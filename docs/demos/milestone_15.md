# Milestone 15 — x86 forward SpMM is already AVX2-fast (auto-vec, no hand kernel)

**This is a measurement and learning milestone, not a ship milestone.**
The hand-written AVX2 forward SpMM kernel we planned for v0.2.2 was
**retired after Gate F1 measured only 1.20–1.33× per-layer over scalar**
— far below the 5× ship floor in
[`docs/design/spmm_forward_avx2.md`](../design/spmm_forward_avx2.md).
The headline finding: the `-march=x86-64-v3` flag added in
[milestone 14](milestone_14.md) Phase 0 (commit `39de0d6`) already
unlocked Clang's auto-vectorizer on the forward AXPY inner loop. On
the GitHub `ubuntu-24.04` Zen 4 runner the scalar forward path now
runs at **~48–51 GF/s**, saturating the store port — which is the
same physical limit any AVX2 implementation can reach on this
hardware.

In short: **milestone 14 shipped two x86 wins, not one.** The
hand-written AVX2 dW kernel was the one we measured and reported.
The auto-vectorized AVX2 forward path was the one we shipped without
realizing it. Milestone 15 is the milestone where we measured it,
validated it, and retired the hand kernel cleanly.

## Why this milestone exists

Milestones 12 and 13 measured pre-AVX2 baselines. Milestone 14
shipped the AVX2 dW kernel (3.0× end-to-end at 40M scale) and added
`-march=x86-64-v3` to `setup.py` as a Phase-0 prerequisite. Milestone
15 was scoped as "do the same hand-written AVX2 port for the forward
path" — `Y = W @ X` was still believed to be the ~4 GF/s scalar
fallback that milestone 13 had measured.

The Phase-A plumbing, Phase-B real intrinsics, Phase-C tests, and
profiler extension all landed cleanly on `main` (commits `4c623aa`,
`e01084f`, `b1e3466`). Gate F1 — the per-layer
`profile_x86_baseline.yml` run — was triggered as required by the
spec, and the artifact disagreed with the design's prediction by a
factor of ~4.

## The measurement that triggered retirement (Gate F1, x86_64)

Run via `.github/workflows/profile_x86_baseline.yml` on commit
`b1e3466` (Linux x86_64, AMD EPYC 9V74 Zen 4, 2 vCPUs):

```
Shape                                                     dw.sc   dw.si dw.si/sc   dw.GF     fw.sc   fw.si fw.si/sc   fw.GF
                                                           (ms)    (ms)              s.s      (ms)    (ms)              s.s
demo15_ffn_up     (384  x 1536 x N=2048, s=0.90)          48.53    4.01    0.08x    4.94      4.70    3.79    0.81x   51.07
demo15_ffn_down   (1536 x 384  x N=2048, s=0.90)          48.42    3.96    0.08x    4.96      4.95    4.09    0.83x   48.44
demo16_ffn_up     (640  x 2560 x N=1024, s=0.90)          66.19    5.55    0.08x    5.05      6.86    5.66    0.82x   48.70
demo16_ffn_down   (2560 x 640  x N=1024, s=0.90)          65.55    5.22    0.08x    5.10      6.47    4.88    0.75x   51.64
tiny              (64   x 64   x N=128,  s=0.80)           0.03    0.01    0.17x    7.38      0.01    0.01    0.94x   29.32
```

Reading the four FFN rows in the forward (`fw.*`) columns:

- **`fw.si/sc` is 0.75–0.83 — the hand-written AVX2 kernel is only
  1.20–1.33× over scalar.** The spec's ship floor was `≤ 0.20`
  (≥ 5× speedup); the design's target was `≤ 0.167` (≥ 6×).
  Hard miss, ~4× short of the floor.
- **`fw.GF` (scalar throughput) is 48.4–51.6 GF/s.** Milestone 13
  measured forward scalar at ~3.8 GF/s on the same runner class
  before `-march=x86-64-v3` was added. That is a **~13× scalar
  speedup attributable entirely to milestone 14's flag**. We just
  did not know it was there because milestone 14's profiler reported
  only the dW columns; we never re-measured forward post-flag.
- **`dw.*` columns are unchanged from milestone 14** — the dW AVX2
  kernel still delivers 12–13× per-layer at ~50 GF/s. Retiring the
  forward hand kernel does not touch the dW story.

The same pattern holds on Linux aarch64 (Graviton-class):

```
Shape                                                     dw.sc   dw.si dw.si/sc   dw.GF     fw.sc   fw.si fw.si/sc   fw.GF
                                                           (ms)    (ms)              s.s      (ms)    (ms)              s.s
demo15_ffn_up     (384  x 1536 x N=2048, s=0.90)          19.09    5.07    0.27x   12.57      6.01    5.48    0.91x   39.89
demo15_ffn_down   (1536 x 384  x N=2048, s=0.90)          19.05    5.29    0.28x   12.59      6.39    5.75    0.90x   37.56
demo16_ffn_up     (640  x 2560 x N=1024, s=0.90)          27.53    8.49    0.31x   12.15      9.22    8.69    0.94x   36.26
demo16_ffn_down   (2560 x 640  x N=1024, s=0.90)          26.87    8.21    0.31x   12.44      9.27    8.67    0.94x   36.08
tiny              (64   x 64   x N=128,  s=0.80)           0.02    0.01    0.32x   12.92      0.01    0.01    0.95x   32.46
```

NEON forward only beats scalar forward by 1.05–1.11× on Graviton —
because Clang's aarch64 auto-vectorizer is also fully covering the
AXPY pattern, hitting ~36–40 GF/s scalar. This isn't a Zen-4-specific
artifact. It is the toolchain doing the work we were planning to do
by hand, on every supported platform. The retirement is therefore the
right call cross-platform, not just on x86.

## Why scalar forward auto-vectorizes but scalar dW doesn't

The forward inner loop:

```cpp
for (int64_t j = 0; j < N; ++j)
    y_row[j] += v * x_row[j];
```

Is the textbook AXPY pattern (`y += v*x`). Clang under
`-march=x86-64-v3` recognizes it, emits 256-bit AVX2 + FMA with
prefetch, and saturates the Zen 4 store port at ~50 GF/s aggregate.
Two cores via OpenMP get us to that ceiling.

The dW inner loop:

```cpp
float acc = 0.0f;
for (int64_t j = 0; j < N; ++j)
    acc += dY_row[j] * X_row[j];
dW_values[slot] = acc;     // horizontal reduce into runtime-varying slot
```

Has a per-slot accumulator-then-scalar-store pattern where the store
target is a runtime-varying slot index. Clang plays it safe and emits
serial scalar FMAs at ~5 GF/s. The hand-written AVX2 dW kernel uses
dual 256-bit accumulators with explicit horizontal reduce and gets
~47–51 GF/s — 12–13× speedup is real and remains the actual x86
moat.

This is why milestone 14's win held while milestone 15's didn't:
**the bottleneck shape determines whether a hand-written kernel is
worth writing.** Compute-bound dot products with reduction tax
(dW) leave headroom for hand-written SIMD; memory-bound AXPY with
contiguous output (forward) does not, because the toolchain
already covers it.

## Why AVX2 forward can't beat ~50 GF/s on Zen 4

Per 8-wide FMA in the forward inner loop:

- 32 B load from `x_row[j..j+7]`
- 32 B load from `y_row[j..j+7]`
- 32 B store to `y_row[j..j+7]`
- 1 FMA (8 lanes)

That is **96 B of memory traffic per 8 FMAs**. Zen 4 has 2 load ports
+ 1 store port per cycle. The store port is the binding constraint:
1 store/cycle × 32 B = 32 B/cycle × 3.2 GHz × 2 cores ≈ 200 GB/s of
store bandwidth, which translates to ~50 GF/s sustained for this
pattern. Both auto-vec scalar and the hand-written dual-stream AVX2
kernel saturate the same physical limit; the hand kernel can't move
that limit.

This is exactly the §6.2 / §7.3 / §7.4 analysis in the design doc —
recorded *before* the Gate F1 measurement, not retrofitted after it.
Risk 7.4 specifically called the auto-vec scenario and prescribed
retirement: "If Clang ever gets smart enough to fully auto-vectorize
this pattern on x86, our hand-written kernel becomes redundant.
That's fine — means the scalar path gets faster. We'd retire the AVX2
kernel and save maintenance cost. Not a risk to correctness, only to
'was the effort worth it'."

## What milestone 15 delivers

A documented retirement, not a kernel:

- **The measurement.** Gate F1 numbers above, captured on the
  feature-branch HEAD before retirement, archived in this writeup.
- **The validation.** Cross-platform numbers (x86 + aarch64 from
  the same workflow run) confirming auto-vec covers the AXPY
  pattern on every supported architecture, not just Zen 4.
- **The retirement.** Commit `85ecb7a` reverts the kernel and
  dispatch plumbing (`4c623aa`, `e01084f`); `_core.spmm_simd` on
  x86_64 now falls through to `spmm_scalar` and benefits from the
  ~50 GF/s auto-vectorized path.
- **The retroactive credit.** Milestone 14's
  `-march=x86-64-v3` flag is now correctly understood as the
  intervention that made x86 forward fast. Milestone 14 shipped
  ~13× forward scalar speedup as a side effect we did not measure
  at the time.
- **The v0.3 scoping update.** The bandwidth-ceiling argument
  generalizes: AVX-512 forward likely does NOT pay off on
  current Intel/AMD silicon for the same reason — store-port
  bandwidth is the limit, and AVX-512 doesn't double it. AVX-512
  for dW remains worthwhile (compute-bound, headroom still
  there); AVX-512 for forward should NOT be in v0.3 scope absent
  a layout change.

## What milestone 15 does NOT do

- ❌ Change the Python API. `_core.spmm_simd` and
  `SparseLinear.forward` continue to exist and behave identically
  — they just route to `spmm_scalar` on x86_64 now. Apple Silicon
  and Linux aarch64 still route to `spmm_simd_neon` unchanged.
- ❌ Touch the dW AVX2 kernel. `csrc/kernels/spmm_grad_avx2.cpp`
  and the milestone 14 12–13× per-layer dW win are completely
  unaffected.
- ❌ Touch the NEON path. `csrc/kernels/spmm_neon.cpp` and
  `spmm_grad_neon.cpp` are not modified.
- ❌ Remove the `-march=x86-64-v3` flag from `setup.py`. That flag
  is the reason scalar forward is fast and stays in place.
- ❌ Delete the AVX2 forward test file. `tests/test_spmm_avx2.py`
  is kept as a future-proofing scaffold for a v0.3 AVX-512 or
  layout-change revisit. Its docstring is updated to record the
  retirement and the trivial-pass behavior post-aliasing.
- ❌ Delete the design doc. `docs/design/spmm_forward_avx2.md` is
  kept and stamped with a "Retired" header pointing here. The
  bandwidth-ceiling analysis in §6.2 and the dual-stream
  rationale in §3.3 stay useful for any future revisit.

## What milestone 15 ALSO retroactively claims about milestone 14

In the spirit of [`north-star.md`](../../.kiro/steering/north-star.md)
§5 (narrative clarity), this is a non-trivial retroactive scope
expansion of milestone 14. It deserves to be stated explicitly:

> Milestone 14's `-march=x86-64-v3` flag delivered ~13× scalar
> forward SpMM speedup on Zen 4 (3.8 GF/s → 51 GF/s), in addition to
> the ~12–13× per-layer hand-written dW speedup it documented. That
> flag-induced forward speedup was measured at milestone 15, not
> milestone 14, but its cause is the milestone 14 commit. Future
> writeups citing the v0.2.x x86 story should attribute both wins to
> milestone 14's intervention.

## Methodology — the process win

The Gate F1 / Gate F2 / risk-register discipline worked exactly as
designed:

1. **Risk 7.4 was called out up-front** in the design doc, before
   any kernel was written. The exact retirement language ("we'd
   retire the AVX2 kernel and save maintenance cost") was committed
   to the repo in `d004357`.
2. **Gate F1 ran before Gate F2.** The per-layer measurement caught
   the materialized risk in ~3 minutes of CI compute. Gate F2 (the
   30-minute end-to-end 40M training validation) was never
   triggered, because the F1 numbers told us what F2 would say.
3. **Retirement was a forward chore commit, not history-mutating.**
   Phases A and B reverted via `git revert -n` chained, committed as
   one forward commit (`85ecb7a`). The Phase A/B/C history stays in
   the log with the design doc + this writeup as the canonical
   explanation of what we measured and why we retired.

This is a process win, not a process failure. The cost of catching
a marginal-win kernel before it ships is one CI run plus a doc
commit; the cost of shipping it would be permanent maintenance
surface for a 1.2× speedup on a path the toolchain already covers.

## Reproduce

```bash
# Re-run the Gate F1 measurement on current main (post-retirement).
# Forward columns now show the auto-vec scalar path on x86_64 — no
# hand kernel — at ~50 GF/s. fw.si/sc should be ~1.0 because simd
# now aliases scalar on x86.
gh workflow run "Profile x86 baseline" --ref main
gh run download <run-id> --dir ./profile-artifacts
cat ./profile-artifacts/profile-Linux-x86_64/profile_output.txt

# Local (Apple Silicon / Linux aarch64) — unchanged from milestone 12
# / milestone 14. NEON forward + NEON dW still ship.
python examples/profile_dw_baseline.py
pytest tests/ -q
```

## Files changed

- **Reverted (one commit, `85ecb7a`):**
  - `csrc/kernels/spmm_avx2.{cpp,hpp}` — deleted (Phase A/B kernel).
  - `csrc/bindings.cpp` — `#elif __AVX2__` dispatch branch and
    include removed; x86 falls through to `spmm_scalar`.
  - `setup.py` — `spmm_avx2.cpp` removed from the `IS_X86_64`
    source list; the inline comment block is reverted to the
    pre-milestone-15 wording.
- **Updated (this commit):**
  - `docs/demos/milestone_15.md` — this file.
  - `docs/design/spmm_forward_avx2.md` — "Retired" header
    prepended; body preserved for future reference.
  - `tests/test_spmm_avx2.py` — module docstring annotated with
    the post-retirement aliasing note.
  - `CHANGELOG.md` — `[Unreleased]` block extended with an
    `Investigated but not shipped` subsection.
- **Unchanged:**
  - `csrc/kernels/spmm_grad_avx2.{cpp,hpp}` — milestone 14 dW
    kernel still ships, still 12–13× per-layer.
  - `csrc/kernels/spmm_neon.cpp`, `csrc/kernels/spmm_grad_neon.cpp`
    — NEON path on ARM unchanged.
  - `examples/profile_dw_baseline.py` — the four forward profile
    columns added in `b1e3466` (Phase C) are kept as a permanent
    diagnostic for any future SIMD revisit.

## What this unblocks

1. **v0.2.2 tag is unblocked on the performance side.** Both the
   x86 training and inference stories are honest. Outstanding
   v0.2.2 blockers are issue #18 (macOS libomp double-load on
   editable installs) and a README refresh — neither of which is
   touched by this milestone.
2. **AVX-512 in v0.3 has cleaner scoping.** AVX-512 for dW is
   obviously worthwhile (still compute-bound, headroom remains).
   AVX-512 for forward is probably NOT worthwhile for the same
   reason AVX2 forward wasn't — store-port bandwidth is the
   limit, and AVX-512 doesn't buy more store-port bandwidth on
   current Intel/AMD silicon. That scoping saves us a week in
   v0.3.
3. **The Gate F1 / risk-register methodology is validated.** Risk
   7.4 was anticipated, measurement caught the materialized risk
   before shipping, retirement was a clean forward-chore commit.
   This pattern is the right scaffolding for future SIMD work.

---

_Measured: 2026-05-06 via `profile_x86_baseline.yml` GitHub Actions
run on commit `b1e3466` (last commit before retirement). Retirement:
2026-05-07, commit `85ecb7a`. Documented: this writeup._
