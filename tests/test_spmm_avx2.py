"""
AVX2-specific tests for spmm_simd (forward SpMM) on x86_64.

Status (post-milestone-15)
──────────────────────────
The hand-written AVX2 forward kernel was retired in milestone 15
(see docs/demos/milestone_15.md) after Gate F1 measured only
1.20–1.33× per-layer speedup vs scalar — well below the 5× ship
floor. Clang's auto-vectorizer under -march=x86-64-v3 (added in
milestone 14) already covers the forward AXPY inner loop and
delivers ~50 GF/s scalar on Zen 4, saturating the same store-port
limit any AVX2 implementation hits.

After retirement, on x86_64 _core.spmm_simd falls through to
spmm_scalar (the auto-vectorized scalar path). The 46 tests below
therefore trivially pass — both the "scalar" and "simd" paths in
this file route to the same kernel. This file is kept as a
future-proofing asset: when v0.3 revisits x86 SIMD (AVX-512 port
or a different layout), the N % 16 residue sweep, OpenMP
determinism check, and structural edge cases below are still the
right correctness scaffold to rebuild against. Restoring meaningful
coverage at that point is a single commit (re-enable a hand-written
kernel + its #elif dispatch branch in bindings.cpp) — none of the
test logic needs to change.

Purpose
───────
tests/test_spmm.py parametrizes the 23 oracle tests over both scalar
and simd kernels, so scalar-vs-AVX2 correctness is covered on the
canonical shapes. This file adds cases that specifically exercise
AVX2's Phase A/B/C structure (16-wide dual-stream main loop / 8-wide
trail / scalar 0-7 residue), OpenMP determinism, and structural edge
cases that can expose SIMD-only bugs.

This is the x86 forward analog of tests/test_spmm_grad_avx2.py (which
shipped with milestone 14 for the dW kernel) and tests/test_spmm_neon.py
(which covers the ARM side). Same overall shape; retargeted to the
forward `Y = W @ X` semantics and the AVX2 16-lane Phase-A width.

Design rationale
────────────────
The AVX2 forward inner loop has three control-flow phases that the
oracle tests don't hit individually. A shape like M=16, K=16, N=16
enters Phase A once and skips B and C entirely; M=16, K=16, N=17
enters A once, B zero times, and C for the single trailing float.
We parametrize N across every mod-16 residue from 1..65 plus
boundary neighbors to guarantee every path executes and every
A → B → C transition is hit by at least one test.

Unlike the dW kernel, forward does write-through FMA (`y[j] += v*x[j]`)
with no per-slot horizontal reduction — so reordering of the scalar-
vs-AVX2 output is minimal (lane-parallel within an 8-wide SIMD step)
and outputs are often bit-identical. We still compare at rtol=atol=1e-5
because the single-accumulator scalar-j-loop order does differ from
the 8-lane-parallel AVX2 order enough to produce ~1 ULP drift over
long N.

Platform gating
───────────────
This file is skipped on non-x86_64 platforms. On ARM64 the AVX2
kernel isn't compiled (setup.py's IS_X86_64 gate excludes the source);
calling _core.spmm_simd there routes to spmm_simd_neon, which is
already covered by test_spmm_neon.py. Running these tests on NEON
would not break — _core.spmm_simd has the same interface on both
platforms and would still agree with scalar — but the N % 16
parametrization would be redundant with NEON's N % 8 coverage.

Tolerance
─────────
Scalar-vs-AVX2 agreement uses rtol=atol=1e-5 matching the rest of
test_spmm.py and the dW AVX2 test suite. Forward is memory-bound
write-through so per-slot reordering is smaller than dW's, but we
hold the same tolerance bar for consistency.

Run with:   pytest tests/test_spmm_avx2.py -v
"""

from __future__ import annotations

import platform
import warnings

import numpy as np
import pytest
import torch

from sparselab import PaddedCSR, _core


# ─────────────────────────────────────────────────────────────────────
#  Platform gate
#
#  Skip the whole module on non-x86_64 machines. On ARM the AVX2
#  kernel isn't compiled (setup.py IS_X86_64 source gate), and these
#  tests would duplicate test_spmm_neon.py's coverage anyway.
# ─────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"),
    reason="AVX2 forward kernel only compiled on x86_64 (setup.py IS_X86_64 gate).",
)


@pytest.fixture(autouse=True)
def _suppress_known_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        yield


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_sparse_W(
    M: int, K: int, sparsity: float, seed: int = 0
) -> tuple[PaddedCSR, torch.Tensor]:
    """Build (PaddedCSR, dense ground-truth) with given sparsity.

    Returns both so the caller can compare AVX2 output against (a) the
    scalar kernel and (b) the dense torch.matmul oracle — the
    triangular correctness check from test_spmm_neon.py.
    """
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(M, K, generator=gen, dtype=torch.float32)
    keep_mask = torch.rand(M, K, generator=gen) >= sparsity
    W_dense = W * keep_mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)
    return W_csr, W_dense


# ─────────────────────────────────────────────────────────────────────
#  Group F2a — scalar/AVX2 bit-tolerance agreement over random shapes
#
#  20 randomly-sized problems, varied sparsity. If any output cell's
#  AVX2 result diverges from scalar by more than rtol=atol=1e-5, the
#  lane-parallel reordering is producing un-tolerated noise and we
#  have a numeric bug to investigate. Also spot-checks against the
#  torch.matmul dense oracle — if BOTH kernels drift from the oracle
#  at the same point, the bug is upstream (e.g. in PaddedCSR).
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rng_seed", list(range(20)))
def test_scalar_and_avx2_agree_on_random_shapes(rng_seed):
    """20 random (M, K, N, sparsity) draws: scalar and AVX2 must agree."""
    rng = np.random.default_rng(rng_seed)
    M = int(rng.integers(4, 64))
    K = int(rng.integers(4, 64))
    N = int(rng.integers(1, 64))
    sparsity = float(rng.uniform(0.3, 0.95))

    W_csr, W_dense = _make_sparse_W(M, K, sparsity, seed=rng_seed)
    X = torch.randn(K, N, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(rng_seed + 1))

    Y_scalar = _core.spmm_scalar(W_csr, X.numpy())
    Y_avx2   = _core.spmm_simd(W_csr, X.numpy())

    assert Y_avx2.shape == (M, N)
    assert Y_avx2.dtype == np.float32

    # AVX2 vs scalar — isolates AVX2-specific bugs.
    assert np.allclose(Y_scalar, Y_avx2, rtol=1e-5, atol=1e-5), (
        f"Scalar/AVX2 disagree at seed={rng_seed} "
        f"(M={M}, K={K}, N={N}, s={sparsity:.2f}). "
        f"Max abs diff: {np.abs(Y_scalar - Y_avx2).max():.3e}"
    )

    # AVX2 vs torch oracle — confirms we're computing the right math
    # and that PaddedCSR round-trips correctly.
    Y_oracle = (W_dense @ X).numpy()
    assert np.allclose(Y_avx2, Y_oracle, rtol=1e-5, atol=1e-5), (
        f"AVX2 vs dense-oracle mismatch at seed={rng_seed} "
        f"(M={M}, K={K}, N={N}, s={sparsity:.2f}). "
        f"Max abs diff: {np.abs(Y_avx2 - Y_oracle).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group F2b — N-residue coverage
#
#  Our forward AVX2 kernel has three internal phases based on j's
#  position in the inner loop:
#    Phase A: j + 16 <= N  (16-wide dual-stream main)
#    Phase B: j + 8  <= N  (one 8-wide iter if 8-15 remain after A)
#    Phase C: j < N        (scalar 0-7 residue)
#
#  We parametrize N across every mod-16 residue from 1..65 plus
#  boundary neighbors to guarantee every phase boundary is exercised
#  by at least one test. Key cases (see design §3.2 / §3.4):
#
#    N=1..7   → only Phase C (A and B both skip)
#    N=8      → Phase B once, no C
#    N=9..15  → Phase B + 1..7 Phase C residue
#    N=15     → A=0, B=1, C=7 (worst-case tail before first A iter)
#    N=16     → Phase A once, no B, no C (clean main-loop multiple)
#    N=17     → Phase A once + 1 Phase C (post-A no-B case)
#    N=24     → A once + B once + no C (Phase A + Phase B exactly fills)
#    N=31     → A once + B once + C=7 (maximum tail after both SIMD phases)
#    N=32     → 2× Phase A, no B, no C
#    N=33     → 2× Phase A + 1 Phase C
#    N=47     → 2× Phase A + B + 7 Phase C
#    N=48     → 3× Phase A, no B, no C
#    N=64, 65 → stress multi-iter Phase A with/without trailing residue
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "N",
    [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65],
)
def test_all_n_residues_match_scalar(N):
    """Every N % 16 residue must produce the same output as scalar."""
    M, K = 8, 12  # small; we care about N-shape variance, not M or K.
    W_csr, W_dense = _make_sparse_W(M, K, sparsity=0.5, seed=123)
    gen = torch.Generator().manual_seed(7)
    X = torch.randn(K, N, dtype=torch.float32, generator=gen)

    Y_scalar = _core.spmm_scalar(W_csr, X.numpy())
    Y_avx2   = _core.spmm_simd(W_csr, X.numpy())

    assert np.allclose(Y_scalar, Y_avx2, rtol=1e-5, atol=1e-5), (
        f"AVX2 diverged from scalar at N={N} (mod 16 = {N % 16}). "
        f"Max abs diff: {np.abs(Y_scalar - Y_avx2).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group F2c — structural edge cases
#
#  Patterns that can expose SIMD-only bugs:
#
#  Empty-row interleaving: rows of W alternate between empty and
#  populated. Tests that OpenMP's static schedule doesn't assume
#  balanced per-thread work, AND that empty rows remain exactly zero
#  (the self-zero memset at entry is doing its job).
#
#  Single-slot-per-row: every row has exactly nnz=1 with a known
#  column and value, so the outer live-slot loop runs exactly one
#  pass per row. When combined with tiny N, this stresses the Phase-C
#  scalar tail (no Phase A or B iterates).
#
#  Padding-slot safety: padding_ratio=1.0 means HALF of W.values is
#  padding (col_indices[padding_slot] = -1 sentinel). If the kernel
#  walked capacity instead of row_nnz, it would either segfault (c=-1
#  as huge unsigned index into X) or produce silent garbage. Oracle
#  compare catches both.
# ─────────────────────────────────────────────────────────────────────

def test_empty_rows_interleaved():
    """W with every other row empty — OpenMP static schedule robustness."""
    # N=31 → exercises A once, B once, and C=7 all in one slot — the
    # most complex phase-transition case we can pick for this stress.
    M, K, N = 20, 16, 31
    gen = torch.Generator().manual_seed(4)
    W_dense = torch.randn(M, K, dtype=torch.float32, generator=gen)
    keep_mask = torch.zeros(M, K, dtype=torch.bool)
    for i in range(0, M, 2):  # only even rows keep any connections
        keep_mask[i] = torch.rand(K, generator=gen) >= 0.3
    W_dense = W_dense * keep_mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)

    X = torch.randn(K, N, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(5))

    Y_scalar = _core.spmm_scalar(W_csr, X.numpy())
    Y_avx2   = _core.spmm_simd(W_csr, X.numpy())
    assert np.allclose(Y_scalar, Y_avx2, rtol=1e-5, atol=1e-5)

    # Empty rows (odd indices) must be exactly zero — the kernel
    # never ran the inner loop for them, so the self-zeroing memset
    # is the only thing that touched those rows.
    for i in range(1, M, 2):
        np.testing.assert_array_equal(Y_avx2[i], np.zeros(N, dtype=np.float32))


def test_single_slot_per_row_tiny_n():
    """Each row has exactly 1 live slot; tiny N stresses the Phase-C tail."""
    # N=3 → A=0, B=0, C=3; exercises scalar-only path in AVX2 kernel.
    M, K, N = 16, 32, 3
    W_dense = torch.zeros(M, K, dtype=torch.float32)
    for i in range(M):
        W_dense[i, i % K] = float(i + 1) * 0.1
    W_csr = PaddedCSR.from_dense(W_dense)

    X = torch.randn(K, N, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(9))

    Y_scalar = _core.spmm_scalar(W_csr, X.numpy())
    Y_avx2   = _core.spmm_simd(W_csr, X.numpy())
    assert np.allclose(Y_scalar, Y_avx2, rtol=1e-5, atol=1e-5)


def test_padding_slots_not_touched():
    """
    padding_ratio=1.0 means half of W.values is padding. If the kernel
    walked W.total_capacity() instead of row_nnz[i], it would read
    col_indices[padding_slot] = -1 and dereference X[(uint32_t)-1, :]
    → segfault or silent garbage. Oracle compare catches both.
    """
    gen = torch.Generator().manual_seed(2)
    W_dense = torch.randn(24, 32, generator=gen) * (
        torch.rand(24, 32, generator=gen) > 0.75).float()
    W_csr = PaddedCSR.from_dense(W_dense, padding_ratio=1.0)
    # N=13 (mod 16 == 13) also hits Phase A once + Phase B once + C=5.
    X = torch.randn(32, 13, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(3))

    Y_scalar = _core.spmm_scalar(W_csr, X.numpy())
    Y_avx2   = _core.spmm_simd(W_csr, X.numpy())
    assert np.allclose(Y_scalar, Y_avx2, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group F2d — determinism under OpenMP
#
#  With schedule(static) the same work goes to the same thread every
#  time, so the final Y output must be bit-identical across repeated
#  calls. If we ever switch to schedule(dynamic) this would flake —
#  the test defends against a future well-intentioned change that
#  silently breaks training reproducibility (AC-4.1 / CP-4).
#
#  Per-slot write-through in the forward kernel naturally preserves
#  per-row lane order, but across rows the OpenMP scheduling could
#  in principle cause issues (it doesn't with static) — hence this
#  belt-and-suspenders check.
# ─────────────────────────────────────────────────────────────────────

def test_avx2_is_deterministic_across_calls():
    """Same inputs → byte-identical outputs every call (bit-stable)."""
    W_csr, _ = _make_sparse_W(64, 96, sparsity=0.7, seed=11)
    X = torch.randn(96, 48, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(12)).numpy()

    y1 = _core.spmm_simd(W_csr, X)
    y2 = _core.spmm_simd(W_csr, X)
    y3 = _core.spmm_simd(W_csr, X)

    # np.array_equal is the bit-identical check; np.allclose would
    # miss the bug where static schedule silently drifted.
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(y2, y3)


# ─────────────────────────────────────────────────────────────────────
#  Group F2e — self-zeroing contract
#
#  With W = zeros(M, K) the kernel's inner loop never runs (row_nnz[i]
#  is zero for every row). The only thing that touches Y is the
#  memset at the top of the kernel. If that memset is missing or sized
#  incorrectly, Y comes back holding whatever the py::array_t
#  allocator returned — garbage on most platforms. Verifies CP-2 /
#  AC-2.2 self-zeroing.
# ─────────────────────────────────────────────────────────────────────

def test_avx2_fully_sparse_returns_exact_zero():
    """All-zero W must produce exactly zero output (self-zeroing contract)."""
    W_dense = torch.zeros(8, 16, dtype=torch.float32)
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(16, 7, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(6)).numpy()

    Y = _core.spmm_simd(W_csr, X)
    np.testing.assert_array_equal(Y, np.zeros((8, 7), dtype=np.float32))
