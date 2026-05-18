# SparseLab

![v0.2](https://img.shields.io/badge/version-0.2.1-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyPI](https://img.shields.io/pypi/v/sparselab)
![tests](https://img.shields.io/badge/tests-442%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DarshanFofadiya/sparselab/blob/main/examples/colab_try_sparselab.ipynb)

**Masking is not sparsity.**

---

## TL;DR — the 60-second read

SparseLab is a PyTorch library for training sparse neural networks
*from scratch*, with real sparse storage and real sparse kernels.
Not mask-on-dense. Not post-training pruning. Actual sparsity
at training time, on commodity hardware.

**Why it matters:**

- **Most neural networks are mostly unnecessary.** 90%-sparse MNIST
  reaches **97.45% vs 98.06% dense — 0.61 pp gap for 82% memory
  reduction**. A 10M-param Tiny Shakespeare transformer tracks
  dense val loss within 0.055 nats at **17.5% of dense parameters**.

- **Nobody else is doing this.** Cerebras trains sparse but uses
  dense storage + a binary mask on wafer-scale chips. Neural
  Magic does post-training pruning for CPU inference (not
  training). torchao is GPU-structured-2:4. rigl-torch is a
  single algorithm and mask-simulated. Nobody ships an
  actually-sparse training stack you can `pip install` on a
  laptop. (See the table below for specifics.)

- **v0.1 proves the paradigm on a laptop.** 10M-param transformer
  on a MacBook CPU: 37% of dense memory, quality tracking dense,
  real sparse storage end-to-end. Not a simulation.

- **v0.2 makes the kernels actually fast on every supported CPU.**
  Hand-written NEON dW (Apple Silicon + Linux aarch64, milestone 12)
  and AVX2 dW (Linux x86_64, milestone 14) deliver
  **3.0× end-to-end** sparse training speedup at 40M-param transformer
  scale on x86 and **~2× end-to-end** on Apple Silicon. Forward
  SpMM is fast cross-platform — auto-vectorized to ~50 GF/s on
  Zen 4 once the `-march=x86-64-v3` flag landed in milestone 14;
  [milestone 15](docs/demos/milestone_15.md) measured this and
  retired a planned hand-written AVX2 forward kernel because the
  compiler had already done the work. Training dynamics unchanged —
  same seed produces identical val loss to four decimal places.

- **v0.2 also lays the path to clusters.** CPU-cluster data
  parallelism turns "1B dense model → 100M live sparse" into a
  realistic workload on commodity CPU infrastructure. A 10-machine
  CPU cluster with 128 GB RAM each is a few thousand dollars; an
  8×H100 DGX node that handles an equivalent dense workload runs
  $300K+ upfront or $20K+/month in the cloud.

- **It's also a hardware problem, not just software.** GPUs are
  built for dense; sparse accelerators have no training stack to
  target. SparseLab is the software the hardware ecosystem has
  been waiting for.

**For whom:** DST researchers. PyTorch users without GPU access.
Contributors who care about low-level CPU performance. Anyone
building toward purpose-built sparse hardware.

**Get it:** `pip install sparselab`. Pre-built wheels for macOS
arm64 and Linux x86_64/aarch64, Python 3.11–3.13. MIT license.
442 tests including autograd gradcheck.

---

## What no one else is shipping

We audited the ecosystem before building this. Here's the concrete
gap:

| Project | Storage | Training | Hardware | `pip install` on a laptop |
|-|-|-|-|-|
| **SparseLab** (us) | **Real sparse (Padded-CSR)** | **From scratch, pluggable DST** | **CPU (NEON + AVX2 + OpenMP)** | **Yes** |
| Cerebras `cstorch.sparse` | Dense + mask | From scratch, pluggable DST | Wafer-scale only | No |
| Neural Magic SparseML | Dense + mask | Post-training pruning | CPU (inference) | Yes (inference only) |
| rigl-torch | Dense + mask | From scratch, RigL only | CPU/GPU | Yes (mask-simulated) |
| torchao.sparsity | Structured 2:4 | Post-training | GPU | Yes (GPU, structured) |
| `torch.sparse` | Real sparse | Not really supported | CPU/GPU | Yes (no training support) |

**The corner we're in that nobody else occupies: actually-sparse,
unstructured, training-from-scratch, CPU-native, with a pluggable
DST algorithm interface.** That's the specific claim; `docs/LANDSCAPE.md`
walks through each project in detail with what we learned from
them and what we explicitly diverge on.

If you know of a library doing the same thing we're doing,
please file an issue — keeping this comparison honest is part of
how we want to operate.

---

## Most neural networks are mostly unnecessary

That's not a claim we're making in the abstract. It's what our
measurements show, and what the DST literature has been pointing
at for years.

From our own demos in this repo:

- **MNIST (2-layer MLP), trained to convergence:** 90%-sparse
  reaches **97.45%** accuracy vs dense **98.06%** — a **0.61 pp
  gap for 82% memory reduction**. The caveat: sparse needs ~1.8×
  more epochs to reach its plateau (the cost of a random-and-
  frozen mask). Smarter DST routers (RigL / SET) close that
  gap in fewer epochs; the v0.1 demo uses random masks to
  establish the floor. See `docs/demos/milestone_05.md` and
  `docs/demos/milestone_08.md`.
- **10M-parameter transformer on Tiny Shakespeare (10k steps):**
  Keeping only **17.5% of weights** (attention 70% + FFN 90%
  sparse) tracks dense val-loss to within 0.055 nats — within
  run-to-run noise for char-level LM. Memory footprint: **37%
  of dense** at inference. See `docs/demos/milestone_10.md`.

The pattern is consistent across every model size and task we've
tried: you can keep comparable quality at roughly 10–20% of dense
parameters, and competitive quality at 30%. The extra 80-90% of
weights in a trained dense model are mostly noise around the
small fraction that does the actual work.

**Sparsity isn't a lossy compression; it's the actual information
structure of the learned model.** Dense training can't cheaply
tell the difference because it has to compute the whole matrix
regardless of what's doing the work. The reason you can't just
run sparse training as a drop-in is that nobody's shipped a
software stack that treats sparsity as first-class at training
time.

That's the problem SparseLab is built to solve.

---

## This is a software AND hardware problem

Two things have to be true for sparse to win:

1. **The software stack has to treat sparsity as first-class.** Not
   a mask over a dense tensor. Not a post-training step. The
   storage format, kernels, autograd integration, and training
   loop all have to work with live weights directly. That's what
   SparseLab is.

2. **The hardware has to be built for it.** Current GPUs are
   engineered for the dense-matmul workload and they optimize it
   ruthlessly. They handle sparse poorly because nobody's asked
   them to; vendor roadmaps don't prioritize what researchers
   don't actually run. Purpose-built sparse accelerators — the
   neuromorphic-style chips the industry has been theorizing
   about for a decade — have no training software to target, so
   they stay theoretical.

SparseLab's bet is that the software stack has to exist first,
so the hardware has something real to optimize for.

The brain runs on roughly **20 watts** — about a dim light bulb.
The GPUs that approximate a fraction of what it does draw
kilowatts each, scaled into datacenters that draw gigawatts.
That gap isn't fundamental. Part is software: the mask-on-dense
paradigm wastes most of the compute. Part is hardware: we built
silicon for the wrong workload. Both have to be fixed, and the
software is the one researchers can move first.

---

## What v0.1 delivers

**v0.1 is the proof that actually-sparse training is viable on
commodity hardware.** A 10-million-parameter transformer on a
MacBook CPU, trained from scratch at 37% of dense memory, quality
tracking dense. Not a simulation — real sparse kernels, real
at-rest memory.

What becomes possible on top of this foundation:

- **Researchers without GPU access can run real experiments.** A
  Mac and a weekend replaces a cloud bill for a lot of research.
- **Community kernel optimization over time.** The SpMM, dW, and
  transpose kernels are all contributor-shaped problems. Every
  optimization PR is a speedup everyone inherits.
- **Purpose-built sparse hardware has a software stack to target.**
  Sparse accelerators have been a research topic for 10+ years;
  SparseLab is the first real end-to-end sparse training stack
  they can plug into.

---

## Foundational results (v0.1)

10M-parameter decoder-only transformer (6 layers, d_model=384),
trained from scratch on Tiny Shakespeare for 10,000 steps on an
M3 Pro MacBook. These are the v0.1 quality numbers that the
v0.2.x kernel work was built on top of — they're the proof that
actually-sparse training reaches dense quality, not the latest
performance numbers (see [Performance progression](#performance-progression--v01--v022)
for those):

| | Dense | Sparse FFN 90% | Sparse all (attn 70% + FFN 90%) |
|-|-|-|-|
| **Parameters** | 10.7M | 4.4M live | **1.9M live** |
| **Inference memory** | 41.0 MB | 19.9 MB (48%) | **15.3 MB (37%)** |
| **Training memory (weight+grad+padding)** | 81.8 MB | 35.9 MB | **25.2 MB (31%)** |
| **Final validation loss** | 2.534 | 2.582 | 2.589 |

**Memory footprint of the all-sparse model: 37% of dense at
inference, 31% at training.** Real, at-rest, not simulated.

**Quality tracks dense to within 0.055 nats** after 10,000 steps
— within noise for char-level language modeling at this scale. No
sparse-specific pathology. Full writeup: [`docs/demos/milestone_10.md`](docs/demos/milestone_10.md).

### On speed: where we are now

v0.1 was 2.4× slower than dense per step (FFN-only) and 4.6× slower
(all-sparse) on CPU. The v0.2.x kernel work has narrowed that
significantly — measured numbers in the
[Performance progression](#performance-progression--v01--v022) table
above (3.0× end-to-end speedup on Linux x86_64, ~2× on Apple
Silicon, both vs the v0.1 scalar baseline).

We are still slower per step than a dense GPU on small models;
that's a real cost and we don't hide it. Two things continue to
narrow the gap:

1. **Sparse kernels have fixed per-layer overhead.** At the matrix
   sizes in v0.1's demos, that overhead dominates. It does not at
   larger scale — the break-even point is when weight matrices
   become memory-bandwidth bound, typically ~1024+ hidden size and
   above.
2. **Per-step speed is not the only frame.** CPU-cluster data
   parallelism (next on the roadmap) changes the scaling story
   entirely — see "The trajectory" below.

---

## The trajectory

v0.1 ran on one machine. That was the proof-of-concept phase — show
that actually-sparse training works, ship a real library with
wheels, tests, and demos.

**v0.2.x has been the kernel-performance phase.** Hand-written NEON
dW (Apple Silicon + Linux aarch64, milestone 12) and AVX2 dW
(Linux x86_64, milestone 14) brought sparse training within 2–3×
of v0.1's dense-equivalent step times — see
[Performance progression](#performance-progression--v01--v022).

**Data parallelism across CPU cores and machines** is the next
phase, tracked as
[issue #4](https://github.com/DarshanFofadiya/sparselab/issues/4).
This is the scaling story that matters:

- 1B dense parameters at 90% sparsity = 100M live weights.
- 100M live weights ≈ 400 MB at training-time precision. Fits in
  RAM on any modern laptop.
- With CPU-cluster DDP, training it across 10 machines with 128 GB
  RAM each is a realistic configuration. Total hardware cost:
  a few thousand dollars of commodity workstations, or pennies-
  on-the-dollar compared to their GPU equivalent in the cloud.
- The GPU equivalent for a 1B-dense workload today is an 8×H100
  DGX node: roughly **$300K–$400K to purchase outright**, or
  ~$20K/month sustained in cloud at mid-market rates. Not
  available to most researchers at any university lab, lab-
  adjacent startup, or geography without GPU allocation.

CPU clusters are accessible to nearly any researcher, any
university lab, any startup without GPU allocation. H100 nodes
aren't. That's the asymmetry we're building toward.

**v0.3 and beyond: the hardware question.** If CPU-native actually-
sparse training works at scale, the next step is hardware that's
purpose-built for it. Not general-purpose GPUs doing sparse poorly,
not wafer-scale chips with dense-mask simulation — actual sparse
accelerators that match the brain's efficiency profile. The
neuromorphic industry wants this. The problem is nobody has a
training stack to target. SparseLab intends to be that stack.

We're not claiming to beat GPUs today. We are claiming the
paradigm is wrong, and that CPU-native actually-sparse training
deserves to exist as a serious research platform that can
eventually scale into specialized hardware.

---

## Who this is for

- **DST researchers** tired of reinventing scaffolding for every
  algorithm. Write your next drop/grow rule as a ~50-line subclass
  of `SparsityAlgorithm` on top of real sparse storage. No more
  mask-on-dense simulation.
- **Researchers without GPU access.** A MacBook or workstation CPU
  is enough to run real experiments on 10K – 10M parameter models
  today, and larger with v0.2 DDP.
- **Contributors who care about low-level performance.** The SpMM
  and dW kernels are the moats; every optimization compounds
  forever. NEON + AVX2 today; AVX-512 dW for newer Intel/AMD parts
  and ARM SVE for server-class arm tomorrow.
- **Anyone curious about sparse-first ML.** The code is
  intentionally readable and well-commented. A grad student can
  read the NEON inner loop and understand it.

---

## Quick look

```python
import torch
import sparselab

# One-line swap: nn.Linear → sparselab.SparseLinear.
model = torch.nn.Sequential(
    sparselab.SparseLinear(784, 512, sparsity=0.9),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Pluggable DST: add SET topology mutation in 2 lines.
algo = sparselab.SET(sparsity=0.9, drop_fraction=0.3, update_freq=100)
model.apply(algo)        # attaches to every SparseLinear in the tree

# Rest of your training loop is normal PyTorch.
for step in range(1000):
    x = torch.randn(128, 784)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    opt.step()
    algo.step()          # drives topology mutation on the schedule
    opt.zero_grad()
```

`SparseLinear` is a standard `nn.Module`. Its parameters are standard
`nn.Parameter`s. It loads into standard `torch.optim` optimizers.
The only thing different is that under the hood, the weight tensor
is stored as a Padded-CSR and the forward/backward go through our
sparse kernels.

### Prove the memory claim yourself

```python
import torch
import sparselab

# A 784 × 512 layer, dense vs 90% sparse.
dense  = torch.nn.Linear(784, 512, bias=False)
sparse = sparselab.SparseLinear(784, 512, sparsity=0.9, bias=False)

# Dense: 4 bytes per weight (float32).
dense_bytes = dense.weight.numel() * 4

# Sparse: 4 bytes per live value + 4 bytes per column index = 8 bytes per live.
# (Plus O(nrows) for the tiny row-metadata arrays — negligible at this scale.)
sparse_bytes = sparse.nnz * 8

print(f"Dense:  {dense_bytes / 1024:.1f} KB")
print(f"Sparse: {sparse_bytes / 1024:.1f} KB  ({100 * sparse_bytes / dense_bytes:.0f}% of dense)")
# Dense:  1568.0 KB
# Sparse: 310.5 KB  (20% of dense)
```

That's 20% of dense memory for the same 784 × 512 Linear layer at
90% sparsity. Real bytes, not a mask. The column-index array is
what makes it 20% rather than the naive "10% of dense" — every
live weight carries a 4-byte index so the kernel knows which
column it belongs to. That index overhead is the cost of being
actually sparse; it's also why the break-even point is around
50% sparsity (below that, dense storage is smaller).

---

## Install

```bash
pip install sparselab
```

Pre-built wheels are published for the following platforms, with
OpenMP and the SIMD kernels bundled inside — no system libraries to
install, no compiler required:

| Platform | Arch | Python versions | Kernels |
|-|-|-|-|
| macOS | arm64 (Apple Silicon) | 3.11, 3.12, 3.13 | NEON forward + NEON dW + OpenMP |
| Linux | x86_64 (manylinux, Haswell/Zen 1+) | 3.11, 3.12, 3.13 | AVX2+FMA dW (hand-written) + AVX2 forward (auto-vec at `-march=x86-64-v3`) + OpenMP |
| Linux | aarch64 (manylinux, ARMv8.2-A+) | 3.11, 3.12, 3.13 | NEON forward + NEON dW + OpenMP |

**Minimum CPU requirement on Linux x86_64:** Haswell (Intel, 2013+) or
Zen 1 (AMD, 2017+). The `-march=x86-64-v3` build target emits AVX2 +
FMA instructions; pre-2013 x86 CPUs will hit `Illegal instruction` at
import. Every Linux distribution currently supported by PyTorch 2.8+
targets Haswell+ / Zen+ in practice. (See CHANGELOG `[Unreleased]` /
v0.2.2 for the rationale.)

**Windows & Intel Mac:** not yet, and Intel Mac is unlikely.
- **Windows users:** native Windows wheels are tracked as
  [issue #8](https://github.com/DarshanFofadiya/sparselab/issues/8).
  In the meantime use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
  with our Linux wheel — that path works today.
- **Intel Mac users:** [upstream PyTorch deprecated macOS x86_64 wheels
  in January 2024](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)
  and the last torch release published for Intel Mac is 2.2.2. We can
  build a SparseLab wheel for the platform, but `pip install` can't
  resolve our `torch>=2.8` requirement on it — the wheel would be
  unusable. Workaround: build SparseLab from sdist with `torch<=2.2.2`
  pinned (`pip install torch==2.2.2 && pip install sparselab --no-binary sparselab`).
  Requires a C++ toolchain. See CHANGELOG v0.1.1 "Investigated but not
  shipped" for the full reasoning.

If you're on a platform not in the table above, pip falls back to
compiling from source. For that you'll need:

- **Python** 3.11+
- **PyTorch** ≥ 2.8 (pulled in automatically)
- **C++17 compiler** — clang 14+ or gcc 9+
- **libomp** on macOS: `brew install libomp`. On Linux it ships with
  gcc/clang. Without it the build still succeeds but runs sequentially
  (4–6× slower).

### Development install

For hacking on SparseLab itself:

```bash
git clone https://github.com/DarshanFofadiya/sparselab.git
cd sparselab
brew install libomp        # macOS only

# Pre-install build deps into the runtime env, then editable install
# with --no-build-isolation. The flag is REQUIRED on macOS (see
# docs/development.md for why) and recommended everywhere for speed
# and consistency.
pip install --upgrade setuptools wheel pybind11 'torch>=2.8'
pip install -e '.[dev]' --no-build-isolation
```

The editable install rebuilds the C++ kernels whenever you touch a
file in `csrc/`. First build takes ~45 seconds.

**If `pip install -e '.[dev]'` fails on macOS with `ImportError:
Library not loaded: @rpath/libomp.dylib`, you forgot
`--no-build-isolation`.** The full reasoning + recovery commands are
in [`docs/development.md`](docs/development.md). [`CONTRIBUTING.md`](CONTRIBUTING.md)
has the dev quickstart.

### Verify install

```python
import sparselab
print(sparselab.__version__)          # should print 0.2.1 or newer

# Quick smoke test — this should run in under a second
import torch
W = sparselab.PaddedCSR.random(256, 128, sparsity=0.9, seed=0)
X = torch.randn(128, 32)
Y = sparselab.spmm(W, X)
print(Y.shape)                          # torch.Size([256, 32])
```

If you installed from source, the full test suite is also available:

```bash
pytest
# 442 passed, ~3s on Apple Silicon / ~5s on Linux x86
```

If something doesn't work, please [open an issue with the output](https://github.com/DarshanFofadiya/sparselab/issues)
— we want to hear about install failures, especially on platforms
or environments we may not have tested.

### Install troubleshooting

Most install failures fall into one of five categories. The error
message is usually enough to pick the right fix.

**`ERROR: Could not find a version that satisfies the requirement sparselab`**

pip can't find a wheel that matches your platform. Run
`pip debug --verbose` and check the "Compatible tags" list. Your
Python's compatible tags must include at least one of:

  - `cp311-cp311-macosx_11_0_arm64` / `cp312-...` / `cp313-...` (Apple Silicon)
  - `cp311-cp311-manylinux_2_28_x86_64` / `cp312-...` / `cp313-...` (Linux x86_64)
  - `cp311-cp311-manylinux_2_28_aarch64` / `cp312-...` / `cp313-...` (Linux aarch64)

Common causes:

- **Intel Mac (macOS x86_64).** We don't ship a wheel for this
  platform because upstream PyTorch stopped publishing Intel Mac
  wheels after torch 2.2.2. Your options: (a) use a machine with an
  Apple Silicon Mac or a Linux host, or (b) install from sdist with
  an older torch pinned: `pip install torch==2.2.2 && pip install sparselab --no-binary sparselab` (requires a C++ toolchain).
- **Free-threaded Python 3.13t (PEP 703).** Its tags are `cp313t-...`,
  not `cp313-...`, so our wheels don't match. Use a regular (GIL-enabled)
  CPython 3.11/3.12/3.13 for now.
- **Python 3.14 or newer.** We haven't built wheels for it yet. Use 3.11/3.12/3.13.
- **Old pip on an old distro.** If `pip --version` shows < 21.0,
  run `pip install --upgrade pip` first. Older pip doesn't know about
  `manylinux_2_28`.

**`bad interpreter: /path/to/python3.X: no such file or directory`** (before the pip error)

Your virtualenv is pointing at a Python binary that no longer exists
— a stale venv from a Python upgrade or a deleted project. Not a
sparselab problem. Recreate the venv:

```bash
python3 -m venv ~/my-sparselab-env
source ~/my-sparselab-env/bin/activate
pip install sparselab
```

**macOS: `Symbol not found`, `OMP: Error #15`, or `ImportError: @rpath/libomp.dylib`**

PyPI wheels reuse PyTorch's bundled libomp at import time, so this
should not happen from a plain `pip install sparselab`. The two
common causes if it does:

- **You're running an editable install (`pip install -e .`) without
  `--no-build-isolation`.** That flag is required on macOS for the
  reasons documented in
  [`docs/development.md`](docs/development.md#why---no-build-isolation-matters-on-macos).
  Recovery is one command: `pip install -e '.[dev]'
  --no-build-isolation --force-reinstall`.
- **A non-standard `DYLD_LIBRARY_PATH`** or a global libomp install
  that the macOS dynamic loader is finding first. Try a fresh venv
  with no environment overrides.

**Rosetta Python on an M-series Mac**

If you're on Apple Silicon but running an x86_64 Python (e.g., an old
Conda environment migrated from an Intel Mac), `platform.machine()`
returns `x86_64` and pip will look for an Intel Mac wheel we don't
ship. Check with:

```bash
python -c "import platform; print(platform.machine())"
# Should print: arm64
```

If it prints `x86_64`, you're on a Rosetta Python. Install a native
arm64 Python (e.g., from python.org or `conda create -n sc python=3.11`
with the arm64 Miniforge installer) and retry.

**Still stuck?**

Open an issue with the output of these four commands and we'll take
a look:

```bash
python --version
python -c "import platform; print(platform.machine(), platform.platform())"
pip --version
pip debug --verbose 2>&1 | grep -A 3 "Compatible tags" | head -8
```

---

## Performance progression — v0.1 → v0.2.2

End-to-end speedups vs. the v0.1.x scalar baseline, measured on the
same hardware in each row. All numbers come from
`.github/workflows/validate_40m_scalar.yml` (Gate F2) at 40M-param
mini-GPT scale: 8 layers, d_model=640, d_ff=2560, FFN 90% sparse,
200 SGD steps, fixed seed. Same training dynamics — bit-stable val
loss to four decimal places across every row below.

| Platform | Pre-v0.2 (scalar) | Post-v0.2.2 (SIMD) | Speedup | What shipped |
|-|-|-|-|-|
| **Apple Silicon** (M3 Pro, 6 perf cores) | ~110 ms/step | **~56 ms/step** | **~1.96×** | NEON `dW` kernel ([milestone 12](docs/demos/milestone_12.md)) |
| **Linux x86_64** (Zen 4 on GitHub Actions, 2 vCPUs) | 4316 ms/step | **1436 ms/step** | **~3.0×** | AVX2+FMA `dW` kernel ([milestone 14](docs/demos/milestone_14.md)) + auto-vectorized AVX2 forward via `-march=x86-64-v3` ([milestone 15](docs/demos/milestone_15.md)) |
| **Linux aarch64** (Graviton-class, 4 cores) | not directly Gate-F2-measured at 40M | (~1.7–2.2× extrapolated from milestone 13's per-layer numbers) | NEON `dW` kernel — same code path as Apple Silicon |

Notes on what those speedups *are*:

- **Per-step wallclock**, not per-FFN-layer. Per-layer kernel
  speedups are larger (e.g., AVX2 dW is 12–13× per layer on Zen 4)
  but Amdahl-bounded by the embedding / attention / softmax / loss
  share of a step.
- **Same correctness bar.** All three platforms still pass the full
  442-test suite at `rtol=atol=1e-5` against PyTorch oracles, plus
  `torch.autograd.gradcheck` at default tolerances.
- **x86 forward SpMM also got fast as a side effect** of milestone
  14's `-march=x86-64-v3` flag — Clang auto-vectorized the
  forward AXPY inner loop to ~50 GF/s on Zen 4. That's why
  milestone 15 retired a planned hand-written AVX2 forward
  kernel; the compiler had already done the work and a hand kernel
  only delivered 1.20–1.33× over auto-vec.

What didn't change between v0.1 and v0.2.2:

- **The Padded-CSR memory footprint.** 90%-sparse 40M-param model
  still uses ~37% of dense memory. The v0.2.x work is purely
  speed; storage cost was already settled in v0.1.
- **The Python API.** `sparselab.spmm`, `SparseLinear`,
  `SparsityAlgorithm`, `Static` / `SET` / `RigL` are all unchanged
  from v0.1.x. Users on `kernel="auto"` (the default) pick up the
  speedup transparently — no code change required.

Reproduce the x86 numbers:

```bash
gh workflow run "Validate 40M scalar baseline" --ref main
# Wait ~30 minutes, then:
gh run list --workflow="validate_40m_scalar.yml" --limit 1
gh run download <run-id> --dir ./validate-artifacts
cat ./validate-artifacts/validate-40m-x86_64/validate_40m.txt
```

---

## Demos

Runnable examples, each a single file with a banner explaining what
it proves. Run them top to bottom; each adds one more concept:

```bash
python examples/demo_01_bridge.py                  # pybind11 "hello world"
python examples/demo_02_dot.py                     # NEON SIMD dot product
python examples/demo_03_spmm.py                    # sparse matmul benchmark
python examples/demo_04_autograd.py                # sparse backward pass
python examples/demo_05_mnist.py                   # MNIST at 7 sparsity levels
python examples/demo_08_sparse_full_convergence.py # dense vs sparse @ 90%, converged
python examples/demo_09_parallel_speedup.py        # OpenMP thread scaling
python examples/demo_11_rigl_vs_set_vs_static.py   # RigL vs SET vs Static
python examples/demo_13_tiny_transformer.py        # 200-step char transformer
python examples/demo_14_sparse_attention.py        # sparse attention (not promoted to API)
python examples/demo_15_mini_gpt.py                # 10M-param GPT, 3-way comparison
```

Demos that need visualization or datasets (MNIST, transformer) pull
in matplotlib and torchvision:

```bash
pip install -e '.[demos]'
```

---

## What works today

- `sparselab.PaddedCSR` — sparse storage with O(1) slot insert,
  cached transpose, round-trip with `torch.sparse_csr`.
- `sparselab.spmm(W, X)` — sparse-dense matmul, autograd-aware.
  Forward and backward are both vectorized: NEON on Apple Silicon
  + Linux aarch64, AVX2+FMA on Linux x86_64, scalar fallback on
  unrecognized platforms. OpenMP-parallel over the row dimension.
- `sparselab.SparseLinear(nn.Module)` — drop-in `nn.Linear`
  replacement. Standard `nn.Parameter`, standard `state_dict`.
- `sparselab.SparsityAlgorithm`, `Static`, `SET`, `RigL` — pluggable
  DST API. Inspired by Cerebras's `cstorch.sparse.SparsityAlgorithm`;
  see `docs/LANDSCAPE.md`.
- 442 tests, including gradcheck against PyTorch autograd and
  dense-equivalence oracles at 1e-5 tolerance. CI gates on three
  required platforms: macOS-arm64, Linux x86_64, Linux aarch64.
- 15 demos, end-to-end from "hello pybind" through "10M-param
  mini-GPT trained on Shakespeare at 90% sparsity."

## Known limitations (we'd rather tell you upfront)

- **Single machine only** in v0.2. Multi-machine DDP
  ([issue #4](https://github.com/DarshanFofadiya/sparselab/issues/4))
  is the next major scaling step.
- **CPU only.** GPU is a v0.3+ contribution target
  ([issue #3](https://github.com/DarshanFofadiya/sparselab/issues/3)).
- **Slower per-step than dense on small models** below ~50% sparsity.
  Padded-CSR's per-live-weight column-index overhead doesn't pay off
  there. At FFN-shape, 90%-sparse training step is now ~3× the
  scalar baseline on x86 and ~2× on Apple Silicon.
- **Pre-2013 x86 CPUs not supported.** The Linux x86_64 wheel
  requires AVX2 + FMA (Haswell / Zen 1+).
- **Transpose cache has a theoretical `id()` collision risk** when a
  `PaddedCSR` is garbage-collected and Python reuses its id for a new
  same-shape, same-topology-version CSR. Documented in
  `sparselab/ops.py`; has not been observed in practice but is real.
- **No AVX-512 yet** ([issue #3 / v0.3 scope](https://github.com/DarshanFofadiya/sparselab/issues/3)).
  AVX-512 is worth pursuing for the dW kernel (compute-bound,
  headroom remains); for forward SpMM the bottleneck is store-port
  bandwidth, not FMA — see [milestone 15](docs/demos/milestone_15.md).
- **Sparse attention is not a primitive** in v0.2. We verified it
  works end-to-end (see demo_14 and demo_15 all-sparse) but didn't
  promote it to a first-class API
  ([issue #9](https://github.com/DarshanFofadiya/sparselab/issues/9)).
- **Fixed row capacity in Padded-CSR.** Each row's capacity is frozen
  at layer construction (initial `nnz × 1.2`). This gives us O(1)
  insertion during topology mutation. Algorithms that grow a row's
  live count beyond initial capacity will fail — SET and RigL work
  fine because they keep per-row `nnz` constant. Adaptive-density
  DST would need a `compact_all()` primitive
  ([issue #6](https://github.com/DarshanFofadiya/sparselab/issues/6)).

---

## Roadmap

**v0.1 (shipped).** The pluggable DST foundation. Kernels, storage,
autograd, `SparseLinear`, `SparsityAlgorithm` base,
`Static` / `SET` / `RigL`, end-to-end 10M-param transformer demo,
pre-built PyPI wheels for macOS and Linux.

**v0.2 (in progress).** The performance and scaling phase.
- ✅ **Hand-tuned NEON `dW` kernel.** Shipped in v0.2.1 (milestone
  12). 1.96× end-to-end on Apple Silicon at 40M-param transformer
  scale; ~6.5× per-layer dW.
- ✅ **AVX2 + FMA `dW` kernel for Linux x86_64.** Shipped in v0.2.2
  (milestone 14). 3.0× end-to-end on Zen 4 CI runners; 12–13×
  per-layer dW. `-march=x86-64-v3` raised the x86 scalar baseline
  ~13× as a side effect, making forward SpMM auto-vectorize cleanly
  ([milestone 15](docs/demos/milestone_15.md) measured this and
  retired a planned hand-written AVX2 forward kernel).
- ✅ **macOS editable-install libomp double-load fix.** Shipped in
  v0.2.2 (issue #18). macOS-arm64 is now a required gated CI leg.
- 🚧 **README + docs refresh, version bump, v0.2.2 tag.**
- 📋 **CPU-cluster data parallelism via PyTorch DDP**
  ([issue #4](https://github.com/DarshanFofadiya/sparselab/issues/4)).
  The biggest scaling step: training 100M-param-live sparse models
  (1B dense equivalent) across commodity CPU clusters that cost a
  few thousand dollars.
- 📋 **Buffer reuse / arena in the backward path**
  ([issue #7](https://github.com/DarshanFofadiya/sparselab/issues/7)).
- 📋 **`PaddedCSR.compact_all()` primitive** for adaptive-density
  DST algorithms
  ([issue #6](https://github.com/DarshanFofadiya/sparselab/issues/6)).
- 📋 **Windows native wheels**
  ([issue #8](https://github.com/DarshanFofadiya/sparselab/issues/8)).
- 📋 **More DST algorithms** — Sparse Momentum
  ([#5](https://github.com/DarshanFofadiya/sparselab/issues/5)),
  Top-KAST ([#12](https://github.com/DarshanFofadiya/sparselab/issues/12))
  — from community PRs.

**v0.3 (post-launch community phase).**
- **AVX-512 dW kernel for newer Intel/AMD CPUs**
  ([issue #3 covers GPU; AVX-512 lives in the same v0.3 scope](https://github.com/DarshanFofadiya/sparselab/issues/3)).
  AVX-512 is worth pursuing for compute-bound dW, NOT for
  forward SpMM — store-port bandwidth is the bottleneck on
  current Intel/AMD silicon, and AVX-512 doesn't increase
  store bandwidth (see [milestone 15](docs/demos/milestone_15.md)
  for the bandwidth-ceiling analysis).
- **Memory-mapped weights** for models that exceed node RAM
  ([#14](https://github.com/DarshanFofadiya/sparselab/issues/14)).
- **Sparse attention as a first-class primitive**
  ([#9](https://github.com/DarshanFofadiya/sparselab/issues/9)).
- **GPU backend** as a community-led contribution opportunity
  ([#3](https://github.com/DarshanFofadiya/sparselab/issues/3)).
- **Hardware-vendor partnerships** for sparse accelerators once
  the software stack proves itself at scale.

---

## Positioning vs other projects

| | What it is | How we relate |
|-|-|-|
| **Cerebras `cstorch.sparse`** | Production sparse training on wafer-scale chips | We adopt their `SparsityAlgorithm` API shape. They use dense+mask simulation; we use Padded-CSR. Complementary. |
| **Neural Magic SparseML** | Post-training pruning for inference | Different workflow. They compress trained models; we train sparse from scratch. |
| **rigl-torch** | Community PyTorch port of RigL | Single-algorithm, mask-simulated. We're the pluggable multi-algorithm version with real sparse storage. |
| **torchao.sparsity** | GPU structured (2:4) sparsity | Different axis: structured-GPU-posttraining vs unstructured-CPU-fromscratch. |

Full details: `docs/LANDSCAPE.md`.

---

## Documentation

- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) — project thesis and architecture.
- [`docs/LANDSCAPE.md`](docs/LANDSCAPE.md) — honest survey of the sparse-ML ecosystem.
- [`docs/development.md`](docs/development.md) — canonical
  editable-install setup (especially the `--no-build-isolation`
  story on macOS) and developer troubleshooting.
- [`docs/design/*.md`](docs/design/) — design docs written before
  the code they describe (Padded-CSR, SpMM, SparseLinear, Router,
  RigL, NEON dW, AVX2 dW, retired AVX2 forward).
- [`docs/demos/milestone_*.md`](docs/demos/) — per-milestone writeups
  with measured results and text samples. Highlights:
  - [`milestone_10.md`](docs/demos/milestone_10.md) — the v0.1
    launch artifact (10M-param transformer end-to-end).
  - [`milestone_12.md`](docs/demos/milestone_12.md) — NEON dW
    kernel, ~2× end-to-end on Apple Silicon.
  - [`milestone_14.md`](docs/demos/milestone_14.md) — AVX2 dW
    kernel, 3.0× end-to-end on Linux x86_64.
  - [`milestone_15.md`](docs/demos/milestone_15.md) — measurement
    and learning milestone: x86 forward SpMM is already
    AVX2-fast, so a planned hand-written AVX2 forward kernel was
    retired with documented reasoning.

---

## Contributing

Pull requests are welcome. The codebase is intentionally small and
readable; we'd rather merge a thoughtful 50-line PR than a
1,000-line refactor. See `docs/design/` for the design philosophy
and `tests/` for how we oracle-test every kernel.

If you're thinking about a new DST algorithm, start with
`sparselab/router.py` — `SET` and `RigL` are both ~50 lines of
real logic, good templates for a new subclass.

If you're thinking about kernel optimization (NEON / AVX / GPU),
the moats are in `csrc/kernels/`. Every improvement compounds for
every user forever.

---

## Acknowledgments

- The `SparsityAlgorithm` API shape is inspired by
  [Cerebras's `cstorch.sparse.SparsityAlgorithm`](https://training-api.cerebras.ai/en/latest/wsc/tutorials/sparsity.html).
  Their production-hardened API on wafer-scale hardware is the
  industrial reference for sparse training; we borrow the shape,
  diverge on the storage substrate (they use dense+mask, we use
  Padded-CSR for commodity hardware).
- The Padded-CSR layout, the NEON SpMM / dW / dense-grad kernels,
  the transpose cache, and the pluggable router design are
  original work.
- Build patterns for CI + wheel packaging learned from the
  scientific-Python ecosystem (cibuildwheel, delocate, auditwheel
  practices).
- `docs/LANDSCAPE.md` has the full audit of prior art and what we
  learned from each project.

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 [Darshan Fofadiya](https://github.com/DarshanFofadiya).
