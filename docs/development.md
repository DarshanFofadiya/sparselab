# Development setup

Canonical reference for setting up a SparseLab development environment.
The recommended editable-install command is:

```bash
# Pre-install build dependencies into the runtime environment.
pip install --upgrade setuptools wheel pybind11 'torch>=2.8'

# Then editable install with build isolation disabled.
pip install -e '.[dev]' --no-build-isolation
```

On **macOS** the `--no-build-isolation` flag is **required**, not
optional. On **Linux** it is recommended (faster, consistent with
macOS). This page explains why.

If you just want to get going, follow [CONTRIBUTING.md](../CONTRIBUTING.md)
and come back here only if something goes wrong.

---

## Why `--no-build-isolation` matters on macOS

### The problem (issue #18)

PyTorch ships its own `libomp.dylib` inside its wheel
(`<site-packages>/torch/lib/libomp.dylib`). Our C++ extension also
needs OpenMP, so it links against `libomp` at build time. macOS does
not have a system-wide `libomp` — developers install it via
`brew install libomp`, which lives at
`/opt/homebrew/opt/libomp/lib/libomp.dylib` (Apple Silicon) or
`/usr/local/opt/libomp/lib/libomp.dylib` (Intel).

If our extension and PyTorch both load *different* libomp libraries
into the same Python process, the OpenMP runtime detects the
duplicate and either aborts (`OMP: Error #15`) or silently produces
undefined behavior — segfaults are common. We need our `.so` to
resolve `libomp.dylib` to the *same* libomp PyTorch already loaded.

### How we ensure that on macOS

`setup.py` has a custom `BuildExtWithRepair` build_ext that runs
after each compiled extension and rewrites the libomp install_name
inside the `.so`. It does one of two things, depending on whether
`import torch` succeeds at build time:

| Situation | What we do | When this applies |
|---|---|---|
| Torch is importable at build time and `torch/lib/libomp.dylib` exists | Rewrite install_name to the **absolute path** of that libomp | Editable install with `--no-build-isolation` |
| Torch is not importable at build time | Rewrite install_name to `@rpath/libomp.dylib` | Wheel build (cibuildwheel) — `repair_wheel_macos.sh` adds the right rpath afterwards |

The editable-install path needs the absolute install_name because the
`.so` lives in `<repo>/sparselab/_core.so` (in your source tree) but
torch lives in `<site-packages>/torch/lib/`. There is no relative
path that connects them robustly across pip's editable-install
mechanism, so we hard-code the absolute path. That path is local to
your machine, baked into a `.so` that never leaves your repo.

### Why build isolation breaks this

`pip install -e .` (without the flag) creates a fresh, ephemeral
build environment to run `setup.py`. That environment installs its
own copy of torch — a *different* torch from the one in your
runtime venv — at a temporary path that gets deleted after the
build. If we hard-code that path into the `.so`'s install_name, the
absolute path won't exist at runtime, and `import sparselab` raises
`ImportError: Library not loaded: @rpath/libomp.dylib`.

`--no-build-isolation` tells pip to use the *current* environment's
torch as the build-time torch, so the path baked into the `.so` is
the same path the user imports at runtime. That's why the flag is
mandatory.

### The "I forgot the flag" symptom

```
ImportError: dlopen(.../sparselab/_core.cpython-311-darwin.so):
Library not loaded: @rpath/libomp.dylib
Reason: tried: '/path/to/build-env/torch/lib/libomp.dylib'
        (no such file)
```

Recovery is one command:

```bash
pip install -e '.[dev]' --no-build-isolation --force-reinstall
```

The `--force-reinstall` is so pip rebuilds the `.so` (the extant
one has the wrong baked path).

---

## Why `--no-build-isolation` is recommended on Linux too

Two reasons:

1. **Speed.** Skipping the ephemeral build env is ~2× faster on
   incremental rebuilds (no torch + pybind11 + setuptools download
   + extract per build).
2. **Consistency.** One command across platforms is one fewer thing
   to remember and one fewer source of CI / local divergence.

Linux uses `libgomp` (shipped with gcc) rather than `libomp`, and
`libgomp` doesn't have the duplicate-loaded-library issue, so the
flag isn't *required* on Linux — but there's no reason not to use it.

---

## What the wheels do (for reference, not for daily use)

PyPI wheels (`pip install sparselab` from the index) take a different
code path entirely:

1. cibuildwheel builds the extension with no torch in the build env.
2. `BuildExtWithRepair` rewrites install_name to `@rpath/libomp.dylib`
   (the wheel-build branch).
3. `scripts/repair_wheel_macos.sh` runs as cibuildwheel's
   repair-wheel-command. It adds rpath `@loader_path/../torch/lib`
   (a relative path that resolves correctly inside the eventual
   `site-packages/sparselab/` ↔ `site-packages/torch/lib/`
   layout) and strips any absolute build-time rpaths.
4. The wheel ships with an `@rpath/libomp.dylib` install_name plus
   one relative rpath. At end-user import time, that rpath resolves
   to torch's bundled libomp inside the user's site-packages.

End users never run `setup.py` directly; they install the wheel and
the post-built rpath layout works for them. Issue #18 only affects
developers running editable installs.

---

## Quick reference card

| Task | Command |
|---|---|
| First-time setup (macOS or Linux) | `pip install --upgrade setuptools wheel pybind11 'torch>=2.8'` then `pip install -e '.[dev]' --no-build-isolation` |
| Fast incremental rebuild after editing C++ | `pip install -e '.[dev]' --no-build-isolation -v` |
| Run the test suite | `pytest tests/ -q` |
| Run a specific demo | `python examples/demo_XX_*.py` |
| Re-run on a totally fresh build | `pip uninstall -y sparselab && rm -rf build/` then re-run the install command |
| Diagnose a libomp issue on macOS | `otool -L sparselab/_core*.so` (look at the libomp line) and `otool -l sparselab/_core*.so \| grep -A2 LC_RPATH` |
