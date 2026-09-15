# uv Environment

ClearVLA uses Python 3.12. The uv project describes the shared dependency
contract, while CUDA-specific Torch installation remains a machine concern.
This avoids silently replacing a working CUDA build with a different wheel.

## Clean environment

For a machine without an existing Torch environment:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --locked
```

Optional features are installed only when needed:

```powershell
uv sync --locked --extra vision
uv sync --locked --extra analysis
uv sync --locked --extra reference-models
```

`opencv-python-headless` is optional because image decoding and resize already
fall back to Pillow. Transformers and Diffusers are used only by reference
model paths. Matplotlib is used only by latent inspection.

## Preserve an existing CUDA Torch

When a compatible Python 3.12 environment already provides the required CUDA
Torch build, create the project environment with access to those site packages
and ask uv not to replace Torch:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$torchPython = "C:\path\to\existing-torch-env\python.exe"
uv venv .venv --clear `
  --python $torchPython `
  --system-site-packages
uv sync --locked --inexact --no-install-package torch
```

Use `uv run --frozen --no-sync ...` with this bridge environment. The
`--no-sync` flag is deliberate: an automatic exact sync could install a
different Torch wheel into `.venv` and shadow the existing CUDA build.

## Validation without the full trunk

Static checks do not import or instantiate the model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_static.ps1
```

The lightweight test suite compiles the source and runs only parser, dataset
probe, temporal DCT, and physical action codec tests:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_light.ps1
```

These checks intentionally do not run a complete policy forward, load a
checkpoint, open the training dataset, start CUDA training, or invoke any
versioned mainline launcher. CUDA smoke and full-model acceptance remain
separate server-side steps under the current architecture contract.

## Lock-file policy

Commit `pyproject.toml`, `.python-version`, and `uv.lock`. Update the lock only
when the dependency contract changes:

```powershell
uv lock
uv lock --check
```
