"""Small CUDA probe for deterministic grid-sample alternatives.

This is intentionally standalone: it does not import the ClearVLA model.  It
compares the native CUDA sampler with PyTorch's Inductor decomposition and a
native-forward/decomposition-backward autograd wrapper on the shapes used by
the raw-flow refiners.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch._decomp.decompositions import _grid_sampler_2d


def _decomp(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return _grid_sampler_2d(
        value,
        grid,
        interpolation_mode=0,
        padding_mode=0,
        align_corners=True,
        _expand_grid=False,
    )


def _manual_bilinear_zero(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Inline the decomposition without nested helper/generator functions."""

    n, channels, height, width = value.shape
    # Match ATen's CUDA expression tree exactly; the algebraically equivalent
    # ``coord * scale + offset`` rounds differently in FP32.
    x = ((grid[..., 0] + 1.0) / 2.0) * (width - 1)
    y = ((grid[..., 1] + 1.0) / 2.0) * (height - 1)
    x0 = x.floor()
    y0 = y.floor()
    x1 = x0 + 1
    y1 = y0 + 1
    w00 = (x1 - x) * (y1 - y)
    w10 = (x - x0) * (y1 - y)
    w01 = (x1 - x) * (y - y0)
    w11 = (x - x0) * (y - y0)
    nidx = torch.arange(n, device=value.device).view(n, 1, 1, 1)
    cidx = torch.arange(channels, device=value.device).view(1, channels, 1, 1)

    valid00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    valid10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    valid01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    valid11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)
    ix00 = torch.where(valid00, x0, 0).to(torch.int64).view(n, 1, *x0.shape[1:])
    ix10 = torch.where(valid10, x1, 0).to(torch.int64).view(n, 1, *x1.shape[1:])
    ix01 = torch.where(valid01, x0, 0).to(torch.int64).view(n, 1, *x0.shape[1:])
    ix11 = torch.where(valid11, x1, 0).to(torch.int64).view(n, 1, *x1.shape[1:])
    iy00 = torch.where(valid00, y0, 0).to(torch.int64).view(n, 1, *y0.shape[1:])
    iy10 = torch.where(valid10, y0, 0).to(torch.int64).view(n, 1, *y0.shape[1:])
    iy01 = torch.where(valid01, y1, 0).to(torch.int64).view(n, 1, *y1.shape[1:])
    iy11 = torch.where(valid11, y1, 0).to(torch.int64).view(n, 1, *y1.shape[1:])
    result00 = value[nidx, cidx, iy00, ix00] * w00.view(n, 1, *w00.shape[1:])
    result10 = value[nidx, cidx, iy10, ix10] * w10.view(n, 1, *w10.shape[1:])
    result01 = value[nidx, cidx, iy01, ix01] * w01.view(n, 1, *w01.shape[1:])
    result11 = value[nidx, cidx, iy11, ix11] * w11.view(n, 1, *w11.shape[1:])
    return (
        result00 * valid00.view(n, 1, *valid00.shape[1:])
        + result10 * valid10.view(n, 1, *valid10.shape[1:])
        + result01 * valid01.view(n, 1, *valid01.shape[1:])
        + result11 * valid11.view(n, 1, *valid11.shape[1:])
    )


class _NativeForwardDecompBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value, grid)
        return F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        value, grid = ctx.saved_tensors
        with torch.enable_grad():
            value_req = value.detach().requires_grad_(True)
            grid_req = grid.detach().requires_grad_(True)
            output = _decomp(value_req, grid_req)
            grad_value, grad_grid = torch.autograd.grad(
                output,
                (value_req, grid_req),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grad_value, grad_grid


def _native(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def _run(
    name: str,
    value: torch.Tensor,
    grid: torch.Tensor,
    fn: Any,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        x = value.detach().clone().requires_grad_(True)
        g = grid.detach().clone().requires_grad_(True)
        y = fn(x, g)
        y.square().mean().backward()
    torch.cuda.synchronize(value.device)
    start = time.perf_counter()
    out = None
    gx = None
    gg = None
    for _ in range(steps):
        x = value.detach().clone().requires_grad_(True)
        g = grid.detach().clone().requires_grad_(True)
        out = fn(x, g)
        out.square().mean().backward()
        gx = x.grad
        gg = g.grad
    torch.cuda.synchronize(value.device)
    elapsed = time.perf_counter() - start
    assert out is not None and gx is not None and gg is not None
    return {
        "name": name,
        "seconds": elapsed,
        "iterations_per_second": steps / elapsed,
        "output": out.detach(),
        "grad_value": gx.detach(),
        "grad_grid": gg.detach(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--side", type=int, default=84)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(123)
    value = torch.randn(
        args.batch, args.channels, args.side, args.side, device=device
    )
    grid = torch.randn(args.batch, args.side, args.side, 2, device=device).tanh()
    native = _run("native", value, grid, _native, args.warmup, args.steps)
    decomp = _run("decomposition", value, grid, _decomp, args.warmup, args.steps)
    manual = _run(
        "manual", value, grid, _manual_bilinear_zero, args.warmup, args.steps
    )
    compiled_manual = None
    compile_errors: dict[str, str] = {}
    if args.compile:
        try:
            compiled_manual_fn = torch.compile(
                _manual_bilinear_zero, fullgraph=True, dynamic=False
            )
            compiled_manual = _run(
                "compiled_manual",
                value,
                grid,
                compiled_manual_fn,
                args.warmup,
                args.steps,
            )
        except Exception as error:  # pragma: no cover - environment probe
            compile_errors["manual"] = f"{type(error).__name__}: {error}"
    compiled_decomp = None
    if args.compile:
        try:
            compiled_fn = torch.compile(_decomp, fullgraph=True, dynamic=False)
            compiled_decomp = _run(
                "compiled_decomposition",
                value,
                grid,
                compiled_fn,
                args.warmup,
                args.steps,
            )
        except Exception as error:  # pragma: no cover - environment probe
            compile_errors["decomposition"] = f"{type(error).__name__}: {error}"
    custom = _run(
        "native_forward_decomp_backward",
        value,
        grid,
        _NativeForwardDecompBackward.apply,
        args.warmup,
        args.steps,
    )

    def stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
        delta = (lhs.float() - rhs.float()).abs()
        return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}

    timing_results = {
        "native": native,
        "decomposition": decomp,
        "manual": manual,
        "native_forward_decomp_backward": custom,
    }
    if compiled_manual is not None:
        timing_results["compiled_manual"] = compiled_manual
    if compiled_decomp is not None:
        timing_results["compiled_decomposition"] = compiled_decomp
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": [args.batch, args.channels, args.side, args.side],
        "native_vs_decomp": {
            "output": stats(native["output"], decomp["output"]),
            "grad_value": stats(native["grad_value"], decomp["grad_value"]),
            "grad_grid": stats(native["grad_grid"], decomp["grad_grid"]),
        },
        "native_vs_manual": {
            "output": stats(native["output"], manual["output"]),
            "grad_value": stats(native["grad_value"], manual["grad_value"]),
            "grad_grid": stats(native["grad_grid"], manual["grad_grid"]),
        },
        "native_vs_custom": {
            "output": stats(native["output"], custom["output"]),
            "grad_value": stats(native["grad_value"], custom["grad_value"]),
            "grad_grid": stats(native["grad_grid"], custom["grad_grid"]),
        },
        "timing": {
            key: {
                "seconds": item["seconds"],
                "iterations_per_second": item["iterations_per_second"],
            }
            for key, item in timing_results.items()
        },
    }
    if compiled_decomp is not None:
        result["native_vs_compiled_decomp"] = {
            "output": stats(native["output"], compiled_decomp["output"]),
            "grad_value": stats(native["grad_value"], compiled_decomp["grad_value"]),
            "grad_grid": stats(native["grad_grid"], compiled_decomp["grad_grid"]),
        }
    if compiled_manual is not None:
        result["native_vs_compiled_manual"] = {
            "output": stats(native["output"], compiled_manual["output"]),
            "grad_value": stats(native["grad_value"], compiled_manual["grad_value"]),
            "grad_grid": stats(native["grad_grid"], compiled_manual["grad_grid"]),
        }
    if compile_errors:
        result["compile_errors"] = compile_errors
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
