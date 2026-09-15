from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        summary.update({"finite_ratio": 1.0, "min": None, "max": None, "mean": None, "std": None})
        return summary
    if not (detached.is_floating_point() or detached.is_complex()):
        summary.update(
            {
                "finite_ratio": 1.0,
                "min": int(detached.min().cpu()),
                "max": int(detached.max().cpu()),
                "mean": float(detached.float().mean().cpu()),
                "std": float(detached.float().std(unbiased=False).cpu()),
            }
        )
        return summary
    finite = torch.isfinite(detached)
    finite_ratio = float(finite.float().mean().cpu())
    summary["finite_ratio"] = finite_ratio
    if bool(finite.any()):
        finite_values = detached[finite].float()
        summary.update(
            {
                "min": float(finite_values.min().cpu()),
                "max": float(finite_values.max().cpu()),
                "mean": float(finite_values.mean().cpu()),
                "std": float(finite_values.std(unbiased=False).cpu()),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None, "std": None})
    return summary


def assert_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if not (value.is_floating_point() or value.is_complex()):
        return
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite tensor {name}: {tensor_summary(value)}")


def assert_finite_batch(batch: Mapping[str, torch.Tensor]) -> None:
    for key, value in batch.items():
        assert_finite_tensor(value, name=f"batch.{key}")


def assert_finite_gradients(
    parameters: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    for name, parameter in _named_parameters(parameters):
        if parameter.grad is not None:
            assert_finite_tensor(parameter.grad, name=f"gradient.{name}")


def assert_finite_parameters(
    parameters: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    for name, parameter in _named_parameters(parameters):
        assert_finite_tensor(parameter, name=f"parameter.{name}")


def assert_finite_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    assert_finite_tensor(
                        value, name=f"optimizer.group{group_idx}.param{param_idx}.{key}"
                    )


def _named_parameters(
    parameters: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for idx, item in enumerate(parameters):
        if isinstance(item, tuple):
            name, parameter = item
            yield str(name), parameter
        else:
            yield str(idx), item


def cpu_batch_copy(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in batch.items()}


def save_nan_debug_bundle(
    path: Path,
    *,
    batch: Mapping[str, torch.Tensor],
    epoch: int,
    batch_index: int,
    global_step: int,
    phase: str,
    error: BaseException,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "clearvla-nan-debug-v1",
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "global_step": int(global_step),
        "phase": str(phase),
        "error_type": type(error).__name__,
        "error": str(error),
        "batch": cpu_batch_copy(batch),
        "batch_summaries": {key: tensor_summary(value) for key, value in batch.items()},
        "extra": dict(extra or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
