"""Strict loader for a precomputed T5 goal condition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def load_t5_condition(
    path: str | Path,
    *,
    max_tokens: int,
    expected_width: int,
    allow_null: bool = False,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return CPU float32 ``[1,L,D]`` tokens and a boolean mask.

    Formal runs fail when the file is absent.  The all-zero null condition is
    available only through the explicit smoke-only ``allow_null`` argument.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        if not allow_null:
            raise FileNotFoundError(f"required T5 condition does not exist: {source}")
        tokens = torch.zeros(1, 1, expected_width, dtype=torch.float32)
        mask = torch.ones(1, 1, dtype=torch.bool)
        return tokens, mask, {"source": "explicit_null_goal_smoke"}
    if source.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(f"T5 condition must be .pt/.pth, got {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    mask_value: object | None = None
    if isinstance(payload, dict):
        token_value: object | None = None
        for name in (
            "tokens",
            "embeddings",
            "embedding",
            "language_embedding",
            "last_hidden_state",
        ):
            if name in payload:
                token_value = payload[name]
                break
        for name in ("mask", "attention_mask", "language_mask"):
            if name in payload:
                mask_value = payload[name]
                break
        if token_value is None:
            ignored = {"mask", "attention_mask", "language_mask"}
            tensors = [
                value
                for name, value in payload.items()
                if name not in ignored and torch.is_tensor(value)
            ]
            if len(tensors) != 1:
                raise ValueError("T5 dict does not contain one identifiable token tensor")
            token_value = tensors[0]
    else:
        token_value = payload
    raw = torch.as_tensor(token_value)
    original_shape = tuple(int(value) for value in raw.shape)
    original_dtype = str(raw.dtype).removeprefix("torch.")
    tokens = raw.detach().to(dtype=torch.float32, device="cpu")
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or int(tokens.shape[0]) != 1:
        raise ValueError(f"T5 tokens must be [L,D] or [1,L,D], got {original_shape}")
    tokens = tokens[:, :max_tokens]
    if int(tokens.shape[-1]) != expected_width:
        raise ValueError(
            f"T5 width {tokens.shape[-1]} does not match configured width {expected_width}"
        )
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("T5 condition contains NaN or infinity")
    if mask_value is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask_value, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask[:, : tokens.shape[1]]
    if tuple(mask.shape) != tuple(tokens.shape[:2]) or not bool(mask.any()):
        raise ValueError("T5 mask must align with and retain at least one token")
    return (
        tokens.contiguous(),
        mask.contiguous(),
        {
            "source": "precomputed_t5_condition",
            "path": str(source.resolve()),
            "original_shape": list(original_shape),
            "original_dtype": original_dtype,
            "effective_tokens": int(tokens.shape[1]),
        },
    )


__all__ = ["load_t5_condition"]
