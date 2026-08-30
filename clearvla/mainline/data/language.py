"""Strict loader for a precomputed T5 goal condition."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

T5_INSTRUCTION_CACHE_SCHEMA = "clearvla-t5-instruction-cache-v1"
T5_ENCODER_ID = "google/t5-v1_1-xxl"
T5_SOURCE_MAX_TOKENS = 120


def instruction_sha256(instruction: str) -> str:
    value = str(instruction)
    if not value.strip():
        raise ValueError("instruction must be non-empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def instruction_inventory_sha256(instructions: Sequence[str]) -> str:
    encoded = json.dumps(
        [str(value) for value in instructions],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_instruction_inventory_sha256(instructions: Sequence[str]) -> str:
    """Digest the exact source instruction multiset used to build a bank."""

    counts = Counter(str(value) for value in instructions)
    if not counts or any(not value.strip() for value in counts):
        raise ValueError("source instructions must be non-empty text")
    encoded = json.dumps(
        sorted((instruction, int(count)) for instruction, count in counts.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class T5ConditionBank:
    """CPU condition rows plus an exact instruction-to-row mapping."""

    tokens: Tensor  # float32 [N,L,D]
    mask: Tensor  # bool [N,L]
    instructions: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def is_instruction_bank(self) -> bool:
        return bool(self.instructions)

    def condition_indices(self, values: Sequence[str | None]) -> Tensor:
        if not self.is_instruction_bank:
            return torch.zeros(len(values), dtype=torch.long)
        index = {instruction: row for row, instruction in enumerate(self.instructions)}
        missing = sorted(
            {
                "<missing>" if value is None else str(value)
                for value in values
                if value is None or str(value) not in index
            }
        )
        if missing:
            raise KeyError(
                "T5 instruction cache does not cover episode instructions: "
                f"{missing[:5]}"
            )
        return torch.tensor([index[str(value)] for value in values], dtype=torch.long)


def _load_instruction_bank(
    source: Path,
    payload: dict[str, Any],
    *,
    max_tokens: int,
    expected_width: int,
) -> T5ConditionBank:
    if str(payload.get("schema", "")) != T5_INSTRUCTION_CACHE_SCHEMA:
        raise ValueError("unsupported T5 instruction cache schema")
    if str(payload.get("encoder_id", "")) != T5_ENCODER_ID:
        raise ValueError(f"instruction cache must use {T5_ENCODER_ID}")
    if int(payload.get("source_tokenizer_max_length", 0)) != T5_SOURCE_MAX_TOKENS:
        raise ValueError(
            f"instruction cache source tokenizer length must be {T5_SOURCE_MAX_TOKENS}"
        )
    instructions_value = payload.get("instructions")
    digests_value = payload.get("instruction_sha256")
    if not isinstance(instructions_value, (tuple, list)) or not isinstance(
        digests_value, (tuple, list)
    ):
        raise TypeError("instruction cache identities must be sequences")
    instructions = tuple(str(value) for value in instructions_value)
    digests = tuple(str(value) for value in digests_value)
    if not instructions or any(not value.strip() for value in instructions):
        raise ValueError("instruction cache must contain non-empty instructions")
    if tuple(sorted(instructions)) != instructions or len(set(instructions)) != len(instructions):
        raise ValueError("instruction cache instructions must be sorted and unique")
    expected_digests = tuple(instruction_sha256(value) for value in instructions)
    if digests != expected_digests:
        raise ValueError("instruction cache text digests are inconsistent")
    if str(payload.get("instruction_inventory_sha256", "")) != instruction_inventory_sha256(
        instructions
    ):
        raise ValueError("instruction cache inventory digest is inconsistent")
    source_episode_count = int(payload.get("source_episode_count", 0))
    source_inventory_digest = str(
        payload.get("source_instruction_inventory_sha256", "")
    ).lower()
    if source_episode_count <= 0:
        raise ValueError("instruction cache source episode count must be positive")
    if len(source_inventory_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_inventory_digest
    ):
        raise ValueError("instruction cache source inventory identity must be SHA-256")

    if "tokens" not in payload or "attention_mask" not in payload:
        raise KeyError("instruction cache is missing tokens or attention_mask")
    raw_tokens = torch.as_tensor(payload["tokens"])
    raw_mask = torch.as_tensor(payload["attention_mask"], dtype=torch.bool)
    if raw_tokens.ndim != 3 or raw_mask.ndim != 2:
        raise ValueError("instruction cache tensors must be [N,L,D] and [N,L]")
    stored_max_tokens = int(payload.get("policy_max_tokens", 0))
    if stored_max_tokens != int(raw_tokens.shape[1]) or stored_max_tokens < int(max_tokens):
        raise ValueError("instruction cache does not cover the requested policy token length")
    if tuple(raw_mask.shape) != tuple(raw_tokens.shape[:2]):
        raise ValueError("instruction cache mask does not align with tokens")
    if int(raw_tokens.shape[0]) != len(instructions):
        raise ValueError("instruction cache tensor rows do not match instruction identities")
    if int(raw_tokens.shape[2]) != int(expected_width):
        raise ValueError(
            f"T5 width {raw_tokens.shape[2]} does not match configured width {expected_width}"
        )
    if int(payload.get("embedding_width", 0)) != int(raw_tokens.shape[2]):
        raise ValueError("instruction cache embedding width metadata is inconsistent")
    tokens = raw_tokens[:, :max_tokens].detach().to(device="cpu", dtype=torch.float32)
    mask = raw_mask[:, :max_tokens].detach().to(device="cpu", dtype=torch.bool)
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("instruction cache contains NaN or infinity")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every instruction cache row must retain at least one token")
    if bool(tokens.masked_select(~mask[..., None].expand_as(tokens)).ne(0).any()):
        raise ValueError("masked instruction-cache tokens must be exact zero")
    metadata = {
        "source": "precomputed_t5_instruction_cache",
        "path": str(source.resolve()),
        "schema": T5_INSTRUCTION_CACHE_SCHEMA,
        "encoder_id": T5_ENCODER_ID,
        "instructions": len(instructions),
        "instruction_inventory_sha256": instruction_inventory_sha256(instructions),
        "source_episode_count": source_episode_count,
        "source_instruction_inventory_sha256": source_inventory_digest,
        "original_shape": list(raw_tokens.shape),
        "original_dtype": str(raw_tokens.dtype).removeprefix("torch."),
        "effective_tokens": int(tokens.shape[1]),
    }
    return T5ConditionBank(
        tokens=tokens.contiguous(),
        mask=mask.contiguous(),
        instructions=instructions,
        metadata=metadata,
    )


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


def load_t5_condition_bank(
    path: str | Path,
    *,
    max_tokens: int,
    expected_width: int,
    allow_null: bool = False,
) -> T5ConditionBank:
    """Load either the legacy single condition or a typed instruction bank."""

    source = Path(path).expanduser()
    if source.is_file():
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("schema") == T5_INSTRUCTION_CACHE_SCHEMA:
            return _load_instruction_bank(
                source,
                payload,
                max_tokens=max_tokens,
                expected_width=expected_width,
            )
    tokens, mask, metadata = load_t5_condition(
        source,
        max_tokens=max_tokens,
        expected_width=expected_width,
        allow_null=allow_null,
    )
    return T5ConditionBank(
        tokens=tokens,
        mask=mask,
        instructions=(),
        metadata=metadata,
    )


__all__ = [
    "T5ConditionBank",
    "T5_ENCODER_ID",
    "T5_INSTRUCTION_CACHE_SCHEMA",
    "T5_SOURCE_MAX_TOKENS",
    "instruction_inventory_sha256",
    "instruction_sha256",
    "load_t5_condition",
    "load_t5_condition_bank",
    "source_instruction_inventory_sha256",
]
