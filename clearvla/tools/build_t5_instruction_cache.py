"""Precompute the source-backed T5-XXL condition bank for RDT instructions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import h5py
import torch
from torch import Tensor

from clearvla.data.hdf5_episode import (
    decode_hdf5_instruction,
    find_hdf5_files,
)
from clearvla.mainline.data.language import (
    T5_ENCODER_ID,
    T5_INSTRUCTION_CACHE_SCHEMA,
    T5_SOURCE_MAX_TOKENS,
    instruction_inventory_sha256,
    instruction_sha256,
    load_t5_condition_bank,
    source_instruction_inventory_sha256,
)


def collect_hdf5_instructions(
    root: Path,
    *,
    pattern: str = "**/*.hdf5",
) -> tuple[tuple[str, ...], Counter[str]]:
    source = Path(root).expanduser().resolve()
    counts: Counter[str] = Counter()
    for path in find_hdf5_files(source, pattern):
        with h5py.File(path, "r") as handle:
            instruction_dataset = handle.get("instruction")
            if not isinstance(instruction_dataset, h5py.Dataset):
                raise KeyError(f"{path}: missing scalar HDF5 instruction")
            counts[decode_hdf5_instruction(instruction_dataset[()])] += 1
    instructions = tuple(sorted(counts))
    if not instructions:
        raise RuntimeError("no HDF5 instructions were found")
    return instructions, counts


def build_t5_instruction_cache_payload(
    *,
    instructions: Sequence[str],
    tokens: Tensor,
    attention_mask: Tensor,
    model_source: str,
    source_episode_count: int,
    source_instruction_inventory_sha256: str,
) -> dict[str, Any]:
    ordered = tuple(str(value) for value in instructions)
    if (
        not ordered
        or any(not value.strip() for value in ordered)
        or tuple(sorted(set(ordered))) != ordered
    ):
        raise ValueError("instructions must be non-empty, sorted, and unique")
    if int(source_episode_count) <= 0:
        raise ValueError("source_episode_count must be positive")
    source_digest = str(source_instruction_inventory_sha256)
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_digest.lower()
    ):
        raise ValueError("source instruction inventory identity must be SHA-256")
    token_tensor = torch.as_tensor(tokens).detach().to(device="cpu").contiguous()
    mask = torch.as_tensor(attention_mask, dtype=torch.bool).detach().to(device="cpu")
    if token_tensor.ndim != 3 or mask.ndim != 2:
        raise ValueError("condition tensors must be [N,L,D] and [N,L]")
    if tuple(mask.shape) != tuple(token_tensor.shape[:2]) or len(ordered) != len(
        token_tensor
    ):
        raise ValueError("condition rows, tokens, and mask do not align")
    if not bool(mask.any(dim=1).all()) or not bool(torch.isfinite(token_tensor).all()):
        raise ValueError("every condition row must be finite and contain a valid token")
    token_tensor = torch.where(mask[..., None], token_tensor, torch.zeros_like(token_tensor))
    return {
        "schema": T5_INSTRUCTION_CACHE_SCHEMA,
        "encoder_id": T5_ENCODER_ID,
        "model_source": str(model_source),
        "source_tokenizer_max_length": T5_SOURCE_MAX_TOKENS,
        "policy_max_tokens": int(token_tensor.shape[1]),
        "embedding_width": int(token_tensor.shape[2]),
        "instructions": list(ordered),
        "instruction_sha256": [instruction_sha256(value) for value in ordered],
        "instruction_inventory_sha256": instruction_inventory_sha256(ordered),
        "source_episode_count": int(source_episode_count),
        "source_instruction_inventory_sha256": source_digest,
        "tokens": token_tensor,
        "attention_mask": mask.contiguous(),
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def encode_instruction_bank(
    instructions: Sequence[str],
    *,
    model_source: str,
    device: torch.device,
    dtype: torch.dtype,
    policy_max_tokens: int,
    batch_size: int,
    local_files_only: bool,
) -> tuple[Tensor, Tensor, str]:
    if policy_max_tokens <= 0 or policy_max_tokens > T5_SOURCE_MAX_TOKENS:
        raise ValueError(
            f"policy_max_tokens must be in [1,{T5_SOURCE_MAX_TOKENS}]"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import transformers
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:  # pragma: no cover - optional preparation dependency
        raise RuntimeError("T5 cache construction requires transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        model_max_length=T5_SOURCE_MAX_TOKENS,
        local_files_only=local_files_only,
    )
    encoder = T5EncoderModel.from_pretrained(
        model_source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device)
    encoder.requires_grad_(False).eval()
    width = int(encoder.config.d_model)
    if width != 4096:
        raise ValueError(f"{T5_ENCODER_ID} must expose width 4096, got {width}")
    output = torch.zeros(
        len(instructions),
        policy_max_tokens,
        width,
        dtype=dtype,
        device="cpu",
    )
    output_mask = torch.zeros(
        len(instructions), policy_max_tokens, dtype=torch.bool, device="cpu"
    )
    for start in range(0, len(instructions), batch_size):
        end = min(len(instructions), start + batch_size)
        encoded = tokenizer(
            list(instructions[start:end]),
            max_length=T5_SOURCE_MAX_TOKENS,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device=device)
        attention_mask = encoded["attention_mask"].to(device=device)
        with torch.no_grad():
            hidden = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        retained = min(policy_max_tokens, int(hidden.shape[1]))
        retained_mask = attention_mask[:, :retained].to(device="cpu", dtype=torch.bool)
        retained_tokens = hidden[:, :retained].detach().to(device="cpu", dtype=dtype)
        output[start:end, :retained] = torch.where(
            retained_mask[..., None],
            retained_tokens,
            torch.zeros_like(retained_tokens),
        )
        output_mask[start:end, :retained] = retained_mask
        print(
            f"[t5-instruction-cache] encoded={end}/{len(instructions)}",
            flush=True,
        )
    return output, output_mask, str(transformers.__version__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode exact HDF5 instructions with the official RDT T5-v1.1-XXL contract"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=T5_ENCODER_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--policy-max-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("T5 instruction cache output must be .pt/.pth")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing cache: {output_path}")
    device = _device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    instructions, counts = collect_hdf5_instructions(args.data_root, pattern=args.glob)
    tokens, mask, transformers_version = encode_instruction_bank(
        instructions,
        model_source=args.model,
        device=device,
        dtype=dtype,
        policy_max_tokens=args.policy_max_tokens,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    payload = build_t5_instruction_cache_payload(
        instructions=instructions,
        tokens=tokens,
        attention_mask=mask,
        model_source=args.model,
        source_episode_count=sum(counts.values()),
        source_instruction_inventory_sha256=source_instruction_inventory_sha256(
            [
                instruction
                for instruction, count in counts.items()
                for _ in range(int(count))
            ]
        ),
    )
    payload["transformers_version"] = transformers_version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    loaded = load_t5_condition_bank(
        output_path,
        max_tokens=args.policy_max_tokens,
        expected_width=int(tokens.shape[-1]),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "episodes": sum(counts.values()),
                "instructions": len(loaded.instructions),
                "tokens": list(loaded.tokens.shape),
                "encoder_id": T5_ENCODER_ID,
                "source_tokenizer_max_length": T5_SOURCE_MAX_TOKENS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_t5_instruction_cache_payload",
    "collect_hdf5_instructions",
    "encode_instruction_bank",
]
