"""Precompute the exact per-instruction T5-XXL bank used by ClearVLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from clearvla.data.instructions import instruction_key, normalize_instruction
from clearvla.mainline.data.language import CALVIN_LANGUAGE_BANK_SCHEMA, T5_ENCODER_ID


def _instruction_inventory(inventory_path: Path) -> dict[str, str]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != "clearvla-instruction-inventory-v1":
        raise ValueError("instruction inventory schema differs")
    rows = inventory.get("instructions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("instruction inventory is empty")
    instructions: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("instruction inventory rows must be mappings")
        key = str(row["key"])
        instruction = normalize_instruction(str(row["instruction"]))
        if instruction_key(instruction) != key:
            raise ValueError("instruction inventory key/content mismatch")
        previous = instructions.get(key)
        if previous is not None and previous != instruction:
            raise ValueError("instruction inventory key has conflicting text")
        instructions[key] = instruction
    return instructions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_language_bank(
    inventory_path: Path,
    output: Path,
    *,
    model_name: str,
    max_tokens: int,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, object]:
    from transformers import AutoTokenizer, T5EncoderModel

    instructions = _instruction_inventory(inventory_path)
    # T5's SentencePiece tokenizer is intentionally loaded in its slow/native
    # form.  Recent Transformers releases otherwise try a tiktoken/protobuf
    # conversion for ``T5TokenizerFast``; that conversion is unnecessary for
    # this one-time, content-addressed bank build and makes an otherwise
    # healthy isolated environment fail merely because optional fast-tokenizer
    # packages are absent.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=local_files_only,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device).eval()
    conditions: dict[str, dict[str, object]] = {}
    with torch.inference_mode():
        for key in sorted(instructions):
            instruction = instructions[key]
            encoded = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
            input_ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            tokens = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            if int(tokens.shape[-1]) != 4096:
                raise ValueError(f"T5 encoder width must be 4096, got {tokens.shape[-1]}")
            conditions[key] = {
                "instruction": instruction,
                "tokens": tokens.to(device="cpu", dtype=torch.float16).contiguous(),
                "mask": mask.to(device="cpu", dtype=torch.bool).contiguous(),
            }
    payload = {
        "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
        "encoder_model": model_name,
        "storage_dtype": "float16",
        "max_tokens": int(max_tokens),
        "conditions": conditions,
    }
    _atomic_torch_save(payload, output)
    return {
        "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
        "output": str(output.resolve()),
        "encoder_model": model_name,
        "conditions": len(conditions),
        "size_bytes": output.stat().st_size,
    }


def derive_language_bank(
    source_bank: Path,
    inventory_path: Path,
    output: Path,
) -> dict[str, object]:
    """Copy an exact instruction subset from an existing external bank.

    This keeps benchmark artifact preparation independent of the training
    graph and avoids loading T5-XXL again when a content-addressed superset is
    already available.
    """

    source_bank = source_bank.expanduser().resolve()
    if not source_bank.is_file():
        raise FileNotFoundError(f"source language bank does not exist: {source_bank}")
    payload = torch.load(source_bank, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("source language bank must be a mapping")
    if str(payload.get("schema", "")) != CALVIN_LANGUAGE_BANK_SCHEMA:
        raise ValueError("source language bank schema differs")
    if str(payload.get("encoder_model", "")) != T5_ENCODER_ID:
        raise ValueError(f"source language bank must use {T5_ENCODER_ID}")
    source_conditions = payload.get("conditions")
    if not isinstance(source_conditions, Mapping) or not source_conditions:
        raise ValueError("source language bank has no conditions")

    instructions = _instruction_inventory(inventory_path)
    selected: dict[str, object] = {}
    for key in sorted(instructions):
        raw_condition = source_conditions.get(key)
        if not isinstance(raw_condition, Mapping):
            raise KeyError(f"source language bank does not cover instruction key {key!r}")
        expected_instruction = instructions[key]
        actual_instruction = normalize_instruction(
            str(raw_condition.get("instruction", ""))
        )
        if actual_instruction != expected_instruction:
            raise ValueError(
                f"source language condition {key!r} has mismatched instruction text"
            )
        raw_tokens = torch.as_tensor(raw_condition.get("tokens"))
        if raw_tokens.ndim == 2:
            raw_tokens = raw_tokens.unsqueeze(0)
        if (
            raw_tokens.ndim != 3
            or int(raw_tokens.shape[0]) != 1
            or int(raw_tokens.shape[1]) <= 0
            or int(raw_tokens.shape[2]) != 4096
            or not bool(torch.isfinite(raw_tokens).all())
        ):
            raise ValueError(f"source language condition {key!r} has invalid tokens")
        raw_mask = raw_condition.get("mask", raw_condition.get("attention_mask"))
        if raw_mask is not None:
            mask = torch.as_tensor(raw_mask, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape) != tuple(raw_tokens.shape[:2]) or not bool(mask.any()):
                raise ValueError(f"source language condition {key!r} has invalid mask")
        selected[key] = dict(raw_condition)

    derived_payload = {
        "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
        "encoder_model": T5_ENCODER_ID,
        "storage_dtype": str(payload.get("storage_dtype", "unknown")),
        "max_tokens": int(payload.get("max_tokens", 0)),
        "conditions": selected,
        "derived_from": {
            "path": str(source_bank),
            "size_bytes": int(source_bank.stat().st_size),
            "sha256": _sha256(source_bank),
        },
        "instruction_inventory": {
            "path": str(inventory_path.expanduser().resolve()),
            "sha256": _sha256(inventory_path.expanduser().resolve()),
        },
    }
    _atomic_torch_save(derived_payload, output)
    return {
        "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
        "output": str(output.resolve()),
        "encoder_model": T5_ENCODER_ID,
        "conditions": len(selected),
        "size_bytes": output.stat().st_size,
        "source_bank_sha256": derived_payload["derived_from"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-bank",
        type=Path,
        help="Derive an exact subset from this existing external language bank.",
    )
    parser.add_argument("--model", default=T5_ENCODER_ID)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.source_bank is not None:
        result = derive_language_bank(args.source_bank, args.inventory, args.output)
    else:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else torch.device(args.device)
        )
        result = build_language_bank(
            args.inventory,
            args.output,
            model_name=args.model,
            max_tokens=args.max_tokens,
            device=device,
            local_files_only=args.local_files_only,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
