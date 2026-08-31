"""Assemble an auditable T5 instruction bank from RDT's precomputed rows.

The released RDT fine-tuning data already contains ``lang_embed_0.pt`` beside
each task's HDF5 files.  Those files are the embedding for the task's original
HDF5 instruction.  This tool packages one deterministic row per exact UTF-8
instruction without loading (or downloading) T5 weights.  It deliberately
keeps the raw-text inventory and source-file identities in the resulting bank;
the generic T5 encoder builder remains available when a fresh canonical
re-encoding is explicitly desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import torch
from torch import Tensor

from clearvla.data.hdf5_episode import decode_hdf5_instruction, find_hdf5_files
from clearvla.mainline.data.language import (
    T5_ENCODER_ID,
    T5_SOURCE_MAX_TOKENS,
    instruction_sha256,
    load_t5_condition_bank,
    source_instruction_inventory_sha256,
)
from clearvla.tools.build_t5_instruction_cache import build_t5_instruction_cache_payload

RDT_PRECOMPUTED_EMBEDDING_NAME = "lang_embed_0.pt"
RDT_PRECOMPUTED_SOURCE = "rdt_precomputed_lang_embed_0"


@dataclass(frozen=True)
class RDTEmbeddingCandidate:
    """One task-local source row and its stable relative identity."""

    path: Path
    relative_path: str
    tensor: Tensor
    file_sha256: str
    tensor_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    # NumPy does not expose bfloat16 on all supported runtimes.  Hash the
    # exact 16-bit representation in that case instead of silently converting
    # it to float32 and losing source identity.
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _relative_path(path: Path, root: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"<{label}>/{path.name}"


def _load_embedding(path: Path) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        value: object | None = None
        for name in (
            "tokens",
            "embeddings",
            "embedding",
            "language_embedding",
            "last_hidden_state",
        ):
            if name in payload:
                value = payload[name]
                break
        if value is None:
            tensors = [item for item in payload.values() if torch.is_tensor(item)]
            if len(tensors) != 1:
                raise ValueError(
                    f"{path}: expected one identifiable language tensor in mapping"
                )
            value = tensors[0]
    else:
        value = payload
    tensor = torch.as_tensor(value).detach().to(device="cpu").contiguous()
    if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(
            f"{path}: RDT language embedding must be [L,D] (or [1,L,D]), "
            f"got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) != 4096:
        raise ValueError(
            f"{path}: expected non-empty [L,4096] language embedding, "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{path}: language embedding must be floating point")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{path}: language embedding contains NaN or infinity")
    return tensor


def collect_rdt_embedding_candidates(
    data_root: Path,
    *,
    pattern: str = "**/*.hdf5",
    embedding_root: Path | None = None,
) -> tuple[tuple[str, ...], Counter[str], dict[str, tuple[RDTEmbeddingCandidate, ...]]]:
    """Collect exact source instructions and task-local ``lang_embed_0`` rows.

    Every HDF5 file must have a sibling task-level ``lang_embed_0.pt``.  A
    task directory is required to contain one instruction; otherwise one
    task-local row could be silently assigned to multiple texts.
    """

    source_root = data_root.expanduser().resolve()
    row_root = (embedding_root or data_root).expanduser().resolve()
    counts: Counter[str] = Counter()
    task_instructions: dict[Path, set[str]] = defaultdict(set)
    task_candidates: dict[Path, Path] = {}
    for hdf5_path in find_hdf5_files(source_root, pattern):
        with h5py.File(hdf5_path, "r") as handle:
            instruction_dataset = handle.get("instruction")
            if not isinstance(instruction_dataset, h5py.Dataset):
                raise KeyError(f"{hdf5_path}: missing scalar HDF5 instruction")
            instruction = decode_hdf5_instruction(instruction_dataset[()])
        counts[instruction] += 1
        task_root = hdf5_path.parent.resolve()
        task_instructions[task_root].add(instruction)
        task_relative = task_root.relative_to(source_root)
        candidate_path = row_root / task_relative / RDT_PRECOMPUTED_EMBEDDING_NAME
        previous = task_candidates.setdefault(task_root, candidate_path)
        if previous != candidate_path:
            raise AssertionError("task candidate path changed during collection")

    if not counts:
        raise RuntimeError("no RDT HDF5 instructions were found")
    ambiguous = {
        _relative_path(task, source_root, label="data-root"): sorted(values)
        for task, values in task_instructions.items()
        if len(values) != 1
    }
    if ambiguous:
        raise ValueError(
            "each RDT task directory must have exactly one instruction; "
            f"ambiguous tasks={list(ambiguous.items())[:5]}"
        )

    by_instruction: dict[str, dict[Path, RDTEmbeddingCandidate]] = defaultdict(dict)
    for task_root, values in task_instructions.items():
        instruction = next(iter(values))
        candidate_path = task_candidates[task_root]
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"{candidate_path}: missing {RDT_PRECOMPUTED_EMBEDDING_NAME} "
                f"for instruction {instruction!r}"
            )
        tensor = _load_embedding(candidate_path)
        candidate = RDTEmbeddingCandidate(
            path=candidate_path,
            relative_path=_relative_path(candidate_path, row_root, label="embedding-root"),
            tensor=tensor,
            file_sha256=_sha256_file(candidate_path),
            tensor_sha256=_sha256_tensor(tensor),
        )
        by_instruction[instruction][candidate_path] = candidate
    return (
        tuple(sorted(counts)),
        counts,
        {
            instruction: tuple(
                sorted(values.values(), key=lambda item: item.relative_path)
            )
            for instruction, values in by_instruction.items()
        },
    )


def _select_candidates(
    candidates: tuple[RDTEmbeddingCandidate, ...],
    *,
    instruction: str,
    duplicate_policy: str,
) -> tuple[RDTEmbeddingCandidate, dict[str, Any]]:
    if duplicate_policy not in {"error", "lexicographic"}:
        raise ValueError(f"unsupported duplicate policy: {duplicate_policy!r}")
    if not candidates:
        raise ValueError("an instruction must have at least one embedding candidate")
    distinct = {candidate.tensor_sha256 for candidate in candidates}
    if len(distinct) > 1 and duplicate_policy == "error":
        raise ValueError(
            "duplicate RDT instruction has non-identical lang_embed_0 rows; "
            "rerun with --duplicate-policy lexicographic after reviewing the "
            f"source candidates: {[candidate.relative_path for candidate in candidates]}"
        )
    selected = candidates[0]
    record = {
        "instruction_sha256": instruction_sha256(instruction),
        "selected_relative_path": selected.relative_path,
        "selected_file_sha256": selected.file_sha256,
        "selected_tensor_sha256": selected.tensor_sha256,
        "candidate_count": len(candidates),
        "distinct_tensor_count": len(distinct),
        "candidates": [
            {
                "relative_path": candidate.relative_path,
                "file_sha256": candidate.file_sha256,
                "tensor_sha256": candidate.tensor_sha256,
                "shape": list(candidate.tensor.shape),
                "dtype": str(candidate.tensor.dtype).removeprefix("torch."),
            }
            for candidate in candidates
        ],
    }
    return selected, record


def _verify_token_lengths(
    instructions: tuple[str, ...],
    selected: tuple[RDTEmbeddingCandidate, ...],
    *,
    tokenizer_source: str,
) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional preparation dependency
        raise RuntimeError(
            "tokenizer verification requires transformers; use "
            "--skip-tokenizer-check only after reviewing source provenance"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        model_max_length=T5_SOURCE_MAX_TOKENS,
        local_files_only=True,
    )
    for instruction, candidate in zip(instructions, selected, strict=True):
        token_count = len(
            tokenizer(instruction, add_special_tokens=True)["input_ids"]
        )
        if int(candidate.tensor.shape[0]) != token_count:
            raise ValueError(
                "RDT lang_embed_0 sequence length disagrees with the local "
                f"{T5_ENCODER_ID} tokenizer for instruction {instruction!r}: "
                f"embedding={candidate.tensor.shape[0]} tokenizer={token_count}"
            )


def _inventory_digest(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_rdt_precomputed_instruction_cache(
    *,
    data_root: Path,
    output: Path,
    pattern: str = "**/*.hdf5",
    embedding_root: Path | None = None,
    policy_max_tokens: int = 32,
    duplicate_policy: str = "error",
    tokenizer_source: str = T5_ENCODER_ID,
    verify_tokenizer: bool = True,
) -> dict[str, Any]:
    """Build and atomically write a typed bank from existing RDT embeddings."""

    if policy_max_tokens <= 0 or policy_max_tokens > T5_SOURCE_MAX_TOKENS:
        raise ValueError(
            f"policy_max_tokens must be in [1,{T5_SOURCE_MAX_TOKENS}]"
        )
    instructions, counts, candidates_by_instruction = collect_rdt_embedding_candidates(
        data_root,
        pattern=pattern,
        embedding_root=embedding_root,
    )
    selected: list[RDTEmbeddingCandidate] = []
    records: list[dict[str, Any]] = []
    for instruction in instructions:
        candidate, record = _select_candidates(
            candidates_by_instruction[instruction],
            instruction=instruction,
            duplicate_policy=duplicate_policy,
        )
        selected.append(candidate)
        records.append(record)
    if verify_tokenizer:
        _verify_token_lengths(instructions, tuple(selected), tokenizer_source=tokenizer_source)

    dtypes = [candidate.tensor.dtype for candidate in selected]
    output_dtype = dtypes[0]
    for dtype in dtypes[1:]:
        output_dtype = torch.promote_types(output_dtype, dtype)
    tokens = torch.zeros(
        len(selected), policy_max_tokens, 4096, dtype=output_dtype, device="cpu"
    )
    mask = torch.zeros(len(selected), policy_max_tokens, dtype=torch.bool, device="cpu")
    for row, candidate in enumerate(selected):
        retained = min(policy_max_tokens, int(candidate.tensor.shape[0]))
        value = candidate.tensor[:retained].to(dtype=output_dtype)
        tokens[row, :retained] = value
        mask[row, :retained] = True
        records[row]["selected_policy_tokens"] = retained
        records[row]["selected_policy_tensor_sha256"] = _sha256_tensor(
            tokens[row, :retained]
        )

    source_instructions = [
        instruction
        for instruction, count in counts.items()
        for _ in range(int(count))
    ]
    payload = build_t5_instruction_cache_payload(
        instructions=instructions,
        tokens=tokens,
        attention_mask=mask,
        model_source=RDT_PRECOMPUTED_SOURCE,
        source_episode_count=sum(counts.values()),
        source_instruction_inventory_sha256=source_instruction_inventory_sha256(
            source_instructions
        ),
    )
    distinct_variant_groups = sum(
        int(record["distinct_tensor_count"] > 1) for record in records
    )
    payload.update(
        {
            "transformers_version": "not_used_precomputed",
            "embedding_source": RDT_PRECOMPUTED_SOURCE,
            "embedding_name": RDT_PRECOMPUTED_EMBEDDING_NAME,
            "embedding_duplicate_policy": duplicate_policy,
            "embedding_tokenizer_verification": (
                "local_only" if verify_tokenizer else "skipped_explicitly"
            ),
            "embedding_candidate_file_count": sum(
                int(record["candidate_count"]) for record in records
            ),
            "embedding_variant_group_count": distinct_variant_groups,
            "embedding_inventory_sha256": _inventory_digest(records),
            "embedding_records": records,
        }
    )
    output_path = output.expanduser().resolve()
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
        max_tokens=policy_max_tokens,
        expected_width=4096,
    )
    return {
        "output": str(output_path),
        "episodes": sum(counts.values()),
        "instructions": len(loaded.instructions),
        "tokens": list(loaded.tokens.shape),
        "dtype": str(tokens.dtype).removeprefix("torch."),
        "embedding_source": RDT_PRECOMPUTED_SOURCE,
        "embedding_duplicate_policy": duplicate_policy,
        "embedding_variant_group_count": distinct_variant_groups,
        "embedding_inventory_sha256": payload["embedding_inventory_sha256"],
        "source_instruction_inventory_sha256": payload[
            "source_instruction_inventory_sha256"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a typed T5 bank from RDT lang_embed_0.pt rows without T5 weights"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--policy-max-tokens", type=int, default=32)
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "lexicographic"),
        default="error",
    )
    parser.add_argument("--tokenizer", default=T5_ENCODER_ID)
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing bank: {output}")
    summary = build_rdt_precomputed_instruction_cache(
        data_root=args.data_root,
        output=output,
        pattern=args.glob,
        embedding_root=args.embedding_root,
        policy_max_tokens=args.policy_max_tokens,
        duplicate_policy=args.duplicate_policy,
        tokenizer_source=args.tokenizer,
        verify_tokenizer=not args.skip_tokenizer_check,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RDTEmbeddingCandidate",
    "RDT_PRECOMPUTED_EMBEDDING_NAME",
    "RDT_PRECOMPUTED_SOURCE",
    "build_rdt_precomputed_instruction_cache",
    "collect_rdt_embedding_candidates",
]
