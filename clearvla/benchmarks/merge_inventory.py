"""Merge content-addressed benchmark instruction inventories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.data.instructions import instruction_key, normalize_instruction

from .common import atomic_json


def merge_instruction_inventories(
    inputs: tuple[Path, ...],
    output: Path,
) -> dict[str, object]:
    """Union one or more ``instructions.json`` files without losing exact text."""

    if not inputs:
        raise ValueError("at least one instruction inventory is required")
    merged: dict[str, str] = {}
    sources: list[str] = []
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"instruction inventory does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("instructions") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"instruction inventory is empty or malformed: {path}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"instruction inventory row is malformed: {path}")
            instruction = normalize_instruction(str(row.get("instruction", "")))
            key = str(row.get("key", ""))
            if instruction_key(instruction) != key:
                raise ValueError(f"instruction key/content mismatch in {path}: {key!r}")
            previous = merged.get(key)
            if previous is not None and previous != instruction:
                raise ValueError(f"hash collision with different text for key {key!r}")
            merged[key] = instruction
        sources.append(str(path.resolve()))
    result = {
        "schema": "clearvla-instruction-inventory-v1",
        "merged_from": sources,
        "instructions": [
            {"key": key, "instruction": merged[key]} for key in sorted(merged)
        ],
    }
    atomic_json(output, result)
    return {
        "schema": result["schema"],
        "output": str(output.resolve()),
        "source_count": len(inputs),
        "instruction_count": len(merged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge_instruction_inventories(tuple(args.inputs), args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["merge_instruction_inventories"]


