from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from clearvla.benchmarks.language_bank import derive_language_bank
from clearvla.data.instructions import instruction_key
from clearvla.mainline.data.language import CALVIN_LANGUAGE_BANK_SCHEMA, T5_ENCODER_ID


def _condition(instruction: str, value: float) -> dict[str, object]:
    return {
        "instruction": instruction,
        "tokens": torch.full((1, 3, 4096), value, dtype=torch.float16),
        "mask": torch.tensor([[True, True, True]]),
    }


def _inventory(path: Path, instruction: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "clearvla-instruction-inventory-v1",
                "instructions": [
                    {"key": instruction_key(instruction), "instruction": instruction}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_derive_language_bank_copies_only_requested_condition(tmp_path: Path) -> None:
    keep = "pick up the black bowl"
    drop = "put the mug on the plate"
    source = tmp_path / "full.pt"
    torch.save(
        {
            "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
            "encoder_model": T5_ENCODER_ID,
            "storage_dtype": "float16",
            "max_tokens": 32,
            "conditions": {
                instruction_key(drop): _condition(drop, 2.0),
                instruction_key(keep): _condition(keep, 1.0),
            },
        },
        source,
    )
    inventory = _inventory(tmp_path / "instructions.json", keep)
    output = tmp_path / "task0.pt"

    report = derive_language_bank(source, inventory, output)

    derived = torch.load(output, map_location="cpu", weights_only=False)
    assert list(derived["conditions"]) == [instruction_key(keep)]
    assert torch.equal(
        derived["conditions"][instruction_key(keep)]["tokens"],
        _condition(keep, 1.0)["tokens"],
    )
    assert derived["derived_from"]["sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert report["conditions"] == 1


def test_derive_language_bank_fails_closed_on_missing_instruction(tmp_path: Path) -> None:
    source_instruction = "put the mug on the plate"
    requested_instruction = "pick up the black bowl"
    source = tmp_path / "full.pt"
    torch.save(
        {
            "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
            "encoder_model": T5_ENCODER_ID,
            "conditions": {
                instruction_key(source_instruction): _condition(source_instruction, 1.0)
            },
        },
        source,
    )
    inventory = _inventory(tmp_path / "instructions.json", requested_instruction)
    output = tmp_path / "task0.pt"

    with pytest.raises(KeyError, match="does not cover instruction key"):
        derive_language_bank(source, inventory, output)
    assert not output.exists()
