from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from clearvla.mainline.data.language import load_t5_condition_bank
from clearvla.tools.build_t5_instruction_cache import (
    build_t5_instruction_cache_payload,
    collect_hdf5_instructions,
)


def _instruction_episode(path: Path, instruction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((1, 14), dtype=np.float32))
        handle.create_dataset("instruction", data=np.bytes_(instruction))


def test_instruction_inventory_and_condition_bank_are_exact_and_deduplicated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    _instruction_episode(root / "rdt_data" / "a" / "episode_0.hdf5", "move a")
    _instruction_episode(root / "rdt_data" / "a" / "episode_1.hdf5", "move a")
    _instruction_episode(root / "rdt_data" / "b" / "episode_0.hdf5", "move b")

    instructions, counts = collect_hdf5_instructions(root)
    assert instructions == ("move a", "move b")
    assert counts == {"move a": 2, "move b": 1}

    tokens = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12) + 1
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]],
        dtype=torch.bool,
    )
    payload = build_t5_instruction_cache_payload(
        instructions=instructions,
        tokens=tokens,
        attention_mask=mask,
        model_source="local/google/t5-v1_1-xxl",
        source_episode_count=3,
        source_instruction_inventory_sha256="a" * 64,
    )
    path = tmp_path / "instructions.pt"
    torch.save(payload, path)

    bank = load_t5_condition_bank(path, max_tokens=3, expected_width=12)
    assert bank.instructions == instructions
    assert tuple(bank.tokens.shape) == (2, 3, 12)
    assert bank.tokens.dtype == torch.float32
    assert torch.count_nonzero(bank.tokens[0, 2]) == 0
    assert bank.condition_indices(["move b", "move a"]).tolist() == [1, 0]
    with pytest.raises(KeyError, match="does not cover"):
        bank.condition_indices(["move c"])


def test_instruction_cache_rejects_nonzero_masked_values(tmp_path: Path) -> None:
    payload = build_t5_instruction_cache_payload(
        instructions=("move a",),
        tokens=torch.ones(1, 2, 12),
        attention_mask=torch.tensor([[True, False]]),
        model_source="google/t5-v1_1-xxl",
        source_episode_count=1,
        source_instruction_inventory_sha256="b" * 64,
    )
    payload["tokens"][0, 1, 0] = 1
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="exact zero"):
        load_t5_condition_bank(path, max_tokens=2, expected_width=12)
