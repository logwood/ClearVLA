from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from clearvla.mainline.data.language import load_t5_condition_bank
from clearvla.tools.build_rdt_t5_instruction_cache import (
    RDT_PRECOMPUTED_SOURCE,
    build_rdt_precomputed_instruction_cache,
    collect_rdt_embedding_candidates,
)
from clearvla.tools.build_t5_instruction_cache import (
    build_t5_instruction_cache_payload,
    collect_hdf5_instructions,
)


def _instruction_episode(path: Path, instruction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((1, 14), dtype=np.float32))
        handle.create_dataset("instruction", data=np.bytes_(instruction))


def _rdt_task(
    root: Path,
    task: str,
    instruction: str,
    embedding: torch.Tensor,
    *,
    episodes: int = 1,
) -> None:
    task_root = root / "rdt_data" / task
    for index in range(episodes):
        _instruction_episode(task_root / f"episode_{index}.hdf5", instruction)
    torch.save(embedding, task_root / "lang_embed_0.pt")


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


def test_rdt_precomputed_rows_build_a_typed_bank_without_encoder_weights(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    move_a = torch.arange(3 * 4096, dtype=torch.float32).reshape(3, 4096).to(
        torch.bfloat16
    )
    move_b = torch.full((2, 4096), 3.0, dtype=torch.bfloat16)
    _rdt_task(root, "a", "move a", move_a, episodes=2)
    _rdt_task(root, "a_copy", "move a", move_a)
    _rdt_task(root, "b", "move b", move_b)

    instructions, counts, candidates = collect_rdt_embedding_candidates(root)
    assert instructions == ("move a", "move b")
    assert counts == {"move a": 3, "move b": 1}
    assert len(candidates["move a"]) == 2

    output = tmp_path / "typed.pt"
    summary = build_rdt_precomputed_instruction_cache(
        data_root=root,
        output=output,
        policy_max_tokens=4,
        duplicate_policy="error",
        verify_tokenizer=False,
    )
    assert summary["episodes"] == 4
    assert summary["instructions"] == 2
    assert summary["embedding_source"] == RDT_PRECOMPUTED_SOURCE
    assert summary["embedding_variant_group_count"] == 0

    bank = load_t5_condition_bank(output, max_tokens=4, expected_width=4096)
    assert bank.instructions == ("move a", "move b")
    assert bank.condition_indices(["move b", "move a"]).tolist() == [1, 0]
    assert bank.metadata["embedding_candidate_file_count"] == 3
    assert bank.metadata["embedding_variant_group_count"] == 0
    assert bank.metadata["embedding_tokenizer_verification"] == "skipped_explicitly"
    assert torch.equal(bank.tokens[0, :3].to(torch.bfloat16), move_a)
    assert torch.count_nonzero(bank.tokens[0, 3]) == 0
    assert bank.mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]


def test_rdt_precomputed_duplicate_conflict_is_explicit_and_audited(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    first = torch.zeros(2, 4096, dtype=torch.bfloat16)
    second = torch.ones(2, 4096, dtype=torch.bfloat16)
    _rdt_task(root, "a", "same instruction", first)
    _rdt_task(root, "z", "same instruction", second)

    rejected = tmp_path / "rejected.pt"
    with pytest.raises(ValueError, match="non-identical lang_embed_0"):
        build_rdt_precomputed_instruction_cache(
            data_root=root,
            output=rejected,
            duplicate_policy="error",
            verify_tokenizer=False,
        )
    assert not rejected.exists()

    output = tmp_path / "selected.pt"
    summary = build_rdt_precomputed_instruction_cache(
        data_root=root,
        output=output,
        policy_max_tokens=3,
        duplicate_policy="lexicographic",
        verify_tokenizer=False,
    )
    assert summary["embedding_variant_group_count"] == 1
    payload = torch.load(output, map_location="cpu", weights_only=False)
    record = payload["embedding_records"][0]
    assert record["selected_relative_path"] == "rdt_data/a/lang_embed_0.pt"
    assert record["candidate_count"] == 2
    assert record["distinct_tensor_count"] == 2
    assert record["selected_policy_tokens"] == 2
    assert len(record["selected_policy_tensor_sha256"]) == 64
    assert torch.count_nonzero(payload["tokens"]) == 0

    bank = load_t5_condition_bank(output, max_tokens=3, expected_width=4096)
    assert bank.metadata["embedding_duplicate_policy"] == "lexicographic"
    assert bank.metadata["embedding_variant_group_count"] == 1

    payload["embedding_inventory_sha256"] = "0" * 64
    invalid = tmp_path / "invalid-provenance.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="provenance inventory digest"):
        load_t5_condition_bank(invalid, max_tokens=3, expected_width=4096)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload["tokens"][0, 0, 0] = 2
    mutated = tmp_path / "mutated-row.pt"
    torch.save(payload, mutated)
    with pytest.raises(ValueError, match="policy row does not match"):
        load_t5_condition_bank(mutated, max_tokens=3, expected_width=4096)


def test_rdt_precomputed_bank_requires_one_task_local_original_row(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    _instruction_episode(
        root / "rdt_data" / "missing" / "episode_0.hdf5",
        "missing embedding",
    )
    with pytest.raises(FileNotFoundError, match="lang_embed_0.pt"):
        collect_rdt_embedding_candidates(root)

    task = root / "rdt_data" / "ambiguous"
    _instruction_episode(task / "episode_0.hdf5", "first")
    _instruction_episode(task / "episode_1.hdf5", "second")
    torch.save(torch.ones(2, 4096), task / "lang_embed_0.pt")
    with pytest.raises(ValueError, match="exactly one instruction"):
        collect_rdt_embedding_candidates(root)
