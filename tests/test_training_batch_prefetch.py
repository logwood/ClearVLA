from __future__ import annotations

import dataclasses

import pytest
import torch
from test_mainline_data import _config

from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.training.prefetch import (
    CudaTrainingBatchPrefetcher,
    _unpinned_cpu_tensor_paths,
)
from clearvla.tools.profile_mainline_training import _parser as profile_parser


def _raw_batch(*, marker: int) -> dict[str, torch.Tensor]:
    config = _config()
    batch = 2
    generator = torch.Generator().manual_seed(8_100 + marker)
    result = {
        "history_dinov2_tokens": torch.randn(
            batch, 3, 2, 64, 16, generator=generator
        ).half(),
        "history_obs_image": torch.rand(
            batch, 3, 2, 3, 32, 32, generator=generator
        ),
        "state": torch.randn(batch, 7, generator=generator),
        "state_raw": torch.randn(batch, 7, generator=generator),
        "action_state_raw": torch.randn(batch, 7, generator=generator),
        "action_state": torch.randn(batch, 7, generator=generator),
        "gripper_transition_boundary": torch.randn(
            batch, 7, generator=generator
        ),
        "gripper_transition_boundary_raw": torch.randn(
            batch, 7, generator=generator
        ),
        "history_state": torch.randn(batch, 3, 7, generator=generator),
        "executed_action_history": torch.randn(
            batch,
            config.dimensions.executed_history_length,
            7,
            generator=generator,
        ),
        "policy_action": torch.randn(batch, 24, 7, generator=generator),
        "policy_action_raw": torch.randn(batch, 24, 7, generator=generator),
        "action": torch.randn(batch, 48, 7, generator=generator),
        "future_state": torch.randn(batch, 48, 7, generator=generator),
        "target_future_dinov2_tokens": torch.randn(
            batch, 12, 2, 64, 16, generator=generator
        ).half(),
        "target_future_offsets": torch.arange(4, 49, 4)[None]
        .expand(batch, -1)
        .clone(),
        "sample_index": torch.tensor([marker * 2, marker * 2 + 1]),
        "episode_idx": torch.tensor([0, 1], dtype=torch.long),
        "frame_progress": torch.tensor(
            [marker / 10.0, (marker + 1) / 10.0], dtype=torch.float32
        ),
    }
    return result


def _goal() -> GoalTemplate:
    return GoalTemplate(
        tokens=torch.arange(2 * 5 * 12, dtype=torch.float32).reshape(2, 5, 12),
        mask=torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
        metadata={"source": "prefetch-test"},
        episode_condition_indices=torch.tensor([1, 0]),
    )


def _tensor_rows(value: object, *, path: str = "batch") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {path: value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(value):
            result.update(
                _tensor_rows(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
            )
        return result
    return {}


def test_prefetch_pin_audit_reports_nested_pageable_tensor_paths() -> None:
    value = {
        "pinned_or_cuda": [],
        "nested": [torch.zeros(2), {"leaf": torch.ones(1)}],
    }
    assert _unpinned_cpu_tensor_paths(value) == [
        "batch.nested[0]",
        "batch.nested[1].leaf",
    ]


def test_profile_parser_exposes_opt_in_cuda_batch_prefetch() -> None:
    args = profile_parser().parse_args(
        [
            "--cuda-batch-prefetch",
            "--cudnn-benchmark",
            "--cudnn-default-prewarm-scope",
            "raw_pyramid",
        ]
    )
    assert args.cuda_batch_prefetch is True
    assert args.cudnn_benchmark is True
    assert args.cudnn_default_prewarm_scope == ["raw_pyramid"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prefetch requires CUDA")
def test_cuda_prefetch_matches_normal_conversion_bitwise_order_and_rng() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config()
    goal = _goal()
    raw_batches = [
        {name: value.pin_memory() for name, value in _raw_batch(marker=index).items()}
        for index in range(3)
    ]

    expected = [
        to_training_batch(raw, goal=goal, config=config, device=device)
        for raw in raw_batches
    ]
    torch.cuda.synchronize(device)
    torch.manual_seed(9_301)
    torch.cuda.manual_seed_all(9_302)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    with CudaTrainingBatchPrefetcher(
        raw_batches,
        converter=lambda raw: to_training_batch(
            raw,
            goal=goal,
            config=config,
            device=device,
        ),
        device=device,
    ) as prefetcher:
        actual = list(prefetcher)
        torch.cuda.current_stream(device).synchronize()

    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert len(actual) == len(expected)
    for index, (actual_batch, expected_batch) in enumerate(
        zip(actual, expected, strict=True)
    ):
        actual_rows = _tensor_rows(actual_batch)
        expected_rows = _tensor_rows(expected_batch)
        assert tuple(actual_rows) == tuple(expected_rows)
        for name in actual_rows:
            assert torch.equal(actual_rows[name], expected_rows[name]), (
                index,
                name,
            )
        assert actual_batch.audit.sample_index is not None
        assert torch.equal(
            actual_batch.audit.sample_index,
            torch.tensor([index * 2, index * 2 + 1]),
        )
        assert actual_batch.audit.sample_index.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prefetch requires CUDA")
def test_cuda_prefetch_rejects_pageable_worker_tensors() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(ValueError, match="requires pinned worker tensors"):
        CudaTrainingBatchPrefetcher(
            [_raw_batch(marker=0)],
            converter=lambda _raw: pytest.fail("converter must not run"),
            device=device,
        )
