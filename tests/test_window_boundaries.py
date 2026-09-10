from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from clearvla.data.hdf5_episode import (
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
    LoadedEpisode,
)
from clearvla.data.samplers import (
    BoundaryAwareInformationBatchSampler,
    EvenlySpacedPanelBatchSampler,
    InformationBalancedSamplerConfig,
)
from clearvla.data.window_boundaries import (
    BOUNDARY_REGION_TO_INDEX,
    CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
    CAUSAL_PREFIX_V1,
    PREFIX_REGION,
    STRICT_COMPLETE_V1,
    STRICT_REGION,
    TAIL_REGION,
    WindowBoundaryMetadata,
    policy_action_union_coverage,
    resolve_window_boundary_plan,
)
from clearvla.mainline.data.dataset import (
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.normalizer import ArrayNormalizer


def _plan(contract: str):
    if contract == CAUSAL_PREFIX_TERMINAL_SUFFIX_V2:
        metadata = WindowBoundaryMetadata(
            length=148,
            valid_center_start=0,
            valid_center_end=99,
            strict_valid_center_start=24,
            strict_valid_center_end=51,
            terminal_state_index=100,
            source_action_count=100,
            terminal_padding_mode=LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
        )
        complete_future_end = 99
    else:
        metadata = WindowBoundaryMetadata(
            length=100,
            valid_center_start=24 if contract == STRICT_COMPLETE_V1 else 0,
            valid_center_end=51,
            strict_valid_center_start=(
                None if contract == STRICT_COMPLETE_V1 else 24
            ),
            strict_valid_center_end=(
                None if contract == STRICT_COMPLETE_V1 else 51
            ),
        )
        complete_future_end = 51
    return resolve_window_boundary_plan(
        metadata,
        contract=contract,
        strict_computed_start=24,
        complete_future_end=complete_future_end,
        expected_terminal_padding_mode=(
            LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
            if contract == CAUSAL_PREFIX_TERMINAL_SUFFIX_V2
            else None
        ),
    )


def test_window_boundary_plans_have_exact_disjoint_regions_and_coverage() -> None:
    strict = _plan(STRICT_COMPLETE_V1)
    prefix = _plan(CAUSAL_PREFIX_V1)
    terminal = _plan(CAUSAL_PREFIX_TERMINAL_SUFFIX_V2)

    assert strict.regions == ((STRICT_REGION, 24, 51),)
    assert prefix.regions == (
        (PREFIX_REGION, 0, 23),
        (STRICT_REGION, 24, 51),
    )
    assert terminal.regions == (
        (PREFIX_REGION, 0, 23),
        (STRICT_REGION, 24, 51),
        (TAIL_REGION, 52, 99),
    )
    assert terminal.region_counts == {PREFIX_REGION: 24, STRICT_REGION: 28, TAIL_REGION: 48}

    old_coverage = policy_action_union_coverage(
        {0: range(24, 52)}, source_action_counts={0: 100}, policy_horizon=24
    )
    prefix_coverage = policy_action_union_coverage(
        {0: range(0, 52)}, source_action_counts={0: 100}, policy_horizon=24
    )
    terminal_coverage = policy_action_union_coverage(
        {0: range(0, 100)}, source_action_counts={0: 100}, policy_horizon=24
    )
    assert old_coverage["supervised_real_action_count"] == 51
    assert prefix_coverage["supervised_real_action_count"] == 75
    assert terminal_coverage["supervised_real_action_count"] == 100


@pytest.mark.parametrize(
    ("contract", "expected"),
    (
        (CAUSAL_PREFIX_V1, {PREFIX_REGION: 1, STRICT_REGION: 7, TAIL_REGION: 0}),
        (
            CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
            {PREFIX_REGION: 1, STRICT_REGION: 6, TAIL_REGION: 1},
        ),
    ),
)
def test_boundary_sampler_exact_quota_length_determinism_and_no_duplicates(
    contract: str,
    expected: dict[str, int],
) -> None:
    plan = _plan(contract)
    regions = [region for _center, region in plan.iter_centers()]
    score = np.linspace(0.0, 1.0, len(regions), dtype=np.float64)
    event = np.arange(len(regions)) % 11 == 0
    config = InformationBalancedSamplerConfig(batch_size=8, seed=37)
    sampler = BoundaryAwareInformationBatchSampler(
        score,
        event,
        regions,
        contract=contract,
        config=config,
    )
    assert len(sampler) == 4
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second
    for batch in first:
        assert len(batch) == len(set(batch)) == 8
        assert Counter(regions[index] for index in batch) == Counter(expected)
    sampler.set_epoch(1)
    assert list(iter(sampler)) != first


def test_evenly_spaced_boundary_panel_is_bounded_and_includes_endpoints() -> None:
    sampler = EvenlySpacedPanelBatchSampler(101, batch_size=8, max_batches=4)
    batches = list(iter(sampler))
    flat = [value for batch in batches for value in batch]
    assert len(batches) == 4
    assert len(flat) == len(set(flat)) == 32
    assert flat[0] == 0 and flat[-1] == 100


def test_terminal_contract_rejects_missing_or_wrong_suffix_metadata() -> None:
    metadata = WindowBoundaryMetadata(
        length=148,
        valid_center_start=0,
        valid_center_end=99,
        strict_valid_center_start=24,
        strict_valid_center_end=51,
        terminal_state_index=100,
        source_action_count=100,
        terminal_padding_mode="wrong",
    )
    with pytest.raises(ValueError, match="padding mode"):
        resolve_window_boundary_plan(
            metadata,
            contract=CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
            strict_computed_start=24,
            complete_future_end=99,
            expected_terminal_padding_mode=LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
        )


@pytest.mark.parametrize(
    "metadata",
    (
        WindowBoundaryMetadata(
            length=100,
            valid_center_start=24,
            valid_center_end=51,
        ),
        WindowBoundaryMetadata(
            length=100,
            valid_center_start=0,
            valid_center_end=51,
        ),
    ),
)
def test_causal_prefix_rejects_legacy_or_incomplete_converter_metadata(
    metadata: WindowBoundaryMetadata,
) -> None:
    with pytest.raises(ValueError, match="converter-declared"):
        resolve_window_boundary_plan(
            metadata,
            contract=CAUSAL_PREFIX_V1,
            strict_computed_start=24,
            complete_future_end=51,
        )


class _BoundaryImageStore:
    def validate_episode(self, _episode: LoadedEpisode) -> None:
        pass

    def load_window(
        self, _episode: LoadedEpisode, indices: np.ndarray
    ) -> dict[str, torch.Tensor]:
        rows = torch.as_tensor(indices, dtype=torch.uint8).reshape(-1, 1, 1, 1)
        frames = rows.expand(-1, 3, 1, 1).clone()
        return {"top": frames, "wrist": frames}


def _boundary_dataset(
    *,
    terminal_suffix: bool,
    contract: str,
) -> tuple[ObservedStateWindowDataset, np.ndarray, np.ndarray]:
    source_count = 73
    length = source_count + (48 if terminal_suffix else 0)
    row = np.arange(length, dtype=np.float32)[:, None]
    actions = row + np.arange(7, dtype=np.float32)[None] / 10.0
    states = row + np.arange(7, dtype=np.float32)[None] / 20.0
    action_states = np.concatenate(
        (np.zeros((1, 7), dtype=np.float32), actions[:-1]), axis=0
    )
    if terminal_suffix:
        actions[source_count:, :6] = 0.0
        actions[source_count:, 6] = actions[source_count - 1, 6]
        states[source_count:] = states[source_count - 1]
        action_states[1:] = actions[:-1]
    episode = LoadedEpisode(
        path=Path("boundary.hdf5"),
        episode_id="boundary",
        source_partition="libero_spatial",
        task_id="task",
        action_key="action",
        camera_keys={"top": "top", "wrist": "wrist"},
        actions_raw=actions,
        states_raw=states,
        action_states_raw=action_states,
        valid_center_start=0,
        valid_center_end=source_count - 1 if terminal_suffix else 24,
        strict_valid_center_start=24,
        strict_valid_center_end=24,
        terminal_state_index=source_count if terminal_suffix else None,
        source_action_count=source_count if terminal_suffix else None,
        terminal_padding_mode=(
            LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
            if terminal_suffix
            else ""
        ),
    )
    dataset = ObservedStateWindowDataset(
        [episode],
        [0],
        image_store=_BoundaryImageStore(),
        camera_names=("top", "wrist"),
        state_normalizer=ArrayNormalizer.fit_zscore([states[:source_count]]),
        action_normalizer=ArrayNormalizer.fit_zscore([actions[:source_count]]),
        config=ObservedStateDatasetConfig(
            causal_reset_padding=contract != STRICT_COMPLETE_V1,
            window_boundary_contract=contract,
        ),
        gripper_transition_boundary="previous_command",
    )
    return dataset, actions, action_states


def test_causal_dataset_pads_only_pre_reset_history_and_stops_at_real_actions() -> None:
    prefix, prefix_actions, prefix_action_states = _boundary_dataset(
        terminal_suffix=False,
        contract=CAUSAL_PREFIX_V1,
    )
    assert [(ref.center, ref.boundary_region) for ref in prefix.refs] == [
        *((center, PREFIX_REGION) for center in range(24)),
        (24, STRICT_REGION),
    ]
    reset = prefix[0]
    expected_reset_state = prefix.state_normalizer.encode(
        np.repeat(prefix.episodes[0].states_raw[0:1], 3, axis=0)
    )
    expected_reset_action = prefix.action_normalizer.encode(
        np.repeat(prefix_action_states[0:1], 8, axis=0)
    )
    np.testing.assert_allclose(reset["history_state"], expected_reset_state)
    np.testing.assert_allclose(
        reset["executed_action_history"], expected_reset_action
    )
    np.testing.assert_array_equal(
        reset["gripper_transition_boundary_raw"], prefix_action_states[0]
    )
    assert int(reset["center_index"]) == 0
    assert int(reset["boundary_region"]) == BOUNDARY_REGION_TO_INDEX[PREFIX_REGION]

    after_first_action = prefix[1]
    executed_raw = prefix.action_normalizer.decode(
        np.asarray(after_first_action["executed_action_history"])
    )
    np.testing.assert_allclose(
        executed_raw[:-1], np.repeat(prefix_action_states[0:1], 7, axis=0)
    )
    np.testing.assert_allclose(executed_raw[-1], prefix_actions[0], atol=2.0e-6)

    terminal, terminal_actions, _ = _boundary_dataset(
        terminal_suffix=True,
        contract=CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
    )
    assert len(terminal) == 73
    assert terminal.refs[-1].center == 72
    assert sum(ref.boundary_region == TAIL_REGION for ref in terminal.refs) == 48
    assert terminal.boundary_summary()["supervised_real_action_count"] == 73
    assert terminal.boundary_summary()["uncovered_real_action_count"] == 0
    last_real_center = terminal[len(terminal) - 1]
    np.testing.assert_array_equal(
        last_real_center["policy_action_raw"][0], terminal_actions[72]
    )
    np.testing.assert_array_equal(
        last_real_center["policy_action_raw"][1:, :6],
        np.zeros((23, 6), dtype=np.float32),
    )

    strict, _, _ = _boundary_dataset(
        terminal_suffix=True,
        contract=STRICT_COMPLETE_V1,
    )
    assert [(ref.center, ref.boundary_region) for ref in strict.refs] == [
        (24, STRICT_REGION)
    ]
