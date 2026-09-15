from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from clearvla.data.hdf5_episode import LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
from clearvla.data.samplers import (
    BoundaryAwareInformationBatchSampler,
    EvenlySpacedPanelBatchSampler,
    InformationBalancedSamplerConfig,
)
from clearvla.data.window_boundaries import (
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


def test_boundary_sampler_can_reserve_terminal_release_first_lane() -> None:
    plan = _plan(CAUSAL_PREFIX_TERMINAL_SUFFIX_V2)
    regions = [region for _center, region in plan.iter_centers()]
    score = np.linspace(0.0, 1.0, len(regions), dtype=np.float64)
    # Mark two tail centers as directional opening transitions.  Every tail
    # quota row should come from that pool when the repair is fully enabled.
    release = np.asarray(
        [region == TAIL_REGION and index in {52, 70} for index, region in enumerate(regions)],
        dtype=bool,
    )
    sampler = BoundaryAwareInformationBatchSampler(
        score,
        np.zeros_like(release),
        regions,
        contract=CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
        config=InformationBalancedSamplerConfig(batch_size=8, seed=19),
        release_first_events=release,
        release_first_fraction=1.0,
    )
    batches = list(iter(sampler))
    assert sampler.summary["release_first_action_windows"] == 2
    assert sampler.summary["release_first_action_lane"] == "terminal_tail_quota"
    for batch in batches:
        tail = [index for index in batch if regions[index] == TAIL_REGION]
        assert len(tail) == 1
        assert release[tail[0]]


def test_boundary_sampler_accepts_explicit_full_window_epoch_budget() -> None:
    plan = _plan(CAUSAL_PREFIX_TERMINAL_SUFFIX_V2)
    regions = [region for _center, region in plan.iter_centers()]
    score = np.linspace(0.0, 1.0, len(regions), dtype=np.float64)
    config = InformationBalancedSamplerConfig(
        batch_size=8,
        batches_per_epoch=9,
        seed=41,
    )
    sampler = BoundaryAwareInformationBatchSampler(
        score,
        np.zeros(len(regions), dtype=bool),
        regions,
        contract=CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
        config=config,
    )
    assert len(sampler) == 9
    assert sampler.summary["epoch_length_reference"] == (
        "explicit_configured_batches_per_epoch"
    )
    batches = list(iter(sampler))
    assert len(batches) == 9
    assert all(len(batch) == len(set(batch)) == 8 for batch in batches)


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
