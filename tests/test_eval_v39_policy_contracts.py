from types import SimpleNamespace

import pytest

from clearvla.cli.eval_v39_policy import (
    _resolve_model_path_contract,
    _serialized_model_path_capability,
    _versioned_contract_at_least,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    _summarize_current_context_mask_comparison,
)


def _capability_config(*, grounded: int = 0, differential: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        flow_jepa_grounded_intent_effect_mainline=grounded,
        flow_jepa_differential_intent_effect_mainline=differential,
    )


def test_grounded_auto_contract_does_not_replay_v111_ancestry() -> None:
    config = _capability_config(grounded=1)
    assert _serialized_model_path_capability(config) == "grounded_intent_effect_323"
    assert (
        _resolve_model_path_contract(
            "auto",
            policy_config=config,
            newest_versioned_contract="v103",
        )
        == "grounded_intent_effect_323"
    )
    assert (
        _resolve_model_path_contract(
            "grounded_intent_effect_323",
            policy_config=config,
            newest_versioned_contract="v103",
        )
        == "grounded_intent_effect_323"
    )
    for ancestor in ("v111", "v112", "v113"):
        assert not _versioned_contract_at_least(
            "grounded_intent_effect_323",
            ancestor,
        )


def test_differential_auto_contract_is_also_a_sibling_graph() -> None:
    config = _capability_config(differential=1)
    assert (
        _resolve_model_path_contract(
            "auto",
            policy_config=config,
            newest_versioned_contract="v117",
        )
        == "differential_intent_effect_323"
    )
    assert not _versioned_contract_at_least(
        "differential_intent_effect_323",
        "v111",
    )


def test_explicit_contract_and_versioned_order_remain_unchanged() -> None:
    config = _capability_config(grounded=1)
    assert (
        _resolve_model_path_contract(
            "v113",
            policy_config=config,
            newest_versioned_contract="v103",
        )
        == "v113"
    )
    assert _versioned_contract_at_least("v117", "v111")
    assert _versioned_contract_at_least("v113", "v113")
    assert not _versioned_contract_at_least("v110", "v111")


def test_grounded_probe_skips_v113_matched_mask_coverage() -> None:
    assert (
        _summarize_current_context_mask_comparison(
            enabled=False,
            finished_batches=4,
            intervention_samples=32,
            comparison_batches=0,
            comparison_weight=0,
            metric_sums={"unmasked": {}, "masked": {}},
            boundary_sums={},
        )
        is None
    )


def test_enabled_v113_matched_mask_still_requires_full_coverage() -> None:
    with pytest.raises(RuntimeError, match="did not cover every selected"):
        _summarize_current_context_mask_comparison(
            enabled=True,
            finished_batches=4,
            intervention_samples=32,
            comparison_batches=0,
            comparison_weight=0,
            metric_sums={"unmasked": {}, "masked": {}},
            boundary_sums={},
        )


def test_enabled_v113_matched_mask_summary_preserves_paired_delta() -> None:
    summary = _summarize_current_context_mask_comparison(
        enabled=True,
        finished_batches=1,
        intervention_samples=2,
        comparison_batches=1,
        comparison_weight=2,
        metric_sums={
            "unmasked": {"future_loss": 2.0},
            "masked": {"future_loss": 3.0},
        },
        boundary_sums={"mask_fraction": 1.0},
    )
    assert summary is not None
    assert summary["masked_minus_unmasked"]["future_loss"] == pytest.approx(
        0.5
    )
    assert summary["masked_boundary"]["mask_fraction"] == pytest.approx(0.5)
