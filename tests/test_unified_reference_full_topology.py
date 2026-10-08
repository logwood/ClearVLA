"""Full-width selected candidate with independently masked reference evidence.

Inputs and weights are artificial; the reference mask only declares availability.
This does not qualify pretrained DINO/T5, CUDA BS8, or any robot skill.
"""

import json
from collections.abc import Callable

import pytest


@pytest.mark.parametrize("variant", ["A", "B"])
def test_selected_conditional_values_and_partial_reference_at_full_width(
    variant: str, record_testsuite_property: Callable[[str, object], None]
):
    from scripts.check_unified_model_flow import run

    result = run(
        "production",
        variant,
        updates=2,
        raw_side=336,
        completed_step=1200,
        typed_object_values="conditional_object_v1",
        reference_support="alternating",
    )
    assert result["passed"]
    assert result["dimensions"]["hidden_size"] == 512
    assert result["raw_rgb_side"] == 336
    assert result["sampled_action_shape"] == [1, 24, 7]
    assert result["typed_object_value_mode"] == "conditional_object_v1"
    assert result["observation_measurement_mode"] == "source_consistent_v1"
    assert result["reference_support_mode"] == "alternating"
    assert 2 * result["reference_observed_cells"] == result["reference_total_cells"]
    assert not result["real_data_passed"] and not result["behavior_passed"]
    assert not [
        row["name"]
        for row in result["parameter_ledger"][-1]["entries"]
        if row["gradient_status"] == "missing"
    ]
    receipt = {key: value for key, value in result.items() if key != "parameter_ledger"}
    record_testsuite_property("full_width_partial_reference_" + variant, json.dumps(receipt))


def test_reference_fixture_does_not_accept_implicit_mask_modes():
    from scripts.check_unified_model_flow import run

    with pytest.raises(ValueError, match="reference support"):
        run("small", "A", updates=2, raw_side=32, reference_support="guess")
