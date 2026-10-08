"""Separate integration qualifications, not a pretrained/data/GPU acceptance.

Small BS8 uses real CPU BF16 autocast. Full width uses BS1/FP32 and 336px RGB.
Both use artificial values and train from random initialization. Clock 1200
opens execution gates; it is not evidence of 1200 completed learning steps.
"""

import json
from collections.abc import Callable

import pytest


@pytest.mark.parametrize("variant", ["A", "B"])
@pytest.mark.parametrize("shape", ["small", "production"])
def test_measured_robot_results_in_selected_complete_graph(
    variant: str, shape: str, record_testsuite_property: Callable[[str, object], None]
):
    from scripts.check_unified_model_flow import run

    result = run(
        shape,
        variant,
        updates=2,
        raw_side=32 if shape == "small" else 336,
        completed_step=1200,
        typed_object_values="conditional_object_v1",
        compute_dtype="bf16" if shape == "small" else "fp32",
        batch_size=8 if shape == "small" else 1,
        reference_support="alternating",
        outcome_mode="robot_world_before_proposal_v2",
    )
    assert result["passed"]
    assert (
        result["actual_batch_size"]
        == result["optimizer_batch_size"]
        == (8 if shape == "small" else 1)
    )
    assert result["sampled_action_shape"] == [result["actual_batch_size"], 24, 7]
    assert result["observed_outcome_mode"] == "robot_world_before_proposal_v2"
    assert not result["real_data_passed"] and not result["behavior_passed"]
    rows = result["parameter_ledger"][-1]["entries"]
    assert not [r["name"] for r in rows if r["gradient_status"] == "missing"]
    new = [r for r in rows if ".observed_robot_outcome." in r["name"]]
    assert len(new) == 3
    assert all(r["gradient_status"] == "nonzero" and r["update_l2"] > 0 for r in new)
    receipt = {key: value for key, value in result.items() if key != "parameter_ledger"}
    receipt["robot_outcome_parameters"] = new
    record_testsuite_property(shape + "_" + variant, json.dumps(receipt))
