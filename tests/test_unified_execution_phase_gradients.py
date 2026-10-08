"""Execution-phase reachability is not inferred from an aggregate gradient.

Inputs/weights are synthetic. The mature clock only selects execution gates;
it does not pretend that these parameters completed 1,200 training updates.
"""

from importlib import import_module

import pytest

GATED = {
    "execution_bottom.decoder.execution_controller.block_queries",
    "execution_bottom.decoder.execution_controller.operation_query.weight",
    "execution_bottom.decoder.execution_controller.operation_key.weight",
    "execution_bottom.decoder.execution_controller.operation_value.weight",
    "execution_bottom.decoder.execution_controller.capacity_head.weight",
    "execution_bottom.decoder.execution_controller.capacity_head.bias",
    *{f"execution_bottom.decoder.operator_contractions.{i}.basis_raw" for i in range(3)},
}


@pytest.mark.parametrize("variant", ["A", "B"])
def test_ordinary_backward_reaches_execution_paths_after_warmup(variant):
    run = import_module("scripts.check_unified_model_flow").run
    warm = run("small", variant, updates=2, raw_side=32, completed_step=0)
    mature = run("small", variant, updates=2, raw_side=32, completed_step=1200)
    before = {row["name"]: row for row in warm["parameter_ledger"][-1]["entries"]}
    after = {row["name"]: row for row in mature["parameter_ledger"][-1]["entries"]}
    assert before.keys() == after.keys()
    assert {name for name, row in before.items() if row["gradient_status"] == "missing"} == GATED
    assert not [name for name, row in after.items() if row["gradient_status"] == "missing"]
    for name in GATED:
        assert after[name]["gradient_status"] == "nonzero", name
        assert after[name]["gradient_l2"] > 0, name
        assert after[name]["update_l2"] > 0, name
    # Branch-specific exact zeros are accounted for instead of being hidden.
    assert any(row["gradient_status"] == "zero" for row in after.values())
    assert warm["synthetic_completed_step"] == 0
    assert mature["synthetic_completed_step"] == 1200
    assert not mature["real_data_passed"] and not mature["behavior_passed"]
