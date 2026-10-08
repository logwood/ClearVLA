"""Actual CPU autocast and BS8 source/gradient paths, with artificial inputs.

These tests do not load DINO/T5 assets or qualify CUDA kernels, learned skills,
or a real training dataset. Actual batch and dtype are recorded explicitly.
"""

import pytest


@pytest.mark.parametrize("variant", ["A", "B"])
def test_conditional_values_in_real_autocast_and_eight_row_batch(variant):
    from scripts.check_unified_model_flow import run

    result = run(
        "small",
        variant,
        updates=2,
        raw_side=32,
        completed_step=1200,
        typed_object_values="conditional_object_v1",
        compute_dtype="bf16",
        batch_size=8,
    )
    assert result["actual_batch_size"] == result["optimizer_batch_size"] == 8
    assert result["compute_dtype"] == "bf16"
    assert result["training_autocast_dtype"] == "torch.bfloat16"
    assert result["sampled_action_shape"] == [8, 24, 7]
    assert not result["real_data_passed"] and not result["behavior_passed"]
    assert not [
        p["name"]
        for p in result["parameter_ledger"][-1]["entries"]
        if p["gradient_status"] == "missing"
    ]


def test_fixture_rejects_unknown_dtype_and_invalid_actual_batch():
    from scripts.check_unified_model_flow import configuration, run

    with pytest.raises(ValueError):
        configuration("small", "A", compute_dtype="guess")
    for batch in (0, -1, True):
        with pytest.raises(ValueError, match="batch size"):
            run("small", "A", updates=2, raw_side=32, batch_size=batch)
