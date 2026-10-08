"""Tests of receipt validation, NOT synthetic substitutes for real training."""

import json
from typing import Any

import pytest

from scripts.run_unified_real_data_check import verify_training_receipt


def metadata_fixture(tmp_path) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    admission = dict(source_commit="a" * 40, batch_size=8, planned_epochs=1, planned_updates=2)
    context = dict(
        identity={"git_commit": "a" * 40},
        execution_mode="training_model_initialized",
        config={"optimizer": {"batch_size": 8}},
        initialization_checkpoint={"initial_global_step": 11012},
    )
    rows = [
        dict(
            kind="epoch",
            epoch=1,
            step=11014,
            train={"loss_total": 0.5, "gradient_global_preclip_l2": 1.0},
            validation={"validation_action_rmse_normalized": 0.2},
        )
    ]
    (tmp_path / "checkpoints").mkdir()
    # Only nonempty-file/hash validation is exercised; this is NOT a valid
    # torch checkpoint or evidence that any optimizer update occurred.
    (tmp_path / "checkpoints" / "latest.pt").write_bytes(b"receipt-test-fixture")
    return admission, context, rows


def save_metadata(tmp_path, context, rows):
    (tmp_path / "run_context.json").write_text(json.dumps(context))
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_receipt_requires_all_artifacts_even_after_process_success(tmp_path):
    result = verify_training_receipt(tmp_path, dict())
    assert not result["passed"]
    assert len(result["errors"]) == 3


def test_complete_metadata_receipt_is_separate_from_weight_or_behavior_admission(tmp_path):
    admission, context, rows = metadata_fixture(tmp_path)
    save_metadata(tmp_path, context, rows)
    result = verify_training_receipt(tmp_path, admission)
    assert result["passed"]
    assert "not weight reload or behavior" in result["scope"]
    assert result["artifacts"]["checkpoints/latest.pt"]["size_bytes"] > 0


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_source",
        "wrong_batch",
        "validation_only",
        "missing_epoch",
        "wrong_step",
        "duplicate_epoch",
        "nan_train",
        "infinite_validation",
        "missing_loss",
        "missing_validation",
        "gradient_failure",
        "boolean_initial",
        "nonobject_row",
        "empty_checkpoint",
    ],
)
def test_incomplete_or_inconsistent_receipts_are_not_training_success(tmp_path, failure):
    admission, context, rows = metadata_fixture(tmp_path)
    if failure == "wrong_source":
        context["identity"]["git_commit"] = "b" * 40
    elif failure == "wrong_batch":
        context["config"]["optimizer"]["batch_size"] = 1
    elif failure == "validation_only":
        context["execution_mode"] = "validation_only"
    elif failure == "missing_epoch":
        rows.clear()
    elif failure == "wrong_step":
        rows[0]["step"] = 11013
    elif failure == "duplicate_epoch":
        rows.append(rows[0].copy())
    elif failure == "nan_train":
        rows[0]["train"]["loss_total"] = float("nan")
    elif failure == "infinite_validation":
        rows[0]["validation"]["validation_action_rmse_normalized"] = float("inf")
    elif failure == "missing_loss":
        del rows[0]["train"]["loss_total"]
    elif failure == "missing_validation":
        rows[0]["validation"] = {}
    elif failure == "gradient_failure":
        rows.append({"kind": "gradient_failure"})
    elif failure == "boolean_initial":
        context["initialization_checkpoint"]["initial_global_step"] = True
    elif failure == "nonobject_row":
        rows.append([])
    elif failure == "empty_checkpoint":
        (tmp_path / "checkpoints" / "latest.pt").write_bytes(b"")
    save_metadata(tmp_path, context, rows)
    result = verify_training_receipt(tmp_path, admission)
    assert not result["passed"] and result["errors"]
