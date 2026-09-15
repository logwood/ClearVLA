from __future__ import annotations

import json
from pathlib import Path

from clearvla.tools.compare_mainline_early_losses import (
    build_report,
    codec_parity,
    load_legacy_train_rows,
)


def test_codec_boundary_is_exact() -> None:
    result = codec_parity(seed=3)
    assert result["exact"] is True
    assert result["physical_shape"] == [3, 24, 18]


def test_early_loss_report_uses_scale_not_exact_equality(tmp_path: Path) -> None:
    legacy_log = tmp_path / "legacy.log"
    legacy_log.write_text(
        "\n".join(
            (
                "[v122-train] epoch=001 batch=0001 loss_total=1.20 "
                "flow_loss=1.00 native_velocity_mse=0.90 decode_loss=0.20 "
                "event_loss=0.60 motion_loss=0.70 proposal_loss=0.10 "
                "flow_first8=1.10 flow_tail=0.95 "
                "loss_groups=action:1.10/representation:0.10",
                "[v122-train] epoch=001 batch=0002 loss_total=1.00 "
                "flow_loss=0.80 native_velocity_mse=0.72 decode_loss=0.16 "
                "event_loss=0.55 motion_loss=0.62 proposal_loss=0.08 "
                "flow_first8=0.90 flow_tail=0.75 "
                "loss_groups=action:0.91/representation:0.09",
            )
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.jsonl"
    rows = (
        {
            "kind": "train",
            "metrics": {
                "loss_total": 1.26,
                "loss_action_flow": 1.05,
                "loss_action_flow_native": 0.94,
                "loss_decoded_action": 0.21,
                "loss_event": 0.63,
                "loss_motion": 0.72,
                "loss_history_action_proposal": 0.11,
                "loss_action_flow_first8": 1.14,
                "loss_action_flow_tail": 1.00,
                "loss_group_action": 1.15,
                "loss_group_representation": 0.11,
            },
        },
        {
            "kind": "train",
            "metrics": {
                "loss_total": 1.08,
                "loss_action_flow": 0.86,
                "loss_action_flow_native": 0.76,
                "loss_decoded_action": 0.17,
                "loss_event": 0.57,
                "loss_motion": 0.65,
                "loss_history_action_proposal": 0.09,
                "loss_action_flow_first8": 0.96,
                "loss_action_flow_tail": 0.80,
                "loss_group_action": 0.98,
                "loss_group_representation": 0.10,
            },
        },
    )
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = build_report(
        legacy_log=legacy_log,
        mainline_metrics=metrics,
        steps=2,
        legacy_manifest=None,
        mainline_context=None,
    )
    assert report["codec_parity"]["exact"] is True
    assert report["metrics"]["action_flow"]["status"] == "close"
    assert report["conclusion"]["comparable_early_action_scale"] is True
    assert report["conclusion"]["bitwise_equality_expected"] is False
    parsed = load_legacy_train_rows(legacy_log, limit=1)
    assert parsed[0]["loss_group_action"] == 1.10
