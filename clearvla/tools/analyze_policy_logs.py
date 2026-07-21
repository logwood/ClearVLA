from __future__ import annotations

"""Compare dense ClearVLA policy logs without importing the training stack."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


LAYER_PREFIX = "[v39-layer]"
TOKEN_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_]*)=(?P<value>[^\s]+)")
HEADER_RE = re.compile(r"^\[(?P<name>v\d+(?:-[^\]]+)?)\]\s+(?P<body>.*)$")

# Component order is the order used by _owned_serial_log_line. Older logs may
# contain a shorter prefix of a group; those components remain valid.
GROUP_COMPONENTS: dict[str, tuple[str, ...]] = {
    "asrc": ("residual_rms", "delta_rms", "acceleration_rms"),
    "asexp": ("residual_rms", "delta_rms", "acceleration_rms"),
    "asgeo": ("effective_dimension", "condition"),
    "asfirst": ("observed_rms", "expected_std"),
    "hmchart": ("active", "complete", "position_alignment"),
    "hmspec": (
        "state",
        "final_progress",
        "final_arm_mask",
        "final_gripper_mask",
        "competition_loss",
        "coefficient_flow_mse",
    ),
    "hmswarp": ("warp_rms", "spacing_min", "spacing_max", "global_shift_rms"),
    "hmsgeo": ("roundtrip_mse", "bridge_null", "target_null", "prediction_null"),
    "hmtan": ("arm_null", "gripper_null", "noisy_gripper_null"),
    "icdiv": ("global", "stage", "read"),
    "hmbcos": ("noisy_stage", "stage_low", "noisy_low"),
    "hmdepth": ("noisy", "stage", "low"),
    "hmraw": ("noisy", "stage", "low"),
    "hmkeep": ("self", "noisy", "stage", "low", "ffn"),
    "hmedepth": ("noisy", "stage", "low"),
    "hmcontract": ("noisy", "stage", "low"),
    "hmhost": ("noisy", "stage", "low"),
    "hmcover": ("noisy", "stage", "low"),
    "hmremove": ("noisy", "stage", "low"),
    "hmdepthreg": ("progress", "regularizer"),
    "hmdwell": ("learned", "warmup", "value_shadow", "decision_shadow"),
    "hmpfx": ("initial", "final", "gain", "positive_fraction"),
    "hmval": (
        "loss",
        "weight",
        "target_spread",
        "predicted_spread",
        "correlation",
        "accuracy",
        "coverage",
    ),
    "hmvalq": ("target_p25", "target_p50", "target_p75", "reliability", "common_mode"),
    "hmctrl": (
        "direction_participation",
        "pair_cosine",
        "recurrent_change",
        "operation_value_rms",
        "operation_block_spread",
    ),
    "hmvctx": ("memory", "action"),
    "hmpriv": ("pair_cosine", "centered_energy", "global_energy", "value_rms"),
    "hmread": ("operator_memory", "spectral_memory"),
    "hmmem": ("operator_global", "operator_private"),
    "hmrdiv": ("operator_attention", "spectral_local_change", "family_attention"),
    "hmfunc": ("operator_diversity", "spectral_local_change", "state_centered_energy"),
    "hmfcand": ("cosine", "diversity", "update_rms", "update_spread", "valid_count"),
    "hmcomp": ("source_effective", "owner_max", "load_effective", "load_max"),
    "hmctl": ("raw_update", "raw_depth", "continue_keep", "update_depth_corr", "joint_suppression"),
    "hmwi": (
        "state_norm",
        "slot_diversity",
        "low_query",
        "stage_query",
        "promote_mean",
        "promote_std",
    ),
    "hmwic": (
        "low_effective",
        "low_load_effective",
        "low_diversity",
        "stage_effective",
        "stage_load_effective",
        "stage_diversity",
    ),
    "hmca": (
        "intent",
        "flow_time",
        "refine_time",
        "action",
        "evidence",
        "stage_role",
        "stage_content",
        "feedback",
    ),
    "hmce": ("geom", "transition", "event", "state", "layer"),
    "hmop": (
        "loss",
        "weight",
        "target_entropy",
        "predicted_entropy",
        "gain_spread",
        "accuracy",
        "coverage",
    ),
    "hmsel": ("entropy", "max", "exploration"),
    "hmselq": ("query_change", "same_block_query_change"),
    "hmexit": ("probability", "candidate_rate"),
    "hmopgain": ("noisy", "stage", "low"),
    "hmdir": ("noisy", "stage", "low"),
    "hmbcos2": ("noisy", "stage", "low"),
    "hmbgain": ("noisy", "stage", "low"),
    "hmgate": ("self", "noisy", "stage", "low", "ffn"),
    "hmegate": ("self", "noisy", "stage", "low", "ffn"),
    "hmkerr": ("noisy", "stage", "low"),
    "hmnrms": ("pre", "post"),
    "hexh": (
        "executed_steps",
        "action_response_rel",
        "stage_pressure_rel",
        "refine_gain",
        "response_gain_corr",
        "unresolved_rate",
        "budget_exhausted_rate",
        "final_block",
        "final_stage",
        "block_advance_rate",
        "stage_advance_rate",
        "fixed_path_agreement",
        "monotonic_violation",
    ),
    "hmshadow": ("executed_steps", "step_saving", "error_ratio", "stay", "advance"),
    "hmuresp": ("arm", "gripper", "arm_null", "gripper_null"),
    "hmuq": ("p25", "p50", "p75"),
    "hmpq": ("p25", "p50", "p75"),
    "hmuT50": ("t0", "t1", "t2"),
    "hmpT50": ("t0", "t1", "t2"),
    "hmucT": ("t0", "t1", "t2"),
    "hmpcT": ("t0", "t1", "t2"),
}

CORE_KEYS = (
    "loss",
    "pflow",
    "pfn",
    "afmd",
    "gfmf",
    "gfar",
    "decode",
    "rollout",
    "first8",
    "tail",
    "event",
    "d_shuffle",
    "stdr",
    "dnratio",
    "hmdu",
    "hmur",
    "hmcan",
    "hmxcan",
    "hmnf",
    "hmsf",
    "hmlf",
    "hmnw",
    "spb",
)

EPOCH_KEYS = (
    "full_mse",
    "full_rmse",
    "first_rmse",
    "first4_rmse",
    "first8_rmse",
    "tail_rmse",
    "arm_full_rmse",
    "arm_first8_rmse",
    "arm_tail_rmse",
    "gripper_full_rmse",
    "gripper_first8_rmse",
    "gripper_tail_rmse",
    "gripper_precision",
    "gripper_recall",
    "physical_flow",
    "physical_flow_native",
    "arm_fm_per_dim",
    "gripper_fm_field",
    "gripper_arm_fm_ratio",
    "decoded_action",
    "rollout_dynamics",
    "first8_physical_flow",
    "tail_physical_flow",
    "event",
    "event_head_precision",
    "event_head_recall",
    "motion_head_precision",
    "motion_head_recall",
    "rollout_pred_std_ratio",
    "rollout_milestone_delta_norm_ratio",
    "arm_noise_abs_std",
    "arm_noise_delta_std",
    "arm_target_abs_std",
    "arm_target_delta_std",
    "gripper_fm_event",
    "gripper_fm_hold",
    "gripper_fm_event_rate",
    "rollout_base_norm",
    "rollout_delta_norm",
    "rollout_base_is_fixed_zero",
    "sample_arm_null_preproject_fraction",
    "sample_grip_null_preproject_fraction",
    "balanced_score",
    "deploy_eligible",
)


@dataclass
class LayerRow:
    epoch: int
    batch: int
    values: dict[str, float]


@dataclass
class ParsedLog:
    path: Path
    label: str
    headers: list[str] = field(default_factory=list)
    rows: list[LayerRow] = field(default_factory=list)
    epoch_records: list[dict[str, Any]] = field(default_factory=list)
    malformed_json: int = 0


def _number(value: str) -> float | None:
    try:
        result = float(value.rstrip(",;"))
    except ValueError:
        return None
    return result if math.isfinite(result) else result


def _parse_tokens(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in TOKEN_RE.finditer(line):
        key = match.group("key")
        raw = match.group("value").rstrip(",;")
        if "/" not in raw:
            value = _number(raw)
            if value is not None:
                values[key] = value
            continue
        parts = raw.split("/")
        parsed = [_number(part) for part in parts]
        if any(value is None for value in parsed):
            continue
        names = GROUP_COMPONENTS.get(key, ())
        for index, value in enumerate(parsed):
            component = names[index] if index < len(names) else str(index)
            values[f"{key}.{component}"] = float(value)  # type: ignore[arg-type]
    return values


def parse_log(path: Path, label: str | None = None) -> ParsedLog:
    parsed = ParsedLog(path=path, label=label or path.parent.name)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(LAYER_PREFIX):
                values = _parse_tokens(line)
                if "epoch" in values and "batch" in values:
                    parsed.rows.append(
                        LayerRow(int(values.pop("epoch")), int(values.pop("batch")), values)
                    )
                continue
            if line.startswith("{") and '"epoch"' in line:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    parsed.malformed_json += 1
                else:
                    if isinstance(record, dict):
                        parsed.epoch_records.append(record)
                continue
            header_match = HEADER_RE.match(line)
            if header_match and not line.startswith("[v39-init]"):
                parsed.headers.append(line)
    return parsed


def _median(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = fraction * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _tail_stats(rows: Sequence[LayerRow], key: str, tail: int) -> dict[str, float | None]:
    available = [row.values[key] for row in rows if key in row.values]
    selected = available[-tail:]
    if not selected:
        return {"first": None, "last": None, "median": None, "p25": None, "p75": None}
    return {
        "first": available[0],
        "last": available[-1],
        "median": _median(selected),
        "p25": _quantile(selected, 0.25),
        "p75": _quantile(selected, 0.75),
    }


def _epoch_tail_rows(log: ParsedLog, epoch: int, tail: int) -> list[LayerRow]:
    return [row for row in log.rows if row.epoch == epoch][-tail:]


def _milestone(rows: Sequence[LayerRow], key: str, threshold: float) -> str:
    for row in rows:
        value = row.values.get(key)
        if value is not None and value <= threshold:
            return f"e{row.epoch}:b{row.batch}"
    return "-"


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return str(value)
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e4):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _record_sections(record: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for name in ("train", "val", "validation", "eval", "evaluation", "sample"):
        section = record.get(name)
        if isinstance(section, Mapping):
            yield name, section


def _epoch_record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "epoch": record.get("epoch"),
        "global_step": record.get("global_step"),
        "sections": {},
    }
    sections: dict[str, dict[str, float]] = {}
    for name, section in _record_sections(record):
        values: dict[str, float] = {}
        for key in EPOCH_KEYS:
            value = section.get(key)
            if isinstance(value, (int, float)):
                values[key] = float(value)
        for key in (
            "sample_arm_null_preproject_fraction",
            "sample_grip_null_preproject_fraction",
            "hierarchical_mmdit_controller_state_pair_cosine",
            "hierarchical_mmdit_controller_state_direction_participation",
            "hierarchical_mmdit_controller_private_pair_cosine",
            "hierarchical_mmdit_controller_state_centered_energy_ratio",
            "hierarchical_mmdit_action_noisy_update_fraction",
            "hierarchical_mmdit_action_stage_update_fraction",
            "hierarchical_mmdit_action_low_update_fraction",
            "hierarchical_mmdit_executed_steps",
        ):
            value = section.get(key)
            if isinstance(value, (int, float)):
                values[key] = float(value)
        sections[name] = values
    result["sections"] = sections
    return result


def _health_findings(log: ParsedLog, tail: int) -> list[str]:
    findings: list[str] = []
    rows = log.rows[-tail:]

    def med(key: str) -> float | None:
        return _median(row.values[key] for row in rows if key in row.values)

    pair_cos = med("hmctrl.pair_cosine")
    participation = med("hmctrl.direction_participation")
    if pair_cos is not None and pair_cos > 0.90:
        findings.append(
            f"controller common-mode severe: pair_cos={pair_cos:.3f}, "
            f"direction_participation={_fmt(participation, 2)}"
        )
    elif pair_cos is not None and pair_cos > 0.70:
        findings.append(f"controller common-mode elevated: pair_cos={pair_cos:.3f}")
    if participation is not None:
        if participation < 1.5:
            findings.append(
                f"controller directions nearly one-dimensional: participation={participation:.2f}"
            )
        elif participation < 2.5:
            findings.append(
                "controller directional participation is low: "
                f"participation={participation:.2f}, pair_cos={_fmt(pair_cos, 3)}"
            )
        elif participation < 4.0:
            findings.append(
                "controller uses only a limited directional subspace: "
                f"participation={participation:.2f}, pair_cos={_fmt(pair_cos, 3)}"
            )

    latest_train: Mapping[str, Any] = {}
    if log.epoch_records:
        candidate = log.epoch_records[-1].get("train", {})
        if isinstance(candidate, Mapping):
            latest_train = candidate
    private_participation = latest_train.get(
        "hierarchical_mmdit_controller_private_direction_participation"
    )
    if isinstance(private_participation, (int, float)) and private_participation < 1.5:
        findings.append(
            "private controller slots collapsed to an almost one-direction state: "
            f"participation={private_participation:.2f}"
        )
    recurrent_change = latest_train.get("hierarchical_mmdit_controller_recurrent_change")
    if isinstance(recurrent_change, (int, float)) and recurrent_change < 0.05:
        findings.append(
            f"controller recurrence is close to a fixed point: change={recurrent_change:.3f}"
        )

    initial_error = latest_train.get("hierarchical_mmdit_refine_error_initial")
    final_error = latest_train.get("hierarchical_mmdit_refine_error_final")
    if (
        isinstance(initial_error, (int, float))
        and isinstance(final_error, (int, float))
        and initial_error > 0.0
    ):
        findings.append(
            "normalized refine residual="
            f"{final_error / initial_error:.3f} "
            f"({final_error:.4f}/{initial_error:.4f})"
        )

    expected_first = med("asfirst.expected_std")
    expected_terminal = med("astail")
    if (
        expected_first is not None
        and expected_terminal is not None
        and expected_first > 0.0
        and expected_terminal / expected_first > 1.5
    ):
        findings.append(
            "arm source variance is horizon-imbalanced: terminal/first="
            f"{expected_terminal / expected_first:.2f}"
        )

    noisy, stage, low = med("hmnf"), med("hmsf"), med("hmlf")
    if noisy is not None and stage is not None and low is not None:
        findings.append(f"branch fractions noisy/stage/low={noisy:.3f}/{stage:.3f}/{low:.3f}")
        if noisy > max(stage, low) * 1.8:
            findings.append("noisy branch dominates the refinement write budget")

    arm_null, grip_null = med("anull"), med("gnull")
    if arm_null is not None or grip_null is not None:
        findings.append(f"training output null arm/gripper={_fmt(arm_null)}/{_fmt(grip_null)}")

    current_epoch = log.rows[-1].epoch if log.rows else None
    pflow = [
        row.values["pflow"]
        for row in log.rows
        if row.epoch == current_epoch and "pflow" in row.values
    ]
    if len(pflow) >= 12:
        window = max(5, min(tail, len(pflow) // 4))
        before = _median(pflow[-2 * window : -window])
        after = _median(pflow[-window:])
        if before and after:
            relative = (after - before) / before
            if relative > 0.10:
                findings.append(f"pflow tail regressed by {relative:+.1%}")
            elif relative > -0.05:
                findings.append(f"pflow tail is near a plateau ({relative:+.1%})")
            else:
                findings.append(f"pflow tail still descends ({relative:+.1%})")

    rollout = med("rollout")
    decode = med("decode")
    if rollout is not None and decode is not None and rollout > max(0.12, decode * 5.0):
        findings.append(
            f"rollout remains the slower auxiliary objective: {_fmt(rollout)} vs decode {_fmt(decode)}"
        )
    return findings


def build_summary(log: ParsedLog, tail: int) -> dict[str, Any]:
    epochs = sorted({row.epoch for row in log.rows})
    tail_stats: dict[str, Any] = {}
    available_keys = sorted({key for row in log.rows for key in row.values})
    for key in available_keys:
        tail_stats[key] = _tail_stats(log.rows, key, tail)
    epoch_stats: dict[str, Any] = {}
    epoch_snapshots: dict[str, Any] = {}
    for epoch in epochs:
        all_epoch_rows = [row for row in log.rows if row.epoch == epoch]
        rows = all_epoch_rows[-tail:]
        epoch_stats[str(epoch)] = {
            key: _tail_stats(rows, key, len(rows))["median"]
            for key in CORE_KEYS
            if any(key in row.values for row in rows)
        }
        snapshots = []
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            index = round(fraction * (len(all_epoch_rows) - 1))
            row = all_epoch_rows[index]
            snapshots.append(
                {
                    "fraction": fraction,
                    "batch": row.batch,
                    "values": {
                        key: row.values[key]
                        for key in (
                            "pflow",
                            "pfn",
                            "rollout",
                            "event",
                            "hmctrl.direction_participation",
                            "hmctrl.pair_cosine",
                            "hmpriv.pair_cosine",
                            "hmfunc.state_centered_energy",
                            "hmnf",
                            "hmsf",
                            "hmlf",
                            "hmdepth.noisy",
                            "hmdepth.stage",
                            "hmdepth.low",
                        )
                        if key in row.values
                    },
                }
            )
        epoch_snapshots[str(epoch)] = snapshots
    return {
        "label": log.label,
        "path": str(log.path),
        "headers": log.headers,
        "batch_rows": len(log.rows),
        "range": {
            "first": [log.rows[0].epoch, log.rows[0].batch] if log.rows else None,
            "last": [log.rows[-1].epoch, log.rows[-1].batch] if log.rows else None,
        },
        "milestones": {
            str(threshold): _milestone(log.rows, "pflow", threshold)
            for threshold in (1.0, 0.5, 0.2, 0.1, 0.05)
        },
        "tail": tail_stats,
        "epochs": epoch_stats,
        "epoch_snapshots": epoch_snapshots,
        "epoch_records": [_epoch_record_summary(record) for record in log.epoch_records],
        "malformed_json": log.malformed_json,
        "findings": _health_findings(log, tail),
    }


def _print_report(summaries: Sequence[Mapping[str, Any]], tail: int) -> None:
    print(f"Policy log comparison (tail median over {tail} batch rows)")
    print()
    for summary in summaries:
        print(f"== {summary['label']} ==")
        print(summary["path"])
        for header in summary["headers"]:
            print(f"  {header}")
        print(
            f"  rows={summary['batch_rows']} range={summary['range']['first']}..{summary['range']['last']} "
            f"milestones={summary['milestones']}"
        )
        print("  epoch tail medians:")
        print("    epoch   pflow     pfn     decode  rollout first8   tail    event    spb")
        for epoch, values in summary["epochs"].items():
            print(
                f"    {int(epoch):>5} "
                + " ".join(
                    f"{_fmt(values.get(key), 4):>8}"
                    for key in (
                        "pflow",
                        "pfn",
                        "decode",
                        "rollout",
                        "first8",
                        "tail",
                        "event",
                        "spb",
                    )
                )
            )
        print("  progress snapshots:")
        for epoch, snapshots in summary["epoch_snapshots"].items():
            for snapshot in snapshots:
                values = snapshot["values"]
                rendered = " ".join(
                    f"{key}={_fmt(values.get(key), 3)}"
                    for key in (
                        "pflow",
                        "rollout",
                        "event",
                        "hmctrl.direction_participation",
                        "hmctrl.pair_cosine",
                        "hmnf",
                        "hmsf",
                        "hmlf",
                    )
                    if key in values
                )
                print(f"    e{epoch} {snapshot['fraction']:.0%} b{snapshot['batch']}: {rendered}")
        tail_stats = summary["tail"]
        print("  structural tail medians:")
        for keys in (
            (
                "hmctrl.direction_participation",
                "hmctrl.pair_cosine",
                "hmpriv.pair_cosine",
                "hmfunc.state_centered_energy",
            ),
            ("hmnf", "hmsf", "hmlf", "hmnw"),
            (
                "hmdepth.noisy",
                "hmdepth.stage",
                "hmdepth.low",
                "hmkeep.noisy",
                "hmkeep.stage",
                "hmkeep.low",
            ),
            ("anull", "gnull", "hmtan.arm_null", "hmtan.gripper_null"),
            ("hmcan", "hmorth", "hmxcan", "hmbdot"),
        ):
            rendered = []
            for key in keys:
                stat = tail_stats.get(key)
                if stat and stat.get("median") is not None:
                    rendered.append(f"{key}={_fmt(stat['median'])}")
            if rendered:
                print("    " + " ".join(rendered))
        if summary["epoch_records"]:
            print("  epoch JSON summaries:")
            for record in summary["epoch_records"]:
                for section, values in record["sections"].items():
                    selected = " ".join(
                        f"{key}={_fmt(values[key])}" for key in EPOCH_KEYS if key in values
                    )
                    print(
                        f"    e{record['epoch']} step={record['global_step']} {section}: {selected}"
                    )
        print("  findings:")
        for finding in summary["findings"]:
            print(f"    - {finding}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label for the corresponding positional log; may be repeated.",
    )
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tail <= 0:
        raise SystemExit("--tail must be positive")
    if args.label and len(args.label) != len(args.logs):
        raise SystemExit("repeat --label once for every log, or omit all labels")
    logs = [
        parse_log(path, args.label[index] if args.label else f"log{index + 1}")
        for index, path in enumerate(args.logs)
    ]
    summaries = [build_summary(log, args.tail) for log in logs]
    _print_report(summaries, args.tail)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {"schema": "clearvla-policy-log-comparison-v1", "logs": summaries}, indent=2
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
