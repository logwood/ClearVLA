"""Compare the frozen V122 and independent-mainline early loss scales.

This is deliberately a *statistical* comparison.  The two graphs have
different parameter registration order and different private RNG ownership,
so equality of per-step losses would be a false requirement.  The paired
launcher keeps the dataset, sampler, seed, batch size and public objective
weights fixed; this module checks the exact shared action chart and compares
the first N logged updates by semantic loss name.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

_LEGACY_TRAIN = re.compile(r"^\[v(?:121|122)-train\]\s")

# New-name -> (legacy stdout name, independent-mainline metrics.jsonl name).
# Only rows whose mathematics has the same external meaning are included.
_COMMON_METRICS: dict[str, tuple[str, str]] = {
    "total": ("loss_total", "loss_total"),
    "action_flow": ("flow_loss", "loss_action_flow"),
    "native_action_flow": ("native_velocity_mse", "loss_action_flow_native"),
    "decoded_action": ("decode_loss", "loss_decoded_action"),
    "motion": ("motion_loss", "loss_motion"),
    "history_proposal": ("proposal_loss", "loss_history_action_proposal"),
    "flow_first8": ("flow_first8", "loss_action_flow_first8"),
    "flow_tail": ("flow_tail", "loss_action_flow_tail"),
    "action_group": ("loss_group_action", "loss_group_action"),
    "representation_group": (
        "loss_group_representation",
        "loss_group_representation",
    ),
}


@dataclass(frozen=True)
class SeriesSummary:
    count: int
    first: float
    last: float
    mean: float
    median: float
    standard_deviation: float
    first_half_mean: float
    second_half_mean: float
    half_change_fraction: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "first": self.first,
            "last": self.last,
            "mean": self.mean,
            "median": self.median,
            "standard_deviation": self.standard_deviation,
            "first_half_mean": self.first_half_mean,
            "second_half_mean": self.second_half_mean,
            "half_change_fraction": self.half_change_fraction,
        }


def _finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _parse_legacy_fields(line: str) -> dict[str, float]:
    fields: dict[str, float] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        name, raw = token.split("=", 1)
        if name == "loss_groups":
            for group in raw.split("/"):
                group_name, separator, group_value = group.partition(":")
                value = _finite_float(group_value) if separator else None
                if value is not None:
                    fields[f"loss_group_{group_name}"] = value
            continue
        value = _finite_float(raw.rstrip(","))
        if value is not None:
            fields[name] = value
    # V122 emits the explicit alias.  Accept the ancestral name only for a
    # source snapshot whose compact logger predates that alias.
    if "native_velocity_mse" not in fields and "native_flow" in fields:
        fields["native_velocity_mse"] = fields["native_flow"]
    return fields


def load_legacy_train_rows(path: Path, *, limit: int) -> list[dict[str, float]]:
    rows = [
        _parse_legacy_fields(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if _LEGACY_TRAIN.match(line)
    ]
    rows = [row for row in rows if "loss_total" in row]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no V121/V122 train rows found in {path}")
    return rows


def load_mainline_train_rows(path: Path, *, limit: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("kind") != "train":
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError("mainline train row has no metrics mapping")
            row = {
                str(name): float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no mainline train rows found in {path}")
    return rows


def _series(rows: Iterable[Mapping[str, float]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row]


def summarize(values: Sequence[float]) -> SeriesSummary:
    if not values:
        raise ValueError("cannot summarize an empty loss series")
    middle = max(1, len(values) // 2)
    first_half = values[:middle]
    second_half = values[middle:] or values[-1:]
    first_mean = statistics.fmean(first_half)
    second_mean = statistics.fmean(second_half)
    return SeriesSummary(
        count=len(values),
        first=float(values[0]),
        last=float(values[-1]),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        standard_deviation=(statistics.pstdev(values) if len(values) > 1 else 0.0),
        first_half_mean=first_mean,
        second_half_mean=second_mean,
        half_change_fraction=(second_mean - first_mean) / max(abs(first_mean), 1e-12),
    )


def _scale_label(ratio: float) -> str:
    if 0.75 <= ratio <= 4.0 / 3.0:
        return "close"
    if 0.5 <= ratio <= 2.0:
        return "same_order"
    return "different_scale"


def compare_rows(
    legacy_rows: Sequence[Mapping[str, float]],
    mainline_rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for semantic_name, (legacy_name, mainline_name) in _COMMON_METRICS.items():
        legacy_values = _series(legacy_rows, legacy_name)
        mainline_values = _series(mainline_rows, mainline_name)
        if not legacy_values or not mainline_values:
            comparison[semantic_name] = {
                "status": "not_comparable",
                "legacy_key": legacy_name,
                "mainline_key": mainline_name,
                "legacy_count": len(legacy_values),
                "mainline_count": len(mainline_values),
            }
            continue
        legacy_summary = summarize(legacy_values)
        mainline_summary = summarize(mainline_values)
        ratio = mainline_summary.mean / max(abs(legacy_summary.mean), 1e-12)
        comparison[semantic_name] = {
            "status": _scale_label(ratio),
            "legacy_key": legacy_name,
            "mainline_key": mainline_name,
            "mean_ratio_mainline_over_v122": ratio,
            "legacy": legacy_summary.as_dict(),
            "mainline": mainline_summary.as_dict(),
        }
    return comparison


def codec_parity(*, seed: int = 7642077) -> dict[str, Any]:
    """Prove the shared 7-D <-> 18-D action boundary is exactly preserved."""

    from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
    from clearvla.policy.codec import PhysicalActionCodec
    from clearvla.policy.config import V39PolicyConfig

    legacy_config = V39PolicyConfig(
        action_dim=7,
        state_dim=7,
        action_horizon=24,
        gripper_field_dim=6,
        gripper_field_mode="legacy_handcrafted",
        arm_flow_mode="legacy_independent",
        physical_decode_delta_blend=0.25,
    )
    legacy_config.validate()
    legacy = PhysicalActionCodec(legacy_config)
    mainline = PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        gripper_field_dim=6,
        decode_delta_blend=0.25,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    action = torch.randn(3, 24, 7, generator=generator, dtype=torch.float32)
    state = torch.randn(3, 7, generator=generator, dtype=torch.float32)
    legacy_field = legacy.encode(action, state)
    mainline_field = mainline.encode(action, state)
    legacy_noise_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    mainline_noise_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    legacy_noise = legacy.sample_noise(
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=legacy_noise_generator,
        action_state=state,
    )
    mainline_noise = mainline.sample_noise(
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=mainline_noise_generator,
    )
    errors = {
        "encode_max_abs": float((legacy_field - mainline_field).abs().max()),
        "decode_max_abs": float(
            (
                legacy.decode(legacy_field, state)
                - mainline.decode(mainline_field, state)
            )
            .abs()
            .max()
        ),
        "source_noise_max_abs": float((legacy_noise - mainline_noise).abs().max()),
    }
    return {
        "exact": all(value == 0.0 for value in errors.values()),
        "physical_shape": list(mainline_field.shape),
        **errors,
    }


def _deep_get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for name in path.split("."):
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    return current


_CONFIG_FIELDS: dict[str, tuple[str, str]] = {
    "data_seed": ("performance.data_loader_seed", "data.seed"),
    "action_dim": ("policy.action_dim", "dimensions.action_dim"),
    "state_dim": ("policy.state_dim", "dimensions.state_dim"),
    "action_horizon": ("policy.action_horizon", "dimensions.action_horizon"),
    "hidden_size": ("policy.hidden_size", "dimensions.hidden_size"),
    "num_heads": ("policy.num_heads", "dimensions.num_heads"),
    "visual_token_dim": (
        "policy.visual_token_dim",
        "dimensions.visual_token_dim",
    ),
    "goal_token_dim": (
        "policy.goal_language_dim",
        "dimensions.goal_token_dim",
    ),
    "goal_max_tokens": (
        "policy.goal_language_max_tokens",
        "dimensions.goal_max_tokens",
    ),
    "num_cameras": ("policy.num_cameras", "dimensions.num_cameras"),
    "patches_per_camera": (
        "policy.patches_per_camera",
        "dimensions.patches_per_camera",
    ),
    "executed_history_length": (
        "policy.executed_history_length",
        "dimensions.executed_history_length",
    ),
    "action_basis_tokens": (
        "policy.action_basis_tokens",
        "dimensions.action_basis_tokens",
    ),
    "physical_gripper_channels": (
        "policy.gripper_field_dim",
        "bottom.gripper_field_dim",
    ),
    "physical_decode_delta_blend": (
        "policy.physical_decode_delta_blend",
        "bottom.physical_decode_delta_blend",
    ),
    "flow_time_distribution": (
        "policy.flow_matching_time_distribution",
        "bottom.flow_time_distribution",
    ),
    "evidence_mmdit_depth": (
        "policy.latent_cvae_mmdit_depth",
        "bottom.evidence_depth",
    ),
    "operator_rank": (
        "policy.latent_cvae_mmdit_operator_rank",
        "bottom.operator_rank",
    ),
    "operator_groups": (
        "policy.latent_cvae_mmdit_operator_groups",
        "bottom.operator_groups",
    ),
    "controller_tokens": (
        "policy.latent_cvae_mmdit_control_tokens",
        "bottom.controller_tokens",
    ),
    "controller_depth": (
        "policy.latent_cvae_mmdit_controller_depth",
        "bottom.controller_depth",
    ),
    "controller_heads": (
        "policy.latent_cvae_mmdit_controller_heads",
        "bottom.controller_heads",
    ),
    "learning_rate": ("trainer.lr", "optimizer.learning_rate"),
    "weight_decay": ("trainer.weight_decay", "optimizer.weight_decay"),
    "beta1": ("trainer.beta1", "optimizer.beta1"),
    "beta2": ("trainer.beta2", "optimizer.beta2"),
    "epsilon": ("trainer.eps", "optimizer.epsilon"),
    "grad_clip": ("trainer.grad_clip", "optimizer.grad_clip"),
    "warmup_steps": ("trainer.warmup_steps", "optimizer.warmup_steps"),
    "min_lr_ratio": ("trainer.min_lr_ratio", "optimizer.min_lr_ratio"),
    "future_weight": (
        "trainer.flow_jepa_future_loss_weight",
        "objectives.future_dynamics",
    ),
    "intent_weight": (
        "trainer.flow_jepa_interval_stage_loss_weight",
        "objectives.intent_structure",
    ),
    "proposal_weight": ("trainer.proposal_loss_weight", "objectives.proposal"),
    "motion_weight": ("trainer.arm_motion_loss_weight", "objectives.motion"),
    "decoded_action_weight": (
        "trainer.decoded_action_loss_weight",
        "objectives.decoded_action",
    ),
    "smooth_delta_weight": ("trainer.smooth_delta_weight", "objectives.smooth_delta"),
    "physical_delta_weight": (
        "trainer.physical_delta_consistency_weight",
        "objectives.physical_delta_consistency",
    ),
    "flow_warp_weight": (
        "trainer.flow_jepa_warp_loss_weight",
        "objectives.flow_warp",
    ),
    "flow_identity_advantage_weight": (
        "trainer.flow_jepa_identity_advantage_loss_weight",
        "objectives.flow_identity_advantage",
    ),
    "flow_static_identity_weight": (
        "trainer.flow_jepa_static_identity_loss_weight",
        "objectives.flow_static_identity",
    ),
    "flow_cycle_weight": (
        "trainer.flow_jepa_cycle_loss_weight",
        "objectives.flow_cycle",
    ),
    "flow_smoothness_weight": (
        "trainer.flow_jepa_smoothness_loss_weight",
        "objectives.flow_smoothness",
    ),
    "flow_uncertainty_weight": (
        "trainer.flow_jepa_uncertainty_nll_weight",
        "objectives.flow_uncertainty",
    ),
    "flow_refinement_weight": (
        "trainer.flow_jepa_refinement_sequence_loss_weight",
        "objectives.flow_refinement_sequence",
    ),
}


def compare_resolved_configs(
    legacy_manifest_path: Path,
    mainline_context_path: Path,
) -> dict[str, Any]:
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    mainline_context = json.loads(mainline_context_path.read_text(encoding="utf-8"))
    contract = legacy_manifest.get("contract")
    config = mainline_context.get("config")
    if not isinstance(contract, Mapping) or not isinstance(config, Mapping):
        raise ValueError("run manifests do not contain resolved configurations")
    legacy_view = {
        "policy": contract.get("policy_config", {}),
        "trainer": contract.get("trainer_contract", {}),
        "performance": contract.get("performance_contract", {}),
    }
    rows: dict[str, Any] = {}
    mismatch: list[str] = []
    unknown: list[str] = []
    for name, (legacy_path, mainline_path) in _CONFIG_FIELDS.items():
        legacy_value = _deep_get(legacy_view, legacy_path)
        mainline_value = _deep_get(config, mainline_path)
        if legacy_value is None or mainline_value is None:
            status = "unknown"
            unknown.append(name)
        elif isinstance(legacy_value, (int, float)) and isinstance(
            mainline_value, (int, float)
        ):
            status = (
                "match"
                if math.isclose(
                    float(legacy_value),
                    float(mainline_value),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                else "mismatch"
            )
        else:
            status = "match" if legacy_value == mainline_value else "mismatch"
        if status == "mismatch":
            mismatch.append(name)
        rows[name] = {
            "legacy": legacy_value,
            "mainline": mainline_value,
            "status": status,
        }
    legacy_splits = contract.get("splits")
    mainline_splits = mainline_context.get("splits")
    split_status = (
        "unknown"
        if not isinstance(legacy_splits, Mapping)
        or not isinstance(mainline_splits, Mapping)
        else "match"
        if dict(legacy_splits) == dict(mainline_splits)
        else "mismatch"
    )
    if split_status == "mismatch":
        mismatch.append("dataset_splits")
    elif split_status == "unknown":
        unknown.append("dataset_splits")
    rows["dataset_splits"] = {
        "legacy": legacy_splits,
        "mainline": mainline_splits,
        "status": split_status,
    }
    return {
        "all_known_fields_match": not mismatch and not unknown,
        "mismatches": mismatch,
        "unknown": unknown,
        "fields": rows,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V122 vs independent mainline: early-loss comparison",
        "",
        "This report compares scale and early direction, not bitwise loss equality.",
        "",
        "## Hard boundary",
        "",
    ]
    codec = report["codec_parity"]
    lines.append(
        f"- 7-D/18-D action codec exact: **{codec['exact']}** "
        f"(encode={codec['encode_max_abs']:.3g}, decode={codec['decode_max_abs']:.3g}, "
        f"noise={codec['source_noise_max_abs']:.3g})"
    )
    config = report.get("resolved_config")
    if isinstance(config, Mapping):
        lines.append(
            f"- Shared resolved fields match: **{config['all_known_fields_match']}**; "
            f"mismatches={config['mismatches']}; unknown={config['unknown']}"
        )
    lines.extend(
        [
            "",
            "## Early loss scale",
            "",
            "| Semantic metric | V122 mean | Mainline mean | Ratio | Scale | "
            "V122 half Δ | Mainline half Δ |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for name, row in report["metrics"].items():
        if row["status"] == "not_comparable":
            lines.append(f"| {name} | — | — | — | not comparable | — | — |")
            continue
        old = row["legacy"]
        new = row["mainline"]
        lines.append(
            f"| {name} | {old['mean']:.6g} | {new['mean']:.6g} | "
            f"{row['mean_ratio_mainline_over_v122']:.4f} | {row['status']} | "
            f"{old['half_change_fraction']:+.3%} | "
            f"{new['half_change_fraction']:+.3%} |"
        )
    lines.extend(
        [
            "",
            "`close` means a mean ratio in [0.75, 1.333]; `same_order` means "
            "[0.5, 2.0]. Total and representation losses include architecture-specific "
            "terms, so action-flow rows are the primary migration signal.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *,
    legacy_log: Path,
    mainline_metrics: Path,
    steps: int,
    legacy_manifest: Path | None,
    mainline_context: Path | None,
) -> dict[str, Any]:
    legacy_rows = load_legacy_train_rows(legacy_log, limit=steps)
    mainline_rows = load_mainline_train_rows(mainline_metrics, limit=steps)
    report: dict[str, Any] = {
        "schema": "clearvla-mainline-early-loss-comparison-v1",
        "criterion": "shared-scale-and-early-direction-not-bitwise-equality",
        "requested_steps": int(steps),
        "observed_steps": {
            "legacy": len(legacy_rows),
            "mainline": len(mainline_rows),
        },
        "codec_parity": codec_parity(),
        "metrics": compare_rows(legacy_rows, mainline_rows),
    }
    if legacy_manifest is not None and mainline_context is not None:
        report["resolved_config"] = compare_resolved_configs(
            legacy_manifest,
            mainline_context,
        )
    core = report["metrics"]
    core_status = [
        core[name]["status"]
        for name in ("action_flow", "native_action_flow")
        if core[name]["status"] != "not_comparable"
    ]
    report["conclusion"] = {
        "comparable_early_action_scale": bool(core_status)
        and all(status in {"close", "same_order"} for status in core_status),
        "bitwise_equality_expected": False,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen V122 and independent-mainline early losses",
    )
    parser.add_argument("--legacy-log", type=Path, required=True)
    parser.add_argument("--mainline-metrics", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--mainline-context", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    report = build_report(
        legacy_log=args.legacy_log,
        mainline_metrics=args.mainline_metrics,
        steps=args.steps,
        legacy_manifest=args.legacy_manifest,
        mainline_context=args.mainline_context,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report), flush=True)


if __name__ == "__main__":
    main()
