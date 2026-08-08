"""Summarize paired V101+ model-path intervention results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _effect_label(row: dict[str, Any]) -> str:
    interval = row["mse_delta_ci"]
    low = float(interval["ci95_low"])
    high = float(interval["ci95_high"])
    if low > 0.0:
        return "removal/mismatch is significantly harmful: causal use detected"
    if high < 0.0:
        return "removal/mismatch is significantly helpful: path is currently harmful"
    return "CI crosses zero: no significant causal effect at this sample size"


def _percent(value: float) -> str:
    return f"{100.0 * float(value):+.3f}%"


def _model_path_version(schema: str) -> str:
    supported = {
        "clearvla-v101-action-path-intervention-v1": "v101",
        "clearvla-v101-action-path-intervention-v2": "v101",
        "clearvla-v101-action-path-intervention-v3": "v101",
        "clearvla-v103-model-path-intervention-v1": "v103",
        "clearvla-v103-model-path-intervention-v2": "v103",
        "clearvla-v103-model-path-intervention-v3": "v103",
        "clearvla-v104-model-path-intervention-v3": "v104",
        "clearvla-v105-model-path-intervention-v4": "v105",
        "clearvla-v106-model-path-intervention-v5": "v106",
        "clearvla-v107-model-path-intervention-v6": "v107",
        "clearvla-v108-model-path-intervention-v7": "v108",
        "clearvla-v109-model-path-intervention-v8": "v109",
        "clearvla-v110-model-path-intervention-v9": "v110",
        "clearvla-v111-model-path-intervention-v10": "v111",
        "clearvla-v112-model-path-intervention-v11": "v112",
        "clearvla-v113-model-path-intervention-v12": "v113",
        "clearvla-v113-model-path-intervention-v13": "v113",
        "clearvla-v114-model-path-intervention-v14": "v114",
        "clearvla-v115-model-path-intervention-v15": "v115",
        "clearvla-v116-model-path-intervention-v16": "v116",
        "clearvla-v117-model-path-intervention-v17": "v117",
        "clearvla-differential-intent-effect-model-path-v18": (
            "differential-intent-effect-323"
        ),
        "clearvla-grounded-intent-effect-323-model-path-v1": (
            "grounded-intent-effect-323"
        ),
    }
    try:
        return supported[str(schema)]
    except KeyError as exc:
        raise ValueError("input is not a supported model-path intervention result") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a paired ClearVLA model-path intervention result."
    )
    parser.add_argument("result", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    probe = payload.get(
        "model_path_intervention",
        payload.get("action_path_intervention", payload),
    )
    version = _model_path_version(str(probe.get("schema", "")))

    identity = payload.get("run_identity", {})
    episode_clusters = probe.get("episode_clusters", probe.get("selected_episode_count", "unknown"))
    candidate_episode_count: int | str = len(probe.get("candidate_episode_ids", [])) or "unknown"
    print(
        f"[{version}-{'action-path' if version == 'v101' else 'model-path'}-summary] "
        f"checkpoint_epoch={identity.get('checkpoint_epoch')} "
        f"step={identity.get('checkpoint_global_step')} "
        f"samples={probe['intervention_samples']} "
        f"batches={probe['finished_intervention_batches']}/"
        f"{probe['planned_batches']} "
        f"selection={probe.get('selection_strategy', 'legacy_uniform')} "
        f"event_batches={probe.get('selected_event_batches', 'unknown')}/"
        f"{probe.get('event_candidate_batches', 'unknown')} "
        f"episodes={episode_clusters}/{candidate_episode_count} "
        f"target_events={probe['modes']['baseline'].get('gripper_target_events')} "
        f"baseline_identity_max={probe['patched_baseline_max_abs_delta']:.3e} "
        f"replay_batches={probe.get('baseline_identity_checked_batches', 'legacy')}"
    )
    replay_tolerance = float(probe.get("baseline_identity_tolerance", 1e-8))
    replay_checked = probe.get("baseline_identity_checked_batches")
    replay_coverage_valid = (
        replay_checked is None
        or int(replay_checked) == int(probe["finished_intervention_batches"])
    )
    if (
        float(probe["patched_baseline_max_abs_delta"]) > replay_tolerance
        or not replay_coverage_valid
    ):
        print(
            "[model-path-replay-invalid] explicit none does not match ordinary "
            "deployment on every selected batch; intervention effect sizes "
            "and confidence intervals are provisional"
        )
    mode_width = max(33, *(len(str(mode)) for mode in probe["modes"]))
    print(
        f"{'mode':{mode_width}s} full_rmse   band_1_4   band_5_12  "
        "band_13_24  rel_mse_delta       ci95"
    )
    baseline = probe["modes"]["baseline"]
    print(
        f"{'baseline':{mode_width}s} "
        f"{baseline['full_rmse']:9.6f} "
        f"{baseline['action_band_1_4_rmse']:10.6f} "
        f"{baseline['action_band_5_12_rmse']:11.6f} "
        f"{baseline['action_band_13_24_rmse']:12.6f} "
        f"{'-':>13s} {'-':>18s}"
    )
    for mode, row in probe["paired"].items():
        metrics = probe["modes"][mode]
        interval = row["mse_delta_ci"]
        ci = f"[{float(interval['ci95_low']):+.3e},{float(interval['ci95_high']):+.3e}]"
        print(
            f"{mode:{mode_width}s} "
            f"{metrics['full_rmse']:9.6f} "
            f"{metrics['action_band_1_4_rmse']:10.6f} "
            f"{metrics['action_band_5_12_rmse']:11.6f} "
            f"{metrics['action_band_13_24_rmse']:12.6f} "
            f"{_percent(row['relative_mse_delta']):>13s} {ci:>18s}"
        )

    print("\nCausal interpretation:")
    for mode, row in probe["paired"].items():
        print(
            f"- {mode}: {_effect_label(row)}; "
            f"action_delta_rmse={float(row['action_delta_rmse']):.6f}; "
            f"relative_mse_delta={_percent(row['relative_mse_delta'])}"
        )

    print("\nRelative MSE change by horizon:")
    for mode, row in probe["paired"].items():
        bands = " / ".join(
            f"{label}:{_percent(band['relative_mse_delta'])}"
            for label, band in row["bands"].items()
        )
        print(f"- {mode}: {bands}")

    reader = probe.get("reader_intervention_diagnostics", {})
    if reader:
        print("\nPost-reader detail intervention:")
        for mode in (
            "baseline",
            "raw_detail_zero",
            "raw_detail_spatial_shuffle",
            "raw_value_zero",
            "raw_value_spatial_shuffle",
        ):
            row = reader.get(mode)
            if not row:
                continue
            print(
                f"- {mode}: "
                f"selector_residual="
                f"{float(row.get('flow_jepa_raw_post_reader_detail_selector_residual_norm', 0.0)):.6f} "
                f"value_residual="
                f"{float(row.get('flow_jepa_raw_post_reader_detail_value_residual_norm', 0.0)):.6f} "
                f"selector_delta="
                f"{float(row.get('flow_jepa_raw_post_reader_detail_selector_intervention_delta', 0.0)):.6f} "
                f"value_delta="
                f"{float(row.get('flow_jepa_raw_post_reader_detail_value_intervention_delta', 0.0)):.6f}"
            )
    representation = probe.get("representation", {})
    reliability_keys = (
        "grounded_p2_effect_value_pre_mask_rms",
        "grounded_p2_effect_value_post_validity_rms",
        "grounded_p2_effect_value_post_reliability_rms",
        "grounded_p2_effect_reliability_valid_mean",
        "grounded_p2_effect_reliability_attenuation_ratio",
    )
    if any(key in representation for key in reliability_keys):
        print("\nGrounded P2 reliability boundary:")
        print(
            "- "
            + " ".join(
                f"{key}={float(representation[key]):.6g}"
                for key in reliability_keys
                if key in representation
            )
        )
    boundary = probe.get("boundary_diagnostics", {})
    if boundary:
        print("\nIntervened boundary deltas:")
        for mode, row in boundary.items():
            if not row:
                continue
            values = " ".join(f"{key}={float(value):.6g}" for key, value in sorted(row.items()))
            print(f"- {mode}: {values}")

    current_mask = probe.get("current_context_mask_comparison")
    if current_mask:
        print("\nMatched current-context mask comparison:")
        modes = current_mask.get("modes", {})
        deltas = current_mask.get("masked_minus_unmasked", {})
        for key in (
            "flow_jepa_future_prediction",
            "flow_jepa_future_active_direction_loss",
            "flow_jepa_future_active_composite_loss",
            "flow_jepa_horizon_address",
        ):
            if key in modes.get("unmasked", {}) and key in modes.get("masked", {}):
                print(
                    f"- {key}: "
                    f"unmasked={float(modes['unmasked'][key]):.6g} "
                    f"masked={float(modes['masked'][key]):.6g} "
                    f"delta={float(deltas[key]):+.6g}"
                )
        active_offsets = sorted(
            {
                int(key.split("_")[4])
                for key in modes.get("masked", {})
                if key.startswith("flow_jepa_future_horizon_") and key.endswith("_active_loss")
            }
        )
        for offset in active_offsets:
            active_key = f"flow_jepa_future_horizon_{offset}_active_loss"
            direction_key = f"flow_jepa_future_horizon_{offset}_active_direction"
            print(
                f"- H{offset}: "
                f"active={float(modes['unmasked'][active_key]):.6g}->"
                f"{float(modes['masked'][active_key]):.6g} "
                f"direction={float(modes['unmasked'][direction_key]):.6g}->"
                f"{float(modes['masked'][direction_key]):.6g}"
            )
        mask_boundary = current_mask.get("masked_boundary", {})
        if mask_boundary:
            print(
                "- mask boundary: "
                + " ".join(
                    f"{key}={float(value):.6g}" for key, value in sorted(mask_boundary.items())
                )
            )

    matrix = probe.get("acceptance_matrix", {})
    if matrix:
        replay = matrix.get("replay", {})
        aggregate = matrix.get("aggregate", {})
        print("\nModel-path acceptance matrix:")
        print(
            "- matched replay: "
            f"{'pass' if replay.get('numerically_identical') else 'fail'} "
            f"(max_abs_delta={float(replay.get('baseline_max_abs_delta', 0.0)):.3e})"
        )
        for label, key in (
            ("spatial boundary accessible", "spatial_boundary_changed"),
            ("spatial path reaches action", "spatial_path_reaches_action"),
            ("detail boundary accessible", "detail_boundary_changed"),
            ("detail path reaches action", "detail_path_reaches_action"),
            ("goal path reaches action", "goal_path_reaches_action"),
            ("history path reaches action", "history_path_reaches_action"),
            (
                "direct history condition reaches action",
                "history_condition_path_reaches_action",
            ),
            (
                "history proposal boundary accessible",
                "history_proposal_boundary_changed",
            ),
            (
                "history proposal reaches action",
                "history_proposal_path_reaches_action",
            ),
            ("phase path reaches action", "phase_path_reaches_action"),
            (
                "functional W routes accessible",
                "functional_world_route_boundary_changed",
            ),
            (
                "functional W routes reach action",
                "functional_world_route_reaches_action",
            ),
            (
                "P1 appearance gateway accessible",
                "p1_appearance_gateway_boundary_changed",
            ),
            (
                "P1 appearance gateway reaches action",
                "p1_appearance_gateway_reaches_action",
            ),
            (
                "current-context mask changes boundary",
                "current_context_mask_boundary_changed",
            ),
            (
                "current-context mask reaches action",
                "current_context_mask_reaches_action",
            ),
        ):
            value = aggregate.get(key)
            status = "not enabled" if value is None else "pass" if value else "fail"
            print(f"- {label}: {status}")
        slots = matrix.get("address_slot_structure", {})
        if slots:
            observed = bool(slots.get("observed"))
            print(
                "- multi-slot address structure: "
                f"{'observed' if observed else 'not observed'} "
                f"(slots={slots.get('configured_slot_count')}, "
                f"center_distance_norm="
                f"{slots.get('coarse_center_pair_distance_normalized')}, "
                f"posterior_hellinger="
                f"{slots.get('coarse_posterior_pair_hellinger')}, "
                f"policy_effective_slots="
                f"{slots.get('policy_slot_effective_count')}, "
                f"query_variation="
                f"{slots.get('policy_slot_query_variation')})"
            )
            if observed:
                print(
                    "- multi-slot numerical checks: "
                    f"centers={'pass' if slots.get('coarse_centers_numerically_distinct') else 'fail'} "
                    f"posteriors={'pass' if slots.get('coarse_posteriors_numerically_distinct') else 'fail'} "
                    f"policy_mass={'pass' if slots.get('policy_uses_multiple_slots_numerically') else 'fail'} "
                    f"query_conditioning={'pass' if slots.get('policy_slot_route_varies_by_query') else 'fail'}"
                )
        for route_name, route in matrix.get("typed_route_structure", {}).items():
            print(
                f"- typed route {route_name}: "
                f"observed={route.get('observed')} "
                f"effective_sources={route.get('source_effective_count')} "
                f"multi_source="
                f"{route.get('uses_multiple_sources_numerically')} "
                f"query_varies="
                f"{route.get('query_axes_vary_numerically')}"
            )
        far = matrix.get("long_horizon_pairwise", {})
        if far:
            print(
                "- +48 joint distinguishability: "
                f"{'pass' if far.get('joint_distinguishable_from_each_single_path') else 'fail'} "
                f"(joint-vs-typed="
                f"{float(far.get('joint_vs_typed_only_action_delta_rmse', 0.0)):.6g}, "
                f"joint-vs-bottom="
                f"{float(far.get('joint_vs_bottom_only_action_delta_rmse', 0.0)):.6g})"
            )
        print(
            "- utility: inspect each paired CI above; causal accessibility alone "
            "is not a utility pass"
        )


if __name__ == "__main__":
    main()
