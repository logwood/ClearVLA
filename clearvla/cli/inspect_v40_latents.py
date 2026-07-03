from __future__ import annotations

"""Inspect V40.1 unified intervention latents on a dataset split.

The V40 policy keeps the V39 model/checkpoint classes, but the meaning of its
layer contract is different: ``rollout_effect_pred`` comes from one unified
intervention head and every layer also exposes neutral and intervention latent
states.  This command retains the V39.2 contract-inspection workflow while
adding diagnostics for those V40-specific tensors.

This is a teacher-forced contract diagnostic.  It is not a deployable policy
evaluation and the generated metadata states that explicitly.
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.cli.inspect_v39_2_latents import (
    _build_loader_and_conditioner,
    _cov_rank_stats,
    _effect_distance_np,
    _flatten_tokens,
    _maybe_import_matplotlib,
    _pca_2d,
    _safe_ratio,
    _sample_indices,
    _write_json,
)
from clearvla.experiments.classic_policy_lab.cli_common import add_data_args, resolve_device
from clearvla.experiments.observed_state_lab.policy_v39 import V39PolicyConfig, V39PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    V39PolicyTrainerConfig,
    gripper_event_labels,
    prepare_v39_policy_sample,
)
from clearvla.experiments.observed_state_lab.world_runtime import autocast_context, jsonable


V40_CHECKPOINT_SCHEMA = "clearvla-v40-policy-checkpoint-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect V40.1 unified intervention latents and state-shuffle diagnostics."
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--pca-max-points", type=int, default=1200)
    parser.add_argument(
        "--state-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the optional V40 same-action/state-canvas shuffle at inspection time. "
            "This is enabled for inspection even when it was disabled during training."
        ),
    )
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-prefix", default="v40_latent")
    return parser.parse_args()


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size:
        return float("nan")
    denom = np.maximum(np.linalg.norm(a, axis=1), 1e-8) * np.maximum(
        np.linalg.norm(b, axis=1), 1e-8
    )
    return float(np.mean(np.sum(a * b, axis=1) / denom))


def neutral_intervention_metrics(
    neutral: np.ndarray,
    intervention: np.ndarray,
    effect: np.ndarray,
) -> dict[str, float | int]:
    """Summarize paired neutral/intervention vectors for one layer."""

    neutral = np.asarray(neutral, dtype=np.float64)
    intervention = np.asarray(intervention, dtype=np.float64)
    effect = np.asarray(effect, dtype=np.float64)
    if neutral.shape != intervention.shape or neutral.shape != effect.shape:
        raise ValueError(
            "neutral, intervention, and effect vectors must have identical shapes, "
            f"got {neutral.shape}, {intervention.shape}, {effect.shape}"
        )
    if neutral.ndim != 2:
        raise ValueError(f"expected pooled [N,D] vectors, got {neutral.shape}")
    if neutral.shape[0] == 0:
        return {
            "n": 0,
            "paired_l2_mean": float("nan"),
            "paired_rms": float("nan"),
            "paired_cosine_distance": float("nan"),
            "centroid_distance": float("nan"),
            "within_class_rms": float("nan"),
            "centroid_separation_ratio": float("nan"),
            "effect_reconstruction_mse": float("nan"),
            "effect_alignment_cosine": float("nan"),
            "neutral_norm": float("nan"),
            "intervention_norm": float("nan"),
        }
    delta = intervention - neutral
    neutral_centered = neutral - neutral.mean(axis=0, keepdims=True)
    intervention_centered = intervention - intervention.mean(axis=0, keepdims=True)
    # Compare centroid L2 distance with within-class L2 spread in the same
    # units.  Using per-coordinate RMS here would inflate the ratio by sqrt(D).
    within_rms = float(
        np.sqrt(
            0.5
            * (
                np.mean(np.sum(neutral_centered**2, axis=1))
                + np.mean(np.sum(intervention_centered**2, axis=1))
            )
        )
    )
    centroid_distance = float(np.linalg.norm(intervention.mean(axis=0) - neutral.mean(axis=0)))
    return {
        "n": int(neutral.shape[0]),
        "paired_l2_mean": float(np.linalg.norm(delta, axis=1).mean()),
        "paired_rms": float(np.sqrt(np.mean(delta**2))),
        "paired_cosine_distance": float(1.0 - _mean_row_cosine(neutral, intervention)),
        "centroid_distance": centroid_distance,
        "within_class_rms": within_rms,
        "centroid_separation_ratio": _safe_ratio(centroid_distance, within_rms),
        "effect_reconstruction_mse": float(np.mean((delta - effect) ** 2)),
        "effect_alignment_cosine": _mean_row_cosine(delta, effect),
        "neutral_norm": float(np.linalg.norm(neutral, axis=1).mean()),
        "intervention_norm": float(np.linalg.norm(intervention, axis=1).mean()),
    }


def state_shuffle_metrics(
    effect: np.ndarray,
    shuffled_effect: np.ndarray,
    target: np.ndarray,
    *,
    step_delta: np.ndarray | None = None,
    shuffled_step_delta: np.ndarray | None = None,
    step_target: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute V40 state-shuffle distance and prediction-change diagnostics."""

    effect = np.asarray(effect, dtype=np.float64)
    shuffled_effect = np.asarray(shuffled_effect, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if effect.shape != shuffled_effect.shape or effect.shape != target.shape:
        raise ValueError(
            "effect, shuffled effect, and target vectors must have identical shapes, "
            f"got {effect.shape}, {shuffled_effect.shape}, {target.shape}"
        )
    real_distance = _effect_distance_np(effect, target)
    shuffled_distance = _effect_distance_np(shuffled_effect, target)
    row: dict[str, float | int] = {
        "n": int(effect.shape[0]),
        "real_distance": float(real_distance.mean()) if real_distance.size else float("nan"),
        "state_shuffle_distance": (
            float(shuffled_distance.mean()) if shuffled_distance.size else float("nan")
        ),
        "d_state_shuffle": (
            float(shuffled_distance.mean() - real_distance.mean())
            if real_distance.size
            else float("nan")
        ),
        "effect_change_state_shuffle": (
            float(np.mean((shuffled_effect - effect) ** 2)) if effect.size else float("nan")
        ),
    }
    supplied = (step_delta, shuffled_step_delta, step_target)
    if any(item is not None for item in supplied):
        if not all(item is not None for item in supplied):
            raise ValueError("all step-delta arrays must be supplied together")
        step_delta = np.asarray(step_delta, dtype=np.float64)
        shuffled_step_delta = np.asarray(shuffled_step_delta, dtype=np.float64)
        step_target = np.asarray(step_target, dtype=np.float64)
        if step_delta.shape != shuffled_step_delta.shape or step_delta.shape != step_target.shape:
            raise ValueError("step-delta vectors must have identical shapes")
        step_real = _effect_distance_np(step_delta, step_target)
        step_shuffled = _effect_distance_np(shuffled_step_delta, step_target)
        row.update(
            {
                "step_real_distance": float(step_real.mean()),
                "step_state_shuffle_distance": float(step_shuffled.mean()),
                "step_d_state_shuffle": float(step_shuffled.mean() - step_real.mean()),
                "step_change_state_shuffle": float(
                    np.mean((shuffled_step_delta - step_delta) ** 2)
                ),
            }
        )
    return row


def _milestone_step_target(target: torch.Tensor, config: V39PolicyConfig) -> torch.Tensor:
    grid = int(config.num_cameras) * int(config.future_grid_size) ** 2
    anchors = int(config.future_anchors)
    expected = anchors * grid
    if target.ndim != 3 or int(target.shape[1]) != expected:
        raise ValueError(
            f"rollout target must be [B,{expected},H] for milestone diagnostics, got {tuple(target.shape)}"
        )
    grouped = target.float().reshape(target.shape[0], anchors, grid, target.shape[-1])
    first = grouped[:, :1]
    rest = grouped[:, 1:] - grouped[:, :-1]
    return torch.cat([first, rest], dim=1).reshape_as(target)


def _effect_distance_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match policy_runtime_v39._effect_distance without importing a private runtime helper."""

    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"effect tensors must be matching [B,N,H] arrays, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    diff = pred.float() - target.float().detach()
    mse = diff.square().mean(dim=(1, 2))
    pred_n = torch.nn.functional.normalize(pred.float(), dim=-1)
    target_n = torch.nn.functional.normalize(target.float().detach(), dim=-1)
    cosine = 1.0 - (pred_n * target_n).sum(dim=-1).mean(dim=1)
    return mse + 0.10 * cosine


def _plot_layer_metrics(plt, rows: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(row["layer"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    for key in ("latent_mse", "std_ratio", "norm_ratio"):
        axes[0].plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    for key in ("d_hold", "d_shuffle", "effect_change_shuffle"):
        axes[1].plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    axes[0].set_title("V40.1 unified intervention contract by layer")
    axes[1].set_xlabel("DiT layer")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_action_counterfactuals(plt, rows: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(row["layer"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for key in ("real_distance", "hold_distance", "shuffle_distance"):
        ax.plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    ax.set_xlabel("DiT layer")
    ax.set_title("V40.1 action counterfactual distance to target")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _available_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("available"))]


def _plot_state_shuffle(plt, rows: list[dict[str, Any]], out_path: Path) -> None:
    rows = _available_rows(rows)
    if not rows:
        return
    layers = [int(row["layer"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    for key in ("real_distance", "state_shuffle_distance"):
        axes[0].plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    for key in (
        "d_state_shuffle",
        "effect_change_state_shuffle",
        "step_d_state_shuffle",
        "step_change_state_shuffle",
    ):
        if all(key in row for row in rows):
            axes[1].plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    axes[0].set_title("Experimental same-action state/canvas shuffle")
    axes[1].set_xlabel("DiT layer")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_separation(plt, rows: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(row["layer"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    for key in ("paired_l2_mean", "centroid_distance", "neutral_norm", "intervention_norm"):
        axes[0].plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    if all("token_paired_l2_mean" in row for row in rows):
        axes[0].plot(
            layers,
            [float(row["token_paired_l2_mean"]) for row in rows],
            marker="o",
            label="token_paired_l2_mean",
        )
    for key in ("paired_cosine_distance", "centroid_separation_ratio", "effect_alignment_cosine"):
        axes[1].plot(layers, [float(row[key]) for row in rows], marker="o", label=key)
    axes[0].set_title("Neutral vs intervention latent separation")
    axes[1].set_xlabel("DiT layer")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_neutral_intervention_pca(
    plt,
    neutral_coords: np.ndarray,
    intervention_coords: np.ndarray,
    out_path: Path,
    *,
    layer: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for n, z in zip(neutral_coords, intervention_coords):
        ax.plot([n[0], z[0]], [n[1], z[1]], color="0.75", alpha=0.22, linewidth=0.6)
    ax.scatter(neutral_coords[:, 0], neutral_coords[:, 1], s=14, alpha=0.75, label="neutral")
    ax.scatter(
        intervention_coords[:, 0], intervention_coords[:, 1], s=14, alpha=0.75, label="intervention"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Layer {layer}: neutral vs intervention latent PCA")
    ax.grid(True, alpha=0.20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _write_markdown_report(
    out_dir: Path,
    *,
    metadata: dict[str, Any],
    layer_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    separation_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# V40.1 latent inspection",
        "",
        "> Teacher-forced flow-contract inspection. Action metrics are not deployable policy evaluation.",
        "",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Split: `{metadata['split']}`",
        f"- Processed batches/samples: {metadata['processed_batches']} / {metadata['processed_samples']}",
        f"- State shuffle requested/available: {metadata['state_shuffle_requested']} / {metadata['state_shuffle_available']}",
        "",
        "The state shuffle keeps the candidate action but swaps the layer canvas and explicit state/history context. "
        "It is an experimental diagnostic, not a strict causal estimate, because the layer canvas may already contain action information.",
        "",
        "## Layer summary",
        "",
        "| layer | latent MSE | std ratio | action d_shuffle | state d_shuffle | neutral/intervention L2 | separation ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    state_by_layer = {int(row["layer"]): row for row in state_rows}
    separation_by_layer = {int(row["layer"]): row for row in separation_rows}
    for row in layer_rows:
        layer = int(row["layer"])
        state = state_by_layer[layer]
        separation = separation_by_layer[layer]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(layer),
                    _format_metric(row["latent_mse"]),
                    _format_metric(row["std_ratio"]),
                    _format_metric(row["d_shuffle"]),
                    _format_metric(state.get("d_state_shuffle")),
                    _format_metric(separation["token_paired_l2_mean"]),
                    _format_metric(separation["centroid_separation_ratio"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading the diagnostics",
            "",
            "- Positive `d_shuffle` means the real candidate action is closer to the target than the hard action negative.",
            "- Positive `d_state_shuffle` means swapping state/canvas context makes the same-action prediction worse.",
            "- `effect_change_state_shuffle` measures sensitivity even when target-distance ordering is ambiguous.",
            "- Neutral/intervention `effect_reconstruction_mse` and `token_effect_reconstruction_mse` should be near numerical zero: their paired difference is the unified effect by construction.",
            "- Separation magnitude alone is not evidence of correctness; read it with target error and action/state counterfactual deltas.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_html_index(out_dir: Path, *, layer_count: int, prefix: str, has_state: bool) -> None:
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>V40.1 latent inspection</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1100px} img{max-width:100%;border:1px solid #ddd;margin:8px 0 24px} code{background:#f4f4f4;padding:2px 4px}</style>",
        "<h1>V40.1 unified intervention latent inspection</h1>",
        "<p><strong>Teacher-forced contract diagnostic:</strong> action metrics are not deployable policy evaluation.</p>",
        f"<h2>Layer contract</h2><img src='{prefix}_layer_metrics.png'>",
        f"<h2>Action counterfactuals</h2><img src='{prefix}_action_counterfactuals.png'>",
    ]
    if has_state:
        lines.append(f"<h2>Experimental state shuffle</h2><img src='{prefix}_state_shuffle.png'>")
    else:
        lines.append("<h2>Experimental state shuffle</h2><p>Unavailable. It requires --state-shuffle and batches with at least two samples.</p>")
    lines.extend(
        [
            f"<h2>Neutral vs intervention separation</h2><img src='{prefix}_neutral_intervention.png'>",
            "<h2>Neutral/intervention PCA by layer</h2>",
        ]
    )
    for layer in range(layer_count):
        lines.append(f"<h3>Layer {layer}</h3><img src='{prefix}_pca_layer{layer}_neutral_intervention.png'>")
    lines.append("<h2>Data files</h2><ul>")
    for name in (
        "report.md",
        "summary.json",
        "latent_probe_table.csv",
        "latent_counterfactual_by_layer.csv",
        "state_shuffle_by_layer.csv",
        "neutral_intervention_by_layer.csv",
        "latent_vectors.npz",
    ):
        lines.append(f"<li><code>{name}</code></li>")
    lines.append("</ul>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def inspect_latents(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != V40_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"--checkpoint must use {V40_CHECKPOINT_SCHEMA!r}; got {payload.get('schema')!r}"
        )
    checkpoint_config = V39PolicyConfig(**payload["policy_config"])
    if not int(checkpoint_config.layer_contract_adapters) or not int(
        checkpoint_config.layer_recurrent_consequence
    ):
        raise ValueError("checkpoint does not enable the V40 unified layer intervention contract")

    loader, conditioner, cameras, action_norm, state_norm, skipped = _build_loader_and_conditioner(
        args, payload, device, dtype
    )
    del action_norm, state_norm
    runtime_config = replace(
        checkpoint_config, layer_state_counterfactual=int(bool(args.state_shuffle))
    )
    trainer = V39PolicyTrainerConfig(**payload["trainer_config"])
    system = V39PolicySystem(runtime_config)
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32)
    system.eval()

    keys = (
        "effect",
        "delta",
        "target",
        "hold_effect",
        "shuffle_effect",
        "state_effect",
        "state_real_effect",
        "state_residual_target",
        "state_distance_real",
        "state_distance_shuffle",
        "state_effect_change",
        "state_step_delta",
        "state_real_step_delta",
        "state_step_target",
        "state_step_distance_real",
        "state_step_distance_shuffle",
        "state_step_change",
        "neutral",
        "intervention",
        "token_paired_l2",
        "token_paired_rms",
        "token_paired_cosine_distance",
        "token_effect_reconstruction_mse",
    )
    layer_count = int(runtime_config.depth)
    by_layer: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {key: [] for key in keys} for layer in range(layer_count)
    }
    action_norm_rows: list[np.ndarray] = []
    gripper_any_rows: list[np.ndarray] = []
    processed_batches = 0
    processed_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if args.max_batches and batch_index > args.max_batches:
                break
            sample = prepare_v39_policy_sample(
                batch,
                conditioner=conditioner,
                system=system,
                camera_names=cameras,
                device=device,
                dtype=dtype,
                include_target_visual=True,
            )
            with autocast_context(device, dtype):
                target_pack = system.build_rollout_target_pack(
                    sample["visual"], sample["target_visual"]
                )
                output = system.flow_training_forward(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    sample["policy_action"],
                    action_state=sample["action_state"],
                    rollout_target_pack=target_pack,
                    make_counterfactuals=True,
                )
            layers = output.get("layer_contracts")
            if not isinstance(layers, list) or len(layers) != layer_count:
                raise ValueError("checkpoint/model did not return one V40 layer contract per DiT layer")
            labels = gripper_event_labels(
                target_raw=sample["policy_action_raw"],
                current_raw=sample["state_raw"],
                gripper_index=system.policy_config.gripper_index,
                threshold=trainer.gripper_event_threshold,
            )
            action_norm_rows.append(
                sample["policy_action"].float().norm(dim=-1).mean(dim=1).detach().cpu().numpy()
            )
            gripper_any_rows.append((labels != 0).any(dim=1).float().detach().cpu().numpy())
            target = output["rollout_effect_target"]
            target_vec = _flatten_tokens(target).detach().cpu().numpy()
            for layer_index, entry in enumerate(layers):
                required = (
                    "rollout_effect_pred",
                    "rollout_delta_pred",
                    "unified_intervention_latent_pred",
                    "neutral_latent_pred",
                )
                missing = [key for key in required if key not in entry]
                if missing:
                    raise ValueError(
                        f"layer {layer_index} is missing V40 unified outputs: {', '.join(missing)}"
                    )
                store = by_layer[layer_index]
                store["effect"].append(
                    _flatten_tokens(entry["rollout_effect_pred"]).detach().cpu().numpy()
                )
                store["delta"].append(
                    _flatten_tokens(entry["rollout_delta_pred"]).detach().cpu().numpy()
                )
                store["target"].append(target_vec)
                residual_target = target.float().detach()
                if "rollout_base_effect_pred" in entry:
                    residual_target = residual_target - entry["rollout_base_effect_pred"].float().detach()
                neutral_tensor = entry["neutral_latent_pred"].float()
                intervention_tensor = entry["unified_intervention_latent_pred"].float()
                effect_tensor = entry["rollout_effect_pred"].float()
                if (
                    neutral_tensor.shape != intervention_tensor.shape
                    or neutral_tensor.shape != effect_tensor.shape
                ):
                    raise ValueError(
                        f"layer {layer_index} neutral/intervention/effect shapes do not align: "
                        f"{tuple(neutral_tensor.shape)}, {tuple(intervention_tensor.shape)}, "
                        f"{tuple(effect_tensor.shape)}"
                    )
                store["neutral"].append(
                    _flatten_tokens(neutral_tensor).detach().cpu().numpy()
                )
                store["intervention"].append(
                    _flatten_tokens(intervention_tensor).detach().cpu().numpy()
                )
                token_delta = intervention_tensor - neutral_tensor
                batch_size = int(token_delta.shape[0])
                token_delta_flat = token_delta.reshape(batch_size, -1, token_delta.shape[-1])
                neutral_flat = neutral_tensor.reshape(
                    batch_size, -1, neutral_tensor.shape[-1]
                )
                intervention_flat = intervention_tensor.reshape(
                    batch_size, -1, intervention_tensor.shape[-1]
                )
                effect_flat = effect_tensor.reshape(batch_size, -1, effect_tensor.shape[-1])
                store["token_paired_l2"].append(
                    token_delta_flat.norm(dim=-1).mean(dim=1).detach().cpu().numpy()
                )
                store["token_paired_rms"].append(
                    token_delta_flat.square().mean(dim=(1, 2)).sqrt().detach().cpu().numpy()
                )
                store["token_paired_cosine_distance"].append(
                    (1.0 - torch.cosine_similarity(neutral_flat, intervention_flat, dim=-1))
                    .mean(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                store["token_effect_reconstruction_mse"].append(
                    (token_delta_flat - effect_flat)
                    .square()
                    .mean(dim=(1, 2))
                    .detach()
                    .cpu()
                    .numpy()
                )
                if "rollout_effect_pred_hold_action" in entry:
                    store["hold_effect"].append(
                        _flatten_tokens(entry["rollout_effect_pred_hold_action"])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                if "rollout_effect_pred_shuffle_action" in entry:
                    store["shuffle_effect"].append(
                        _flatten_tokens(entry["rollout_effect_pred_shuffle_action"])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                if "rollout_effect_pred_shuffle_state" in entry:
                    shuffled_state_tensor = entry["rollout_effect_pred_shuffle_state"].float()
                    store["state_real_effect"].append(
                        _flatten_tokens(entry["rollout_effect_pred"]).detach().cpu().numpy()
                    )
                    store["state_effect"].append(
                        _flatten_tokens(shuffled_state_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_residual_target"].append(
                        _flatten_tokens(residual_target).detach().cpu().numpy()
                    )
                    store["state_distance_real"].append(
                        _effect_distance_per_sample(effect_tensor, residual_target)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_distance_shuffle"].append(
                        _effect_distance_per_sample(shuffled_state_tensor, residual_target)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_effect_change"].append(
                        (effect_tensor - shuffled_state_tensor)
                        .square()
                        .mean(dim=(1, 2))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                if "milestone_step_delta_pred_shuffle_state" in entry:
                    real_step_tensor = entry["milestone_step_delta_pred"].float()
                    shuffled_step_tensor = entry[
                        "milestone_step_delta_pred_shuffle_state"
                    ].float()
                    step_target_tensor = _milestone_step_target(
                        residual_target, runtime_config
                    )
                    store["state_real_step_delta"].append(
                        _flatten_tokens(real_step_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_step_delta"].append(
                        _flatten_tokens(shuffled_step_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_step_target"].append(
                        _flatten_tokens(step_target_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_step_distance_real"].append(
                        _effect_distance_per_sample(real_step_tensor, step_target_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_step_distance_shuffle"].append(
                        _effect_distance_per_sample(shuffled_step_tensor, step_target_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    store["state_step_change"].append(
                        (real_step_tensor - shuffled_step_tensor)
                        .square()
                        .mean(dim=(1, 2))
                        .detach()
                        .cpu()
                        .numpy()
                    )
            processed_batches += 1
            processed_samples += int(target.shape[0])

    if processed_batches == 0:
        raise ValueError("inspection split produced no batches")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_np = {
        "action_norm": np.concatenate(action_norm_rows),
        "gripper_event_any": np.concatenate(gripper_any_rows),
    }
    vector_dump: dict[str, np.ndarray] = {**labels_np}
    layer_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    separation_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []

    for layer in range(layer_count):
        data = by_layer[layer]
        arrays = {
            key: np.concatenate(values) if values else None for key, values in data.items()
        }
        effect = arrays["effect"]
        delta = arrays["delta"]
        target = arrays["target"]
        neutral = arrays["neutral"]
        intervention = arrays["intervention"]
        assert effect is not None and delta is not None and target is not None
        assert neutral is not None and intervention is not None
        hold = arrays["hold_effect"] if arrays["hold_effect"] is not None else effect.copy()
        shuffled = (
            arrays["shuffle_effect"] if arrays["shuffle_effect"] is not None else effect.copy()
        )
        real_dist = _effect_distance_np(effect, target)
        hold_dist = _effect_distance_np(hold, target)
        shuffled_dist = _effect_distance_np(shuffled, target)
        pred_std = float(np.linalg.norm(effect.std(axis=0)))
        target_std = float(np.linalg.norm(target.std(axis=0)))
        pred_norm = float(np.linalg.norm(effect, axis=1).mean())
        target_norm = float(np.linalg.norm(target, axis=1).mean())
        top1, top5, rank90 = _cov_rank_stats(effect)
        layer_row = {
            "layer": layer,
            "n": int(effect.shape[0]),
            "latent_mse": float(np.mean((effect - target) ** 2)),
            "latent_cosine": _mean_row_cosine(effect, target),
            "pred_std_norm": pred_std,
            "target_std_norm": target_std,
            "std_ratio": _safe_ratio(pred_std, target_std),
            "pred_norm": pred_norm,
            "target_norm": target_norm,
            "norm_ratio": _safe_ratio(pred_norm, target_norm),
            "cov_explained_top1": top1,
            "cov_explained_top5": top5,
            "cov_rank90": rank90,
            "delta_norm": float(np.linalg.norm(delta, axis=1).mean()),
            "real_distance": float(real_dist.mean()),
            "hold_distance": float(hold_dist.mean()),
            "shuffle_distance": float(shuffled_dist.mean()),
            "d_hold": float(hold_dist.mean() - real_dist.mean()),
            "d_shuffle": float(shuffled_dist.mean() - real_dist.mean()),
            "effect_change_hold": float(np.mean((hold - effect) ** 2)),
            "effect_change_shuffle": float(np.mean((shuffled - effect) ** 2)),
        }
        layer_rows.append(layer_row)
        action_rows.append(
            {
                key: layer_row[key]
                for key in (
                    "layer",
                    "real_distance",
                    "hold_distance",
                    "shuffle_distance",
                    "d_hold",
                    "d_shuffle",
                    "effect_change_hold",
                    "effect_change_shuffle",
                )
            }
        )
        separation = {"layer": layer, **neutral_intervention_metrics(neutral, intervention, effect)}
        separation.update(
            {
                "token_paired_l2_mean": float(np.mean(arrays["token_paired_l2"])),
                "token_paired_rms": float(np.mean(arrays["token_paired_rms"])),
                "token_paired_cosine_distance": float(
                    np.mean(arrays["token_paired_cosine_distance"])
                ),
                "token_effect_reconstruction_mse": float(
                    np.mean(arrays["token_effect_reconstruction_mse"])
                ),
            }
        )
        separation_rows.append(separation)

        state_effect = arrays["state_effect"]
        if state_effect is None:
            state_rows.append(
                {
                    "layer": layer,
                    "available": False,
                    "reason": (
                        "disabled_by_cli"
                        if not args.state_shuffle
                        else "requires_batch_size_greater_than_one"
                    ),
                }
            )
        else:
            kwargs: dict[str, np.ndarray] = {}
            if (
                arrays["state_real_step_delta"] is not None
                and arrays["state_step_delta"] is not None
            ):
                kwargs = {
                    "step_delta": arrays["state_real_step_delta"],
                    "shuffled_step_delta": arrays["state_step_delta"],
                    "step_target": arrays["state_step_target"],
                }
            state_real_effect = arrays["state_real_effect"]
            state_residual_target = arrays["state_residual_target"]
            assert state_real_effect is not None and state_residual_target is not None
            pooled = state_shuffle_metrics(
                state_real_effect, state_effect, state_residual_target, **kwargs
            )
            real_distance = float(np.mean(arrays["state_distance_real"]))
            shuffled_distance = float(np.mean(arrays["state_distance_shuffle"]))
            effect_change = float(np.mean(arrays["state_effect_change"]))
            exact: dict[str, float] = {
                "real_distance": real_distance,
                "state_shuffle_distance": shuffled_distance,
                "d_state_shuffle": shuffled_distance - real_distance,
                "effect_change_state_shuffle": effect_change,
                "rollout_delta_state_shuffle": shuffled_distance - real_distance,
                "rollout_effect_change_state_shuffle": effect_change,
                "rollout_full_effect_change_state_shuffle": effect_change,
            }
            if arrays["state_step_distance_real"] is not None:
                step_real_distance = float(np.mean(arrays["state_step_distance_real"]))
                step_shuffled_distance = float(
                    np.mean(arrays["state_step_distance_shuffle"])
                )
                step_change = float(np.mean(arrays["state_step_change"]))
                exact.update(
                    {
                        "step_real_distance": step_real_distance,
                        "step_state_shuffle_distance": step_shuffled_distance,
                        "step_d_state_shuffle": step_shuffled_distance
                        - step_real_distance,
                        "step_change_state_shuffle": step_change,
                        "step_delta_state_shuffle": step_shuffled_distance
                        - step_real_distance,
                        "step_delta_change_state_shuffle": step_change,
                    }
                )
            state_rows.append(
                {
                    "layer": layer,
                    "available": True,
                    # Match the training/runtime state-shuffle diagnostic: the
                    # unified delta is measured against target minus the weak
                    # direct/base prediction. The unprefixed values match the
                    # token-level runtime metric; pooled values preserve the
                    # V39.2 inspector's compact-vector view.
                    **{
                        f"pooled_{key}": value
                        for key, value in pooled.items()
                        if key != "n"
                    },
                    "n": int(pooled["n"]),
                    **exact,
                }
            )

        idx = _sample_indices(effect.shape[0], int(args.pca_max_points), seed=4000 + layer)
        paired = np.concatenate([neutral[idx], intervention[idx]], axis=0)
        paired_coords, explained = _pca_2d(paired)
        n_points = int(idx.shape[0])
        vector_dump[f"layer{layer}_effect"] = effect.astype(np.float32)
        vector_dump[f"layer{layer}_neutral"] = neutral.astype(np.float32)
        vector_dump[f"layer{layer}_intervention"] = intervention.astype(np.float32)
        vector_dump[f"layer{layer}_pca_indices"] = idx.astype(np.int64)
        vector_dump[f"layer{layer}_neutral_pca"] = paired_coords[:n_points].astype(np.float32)
        vector_dump[f"layer{layer}_intervention_pca"] = paired_coords[n_points:].astype(np.float32)
        if state_effect is not None:
            vector_dump[f"layer{layer}_state_shuffle_effect"] = state_effect.astype(np.float32)
            vector_dump[f"layer{layer}_state_shuffle_real_effect"] = arrays[
                "state_real_effect"
            ].astype(np.float32)
            vector_dump[f"layer{layer}_state_shuffle_residual_target"] = arrays[
                "state_residual_target"
            ].astype(np.float32)
        pca_rows.append(
            {
                "layer": layer,
                "points_per_class": n_points,
                "pc1_explained": float(explained[0]) if explained.size else 0.0,
                "pc2_explained": float(explained[1]) if explained.size > 1 else 0.0,
            }
        )

    state_available = any(bool(row["available"]) for row in state_rows)
    metadata = {
        "schema": "clearvla-v40-1-latent-inspect-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_schema": payload["schema"],
        "split": args.split,
        "mode": "contract",
        "teacher_forced_flow_input": True,
        "target_action_used_as_flow_training_input": True,
        "action_metrics_are_not_policy_eval": True,
        "state_shuffle_is_experimental_non_strict": True,
        "state_shuffle_requested": bool(args.state_shuffle),
        "state_shuffle_enabled_in_checkpoint": bool(
            checkpoint_config.layer_state_counterfactual
        ),
        "state_shuffle_enabled_for_inspection": bool(runtime_config.layer_state_counterfactual),
        "state_shuffle_available": state_available,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "max_batches": int(args.max_batches),
        "layers": layer_count,
        "skipped": skipped,
        "policy_config": payload["policy_config"],
        "trainer_config": payload["trainer_config"],
    }
    _write_json(out_dir / "metadata.json", metadata)
    tables = (
        ("latent_probe_table", "clearvla-v40-1-latent-probe-table-v1", layer_rows),
        (
            "latent_counterfactual_by_layer",
            "clearvla-v40-1-action-counterfactual-v1",
            action_rows,
        ),
        ("state_shuffle_by_layer", "clearvla-v40-1-state-shuffle-v1", state_rows),
        (
            "neutral_intervention_by_layer",
            "clearvla-v40-1-neutral-intervention-v1",
            separation_rows,
        ),
        ("latent_pca_explained_by_layer", "clearvla-v40-1-latent-pca-v1", pca_rows),
    )
    for stem, schema, rows in tables:
        _write_json(out_dir / f"{stem}.json", {"schema": schema, "rows": rows})
        _save_csv(out_dir / f"{stem}.csv", rows)
    if args.save_vectors:
        np.savez_compressed(out_dir / "latent_vectors.npz", **vector_dump)

    _write_markdown_report(
        out_dir,
        metadata=metadata,
        layer_rows=layer_rows,
        state_rows=state_rows,
        separation_rows=separation_rows,
    )
    if args.make_plots:
        plt = _maybe_import_matplotlib()
        if plt is not None:
            _plot_layer_metrics(plt, layer_rows, out_dir / f"{args.plot_prefix}_layer_metrics.png")
            _plot_action_counterfactuals(
                plt, action_rows, out_dir / f"{args.plot_prefix}_action_counterfactuals.png"
            )
            _plot_state_shuffle(
                plt, state_rows, out_dir / f"{args.plot_prefix}_state_shuffle.png"
            )
            _plot_separation(
                plt,
                separation_rows,
                out_dir / f"{args.plot_prefix}_neutral_intervention.png",
            )
            for layer in range(layer_count):
                _plot_neutral_intervention_pca(
                    plt,
                    vector_dump[f"layer{layer}_neutral_pca"],
                    vector_dump[f"layer{layer}_intervention_pca"],
                    out_dir / f"{args.plot_prefix}_pca_layer{layer}_neutral_intervention.png",
                    layer=layer,
                )
            _write_html_index(
                out_dir,
                layer_count=layer_count,
                prefix=args.plot_prefix,
                has_state=state_available,
            )

    summary = {
        "schema": "clearvla-v40-1-latent-inspect-summary-v1",
        "out_dir": str(out_dir),
        "metadata": metadata,
        "best_by_latent_mse": min(layer_rows, key=lambda row: float(row["latent_mse"])),
        "best_by_action_shuffle_delta": max(
            layer_rows, key=lambda row: float(row["d_shuffle"])
        ),
        "best_by_state_shuffle_delta": (
            max(_available_rows(state_rows), key=lambda row: float(row["d_state_shuffle"]))
            if state_available
            else None
        ),
        "largest_neutral_intervention_separation": max(
            separation_rows, key=lambda row: float(row["token_paired_l2_mean"])
        ),
        "rows": layer_rows,
        "state_shuffle_rows": state_rows,
        "neutral_intervention_rows": separation_rows,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = inspect_latents(args)
    print(json.dumps(jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
