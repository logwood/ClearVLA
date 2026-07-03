from __future__ import annotations

"""Inspect and visualize V39.2 multi-layer latent heads on a dataset split.

This tool is diagnostic, not a policy evaluator.  In the default ``contract``
mode it uses the flow-matching training path on validation windows so the layer
latent heads can be compared against their future-latent targets.  The script
writes explicit metadata marking this as teacher-forced/contract inspection; do
not use its action RMSE as deployable policy performance.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.dataset import (
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
    PolicyWindowDataset,
)
from clearvla.experiments.observed_state_lab.policy_v39 import V39PolicyConfig, V39PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
    gripper_event_labels,
    prepare_v39_policy_sample,
    rollout_diagnostics,
    rollout_dynamics_loss,
    rollout_delta_loss,
)
from clearvla.experiments.observed_state_lab.world_runtime import autocast_context, jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize V39.2 layer latent heads on a split.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--pca-max-points", type=int, default=1200)
    parser.add_argument("--mode", choices=["contract"], default="contract", help="contract = flow-matching contract inspection; it is teacher-forced by design.")
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-prefix", default="latent")
    return parser.parse_args()


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """Return [B, D] by averaging all token/time axes except batch and dim."""
    if x.ndim < 2:
        raise ValueError(f"expected at least 2D tensor, got {tuple(x.shape)}")
    if x.ndim == 2:
        return x.float()
    return x.float().reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)


def _flat_all(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(x.shape[0], -1)


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a.float(), b.float()).detach().cpu())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = _flat_all(a)
    bb = _flat_all(b)
    return float(F.cosine_similarity(aa, bb, dim=-1).mean().detach().cpu())


def _effect_distance_np(pred_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    diff = pred_vec - target_vec
    mse = np.mean(diff * diff, axis=1)
    pred_norm = pred_vec / np.maximum(np.linalg.norm(pred_vec, axis=1, keepdims=True), 1e-8)
    target_norm = target_vec / np.maximum(np.linalg.norm(target_vec, axis=1, keepdims=True), 1e-8)
    cosine = 1.0 - np.sum(pred_norm * target_norm, axis=1)
    return mse + 0.10 * cosine


def _safe_ratio(a: float, b: float) -> float:
    return float(a / max(abs(b), 1e-12))


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    # Economical SVD is stable and avoids an sklearn dependency.
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    var = s * s
    denom = float(var.sum()) if float(var.sum()) > 0 else 1.0
    explained = (var[:2] / denom).astype(np.float32)
    return coords.astype(np.float32), explained


def _cov_rank_stats(x: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        return 0.0, 0.0, 0
    x = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s * s
    total = float(var.sum())
    if total <= 0:
        return 0.0, 0.0, 0
    explained = var / total
    rank90 = int(np.searchsorted(np.cumsum(explained), 0.90) + 1)
    return float(explained[0]), float(explained[: min(5, explained.shape[0])].sum()), rank90


def _sample_indices(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")


def _maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[latent-inspect] matplotlib unavailable; skipping plots: {exc}", flush=True)
        return None


def _plot_layer_metric_lines(plt, rows: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(r["layer"]) for r in rows]
    fig = plt.figure(figsize=(8.5, 5.0))
    ax = fig.add_subplot(111)
    for key in ("latent_mse", "std_ratio", "norm_ratio", "d_shuffle"):
        vals = [float(r.get(key, float("nan"))) for r in rows]
        ax.plot(layers, vals, marker="o", label=key)
    ax.set_xlabel("DiT layer")
    ax.set_title("V39.2 latent diagnostics by layer")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_counterfactual_lines(plt, rows: list[dict[str, Any]], out_path: Path) -> None:
    layers = [int(r["layer"]) for r in rows]
    fig = plt.figure(figsize=(8.5, 5.0))
    ax = fig.add_subplot(111)
    for key in ("real_distance", "hold_distance", "shuffle_distance"):
        vals = [float(r.get(key, float("nan"))) for r in rows]
        ax.plot(layers, vals, marker="o", label=key)
    ax.set_xlabel("DiT layer")
    ax.set_title("Counterfactual latent distance to target")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pca_scatter(plt, coords: np.ndarray, values: np.ndarray, title: str, out_path: Path, *, value_name: str) -> None:
    fig = plt.figure(figsize=(6.0, 5.2))
    ax = fig.add_subplot(111)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=12, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.20)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(value_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class _Geometry:
    def __init__(self, row: dict[str, Any]) -> None:
        self.history_length = int(row["history_length"])
        self.num_future = int(row["future_count"])
        self.num_cameras = int(row["num_cameras"])
        self.patches_per_camera = int(row["patches_per_camera"])
        self.latent_dim = int(row["latent_dim"])


def _build_loader_and_conditioner(args: argparse.Namespace, payload: dict[str, Any], device: torch.device, dtype: torch.dtype):
    context = payload["context"]
    cameras = tuple(str(x) for x in args.cameras)
    dataset_config = ObservedStateDatasetConfig(**context["dataset"])
    visual_geometry = context.get("visual_geometry")
    if visual_geometry is None:
        raise ValueError("checkpoint context is missing visual_geometry")
    geometry = _Geometry(visual_geometry)
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    min_length = dataset_config.world_horizon + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets)) + 2
    episodes, train_ids, val_ids, test_ids, _, _, image_store, skipped = load_data(
        args,
        min_length=min_length,
        normalizer_mode=action_norm.mode,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        splits=context["splits"],
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    effective = ObservedStateDatasetConfig(**{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"})
    dataset = PolicyWindowDataset(
        ObservedStateWindowDataset(
            episodes,
            split_ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_norm,
            action_normalizer=action_norm,
            config=effective,
        )
    )
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=geometry.latent_dim,
        debug_patches_per_camera=geometry.patches_per_camera,
        device=device,
        dtype=dtype,
    )
    if latent_dim != geometry.latent_dim or (patches is not None and patches != geometry.patches_per_camera):
        raise ValueError("conditioner geometry does not match checkpoint")
    return loader, conditioner, cameras, action_norm, state_norm, skipped


def inspect_latents(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
        raise ValueError("--checkpoint must be a V39/V40 policy checkpoint")
    loader, conditioner, cameras, action_norm, state_norm, skipped = _build_loader_and_conditioner(args, payload, device, dtype)
    del action_norm, state_norm
    policy_config = V39PolicyConfig(**payload["policy_config"])
    trainer = V39PolicyTrainerConfig(**payload["trainer_config"])
    system = V39PolicySystem(policy_config)
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32)
    system.eval()

    by_layer: dict[int, dict[str, list[np.ndarray]]] = {}
    target_vec_rows: list[np.ndarray] = []
    action_norm_rows: list[np.ndarray] = []
    gripper_any_rows: list[np.ndarray] = []
    gripper_close_rows: list[np.ndarray] = []
    layer_count = int(policy_config.depth)
    for i in range(layer_count):
        by_layer[i] = {
            "z": [], "effect": [], "delta": [], "target": [], "hold_effect": [], "shuffle_effect": [],
            "event_logits": [], "action_estimate": [],
        }

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
                target_pack = system.build_rollout_target_pack(sample["visual"], sample["target_visual"])
                out = system.flow_training_forward(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    sample["policy_action"],
                    action_state=sample["action_state"],
                    rollout_target_pack=target_pack,
                    make_counterfactuals=True,
                )
            labels = gripper_event_labels(
                target_raw=sample["policy_action_raw"],
                current_raw=sample["state_raw"],
                gripper_index=system.policy_config.gripper_index,
                threshold=trainer.gripper_event_threshold,
            )
            action_norm = sample["policy_action"].float().norm(dim=-1).mean(dim=1)
            gripper_any = (labels != 0).any(dim=1).float()
            gripper_close = (labels == 2).any(dim=1).float()
            action_norm_rows.append(action_norm.detach().cpu().numpy())
            gripper_any_rows.append(gripper_any.detach().cpu().numpy())
            gripper_close_rows.append(gripper_close.detach().cpu().numpy())
            target_vec = _flatten_tokens(out["rollout_effect_target"]).detach().cpu().numpy()
            target_vec_rows.append(target_vec)
            layers = out.get("layer_contracts")
            if not isinstance(layers, list) or not layers:
                raise ValueError("checkpoint/model did not return layer_contracts; enable V39.2 layer adapters")
            for i, entry in enumerate(layers):
                effect = _flatten_tokens(entry["rollout_effect_pred"]).detach().cpu().numpy()
                delta = _flatten_tokens(entry["rollout_delta_pred"]).detach().cpu().numpy()
                z = np.concatenate([effect, delta], axis=1)
                by_layer[i]["z"].append(z)
                by_layer[i]["effect"].append(effect)
                by_layer[i]["delta"].append(delta)
                by_layer[i]["target"].append(target_vec)
                if "rollout_effect_pred_hold_action" in entry:
                    by_layer[i]["hold_effect"].append(_flatten_tokens(entry["rollout_effect_pred_hold_action"]).detach().cpu().numpy())
                if "rollout_effect_pred_shuffle_action" in entry:
                    by_layer[i]["shuffle_effect"].append(_flatten_tokens(entry["rollout_effect_pred_shuffle_action"]).detach().cpu().numpy())
                if "event_logits" in entry:
                    by_layer[i]["event_logits"].append(entry["event_logits"].detach().float().cpu().numpy())
                if "pred_action_estimate" in entry:
                    by_layer[i]["action_estimate"].append(entry["pred_action_estimate"].detach().float().cpu().numpy())

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_np = {
        "action_norm": np.concatenate(action_norm_rows) if action_norm_rows else np.zeros((0,), dtype=np.float32),
        "gripper_event_any": np.concatenate(gripper_any_rows) if gripper_any_rows else np.zeros((0,), dtype=np.float32),
        "gripper_close_any": np.concatenate(gripper_close_rows) if gripper_close_rows else np.zeros((0,), dtype=np.float32),
    }
    target_all = np.concatenate(target_vec_rows) if target_vec_rows else np.zeros((0, int(policy_config.hidden_size)), dtype=np.float32)

    layer_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    vector_dump: dict[str, np.ndarray] = {**labels_np, "target_effect_vec": target_all}
    for i in range(layer_count):
        data = by_layer[i]
        effect = np.concatenate(data["effect"]) if data["effect"] else np.zeros_like(target_all)
        delta = np.concatenate(data["delta"]) if data["delta"] else np.zeros((target_all.shape[0], int(policy_config.hidden_size)), dtype=np.float32)
        z = np.concatenate(data["z"]) if data["z"] else np.concatenate([effect, delta], axis=1)
        target = np.concatenate(data["target"]) if data["target"] else target_all
        real_dist = _effect_distance_np(effect, target)
        hold = np.concatenate(data["hold_effect"]) if data["hold_effect"] else effect.copy()
        shuf = np.concatenate(data["shuffle_effect"]) if data["shuffle_effect"] else effect.copy()
        hold_dist = _effect_distance_np(hold, target)
        shuf_dist = _effect_distance_np(shuf, target)
        pred_std = float(np.linalg.norm(effect.std(axis=0))) if effect.size else 0.0
        target_std = float(np.linalg.norm(target.std(axis=0))) if target.size else 0.0
        pred_norm = float(np.linalg.norm(effect, axis=1).mean()) if effect.size else 0.0
        target_norm = float(np.linalg.norm(target, axis=1).mean()) if target.size else 0.0
        top1, top5, rank90 = _cov_rank_stats(effect)
        row = {
            "layer": i,
            "n": int(effect.shape[0]),
            "latent_mse": float(np.mean((effect - target) ** 2)) if effect.size else float("nan"),
            "latent_cosine": float(np.mean(np.sum(effect * target, axis=1) / (np.maximum(np.linalg.norm(effect, axis=1), 1e-8) * np.maximum(np.linalg.norm(target, axis=1), 1e-8)))) if effect.size else float("nan"),
            "pred_std_norm": pred_std,
            "target_std_norm": target_std,
            "std_ratio": _safe_ratio(pred_std, target_std),
            "pred_norm": pred_norm,
            "target_norm": target_norm,
            "norm_ratio": _safe_ratio(pred_norm, target_norm),
            "cov_explained_top1": top1,
            "cov_explained_top5": top5,
            "cov_rank90": rank90,
            "delta_norm": float(np.linalg.norm(delta, axis=1).mean()) if delta.size else 0.0,
            "real_distance": float(real_dist.mean()) if real_dist.size else float("nan"),
            "hold_distance": float(hold_dist.mean()) if hold_dist.size else float("nan"),
            "shuffle_distance": float(shuf_dist.mean()) if shuf_dist.size else float("nan"),
            "d_hold": float(hold_dist.mean() - real_dist.mean()) if real_dist.size else float("nan"),
            "d_shuffle": float(shuf_dist.mean() - real_dist.mean()) if real_dist.size else float("nan"),
            "effect_change_hold": float(np.mean((hold - effect) ** 2)) if effect.size else float("nan"),
            "effect_change_shuffle": float(np.mean((shuf - effect) ** 2)) if effect.size else float("nan"),
        }
        layer_rows.append(row)
        counter_rows.append({k: row[k] for k in ("layer", "real_distance", "hold_distance", "shuffle_distance", "d_hold", "d_shuffle", "effect_change_hold", "effect_change_shuffle")})
        idx = _sample_indices(z.shape[0], int(args.pca_max_points), seed=1000 + i)
        coords, explained = _pca_2d(z[idx])
        pca_rows.append({"layer": i, "points": int(idx.shape[0]), "pc1_explained": float(explained[0]) if explained.shape[0] > 0 else 0.0, "pc2_explained": float(explained[1]) if explained.shape[0] > 1 else 0.0})
        vector_dump[f"layer{i}_z"] = z.astype(np.float32)
        vector_dump[f"layer{i}_effect"] = effect.astype(np.float32)
        vector_dump[f"layer{i}_delta"] = delta.astype(np.float32)
        vector_dump[f"layer{i}_pca_coords"] = coords.astype(np.float32)
        vector_dump[f"layer{i}_pca_indices"] = idx.astype(np.int64)

    metadata = {
        "schema": "clearvla-v39-2-latent-inspect-v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mode": args.mode,
        "teacher_forced_flow_input": True,
        "target_action_used_as_flow_training_input": True,
        "action_metrics_are_not_policy_eval": True,
        "max_batches": int(args.max_batches),
        "layers": int(layer_count),
        "skipped": skipped,
        "policy_config": payload["policy_config"],
        "trainer_config": payload["trainer_config"],
    }
    _write_json(out_dir / "metadata.json", metadata)
    _write_json(out_dir / "latent_probe_table.json", {"schema": "clearvla-v39-2-latent-probe-table-v1", "rows": layer_rows})
    _write_json(out_dir / "latent_counterfactual_by_layer.json", {"schema": "clearvla-v39-2-latent-counterfactual-v1", "rows": counter_rows})
    _write_json(out_dir / "latent_pca_explained_by_layer.json", {"schema": "clearvla-v39-2-latent-pca-v1", "rows": pca_rows})
    _save_csv(out_dir / "latent_probe_table.csv", layer_rows)
    _save_csv(out_dir / "latent_counterfactual_by_layer.csv", counter_rows)
    _save_csv(out_dir / "latent_pca_explained_by_layer.csv", pca_rows)
    if args.save_vectors:
        np.savez_compressed(out_dir / "latent_vectors.npz", **vector_dump)

    if args.make_plots:
        plt = _maybe_import_matplotlib()
        if plt is not None:
            _plot_layer_metric_lines(plt, layer_rows, out_dir / f"{args.plot_prefix}_layer_metrics.png")
            _plot_counterfactual_lines(plt, counter_rows, out_dir / f"{args.plot_prefix}_counterfactual_distances.png")
            for i in range(layer_count):
                coords = vector_dump[f"layer{i}_pca_coords"]
                idx = vector_dump[f"layer{i}_pca_indices"]
                if coords.shape[0] == 0:
                    continue
                _plot_pca_scatter(
                    plt,
                    coords,
                    labels_np["action_norm"][idx],
                    f"Layer {i} latent PCA by action norm",
                    out_dir / f"{args.plot_prefix}_pca_layer{i}_action_norm.png",
                    value_name="Mean action norm",
                )
                _plot_pca_scatter(
                    plt,
                    coords,
                    labels_np["gripper_event_any"][idx],
                    f"Layer {i} latent PCA by gripper event",
                    out_dir / f"{args.plot_prefix}_pca_layer{i}_gripper_event.png",
                    value_name="Any gripper event",
                )
            _write_html_index(out_dir, layer_count=layer_count, prefix=args.plot_prefix)

    summary = {
        "schema": "clearvla-v39-2-latent-inspect-summary-v1",
        "out_dir": str(out_dir),
        "metadata": metadata,
        "best_by_latent_mse": min(layer_rows, key=lambda r: float(r["latent_mse"])) if layer_rows else None,
        "best_by_std_ratio_closest_to_one": min(layer_rows, key=lambda r: abs(float(r["std_ratio"]) - 1.0)) if layer_rows else None,
        "rows": layer_rows,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _write_html_index(out_dir: Path, *, layer_count: int, prefix: str) -> None:
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>V39.2 latent inspection</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1100px} img{max-width:100%;border:1px solid #ddd;margin:8px 0 24px 0} code{background:#f4f4f4;padding:2px 4px}</style>",
        "<h1>V39.2 latent inspection</h1>",
        "<p>This is a contract/teacher-forced latent inspection, not deployable policy evaluation.</p>",
        f"<h2>Layer metrics</h2><img src='{prefix}_layer_metrics.png'>",
        f"<h2>Counterfactual distances</h2><img src='{prefix}_counterfactual_distances.png'>",
        "<h2>PCA by layer</h2>",
    ]
    for i in range(layer_count):
        lines.append(f"<h3>Layer {i}</h3>")
        lines.append(f"<img src='{prefix}_pca_layer{i}_action_norm.png'>")
        lines.append(f"<img src='{prefix}_pca_layer{i}_gripper_event.png'>")
    lines.append("<h2>Data files</h2><ul>")
    for name in ("summary.json", "latent_probe_table.csv", "latent_counterfactual_by_layer.csv", "latent_pca_explained_by_layer.csv", "latent_vectors.npz"):
        lines.append(f"<li><code>{name}</code></li>")
    lines.append("</ul>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = inspect_latents(args)
    print(json.dumps(jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
