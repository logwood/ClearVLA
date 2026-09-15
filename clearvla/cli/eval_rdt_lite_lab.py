from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from clearvla.cli.common import resolve_device
from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.split import split_episode_ids
from clearvla.experiments.rdt_lite_lab.codec import RDTLiteCodecs, apply_rdt_lite_codecs
from clearvla.experiments.rdt_lite_lab.dataset import (
    RDTLiteDataset,
    RDTLiteDatasetConfig,
    compute_rdt_lite_event_scores,
)
from clearvla.experiments.rdt_lite_lab.evaluation import (
    evaluate_rdt_lite_model,
    visual_dependency_report,
)
from clearvla.experiments.rdt_lite_lab.model import RDTLiteModel, RDTLiteModelConfig
from clearvla.experiments.rdt_lite_lab.schedule import (
    CosineDiffusionSchedule,
    DiffusionScheduleConfig,
)
from clearvla.experiments.vision_usage_lab.dataset import LabEventScoreConfig, LabVisualMode
from clearvla.experiments.vision_usage_lab.latent_cache import VisionLatentCacheStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate corrected lightweight RDT-style reference")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--latent-cache-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--state-key", default="qpos")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")
    p.add_argument("--sampling-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--event-quantile", type=float, default=0.70)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def _read_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    payload = _read_payload(args.checkpoint)
    model_config = RDTLiteModelConfig.from_dict(dict(payload["model_config"]))
    codecs = RDTLiteCodecs.from_dict(dict(payload["codecs"]))
    loss_data = dict(payload.get("loss_config") or payload.get("summary", {}).get("loss") or {})
    trainer_data = dict(
        payload.get("trainer_config") or payload.get("summary", {}).get("trainer") or {}
    )
    schedule_data = dict(
        payload.get("diffusion_schedule_config")
        or payload.get("summary", {}).get("diffusion_schedule")
        or {}
    )
    objective = str(
        loss_data.get("objective", payload.get("summary", {}).get("objective", "rdt_denoise"))
    )
    sampling_steps = int(
        args.sampling_steps
        or trainer_data.get("sampling_steps")
        or (5 if objective == "rdt_denoise" else 10)
    )
    schedule_config = (
        DiffusionScheduleConfig(**schedule_data) if schedule_data else DiffusionScheduleConfig()
    )
    context_args = dict(payload.get("context", {}).get("args", {}))
    data_config_data = dict(payload.get("context", {}).get("data_config", {}))
    if not data_config_data:
        data_config_data = {
            "chunk_len": model_config.chunk_len,
            "past_len": int(context_args.get("past_len", 25)),
            "state_history_len": model_config.state_history_len,
            "obs_horizon": model_config.obs_horizon,
            "stride": int(context_args.get("stride", 1)),
            "state_offset": int(context_args.get("state_offset", 0)),
            "image_offset": int(context_args.get("image_offset", 0)),
            "action_offset": int(context_args.get("action_offset", 0)),
            "prior": str(context_args.get("prior", "blend")),
            "prior_beta": float(context_args.get("prior_beta", 0.5)),
            "velocity_mode": str(context_args.get("velocity_mode", "ema")),
            "ema_decay": float(context_args.get("ema_decay", 0.75)),
            "visual_shift": int(context_args.get("visual_shift", 8)),
        }
    data_config = RDTLiteDatasetConfig(**data_config_data)
    cameras = tuple(str(value) for value in args.cameras)
    if cameras != model_config.camera_names:
        raise ValueError(
            f"checkpoint cameras={model_config.camera_names} but CLI cameras={cameras}"
        )
    min_length = (
        max(data_config.past_len, data_config.state_history_len, data_config.obs_horizon)
        + data_config.chunk_len
        + max(
            abs(data_config.state_offset),
            abs(data_config.image_offset),
            abs(data_config.action_offset),
        )
    )
    episodes, skipped = load_episodes(
        args.data_root,
        args.glob,
        cameras=cameras,
        min_length=min_length,
        action_key=args.action_key,
        state_key=args.state_key,
        camera_key_overrides={"top": args.top_key, "wrist": args.wrist_key},
    )
    train_ids, val_ids, test_ids = split_episode_ids(
        len(episodes), args.train_frac, args.val_frac, args.seed
    )
    apply_rdt_lite_codecs(episodes, codecs)
    ids = val_ids if args.split == "val" else test_ids
    visual_pool = ids if len(ids) > 1 else list(range(len(episodes)))
    latent_store = VisionLatentCacheStore(args.latent_cache_dir, camera_names=cameras)
    latent_store.validate_consistent(episodes)
    loaders: dict[str, DataLoader] = {}
    for mode in LabVisualMode:
        ds = RDTLiteDataset(
            episodes,
            ids,
            latent_store=latent_store,
            codecs=codecs,
            config=data_config,
            visual_mode=mode,
            visual_pool_episode_ids=visual_pool,
        )
        ds.attach_event_scores(
            compute_rdt_lite_event_scores(
                ds, LabEventScoreConfig(event_quantile=args.event_quantile)
            )
        )
        loaders[mode.value] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    model = RDTLiteModel(model_config)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.to(device)
    schedule = CosineDiffusionSchedule(schedule_config)
    metrics = {
        mode: evaluate_rdt_lite_model(
            model,
            loader,
            objective=objective,
            device=device,
            codecs=codecs,
            sampling_steps=sampling_steps,
            diffusion_schedule=schedule,
        )
        for mode, loader in loaders.items()
    }
    report = {
        "schema": "clearvla-rdt-lite-lab-eval-v13.1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "objective": objective,
        "sampling_steps": sampling_steps,
        "skipped": skipped,
        "metrics": metrics,
        "dependency": visual_dependency_report(metrics),
    }
    if args.summary_only:
        correct = metrics["correct"]
        print(
            json.dumps(
                {
                    "schema": report["schema"],
                    "checkpoint": report["checkpoint"],
                    "split": args.split,
                    "objective": objective,
                    "sampling_steps": sampling_steps,
                    "action_representation": codecs.action_representation,
                    "correct": {
                        key: correct[key]
                        for key in (
                            "full_mse",
                            "full_rmse",
                            "normalized_mae",
                            "first_rmse",
                            "first4_rmse",
                            "arm_first_rmse",
                            "arm_first4_rmse",
                            "gripper_first_rmse",
                            "gripper_full_rmse",
                            "pred_boundary_jump_norm",
                            "target_boundary_jump_norm",
                            "per_horizon_rmse",
                        )
                        if key in correct
                    },
                    "dependency": report["dependency"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
