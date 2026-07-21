from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.cli.common import resolve_device
from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.samplers import (
    TrajectoryBlockBatchSampler,
    TrajectoryBlockSamplerConfig,
    TrajectorySequentialBatchSampler,
    TrajectorySequentialSamplerConfig,
    TrajectoryShuffledBlockBatchSampler,
    TrajectoryShuffledBlockSamplerConfig,
)
from clearvla.data.split import split_episode_ids
from clearvla.experiments.rdt_lite_lab.codec import apply_rdt_lite_codecs, fit_rdt_lite_codecs
from clearvla.experiments.rdt_lite_lab.dataset import (
    RDTLiteDataset,
    RDTLiteDatasetConfig,
    compute_rdt_lite_event_scores,
)
from clearvla.experiments.rdt_lite_lab.losses import RDTLiteLossConfig, compute_rdt_lite_loss
from clearvla.experiments.rdt_lite_lab.model import RDTLiteModel, RDTLiteModelConfig
from clearvla.experiments.rdt_lite_lab.schedule import (
    CosineDiffusionSchedule,
    DiffusionScheduleConfig,
)
from clearvla.experiments.rdt_lite_lab.trainer import RDTLiteTrainerConfig, train_rdt_lite_lab
from clearvla.experiments.vision_usage_lab.dataset import LabEventScoreConfig, LabVisualMode
from clearvla.experiments.vision_usage_lab.latent_cache import VisionLatentCacheStore


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train corrected lightweight RDT-style direct action references"
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--latent-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--camera-order", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--state-key", default="qpos")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")

    p.add_argument("--objective", choices=["rdt_denoise", "pi_flow"], default="rdt_denoise")
    p.add_argument(
        "--action-representation",
        choices=["absolute", "relative_to_current"],
        default="relative_to_current",
    )
    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--past-len", type=int, default=25)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--state-history-len", type=int, default=1)
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--prior", choices=["hold", "velocity", "ema_velocity", "blend"], default="blend"
    )
    p.add_argument("--prior-beta", type=float, default=0.5)
    p.add_argument("--velocity-mode", choices=["last", "mean", "ema"], default="ema")
    p.add_argument("--ema-decay", type=float, default=0.75)
    p.add_argument("--visual-shift", type=int, default=8)

    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ffn-hidden", type=int, default=384)
    p.add_argument("--img-adaptor-depth", type=int, default=2)
    p.add_argument("--state-adaptor-depth", type=int, default=3)
    p.add_argument("--action-adaptor-depth", type=int, default=3)
    p.add_argument(
        "--conditioning-mode", choices=["concat", "camera_alternate", "alternate"], default="concat"
    )
    p.add_argument("--camera-dropout", type=float, default=0.0)
    p.add_argument("--include-visual-delta-tokens", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--control-frequency-hz", type=float, default=30.0)
    p.add_argument("--decoder-output-init-std", type=float, default=1e-3)

    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--sampler",
        choices=["shuffled-block", "sequential", "event-block"],
        default="shuffled-block",
    )
    p.add_argument("--event-fraction", type=float, default=0.30)
    p.add_argument("--event-quantile", type=float, default=0.70)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=100)
    p.add_argument("--max-train-batches", type=int, default=0)

    p.add_argument("--train-diffusion-steps", type=int, default=1000)
    p.add_argument("--sampling-steps", type=int, default=0)
    p.add_argument("--pi-time-alpha", type=float, default=1.5)
    p.add_argument("--pi-time-beta", type=float, default=1.0)
    p.add_argument("--pi-endpoint-weight", type=float, default=0.0)
    p.add_argument("--first-weight", type=float, default=0.0)
    p.add_argument("--first4-weight", type=float, default=0.0)
    p.add_argument("--gripper-weight", type=float, default=1.0)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _loader(
    dataset: RDTLiteDataset, *, batch_size: int, workers: int, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cameras = tuple(str(value) for value in args.cameras)
    camera_order = tuple(str(value) for value in args.camera_order)
    sampling_steps = int(args.sampling_steps or (5 if args.objective == "rdt_denoise" else 10))
    min_length = (
        max(args.past_len, args.state_history_len, args.obs_horizon)
        + args.chunk_len
        + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset))
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
    codecs = fit_rdt_lite_codecs(
        episodes,
        train_ids,
        action_representation=args.action_representation,
        chunk_len=args.chunk_len,
        past_len=args.past_len,
        state_history_len=args.state_history_len,
        obs_horizon=args.obs_horizon,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
    )
    apply_rdt_lite_codecs(episodes, codecs)
    latent_store = VisionLatentCacheStore(args.latent_cache_dir, camera_names=cameras)
    latent_meta = latent_store.validate_consistent(episodes)
    data_config = RDTLiteDatasetConfig(
        chunk_len=args.chunk_len,
        past_len=args.past_len,
        state_history_len=args.state_history_len,
        obs_horizon=args.obs_horizon,
        stride=args.stride,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        prior=args.prior,
        prior_beta=args.prior_beta,
        velocity_mode=args.velocity_mode,
        ema_decay=args.ema_decay,
        visual_shift=args.visual_shift,
    )
    train_ds = RDTLiteDataset(
        episodes,
        train_ids,
        latent_store=latent_store,
        codecs=codecs,
        config=data_config,
        visual_mode=LabVisualMode.CORRECT,
        visual_pool_episode_ids=train_ids,
    )
    train_scores = compute_rdt_lite_event_scores(
        train_ds, LabEventScoreConfig(event_quantile=args.event_quantile)
    )
    train_ds.attach_event_scores(train_scores)
    if args.sampler == "shuffled-block":
        batch_sampler = TrajectoryShuffledBlockBatchSampler(
            train_ds.refs,
            TrajectoryShuffledBlockSamplerConfig(block_size=args.batch_size, seed=args.seed),
        )
    elif args.sampler == "sequential":
        batch_sampler = TrajectorySequentialBatchSampler(
            train_ds.refs,
            TrajectorySequentialSamplerConfig(block_size=args.batch_size, seed=args.seed),
        )
    else:
        batch_sampler = TrajectoryBlockBatchSampler(
            train_ds.refs,
            train_scores.is_event,
            TrajectoryBlockSamplerConfig(
                block_size=args.batch_size, event_fraction=args.event_fraction, seed=args.seed
            ),
        )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_visual_pool = val_ids if len(val_ids) > 1 else list(range(len(episodes)))
    val_loaders: dict[str, DataLoader] = {}
    for mode in LabVisualMode:
        ds = RDTLiteDataset(
            episodes,
            val_ids,
            latent_store=latent_store,
            codecs=codecs,
            config=data_config,
            visual_mode=mode,
            visual_pool_episode_ids=val_visual_pool,
        )
        ds.attach_event_scores(
            compute_rdt_lite_event_scores(
                ds, LabEventScoreConfig(event_quantile=args.event_quantile)
            )
        )
        val_loaders[mode.value] = _loader(
            ds, batch_size=args.batch_size, workers=args.num_workers, device=device
        )

    time_encoding = "rdt_discrete" if args.objective == "rdt_denoise" else "pi_continuous"
    model_config = RDTLiteModelConfig(
        state_dim=int(episodes[0].states_raw.shape[1]),
        action_dim=int(episodes[0].actions_raw.shape[1]),
        chunk_len=args.chunk_len,
        obs_horizon=args.obs_horizon,
        state_history_len=args.state_history_len,
        camera_names=cameras,
        camera_order=camera_order,
        patch_grid=latent_meta.patch_grid,
        teacher_dim=latent_meta.token_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        img_adaptor_depth=args.img_adaptor_depth,
        state_adaptor_depth=args.state_adaptor_depth,
        action_adaptor_depth=args.action_adaptor_depth,
        conditioning_mode=args.conditioning_mode,
        independent_camera_dropout=args.camera_dropout,
        include_visual_delta_tokens=args.include_visual_delta_tokens,
        dropout=args.dropout,
        time_encoding=time_encoding,
        control_frequency_hz=args.control_frequency_hz,
        decoder_output_init_std=args.decoder_output_init_std,
    )
    model = RDTLiteModel(model_config).to(device)
    schedule_config = DiffusionScheduleConfig(train_timesteps=args.train_diffusion_steps)
    loss_config = RDTLiteLossConfig(
        objective=args.objective,
        first_weight=args.first_weight,
        first4_weight=args.first4_weight,
        gripper_weight=args.gripper_weight,
        pi_endpoint_weight=args.pi_endpoint_weight,
        pi_time_alpha=args.pi_time_alpha,
        pi_time_beta=args.pi_time_beta,
    )
    trainer = RDTLiteTrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every=args.log_every,
        rolling_window=args.rolling_window,
        sampling_steps=sampling_steps,
        max_train_batches=args.max_train_batches,
    )
    context = {
        "schema": "clearvla-rdt-lite-context-v13.1",
        "args": _serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "latent_meta": _serializable(vars(latent_meta)),
        "data_config": _serializable(vars(data_config)),
        "codecs": codecs.to_dict(),
        "model": model_config.to_dict(),
        "loss": _serializable(vars(loss_config)),
        "trainer": _serializable(vars(trainer)),
        "diffusion_schedule": _serializable(vars(schedule_config)),
    }
    print(
        json.dumps(
            {
                "data": {
                    "episodes": len(episodes),
                    "train_windows": len(train_ds),
                    "train_batches": len(train_loader),
                    "sampler": args.sampler,
                },
                "alignment": {
                    "state_offset": args.state_offset,
                    "image_offset": args.image_offset,
                    "action_offset": args.action_offset,
                },
                "action_representation": codecs.action_representation,
                "model": {
                    "parameters": model.parameter_count(),
                    "conditioning": args.conditioning_mode,
                    "time_encoding": time_encoding,
                },
            },
            indent=2,
        )
    )
    if args.dry_run:
        raw = next(iter(train_loader))
        batch = {key: value.to(device) for key, value in raw.items()}
        result = compute_rdt_lite_loss(
            model,
            state_history=batch["state_history"],
            target_actions=batch["target_actions"],
            visual_tokens=batch["visual_tokens"],
            config=loss_config,
            diffusion_schedule=CosineDiffusionSchedule(schedule_config),
        )
        result.total.backward()
        print(
            "dry_run_shapes:",
            {key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")},
        )
        print(
            "dry_run_components:",
            {key: float(value.detach().cpu()) for key, value in result.components.items()},
        )
        print("dry-run passed")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    resume_payload = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    summary = train_rdt_lite_lab(
        model=model,
        train_loader=train_loader,
        val_loaders_by_mode=val_loaders,
        device=device,
        codecs=codecs,
        out_dir=args.out_dir,
        trainer=trainer,
        loss_config=loss_config,
        schedule_config=schedule_config,
        context=context,
        resume_payload=resume_payload,
    )
    print(
        json.dumps(
            {
                "best_full_mse": summary["best_full_mse"],
                "best_arm_first_rmse": summary["best_arm_first_rmse"],
                "summary": str(args.out_dir / "rdt_lite_lab_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
