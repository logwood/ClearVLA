from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.cli.common import load_and_normalize_episodes, resolve_device
from clearvla.data.samplers import TrajectoryBlockBatchSampler, TrajectoryBlockSamplerConfig
from clearvla.experiments.residual_flow_lab.flow import ResidualBridgeConfig, sample_residual_bridge
from clearvla.experiments.residual_flow_lab.losses import ResidualFlowLossConfig, residual_flow_loss
from clearvla.experiments.residual_flow_lab.model import ResidualFlowLabModel, ResidualFlowLabModelConfig
from clearvla.experiments.residual_flow_lab.trainer import LabPhaseEpochs, ResidualFlowTrainerConfig, train_residual_flow_lab
from clearvla.experiments.vision_usage_lab.dataset import LabEventScoreConfig, LabVisualMode, VisionUsageLabDataset, compute_lab_event_scores
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
    p = argparse.ArgumentParser(description="Train history-anchored residual flow with direct action supervision")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--latent-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, default=None, help="Optional v11/v12 checkpoint providing history_source.* weights")
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")

    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--past-len", type=int, default=25)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--prior", choices=["hold", "velocity", "ema_velocity", "blend"], default="blend")
    p.add_argument("--prior-beta", type=float, default=0.5)
    p.add_argument("--velocity-mode", choices=["last", "mean", "ema"], default="ema")
    p.add_argument("--ema-decay", type=float, default=0.75)
    p.add_argument("--visual-shift", type=int, default=8)
    p.add_argument("--negative-visual-min-shift", type=int, default=8)

    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--scene-latents", type=int, default=24)
    p.add_argument("--scene-depth", type=int, default=1)
    p.add_argument("--flow-depth", type=int, default=3)
    p.add_argument("--ffn-hidden", type=int, default=1024)
    p.add_argument("--local-action-kernel", type=int, default=3)
    p.add_argument("--source-residual-scale", type=float, default=0.50)
    p.add_argument("--history-source-noise-std", type=float, default=0.01)
    p.add_argument("--camera-dropout", type=float, default=0.15)
    p.add_argument("--layerscale-init", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--delta-topk", type=int, default=16)
    p.add_argument("--disable-visual-delta-tokens", action="store_true")

    p.add_argument("--source-epochs", type=int, default=4)
    p.add_argument("--residual-flow-epochs", type=int, default=12)
    p.add_argument("--source-lr", type=float, default=1e-4)
    p.add_argument("--flow-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--event-fraction", type=float, default=0.50)
    p.add_argument("--event-quantile", type=float, default=0.70)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=100)
    p.add_argument("--integration-steps", type=int, default=4)

    p.add_argument("--clean-residual-prob", type=float, default=0.50)
    p.add_argument("--mild-residual-prob", type=float, default=0.35)
    p.add_argument("--strong-residual-prob", type=float, default=0.15)
    p.add_argument("--mild-noise-std", type=float, default=0.05)
    p.add_argument("--strong-noise-std", type=float, default=0.15)
    p.add_argument("--mild-velocity-bias-std", type=float, default=0.02)
    p.add_argument("--strong-velocity-bias-std", type=float, default=0.06)
    p.add_argument("--uniform-time-probability", type=float, default=0.30)
    p.add_argument("--source-beta-alpha", type=float, default=1.0)
    p.add_argument("--source-beta-beta", type=float, default=3.0)

    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--endpoint-weight", type=float, default=1.0)
    p.add_argument("--first-weight", type=float, default=0.0)
    p.add_argument("--first4-weight", type=float, default=0.0)
    p.add_argument("--velocity-weight", type=float, default=0.0)
    p.add_argument("--ranking-weight", type=float, default=0.0)
    p.add_argument("--ranking-margin", type=float, default=0.01)
    p.add_argument("--huber-beta", type=float, default=0.03)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _loader(dataset: VisionUsageLabDataset, *, batch_size: int, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")


def _load_source_checkpoint(model: ResidualFlowLabModel, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    source_state = {key.removeprefix("history_source."): value for key, value in state.items() if key.startswith("history_source.")}
    if not source_state:
        raise ValueError(f"checkpoint {checkpoint} does not contain history_source.* weights")
    missing, unexpected = model.history_source.load_state_dict(source_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"source checkpoint mismatch missing={missing} unexpected={unexpected}")


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cameras = tuple(str(value) for value in args.cameras)
    min_length = max(args.past_len, args.obs_horizon - 1) + args.chunk_len
    episodes, skipped, train_ids, val_ids, test_ids, normalizer = load_and_normalize_episodes(
        data_root=args.data_root,
        pattern=args.glob,
        cameras=cameras,
        action_key=args.action_key,
        camera_key_overrides={"top": args.top_key, "wrist": args.wrist_key},
        min_length=min_length,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    latent_store = VisionLatentCacheStore(args.latent_cache_dir, camera_names=cameras)
    latent_meta = latent_store.validate_consistent(episodes)
    dataset_kwargs = dict(
        episodes=episodes,
        latent_store=latent_store,
        chunk_len=args.chunk_len,
        past_len=args.past_len,
        obs_horizon=args.obs_horizon,
        future_visual_horizons=(1,),
        include_future_visual_delta=False,
        stride=args.stride,
        prior=args.prior,
        prior_beta=args.prior_beta,
        velocity_mode=args.velocity_mode,
        ema_decay=args.ema_decay,
        visual_shift=args.visual_shift,
        negative_visual_min_shift=args.negative_visual_min_shift,
    )
    train_ds = VisionUsageLabDataset(
        episode_ids=train_ids,
        visual_pool_episode_ids=train_ids,
        include_negative_visual=args.ranking_weight > 0,
        visual_mode=LabVisualMode.CORRECT,
        **dataset_kwargs,
    )
    train_scores = compute_lab_event_scores(train_ds, LabEventScoreConfig(event_quantile=args.event_quantile))
    train_ds.attach_event_scores(train_scores)
    sampler = TrajectoryBlockBatchSampler(
        train_ds.refs,
        train_scores.is_event,
        TrajectoryBlockSamplerConfig(block_size=args.batch_size, event_fraction=args.event_fraction, seed=args.seed),
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    val_visual_pool = val_ids if len(val_ids) > 1 else list(range(len(episodes)))
    val_datasets: dict[str, VisionUsageLabDataset] = {}
    for mode in LabVisualMode:
        ds = VisionUsageLabDataset(
            episode_ids=val_ids,
            visual_pool_episode_ids=val_visual_pool,
            visual_mode=mode,
            **dataset_kwargs,
        )
        ds.attach_event_scores(compute_lab_event_scores(ds, LabEventScoreConfig(event_quantile=args.event_quantile)))
        val_datasets[mode.value] = ds
    val_loaders = {mode: _loader(ds, batch_size=args.batch_size, workers=args.num_workers, device=device) for mode, ds in val_datasets.items()}

    model_config = ResidualFlowLabModelConfig(
        action_dim=int(episodes[0].actions_raw.shape[1]),
        chunk_len=args.chunk_len,
        past_len=args.past_len,
        obs_horizon=args.obs_horizon,
        camera_names=cameras,
        patch_grid=latent_meta.patch_grid,
        teacher_dim=latent_meta.token_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        scene_latents=args.scene_latents,
        scene_depth=args.scene_depth,
        flow_depth=args.flow_depth,
        ffn_hidden=args.ffn_hidden,
        local_action_kernel=args.local_action_kernel,
        include_visual_delta_tokens=not args.disable_visual_delta_tokens,
        independent_camera_dropout=args.camera_dropout,
        layerscale_init=args.layerscale_init,
        dropout=args.dropout,
        source_residual_scale=args.source_residual_scale,
        history_source_noise_std=args.history_source_noise_std,
        delta_topk=args.delta_topk,
    )
    model = ResidualFlowLabModel(model_config).to(device)
    if args.source_checkpoint is not None:
        _load_source_checkpoint(model, args.source_checkpoint)

    trainer = ResidualFlowTrainerConfig(
        phase_epochs=LabPhaseEpochs(source_pretrain=args.source_epochs, residual_flow=args.residual_flow_epochs),
        source_lr=args.source_lr,
        flow_lr=args.flow_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        rolling_window=args.rolling_window,
        integration_steps=args.integration_steps,
    )
    bridge = ResidualBridgeConfig(
        clean_probability=args.clean_residual_prob,
        mild_probability=args.mild_residual_prob,
        strong_probability=args.strong_residual_prob,
        mild_noise_std=args.mild_noise_std,
        strong_noise_std=args.strong_noise_std,
        mild_velocity_bias_std=args.mild_velocity_bias_std,
        strong_velocity_bias_std=args.strong_velocity_bias_std,
        uniform_time_probability=args.uniform_time_probability,
        source_beta_alpha=args.source_beta_alpha,
        source_beta_beta=args.source_beta_beta,
    )
    loss_config = ResidualFlowLossConfig(
        flow_weight=args.flow_weight,
        endpoint_weight=args.endpoint_weight,
        first_weight=args.first_weight,
        first4_weight=args.first4_weight,
        velocity_weight=args.velocity_weight,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        huber_beta=args.huber_beta,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    context = {
        "args": _serializable(vars(args)),
        "latent_cache": {
            "cache_version": latent_meta.cache_version,
            "patch_grid": list(latent_meta.patch_grid),
            "token_dim": latent_meta.token_dim,
            "teacher_config": latent_meta.teacher_config,
            "teacher_fingerprint": latent_meta.teacher_fingerprint,
        },
        "model": model_config.to_dict(),
        "parameters": parameters,
        "normalizer": normalizer.to_dict(),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "skipped": skipped,
        "event_windows": int(train_scores.is_event.sum()),
        "regular_windows": int((~train_scores.is_event).sum()),
    }
    print("device:", device)
    print(f"episodes loaded={len(episodes)}, skipped={len(skipped)}")
    print(f"train_eps={len(train_ids)}, val_eps={len(val_ids)}, test_eps={len(test_ids)}")
    print(f"samples train={len(train_ds)}, val={len(val_datasets['correct'])}")
    print("model:", model_config.to_dict())
    print("model_parameters:", parameters)
    print("camera_schedule:", model.camera_schedule)
    print("event_windows:", context["event_windows"], "regular_windows:", context["regular_windows"])

    if args.dry_run:
        raw = next(iter(train_loader))
        batch = {key: value.to(device) for key, value in raw.items()}
        with torch.no_grad():
            source, _ = model.predict_source(batch["past"], batch["prior"])
        residual_bridge = sample_residual_bridge(source, batch["future"], bridge)
        prepared = model.prepare_visual(batch["visual_tokens"])
        output = model.predict_residual_velocity_prepared(
            past=batch["past"], learned_source=source, prepared_visual=prepared,
            residual_state=residual_bridge.residual_state, bridge_time=residual_bridge.time,
            step_size=residual_bridge.step_size_hint, noise_level=residual_bridge.noise_level,
        )
        wrong = None
        if args.ranking_weight > 0:
            wrong_prepared = model.prepare_visual(batch["negative_visual_tokens"])
            wrong = model.predict_residual_velocity_prepared(
                past=batch["past"], learned_source=source, prepared_visual=wrong_prepared,
                residual_state=residual_bridge.residual_state, bridge_time=residual_bridge.time,
                step_size=residual_bridge.step_size_hint, noise_level=residual_bridge.noise_level,
            )
        loss = residual_flow_loss(
            correct=output, wrong=wrong, target_actions=batch["future"], target_velocity=residual_bridge.target_velocity, config=loss_config,
        )
        loss.total.backward()
        pred = model.integrate(past=batch["past"], prior=batch["prior"], visual_tokens=batch["visual_tokens"], steps=args.integration_steps)
        print("dry_run_shapes:", {
            "past": tuple(batch["past"].shape),
            "prior": tuple(batch["prior"].shape),
            "visual_tokens": tuple(batch["visual_tokens"].shape),
            "source": tuple(source.shape),
            "residual_state": tuple(residual_bridge.residual_state.shape),
            "endpoint_actions": tuple(output.endpoint_actions.shape),
            "prediction": tuple(pred.shape),
        })
        print("dry_run_loss:", loss.detached_floats())
        print("dry-run passed")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    summary = train_residual_flow_lab(
        model=model,
        train_loader=train_loader,
        val_loaders_by_mode=val_loaders,
        device=device,
        normalizer=normalizer,
        out_dir=args.out_dir,
        trainer=trainer,
        bridge=bridge,
        loss_config=loss_config,
        context=context,
    )
    print(json.dumps({"best_full_mse": summary["best_full_mse"], "summary": str(args.out_dir / "residual_flow_lab_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
