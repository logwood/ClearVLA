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
from clearvla.experiments.vision_usage_lab.dataset import (
    LabEventScoreConfig,
    LabVisualMode,
    VisionUsageLabDataset,
    compute_lab_event_scores,
)
from clearvla.experiments.vision_usage_lab.flow import ActionBridgeConfig, sample_action_bridge
from clearvla.experiments.vision_usage_lab.latent_cache import VisionLatentCacheStore
from clearvla.experiments.vision_usage_lab.losses import (
    VisionUsageLabLossConfig,
    vision_usage_lab_loss,
)
from clearvla.experiments.vision_usage_lab.model import (
    AdaptiveSolverConfig,
    VisionUsageLabModel,
    VisionUsageLabModelConfig,
)
from clearvla.experiments.vision_usage_lab.trainer import (
    LAB_PHASES,
    LabPhaseEpochs,
    VisionUsageLabTrainerConfig,
    train_vision_usage_lab,
)


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the structured visual-latent vision-usage laboratory"
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--latent-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")

    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--past-len", type=int, default=25)
    p.add_argument("--recent-action-len", type=int, default=4)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--future-visual-horizons", type=int, nargs="+", default=(1, 4, 8))
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--prior", choices=["hold", "velocity", "ema_velocity", "blend"], default="blend"
    )
    p.add_argument("--prior-beta", type=float, default=0.5)
    p.add_argument("--velocity-mode", choices=["last", "mean", "ema"], default="ema")
    p.add_argument("--ema-decay", type=float, default=0.75)
    p.add_argument("--visual-shift", type=int, default=8)
    p.add_argument("--negative-visual-min-shift", type=int, default=8)

    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--scene-latents", type=int, default=48)
    p.add_argument("--scene-depth", type=int, default=3)
    p.add_argument("--fusion-depth", type=int, default=6)
    p.add_argument("--ffn-hidden", type=int, default=1536)
    p.add_argument("--dense-cross-every", type=int, default=2)
    p.add_argument("--action-history-dropout", type=float, default=0.30)
    p.add_argument("--layerscale-init", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--local-action-kernel", type=int, default=3)
    p.add_argument("--prefix-len", type=int, default=3)
    p.add_argument("--source-residual-scale", type=float, default=0.50)
    p.add_argument("--history-source-noise-std", type=float, default=0.01)
    p.add_argument("--disable-visual-delta-tokens", action="store_true")

    p.add_argument("--representation-epochs", type=int, default=4)
    p.add_argument("--action-flow-epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--representation-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--event-fraction", type=float, default=0.50)
    p.add_argument("--event-quantile", type=float, default=0.70)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=100)
    p.add_argument("--integration-steps", type=int, default=4)
    p.add_argument("--disable-adaptive-eval", action="store_true")
    p.add_argument("--adaptive-low-threshold", type=float, default=0.20)
    p.add_argument("--adaptive-high-threshold", type=float, default=0.50)
    p.add_argument("--adaptive-low-steps", type=int, default=1)
    p.add_argument("--adaptive-medium-steps", type=int, default=2)
    p.add_argument("--adaptive-high-steps", type=int, default=4)

    p.add_argument("--clean-source-prob", type=float, default=0.50)
    p.add_argument("--mild-source-prob", type=float, default=0.35)
    p.add_argument("--strong-source-prob", type=float, default=0.15)
    p.add_argument("--mild-noise-std", type=float, default=0.05)
    p.add_argument("--strong-noise-std", type=float, default=0.15)
    p.add_argument("--mild-velocity-bias-std", type=float, default=0.02)
    p.add_argument("--strong-velocity-bias-std", type=float, default=0.06)

    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--endpoint-weight", type=float, default=1.0)
    p.add_argument("--first-weight", type=float, default=1.0)
    p.add_argument("--first4-weight", type=float, default=0.5)
    p.add_argument("--velocity-weight", type=float, default=0.25)
    p.add_argument("--source-weight", type=float, default=0.50)
    p.add_argument("--prefix-weight", type=float, default=0.50)
    p.add_argument("--prefix-teacher-weight", type=float, default=0.25)
    p.add_argument("--streaming-weight", type=float, default=0.50)
    p.add_argument("--streaming-teacher-forced-weight", type=float, default=0.25)
    p.add_argument("--streaming-teacher-weight", type=float, default=0.25)
    p.add_argument("--consistency-weight", type=float, default=0.10)
    p.add_argument("--dynamics-weight", type=float, default=1.0)
    p.add_argument("--dynamics-cosine-weight", type=float, default=0.25)
    p.add_argument("--ranking-weight", type=float, default=0.20)
    p.add_argument("--ranking-margin", type=float, default=0.01)
    p.add_argument("--ranking-demand-boost", type=float, default=1.0)
    p.add_argument("--event-weight", type=float, default=0.10)
    p.add_argument("--demand-weight", type=float, default=0.25)
    p.add_argument("--demand-huber-beta", type=float, default=0.10)
    p.add_argument("--huber-beta", type=float, default=0.03)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-num-threads", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _loader(
    dataset: VisionUsageLabDataset, *, batch_size: int, workers: int, device: torch.device
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
    cameras = tuple(str(x) for x in args.cameras)
    future_horizons = tuple(sorted(set(int(x) for x in args.future_visual_horizons)))
    min_length = max(args.past_len, args.obs_horizon - 1) + max(
        args.chunk_len, max(future_horizons) + 1
    )
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
        future_visual_horizons=future_horizons,
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
        include_negative_visual=True,
        visual_mode=LabVisualMode.CORRECT,
        **dataset_kwargs,
    )
    train_scores = compute_lab_event_scores(
        train_ds, LabEventScoreConfig(event_quantile=args.event_quantile)
    )
    train_ds.attach_event_scores(train_scores)
    sampler = TrajectoryBlockBatchSampler(
        train_ds.refs,
        train_scores.is_event,
        TrajectoryBlockSamplerConfig(
            block_size=args.batch_size, event_fraction=args.event_fraction, seed=args.seed
        ),
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Validation corruption sources must remain distinct even when the validation
    # split contains only one episode.  Sources are visual counterfactuals only;
    # action targets still come exclusively from val_ids.
    val_visual_pool = val_ids if len(val_ids) > 1 else list(range(len(episodes)))
    val_datasets: dict[str, VisionUsageLabDataset] = {}
    for mode in LabVisualMode:
        ds = VisionUsageLabDataset(
            episode_ids=val_ids,
            visual_pool_episode_ids=val_visual_pool,
            visual_mode=mode,
            **dataset_kwargs,
        )
        ds.attach_event_scores(
            compute_lab_event_scores(ds, LabEventScoreConfig(event_quantile=args.event_quantile))
        )
        val_datasets[mode.value] = ds
    val_loaders = {
        mode: _loader(ds, batch_size=args.batch_size, workers=args.num_workers, device=device)
        for mode, ds in val_datasets.items()
    }

    model_cfg = VisionUsageLabModelConfig(
        action_dim=int(episodes[0].actions_raw.shape[1]),
        chunk_len=args.chunk_len,
        past_len=args.past_len,
        recent_action_len=args.recent_action_len,
        obs_horizon=args.obs_horizon,
        camera_names=cameras,
        patch_grid=latent_meta.patch_grid,
        teacher_dim=latent_meta.token_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        scene_latents=args.scene_latents,
        scene_depth=args.scene_depth,
        fusion_depth=args.fusion_depth,
        ffn_hidden=args.ffn_hidden,
        dense_cross_every=args.dense_cross_every,
        future_visual_horizons=future_horizons,
        include_visual_delta_tokens=not args.disable_visual_delta_tokens,
        action_history_dropout=args.action_history_dropout,
        layerscale_init=args.layerscale_init,
        dropout=args.dropout,
        local_action_kernel=args.local_action_kernel,
        prefix_len=args.prefix_len,
        source_residual_scale=args.source_residual_scale,
        history_source_noise_std=args.history_source_noise_std,
    )
    model = VisionUsageLabModel(model_cfg).to(device)
    adaptive_solver = AdaptiveSolverConfig(
        low_threshold=args.adaptive_low_threshold,
        high_threshold=args.adaptive_high_threshold,
        low_steps=args.adaptive_low_steps,
        medium_steps=args.adaptive_medium_steps,
        high_steps=args.adaptive_high_steps,
    )
    adaptive_solver.validate()
    trainer_cfg = VisionUsageLabTrainerConfig(
        phase_epochs=LabPhaseEpochs(args.representation_epochs, args.action_flow_epochs),
        lr=args.lr,
        representation_lr=args.representation_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        rolling_window=args.rolling_window,
        integration_steps=args.integration_steps,
        adaptive_eval=not args.disable_adaptive_eval,
        adaptive_solver=adaptive_solver,
    )
    bridge_cfg = ActionBridgeConfig(
        clean_probability=args.clean_source_prob,
        mild_probability=args.mild_source_prob,
        strong_probability=args.strong_source_prob,
        mild_noise_std=args.mild_noise_std,
        strong_noise_std=args.strong_noise_std,
        mild_velocity_bias_std=args.mild_velocity_bias_std,
        strong_velocity_bias_std=args.strong_velocity_bias_std,
    )
    loss_cfg = VisionUsageLabLossConfig(
        flow_weight=args.flow_weight,
        endpoint_weight=args.endpoint_weight,
        first_weight=args.first_weight,
        first4_weight=args.first4_weight,
        velocity_weight=args.velocity_weight,
        source_weight=args.source_weight,
        prefix_weight=args.prefix_weight,
        prefix_teacher_weight=args.prefix_teacher_weight,
        streaming_weight=args.streaming_weight,
        streaming_teacher_forced_weight=args.streaming_teacher_forced_weight,
        streaming_teacher_weight=args.streaming_teacher_weight,
        consistency_weight=args.consistency_weight,
        dynamics_weight=args.dynamics_weight,
        dynamics_cosine_weight=args.dynamics_cosine_weight,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        ranking_demand_boost=args.ranking_demand_boost,
        event_weight=args.event_weight,
        demand_weight=args.demand_weight,
        demand_huber_beta=args.demand_huber_beta,
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
        "model": model_cfg.to_dict(),
        "adaptive_solver": adaptive_solver.to_dict(),
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
    print("latent_cache:", context["latent_cache"])
    print("model:", model_cfg.to_dict())
    print("model_parameters:", parameters)
    print(
        "event_windows:", context["event_windows"], "regular_windows:", context["regular_windows"]
    )

    if args.dry_run:
        raw = next(iter(train_loader))
        batch = {key: value.to(device) for key, value in raw.items()}
        aux = model(
            past=batch["past"],
            prior=batch["prior"],
            visual_tokens=batch["visual_tokens"],
            future_actions=batch["future"],
            compute_action=False,
            compute_auxiliary=True,
        )
        learned_source, learned_source_tokens = model.predict_source(batch["past"], batch["prior"])
        bridge = sample_action_bridge(learned_source.detach(), batch["future"], bridge_cfg)
        output = model(
            past=batch["past"],
            prior=batch["prior"],
            visual_tokens=batch["visual_tokens"],
            action_state=bridge.state,
            bridge_time=bridge.time,
            noise_level=bridge.noise_level,
            future_actions=batch["future"],
            source_trajectory=learned_source,
            compute_action=True,
            compute_auxiliary=True,
        )
        wrong = model(
            past=batch["past"],
            prior=batch["prior"],
            visual_tokens=batch["negative_visual_tokens"],
            action_state=bridge.state,
            bridge_time=bridge.time,
            noise_level=bridge.noise_level,
            source_trajectory=learned_source,
            compute_action=True,
            compute_auxiliary=False,
        )
        result = vision_usage_lab_loss(
            correct=output,
            wrong=wrong,
            target_actions=batch["future"],
            target_velocity=bridge.target_velocity,
            target_visual_delta_tokens=batch["future_visual_delta_tokens"],
            event_flag=batch["event_flag"],
            demand_target=batch["demand_target"],
            config=loss_cfg,
            phase="action_flow",
        )
        result.total.backward()
        adaptive = model.integrate_adaptive(
            past=batch["past"],
            prior=batch["prior"],
            visual_tokens=batch["visual_tokens"],
            solver=adaptive_solver,
        )
        if (
            aux.visual_delta_tokens is None
            or aux.demand_score is None
            or output.velocity is None
            or output.visual_delta_tokens is None
            or output.demand_score is None
        ):
            raise RuntimeError("dry-run model output is incomplete")
        print(
            "dry_run_shapes:",
            {
                "past": tuple(batch["past"].shape),
                "prior": tuple(batch["prior"].shape),
                "visual_tokens": tuple(batch["visual_tokens"].shape),
                "negative_visual_tokens": tuple(batch["negative_visual_tokens"].shape),
                "future_visual_delta_tokens": tuple(batch["future_visual_delta_tokens"].shape),
                "demand_target": tuple(batch["demand_target"].shape),
                "learned_source": tuple(output.learned_source.shape),
                "fast_prefix": tuple(output.fast_prefix.shape),
                "streaming_actions": None
                if output.streaming_actions is None
                else tuple(output.streaming_actions.shape),
                "velocity": tuple(output.velocity.shape),
                "visual_delta_tokens": tuple(output.visual_delta_tokens.shape),
                "demand_score": tuple(output.demand_score.shape),
                "adaptive_prediction": tuple(adaptive.prediction.shape),
                "adaptive_solver_steps": tuple(adaptive.solver_steps.shape),
            },
        )
        print("dry_run_loss:", result.detached_floats())
        print("dry-run passed")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    summary = train_vision_usage_lab(
        model=model,
        train_loaders_by_phase={phase: train_loader for phase in LAB_PHASES},
        val_loaders_by_mode=val_loaders,
        device=device,
        normalizer=normalizer,
        out_dir=args.out_dir,
        trainer=trainer_cfg,
        bridge=bridge_cfg,
        loss_config=loss_cfg,
        context=context,
    )
    print(
        json.dumps(
            {
                "best_selection_full_mse": summary["best_selection_full_mse"],
                "best_selection_mode": summary["best_selection_mode"],
                "summary": str(args.out_dir / "vision_usage_lab_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
