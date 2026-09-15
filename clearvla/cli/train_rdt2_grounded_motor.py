from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    print_context,
    resolve_device,
    serializable,
)
from clearvla.experiments.classic_policy_lab.dataset import RDT2FMDatasetConfig, RDT2FMWindowDataset
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import (
    CachedDinoV2DenseConditioner,
    DebugDenseConditioner,
    DinoV2DenseConditioner,
)
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import DinoV2TokenStore
from clearvla.experiments.classic_policy_lab.rdt2_grounded_motor import (
    GroundedMotorRDT2FM,
    GroundedMotorRDT2FMConfig,
)
from clearvla.experiments.classic_policy_lab.rdt2_grounded_motor_runtime import (
    train_grounded_motor_rdt2_fm,
)
from clearvla.experiments.classic_policy_lab.trainer import RDTTrainerConfig


PRESETS: dict[str, dict[str, int]] = {
    "small": {
        "hidden_size": 256,
        "first_depth": 1,
        "tail_depth": 3,
        "heads": 8,
        "kv_heads": 4,
        "multiple_of": 128,
        "grounding_depth": 1,
        "grounding_queries": 6,
        "history_hidden_size": 128,
        "motion_tokens": 3,
    },
    "medium": {
        "hidden_size": 512,
        "first_depth": 2,
        "tail_depth": 4,
        "heads": 8,
        "kv_heads": 4,
        "multiple_of": 256,
        "grounding_depth": 2,
        "grounding_queries": 8,
        "history_hidden_size": 192,
        "motion_tokens": 4,
    },
    "wide": {
        "hidden_size": 1024,
        "first_depth": 2,
        "tail_depth": 4,
        "heads": 8,
        "kv_heads": 4,
        "multiple_of": 256,
        "grounding_depth": 2,
        "grounding_queries": 8,
        "history_hidden_size": 256,
        "motion_tokens": 4,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the v20 one-time-grounded shallow motor RDT2-FM policy"
    )
    add_data_args(p, default_resize=(224, 224))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prediction-horizon", type=int, default=24)
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--zero-state", action="store_true")
    p.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")

    p.add_argument("--model-size", choices=["small", "medium", "wide", "custom"], default="medium")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--first-depth", type=int, default=None)
    p.add_argument("--tail-depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--multiple-of", type=int, default=None)
    p.add_argument("--norm-eps", type=float, default=1e-5)
    p.add_argument("--inference-steps", type=int, default=5)
    p.add_argument("--no-flash-attn", action="store_true")

    p.add_argument(
        "--condition-mode",
        choices=["debug-dense", "dinov2", "dinov2-cache"],
        default="dinov2-cache",
    )
    p.add_argument("--instruction", default="")
    p.add_argument("--debug-cond-tokens", type=int, default=8)
    p.add_argument("--debug-dense-token-dim", type=int, default=32)
    p.add_argument("--dinov2-model", default="facebook/dinov2-base")
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--visual-adaptor", choices=["linear", "mlp2x_silu"], default="linear")

    p.add_argument("--grounding-depth", type=int, default=None)
    p.add_argument("--grounding-queries", type=int, default=None)
    p.add_argument("--default-task-tokens", type=int, default=2)
    p.add_argument("--history-hidden-size", type=int, default=None)
    p.add_argument("--history-layers", type=int, default=1)
    p.add_argument("--motion-tokens", type=int, default=None)
    p.add_argument("--history-noise-std", type=float, default=0.01)
    p.add_argument("--full-flow-loss-weight", type=float, default=1.0)
    p.add_argument("--first-flow-loss-weight", type=float, default=1.0)
    p.add_argument("--full-first-position-weight", type=float, default=1.0)
    p.add_argument("--no-detach-first-anchor", action="store_true")

    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--scheduler", choices=["constant", "constant_with_warmup", "cosine"], default="constant"
    )
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_shape(args: argparse.Namespace) -> None:
    names = (
        "hidden_size",
        "first_depth",
        "tail_depth",
        "heads",
        "kv_heads",
        "multiple_of",
        "grounding_depth",
        "grounding_queries",
        "history_hidden_size",
        "motion_tokens",
    )
    if args.model_size == "custom":
        missing = [name for name in names if getattr(args, name) is None]
        if missing:
            raise ValueError(f"--model-size custom requires explicit values for {missing}")
    else:
        for name, value in PRESETS[args.model_size].items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    if args.hidden_size % args.heads != 0:
        raise ValueError("hidden-size must be divisible by heads")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv-heads")


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        if device.type != "cuda":
            raise RuntimeError("--dtype bf16 is intended for CUDA formal runs")
        return torch.bfloat16
    return torch.float32


def _build_dense_conditioner(
    args: argparse.Namespace, *, episodes, cameras: tuple[str, ...], device: torch.device
):
    if args.condition_mode == "debug-dense":
        return DebugDenseConditioner(
            token_dim=args.debug_dense_token_dim,
            tokens_per_camera=max(1, args.debug_cond_tokens // max(len(cameras), 1)),
        ).to(device)
    if args.condition_mode == "dinov2":
        return DinoV2DenseConditioner(
            args.dinov2_model, local_files_only=args.dinov2_local_files_only
        ).to(device)
    if args.dinov2_token_cache_dir is None:
        raise ValueError("--condition-mode dinov2-cache requires --dinov2-token-cache-dir")
    store = DinoV2TokenStore(
        args.dinov2_token_cache_dir,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
    )
    return CachedDinoV2DenseConditioner(store).to(device)


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/train_rdt2_grounded_motor.py")
    args = parse_args()
    _resolve_shape(args)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype, device)
    min_length = (
        args.prediction_horizon
        + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset))
        + 1
    )
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = (
        load_data(args, min_length=min_length, normalizer_mode=args.normalizer)
    )
    action_dim = int(episodes[0].actions_raw.shape[1])
    state_dim = int(episodes[0].states_raw.shape[1])
    cameras = tuple(str(value) for value in args.cameras)
    data_config = RDT2FMDatasetConfig(
        prediction_horizon=args.prediction_horizon,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        zero_state=args.zero_state,
    )
    train_ds = RDT2FMWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_norm,
        action_normalizer=action_norm,
        config=data_config,
    )
    val_ds = RDT2FMWindowDataset(
        episodes,
        val_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_norm,
        action_normalizer=action_norm,
        config=data_config,
    )
    train_loader = make_loader(
        train_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device
    )
    val_loader = make_loader(
        val_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device
    )
    conditioner = _build_dense_conditioner(args, episodes=episodes, cameras=cameras, device=device)
    dense_dim = int(conditioner.token_dim)
    model_config = GroundedMotorRDT2FMConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        prediction_horizon=args.prediction_horizon,
        hidden_size=args.hidden_size,
        first_depth=args.first_depth,
        tail_depth=args.tail_depth,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        norm_eps=args.norm_eps,
        multiple_of=args.multiple_of,
        use_flash_attn=not args.no_flash_attn,
        num_inference_timesteps=args.inference_steps,
        dense_token_dim=dense_dim,
        visual_adaptor=args.visual_adaptor,
        grounding_depth=args.grounding_depth,
        grounding_queries=args.grounding_queries,
        default_task_tokens=args.default_task_tokens,
        history_hidden_size=args.history_hidden_size,
        history_layers=args.history_layers,
        motion_tokens=args.motion_tokens,
        history_noise_std=args.history_noise_std,
        full_flow_loss_weight=args.full_flow_loss_weight,
        first_flow_loss_weight=args.first_flow_loss_weight,
        full_first_position_weight=args.full_first_position_weight,
        detach_first_anchor=not args.no_detach_first_anchor,
    )
    model = GroundedMotorRDT2FM(model_config, dtype=dtype).to(device=device, dtype=dtype)
    trainer = RDTTrainerConfig(
        epochs=1 if args.dry_run else args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every=1 if args.dry_run else args.log_every,
        max_train_batches=1 if args.dry_run else args.max_train_batches,
        max_val_batches=1 if args.dry_run else args.max_val_batches,
        eval_every=1 if args.dry_run else args.eval_every,
    )
    context = {
        "schema": "clearvla-rdt2-grounded-motor-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model.config_dict(),
        "trainer": serializable(vars(trainer)),
        "conditioning": {
            "mode": args.condition_mode,
            "dense_token_dim": dense_dim,
            "instruction": args.instruction,
            "dinov2_token_cache_dir": None
            if args.dinov2_token_cache_dir is None
            else str(args.dinov2_token_cache_dir),
        },
        "parameters": model.parameter_count(),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
    }
    print_context(
        {
            "model": "GroundedMotorRDT2FM",
            "model_size": args.model_size,
            "parameters": model.parameter_count(),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "cameras": cameras,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "prediction_horizon": args.prediction_horizon,
            "condition_mode": args.condition_mode,
            "dense_token_dim": dense_dim,
            "first_depth": args.first_depth,
            "tail_depth": args.tail_depth,
            "grounding_depth": args.grounding_depth,
            "grounding_queries": args.grounding_queries,
            "controlled_difference": "one-time semantic visual grounding + explicit motion history + native first-action token stage + shallow tail motor stage",
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(
        json.dumps(serializable(context), indent=2), encoding="utf-8"
    )
    train_grounded_motor_rdt2_fm(
        model=model,
        conditioner=conditioner,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=args.out_dir,
        trainer=trainer,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        context=context,
        inference_steps=args.inference_steps,
        instruction=args.instruction,
        eval_seed=args.eval_seed,
    )
    if args.dry_run:
        print("status: grounded-motor dry-run passed", flush=True)


if __name__ == "__main__":
    main()
