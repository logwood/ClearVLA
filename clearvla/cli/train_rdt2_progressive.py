from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

import numpy as np
import torch

from clearvla.cli.train_rdt2_fm_reference import _build_conditioner, _dtype, _resolve_model_shape
from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    print_context,
    serializable,
)
from clearvla.experiments.classic_policy_lab.dataset import RDT2FMDatasetConfig, RDT2FMWindowDataset
from clearvla.experiments.classic_policy_lab.rdt2_progressive import ProgressiveRDT2FM, ProgressiveRDT2FMConfig
from clearvla.experiments.classic_policy_lab.rdt2_progressive_runtime import train_progressive_rdt2_fm
from clearvla.experiments.classic_policy_lab.trainer import RDTTrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the v19 history-anchored progressive RDT2-FM policy")
    add_data_args(p, default_resize=(224, 224))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prediction-horizon", type=int, default=24)
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--zero-state", action="store_true")
    p.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")

    p.add_argument("--model-size", choices=["small", "medium", "official", "custom"], default="medium")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--register-tokens", type=int, default=None)
    p.add_argument("--multiple-of", type=int, default=None)
    p.add_argument("--norm-eps", type=float, default=1e-5)
    p.add_argument("--inference-steps", type=int, default=5)
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--base-checkpoint", type=Path, default=None, help="Optional v18 ClearVLA checkpoint or released RDT2-FM state dict; only shape-compatible tensors transfer")

    p.add_argument("--condition-mode", choices=["none", "debug-kv", "debug-dense", "dinov2", "dinov2-cache", "rdt2-vq"], default="dinov2-cache")
    p.add_argument("--instruction", default="")
    p.add_argument("--debug-cond-tokens", type=int, default=8)
    p.add_argument("--debug-dense-token-dim", type=int, default=32)
    p.add_argument("--dense-condition-adaptor", choices=["none", "linear", "mlp2x_silu"], default="mlp2x_silu")
    p.add_argument("--dinov2-model", default="facebook/dinov2-base")
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--rdt2-vq-model", default="robotics-diffusion-transformer/RDT2-VQ")
    p.add_argument("--rdt2-vq-processor", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--rdt2-vq-local-files-only", action="store_true")
    p.add_argument("--selected-layers", nargs="+", type=int, default=None)

    p.add_argument("--history-hidden-size", type=int, default=None)
    p.add_argument("--history-layers", type=int, default=1)
    p.add_argument("--prior-residual-scale", type=float, default=1.0)
    p.add_argument("--history-noise-std", type=float, default=0.01)
    p.add_argument("--fast-exit-layer", type=int, default=None)
    p.add_argument("--prefix-exit-layer", type=int, default=None)
    p.add_argument("--prefix-length", type=int, default=4)
    p.add_argument("--visual-start-layer", type=int, default=None)
    p.add_argument("--modulation-rank", type=int, default=None)

    p.add_argument("--first-position-weight", type=float, default=8.0)
    p.add_argument("--first4-position-weight", type=float, default=4.0)
    p.add_argument("--first8-position-weight", type=float, default=2.0)
    p.add_argument("--tail-position-weight", type=float, default=1.0)
    p.add_argument("--prior-loss-weight", type=float, default=0.50)
    p.add_argument("--fast-exit-loss-weight", type=float, default=1.00)
    p.add_argument("--prefix-exit-loss-weight", type=float, default=0.50)
    p.add_argument("--full-flow-loss-weight", type=float, default=1.00)

    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--scheduler", choices=["constant", "constant_with_warmup", "cosine"], default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_progressive_shape(args: argparse.Namespace) -> None:
    _resolve_model_shape(args)
    defaults = {
        "small": {"history_hidden_size": 64, "fast_exit_layer": 2, "prefix_exit_layer": 4, "visual_start_layer": 2, "modulation_rank": 64},
        "medium": {"history_hidden_size": 128, "fast_exit_layer": 2, "prefix_exit_layer": 4, "visual_start_layer": 2, "modulation_rank": 128},
        "official": {"history_hidden_size": 256, "fast_exit_layer": 4, "prefix_exit_layer": 8, "visual_start_layer": 4, "modulation_rank": 256},
        "custom": {"history_hidden_size": 64, "fast_exit_layer": max(1, args.depth // 3), "prefix_exit_layer": max(1, 2 * args.depth // 3), "visual_start_layer": max(1, args.depth // 3), "modulation_rank": max(16, args.hidden_size // 4)},
    }[args.model_size]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/train_rdt2_progressive.py")
    args = parse_args()
    _resolve_progressive_shape(args)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from clearvla.experiments.classic_policy_lab.cli_common import resolve_device
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype, device)
    min_length = args.prediction_horizon + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset)) + 1
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = load_data(args, min_length=min_length, normalizer_mode=args.normalizer)
    action_dim = int(episodes[0].actions_raw.shape[1]); state_dim = int(episodes[0].states_raw.shape[1])
    cameras = tuple(str(value) for value in args.cameras)
    data_config = RDT2FMDatasetConfig(
        prediction_horizon=args.prediction_horizon,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        zero_state=args.zero_state,
    )
    train_ds = RDT2FMWindowDataset(episodes, train_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    val_ds = RDT2FMWindowDataset(episodes, val_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner = _build_conditioner(args, episodes=episodes, cameras=cameras, device=device, dtype=dtype, depth=args.depth, kv_heads=args.kv_heads, head_dim=args.hidden_size // args.heads)
    dense_dim = int(getattr(conditioner, "token_dim")) if hasattr(conditioner, "token_dim") else None
    adaptor = None if dense_dim is None or args.dense_condition_adaptor == "none" else args.dense_condition_adaptor
    if dense_dim is not None and adaptor is None and dense_dim != args.hidden_size:
        raise ValueError("dense condition tokens require an adaptor when token width differs from hidden size")
    config = ProgressiveRDT2FMConfig(
        action_dim=action_dim, state_dim=state_dim, prediction_horizon=args.prediction_horizon,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.heads, num_kv_heads=args.kv_heads,
        num_register_tokens=args.register_tokens, norm_eps=args.norm_eps, multiple_of=args.multiple_of,
        use_flash_attn=not args.no_flash_attn, num_inference_timesteps=args.inference_steps,
        lang_adaptor=adaptor, lang_token_dim=dense_dim,
        history_hidden_size=args.history_hidden_size, history_layers=args.history_layers,
        prior_residual_scale=args.prior_residual_scale, history_noise_std=args.history_noise_std,
        fast_exit_layer=args.fast_exit_layer, prefix_exit_layer=args.prefix_exit_layer,
        prefix_length=args.prefix_length, visual_start_layer=args.visual_start_layer,
        modulation_rank=args.modulation_rank,
        first_position_weight=args.first_position_weight, first4_position_weight=args.first4_position_weight,
        first8_position_weight=args.first8_position_weight, tail_position_weight=args.tail_position_weight,
        prior_loss_weight=args.prior_loss_weight, fast_exit_loss_weight=args.fast_exit_loss_weight,
        prefix_exit_loss_weight=args.prefix_exit_loss_weight, full_flow_loss_weight=args.full_flow_loss_weight,
    )
    model = ProgressiveRDT2FM(config, dtype=dtype).to(device=device, dtype=dtype)
    load_report = None
    if args.base_checkpoint is not None:
        load_report = model.load_compatible_reference_state_dict(args.base_checkpoint)
    trainer = RDTTrainerConfig(
        epochs=1 if args.dry_run else args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, eps=args.adam_eps, grad_clip=args.grad_clip,
        scheduler=args.scheduler, warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        log_every=1 if args.dry_run else args.log_every,
        max_train_batches=1 if args.dry_run else args.max_train_batches,
        max_val_batches=1 if args.dry_run else args.max_val_batches,
        eval_every=1 if args.dry_run else args.eval_every,
    )
    context = {
        "schema": "clearvla-rdt2-progressive-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model.config_dict(),
        "trainer": serializable(vars(trainer)),
        "conditioning": {"mode": args.condition_mode, "dense_token_dim": dense_dim, "instruction": args.instruction},
        "base_load": load_report,
        "parameters": model.parameter_count(),
        "train_windows": len(train_ds), "val_windows": len(val_ds),
    }
    print_context(context)
    train_progressive_rdt2_fm(
        model=model, conditioner=conditioner, train_loader=train_loader, val_loader=val_loader,
        device=device, out_dir=args.out_dir, trainer=trainer, action_normalizer=action_norm,
        state_normalizer=state_norm, context=context, inference_steps=args.inference_steps,
        instruction=args.instruction,
    )
    if args.dry_run:
        print("status: progressive dry-run passed", flush=True)


if __name__ == "__main__":
    main()
