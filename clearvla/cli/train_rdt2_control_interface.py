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
from clearvla.experiments.classic_policy_lab.rdt2_control_interface import (
    ControlInterfaceRDT2FMConfig,
    RDT2ControlInterface,
)
from clearvla.experiments.classic_policy_lab.rdt2_control_interface_runtime import (
    train_control_interface_rdt2_fm,
)
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import DinoV2TokenStore
from clearvla.experiments.classic_policy_lab.trainer import RDTTrainerConfig


PRESETS: dict[str, dict[str, int]] = {
    # Formal controlled shape: match the released-style reference motor core.
    "official": {
        "hidden_size": 1024,
        "depth": 14,
        "heads": 8,
        "kv_heads": 4,
        "register_tokens": 4,
        "multiple_of": 256,
        "interface_hidden_size": 512,
        "interface_heads": 8,
        "interface_kv_heads": 4,
        "interface_multiple_of": 128,
    },
    # Dependency-free local smoke shape only; never use for formal comparisons.
    "debug": {
        "hidden_size": 64,
        "depth": 2,
        "heads": 4,
        "kv_heads": 2,
        "register_tokens": 2,
        "multiple_of": 16,
        "interface_hidden_size": 64,
        "interface_heads": 4,
        "interface_kv_heads": 2,
        "interface_multiple_of": 16,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train v21 RDT2-FM with a control-relevant condition interface"
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

    p.add_argument("--model-size", choices=["official", "debug", "custom"], default="official")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--register-tokens", type=int, default=None)
    p.add_argument("--multiple-of", type=int, default=None)
    p.add_argument("--norm-eps", type=float, default=1e-5)
    p.add_argument("--interface-hidden-size", type=int, default=None)
    p.add_argument("--interface-heads", type=int, default=None)
    p.add_argument("--interface-kv-heads", type=int, default=None)
    p.add_argument("--interface-multiple-of", type=int, default=None)
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

    p.add_argument("--interface-mode", choices=["static", "dynamic"], default="dynamic")
    p.add_argument("--slot-tokens", type=int, default=4)
    p.add_argument("--slot-resampler-depth", type=int, default=1)
    p.add_argument("--scene-tokens", type=int, default=16)
    p.add_argument("--scene-fusion-depth", type=int, default=2)
    p.add_argument("--default-task-tokens", type=int, default=2)
    p.add_argument("--action-summary-tokens", type=int, default=4)
    p.add_argument("--action-summary-depth", type=int, default=1)
    p.add_argument("--control-tokens", type=int, default=8)
    p.add_argument("--control-readout-depth", type=int, default=2)

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
    p.add_argument("--collect-diagnostics", action="store_true")
    p.add_argument("--diagnostic-batches", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_shape(args: argparse.Namespace) -> None:
    names = (
        "hidden_size",
        "depth",
        "heads",
        "kv_heads",
        "register_tokens",
        "multiple_of",
        "interface_hidden_size",
        "interface_heads",
        "interface_kv_heads",
        "interface_multiple_of",
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
    if args.interface_hidden_size % args.interface_heads != 0:
        raise ValueError("interface-hidden-size must be divisible by interface-heads")
    if args.interface_heads % args.interface_kv_heads != 0:
        raise ValueError("interface-heads must be divisible by interface-kv-heads")


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
    require_legacy_rdt2_cli("clearvla/cli/train_rdt2_control_interface.py")
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
    dense_token_dim = int(conditioner.token_dim)
    model_config = ControlInterfaceRDT2FMConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        prediction_horizon=args.prediction_horizon,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        num_register_tokens=args.register_tokens,
        norm_eps=args.norm_eps,
        multiple_of=args.multiple_of,
        use_flash_attn=not args.no_flash_attn,
        num_inference_timesteps=args.inference_steps,
        dense_token_dim=dense_token_dim,
        visual_adaptor=args.visual_adaptor,
        camera_count=len(cameras),
        interface_hidden_size=args.interface_hidden_size,
        interface_num_heads=args.interface_heads,
        interface_num_kv_heads=args.interface_kv_heads,
        interface_multiple_of=args.interface_multiple_of,
        interface_mode=args.interface_mode,
        slot_tokens=args.slot_tokens,
        slot_resampler_depth=args.slot_resampler_depth,
        scene_tokens=args.scene_tokens,
        scene_fusion_depth=args.scene_fusion_depth,
        default_task_tokens=args.default_task_tokens,
        action_summary_tokens=args.action_summary_tokens,
        action_summary_depth=args.action_summary_depth,
        control_tokens=args.control_tokens,
        control_readout_depth=args.control_readout_depth,
    )
    model = RDT2ControlInterface(model_config, dtype=dtype).to(device=device, dtype=dtype)
    trainer = RDTTrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        eval_every=args.eval_every,
    )
    context = {
        "schema": "clearvla-rdt2-control-interface-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model.config_dict(),
        "trainer": serializable(vars(trainer)),
        "conditioning": {
            "mode": args.condition_mode,
            "dense_token_dim": dense_token_dim,
            "instruction": args.instruction,
            "dinov2_token_cache_dir": None
            if args.dinov2_token_cache_dir is None
            else str(args.dinov2_token_cache_dir),
        },
        "controlled_difference": "two-stage condition interface only: static scene-task compiler plus optional action-aware per-flow-step readout",
    }
    print_context(
        {
            "model": "RDT2ControlInterface",
            "model_size": args.model_size,
            "interface_mode": args.interface_mode,
            "parameters": model.parameter_count(),
            "parameter_groups": model.parameter_groups(),
            "motor_core_depth": args.depth,
            "motor_core_hidden_size": args.hidden_size,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "cameras": cameras,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "prediction_horizon": args.prediction_horizon,
            "condition_mode": args.condition_mode,
            "normalizer": args.normalizer,
            "zero_state": args.zero_state,
            "controlled_difference": context["controlled_difference"],
        }
    )
    if args.dry_run:
        batch = next(iter(train_loader))
        state = batch["state"].to(device=device, dtype=dtype)
        actions = batch["action"].to(device=device, dtype=dtype)
        images = batch["obs_image"].to(device=device)
        keys = torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)
        with torch.no_grad():
            condition = conditioner.encode(
                images,
                [args.instruction] * state.shape[0],
                sample_keys=keys,
                image_ablation="normal",
                camera_names=cameras,
            ).to(device=device, dtype=dtype)
        if condition.dense_tokens is None:
            raise AssertionError("dense condition missing")
        losses = model.compute_loss(
            state_tokens=state,
            action_gt=actions,
            dense_tokens=condition.dense_tokens,
            attention_mask=condition.attention_mask,
        )
        losses["loss"].backward()
        with torch.no_grad():
            pred, diagnostics = model.predict_action(
                state_tokens=state,
                dense_tokens=condition.dense_tokens,
                attention_mask=condition.attention_mask,
                inference_steps=args.inference_steps,
                return_diagnostics=True,
            )
        print_context(
            {
                "dry_run_loss": float(losses["loss"].detach().cpu()),
                "condition": tuple(condition.dense_tokens.shape),
                "condition_mask": tuple(condition.attention_mask.shape),
                "prediction": tuple(pred.shape),
                "diagnostic_groups": diagnostics["source_group_names"],
                "diagnostic_flow_steps": len(diagnostics["flow_steps"]),
                "status": "control-interface dry-run passed",
            }
        )
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(
        json.dumps(serializable(context), indent=2), encoding="utf-8"
    )
    train_control_interface_rdt2_fm(
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
        collect_diagnostics=args.collect_diagnostics,
        diagnostic_batches=args.diagnostic_batches,
    )


if __name__ == "__main__":
    main()
