from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    DebugKVConditioner,
    DinoV2DenseConditioner,
    NullKVConditioner,
    RDT2VQKVConditioner,
)
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import DinoV2TokenStore
from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import RDT2FMReference, RDT2FMReferenceConfig
from clearvla.experiments.classic_policy_lab.trainer import RDTTrainerConfig, train_rdt2_fm


MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "small": {"hidden_size": 256, "depth": 6, "heads": 4, "kv_heads": 2, "register_tokens": 4, "multiple_of": 256},
    "medium": {"hidden_size": 512, "depth": 8, "heads": 8, "kv_heads": 4, "register_tokens": 4, "multiple_of": 256},
    "official": {"hidden_size": 1024, "depth": 14, "heads": 8, "kv_heads": 4, "register_tokens": 4, "multiple_of": 256},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the RDT2-FM flow-matching action expert with a pluggable condition source")
    add_data_args(p, default_resize=(384, 384))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prediction-horizon", type=int, default=24)
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--zero-state", action="store_true", help="Match the released post-training path, which currently supplies zero proprioception")
    p.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore", help="Local joint-space scratch runs typically use zscore; official UMI-20 runs should load matching stats externally")

    # Released RDT2-FM core. Presets avoid partially edited experiment commands.
    p.add_argument("--model-size", choices=["small", "medium", "official", "custom"], default="official")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--register-tokens", type=int, default=None)
    p.add_argument("--norm-eps", type=float, default=1e-5)
    p.add_argument("--multiple-of", type=int, default=None)
    p.add_argument("--inference-steps", type=int, default=5)
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--rdt2-fm-weights", type=Path, default=None, help="Optional released RDT2-FM pytorch_model.bin or local compatible state_dict")
    p.add_argument("--rdt2-fm-load-mode", choices=["strict", "compatible"], default="strict", help="strict reproduces the released tensor contract; compatible is for controlled local-head or DINOv2 ablations")

    # Condition plugins.
    p.add_argument("--condition-mode", choices=["none", "debug-kv", "debug-dense", "dinov2", "dinov2-cache", "rdt2-vq"], default="debug-kv")
    p.add_argument("--instruction", default="", help="Fixed instruction for this single-task ClearVLA run")
    p.add_argument("--debug-cond-tokens", type=int, default=8)
    p.add_argument("--debug-dense-token-dim", type=int, default=32)
    p.add_argument("--dense-condition-adaptor", choices=["none", "linear", "mlp2x_silu"], default="mlp2x_silu")
    p.add_argument("--dinov2-model", default="facebook/dinov2-large")
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--rdt2-vq-model", default="robotics-diffusion-transformer/RDT2-VQ")
    p.add_argument("--rdt2-vq-processor", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--rdt2-vq-local-files-only", action="store_true")
    p.add_argument("--selected-layers", nargs="+", type=int, default=None)

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
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_model_shape(args: argparse.Namespace) -> None:
    if args.model_size == "custom":
        missing = [name for name in ("hidden_size", "depth", "heads", "kv_heads") if getattr(args, name) is None]
        if missing:
            raise ValueError(f"--model-size custom requires explicit values for {missing}")
        if args.register_tokens is None:
            args.register_tokens = 4
        if args.multiple_of is None:
            args.multiple_of = 256
    else:
        preset = MODEL_SIZE_PRESETS[args.model_size]
        for name, value in preset.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    if args.hidden_size % args.heads != 0:
        raise ValueError("hidden size must be divisible by heads")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv-heads")
    if args.selected_layers is None:
        args.selected_layers = list(range(args.depth))


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        if device.type != "cuda":
            raise RuntimeError("--dtype bf16 is intended for CUDA formal runs")
        return torch.bfloat16
    return torch.float32


def _build_conditioner(
    args: argparse.Namespace,
    *,
    episodes,
    cameras: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
    depth: int,
    kv_heads: int,
    head_dim: int,
):
    if args.condition_mode == "none":
        return NullKVConditioner(depth=depth, num_kv_heads=kv_heads, head_dim=head_dim).to(device)
    if args.condition_mode == "debug-kv":
        return DebugKVConditioner(depth=depth, num_kv_heads=kv_heads, head_dim=head_dim, tokens=args.debug_cond_tokens).to(device)
    if args.condition_mode == "debug-dense":
        return DebugDenseConditioner(token_dim=args.debug_dense_token_dim, tokens_per_camera=max(1, args.debug_cond_tokens // max(len(cameras), 1))).to(device)
    if args.condition_mode == "dinov2":
        return DinoV2DenseConditioner(args.dinov2_model, local_files_only=args.dinov2_local_files_only).to(device)
    if args.condition_mode == "dinov2-cache":
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
    conditioner = RDT2VQKVConditioner(
        args.rdt2_vq_model,
        selected_layers=args.selected_layers,
        processor_name_or_path=args.rdt2_vq_processor,
        dtype=dtype,
        local_files_only=args.rdt2_vq_local_files_only,
    )
    return conditioner.to(device)


def _sample_keys(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([batch["episode_idx"], batch["image_index"]], dim=1)


def main() -> None:
    args = parse_args()
    _resolve_model_shape(args)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype, device)
    min_length = args.prediction_horizon + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset)) + 1
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = load_data(
        args,
        min_length=min_length,
        normalizer_mode=args.normalizer,
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
    train_ds = RDT2FMWindowDataset(episodes, train_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    val_ds = RDT2FMWindowDataset(episodes, val_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)

    conditioner = _build_conditioner(
        args,
        episodes=episodes,
        cameras=cameras,
        device=device,
        dtype=dtype,
        depth=args.depth,
        kv_heads=args.kv_heads,
        head_dim=args.hidden_size // args.heads,
    )
    dense_token_dim = None
    lang_adaptor = None
    if args.condition_mode in {"debug-dense", "dinov2", "dinov2-cache"}:
        dense_token_dim = int(conditioner.token_dim)
        lang_adaptor = None if args.dense_condition_adaptor == "none" else args.dense_condition_adaptor
        if lang_adaptor is None and dense_token_dim != args.hidden_size:
            raise ValueError("dense condition width differs from hidden size; enable --dense-condition-adaptor")
    if args.condition_mode == "rdt2-vq" and len(args.selected_layers) != args.depth:
        raise ValueError("--selected-layers length must equal --depth for RDT2-VQ KV mode")

    model_config = RDT2FMReferenceConfig(
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
        lang_adaptor=lang_adaptor,
        lang_token_dim=dense_token_dim,
    )
    model = RDT2FMReference(model_config, dtype=dtype).to(device=device, dtype=dtype)
    load_report = None
    if args.rdt2_fm_weights is not None:
        if args.rdt2_fm_load_mode == "strict":
            if not model_config.upstream_compatible:
                raise ValueError("strict released checkpoint loading requires official 20-D state/action, no new dense adaptor, and the released core shape")
            model.load_upstream_state_dict(args.rdt2_fm_weights, strict=True)
            load_report = {"mode": "strict", "matched_tensors": len(model.state_dict())}
        else:
            load_report = {"mode": "compatible", **model.load_compatible_upstream_state_dict(args.rdt2_fm_weights)}

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
        "schema": "clearvla-rdt2-fm-reference-context-v2",
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
            "selected_layers": list(args.selected_layers),
            "experimental_dinov2_replacement": args.condition_mode in {"dinov2", "dinov2-cache"},
            "dinov2_token_cache_dir": None if args.dinov2_token_cache_dir is None else str(args.dinov2_token_cache_dir),
        },
        "weight_load": load_report,
    }
    print_context({
        "model": "RDT2FMReference",
        "model_size": args.model_size,
        "parameters": model.parameter_count(),
        "upstream_compatible_shape": model_config.upstream_compatible,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "cameras": cameras,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "prediction_horizon": args.prediction_horizon,
        "condition_mode": args.condition_mode,
        "rdt2_fm_weights": None if args.rdt2_fm_weights is None else str(args.rdt2_fm_weights),
        "rdt2_fm_load_mode": args.rdt2_fm_load_mode,
        "weight_load": load_report,
        "normalizer": args.normalizer,
        "zero_state": args.zero_state,
    })
    if args.dry_run:
        batch = next(iter(train_loader))
        state = batch["state"].to(device=device, dtype=dtype)
        actions = batch["action"].to(device=device, dtype=dtype)
        images = batch["obs_image"].to(device)
        with torch.no_grad():
            condition = conditioner.encode(
                images,
                [args.instruction] * state.shape[0],
                sample_keys=_sample_keys(batch),
                image_ablation="normal",
                camera_names=cameras,
            ).to(device=device, dtype=dtype)
        loss = model.compute_loss(state_tokens=state, action_gt=actions, lang_tokens=condition.dense_tokens, lang_kv_cache=condition.kv_cache, lang_attn_mask=condition.attention_mask)
        loss.backward()
        with torch.no_grad():
            pred = model.predict_action(state_tokens=state, lang_tokens=condition.dense_tokens, lang_kv_cache=condition.kv_cache, lang_attn_mask=condition.attention_mask, inference_steps=args.inference_steps)
        payload = {
            "dry_run_loss": float(loss.detach().cpu()),
            "condition": "kv_cache" if condition.kv_cache is not None else tuple(condition.dense_tokens.shape),
            "condition_mask": tuple(condition.attention_mask.shape),
            "prediction": tuple(pred.shape),
            "status": "dry-run passed",
        }
        print_context(payload)
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(json.dumps(serializable(context), indent=2), encoding="utf-8")
    train_rdt2_fm(
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
        eval_seed=args.eval_seed,
        instruction=args.instruction,
    )


if __name__ == "__main__":
    main()
