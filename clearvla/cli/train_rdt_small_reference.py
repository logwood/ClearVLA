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
    print_context,
    resolve_device,
    serializable,
)
from clearvla.experiments.classic_policy_lab.dataset import RDTSmallDatasetConfig, RDTSmallWindowDataset
from clearvla.experiments.classic_policy_lab.rdt_small_reference import (
    DebugPatchVisionEncoder,
    EmptyLanguageConditioner,
    RDTSmallReference,
    RDTSmallReferenceConfig,
    SiglipPatchVisionEncoder,
    load_policy_weights,
)
from clearvla.experiments.classic_policy_lab.trainer import RDTTrainerConfig, train_rdt_small


def _asset_empty_lang() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "rdt_empty_lang_embed.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the faithful RDT-170M / RDT-small reference on ClearVLA HDF5 episodes")
    add_data_args(p, default_resize=(384, 384))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prediction-horizon", type=int, default=64)
    p.add_argument("--image-history", type=int, default=2)
    p.add_argument("--max-cameras", type=int, default=3, help="Keep 3 for released RDT checkpoint compatibility; absent views are padded")
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--control-frequency", type=float, default=25.0)
    p.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="identity", help="Released RDT preserves physical semantics; identity is the faithful default")
    p.add_argument("--state-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 10], help="Map local 6-joint + gripper vector into RDT's unified 128-D space")

    # Released RDT-170M core. Exposed for tiny smoke tests; keep defaults for formal runs.
    p.add_argument("--unified-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--max-lang-cond-len", type=int, default=1024)
    p.add_argument("--lang-token-dim", type=int, default=4096)
    p.add_argument("--img-token-dim", type=int, default=1152)
    p.add_argument("--diffusion-train-steps", type=int, default=1000)
    p.add_argument("--inference-steps", type=int, default=5)
    p.add_argument("--sampler", choices=["dpm_solver", "ddpm_debug"], default="dpm_solver")

    # Frozen condition encoders.
    p.add_argument("--vision-encoder", choices=["siglip", "patch-debug"], default="siglip")
    p.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    p.add_argument("--siglip-local-files-only", action="store_true")
    p.add_argument("--patch-grid", type=int, default=27, help="Only for patch-debug; 27 keeps released 729-token image length")
    p.add_argument("--empty-lang-embed", type=Path, default=_asset_empty_lang())
    p.add_argument("--rdt-weights", type=Path, default=None, help="Optional released pytorch_model.bin or local RDT runner checkpoint")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")

    # Released fine-tune optimizer defaults: AdamW, constant 1e-4, weight decay 1e-2.
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
    p.add_argument("--stochastic-sampling", action="store_true", help="Only affects ddpm_debug sampling")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        if device.type != "cuda":
            raise RuntimeError("--dtype bf16 is intended for CUDA formal runs")
        return torch.bfloat16
    return torch.float32


def _vision_encoder(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    if args.vision_encoder == "siglip":
        encoder = SiglipPatchVisionEncoder(
            model_name_or_path=args.siglip_model,
            image_history=args.image_history,
            max_cameras=args.max_cameras,
            local_files_only=args.siglip_local_files_only,
        )
        if encoder.token_dim != args.img_token_dim:
            raise ValueError(f"SigLIP token dim {encoder.token_dim} != --img-token-dim {args.img_token_dim}")
        if encoder.patches_per_image != args.patch_grid * args.patch_grid:
            raise ValueError(f"SigLIP patch count {encoder.patches_per_image} != patch-grid^2 {args.patch_grid ** 2}")
        return encoder.to(device=device, dtype=dtype)
    return DebugPatchVisionEncoder(
        token_dim=args.img_token_dim,
        patch_grid=args.patch_grid,
        image_history=args.image_history,
        max_cameras=args.max_cameras,
    ).to(device)


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype, device)
    if len(args.state_indices) <= 0:
        raise ValueError("--state-indices must not be empty")
    min_length = args.prediction_horizon + args.image_history + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset))
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = load_data(
        args,
        min_length=min_length,
        normalizer_mode=args.normalizer,
    )
    robot_dim = int(episodes[0].actions_raw.shape[1])
    if int(episodes[0].states_raw.shape[1]) != robot_dim:
        raise ValueError("RDT reference currently expects state_dim == action_dim")
    if len(args.state_indices) != robot_dim:
        raise ValueError(f"--state-indices has {len(args.state_indices)} values but robot_dim={robot_dim}")
    cameras = tuple(str(value) for value in args.cameras)
    if len(cameras) > args.max_cameras:
        raise ValueError("number of real cameras exceeds --max-cameras")
    data_config = RDTSmallDatasetConfig(
        prediction_horizon=args.prediction_horizon,
        image_history=args.image_history,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        control_frequency=args.control_frequency,
    )
    train_ds = RDTSmallWindowDataset(episodes, train_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    val_ds = RDTSmallWindowDataset(episodes, val_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    model_config = RDTSmallReferenceConfig(
        unified_dim=args.unified_dim,
        prediction_horizon=args.prediction_horizon,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        max_lang_cond_len=args.max_lang_cond_len,
        lang_token_dim=args.lang_token_dim,
        img_token_dim=args.img_token_dim,
        image_history=args.image_history,
        max_cameras=args.max_cameras,
        patches_per_image=args.patch_grid * args.patch_grid,
        diffusion_train_steps=args.diffusion_train_steps,
        inference_steps=args.inference_steps,
        prediction_type="sample",
        robot_dim=robot_dim,
        state_indices=tuple(args.state_indices),
        control_frequency=args.control_frequency,
    )
    model = RDTSmallReference(model_config).to(device=device, dtype=dtype)
    if args.rdt_weights is not None:
        model.load_upstream_state_dict(load_policy_weights(args.rdt_weights), strict=True)
    vision_encoder = _vision_encoder(args, device=device, dtype=dtype)
    language_conditioner = EmptyLanguageConditioner(token_dim=args.lang_token_dim, embedding_path=args.empty_lang_embed)
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
        "schema": "clearvla-rdt-small-reference-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model_config.to_dict(),
        "trainer": serializable(vars(trainer)),
        "vision": {
            "mode": args.vision_encoder,
            "formal_siglip": args.vision_encoder == "siglip",
            "model": args.siglip_model if args.vision_encoder == "siglip" else None,
            "patch_grid": args.patch_grid,
        },
    }
    print_context({
        "model": "RDTSmallReference",
        "parameters": model.parameter_count(),
        "official_170m_shape": model.architecture_is_official_170m(),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "cameras": cameras,
        "padded_camera_slots": args.max_cameras,
        "action_normalization": action_norm.mode,
        "state_normalization": state_norm.mode,
        "vision_encoder": context["vision"],
        "rdt_weights": None if args.rdt_weights is None else str(args.rdt_weights),
    })
    if args.dry_run:
        batch = next(iter(train_loader))
        batch_size = batch["state"].shape[0]
        state = batch["state"].to(device=device, dtype=dtype)
        actions = batch["action"].to(device=device, dtype=dtype)
        images = batch["obs_image"].to(device)
        ctrl = batch["ctrl_freq"].to(device=device, dtype=dtype)
        with torch.no_grad():
            img_tokens = vision_encoder(images).to(device=device, dtype=dtype)
            lang_tokens, lang_mask = language_conditioner.batch(batch_size, device=device, dtype=dtype)
        loss = model.compute_loss(state=state, actions=actions, lang_tokens=lang_tokens, lang_mask=lang_mask, img_tokens=img_tokens, ctrl_freqs=ctrl)
        loss.backward()
        # ddpm_debug avoids a diffusers dependency during shape-only probes.
        with torch.no_grad():
            pred = model.predict_action(state=state, lang_tokens=lang_tokens, lang_mask=lang_mask, img_tokens=img_tokens, ctrl_freqs=ctrl, inference_steps=min(args.inference_steps, args.diffusion_train_steps), sampler="ddpm_debug")
        print_context({"dry_run_loss": float(loss.detach().cpu()), "img_tokens": tuple(img_tokens.shape), "prediction": tuple(pred.shape), "status": "dry-run passed"})
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(json.dumps(serializable(context), indent=2), encoding="utf-8")
    train_rdt_small(
        model=model,
        vision_encoder=vision_encoder,
        language_conditioner=language_conditioner,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=args.out_dir,
        trainer=trainer,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        context=context,
        inference_steps=args.inference_steps,
        sampler=args.sampler,
        deterministic_sampling=not args.stochastic_sampling,
        eval_seed=args.eval_seed,
    )


if __name__ == "__main__":
    main()
