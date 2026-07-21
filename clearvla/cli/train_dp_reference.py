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
from clearvla.experiments.classic_policy_lab.dataset import DPDatasetConfig, DPWindowDataset
from clearvla.experiments.classic_policy_lab.dp_reference import DPReference, DPReferenceConfig
from clearvla.experiments.classic_policy_lab.trainer import TrainerConfig, train_dp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the faithful image-conditioned U-Net Diffusion Policy reference"
    )
    add_data_args(p, default_resize=(96, 96))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prediction-horizon", type=int, default=16)
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--action-steps", type=int, default=8)
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--crop", nargs=2, type=int, default=[84, 84], metavar=("H", "W"))
    p.add_argument("--diffusion-train-steps", type=int, default=100)
    p.add_argument(
        "--inference-steps",
        type=int,
        default=16,
        help="Use 100 for the official sampling budget; 16 is a faster probe",
    )
    p.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    p.add_argument("--down-dims", nargs="+", type=int, default=[512, 1024, 2048])
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--n-groups", type=int, default=8)
    p.add_argument("--no-cond-predict-scale", action="store_true")
    p.add_argument("--no-obs-encoder-group-norm", action="store_true")
    p.add_argument("--share-rgb-model", action="store_true")
    p.add_argument("--resnet18-weights", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument(
        "--stochastic-sampling",
        action="store_true",
        help="Use DDPM posterior noise during validation sampling; deterministic sampling is easier to compare offline",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.prediction_horizon % 4 != 0:
        raise ValueError("prediction-horizon must be divisible by 4 for the official 3-stage U-Net")
    min_length = (
        args.prediction_horizon
        + args.obs_horizon
        + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset))
    )
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = (
        load_data(args, min_length=min_length, normalizer_mode="limits")
    )
    cameras = tuple(str(x) for x in args.cameras)
    data_config = DPDatasetConfig(
        prediction_horizon=args.prediction_horizon,
        obs_horizon=args.obs_horizon,
        action_steps=args.action_steps,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
    )
    train_ds = DPWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_norm,
        action_normalizer=action_norm,
        config=data_config,
    )
    val_ds = DPWindowDataset(
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
    crop_hw = None if args.crop is None else (int(args.crop[0]), int(args.crop[1]))
    model_config = DPReferenceConfig(
        state_dim=episodes[0].states_raw.shape[1],
        action_dim=episodes[0].actions_raw.shape[1],
        camera_names=cameras,
        prediction_horizon=args.prediction_horizon,
        obs_horizon=args.obs_horizon,
        action_steps=args.action_steps,
        diffusion_train_steps=args.diffusion_train_steps,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        cond_predict_scale=not args.no_cond_predict_scale,
        crop_hw=crop_hw,
        obs_encoder_group_norm=not args.no_obs_encoder_group_norm,
        share_rgb_model=args.share_rgb_model,
        resnet18_weights=args.resnet18_weights,
    )
    model = DPReference(model_config).to(device)
    trainer = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        ema_decay=args.ema_decay,
    )
    context = {
        "schema": "clearvla-dp-reference-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model_config.to_dict(),
        "trainer": serializable(vars(trainer)),
    }
    print_context(
        {
            "model": "DPReference",
            "parameters": model.parameter_count(),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "cameras": cameras,
            "action_normalization": action_norm.mode,
            "state_normalization": state_norm.mode,
            "inference_steps": args.inference_steps,
            "resnet18_weights": args.resnet18_weights,
        }
    )
    if args.dry_run:
        batch = next(iter(train_loader))
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model.compute_loss(batch["obs_image"], batch["obs_state"], batch["action"])
        loss.backward()
        pred = model.predict_action(
            batch["obs_image"][:2],
            batch["obs_state"][:2],
            inference_steps=min(args.inference_steps, 4),
        )
        print("dry_run_shapes:", {key: tuple(value.shape) for key, value in batch.items()})
        print("dry_run_loss:", float(loss.detach().cpu()))
        print("dry_run_prediction:", tuple(pred.shape))
        print("dry-run passed")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    summary = train_dp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=args.out_dir,
        trainer=trainer,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        context=context,
        inference_steps=args.inference_steps,
        use_ema=not args.no_ema,
        deterministic_sampling=not args.stochastic_sampling,
    )
    print(
        json.dumps(
            {
                "best_full_mse": summary["best_full_mse"],
                "best_arm_first_rmse": summary["best_arm_first_rmse"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
