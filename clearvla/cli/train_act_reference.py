from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.act_reference import ACTReference, ACTReferenceConfig
from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    print_context,
    resolve_device,
    serializable,
)
from clearvla.experiments.classic_policy_lab.dataset import ACTDatasetConfig, ACTWindowDataset
from clearvla.experiments.classic_policy_lab.trainer import TrainerConfig, train_act


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the faithful ACT CVAE reference on ClearVLA HDF5 episodes"
    )
    add_data_args(p, default_resize=(128, 128))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--include-tail-padding", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--ffn-dim", type=int, default=3200)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--transformer-encoder-layers", type=int, default=4)
    p.add_argument("--transformer-decoder-layers", type=int, default=7)
    p.add_argument("--style-encoder-layers", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--resnet18-weights", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
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
    min_length = (
        args.chunk_len
        + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset))
        + 1
    )
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = (
        load_data(args, min_length=min_length, normalizer_mode="zscore")
    )
    cameras = tuple(str(x) for x in args.cameras)
    data_config = ACTDatasetConfig(
        chunk_len=args.chunk_len,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        include_tail_padding=args.include_tail_padding,
    )
    train_ds = ACTWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_norm,
        action_normalizer=action_norm,
        config=data_config,
    )
    val_ds = ACTWindowDataset(
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
    model_config = ACTReferenceConfig(
        state_dim=episodes[0].states_raw.shape[1],
        action_dim=episodes[0].actions_raw.shape[1],
        camera_names=cameras,
        chunk_len=args.chunk_len,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        heads=args.heads,
        transformer_encoder_layers=args.transformer_encoder_layers,
        transformer_decoder_layers=args.transformer_decoder_layers,
        style_encoder_layers=args.style_encoder_layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        kl_weight=args.kl_weight,
        resnet18_weights=args.resnet18_weights,
    )
    model = ACTReference(model_config).to(device)
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
    )
    context = {
        "schema": "clearvla-act-reference-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model_config.to_dict(),
        "trainer": serializable(vars(trainer)),
    }
    print_context(
        {
            "model": "ACTReference",
            "parameters": model.parameter_count(),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "cameras": cameras,
            "action_normalization": action_norm.mode,
            "state_normalization": state_norm.mode,
            "resnet18_weights": args.resnet18_weights,
        }
    )
    if args.dry_run:
        batch = next(iter(train_loader))
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model.compute_loss(batch["qpos"], batch["image"], batch["actions"], batch["is_pad"])
        loss["loss"].backward()
        print("dry_run_shapes:", {key: tuple(value.shape) for key, value in batch.items()})
        print("dry_run_loss:", {key: float(value.detach().cpu()) for key, value in loss.items()})
        print("dry-run passed")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_context.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    summary = train_act(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        out_dir=args.out_dir,
        trainer=trainer,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        context=context,
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
