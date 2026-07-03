from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Sequence

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
from clearvla.experiments.dynamic_world_lab.conditioning import (
    build_dense_conditioner,
    infer_dense_geometry,
)
from clearvla.experiments.dynamic_world_lab.dataset import (
    DynamicWorldDatasetConfig,
    DynamicWorldWindowDataset,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.model import (
    DynamicPredictiveWorld,
    DynamicPredictiveWorldConfig,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.representation import (
    DynamicRepresentationLossConfig,
    DynamicRepresentationTrainerConfig,
    train_dynamic_representation,
)


def _parse_offsets(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the V33.4 action-independent temporal dynamics representation. "
            "This stage constructs no action predictor and no policy."
        )
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    parser.add_argument("--future-offsets", nargs="+", type=int, default=[8, 24, 48])
    parser.add_argument("--target-history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--action-offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--control-hz", type=float, default=30.0, help="Reporting only")

    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--debug-token-dim", type=int, default=64)
    parser.add_argument("--debug-patches-per-camera", type=int, default=16)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--dynamic-tokens", type=int, default=16)
    parser.add_argument("--descriptor-projection-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--gripper-open-value", type=float, default=0.0)
    parser.add_argument("--gripper-close-value", type=float, default=1.7459820890426636)

    parser.add_argument("--descriptor-weight", type=float, default=1.0)
    parser.add_argument("--local-motion-weight", type=float, default=0.50)
    parser.add_argument("--context-state-weight", type=float, default=0.25)
    parser.add_argument("--temporal-increment-weight", type=float, default=0.50)
    parser.add_argument("--variance-weight", type=float, default=0.02)
    parser.add_argument("--covariance-weight", type=float, default=0.005)
    parser.add_argument("--token-diversity-weight", type=float, default=0.01)
    parser.add_argument("--embedding-std-target", type=float, default=0.05)
    parser.add_argument("--gripper-transition-boost", type=float, default=3.0)
    parser.add_argument("--gripper-transition-threshold", type=float, default=0.10)

    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype)
    cameras = tuple(str(name) for name in args.cameras)
    history_offsets = _parse_offsets(args.history_offsets)
    future_offsets = _parse_offsets(args.future_offsets)
    target_history_offsets = _parse_offsets(args.target_history_offsets)
    dataset_config = DynamicWorldDatasetConfig(
        action_horizon=args.action_horizon,
        history_offsets=history_offsets,
        future_offsets=future_offsets,
        target_history_offsets=target_history_offsets,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        return_images=args.condition_mode != "dinov2-cache",
    )
    dataset_config.validate()
    max_extent = max(
        args.action_horizon,
        max(future_offsets) - min(target_history_offsets),
    )
    min_length = max_extent + abs(min(history_offsets)) + 2
    (
        episodes,
        train_ids,
        val_ids,
        test_ids,
        action_normalizer,
        state_normalizer,
        image_store,
        skipped,
    ) = load_data(args, min_length=min_length, normalizer_mode=args.normalizer)
    train_dataset = DynamicWorldWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
    val_dataset = DynamicWorldWindowDataset(
        episodes,
        val_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
    test_dataset = DynamicWorldWindowDataset(
        episodes,
        test_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=args.debug_token_dim,
        debug_patches_per_camera=args.debug_patches_per_camera,
        device=device,
        dtype=dtype,
    )
    if patches is None:
        latent_dim, patches = infer_dense_geometry(
            conditioner, train_dataset[0], camera_names=cameras
        )
    model_config = DynamicPredictiveWorldConfig(
        latent_dim=latent_dim,
        action_dim=int(action_normalizer.scale.shape[-1]),
        state_dim=int(state_normalizer.scale.shape[-1]),
        action_horizon=args.action_horizon,
        history_length=len(history_offsets),
        num_cameras=len(cameras),
        patches_per_camera=patches,
        future_offsets=future_offsets,
        hidden_size=args.hidden_size,
        encoder_depth=args.encoder_depth,
        predictor_depth=1,
        action_depth=1,
        num_heads=args.heads,
        context_tokens=args.context_tokens,
        dynamic_tokens=args.dynamic_tokens,
        descriptor_projection_dim=args.descriptor_projection_dim,
        dropout=args.dropout,
        input_mode="full",
        gripper_dim_index=args.gripper_dim_index,
    )
    model = DynamicPredictiveWorld(model_config).to(device=device, dtype=dtype)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    loss_config = DynamicRepresentationLossConfig(
        descriptor_weight=args.descriptor_weight,
        local_motion_weight=args.local_motion_weight,
        context_state_weight=args.context_state_weight,
        temporal_increment_weight=args.temporal_increment_weight,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        token_diversity_weight=args.token_diversity_weight,
        embedding_std_target=args.embedding_std_target,
        gripper_transition_boost=args.gripper_transition_boost,
        gripper_transition_threshold=args.gripper_transition_threshold,
    )
    trainer = DynamicRepresentationTrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    representation_parameters = sum(
        parameter.numel()
        for module in (
            model.online_encoder,
            model.descriptor_head,
            model.local_motion_head,
            model.context_state_head,
        )
        for parameter in module.parameters()
    )
    context = {
        "schema": "clearvla-v33.4-dynamic-representation-context-v1",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "dataset": asdict(dataset_config),
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "trainer": asdict(trainer),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "future_seconds": [float(offset / args.control_hz) for offset in future_offsets],
        "representation_parameter_count": representation_parameters,
        "action_independence_contract": (
            "representation pretraining receives visual histories and synchronized state histories only; "
            "no action trajectory, action predictor, policy, or instance-level InfoNCE is constructed"
        ),
    }
    print_context(context)
    train_dynamic_representation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        conditioner=conditioner,
        device=device,
        dtype=dtype,
        camera_names=cameras,
        out_dir=args.out_dir,
        trainer=trainer,
        loss_config=loss_config,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        context=context,
    )


if __name__ == "__main__":
    main()
