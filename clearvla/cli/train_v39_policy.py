from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    print_context,
    resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import (
    build_dense_conditioner,
    infer_dense_geometry,
)
from clearvla.experiments.observed_state_lab.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
    PolicyWindowDataset,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
    train_v39_policy,
)
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.system import V39PolicySystem


def _parse_offsets(text: str) -> tuple[int, ...]:
    values = tuple(int(x) for x in str(text).replace(",", " ").split())
    if not values:
        raise argparse.ArgumentTypeError("offset list must be non-empty")
    return values


def _legacy_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clearvla-v35-world-checkpoint-v1":
        raise ValueError("--legacy-context-checkpoint must be a V35 world checkpoint")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V39 staged mid-cut latent-contract temporal policy."
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument(
        "--legacy-context-checkpoint",
        type=Path,
        default=None,
        help="Optional migration source for splits/normalizers only; not a model dependency.",
    )
    parser.add_argument("--normalizer", choices=["zscore", "limits", "identity"], default="zscore")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Load a V39 contract-stage checkpoint as model initialization before policy-stage finetuning.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--prefetch-dinov2-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For dinov2-cache mode, load current and compact future DINO tokens in DataLoader workers instead of the main training loop.",
    )
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")

    parser.add_argument("--world-horizon", type=int, default=48)
    parser.add_argument("--policy-horizon", type=int, default=24)
    parser.add_argument("--segment-length", type=int, default=4)
    parser.add_argument("--history-offsets", type=_parse_offsets, default=(-8, -4, 0))
    parser.add_argument("--executed-action-offsets", type=_parse_offsets, default=(-8, -4, -1))
    parser.add_argument("--target-history-offsets", type=_parse_offsets, default=(-8, -4, 0))
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6, help="shared temporal canvas depth")
    parser.add_argument(
        "--midcut-layer",
        type=int,
        default=3,
        help="Expose Z_mid after this many DiT blocks for the simple latent-contract heads.",
    )
    parser.add_argument("--midcut-future-gain-init", type=float, default=0.10)
    parser.add_argument(
        "--layer-contract-adapters",
        type=int,
        default=0,
        help="V39.1: enable tiny contract adapters after every DiT block.",
    )
    parser.add_argument("--layer-contract-adapter-dim", type=int, default=128)
    parser.add_argument("--layer-contract-grad-scale", type=float, default=1.0)
    parser.add_argument("--layer-contract-residual-scale", type=float, default=0.50)
    parser.add_argument(
        "--layer-shared-fm-probe",
        type=int,
        default=0,
        help="V39.2: use one shared flow-matching action probe after each layer latent head.",
    )
    parser.add_argument("--layer-fm-probe-hidden", type=int, default=256)
    parser.add_argument(
        "--layer-recurrent-consequence",
        type=int,
        default=0,
        help="V39.3: roll each layer latent through K action-conditioned consequence milestones.",
    )
    parser.add_argument("--layer-consequence-steps", type=int, default=6)
    parser.add_argument("--layer-consequence-hidden", type=int, default=256)
    parser.add_argument("--layer-consequence-delta-scale", type=float, default=1.0)
    parser.add_argument("--layer-consequence-initial-gain", type=float, default=0.10)
    parser.add_argument(
        "--layer-causal-feedback-depth",
        type=int,
        default=0,
        help="V40.1: optional inner interaction blocks; default 0 because the unified intervention head already encodes state-action jointly.",
    )
    parser.add_argument(
        "--layer-causal-memory-tokens",
        type=int,
        default=4,
        help="V40.1: compact memory/context tokens for unified intervention latent.",
    )
    parser.add_argument(
        "--layer-low-causal-weight",
        type=float,
        default=1.0,
        help="V40: causal gain at the shallowest layer.",
    )
    parser.add_argument(
        "--layer-high-causal-weight",
        type=float,
        default=1.0,
        help="V40.1: retained as diagnostic; unified intervention head is not mixed with a separate latent head.",
    )
    parser.add_argument(
        "--layer-low-latent-weight",
        type=float,
        default=1.0,
        help="V40.1: retained as diagnostic; unified intervention head is not mixed with a separate latent head.",
    )
    parser.add_argument(
        "--layer-high-latent-weight",
        type=float,
        default=1.0,
        help="V40: latent gain at the deepest layer.",
    )
    parser.add_argument(
        "--layer-causal-event-from-effect",
        type=int,
        default=1,
        help="V40: read event logits from causal effect tokens in layer contracts.",
    )
    parser.add_argument(
        "--layer-state-counterfactual",
        type=int,
        default=0,
        help="Experimental non-strict state/frame shuffle diagnostic; disabled by default.",
    )
    parser.add_argument("--proposal-depth", type=int, default=2)
    parser.add_argument("--proposal-dropout", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--event-tokens", type=int, default=3)
    parser.add_argument("--canvas-registers", type=int, default=12)
    parser.add_argument("--future-anchors", type=int, default=4)
    parser.add_argument("--future-grid-size", type=int, default=4)
    parser.add_argument("--action-basis-tokens", type=int, default=4)
    parser.add_argument("--rollout-tail-start-step", type=int, default=8)
    parser.add_argument("--rollout-tail-full-step", type=int, default=13)
    parser.add_argument("--controlled-delta-rank", type=int, default=8)
    parser.add_argument("--base-effect-hidden", type=int, default=128)
    parser.add_argument("--latent-action-tokens", type=int, default=8)
    parser.add_argument("--neutral-action-tokens", type=int, default=4)
    parser.add_argument("--controlled-delta-dropout", type=float, default=0.0)
    parser.add_argument("--role-dropout", type=float, default=0.10)
    parser.add_argument("--visual-memory-dropout", type=float, default=0.0)
    parser.add_argument("--canvas-dropout", type=float, default=0.0)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--first-execution-steps", type=int, default=4)
    parser.add_argument("--mid-execution-steps", type=int, default=8)
    parser.add_argument("--physical-decode-delta-blend", type=float, default=0.25)

    defaults = V39PolicyTrainerConfig()
    for field in V39PolicyTrainerConfig.__dataclass_fields__:
        value = getattr(defaults, field)
        parser.add_argument(
            "--" + field.replace("_", "-"), dest=field, type=type(value), default=value
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)

    legacy = _legacy_payload(args.legacy_context_checkpoint)
    legacy_context = None if legacy is None else dict(legacy["context"])
    legacy_splits = None if legacy_context is None else legacy_context.get("splits")
    action_norm = None if legacy is None else ArrayNormalizer.from_dict(legacy["action_normalizer"])
    state_norm = None if legacy is None else ArrayNormalizer.from_dict(legacy["state_normalizer"])

    dataset_config = ObservedStateDatasetConfig(
        world_horizon=args.world_horizon,
        policy_horizon=args.policy_horizon,
        segment_length=args.segment_length,
        history_offsets=tuple(args.history_offsets),
        executed_action_offsets=tuple(args.executed_action_offsets),
        target_history_offsets=tuple(args.target_history_offsets),
        stride=args.stride,
        return_images=args.condition_mode != "dinov2-cache",
    )
    dataset_config.validate()
    min_length = (
        dataset_config.world_horizon
        + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets))
        + 2
    )
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = (
        load_data(
            args,
            min_length=min_length,
            normalizer_mode=(action_norm.mode if action_norm is not None else args.normalizer),
            action_normalizer=action_norm,
            state_normalizer=state_norm,
            splits=legacy_splits,
        )
    )
    bases = {
        name: ObservedStateWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_norm,
            action_normalizer=action_norm,
            config=dataset_config,
        )
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=768,
        debug_patches_per_camera=256,
        device=device,
        dtype=dtype,
    )
    use_token_prefetch = bool(
        args.prefetch_dinov2_tokens
        and args.condition_mode == "dinov2-cache"
        and hasattr(conditioner, "store")
    )
    if use_token_prefetch:
        token_store = conditioner.store  # type: ignore[attr-defined]
        train_dataset = CachedTokenPolicyWindowDataset(
            bases["train"], token_store=token_store, future_anchors=int(args.future_anchors)
        )
        val_dataset = CachedTokenPolicyWindowDataset(
            bases["val"], token_store=token_store, future_anchors=int(args.future_anchors)
        )
    else:
        train_dataset = PolicyWindowDataset(bases["train"])
        val_dataset = PolicyWindowDataset(bases["val"])
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
    if patches is None:
        probe_sample = bases["train"][0]
        latent_dim, patches = infer_dense_geometry(conditioner, probe_sample, camera_names=cameras)
    visual_geometry = {
        "source": args.condition_mode,
        "latent_dim": int(latent_dim),
        "patches_per_camera": int(patches),
        "num_cameras": len(cameras),
        "history_length": len(dataset_config.history_offsets),
        "future_count": len(dataset_config.future_offsets),
    }

    policy_config = V39PolicyConfig(
        action_dim=int(action_norm.scale.shape[-1]),
        state_dim=int(state_norm.scale.shape[-1]),
        action_horizon=dataset_config.policy_horizon,
        executed_history_length=len(dataset_config.executed_action_offsets),
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        depth=args.depth,
        action_decoder_depth=1,
        proposal_depth=args.proposal_depth,
        proposal_dropout=args.proposal_dropout,
        dropout=args.dropout,
        event_tokens=args.event_tokens,
        gripper_dim_index=args.gripper_dim_index,
        inference_steps=args.inference_steps,
        first_execution_steps=args.first_execution_steps,
        mid_execution_steps=args.mid_execution_steps,
        physical_decode_delta_blend=args.physical_decode_delta_blend,
        visual_token_dim=int(latent_dim),
        visual_history_length=len(dataset_config.history_offsets),
        num_cameras=len(cameras),
        patches_per_camera=int(patches),
        canvas_registers=args.canvas_registers,
        future_anchors=min(int(args.future_anchors), len(dataset_config.future_offsets)),
        target_future_count=len(dataset_config.future_offsets),
        visual_memory_dropout=args.visual_memory_dropout,
        canvas_dropout=args.canvas_dropout,
        role_dropout=args.role_dropout,
        action_basis_tokens=args.action_basis_tokens,
        future_grid_size=args.future_grid_size,
        rollout_tail_start_step=args.rollout_tail_start_step,
        rollout_tail_full_step=args.rollout_tail_full_step,
        controlled_delta_rank=args.controlled_delta_rank,
        base_effect_hidden=args.base_effect_hidden,
        latent_action_tokens=args.latent_action_tokens,
        neutral_action_tokens=args.neutral_action_tokens,
        controlled_delta_dropout=args.controlled_delta_dropout,
        midcut_layer=args.midcut_layer,
        midcut_future_gain_init=args.midcut_future_gain_init,
        layer_contract_adapters=args.layer_contract_adapters,
        layer_contract_adapter_dim=args.layer_contract_adapter_dim,
        layer_contract_grad_scale=args.layer_contract_grad_scale,
        layer_contract_residual_scale=args.layer_contract_residual_scale,
        layer_shared_fm_probe=args.layer_shared_fm_probe,
        layer_fm_probe_hidden=args.layer_fm_probe_hidden,
        layer_recurrent_consequence=args.layer_recurrent_consequence,
        layer_consequence_steps=args.layer_consequence_steps,
        layer_consequence_hidden=args.layer_consequence_hidden,
        layer_consequence_delta_scale=args.layer_consequence_delta_scale,
        layer_consequence_initial_gain=args.layer_consequence_initial_gain,
        layer_causal_feedback_depth=args.layer_causal_feedback_depth,
        layer_causal_memory_tokens=args.layer_causal_memory_tokens,
        layer_low_causal_weight=args.layer_low_causal_weight,
        layer_high_causal_weight=args.layer_high_causal_weight,
        layer_low_latent_weight=args.layer_low_latent_weight,
        layer_high_latent_weight=args.layer_high_latent_weight,
        layer_causal_event_from_effect=args.layer_causal_event_from_effect,
        layer_state_counterfactual=args.layer_state_counterfactual,
    )
    system = V39PolicySystem(policy_config)
    if args.stage1_checkpoint is not None:
        stage_payload = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        if stage_payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
            raise ValueError("--stage1-checkpoint must be a V39/V40 checkpoint")
        missing, unexpected = system.load_state_dict(stage_payload["model"], strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys while loading --stage1-checkpoint: {unexpected[:8]}")
        if missing:
            print(
                f"[v39-init] missing keys from stage1 checkpoint: {missing[:8]} count={len(missing)}",
                flush=True,
            )
    trainer = V39PolicyTrainerConfig(
        **{name: getattr(args, name) for name in V39PolicyTrainerConfig.__dataclass_fields__}
    )
    context = {
        "schema": "clearvla-v39-staged-midcut-contract-context-v1",
        "args": vars(args),
        "legacy_context_checkpoint": None
        if args.legacy_context_checkpoint is None
        else str(args.legacy_context_checkpoint),
        "stage1_checkpoint": None
        if args.stage1_checkpoint is None
        else str(args.stage1_checkpoint),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "dataset": asdict(dataset_config),
        "visual_geometry": visual_geometry,
        "policy_model": asdict(policy_config),
        "trainer": asdict(trainer),
        "parameter_report": system.parameter_report(),
        "performance_contract": {
            "prefetch_dinov2_tokens": bool(use_token_prefetch),
            "target_future_encoding": "future_anchors_last_history_only",
            "future_target_is_input": False,
            "rollout_dynamics_bound": True,
            "controlled_residual_dynamics": True,
            "weak_visual_base_plus_action_delta": True,
            "counterfactual_delta_contrast": True,
            "tail_action_reads_controlled_delta": True,
            "midcut_layer": int(args.midcut_layer),
            "layer_contract_adapters": int(args.layer_contract_adapters),
            "staged_midcut_contract": True,
            "v39_1_multi_layer_adapter_contract": bool(args.layer_contract_adapters),
            "v39_2_multi_layer_latent_head_shared_fm_probe": bool(args.layer_shared_fm_probe),
            "v39_3_recurrent_milestone_consequence": bool(args.layer_recurrent_consequence),
            "v40_layer_causal_latent_split": True,
            "v40_lower_layers_are_causal": True,
            "v40_upper_layers_are_latent": True,
            "v40_hard_action_negative": "within_batch_farthest_action",
            "v39_3_consequence_steps": int(args.layer_consequence_steps),
            "v40_causal_feedback_depth": int(args.layer_causal_feedback_depth),
            "metric_sync": "log_every_and_epoch_end",
            "dino_cache_reads": "episode_grouped_mmap",
        },
        "skipped": skipped,
        "policy_contract": (
            "V40 is independent of old world checkpoints by default. It keeps the V39 staged training contract but replaces the ambiguous action/direct auxiliary heads with "
            "a layer-role split: shallow layers emphasize action-causal step deltas, upper layers emphasize world/future latent alignment, and middle layers bridge them. "
            "Counterfactual contrast is applied to local step deltas using within-batch farthest-action negatives rather than reverse-batch near-neighbor shuffles."
        ),
    }
    print_context(context)
    train_v39_policy(
        system=system,
        train_loader=train_loader,
        val_loader=val_loader,
        conditioner=conditioner,
        device=device,
        dtype=dtype,
        camera_names=cameras,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        trainer=trainer,
        out_dir=args.out_dir,
        context=context,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
