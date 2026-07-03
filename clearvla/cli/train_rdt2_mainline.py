from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
from clearvla.experiments.classic_policy_lab.rdt2_mainline import MainlineRDT2FM, MainlineRDT2FMConfig
from clearvla.experiments.classic_policy_lab.rdt2_mainline_runtime import train_mainline_rdt2_fm
from clearvla.experiments.classic_policy_lab.trainer import RDTTrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the v29 ClearVLA RDT2 chunk policy with an isolated action-conditioned future-DINO residual world model")
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

    p.add_argument("--visual-corrector", choices=["none", "query-latent"], default="none", help="Optional structured fast visual readout; stable baseline is none")
    p.add_argument("--visual-top-query-tokens", type=int, default=2)
    p.add_argument("--visual-wrist-query-tokens", type=int, default=4)
    p.add_argument("--visual-query-hidden-size", type=int, default=None)
    p.add_argument("--visual-query-heads", type=int, default=4)
    p.add_argument("--visual-latent-max-scale", type=float, default=0.10)
    p.add_argument("--visual-latent-init-logit", type=float, default=-3.0)
    p.add_argument("--visual-top-gate-floor", type=float, default=0.0, help="Minimum retained top-camera query-latent gate; 0 keeps v28.2 behavior")

    p.add_argument(
        "--future-latent-variant",
        choices=["none", "world-only", "closed-loop"],
        default="none",
        help=(
            "world-only trains a fully isolated forward world model; closed-loop additionally "
            "updates the policy through conservative demonstration-relative consequence "
            "consistency evaluated by a parameter-detached world model"
        ),
    )
    p.add_argument(
        "--future-latent-offsets", nargs="+", type=int, default=[8, 16, 24],
        help="Future frame offsets relative to the current observation",
    )
    p.add_argument(
        "--match-future-window-support", action="store_true",
        help="With variant=none, crop windows to the same future-frame support as world-model runs for a fair policy baseline",
    )
    p.add_argument(
        "--future-latent-grid-size", type=int, default=8,
        help="Spatial DINO residual grid per camera/time; 8 gives 384 tokens for 3 times x 2 cameras",
    )
    p.add_argument("--future-latent-hidden-size", type=int, default=768)
    p.add_argument("--future-latent-depth", type=int, default=6)
    p.add_argument("--future-latent-heads", type=int, default=8)
    p.add_argument("--future-latent-kv-heads", type=int, default=4)
    p.add_argument("--future-latent-modulation-rank", type=int, default=192)
    p.add_argument("--future-world-loss-weight", type=float, default=0.10)
    p.add_argument("--future-endpoint-loss-weight", type=float, default=0.0)
    p.add_argument("--future-motion-weight", type=float, default=1.0)
    p.add_argument("--future-motion-weight-cap", type=float, default=4.0)
    p.add_argument("--future-dependency-loss-weight", type=float, default=0.01)
    p.add_argument("--future-action-semantic-dim", type=int, default=256)
    p.add_argument("--future-action-semantic-hidden-size", type=int, default=256)
    p.add_argument("--future-action-semantic-depth", type=int, default=2)
    p.add_argument("--future-action-semantic-heads", type=int, default=4)
    p.add_argument("--future-action-semantic-kv-heads", type=int, default=2)
    p.add_argument("--future-align-loss-weight", type=float, default=0.05)
    p.add_argument("--future-inverse-loss-weight", type=float, default=0.10)
    p.add_argument("--future-current-action-baseline-loss-weight", type=float, default=0.02)
    p.add_argument("--future-action-reconstruction-loss-weight", type=float, default=0.05)
    p.add_argument("--future-embedding-variance-loss-weight", type=float, default=0.02)
    p.add_argument("--future-embedding-covariance-loss-weight", type=float, default=0.005)
    p.add_argument("--future-contrastive-temperature", type=float, default=0.10)
    p.add_argument("--future-structured-nce-weight", type=float, default=0.25)
    p.add_argument("--future-contrastive-transition-boost", type=float, default=1.0)
    p.add_argument("--future-contrastive-duplicate-threshold", type=float, default=1e-6)
    p.add_argument("--future-embedding-std-target", type=float, default=0.05)
    p.add_argument("--future-pred-align-loss-weight", type=float, default=0.05)
    p.add_argument("--future-cycle-loss-weight", type=float, default=0.05)
    p.add_argument("--future-align-margin", type=float, default=0.10)
    p.add_argument("--future-semantic-confidence-margin", type=float, default=0.10)
    p.add_argument("--future-inverse-transition-threshold", type=float, default=0.10)
    p.add_argument("--future-semantic-warmup-steps", type=int, default=3217)
    p.add_argument("--future-semantic-ramp-steps", type=int, default=3217)
    p.add_argument("--future-action-cross-scale", type=float, default=0.0, help="Fixed ungated action cross-attention residual scale; <=0 selects 1/sqrt(depth)")
    p.add_argument("--future-semantic-negative-delay", type=int, default=3)
    p.add_argument(
        "--future-dependency-relative-margin",
        type=float,
        default=0.03,
        help="Required relative increase in world-model error for a corrupted action",
    )
    p.add_argument("--future-action-time-power", type=float, default=1.0, help="Emphasize low future-flow times where the noisy future reveals less target information")
    p.add_argument("--future-action-time-floor", type=float, default=0.10)
    p.add_argument("--future-policy-bridge-time-power", type=float, default=1.0, help="Emphasize low policy action-flow bridge times for consequence supervision")
    p.add_argument("--future-policy-bridge-time-floor", type=float, default=0.10)
    p.add_argument(
        "--future-consistency-relative-margin",
        type=float,
        default=0.02,
        help="Allowed relative future-error slack versus the demonstrated action",
    )
    p.add_argument("--future-consistency-regret-cap", type=float, default=2.0)
    p.add_argument(
        "--future-consistency-teacher-weight",
        type=float,
        default=0.25,
        help="Weight of normalized demonstrated-consequence matching inside closed-loop transfer",
    )
    p.add_argument("--future-consistency-teacher-cap", type=float, default=1.0)
    p.add_argument(
        "--future-consistency-world-skill-margin",
        type=float,
        default=0.10,
        help="Relative improvement over zero velocity required for full world-skill confidence",
    )
    p.add_argument(
        "--future-consistency-confidence-floor",
        type=float,
        default=0.0,
        help="Optional minimum closed-loop confidence; zero fully disables untrusted transfer",
    )
    p.add_argument("--future-consistency-weight-cap", type=float, default=4.0)
    p.add_argument("--future-consistency-loss-weight", type=float, default=0.02)
    p.add_argument("--future-consistency-warmup-steps", type=int, default=3217)
    p.add_argument("--future-consistency-ramp-steps", type=int, default=3217)
    p.add_argument("--future-world-lr", type=float, default=None, help="World-model optimizer LR; defaults to --lr")
    p.add_argument("--future-world-weight-decay", type=float, default=None, help="World-model weight decay; defaults to --weight-decay")
    p.add_argument("--future-world-grad-clip", type=float, default=1.0, help="Independent world-model gradient clipping threshold")
    p.add_argument("--future-latent-stat-eps", type=float, default=1e-5)
    p.add_argument(
        "--future-latent-stat-batches", type=int, default=128,
        help="Batches used once to estimate fixed per-time/per-camera/per-channel DINO-residual statistics; 0 uses the full train loader",
    )
    p.add_argument(
        "--no-component-grad-log", action="store_true",
        help="Disable policy-vs-latent shared-gradient norms at log intervals",
    )

    p.add_argument("--horizon-weight-mode", choices=["prefix", "uniform", "chunk-balanced"], default="chunk-balanced", help="v29 default trains the whole 24-step chunk for chunk execution")
    p.add_argument("--first-position-weight", type=float, default=8.0)
    p.add_argument("--first4-position-weight", type=float, default=4.0)
    p.add_argument("--first8-position-weight", type=float, default=2.0)
    p.add_argument("--tail-position-weight", type=float, default=1.0)
    p.add_argument("--chunk-first4-position-weight", type=float, default=1.5)
    p.add_argument("--chunk-middle-position-weight", type=float, default=1.5, help="Weight for steps 5-12 in chunk-balanced mode")
    p.add_argument("--chunk-late-position-weight", type=float, default=1.5, help="Weight for steps 13-20 in chunk-balanced mode")
    p.add_argument("--chunk-tail-position-weight", type=float, default=1.2, help="Weight for steps 21+ in chunk-balanced mode")
    p.add_argument("--prior-loss-weight", type=float, default=0.50)
    p.add_argument("--fast-exit-loss-weight", type=float, default=1.00)
    p.add_argument("--prefix-exit-loss-weight", type=float, default=0.50)
    p.add_argument("--full-flow-loss-weight", type=float, default=1.00)
    p.add_argument("--arm-delta-loss-weight", type=float, default=0.10, help="Endpoint arm delta-matching weight for executable chunks")
    p.add_argument("--align-phase-loss-weight", type=float, default=0.0, help="Extra arm endpoint loss on steps before true close transitions")
    p.add_argument("--align-phase-pre-steps", type=int, default=8)
    p.add_argument("--gripper-dim-index", type=int, default=-1)
    p.add_argument("--gripper-open-value", type=float, default=None, help="Raw physical gripper value for fully open; defaults to the train split minimum")
    p.add_argument("--gripper-close-value", type=float, default=None, help="Raw physical gripper value for fully closed; defaults to the train split maximum")
    p.add_argument("--gripper-openness-residual-scale", type=float, default=1.0, help="Maximum normalized continuous openness correction around the held physical prior")
    p.add_argument("--arm-flow-loss-weight", type=float, default=1.0)
    p.add_argument("--gripper-state-loss-weight", type=float, default=2.0)
    p.add_argument("--gripper-transition-boost", type=float, default=3.0)
    p.add_argument("--gripper-transition-aux-weight", type=float, default=0.50)
    p.add_argument("--gripper-transition-threshold", type=float, default=0.10, help="Transition threshold in continuous openness units [0,1]")
    p.add_argument("--gripper-transition-radius", type=int, default=1)
    p.add_argument("--gripper-smooth-weight", type=float, default=0.02, help="Weight for continuous openness delta-matching loss")

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


def _resolve_mainline_shape(args: argparse.Namespace) -> None:
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
    if args.visual_query_hidden_size is None:
        args.visual_query_hidden_size = min(256, args.hidden_size)


def _resolve_gripper_calibration(args: argparse.Namespace, action_norm, action_dim: int) -> dict[str, float | str]:
    index = int(args.gripper_dim_index)
    if index < 0:
        index += int(action_dim)
    if not (0 <= index < int(action_dim)):
        raise ValueError(f"gripper_dim_index={args.gripper_dim_index} invalid for action_dim={action_dim}")
    minimum = float(action_norm.minimum.reshape(-1)[index])
    maximum = float(action_norm.maximum.reshape(-1)[index])
    explicit = args.gripper_open_value is not None or args.gripper_close_value is not None
    if explicit and (args.gripper_open_value is None or args.gripper_close_value is None):
        raise ValueError("--gripper-open-value and --gripper-close-value must be provided together")
    open_raw = minimum if args.gripper_open_value is None else float(args.gripper_open_value)
    close_raw = maximum if args.gripper_close_value is None else float(args.gripper_close_value)
    if open_raw == close_raw:
        raise ValueError("gripper open and close values must be distinct")
    scale = float(action_norm.scale.reshape(-1)[index])
    offset = float(action_norm.offset.reshape(-1)[index])
    return {
        "source": "explicit-cli" if explicit else "train-split-minmax",
        "open_raw": open_raw,
        "close_raw": close_raw,
        "open_normalized": open_raw * scale + offset,
        "close_normalized": close_raw * scale + offset,
    }


def main() -> None:
    args = parse_args()
    _resolve_mainline_shape(args)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from clearvla.experiments.classic_policy_lab.cli_common import resolve_device
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype, device)
    future_offsets = (
        tuple(int(value) for value in args.future_latent_offsets)
        if args.future_latent_variant != "none" or args.match_future_window_support
        else ()
    )
    future_span = max(future_offsets, default=0) + 1
    min_length = max(args.prediction_horizon, future_span) + max(abs(args.state_offset), abs(args.image_offset), abs(args.action_offset)) + 1
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = load_data(args, min_length=min_length, normalizer_mode=args.normalizer)
    action_dim = int(episodes[0].actions_raw.shape[1]); state_dim = int(episodes[0].states_raw.shape[1])
    gripper_calibration = _resolve_gripper_calibration(args, action_norm, action_dim)
    cameras = tuple(str(value) for value in args.cameras)
    data_config = RDT2FMDatasetConfig(
        prediction_horizon=args.prediction_horizon,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        zero_state=args.zero_state,
        future_latent_offsets=future_offsets,
        return_future_images=(args.future_latent_variant != "none" and args.condition_mode != "dinov2-cache"),
    )
    train_ds = RDT2FMWindowDataset(episodes, train_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    val_ds = RDT2FMWindowDataset(episodes, val_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=data_config)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner = _build_conditioner(args, episodes=episodes, cameras=cameras, device=device, dtype=dtype, depth=args.depth, kv_heads=args.kv_heads, head_dim=args.hidden_size // args.heads)
    dense_dim = int(getattr(conditioner, "token_dim")) if hasattr(conditioner, "token_dim") else None
    if args.visual_corrector == "query-latent" and dense_dim is None:
        raise ValueError("--visual-corrector query-latent requires a dense-token conditioner such as dinov2, dinov2-cache, or debug-dense")
    if args.future_latent_variant != "none":
        if dense_dim is None or args.condition_mode in {"none", "debug-kv", "rdt2-vq"}:
            raise ValueError("future latent dynamics requires dense DINO-style tokens; use dinov2, dinov2-cache, or debug-dense")
        if args.condition_mode == "debug-dense":
            raise ValueError("formal future latent experiments require DINO tokens; debug-dense is only suitable for unit tests")
        if len(cameras) < 1:
            raise ValueError("future latent dynamics requires at least one camera")
    adaptor = None if dense_dim is None or args.dense_condition_adaptor == "none" else args.dense_condition_adaptor
    if dense_dim is not None and adaptor is None and dense_dim != args.hidden_size:
        raise ValueError("dense condition tokens require an adaptor when token width differs from hidden size")
    config = MainlineRDT2FMConfig(
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
        visual_corrector=args.visual_corrector, visual_top_query_tokens=args.visual_top_query_tokens,
        visual_wrist_query_tokens=args.visual_wrist_query_tokens, visual_query_hidden_size=args.visual_query_hidden_size,
        visual_query_heads=args.visual_query_heads, visual_latent_max_scale=args.visual_latent_max_scale,
        visual_latent_init_logit=args.visual_latent_init_logit, visual_top_gate_floor=args.visual_top_gate_floor,
        future_latent_variant=args.future_latent_variant, future_latent_dim=dense_dim,
        future_latent_offsets=future_offsets or tuple(int(value) for value in args.future_latent_offsets),
        future_latent_num_cameras=len(cameras), future_latent_grid_size=args.future_latent_grid_size,
        future_latent_hidden_size=args.future_latent_hidden_size,
        future_latent_depth=args.future_latent_depth,
        future_latent_heads=args.future_latent_heads,
        future_latent_kv_heads=args.future_latent_kv_heads,
        future_latent_modulation_rank=args.future_latent_modulation_rank,
        future_world_loss_weight=args.future_world_loss_weight,
        future_endpoint_loss_weight=args.future_endpoint_loss_weight,
        future_motion_weight=args.future_motion_weight,
        future_motion_weight_cap=args.future_motion_weight_cap,
        future_dependency_loss_weight=args.future_dependency_loss_weight,
        future_action_semantic_dim=args.future_action_semantic_dim,
        future_action_semantic_hidden_size=args.future_action_semantic_hidden_size,
        future_action_semantic_depth=args.future_action_semantic_depth,
        future_action_semantic_heads=args.future_action_semantic_heads,
        future_action_semantic_kv_heads=args.future_action_semantic_kv_heads,
        future_align_loss_weight=args.future_align_loss_weight,
        future_inverse_loss_weight=args.future_inverse_loss_weight,
        future_current_action_baseline_loss_weight=args.future_current_action_baseline_loss_weight,
        future_action_reconstruction_loss_weight=args.future_action_reconstruction_loss_weight,
        future_embedding_variance_loss_weight=args.future_embedding_variance_loss_weight,
        future_embedding_covariance_loss_weight=args.future_embedding_covariance_loss_weight,
        future_contrastive_temperature=args.future_contrastive_temperature,
        future_structured_nce_weight=args.future_structured_nce_weight,
        future_contrastive_transition_boost=args.future_contrastive_transition_boost,
        future_contrastive_duplicate_threshold=args.future_contrastive_duplicate_threshold,
        future_embedding_std_target=args.future_embedding_std_target,
        future_pred_align_loss_weight=args.future_pred_align_loss_weight,
        future_cycle_loss_weight=args.future_cycle_loss_weight,
        future_align_margin=args.future_align_margin,
        future_semantic_confidence_margin=args.future_semantic_confidence_margin,
        future_inverse_transition_threshold=args.future_inverse_transition_threshold,
        future_semantic_warmup_steps=args.future_semantic_warmup_steps,
        future_semantic_ramp_steps=args.future_semantic_ramp_steps,
        future_action_cross_scale=args.future_action_cross_scale,
        future_semantic_negative_delay=args.future_semantic_negative_delay,
        future_dependency_relative_margin=args.future_dependency_relative_margin,
        future_action_time_power=args.future_action_time_power,
        future_action_time_floor=args.future_action_time_floor,
        future_policy_bridge_time_power=args.future_policy_bridge_time_power,
        future_policy_bridge_time_floor=args.future_policy_bridge_time_floor,
        future_consistency_relative_margin=args.future_consistency_relative_margin,
        future_consistency_regret_cap=args.future_consistency_regret_cap,
        future_consistency_teacher_weight=args.future_consistency_teacher_weight,
        future_consistency_teacher_cap=args.future_consistency_teacher_cap,
        future_consistency_world_skill_margin=args.future_consistency_world_skill_margin,
        future_consistency_confidence_floor=args.future_consistency_confidence_floor,
        future_consistency_weight_cap=args.future_consistency_weight_cap,
        future_consistency_loss_weight=args.future_consistency_loss_weight,
        future_consistency_warmup_steps=args.future_consistency_warmup_steps,
        future_consistency_ramp_steps=args.future_consistency_ramp_steps,
        future_latent_stat_eps=args.future_latent_stat_eps,
        horizon_weight_mode=args.horizon_weight_mode,
        first_position_weight=args.first_position_weight, first4_position_weight=args.first4_position_weight,
        first8_position_weight=args.first8_position_weight, tail_position_weight=args.tail_position_weight,
        chunk_first4_position_weight=args.chunk_first4_position_weight,
        chunk_middle_position_weight=args.chunk_middle_position_weight,
        chunk_late_position_weight=args.chunk_late_position_weight,
        chunk_tail_position_weight=args.chunk_tail_position_weight,
        prior_loss_weight=args.prior_loss_weight, fast_exit_loss_weight=args.fast_exit_loss_weight,
        prefix_exit_loss_weight=args.prefix_exit_loss_weight, full_flow_loss_weight=args.full_flow_loss_weight,
        arm_delta_loss_weight=args.arm_delta_loss_weight, align_phase_loss_weight=args.align_phase_loss_weight,
        align_phase_pre_steps=args.align_phase_pre_steps,
        gripper_dim_index=args.gripper_dim_index, arm_flow_loss_weight=args.arm_flow_loss_weight,
        gripper_open_raw=float(gripper_calibration["open_raw"]), gripper_close_raw=float(gripper_calibration["close_raw"]),
        gripper_open_normalized=float(gripper_calibration["open_normalized"]), gripper_close_normalized=float(gripper_calibration["close_normalized"]),
        gripper_openness_residual_scale=args.gripper_openness_residual_scale, gripper_state_loss_weight=args.gripper_state_loss_weight,
        gripper_transition_boost=args.gripper_transition_boost, gripper_transition_aux_weight=args.gripper_transition_aux_weight,
        gripper_transition_threshold=args.gripper_transition_threshold, gripper_transition_radius=args.gripper_transition_radius,
        gripper_smooth_weight=args.gripper_smooth_weight,
    )
    model = MainlineRDT2FM(config, dtype=dtype).to(device=device, dtype=dtype)
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
    used_episode_ids = set(train_ids) | set(val_ids) | set(test_ids)
    unused_episode_ids = [index for index in range(len(episodes)) if index not in used_episode_ids]
    split_episode_names = {
        "train": [episodes[index].stem for index in train_ids],
        "val": [episodes[index].stem for index in val_ids],
        "test": [episodes[index].stem for index in test_ids],
        "unused": [episodes[index].stem for index in unused_episode_ids],
    }
    context = {
        "schema": "clearvla-rdt2-mainline-context-v8-contrastive-action-anchor",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "split_summary": {
            "mode": args.episode_split_mode,
            "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids), "unused": len(unused_episode_ids)},
            "episode_names": split_episode_names,
        },
        "skipped": skipped,
        "data": serializable(vars(data_config)),
        "model": model.config_dict(),
        "trainer": serializable(vars(trainer)),
        "conditioning": {"mode": args.condition_mode, "dense_token_dim": dense_dim, "instruction": args.instruction},
        "future_latent": {
            "variant": args.future_latent_variant,
            "offsets": list(future_offsets),
            "temporal_strides": [future_offsets[0]] + [future_offsets[i] - future_offsets[i - 1] for i in range(1, len(future_offsets))] if future_offsets else [],
            "grid_size": args.future_latent_grid_size,
            "tokens_per_camera_time": (
                args.future_latent_grid_size ** 2 if args.future_latent_variant != "none" else 0
            ),
            "total_tokens": (
                len(future_offsets) * len(cameras) * (args.future_latent_grid_size ** 2)
                if args.future_latent_variant != "none" else 0
            ),
            "matched_window_support_only": bool(
                args.match_future_window_support and args.future_latent_variant == "none"
            ),
            "target": "future_dino_minus_current_dino",
            "normalization_axes": "future_time,camera,channel",
            "action_contract": "complete_normalized_action_plus_boundary_delta",
            "action_prefixes": list(future_offsets) if args.future_latent_variant != "none" else [],
            "action_encoder_attention": "causal",
            "future_stream_attention": "block-causal-by-future-time",
            "world_model_hidden_size": args.future_latent_hidden_size,
            "world_model_depth": args.future_latent_depth,
            "world_model_heads": args.future_latent_heads,
            "world_model_kv_heads": args.future_latent_kv_heads,
            "world_loss_weight": args.future_world_loss_weight,
            "motion_weight": args.future_motion_weight,
            "dependency_weight": args.future_dependency_loss_weight,
            "semantic_contract": "pure-action-prefix <-> future-DINO-change via symmetric InfoNCE, shared inverse decoder, and anti-collapse regularization",
            "semantic_dim": args.future_action_semantic_dim,
            "semantic_hidden_size": args.future_action_semantic_hidden_size,
            "semantic_depth": args.future_action_semantic_depth,
            "semantic_heads": args.future_action_semantic_heads,
            "semantic_kv_heads": args.future_action_semantic_kv_heads,
            "align_weight": args.future_align_loss_weight,
            "alignment_objective": "per-prefix symmetric InfoNCE + structured-negative NCE",
            "contrastive_temperature": args.future_contrastive_temperature,
            "structured_nce_weight": args.future_structured_nce_weight,
            "contrastive_transition_boost": args.future_contrastive_transition_boost,
            "contrastive_duplicate_threshold": args.future_contrastive_duplicate_threshold,
            "inverse_weight": args.future_inverse_loss_weight,
            "current_only_baseline_weight": args.future_current_action_baseline_loss_weight,
            "action_reconstruction_weight": args.future_action_reconstruction_loss_weight,
            "embedding_variance_weight": args.future_embedding_variance_loss_weight,
            "embedding_covariance_weight": args.future_embedding_covariance_loss_weight,
            "embedding_std_target": args.future_embedding_std_target,
            "pred_align_weight": args.future_pred_align_loss_weight,
            "cycle_weight": args.future_cycle_loss_weight,
            "align_margin": args.future_align_margin,
            "semantic_confidence_margin": args.future_semantic_confidence_margin,
            "semantic_warmup_steps": args.future_semantic_warmup_steps,
            "semantic_ramp_steps": args.future_semantic_ramp_steps,
            "action_cross_scale": args.future_action_cross_scale,
            "semantic_negatives": ["matched", "gripper_delay", "gripper_remove", "tail_hold"],
            "dependency_relative_margin": args.future_dependency_relative_margin,
            "action_time_power": args.future_action_time_power,
            "action_time_floor": args.future_action_time_floor,
            "policy_bridge_time_power": args.future_policy_bridge_time_power,
            "policy_bridge_time_floor": args.future_policy_bridge_time_floor,
            "consistency_relative_margin": args.future_consistency_relative_margin,
            "consistency_regret_cap": args.future_consistency_regret_cap,
            "consistency_teacher_weight": args.future_consistency_teacher_weight,
            "consistency_teacher_cap": args.future_consistency_teacher_cap,
            "consistency_world_skill_margin": args.future_consistency_world_skill_margin,
            "consistency_confidence_floor": args.future_consistency_confidence_floor,
            "consistency_weight_cap": args.future_consistency_weight_cap,
            "consistency_weight": args.future_consistency_loss_weight,
            "consistency_warmup_steps": args.future_consistency_warmup_steps,
            "consistency_ramp_steps": args.future_consistency_ramp_steps,
            "world_optimizer_lr": args.lr if args.future_world_lr is None else args.future_world_lr,
            "world_optimizer_weight_decay": args.weight_decay if args.future_world_weight_decay is None else args.future_world_weight_decay,
            "world_optimizer_grad_clip": args.future_world_grad_clip,
            "independent_gradient_clipping": True,
            "parameter_sharing_with_policy": False,
            "policy_transfer": (
                "demonstration-relative conservative consequence consistency through "
                "detached world-model parameters"
            ),
            "stat_batches": args.future_latent_stat_batches,
            "component_grad_logging": not args.no_component_grad_log,
        },
        "gripper_calibration": gripper_calibration,
        "base_load": load_report,
        "parameters": model.parameter_count(),
        "future_latent_parameters": sum(parameter.numel() for parameter in model.future_latent_parameters()),
        "future_world_model_depth": (model.future_dynamics.depth if model.future_dynamics is not None else 0),
        "future_world_model_policy_shared_parameters": 0,
        "train_windows": len(train_ds), "val_windows": len(val_ds),
    }
    print_context(context)
    train_mainline_rdt2_fm(
        model=model, conditioner=conditioner, train_loader=train_loader, val_loader=val_loader,
        device=device, out_dir=args.out_dir, trainer=trainer, action_normalizer=action_norm,
        state_normalizer=state_norm, context=context, inference_steps=args.inference_steps,
        instruction=args.instruction, future_latent_stat_batches=args.future_latent_stat_batches,
        log_component_grad_norms=not args.no_component_grad_log,
        future_world_lr=(args.lr if args.future_world_lr is None else args.future_world_lr),
        future_world_weight_decay=(
            args.weight_decay if args.future_world_weight_decay is None else args.future_world_weight_decay
        ),
        future_world_grad_clip=args.future_world_grad_clip,
    )
    if args.dry_run:
        print("status: mainline dry-run passed", flush=True)


if __name__ == "__main__":
    main()
