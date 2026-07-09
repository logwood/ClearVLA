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
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner, infer_dense_geometry
from clearvla.experiments.observed_state_lab.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
    PolicyWindowDataset,
)
from clearvla.experiments.observed_state_lab.policy_v39 import V39PolicyConfig, V39PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
    train_v39_policy,
)


def _parse_offsets(text: str) -> tuple[int, ...]:
    values = tuple(int(x) for x in str(text).replace(",", " ").split())
    if not values:
        raise argparse.ArgumentTypeError("offset list must be non-empty")
    return values


def _normalizer_fingerprint(normalizer: ArrayNormalizer) -> str:
    """Short stable hash of the normalizer statistics.

    physical_flow_native_uniform is a cross-version anchor metric ONLY between
    runs that share this fingerprint: the native metric is priced in
    normalizer coordinates, so refitting the normalizer silently re-scales it.
    """
    import hashlib

    def _round(value: Any) -> Any:
        array = np.asarray(value)
        if array.dtype.kind in "fc":
            return np.round(array.astype(np.float64), 6).tolist()
        return array.tolist() if array.ndim else value
    payload = json.dumps(
        {key: _round(value) for key, value in sorted(normalizer.to_dict().items())},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _legacy_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clearvla-v35-world-checkpoint-v1":
        raise ValueError("--legacy-context-checkpoint must be a V35 world checkpoint")
    return payload


_STAGE1_DIRTY_ADAPTER_PREFIXES = (
    # V48 migration: keep the DiT trunk/readout warm start, but do not inherit
    # layer-contract interfaces trained under older consequence/action semantics.
    "planner.layer_contract_heads.",
    "planner.layer_consequence_cell.",
    "planner.layer_fm_probe.",
    "planner.event_probe.",
    "planner.motion_probe.",
)

_PARSEVAL_REPLACED_STAGE1_PREFIXES = (
    "planner.seed.noisy_physical_lift.grip_value.",
    "planner.seed.noisy_physical_lift.grip_delta.",
    "planner.seed.noisy_physical_lift.grip_extra.",
    "planner.latent_cvae_action_decoder.noisy_action_lift.",
    "planner.latent_cvae_action_decoder.posterior_action.",
    "planner.latent_cvae_action_decoder.velocity_head.",
)


def _filter_stage1_state_dict(
    state: dict[str, torch.Tensor],
    *,
    reset_dirty_adapters: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not reset_dirty_adapters:
        return state, []
    skipped = [
        key for key in state
        if key.startswith(_STAGE1_DIRTY_ADAPTER_PREFIXES)
    ]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def _filter_shape_mismatched_state_dict(
    state: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    skipped = [
        key for key, value in state.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    ]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def _filter_parseval_replaced_state_dict(
    state: dict[str, torch.Tensor],
    *,
    enabled: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not enabled:
        return state, []
    skipped = [key for key in state if key.startswith(_PARSEVAL_REPLACED_STAGE1_PREFIXES)]
    if not skipped:
        return state, []
    skipped_set = set(skipped)
    return {key: value for key, value in state.items() if key not in skipped_set}, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V40 layer-role causal/latent temporal policy.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--legacy-context-checkpoint", type=Path, default=None, help="Optional migration source for splits/normalizers only; not a model dependency.")
    parser.add_argument("--normalizer", choices=["zscore", "limits", "identity"], default="zscore")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Load a V39 contract-stage checkpoint as model initialization before policy-stage finetuning.")
    parser.add_argument("--stage1-reset-dirty-adapters", type=int, default=0, help="Skip old layer adapter/consequence interface weights when migrating from a pre-fix stage1 checkpoint.")
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--prefetch-dinov2-tokens", action=argparse.BooleanOptionalAction, default=True, help="For dinov2-cache mode, load current and compact future DINO tokens in DataLoader workers instead of the main training loop.")
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
    parser.add_argument("--depth", type=int, default=8, help="shared temporal canvas depth")
    parser.add_argument("--midcut-layer", type=int, default=3, help="Expose Z_mid after this many DiT blocks for the simple latent-contract heads.")
    parser.add_argument("--midcut-future-gain-init", type=float, default=0.10)
    parser.add_argument("--layer-contract-adapters", type=int, default=1, help="V40: enable layer-role contract adapters after every DiT block.")
    parser.add_argument("--layer-contract-adapter-dim", type=int, default=128)
    parser.add_argument("--layer-contract-grad-scale", type=float, default=1.0)
    parser.add_argument("--layer-contract-residual-scale", type=float, default=0.50)
    parser.add_argument("--layer-shared-fm-probe", type=int, default=0, help="Deprecated in V40 by default; keep disabled so action_pred does not become a side branch.")
    parser.add_argument("--layer-fm-probe-hidden", type=int, default=256)
    parser.add_argument("--layer-recurrent-consequence", type=int, default=1, help="V40: enable the layer-role causal/effect branch.")
    parser.add_argument("--layer-consequence-steps", type=int, default=6)
    parser.add_argument("--layer-consequence-hidden", type=int, default=256)
    parser.add_argument("--layer-consequence-delta-scale", type=float, default=1.0)
    parser.add_argument("--layer-consequence-initial-gain", type=float, default=0.10)
    parser.add_argument("--layer-causal-feedback-depth", type=int, default=0, help="V40.1: optional inner interaction blocks; default 0 because the unified intervention head already encodes state-action jointly.")
    parser.add_argument("--layer-causal-memory-tokens", type=int, default=4, help="V40.1: compact memory/context tokens for unified intervention latent.")
    parser.add_argument("--layer-low-causal-weight", type=float, default=1.0, help="V40: causal gain at the shallowest layer.")
    parser.add_argument("--layer-high-causal-weight", type=float, default=1.0, help="V40.1: retained as diagnostic; unified intervention head is not mixed with a separate latent head.")
    parser.add_argument("--layer-low-latent-weight", type=float, default=1.0, help="V40.1: retained as diagnostic; unified intervention head is not mixed with a separate latent head.")
    parser.add_argument("--layer-high-latent-weight", type=float, default=1.0, help="V40: latent gain at the deepest layer.")
    parser.add_argument("--layer-causal-event-from-effect", type=int, default=1, help="V40: read event logits from causal effect tokens in layer contracts.")
    parser.add_argument("--layer-state-counterfactual", type=int, default=0, help="Experimental non-strict state/frame shuffle diagnostic; disabled by default and excluded from the recommended training path.")
    parser.add_argument("--action-consequence-self-condition", type=int, default=0, help="Use a no-grad deployable clean-action preview as the layer consequence action input.")
    parser.add_argument("--layer-zero-base-diagnostic", type=int, default=1, help="Log consequence-output shift when layer rollout tokens are zeroed; diagnostic only, no loss.")
    parser.add_argument("--proposal-depth", type=int, default=2)
    parser.add_argument("--proposal-dropout", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--event-tokens", type=int, default=3)
    parser.add_argument("--canvas-registers", type=int, default=12)
    parser.add_argument("--future-anchors", type=int, default=6)
    parser.add_argument("--future-grid-size", type=int, default=4)
    parser.add_argument("--action-basis-tokens", type=int, default=4)
    parser.add_argument("--rollout-tail-start-step", type=int, default=8)
    parser.add_argument("--rollout-tail-full-step", type=int, default=13)
    parser.add_argument("--controlled-delta-rank", type=int, default=8)
    parser.add_argument("--base-effect-hidden", type=int, default=128)
    parser.add_argument(
        "--controlled-base-mode",
        choices=("learned", "fixed_zero"),
        default="fixed_zero",
        help="Use an identifiable no-change rollout base; learned is retained only for historical checkpoint evaluation.",
    )
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
    parser.add_argument(
        "--arm-flow-mode",
        choices=("legacy_independent", "manifold_native"),
        default="legacy_independent",
    )
    parser.add_argument("--arm-noise-temporal-rho", type=float, default=0.0)
    parser.add_argument("--gripper-field-dim", type=int, default=12)
    parser.add_argument("--gripper-field-mode", choices=("legacy_handcrafted", "parseval_temporal"), default="legacy_handcrafted")
    parser.add_argument("--final-action-decoder", choices=["legacy", "residual_action_flow", "layered_residual_action_flow", "latent_main_action", "latent_cvae_action", "adaptive_recurrent_cvae_action"], default="legacy")
    parser.add_argument("--action-flow-residual-depth", type=int, default=2)
    parser.add_argument("--action-flow-residual-high-slots", type=int, default=4)
    parser.add_argument("--action-flow-residual-max-scale", type=float, default=0.20)
    parser.add_argument("--action-flow-residual-visual-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-context-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-transition-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-layer-memory", type=int, default=1)
    parser.add_argument("--action-flow-residual-layer-pair-schedule", type=str, default="0:1,1:3,3:5,5:7")
    parser.add_argument("--action-flow-residual-layer-detach", type=int, default=1)
    parser.add_argument("--action-flow-residual-stage-router", type=int, default=1)
    parser.add_argument("--action-flow-residual-anchor-memory", type=int, default=1)
    parser.add_argument("--latent-action-decoder-depth", type=int, default=8)
    parser.add_argument("--latent-action-high-slots", type=int, default=4)
    parser.add_argument("--latent-action-layer-schedule", type=str, default="0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7")
    parser.add_argument("--latent-action-visual-memory", type=int, default=0)
    parser.add_argument("--latent-action-context-memory", type=int, default=0)
    parser.add_argument("--latent-action-transition-memory", type=int, default=1)
    parser.add_argument("--latent-action-layer-memory", type=int, default=1)
    parser.add_argument("--latent-action-anchor-memory", type=int, default=1)
    parser.add_argument("--latent-action-stage-router", type=int, default=0)
    parser.add_argument("--latent-action-layer-detach", type=int, default=0)
    parser.add_argument("--latent-action-event-gripper-gate", type=int, default=1)
    parser.add_argument("--latent-action-temporal-depth", type=int, default=0)
    parser.add_argument("--latent-action-near-steps", type=int, default=4)
    parser.add_argument("--latent-action-mid-steps", type=int, default=8)
    parser.add_argument("--latent-action-near-depth", type=int, default=2)
    parser.add_argument("--latent-action-mid-depth", type=int, default=4)
    parser.add_argument("--latent-cvae-z-dim", type=int, default=64)
    parser.add_argument("--latent-cvae-decoder-depth", type=int, default=3)
    parser.add_argument("--latent-cvae-ffn-expansion", type=float, default=2.0)
    parser.add_argument("--latent-cvae-layer-memory", type=int, default=1)
    parser.add_argument("--latent-cvae-transition-memory", type=int, default=1)
    parser.add_argument("--latent-cvae-transition-detach", type=int, default=1)
    parser.add_argument("--latent-cvae-context-memory", type=int, default=0)
    parser.add_argument("--latent-cvae-visual-memory", type=int, default=0)
    parser.add_argument("--latent-cvae-layer-detach", type=int, default=1)
    parser.add_argument("--latent-cvae-layer-grad-scale", type=float, default=0.0)
    parser.add_argument("--latent-cvae-condition-source-norm", type=int, default=1, help="Normalize each CVAE condition source before fusion.")
    parser.add_argument("--latent-cvae-bounded-consequence-fusion", type=int, default=1, help="Fuse world and action-conditioned summaries separately so routing cannot bypass consequence scaling.")
    parser.add_argument("--latent-cvae-consequence-scale-init", type=float, default=0.10, help="Initial strength of action-conditioned layer-contract summaries.")
    parser.add_argument("--latent-cvae-consequence-scale-max", type=float, default=0.50, help="Upper bound for action-conditioned layer-contract summary strength.")
    parser.add_argument("--latent-cvae-event-gripper-gate", type=int, default=1)
    parser.add_argument("--latent-cvae-inference-sample", type=int, default=0)
    parser.add_argument("--latent-cvae-output-init-std", type=float, default=1e-3)
    parser.add_argument("--latent-cvae-mu-bound", type=float, default=1.5)
    parser.add_argument("--latent-cvae-min-std", type=float, default=0.5)
    parser.add_argument("--latent-cvae-causal-attention", type=int, default=1)
    parser.add_argument("--latent-cvae-noisy-gate", type=int, default=0, help="t-gate the direct noisy-action branch of the CVAE decoder.")
    parser.add_argument("--latent-cvae-noisy-gate-min", type=float, default=0.05)
    parser.add_argument("--latent-cvae-noisy-gate-power", type=float, default=1.5)
    parser.add_argument("--latent-cvae-layer-scan", type=int, default=0, help="Use a recurrent scan over ordered layer summaries as the CVAE condition.")
    parser.add_argument("--latent-cvae-layer-scan-alpha", type=float, default=0.2, help="Residual weight for the flat lateral layer condition when layer scan is enabled.")
    parser.add_argument("--latent-cvae-mmdit-decoder", type=int, default=0, help="Use the MMDiT-lite CVAE action-condition token decoder.")
    parser.add_argument("--latent-cvae-mmdit-depth", type=int, default=3)
    parser.add_argument("--latent-cvae-mmdit-cond-update", type=int, default=0, help="Allow condition tokens to update inside the MMDiT-lite decoder.")
    parser.add_argument("--latent-cvae-mmdit-noisy-causal", type=int, default=1, help="Mask future noisy-action condition tokens for action queries.")
    parser.add_argument("--latent-cvae-mmdit-noisy-logit-gate", type=int, default=0, help="V70: LayerNorm noisy condition tokens and move the t-gate to an additive log g(t) attention-logit bias (closes the value-volume degree of freedom).")
    parser.add_argument("--latent-cvae-progress-action-isolation", type=int, default=0, help="V72: cut the action->progress->workspace echo; the per-step progress update no longer receives the raw action summary (zeros fed, parameter shapes unchanged).")
    parser.add_argument("--latent-cvae-horizon-tokens", type=int, default=24, help="Number of z-conditioned evidence workspace tokens supplied to MMDiT.")
    parser.add_argument("--latent-cvae-workspace-noisy-query", type=int, default=0, help="Condition workspace evidence queries on the current noisy flow state.")
    parser.add_argument("--latent-cvae-workspace-trajectory-source", type=int, default=1, help="Expose full-resolution trajectory canvas tokens as workspace values; set 0 for the x_t-echo ablation.")
    parser.add_argument("--latent-cvae-workspace-global-sources", type=int, default=1, help="Expose scan/lateral global condition summaries as workspace values; set 0 to keep them only in cond/z.")
    parser.add_argument("--latent-cvae-workspace-layer-source", type=int, default=1, help="Expose full layer_stack as a static workspace value; set 0 to force layer information through routed_layer/capsules.")
    parser.add_argument("--latent-cvae-workspace-progress-value", type=int, default=1, help="Expose progress as a workspace value; set 0 to use progress only as workspace step/query state.")
    parser.add_argument("--latent-cvae-workspace-time-state", type=int, default=0, help="Inject the existing z+time primary condition into workspace slots as explicit state.")
    parser.add_argument("--adaptive-cvae-refine-steps", type=int, default=3)
    parser.add_argument("--adaptive-cvae-progress-memory", type=int, default=1)
    parser.add_argument("--adaptive-cvae-progress-steps", type=int, default=6)
    parser.add_argument("--adaptive-cvae-prefix-memory", type=int, default=0)
    parser.add_argument("--adaptive-cvae-layer-routing", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-cosine", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-temperature", type=float, default=1.0)
    parser.add_argument("--adaptive-cvae-prefix-detach", type=int, default=1)
    parser.add_argument("--adaptive-cvae-progress-z-injection", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-query-bias", type=int, default=1)
    parser.add_argument("--adaptive-cvae-token-semantic-adapter", type=int, default=1)
    parser.add_argument("--adaptive-cvae-output-adapter", type=int, default=0)
    parser.add_argument("--adaptive-cvae-context-dropout", type=float, default=0.05)
    parser.add_argument("--adaptive-cvae-route-entropy-floor-ratio", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-function-adapters", type=int, default=1)
    parser.add_argument("--adaptive-cvae-function-rank", type=int, default=64)
    parser.add_argument("--adaptive-cvae-progress-role-dim", type=int, default=16)
    parser.add_argument("--adaptive-cvae-route-topk", type=int, default=0)
    parser.add_argument("--adaptive-cvae-route-sparsemax", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-adaptive-temperature", type=int, default=1)
    parser.add_argument("--adaptive-cvae-route-min-temperature", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-route-max-temperature", type=float, default=1.25)
    parser.add_argument("--adaptive-cvae-role-query", type=int, default=1)
    parser.add_argument("--adaptive-cvae-step-roles", type=int, default=1)
    parser.add_argument("--adaptive-cvae-coarse-stride", type=int, default=4)
    parser.add_argument("--adaptive-cvae-coarse-strength", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-seed-scale", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-output-scale", type=float, default=0.05)
    parser.add_argument("--adaptive-cvae-context-capsules", type=int, default=1)
    parser.add_argument("--adaptive-cvae-context-capsule-count", type=int, default=6)
    parser.add_argument("--adaptive-cvae-direct-condition-residual", type=int, default=0)
    parser.add_argument("--adaptive-cvae-condition-strength", type=int, default=0)
    parser.add_argument("--adaptive-cvae-condition-strength-min", type=float, default=0.03)
    parser.add_argument("--adaptive-cvae-condition-strength-max", type=float, default=1.5)
    parser.add_argument("--adaptive-cvae-condition-strength-init", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-micro-control", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-refine-block", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-supervision", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-heun", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-monotonic-progress", type=int, default=1)
    parser.add_argument("--adaptive-cvae-micro-min-step", type=float, default=0.03)
    parser.add_argument("--adaptive-cvae-micro-max-step", type=float, default=0.35)
    parser.add_argument("--adaptive-cvae-micro-step-init", type=float, default=0.12)
    parser.add_argument("--adaptive-cvae-micro-kp-max", type=float, default=0.6)
    parser.add_argument("--adaptive-cvae-micro-kp-init", type=float, default=0.18)
    parser.add_argument("--adaptive-cvae-micro-kd-max", type=float, default=0.45)
    parser.add_argument("--adaptive-cvae-micro-kd-init", type=float, default=0.08)
    parser.add_argument("--adaptive-cvae-micro-update-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-cvae-micro-refine-block-scale", type=float, default=0.3)
    parser.add_argument("--adaptive-cvae-micro-progress-distance-scale", type=float, default=4.0)

    defaults = V39PolicyTrainerConfig()
    for field in V39PolicyTrainerConfig.__dataclass_fields__:
        value = getattr(defaults, field)
        parser.add_argument("--" + field.replace("_", "-"), dest=field, type=type(value), default=value)
    # The V40 entry point is specifically the multi-layer intervention-latent
    # experiment; inheriting V39's midcut default silently skips that branch.
    parser.set_defaults(contract_mode="layer_adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
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
    min_length = dataset_config.world_horizon + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets)) + 2
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = load_data(
        args,
        min_length=min_length,
        normalizer_mode=(action_norm.mode if action_norm is not None else args.normalizer),
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        splits=legacy_splits,
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
    use_token_prefetch = bool(args.prefetch_dinov2_tokens and args.condition_mode == "dinov2-cache" and hasattr(conditioner, "store"))
    if use_token_prefetch:
        token_store = conditioner.store  # type: ignore[attr-defined]
        train_dataset = CachedTokenPolicyWindowDataset(bases["train"], token_store=token_store, future_anchors=int(args.future_anchors))
        val_dataset = CachedTokenPolicyWindowDataset(bases["val"], token_store=token_store, future_anchors=int(args.future_anchors))
    else:
        train_dataset = PolicyWindowDataset(bases["train"])
        val_dataset = PolicyWindowDataset(bases["val"])
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
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
        arm_flow_mode=args.arm_flow_mode,
        arm_noise_temporal_rho=args.arm_noise_temporal_rho,
        gripper_field_dim=args.gripper_field_dim,
        gripper_field_mode=args.gripper_field_mode,
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
        controlled_base_mode=args.controlled_base_mode,
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
        action_consequence_self_condition=args.action_consequence_self_condition,
        layer_zero_base_diagnostic=args.layer_zero_base_diagnostic,
        final_action_decoder=args.final_action_decoder,
        action_flow_residual_depth=args.action_flow_residual_depth,
        action_flow_residual_high_slots=args.action_flow_residual_high_slots,
        action_flow_residual_max_scale=args.action_flow_residual_max_scale,
        action_flow_residual_visual_memory=args.action_flow_residual_visual_memory,
        action_flow_residual_context_memory=args.action_flow_residual_context_memory,
        action_flow_residual_transition_memory=args.action_flow_residual_transition_memory,
        action_flow_residual_layer_memory=args.action_flow_residual_layer_memory,
        action_flow_residual_layer_pair_schedule=args.action_flow_residual_layer_pair_schedule,
        action_flow_residual_layer_detach=args.action_flow_residual_layer_detach,
        action_flow_residual_stage_router=args.action_flow_residual_stage_router,
        action_flow_residual_anchor_memory=args.action_flow_residual_anchor_memory,
        latent_action_decoder_depth=args.latent_action_decoder_depth,
        latent_action_high_slots=args.latent_action_high_slots,
        latent_action_layer_schedule=args.latent_action_layer_schedule,
        latent_action_visual_memory=args.latent_action_visual_memory,
        latent_action_context_memory=args.latent_action_context_memory,
        latent_action_transition_memory=args.latent_action_transition_memory,
        latent_action_layer_memory=args.latent_action_layer_memory,
        latent_action_anchor_memory=args.latent_action_anchor_memory,
        latent_action_stage_router=args.latent_action_stage_router,
        latent_action_layer_detach=args.latent_action_layer_detach,
        latent_action_event_gripper_gate=args.latent_action_event_gripper_gate,
        latent_action_temporal_depth=args.latent_action_temporal_depth,
        latent_action_near_steps=args.latent_action_near_steps,
        latent_action_mid_steps=args.latent_action_mid_steps,
        latent_action_near_depth=args.latent_action_near_depth,
        latent_action_mid_depth=args.latent_action_mid_depth,
        latent_cvae_z_dim=args.latent_cvae_z_dim,
        latent_cvae_decoder_depth=args.latent_cvae_decoder_depth,
        latent_cvae_ffn_expansion=args.latent_cvae_ffn_expansion,
        latent_cvae_layer_memory=args.latent_cvae_layer_memory,
        latent_cvae_transition_memory=args.latent_cvae_transition_memory,
        latent_cvae_transition_detach=args.latent_cvae_transition_detach,
        latent_cvae_context_memory=args.latent_cvae_context_memory,
        latent_cvae_visual_memory=args.latent_cvae_visual_memory,
        latent_cvae_layer_detach=args.latent_cvae_layer_detach,
        latent_cvae_layer_grad_scale=args.latent_cvae_layer_grad_scale,
        latent_cvae_condition_source_norm=args.latent_cvae_condition_source_norm,
        latent_cvae_bounded_consequence_fusion=args.latent_cvae_bounded_consequence_fusion,
        latent_cvae_consequence_scale_init=args.latent_cvae_consequence_scale_init,
        latent_cvae_consequence_scale_max=args.latent_cvae_consequence_scale_max,
        latent_cvae_event_gripper_gate=args.latent_cvae_event_gripper_gate,
        latent_cvae_inference_sample=args.latent_cvae_inference_sample,
        latent_cvae_output_init_std=args.latent_cvae_output_init_std,
        latent_cvae_mu_bound=args.latent_cvae_mu_bound,
        latent_cvae_min_std=args.latent_cvae_min_std,
        latent_cvae_causal_attention=args.latent_cvae_causal_attention,
        latent_cvae_noisy_gate=args.latent_cvae_noisy_gate,
        latent_cvae_noisy_gate_min=args.latent_cvae_noisy_gate_min,
        latent_cvae_noisy_gate_power=args.latent_cvae_noisy_gate_power,
        latent_cvae_layer_scan=args.latent_cvae_layer_scan,
        latent_cvae_layer_scan_alpha=args.latent_cvae_layer_scan_alpha,
        latent_cvae_mmdit_decoder=args.latent_cvae_mmdit_decoder,
        latent_cvae_mmdit_depth=args.latent_cvae_mmdit_depth,
        latent_cvae_mmdit_cond_update=args.latent_cvae_mmdit_cond_update,
        latent_cvae_mmdit_noisy_causal=args.latent_cvae_mmdit_noisy_causal,
        latent_cvae_mmdit_noisy_logit_gate=args.latent_cvae_mmdit_noisy_logit_gate,
        latent_cvae_progress_action_isolation=args.latent_cvae_progress_action_isolation,
        latent_cvae_horizon_tokens=args.latent_cvae_horizon_tokens,
        latent_cvae_workspace_noisy_query=args.latent_cvae_workspace_noisy_query,
        latent_cvae_workspace_trajectory_source=args.latent_cvae_workspace_trajectory_source,
        latent_cvae_workspace_global_sources=args.latent_cvae_workspace_global_sources,
        latent_cvae_workspace_layer_source=args.latent_cvae_workspace_layer_source,
        latent_cvae_workspace_progress_value=args.latent_cvae_workspace_progress_value,
        latent_cvae_workspace_time_state=args.latent_cvae_workspace_time_state,
        adaptive_cvae_refine_steps=args.adaptive_cvae_refine_steps,
        adaptive_cvae_progress_memory=args.adaptive_cvae_progress_memory,
        adaptive_cvae_progress_steps=args.adaptive_cvae_progress_steps,
        adaptive_cvae_prefix_memory=args.adaptive_cvae_prefix_memory,
        adaptive_cvae_layer_routing=args.adaptive_cvae_layer_routing,
        adaptive_cvae_route_cosine=args.adaptive_cvae_route_cosine,
        adaptive_cvae_route_temperature=args.adaptive_cvae_route_temperature,
        adaptive_cvae_prefix_detach=args.adaptive_cvae_prefix_detach,
        adaptive_cvae_progress_z_injection=args.adaptive_cvae_progress_z_injection,
        adaptive_cvae_route_query_bias=args.adaptive_cvae_route_query_bias,
        adaptive_cvae_token_semantic_adapter=args.adaptive_cvae_token_semantic_adapter,
        adaptive_cvae_output_adapter=args.adaptive_cvae_output_adapter,
        adaptive_cvae_context_dropout=args.adaptive_cvae_context_dropout,
        adaptive_cvae_route_entropy_floor_ratio=args.adaptive_cvae_route_entropy_floor_ratio,
        adaptive_cvae_function_adapters=args.adaptive_cvae_function_adapters,
        adaptive_cvae_function_rank=args.adaptive_cvae_function_rank,
        adaptive_cvae_progress_role_dim=args.adaptive_cvae_progress_role_dim,
        adaptive_cvae_route_topk=args.adaptive_cvae_route_topk,
        adaptive_cvae_route_sparsemax=args.adaptive_cvae_route_sparsemax,
        adaptive_cvae_route_adaptive_temperature=args.adaptive_cvae_route_adaptive_temperature,
        adaptive_cvae_route_min_temperature=args.adaptive_cvae_route_min_temperature,
        adaptive_cvae_route_max_temperature=args.adaptive_cvae_route_max_temperature,
        adaptive_cvae_role_query=args.adaptive_cvae_role_query,
        adaptive_cvae_step_roles=args.adaptive_cvae_step_roles,
        adaptive_cvae_coarse_stride=args.adaptive_cvae_coarse_stride,
        adaptive_cvae_coarse_strength=args.adaptive_cvae_coarse_strength,
        adaptive_cvae_seed_scale=args.adaptive_cvae_seed_scale,
        adaptive_cvae_output_scale=args.adaptive_cvae_output_scale,
        adaptive_cvae_context_capsules=args.adaptive_cvae_context_capsules,
        adaptive_cvae_context_capsule_count=args.adaptive_cvae_context_capsule_count,
        adaptive_cvae_direct_condition_residual=args.adaptive_cvae_direct_condition_residual,
        adaptive_cvae_condition_strength=args.adaptive_cvae_condition_strength,
        adaptive_cvae_condition_strength_min=args.adaptive_cvae_condition_strength_min,
        adaptive_cvae_condition_strength_max=args.adaptive_cvae_condition_strength_max,
        adaptive_cvae_condition_strength_init=args.adaptive_cvae_condition_strength_init,
        adaptive_cvae_micro_control=args.adaptive_cvae_micro_control,
        adaptive_cvae_micro_refine_block=args.adaptive_cvae_micro_refine_block,
        adaptive_cvae_micro_supervision=args.adaptive_cvae_micro_supervision,
        adaptive_cvae_micro_heun=args.adaptive_cvae_micro_heun,
        adaptive_cvae_micro_monotonic_progress=args.adaptive_cvae_micro_monotonic_progress,
        adaptive_cvae_micro_min_step=args.adaptive_cvae_micro_min_step,
        adaptive_cvae_micro_max_step=args.adaptive_cvae_micro_max_step,
        adaptive_cvae_micro_step_init=args.adaptive_cvae_micro_step_init,
        adaptive_cvae_micro_kp_max=args.adaptive_cvae_micro_kp_max,
        adaptive_cvae_micro_kp_init=args.adaptive_cvae_micro_kp_init,
        adaptive_cvae_micro_kd_max=args.adaptive_cvae_micro_kd_max,
        adaptive_cvae_micro_kd_init=args.adaptive_cvae_micro_kd_init,
        adaptive_cvae_micro_update_scale=args.adaptive_cvae_micro_update_scale,
        adaptive_cvae_micro_refine_block_scale=args.adaptive_cvae_micro_refine_block_scale,
        adaptive_cvae_micro_progress_distance_scale=args.adaptive_cvae_micro_progress_distance_scale,
    )
    system = V39PolicySystem(policy_config)
    if args.stage1_checkpoint is not None:
        stage_payload = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        if stage_payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
            raise ValueError("--stage1-checkpoint must be a V39/V40 checkpoint")
        stage_state, skipped_stage_keys = _filter_stage1_state_dict(
            stage_payload["model"],
            reset_dirty_adapters=bool(args.stage1_reset_dirty_adapters),
        )
        if skipped_stage_keys:
            print(
                f"[v39-init] skipped dirty stage1 adapter keys: "
                f"{skipped_stage_keys[:8]} count={len(skipped_stage_keys)}",
                flush=True,
            )
        stage_state, skipped_parseval_keys = _filter_parseval_replaced_state_dict(
            stage_state,
            enabled=str(args.gripper_field_mode) == "parseval_temporal",
        )
        if skipped_parseval_keys:
            print(
                f"[v39-init] skipped replaced Parseval-interface keys: "
                f"{skipped_parseval_keys[:8]} count={len(skipped_parseval_keys)}",
                flush=True,
            )
        stage_state, skipped_shape_keys = _filter_shape_mismatched_state_dict(stage_state, system.state_dict())
        if skipped_shape_keys:
            print(
                f"[v39-init] skipped shape-mismatched stage1 keys: "
                f"{skipped_shape_keys[:8]} count={len(skipped_shape_keys)}",
                flush=True,
            )
        missing, unexpected = system.load_state_dict(stage_state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys while loading --stage1-checkpoint: {unexpected[:8]}")
        if missing:
            print(f"[v39-init] missing keys from stage1 checkpoint: {missing[:8]} count={len(missing)}", flush=True)
    trainer = V39PolicyTrainerConfig(**{name: getattr(args, name) for name in V39PolicyTrainerConfig.__dataclass_fields__})
    context = {
        "schema": "clearvla-v40-1-unified-intervention-latent-context-v1",
        "args": vars(args),
        "legacy_context_checkpoint": None if args.legacy_context_checkpoint is None else str(args.legacy_context_checkpoint),
        "stage1_checkpoint": None if args.stage1_checkpoint is None else str(args.stage1_checkpoint),
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
            "final_action_decoder": str(args.final_action_decoder),
            "gripper_field_mode": str(args.gripper_field_mode),
            "gripper_field_dim": int(args.gripper_field_dim),
            "gripper_parseval_shared_native_noise": str(args.gripper_field_mode) == "parseval_temporal",
            "gripper_parseval_all_channels_decode": str(args.gripper_field_mode) == "parseval_temporal",
            "arm_flow_mode": str(args.arm_flow_mode),
            "arm_noise_temporal_rho": float(args.arm_noise_temporal_rho),
            "action_normalizer_fingerprint": _normalizer_fingerprint(action_norm),
            "arm_manifold_native_noise": str(args.arm_flow_mode) == "manifold_native",
            "arm_manifold_projected_sampling": str(args.arm_flow_mode) == "manifold_native",
            "residual_action_flow_safe_start": str(args.final_action_decoder) in {"residual_action_flow", "layered_residual_action_flow"},
            "residual_action_flow_uses_v37_high_action_event_tokens": str(args.final_action_decoder) in {"residual_action_flow", "layered_residual_action_flow"},
            "layered_residual_action_flow_layer_pair_injection": str(args.final_action_decoder) == "layered_residual_action_flow",
            "latent_main_action_decoder": str(args.final_action_decoder) == "latent_main_action",
            "latent_main_action_single_final_path": str(args.final_action_decoder) == "latent_main_action",
            "latent_main_action_every_layer_summary_injected": str(args.final_action_decoder) == "latent_main_action",
            "latent_main_action_no_legacy_velocity_base": str(args.final_action_decoder) == "latent_main_action",
            "latent_main_action_horizon_dependent_depth": bool(int(args.latent_action_temporal_depth)) and str(args.final_action_decoder) == "latent_main_action",
            "latent_cvae_action_decoder": str(args.final_action_decoder) in {"latent_cvae_action", "adaptive_recurrent_cvae_action"},
            "latent_cvae_single_final_path": str(args.final_action_decoder) in {"latent_cvae_action", "adaptive_recurrent_cvae_action"},
            "latent_cvae_no_legacy_velocity_base": str(args.final_action_decoder) in {"latent_cvae_action", "adaptive_recurrent_cvae_action"},
            "adaptive_recurrent_cvae_action_decoder": str(args.final_action_decoder) == "adaptive_recurrent_cvae_action",
            "latent_cvae_mmdit_decoder": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_mmdit_no_direct_noisy_residual": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_mmdit_condition_update": bool(int(args.latent_cvae_mmdit_cond_update)),
            "latent_cvae_mmdit_noisy_causal": bool(int(args.latent_cvae_mmdit_noisy_causal)),
            "latent_cvae_horizon_tokens": int(args.latent_cvae_horizon_tokens),
            "latent_cvae_workspace_noisy_query": bool(int(args.latent_cvae_workspace_noisy_query)),
            "latent_cvae_workspace_trajectory_source": bool(int(args.latent_cvae_workspace_trajectory_source)),
            "latent_cvae_workspace_global_sources": bool(int(args.latent_cvae_workspace_global_sources)),
            "latent_cvae_workspace_layer_source": bool(int(args.latent_cvae_workspace_layer_source)),
            "latent_cvae_workspace_progress_value": bool(int(args.latent_cvae_workspace_progress_value)),
            "latent_cvae_workspace_time_state": bool(int(args.latent_cvae_workspace_time_state)),
            "latent_cvae_z_primary_denoising": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_typed_evidence_workspace": bool(int(args.latent_cvae_mmdit_decoder)),
            "latent_cvae_single_workspace_action_write": bool(int(args.latent_cvae_mmdit_decoder)),
            "adaptive_cvae_refine_steps": int(args.adaptive_cvae_refine_steps),
            "adaptive_cvae_progress_memory": int(args.adaptive_cvae_progress_memory),
            "adaptive_cvae_progress_steps": int(args.adaptive_cvae_progress_steps),
            "adaptive_cvae_prefix_memory": int(args.adaptive_cvae_prefix_memory),
            "adaptive_cvae_layer_routing": int(args.adaptive_cvae_layer_routing),
            "adaptive_cvae_route_cosine": int(args.adaptive_cvae_route_cosine),
            "adaptive_cvae_route_temperature": float(args.adaptive_cvae_route_temperature),
            "adaptive_cvae_prefix_detach": int(args.adaptive_cvae_prefix_detach),
            "adaptive_cvae_progress_z_injection": int(args.adaptive_cvae_progress_z_injection),
            "adaptive_cvae_route_query_bias": int(args.adaptive_cvae_route_query_bias),
            "adaptive_cvae_token_semantic_adapter": int(args.adaptive_cvae_token_semantic_adapter),
            "adaptive_cvae_output_adapter": int(args.adaptive_cvae_output_adapter),
            "adaptive_cvae_context_dropout": float(args.adaptive_cvae_context_dropout),
            "adaptive_cvae_route_entropy_floor_ratio": float(args.adaptive_cvae_route_entropy_floor_ratio),
            "adaptive_cvae_function_adapters": int(args.adaptive_cvae_function_adapters),
            "adaptive_cvae_function_rank": int(args.adaptive_cvae_function_rank),
            "adaptive_cvae_progress_role_dim": int(args.adaptive_cvae_progress_role_dim),
            "adaptive_cvae_route_topk": int(args.adaptive_cvae_route_topk),
            "adaptive_cvae_route_sparsemax": int(args.adaptive_cvae_route_sparsemax),
            "adaptive_cvae_route_adaptive_temperature": int(args.adaptive_cvae_route_adaptive_temperature),
            "adaptive_cvae_route_min_temperature": float(args.adaptive_cvae_route_min_temperature),
            "adaptive_cvae_route_max_temperature": float(args.adaptive_cvae_route_max_temperature),
            "adaptive_cvae_role_query": int(args.adaptive_cvae_role_query),
            "adaptive_cvae_step_roles": int(args.adaptive_cvae_step_roles),
            "adaptive_cvae_coarse_stride": int(args.adaptive_cvae_coarse_stride),
            "adaptive_cvae_coarse_strength": float(args.adaptive_cvae_coarse_strength),
            "adaptive_cvae_seed_scale": float(args.adaptive_cvae_seed_scale),
            "adaptive_cvae_output_scale": float(args.adaptive_cvae_output_scale),
            "adaptive_cvae_context_capsules": int(args.adaptive_cvae_context_capsules),
            "adaptive_cvae_context_capsule_count": int(args.adaptive_cvae_context_capsule_count),
            "adaptive_cvae_direct_condition_residual": int(args.adaptive_cvae_direct_condition_residual),
            "adaptive_cvae_condition_strength": int(args.adaptive_cvae_condition_strength),
            "adaptive_cvae_condition_strength_min": float(args.adaptive_cvae_condition_strength_min),
            "adaptive_cvae_condition_strength_max": float(args.adaptive_cvae_condition_strength_max),
            "adaptive_cvae_condition_strength_init": float(args.adaptive_cvae_condition_strength_init),
            "adaptive_cvae_micro_control": int(args.adaptive_cvae_micro_control),
            "adaptive_cvae_micro_refine_block": int(args.adaptive_cvae_micro_refine_block),
            "adaptive_cvae_micro_supervision": int(args.adaptive_cvae_micro_supervision),
            "adaptive_cvae_micro_heun": int(args.adaptive_cvae_micro_heun),
            "adaptive_cvae_micro_monotonic_progress": int(args.adaptive_cvae_micro_monotonic_progress),
            "adaptive_cvae_micro_min_step": float(args.adaptive_cvae_micro_min_step),
            "adaptive_cvae_micro_max_step": float(args.adaptive_cvae_micro_max_step),
            "adaptive_cvae_micro_step_init": float(args.adaptive_cvae_micro_step_init),
            "adaptive_cvae_micro_kp_max": float(args.adaptive_cvae_micro_kp_max),
            "adaptive_cvae_micro_kp_init": float(args.adaptive_cvae_micro_kp_init),
            "adaptive_cvae_micro_kd_max": float(args.adaptive_cvae_micro_kd_max),
            "adaptive_cvae_micro_kd_init": float(args.adaptive_cvae_micro_kd_init),
            "adaptive_cvae_micro_update_scale": float(args.adaptive_cvae_micro_update_scale),
            "adaptive_cvae_micro_refine_block_scale": float(args.adaptive_cvae_micro_refine_block_scale),
            "adaptive_cvae_micro_progress_distance_scale": float(args.adaptive_cvae_micro_progress_distance_scale),
            "latent_cvae_z_dim": int(args.latent_cvae_z_dim),
            "latent_cvae_decoder_depth": int(args.latent_cvae_decoder_depth),
            "latent_cvae_layer_memory": int(args.latent_cvae_layer_memory),
            "latent_cvae_transition_memory": int(args.latent_cvae_transition_memory),
            "latent_cvae_transition_detach": int(args.latent_cvae_transition_detach),
            "latent_cvae_layer_grad_scale": float(args.latent_cvae_layer_grad_scale),
            "latent_cvae_visual_memory": int(args.latent_cvae_visual_memory),
            "latent_cvae_context_memory": int(args.latent_cvae_context_memory),
            "latent_cvae_mu_bound": float(args.latent_cvae_mu_bound),
            "latent_cvae_min_std": float(args.latent_cvae_min_std),
            "latent_cvae_causal_attention": int(args.latent_cvae_causal_attention),
            "latent_cvae_noisy_gate": int(args.latent_cvae_noisy_gate),
            "latent_cvae_noisy_gate_min": float(args.latent_cvae_noisy_gate_min),
            "latent_cvae_noisy_gate_power": float(args.latent_cvae_noisy_gate_power),
            "latent_cvae_layer_scan": int(args.latent_cvae_layer_scan),
            "latent_cvae_layer_scan_alpha": float(args.latent_cvae_layer_scan_alpha),
            "latent_cvae_mmdit_decoder": int(args.latent_cvae_mmdit_decoder),
            "latent_cvae_mmdit_depth": int(args.latent_cvae_mmdit_depth),
            "latent_cvae_mmdit_cond_update": int(args.latent_cvae_mmdit_cond_update),
            "latent_cvae_mmdit_noisy_causal": int(args.latent_cvae_mmdit_noisy_causal),
            "latent_cvae_horizon_tokens": int(args.latent_cvae_horizon_tokens),
            "latent_cvae_workspace_noisy_query": int(args.latent_cvae_workspace_noisy_query),
            "latent_cvae_workspace_trajectory_source": int(args.latent_cvae_workspace_trajectory_source),
            "latent_cvae_workspace_global_sources": int(args.latent_cvae_workspace_global_sources),
            "latent_cvae_workspace_layer_source": int(args.latent_cvae_workspace_layer_source),
            "latent_cvae_workspace_progress_value": int(args.latent_cvae_workspace_progress_value),
            "latent_cvae_workspace_time_state": int(args.latent_cvae_workspace_time_state),
            "latent_action_decoder_depth": int(args.latent_action_decoder_depth),
            "latent_action_layer_schedule": str(args.latent_action_layer_schedule),
            "latent_action_visual_memory": int(args.latent_action_visual_memory),
            "latent_action_context_memory": int(args.latent_action_context_memory),
            "latent_action_transition_memory": int(args.latent_action_transition_memory),
            "latent_action_event_gripper_gate": int(args.latent_action_event_gripper_gate),
            "latent_action_temporal_depth": int(args.latent_action_temporal_depth),
            "latent_action_near_steps": int(args.latent_action_near_steps),
            "latent_action_mid_steps": int(args.latent_action_mid_steps),
            "latent_action_near_depth": int(args.latent_action_near_depth),
            "latent_action_mid_depth": int(args.latent_action_mid_depth),
            "action_flow_residual_depth": int(args.action_flow_residual_depth),
            "action_flow_residual_max_scale": float(args.action_flow_residual_max_scale),
            "action_flow_residual_layer_pair_schedule": str(args.action_flow_residual_layer_pair_schedule),
            "action_flow_residual_layer_detach": bool(args.action_flow_residual_layer_detach),
            "action_flow_residual_stage_router": bool(args.action_flow_residual_stage_router),
            "midcut_layer": int(args.midcut_layer),
            "layer_contract_adapters": int(args.layer_contract_adapters),
            "staged_midcut_contract": True,
            "v39_1_multi_layer_adapter_contract": bool(args.layer_contract_adapters),
            "v39_2_multi_layer_latent_head_shared_fm_probe": bool(args.layer_shared_fm_probe),
            "v39_3_recurrent_milestone_consequence": bool(args.layer_recurrent_consequence),
            "v40_layer_causal_latent_split": False,
            "v40_1_unified_intervention_latent": True,
            "v40_context_fusion_at_zero_feedback_depth": True,
            "v40_milestone_anchor_alignment": "required_equal",
            "v40_state_counterfactual_enabled": bool(args.layer_state_counterfactual),
            "action_consequence_self_condition": bool(args.action_consequence_self_condition),
            "layer_zero_base_diagnostic": bool(args.layer_zero_base_diagnostic),
            "v40_hard_action_negative": "within_batch_farthest_action",
            "v39_3_consequence_steps": int(args.layer_consequence_steps),
            "v40_causal_feedback_depth": int(args.layer_causal_feedback_depth),
            "metric_sync": "log_every_and_epoch_end",
            "dino_cache_reads": "episode_grouped_mmap",
        },
        "skipped": skipped,
        "policy_contract": (
            "V40.1 is independent of old world checkpoints by default. It keeps the V39 staged training contract but uses one unified intervention-latent head at every supervised layer. "
            "Explicit state/history/action context is fused into the FiLM/gate path even when feedback depth is zero. Counterfactual contrast uses hold-action and within-batch "
            "farthest-action negatives; the non-strict state/frame shuffle remains disabled by default."
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
