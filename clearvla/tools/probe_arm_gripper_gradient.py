"""Measure arm/gripper gradient interaction on a trained mainline checkpoint.

The probe is deliberately outside the training hot path.  It replays one real
training batch with the serialized model/data contract, takes no optimizer
step, and never populates persistent ``.grad`` tensors.  The reported VJPs are
restricted to shared parameter owners; terminal arm/gripper output parameters
are excluded so disjoint heads cannot manufacture a misleading zero cosine.

For a continuous seven-dimensional outlet, the action objective is separated
into the terms whose semantics are unambiguous:

* arm: direct flow, decoded action, adjacent-delta consistency, physical-field
  delta consistency, and the arm-motion auxiliary;
* gripper: direct flow, decoded action, adjacent-delta consistency,
  physical-field delta consistency, and the dedicated gripper trajectory.

The history-proposal term is left unassigned because it is a separate owner
and has no route to the selected shared bottom parameters.  Representation and
execution-value groups are also outside this semantic-channel comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clearvla.mainline.checkpoint import (
    build_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.action_codec import anchor_horizon_weights
try:
    from clearvla.mainline.model.component_contracts import legacy_named_parameters
except ImportError:  # Schema-30 recovery checkout predates the name-map helper.
    def legacy_named_parameters(model: nn.Module):
        return tuple(model.named_parameters())
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy, PolicyStepOutput
from clearvla.mainline.runtime.identity import (
    dataset_identity,
    language_identity,
    v120_normalizer_fingerprint,
)
from clearvla.mainline.runtime.checkpoints import load_checkpoint_for_validation
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training.losses import (
    FlowMatchingState,
    LossLedger,
    compose_losses,
    sample_flow_matching,
)
from clearvla.mainline.training.optimizer import parameter_role


REPORT_SCHEMA = "clearvla-arm-gripper-shared-gradient-v1"
DEFAULT_OWNER_ROLES = (
    "bottom_query",
    "bottom_policy_bridge",
    "bottom_organizer",
    "bottom_mmdit",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe-seed", type=int, default=20260909)
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "physical probe batch; defaults to the serialized training batch "
            "without changing checkpoint/config identity"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="probe DataLoader workers; defaults to the serialized value",
    )
    parser.add_argument(
        "--batch-offset",
        type=int,
        default=0,
        help="zero-based batch offset in the deterministic shuffled train loader",
    )
    parser.add_argument(
        "--owner-role",
        action="append",
        dest="owner_roles",
        help=(
            "shared optimizer role to inspect; repeat to select multiple roles "
            f"(default: {','.join(DEFAULT_OWNER_ROLES)})"
        ),
    )
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _owned_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _autocast(device: torch.device, dtype: torch.dtype):
    enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def _scalar(value: Tensor) -> float:
    return float(value.detach().float().cpu())


def _write_once(path: Path, rendered: str) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite probe report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _repository_state() -> dict[str, object]:
    # A read-only copy of this diagnostic is sometimes staged under /tmp on a
    # remote host.  Let the caller pin the source checkout explicitly so the
    # reported digest still refers to the checkpoint's repository, not /tmp.
    configured_root = os.environ.get("CLEARVLA_REPO_ROOT")
    root = (
        Path(configured_root)
        if configured_root
        else Path(__file__).resolve().parents[2]
    ).expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_dirty": bool(status),
        "worktree_status_rows": len(status),
    }


def _batch_identity(batch: TrainingBatch) -> dict[str, object]:
    def values(value: Tensor | None) -> list[float | int] | None:
        if value is None:
            return None
        return list(value.detach().cpu().tolist())

    return {
        "batch_size": int(batch.online.batch),
        "sample_indices": values(batch.audit.sample_index),
        "episode_indices": values(batch.audit.episode_index),
        "frame_progress": values(batch.audit.frame_progress),
    }


def _selected_parameters(
    model: nn.Module,
    owner_roles: Sequence[str],
) -> tuple[
    tuple[tuple[str, nn.Parameter], ...],
    dict[str, tuple[int, ...]],
]:
    requested = tuple(dict.fromkeys(str(role) for role in owner_roles))
    if not requested:
        raise ValueError("at least one shared owner role is required")
    if "bottom_heads" in requested:
        raise ValueError(
            "bottom_heads is intentionally excluded: its disjoint output "
            "coordinates would make shared-gradient cosine misleading"
        )
    rows: list[tuple[str, nn.Parameter]] = []
    role_indices: dict[str, list[int]] = {role: [] for role in requested}
    seen: set[int] = set()
    for name, parameter in legacy_named_parameters(model):
        if not parameter.requires_grad:
            continue
        role = parameter_role(name)
        if role not in role_indices:
            continue
        if id(parameter) in seen:
            raise RuntimeError("selected parameter inventory contains an alias")
        seen.add(id(parameter))
        role_indices[role].append(len(rows))
        rows.append((name, parameter))
    missing = tuple(role for role, indices in role_indices.items() if not indices)
    if missing:
        raise ValueError("selected shared owners are absent: " + ", ".join(missing))
    return tuple(rows), {
        role: tuple(indices) for role, indices in role_indices.items()
    }


def _semantic_action_losses(
    config: ExperimentConfig,
    model: ClearVLAMainlinePolicy,
    batch: TrainingBatch,
    output: PolicyStepOutput,
    flow_state: FlowMatchingState,
    ledger: LossLedger,
) -> dict[str, Tensor]:
    codec = getattr(model, "outlet_adapter", None)
    if codec is None:
        # Schema-30 recovery names the shared continuous codec directly.
        codec = model.action_codec
    if bool(getattr(codec, "is_binary_command", False)):
        raise ValueError(
            "this semantic decomposition is for continuous arm/gripper outlets"
        )
    arm_dim = int(codec.arm_dim)
    action_dim = arm_dim + 1
    prediction = output.bottom.physical_velocity.float()
    residual = prediction - flow_state.target_physical_velocity.detach().float()
    residual_parts = codec.split(residual)
    horizon_weight = anchor_horizon_weights(
        horizon=config.dimensions.action_horizon,
        tail_emphasis=config.objectives.horizon_tail_emphasis,
        first_step_protection=config.objectives.horizon_first_step_protection,
        device=prediction.device,
    )[None]

    arm_flow_rows = 0.5 * (
        residual_parts.arm_absolute.square() + residual_parts.arm_delta.square()
    )
    gripper_flow_rows = residual_parts.gripper_field.square().mean(dim=-1)
    arm_flow = (
        arm_flow_rows.sum(dim=-1) / float(action_dim) * horizon_weight
    ).mean()
    # The deployed continuous gripper decode reads only field channels 0/1
    # (absolute and local delta).  Channels 2: are compatibility coordinates
    # retained by the 18-D ABI; they still receive the direct field objective.
    # Keep the exact 1/6 weighting of ``gripper_flow`` so these two diagnostics
    # add back to the original direct gripper term.
    gripper_deployed_flow = (
        residual_parts.gripper_field[..., :2].square().sum(dim=-1)
        / 6.0
        / float(action_dim)
        * horizon_weight
    ).mean()
    gripper_compatibility_flow = (
        residual_parts.gripper_field[..., 2:].square().sum(dim=-1)
        / 6.0
        / float(action_dim)
        * horizon_weight
    ).mean()
    gripper_flow = (
        gripper_flow_rows / float(action_dim) * horizon_weight
    ).mean()

    remaining = (1.0 - flow_state.time.float())[:, None, None]
    clean_physical = flow_state.noisy_physical.float() + remaining * prediction
    codec_gripper_boundary = batch.online.history.codec_gripper_boundary.float()
    decoded = codec.decode(
        clean_physical,
        batch.online.history.action_state.float(),
        codec_gripper_boundary=codec_gripper_boundary,
    )
    target = batch.action_target.normalized.float()
    decoded_error = F.smooth_l1_loss(decoded, target, reduction="none")
    decoded_arm = (
        decoded_error[..., :arm_dim].sum(dim=-1)
        / float(action_dim)
        * horizon_weight
    ).mean()
    decoded_gripper = (
        decoded_error[..., -1] / float(action_dim) * horizon_weight
    ).mean()

    transition_start = torch.cat(
        (
            batch.online.history.action_state[:, :arm_dim].float(),
            codec_gripper_boundary,
        ),
        dim=-1,
    )
    target_boundary = torch.cat(
        (transition_start[:, None], target[:, :-1]),
        dim=1,
    )
    target_delta = target - target_boundary
    predicted_boundary = torch.cat(
        (transition_start[:, None], decoded[:, :-1]),
        dim=1,
    )
    predicted_delta = decoded - predicted_boundary
    smooth_error = F.smooth_l1_loss(
        predicted_delta,
        target_delta,
        reduction="none",
    )
    smooth_arm = (
        smooth_error[..., :arm_dim].sum(dim=-1)
        / float(action_dim)
        * horizon_weight
    ).mean()
    smooth_gripper = (
        smooth_error[..., -1] / float(action_dim) * horizon_weight
    ).mean()

    clean_parts = codec.split(clean_physical)
    field_delta = torch.cat(
        (clean_parts.arm_delta, clean_parts.gripper_field[..., 1:2]),
        dim=-1,
    )
    consistency_error = F.smooth_l1_loss(
        predicted_delta,
        field_delta,
        reduction="none",
    )
    consistency_arm = (
        consistency_error[..., :arm_dim].sum(dim=-1)
        / float(action_dim)
        * horizon_weight
    ).mean()
    consistency_gripper = (
        consistency_error[..., -1] / float(action_dim) * horizon_weight
    ).mean()

    objective = config.objectives
    arm_direct = arm_flow
    gripper_direct = gripper_flow
    arm_full = (
        arm_direct
        + objective.decoded_action * decoded_arm
        + objective.smooth_delta * smooth_arm
        + objective.physical_delta_consistency * consistency_arm
        + objective.motion * ledger.terms["motion"]
    )
    gripper_full = (
        gripper_direct
        + objective.decoded_action * decoded_gripper
        + objective.smooth_delta * smooth_gripper
        + objective.physical_delta_consistency * consistency_gripper
        + objective.gripper_trajectory * ledger.terms["gripper_trajectory"]
    )
    return {
        "arm_direct": arm_direct,
        "gripper_direct": gripper_direct,
        "gripper_deployed_direct": gripper_deployed_flow,
        "gripper_compatibility_direct": gripper_compatibility_flow,
        "arm_full": arm_full,
        "gripper_full": gripper_full,
        "arm_decoded": objective.decoded_action * decoded_arm,
        "gripper_decoded": objective.decoded_action * decoded_gripper,
        "arm_smooth_delta": objective.smooth_delta * smooth_arm,
        "gripper_smooth_delta": objective.smooth_delta * smooth_gripper,
        "arm_physical_delta_consistency": (
            objective.physical_delta_consistency * consistency_arm
        ),
        "gripper_physical_delta_consistency": (
            objective.physical_delta_consistency * consistency_gripper
        ),
        "arm_motion": objective.motion * ledger.terms["motion"],
        "gripper_trajectory": (
            objective.gripper_trajectory * ledger.terms["gripper_trajectory"]
        ),
    }


def _vjp_to_cpu(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[Tensor | None, ...]:
    if loss.ndim != 0 or not loss.requires_grad or not bool(torch.isfinite(loss)):
        raise ValueError("VJP loss must be a finite differentiable scalar")
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    result = tuple(
        None if value is None else value.detach().float().cpu()
        for value in gradients
    )
    del gradients
    return result


def _gradient_pair_stats(
    indices: Iterable[int],
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
    *,
    right_subtract: Sequence[Tensor | None] | None = None,
) -> dict[str, float | int | str | None]:
    left_square = 0.0
    right_square = 0.0
    dot = 0.0
    tensor_count = 0
    joint_tensor_count = 0
    element_count = 0
    for index in indices:
        left_value = left[index]
        right_value = right[index]
        subtract_value = None if right_subtract is None else right_subtract[index]
        if right_value is None and subtract_value is None:
            derived_right = None
        elif right_value is None:
            derived_right = -subtract_value
        elif subtract_value is None:
            derived_right = right_value
        else:
            derived_right = right_value - subtract_value
        reference = left_value if left_value is not None else derived_right
        if reference is not None:
            tensor_count += 1
            element_count += int(reference.numel())
        if left_value is not None:
            left_square += float(left_value.square().sum())
        if derived_right is not None:
            right_square += float(derived_right.square().sum())
        if left_value is not None and derived_right is not None:
            joint_tensor_count += 1
            dot += float((left_value * derived_right).sum())
    left_l2 = math.sqrt(max(left_square, 0.0))
    right_l2 = math.sqrt(max(right_square, 0.0))
    combined_square = max(left_square + right_square + 2.0 * dot, 0.0)
    combined_l2 = math.sqrt(combined_square)
    cosine = None
    projection_over_left = None
    signed_projection_l2 = None
    retained_factor = None
    combined_alignment = None
    if left_l2 > 0.0 and right_l2 > 0.0:
        cosine = dot / (left_l2 * right_l2)
        signed_projection_l2 = dot / left_l2
    if left_square > 0.0:
        projection_over_left = dot / left_square
        retained_factor = 1.0 + projection_over_left
    if left_l2 > 0.0 and combined_l2 > 0.0:
        combined_alignment = (left_square + dot) / (left_l2 * combined_l2)
    if right_l2 == 0.0:
        classification = "right_gradient_zero"
    elif dot < 0.0:
        classification = "opposing"
    elif dot > 0.0:
        classification = "aligned"
    else:
        classification = "orthogonal_or_disjoint"
    return {
        "left_gradient_l2": left_l2,
        "right_gradient_l2": right_l2,
        "right_over_left_l2": (
            None if left_l2 == 0.0 else right_l2 / left_l2
        ),
        "dot": dot,
        "cosine": cosine,
        "right_signed_projection_on_left_l2": signed_projection_l2,
        "right_projection_over_left_norm": projection_over_left,
        "left_direction_retained_factor_after_sum": retained_factor,
        "combined_gradient_l2": combined_l2,
        "combined_alignment_with_left": combined_alignment,
        "classification": classification,
        "parameter_tensors": tensor_count,
        "joint_parameter_tensors": joint_tensor_count,
        "parameter_elements": element_count,
    }


def _loss_report(losses: Mapping[str, Tensor], ledger: LossLedger) -> dict[str, float]:
    result = {name: _scalar(value) for name, value in losses.items()}
    result.update(
        {
            "formal_action_group": _scalar(ledger.groups["action"]),
            "formal_action_flow": _scalar(ledger.contributions["action_flow"]),
            "formal_decoded_action": _scalar(ledger.contributions["decoded_action"]),
            "formal_smooth_delta": _scalar(ledger.contributions["smooth_delta"]),
            "formal_physical_delta_consistency": _scalar(
                ledger.contributions["physical_delta_consistency"]
            ),
            "formal_motion": _scalar(ledger.contributions["motion"]),
            "formal_gripper_trajectory": _scalar(
                ledger.contributions["gripper_trajectory"]
            ),
            "formal_proposal": _scalar(ledger.contributions["proposal"]),
        }
    )
    result["action_flow_semantic_gap"] = result["formal_action_flow"] - (
        result["arm_direct"] + result["gripper_direct"]
    )
    result["action_group_unassigned_gap"] = result["formal_action_group"] - (
        result["arm_full"] + result["gripper_full"]
    )
    return result


def run_probe(
    *,
    model: ClearVLAMainlinePolicy,
    config: ExperimentConfig,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    owner_roles: Sequence[str],
    flow_generator: torch.Generator,
    condition_generator: torch.Generator,
    training_step: int,
    probe_seed: int,
) -> dict[str, object]:
    config.validate()
    batch.validate(config)
    model.train()
    model.set_training_step(int(training_step))
    model.zero_grad(set_to_none=True)
    selected, role_indices = _selected_parameters(model, owner_roles)
    parameters = tuple(parameter for _, parameter in selected)
    _seed(probe_seed)

    with _autocast(device, dtype):
        cache, training_state, _ = model.encode_online(
            batch.online,
            training_mask=True,
            collect_diagnostics=False,
            condition_generator=condition_generator,
        )
        top_targets, _ = model.build_training_targets(
            training_state,
            batch.future,
            collect_diagnostics=False,
        )
        outlet_adapter = getattr(model, "outlet_adapter", None)
        if outlet_adapter is None:
            outlet_adapter = model.action_codec
        flow_state = sample_flow_matching(
            batch.action_target.normalized,
            action_state=batch.online.history.action_state,
            codec_gripper_boundary=batch.online.history.codec_gripper_boundary,
            codec=outlet_adapter,
            distribution=config.bottom.flow_time_distribution,
            generator=flow_generator,
        )
        output = model.velocity(
            cache,
            noisy_action_field=flow_state.noisy_physical,
            time=flow_state.time,
            require_execution_supervision=True,
            collect_diagnostics=False,
        )
        ledger = compose_losses(
            config,
            policy_output=output,
            action_target=batch.action_target,
            history=batch.online.history,
            flow_state=flow_state,
            observation=training_state.observation,
            top_targets=top_targets,
            predicted_dynamics=cache.top.predicted_dynamics,
            action_codec=outlet_adapter,
            collect_diagnostics=False,
        )
        losses = _semantic_action_losses(
            config,
            model,
            batch,
            output,
            flow_state,
            ledger,
        )

    arm_direct_gradients = _vjp_to_cpu(
        losses["arm_direct"],
        parameters,
        retain_graph=True,
    )
    gripper_direct_gradients = _vjp_to_cpu(
        losses["gripper_direct"],
        parameters,
        retain_graph=True,
    )
    gripper_deployed_direct_gradients = _vjp_to_cpu(
        losses["gripper_deployed_direct"],
        parameters,
        retain_graph=True,
    )
    gripper_compatibility_direct_gradients = _vjp_to_cpu(
        losses["gripper_compatibility_direct"],
        parameters,
        retain_graph=True,
    )
    arm_full_gradients = _vjp_to_cpu(
        losses["arm_full"],
        parameters,
        retain_graph=True,
    )
    gripper_full_gradients = _vjp_to_cpu(
        losses["gripper_full"],
        parameters,
        retain_graph=False,
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only VJP probe unexpectedly populated .grad")

    all_indices = tuple(range(len(selected)))
    comparisons: dict[str, object] = {}
    for role, indices in (*role_indices.items(), ("selected_shared_total", all_indices)):
        comparisons[role] = {
            "direct_arm_vs_direct_gripper": _gradient_pair_stats(
                indices,
                arm_direct_gradients,
                gripper_direct_gradients,
            ),
            "deployed_gripper_vs_compatibility_gripper": _gradient_pair_stats(
                indices,
                gripper_deployed_direct_gradients,
                gripper_compatibility_direct_gradients,
            ),
            "full_arm_vs_full_gripper": _gradient_pair_stats(
                indices,
                arm_full_gradients,
                gripper_full_gradients,
            ),
            "full_arm_vs_gripper_auxiliary_increment": _gradient_pair_stats(
                indices,
                arm_full_gradients,
                gripper_full_gradients,
                right_subtract=gripper_direct_gradients,
            ),
        }
    event_rate = ledger.terms["action_gripper_event_rate"]
    return {
        "batch": _batch_identity(batch),
        "flow_state": {
            "time_mean": _scalar(flow_state.time.mean()),
            "time_min": _scalar(flow_state.time.amin()),
            "time_max": _scalar(flow_state.time.amax()),
            "gripper_event_row_rate": _scalar(event_rate),
            "gripper_trajectory_mask_fraction": _scalar(
                ledger.terms["gripper_trajectory_mask_fraction"]
            ),
        },
        "losses": _loss_report(losses, ledger),
        "owners": {
            "roles": list(owner_roles),
            "parameter_tensors": len(selected),
            "parameter_elements": sum(
                int(parameter.numel()) for _, parameter in selected
            ),
            "terminal_heads_excluded": True,
        },
        "comparisons": comparisons,
        "scope": {
            "optimizer_constructed": False,
            "optimizer_step_taken": False,
            "persistent_gradients_written": False,
            "training_mode": True,
            "batches": 1,
            "semantic_assignment": (
                "action-regression terms only; representation and execution-value "
                "groups are not attributed to arm or gripper"
            ),
        },
    }


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    supported_profiles = {"libero_relative_7d_v1", "identity_7d_pen"}
    if str(config.data.data_profile) not in supported_profiles:
        raise ValueError(
            "the requested focused probe expects a continuous seven-dimensional "
            f"profile, got {config.data.data_profile!r}"
        )
    if int(config.dimensions.action_dim) != 7:
        raise ValueError("the focused semantic decomposition expects action_dim=7")
    owner_roles = tuple(args.owner_roles or DEFAULT_OWNER_ROLES)
    probe_batch_size = (
        int(config.optimizer.batch_size)
        if args.batch_size is None
        else int(args.batch_size)
    )
    probe_workers = (
        int(config.data.num_workers)
        if args.num_workers is None
        else int(args.num_workers)
    )
    if probe_batch_size <= 0 or probe_workers < 0:
        raise ValueError("probe batch size must be positive and workers non-negative")
    if int(args.batch_offset) < 0:
        raise ValueError("probe batch offset must be non-negative")
    device = _device(args.device)
    dtype = resolve_compute_dtype(config)
    _seed(config.data.seed)
    bundle = load_mainline_data(config)
    train_loader_generator = torch.Generator().manual_seed(config.data.seed + 101)
    flow_generator = _owned_generator(device, int(args.probe_seed) + 102)
    condition_generator = _owned_generator(device, int(args.probe_seed) + 103)
    train_loader = bundle.loader(
        "train",
        batch_size=probe_batch_size,
        workers=probe_workers,
        device=device,
        generator=train_loader_generator,
    )
    model = ClearVLAMainlinePolicy(config).to(device)
    configure_normalizer = getattr(model, "configure_action_normalizer", None)
    if configure_normalizer is not None:
        configure_normalizer(bundle.action_normalizer)
    identity = build_checkpoint_identity(
        config,
        repo_root=Path.cwd(),
        dataset=dataset_identity(bundle, config),
        language=language_identity(bundle, config),
    )
    replay = load_checkpoint_for_validation(
        args.checkpoint,
        model=model,
        config=config,
        identity=identity,
    )
    loader_iterator = iter(train_loader)
    raw_batch = None
    for _ in range(int(args.batch_offset) + 1):
        try:
            raw_batch = next(loader_iterator)
        except StopIteration as error:
            raise ValueError(
                f"probe batch offset {args.batch_offset} exceeds the train loader"
            ) from error
    if raw_batch is None:
        raise RuntimeError("probe failed to select a real training batch")
    batch = to_training_batch(
        raw_batch,
        goal=bundle.goal,
        config=config,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    report = run_probe(
        model=model,
        config=config,
        batch=batch,
        device=device,
        dtype=dtype,
        owner_roles=owner_roles,
        flow_generator=flow_generator,
        condition_generator=condition_generator,
        training_step=replay.global_step,
        probe_seed=int(args.probe_seed),
    )
    report.update(
        {
            "schema": REPORT_SCHEMA,
            "source": _repository_state(),
            "config": str(args.config.expanduser().resolve()),
            "config_digest_without_paths": config.digest(include_paths=False),
            "checkpoint": {
                "path": str(args.checkpoint.expanduser().resolve()),
                "size_bytes": args.checkpoint.stat().st_size,
                "epoch": replay.epoch,
                "global_step": replay.global_step,
                "best_metric": replay.best_metric,
                "saved_source_digest": replay.saved_source_digest,
                "current_source_digest": replay.current_source_digest,
                "changed_source_files": list(replay.changed_source_files),
            },
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "probe_loader": {
                "batch_size": probe_batch_size,
                "batch_offset": int(args.batch_offset),
                "num_workers": probe_workers,
                "serialized_training_batch_size": config.optimizer.batch_size,
                "serialized_training_num_workers": config.data.num_workers,
            },
            "normalizers": {
                "action_v120": v120_normalizer_fingerprint(
                    bundle.action_normalizer
                ),
                "state_v120": v120_normalizer_fingerprint(bundle.state_normalizer),
            },
            "cuda": (
                None
                if device.type != "cuda"
                else {
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / (1024**3),
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                    / (1024**3),
                }
            ),
        }
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        _write_once(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
