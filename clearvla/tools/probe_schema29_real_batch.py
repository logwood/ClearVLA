"""Compare Schema29/30 cache0 and cache1 gradients on one real training batch.

This tool is intentionally outside the training hot path.  It constructs the
formal data/model boundary in the same order as ``clearvla.mainline.train``,
samples one flow state, and compares two dynamic reads without taking an
optimizer step:

``cache0_single``
    One formal velocity read from the coarse-action-conditioned world.

``cache1_self_conditioned``
    One detached endpoint read, one W-only rebuild, then the formal velocity
    read from exactly the same noisy physical field, flow time, and dropout
    entry stream.

Only scalar summaries and SHA-256 fingerprints are emitted.  No checkpoint or
activation tensor is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.manifest import ARCHITECTURE_MANIFEST
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy, PolicyStepOutput
from clearvla.mainline.model.types import PhysicalActionCondition
from clearvla.mainline.runtime.identity import v120_normalizer_fingerprint
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training.losses import (
    LossLedger,
    compose_losses,
    sample_flow_matching,
)
from clearvla.mainline.training.optimizer import parameter_role

REPORT_SCHEMA = "clearvla-schema29-real-batch-gradient-ab-v2"
_ACTION_CONTRIBUTIONS = (
    "action_flow",
    "decoded_action",
    "gripper_trajectory",
    "motion",
    "smooth_delta",
    "physical_delta_consistency",
    "proposal",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare detached cache0 and formal cache1 VJPs on one real training batch"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mainline/object_intent_dynamics_323.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--gripper-event-threshold", type=float)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--output", type=Path)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _owned_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    data = config.data
    objectives = config.objectives
    optimizer = config.optimizer
    runtime = config.runtime
    for field_name, value in (
        ("raw_hdf5_root", args.data_root),
        ("decoded_cache", args.decoded_cache),
        ("dino_cache", args.dino_cache),
        ("t5_condition", args.t5_condition),
    ):
        if value is not None:
            data = replace(data, **{field_name: str(value)})
    if args.num_workers is not None:
        data = replace(data, num_workers=int(args.num_workers))
    if args.batch_size is not None:
        optimizer = replace(optimizer, batch_size=int(args.batch_size))
    if args.dtype is not None:
        runtime = replace(runtime, compute_dtype=str(args.dtype))
    if args.gripper_event_threshold is not None:
        threshold = float(args.gripper_event_threshold)
        data = replace(data, sampling_gripper_event_threshold=threshold)
        objectives = replace(objectives, gripper_event_threshold=threshold)
    result = replace(
        config,
        data=data,
        objectives=objectives,
        optimizer=optimizer,
        runtime=runtime,
    )
    result.validate()
    return result


def _autocast(
    device: torch.device,
    dtype: torch.dtype,
    *,
    cache_enabled: bool = True,
):
    enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=bool(cache_enabled),
    )


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        payload = tensor.view(torch.uint16).numpy().tobytes()
    else:
        payload = tensor.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _rng_sha256(value: Tensor) -> str:
    return _tensor_sha256(value.to(dtype=torch.uint8))


def _parameter_fingerprint(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, object]:
    digest = hashlib.sha256()
    square_sum = 0.0
    count = 0
    names: list[str] = []
    for name, parameter in named_parameters:
        value = parameter.detach().to(device="cpu").contiguous()
        names.append(name)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        if value.dtype == torch.bfloat16:
            digest.update(value.view(torch.uint16).numpy().tobytes())
        else:
            digest.update(value.numpy().tobytes())
        value_f = value.float()
        square_sum += float(value_f.square().sum())
        count += int(value.numel())
    if not names or count <= 0:
        raise ValueError("parameter fingerprint received an empty owner set")
    return {
        "parameter_names": names,
        "parameter_tensors": len(names),
        "parameter_elements": count,
        "parameter_rms": float((square_sum / count) ** 0.5),
        "sha256": digest.hexdigest(),
    }


def _named_owner_parameters(
    model: ClearVLAMainlinePolicy,
) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
    named = tuple(model.named_parameters())
    velocity_output_ids = {
        id(parameter)
        for layer in model.bottom.decoder.velocity_head.output_layers()
        for parameter in layer.parameters()
    }
    owners = {
        "velocity_output": tuple(
            (name, parameter)
            for name, parameter in named
            if id(parameter) in velocity_output_ids
        ),
        "gripper_gate": tuple(
            (name, parameter)
            for name, parameter in named
            if name.startswith("bottom.decoder.velocity_head.gripper_gate.")
        ),
        "motion_head": tuple(
            (name, parameter)
            for name, parameter in named
            if name.startswith("bottom.decoder.motion_head.")
        ),
    }
    if any(not values for values in owners.values()):
        raise RuntimeError("active bottom head parameter ownership is incomplete")
    ids = [id(parameter) for values in owners.values() for _, parameter in values]
    if len(ids) != len(set(ids)):
        raise RuntimeError("diagnostic bottom head owner sets overlap")
    return owners


def _named_trainable_parameters(
    model: ClearVLAMainlinePolicy,
) -> tuple[tuple[str, nn.Parameter], ...]:
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise RuntimeError("gradient probe found no trainable parameters")
    parameter_ids = [id(parameter) for _, parameter in named]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("trainable parameter inventory contains an alias")
    # Fail before the expensive VJP if a newly added parameter has escaped the
    # same optimizer-role map used by formal training.
    for name, _ in named:
        parameter_role(name)
    return named


def _gradient_l2(values: Iterable[Tensor | None]) -> float:
    squares = [
        value.detach().float().square().sum()
        for value in values
        if value is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def _gradient_rms(value: Tensor | None) -> float:
    if value is None:
        return 0.0
    return _rms(value)


def _parameter_gradient_stats(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    gradients: Iterable[Tensor | None],
) -> dict[str, float | int]:
    named = tuple(named_parameters)
    values = tuple(gradients)
    if len(named) != len(values):
        raise ValueError("parameter and gradient owner rows do not align")
    if not named:
        raise ValueError("parameter gradient aggregation received an empty owner")
    elements = sum(int(parameter.numel()) for _, parameter in named)
    present = tuple(value for value in values if value is not None)
    if present:
        square_sum = torch.stack(
            [value.detach().float().square().sum() for value in present]
        ).sum()
        gradient_l2 = float(square_sum.sqrt())
        gradient_rms = float((square_sum / float(max(elements, 1))).sqrt())
    else:
        gradient_l2 = 0.0
        gradient_rms = 0.0
    return {
        "parameter_tensors": len(named),
        "parameter_elements": elements,
        "gradient_present_tensors": len(present),
        "gradient_l2": gradient_l2,
        "gradient_rms_over_parameter_elements": gradient_rms,
    }


def _loss_surfaces(ledger: LossLedger) -> tuple[tuple[str, Tensor], ...]:
    rows = [
        (f"contrib_{name}", ledger.contributions[name])
        for name in _ACTION_CONTRIBUTIONS
    ]
    rows.extend(
        (
            ("group_action", ledger.groups["action"]),
            ("group_representation", ledger.groups["representation"]),
            ("group_execution", ledger.groups["execution"]),
            # Keep total last: its VJP is the final consumer of this dynamic
            # graph and can release the saved forward state.
            ("total", ledger.total),
        )
    )
    return tuple(rows)


def _vjp_report(
    ledger: LossLedger,
    *,
    owners: Mapping[str, tuple[tuple[str, nn.Parameter], ...]],
    trainable_parameters: tuple[tuple[str, nn.Parameter], ...],
    physical_velocity: Tensor,
    velocity_head_input: Tensor,
    retain_graph_after_total: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    owner_parameters = {
        name: tuple(parameter for _, parameter in values)
        for name, values in owners.items()
    }
    ordered_parameters = tuple(
        parameter
        for name in ("velocity_output", "gripper_gate", "motion_head")
        for parameter in owner_parameters[name]
    )
    owner_slices: dict[str, slice] = {}
    start = 0
    for name in ("velocity_output", "gripper_gate", "motion_head"):
        end = start + len(owner_parameters[name])
        owner_slices[name] = slice(start, end)
        start = end
    activation_start = len(ordered_parameters)
    targets = (*ordered_parameters, physical_velocity, velocity_head_input)
    rows: dict[str, dict[str, float]] = {}
    surfaces = _loss_surfaces(ledger)
    if not surfaces or surfaces[-1][0] != "total":
        raise RuntimeError("total loss must be the final VJP surface")
    for name, loss in surfaces[:-1]:
        if loss.ndim != 0:
            raise ValueError(f"diagnostic loss {name!r} must be scalar")
        if loss.requires_grad:
            gradients = torch.autograd.grad(
                loss,
                targets,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            gradients = tuple(None for _ in targets)
        rows[name] = {
            "loss_value": float(loss.detach().float()),
            "velocity_output_parameter_gradient_l2": _gradient_l2(
                gradients[owner_slices["velocity_output"]]
            ),
            "gripper_gate_parameter_gradient_l2": _gradient_l2(
                gradients[owner_slices["gripper_gate"]]
            ),
            "motion_head_parameter_gradient_l2": _gradient_l2(
                gradients[owner_slices["motion_head"]]
            ),
            "physical_velocity_gradient_rms": _gradient_rms(
                gradients[activation_start]
            ),
            "velocity_head_input_gradient_rms": _gradient_rms(
                gradients[activation_start + 1]
            ),
        }

    total_name, total_loss = surfaces[-1]
    if total_loss.ndim != 0:
        raise ValueError("diagnostic total loss must be scalar")
    trainable = tuple(parameter for _, parameter in trainable_parameters)
    total_targets = (*trainable, physical_velocity, velocity_head_input)
    if total_loss.requires_grad:
        total_gradients = torch.autograd.grad(
            total_loss,
            total_targets,
            retain_graph=retain_graph_after_total,
            allow_unused=True,
        )
    else:
        total_gradients = tuple(None for _ in total_targets)
    parameter_gradients = total_gradients[: len(trainable)]
    activation_gradients = total_gradients[len(trainable) :]
    parameter_index = {
        id(parameter): index
        for index, (_, parameter) in enumerate(trainable_parameters)
    }

    def selected_stats(
        selected: Iterable[tuple[str, nn.Parameter]],
    ) -> dict[str, float | int]:
        named = tuple(selected)
        try:
            indices = tuple(parameter_index[id(parameter)] for _, parameter in named)
        except KeyError as error:
            raise RuntimeError("VJP owner is absent from the trainable inventory") from error
        return _parameter_gradient_stats(
            named,
            (parameter_gradients[index] for index in indices),
        )

    rows[total_name] = {
        "loss_value": float(total_loss.detach().float()),
        "velocity_output_parameter_gradient_l2": float(
            selected_stats(owners["velocity_output"])["gradient_l2"]
        ),
        "gripper_gate_parameter_gradient_l2": float(
            selected_stats(owners["gripper_gate"])["gradient_l2"]
        ),
        "motion_head_parameter_gradient_l2": float(
            selected_stats(owners["motion_head"])["gradient_l2"]
        ),
        "physical_velocity_gradient_rms": _gradient_rms(
            activation_gradients[0]
        ),
        "velocity_head_input_gradient_rms": _gradient_rms(
            activation_gradients[1]
        ),
    }
    role_names = sorted({parameter_role(name) for name, _ in trainable_parameters})
    role_stats = {
        role: selected_stats(
            (name, parameter)
            for name, parameter in trainable_parameters
            if parameter_role(name) == role
        )
        for role in role_names
    }
    block_stats: dict[str, dict[str, float | int]] = {}
    for index in range(3):
        prefix = f"bottom.decoder.blocks.{index}."
        selected = tuple(
            (name, parameter)
            for name, parameter in trainable_parameters
            if name.startswith(prefix)
        )
        if not selected:
            raise RuntimeError(f"active Evidence MMDiT block {index} has no parameters")
        block_stats[f"block_{index}"] = selected_stats(selected)
    return rows, {
        "all_trainable": _parameter_gradient_stats(
            trainable_parameters,
            parameter_gradients,
        ),
        "parameter_roles": role_stats,
        "bottom_mmdit_blocks": block_stats,
    }


def _variation_rms(value: Tensor, dim: int) -> float:
    centered = value.detach().float() - value.detach().float().mean(
        dim=dim,
        keepdim=True,
    )
    return _rms(centered)


def _single_tensor_stats(
    value: Tensor,
    *,
    interval_dim: int | None = 1,
) -> dict[str, float]:
    result = {
        "rms": _rms(value),
        "batch_variation_rms": _variation_rms(value, 0),
    }
    if interval_dim is not None and value.ndim > interval_dim:
        result["interval_variation_rms"] = _variation_rms(value, interval_dim)
    return result


def _pair_stats(left: Tensor, right: Tensor, *, interval_dim: int | None = 1) -> dict[str, float]:
    left_f = left.detach().float()
    right_f = right.detach().float()
    if tuple(left_f.shape) != tuple(right_f.shape):
        raise ValueError("paired diagnostic boundaries must have identical shapes")
    cosine = torch.nn.functional.cosine_similarity(
        left_f.flatten(1),
        right_f.flatten(1),
        dim=-1,
        eps=1e-8,
    ).mean()
    result = {
        "left_rms": _rms(left_f),
        "right_rms": _rms(right_f),
        "delta_rms": _rms(right_f - left_f),
        "sample_flat_cosine": float(cosine),
        "left_batch_variation_rms": _variation_rms(left_f, 0),
        "right_batch_variation_rms": _variation_rms(right_f, 0),
    }
    if interval_dim is not None and left_f.ndim > interval_dim:
        result.update(
            {
                "left_interval_variation_rms": _variation_rms(left_f, interval_dim),
                "right_interval_variation_rms": _variation_rms(right_f, interval_dim),
            }
        )
    return result


def _head_input_from_norm_calls(values: list[Tensor]) -> Tensor:
    # The final physical head reads one base state and one gripper-private
    # state through the shared norm.  Prefix/candidate reads may precede it;
    # the penultimate norm input is therefore the base input of the final
    # deployed field read.
    if len(values) < 2:
        raise RuntimeError("velocity head input hook observed fewer than two norm calls")
    base = values[-2]
    private = values[-1]
    if tuple(base.shape) != tuple(private.shape):
        raise RuntimeError("velocity head base/private reads lost a shared shape")
    return base


def _mode_report(
    output: PolicyStepOutput,
    ledger: LossLedger,
    *,
    owners: Mapping[str, tuple[tuple[str, nn.Parameter], ...]],
    trainable_parameters: tuple[tuple[str, nn.Parameter], ...],
    velocity_head_input: Tensor,
    transition_value: Tensor,
    predicted_semantic: Tensor,
    predicted_transport: Tensor,
    retain_graph_after_total: bool = False,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    physical_velocity = output.bottom.physical_velocity
    boundaries = {
        "physical_velocity": physical_velocity.detach().float().cpu(),
        "velocity_head_input": velocity_head_input.detach().float().cpu(),
        "w_semantic": predicted_semantic.detach().float().cpu(),
        "w_transport": predicted_transport.detach().float().cpu(),
        "p2_effect": output.compiled.effect.combined().detach().float().cpu(),
        "consequence": output.compiled.consequence.protected_consequence.detach()
        .float()
        .cpu(),
        "consequence_interaction": output.compiled.consequence.interaction.combined()
        .detach()
        .float()
        .cpu(),
        "controlled_transition": transition_value.detach().float().cpu(),
    }
    losses_and_vjps, total_owner_vjps = _vjp_report(
        ledger,
        owners=owners,
        trainable_parameters=trainable_parameters,
        physical_velocity=physical_velocity,
        velocity_head_input=velocity_head_input,
        retain_graph_after_total=retain_graph_after_total,
    )
    report = {
        "losses_and_vjps": losses_and_vjps,
        "total_owner_vjps": total_owner_vjps,
        "dtypes": {
            "physical_velocity": str(physical_velocity.dtype).removeprefix("torch."),
            "velocity_head_input": str(velocity_head_input.dtype).removeprefix(
                "torch."
            ),
            "w_semantic": str(predicted_semantic.dtype).removeprefix("torch."),
            "w_transport": str(predicted_transport.dtype).removeprefix("torch."),
            "p2_effect": str(output.compiled.effect.combined().dtype).removeprefix(
                "torch."
            ),
            "controlled_transition": str(transition_value.dtype).removeprefix(
                "torch."
            ),
        },
        "forward": {
            "physical_velocity": _single_tensor_stats(boundaries["physical_velocity"]),
            "velocity_head_input": _single_tensor_stats(
                boundaries["velocity_head_input"]
            ),
            "w_semantic": _single_tensor_stats(boundaries["w_semantic"]),
            "w_transport": _single_tensor_stats(boundaries["w_transport"]),
            "p2_effect": _single_tensor_stats(boundaries["p2_effect"]),
            "consequence": _single_tensor_stats(boundaries["consequence"]),
            "consequence_interaction": _single_tensor_stats(
                boundaries["consequence_interaction"]
            ),
            "controlled_transition": _single_tensor_stats(
                boundaries["controlled_transition"]
            ),
        },
    }
    return report, boundaries


def _cuda_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [
        torch.cuda.current_device()
        if device.index is None
        else int(device.index)
    ]


def _relative_decision(left: float, right: float) -> dict[str, object]:
    if left == 0.0 and right == 0.0:
        return {"ratio_cache1_over_cache0": None, "classification": "no_vjp_signal"}
    ratio = right / max(left, 1e-30)
    if left > 0.0 and ratio <= 0.10:
        classification = "cache1_specific_strong_attenuation"
    elif 0.50 <= ratio <= 2.0:
        classification = "no_large_cache0_cache1_difference"
    else:
        classification = "cache0_cache1_difference_requires_interpretation"
    return {
        "ratio_cache1_over_cache0": ratio,
        "classification": classification,
    }


def _relative_total_owner_decisions(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for section in ("parameter_roles", "bottom_mmdit_blocks"):
        left_section = left.get(section)
        right_section = right.get(section)
        if not isinstance(left_section, Mapping) or not isinstance(
            right_section, Mapping
        ):
            raise RuntimeError(f"total-owner VJP section {section!r} is malformed")
        if set(left_section) != set(right_section):
            raise RuntimeError(f"total-owner VJP section {section!r} changed owners")
        rows: dict[str, dict[str, object]] = {}
        for name in sorted(str(value) for value in left_section):
            left_row = left_section[name]
            right_row = right_section[name]
            if not isinstance(left_row, Mapping) or not isinstance(
                right_row, Mapping
            ):
                raise RuntimeError(f"total-owner VJP row {name!r} is malformed")
            rows[name] = _relative_decision(
                float(left_row["gradient_l2"]),
                float(right_row["gradient_l2"]),
            )
        result[section] = rows
    return result


def run_schema29_real_batch_probe(
    *,
    model: ClearVLAMainlinePolicy,
    config: ExperimentConfig,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    flow_generator: torch.Generator,
    condition_generator: torch.Generator,
) -> dict[str, object]:
    """Run the controlled A/B comparison on an already typed real/fake batch."""

    if int(ARCHITECTURE_MANIFEST.schema) < 29:
        raise RuntimeError(
            "the active manifest predates detached cache0/formal cache1 training"
        )
    config.validate()
    batch.validate(config)
    model.train()
    model.set_training_step(0)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("gradient probe requires a pristine model gradient state")
    owners = _named_owner_parameters(model)
    trainable_parameters = _named_trainable_parameters(model)
    initialization = {
        name: _parameter_fingerprint(values)
        for name, values in owners.items()
    }

    with _autocast(device, dtype):
        cache0, training_state, _ = model.encode_online(
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
        flow_state = sample_flow_matching(
            batch.action_target.normalized,
            action_state=batch.online.history.action_state,
            codec=model.action_codec,
            distribution=config.bottom.flow_time_distribution,
            generator=flow_generator,
        )

    dynamic_cpu_rng = torch.get_rng_state().clone()
    cuda_devices = _cuda_devices(device)
    dynamic_cuda_rng = (
        torch.cuda.get_rng_state(cuda_devices[0]).clone()
        if cuda_devices
        else None
    )
    head_norm_inputs: list[Tensor] = []
    transition_values: list[Tensor] = []

    def capture_head_input(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        if not args or not isinstance(args[0], Tensor):
            raise RuntimeError("velocity head norm hook lost its tensor input")
        head_norm_inputs.append(args[0])

    def capture_transition(
        _module: nn.Module,
        _args: tuple[object, ...],
        result: object,
    ) -> None:
        if not isinstance(result, tuple) or not result:
            raise RuntimeError("controlled transition hook lost its typed output")
        state = result[0]
        value = getattr(state, "value", None)
        if not isinstance(value, Tensor):
            raise RuntimeError("controlled transition state lost its value tensor")
        transition_values.append(value)

    head_hook = model.bottom.decoder.velocity_head.norm.register_forward_pre_hook(
        capture_head_input
    )
    transition_hook = model.transition.register_forward_hook(capture_transition)
    try:
        # A is forked so B begins from the exact same dynamic dropout state.
        with torch.random.fork_rng(devices=cuda_devices):
            with _autocast(device, dtype):
                cache0_autocast_cache_enabled = torch.is_autocast_cache_enabled()
                output0 = model.velocity(
                    cache0,
                    noisy_action_field=flow_state.noisy_physical,
                    time=flow_state.time,
                    require_execution_supervision=True,
                    collect_diagnostics=False,
                )
                head_input0 = _head_input_from_norm_calls(head_norm_inputs)
                if not transition_values:
                    raise RuntimeError("cache0 velocity lost controlled transition")
                transition0 = transition_values[-1]
                ledger0 = compose_losses(
                    config,
                    policy_output=output0,
                    action_target=batch.action_target,
                    history=batch.online.history,
                    flow_state=flow_state,
                    observation=training_state.observation,
                    top_targets=top_targets,
                    predicted_dynamics=cache0.top.predicted_dynamics,
                    action_codec=model.action_codec,
                    collect_diagnostics=False,
                )
            report0, boundaries0 = _mode_report(
                output0,
                ledger0,
                owners=owners,
                trainable_parameters=trainable_parameters,
                velocity_head_input=head_input0,
                transition_value=transition0,
                predicted_semantic=cache0.top.predicted_dynamics.semantic_delta,
                predicted_transport=cache0.top.predicted_dynamics.transport_mean,
                retain_graph_after_total=True,
            )
        cache0_fork_restored_cpu = torch.equal(
            torch.get_rng_state(),
            dynamic_cpu_rng,
        )
        cache0_fork_restored_cuda = (
            True
            if dynamic_cuda_rng is None
            else torch.equal(
                torch.cuda.get_rng_state(cuda_devices[0]),
                dynamic_cuda_rng,
            )
        )
        if not cache0_fork_restored_cpu or not cache0_fork_restored_cuda:
            raise RuntimeError("cache0 A/B fork failed to restore the dynamic RNG entry")
        del output0, ledger0, head_input0, transition0

        head_norm_inputs.clear()
        transition_values.clear()
        with _autocast(device, dtype):
            with torch.random.fork_rng(devices=cuda_devices):
                with torch.no_grad():
                    with _autocast(device, dtype, cache_enabled=False):
                        pass0_autocast_cache_enabled = (
                            torch.is_autocast_cache_enabled()
                        )
                        pass0_output = model.velocity(
                            cache0,
                            noisy_action_field=flow_state.noisy_physical,
                            time=flow_state.time,
                            require_execution_supervision=False,
                            collect_diagnostics=False,
                        )
                        pass0_velocity_requires_grad = bool(
                            pass0_output.bottom.physical_velocity.requires_grad
                        )
                        remaining = (1.0 - flow_state.time.to(
                            dtype=flow_state.noisy_physical.dtype
                        ))[:, None, None]
                        pass0_clean_physical = (
                            flow_state.noisy_physical
                            + remaining
                            * pass0_output.bottom.physical_velocity.to(
                                dtype=flow_state.noisy_physical.dtype
                            )
                        )
                        pass0_clean_action = model.action_codec.decode(
                            pass0_clean_physical,
                            cache0.history.action_state,
                        ).detach()
                        pass0_condition = PhysicalActionCondition.from_horizon_action(
                            pass0_clean_action,
                            cache0.history.action_state.detach(),
                        )
                        pass0_velocity_dtype = str(
                            pass0_output.bottom.physical_velocity.dtype
                        ).removeprefix("torch.")
                        del pass0_output, pass0_clean_physical

            pass0_fork_restored_cpu = torch.equal(
                torch.get_rng_state(),
                dynamic_cpu_rng,
            )
            pass0_fork_restored_cuda = (
                True
                if dynamic_cuda_rng is None
                else torch.equal(
                    torch.cuda.get_rng_state(cuda_devices[0]),
                    dynamic_cuda_rng,
                )
            )
            if not pass0_fork_restored_cpu or not pass0_fork_restored_cuda:
                raise RuntimeError("Schema29/30 pass0 failed to restore the formal RNG entry")

            cache1_top, _ = model.top.refine_deployment_world(
                cache0.top,
                action_condition=pass0_condition,
                collect_diagnostics=False,
            )
            cache1 = replace(cache0, top=cache1_top)
            cache1_formal_autocast_cache_enabled = (
                torch.is_autocast_cache_enabled()
            )
            output1 = model.velocity(
                cache1,
                noisy_action_field=flow_state.noisy_physical,
                time=flow_state.time,
                require_execution_supervision=True,
                collect_diagnostics=False,
            )
            head_input1 = _head_input_from_norm_calls(head_norm_inputs)
            if not transition_values:
                raise RuntimeError("cache1 velocity lost controlled transition")
            transition1 = transition_values[-1]
            ledger1 = compose_losses(
                config,
                policy_output=output1,
                action_target=batch.action_target,
                history=batch.online.history,
                flow_state=flow_state,
                observation=training_state.observation,
                top_targets=top_targets,
                predicted_dynamics=cache1.top.predicted_dynamics,
                action_codec=model.action_codec,
                collect_diagnostics=False,
            )
        report1, boundaries1 = _mode_report(
            output1,
            ledger1,
            owners=owners,
            trainable_parameters=trainable_parameters,
            velocity_head_input=head_input1,
            transition_value=transition1,
            predicted_semantic=cache1.top.predicted_dynamics.semantic_delta,
            predicted_transport=cache1.top.predicted_dynamics.transport_mean,
        )
    finally:
        head_hook.remove()
        transition_hook.remove()

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("VJP probe unexpectedly populated persistent .grad tensors")

    pairs = {
        name: _pair_stats(boundaries0[name], boundaries1[name])
        for name in boundaries0
    }
    action_flow0 = report0["losses_and_vjps"]["contrib_action_flow"]
    action_flow1 = report1["losses_and_vjps"]["contrib_action_flow"]
    if not isinstance(action_flow0, Mapping) or not isinstance(action_flow1, Mapping):
        raise RuntimeError("action-flow VJP report is malformed")
    velocity_decision = _relative_decision(
        float(action_flow0["velocity_output_parameter_gradient_l2"]),
        float(action_flow1["velocity_output_parameter_gradient_l2"]),
    )
    motion0 = report0["losses_and_vjps"]["contrib_motion"]
    motion1 = report1["losses_and_vjps"]["contrib_motion"]
    if not isinstance(motion0, Mapping) or not isinstance(motion1, Mapping):
        raise RuntimeError("motion VJP report is malformed")
    motion_decision = _relative_decision(
        float(motion0["motion_head_parameter_gradient_l2"]),
        float(motion1["motion_head_parameter_gradient_l2"]),
    )
    if report0["dtypes"] != report1["dtypes"]:
        raise RuntimeError("cache0/cache1 formal boundaries changed dtype")
    total_owner0 = report0["total_owner_vjps"]
    total_owner1 = report1["total_owner_vjps"]
    if not isinstance(total_owner0, Mapping) or not isinstance(
        total_owner1, Mapping
    ):
        raise RuntimeError("total-owner VJP report is malformed")
    total_owner_decisions = _relative_total_owner_decisions(
        total_owner0,
        total_owner1,
    )
    coarse_fingerprint = cache0.top.action_condition.fingerprint.detach().float()
    refined_fingerprint = pass0_condition.fingerprint.detach().float()
    final_report = {
        "schema": REPORT_SCHEMA,
        "manifest_schema": int(ARCHITECTURE_MANIFEST.schema),
        "manifest_sha256": ARCHITECTURE_MANIFEST.digest(),
        "model_training_step": 0,
        "optimizer_constructed": False,
        "optimizer_step_taken": False,
        "checkpoint_loaded": False,
        "initialization": initialization,
        "rng": {
            "dynamic_cpu_entry_sha256": _rng_sha256(dynamic_cpu_rng),
            "dynamic_cuda_entry_sha256": (
                None if dynamic_cuda_rng is None else _rng_sha256(dynamic_cuda_rng)
            ),
            "cache0_single_fork_restored_cpu": cache0_fork_restored_cpu,
            "cache0_single_fork_restored_cuda": cache0_fork_restored_cuda,
            "cache1_pass0_fork_restored_cpu": pass0_fork_restored_cpu,
            "cache1_pass0_fork_restored_cuda": pass0_fork_restored_cuda,
            # The cache1 formal read legitimately advances the global stream.
            "cache1_formal_read_advanced_cpu_rng": not torch.equal(
                torch.get_rng_state(), dynamic_cpu_rng
            ),
        },
        "flow_state": {
            "time_mean": float(flow_state.time.detach().float().mean()),
            "time_min": float(flow_state.time.detach().float().amin()),
            "time_max": float(flow_state.time.detach().float().amax()),
            "noisy_physical_rms": _rms(flow_state.noisy_physical),
            "target_physical_velocity_rms": _rms(
                flow_state.target_physical_velocity
            ),
            "noisy_physical_sha256": _tensor_sha256(flow_state.noisy_physical),
        },
        "self_conditioning": {
            "pass0_clean_action_rms": _rms(pass0_clean_action),
            "pass0_velocity_dtype": pass0_velocity_dtype,
            "pass0_velocity_requires_grad": pass0_velocity_requires_grad,
            "pass0_action_requires_grad": bool(pass0_clean_action.requires_grad),
            "pass0_condition_interval_action_requires_grad": bool(
                pass0_condition.interval_action.requires_grad
            ),
            "pass0_condition_interval_delta_requires_grad": bool(
                pass0_condition.interval_delta.requires_grad
            ),
            "coarse_to_pass0_condition_rms": _rms(
                refined_fingerprint - coarse_fingerprint
            ),
            "cache0_to_cache1_w_semantic": pairs["w_semantic"],
            "cache0_to_cache1_w_transport": pairs["w_transport"],
        },
        "autocast": {
            "cache0_formal_cache_enabled": cache0_autocast_cache_enabled,
            "cache1_pass0_cache_enabled": pass0_autocast_cache_enabled,
            "cache1_formal_cache_enabled": (
                cache1_formal_autocast_cache_enabled
            ),
        },
        "modes": {
            "cache0_single": report0,
            "cache1_self_conditioned": report1,
        },
        "paired_boundaries": pairs,
        "relative_decision": {
            "action_flow_to_velocity_output_layers": velocity_decision,
            "motion_loss_to_motion_head": motion_decision,
            "total_loss_owners": total_owner_decisions,
            "scope": (
                "relative first-batch attribution only; this does not claim "
                "trained behavior or a fixed-point property"
            ),
        },
    }
    return final_report


def _repository_state(expected_commit: str | None) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if expected_commit is not None and commit != str(expected_commit).strip():
        raise ValueError(
            f"source commit {commit} does not match expected {expected_commit}"
        )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "worktree_dirty": bool(status.strip()),
    }


def _batch_identity(batch: TrainingBatch) -> dict[str, object]:
    def values(value: Tensor | None) -> list[float | int] | None:
        if value is None:
            return None
        raw = value.detach().cpu().tolist()
        return [item for item in raw]

    return {
        "batch_size": int(batch.online.batch),
        "sample_indices": values(batch.audit.sample_index),
        "episode_indices": values(batch.audit.episode_index),
        "frame_progress": values(batch.audit.frame_progress),
    }


def _write_report(path: Path, rendered: str) -> None:
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


def main() -> None:
    args = _parser().parse_args()
    config = _overrides(load_config(args.config), args)
    device = _device(args.device)
    dtype = resolve_compute_dtype(config)

    # Keep the formal initialization order: seed -> complete data bundle ->
    # owned loaders/generators -> model.  Optimizer construction is omitted
    # deliberately because this probe must not create or update optimizer
    # state and AdamW construction consumes no model-forward RNG.
    _seed(config.data.seed)
    bundle = load_mainline_data(config)
    train_loader_generator = torch.Generator().manual_seed(config.data.seed + 101)
    train_flow_generator = _owned_generator(device, config.data.seed + 102)
    train_condition_generator = _owned_generator(device, config.data.seed + 103)
    train_loader = bundle.loader(
        "train",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        generator=train_loader_generator,
    )
    val_loader = bundle.loader(
        "val",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        shuffle=False,
    )
    model = ClearVLAMainlinePolicy(config).to(device)
    raw_batch = next(iter(train_loader))
    batch = to_training_batch(
        raw_batch,
        goal=bundle.goal,
        config=config,
        device=device,
    )
    report = run_schema29_real_batch_probe(
        model=model,
        config=config,
        batch=batch,
        device=device,
        dtype=dtype,
        flow_generator=train_flow_generator,
        condition_generator=train_condition_generator,
    )
    report.update(
        {
            "source": _repository_state(args.expected_source_commit),
            "config": str(args.config.expanduser().resolve()),
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "data": {
                "train_loader_batches": len(train_loader),
                "val_loader_batches": len(val_loader),
                "action_normalizer_v120": v120_normalizer_fingerprint(
                    bundle.action_normalizer
                ),
                "state_normalizer_v120": v120_normalizer_fingerprint(
                    bundle.state_normalizer
                ),
                "batch": _batch_identity(batch),
            },
        }
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        _write_report(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
