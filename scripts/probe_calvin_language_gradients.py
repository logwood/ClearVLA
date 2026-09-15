#!/usr/bin/env python3
"""Read-only parameter-owner VJP probe for the trained CALVIN checkpoint.

The probe builds one ordinary CALVIN training batch, loads a deployment
checkpoint, and evaluates the formal loss ledger without an optimizer step.
It reports gradients at the S typed seams and at the parameter owners that
can carry language/object information into P2.  It is deliberately an
external diagnostic: no checkpoint, cache, source file, or ``.grad`` field is
written.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.mainline.runtime.deployment import deployment_graph_config
from clearvla.mainline.runtime.identity import v120_normalizer_fingerprint
from clearvla.mainline.training.losses import compose_losses, sample_flow_matching
from clearvla.simulation.checkpoint import load_deployment_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="formal training config supplying the CALVIN raw-overlay data paths",
    )
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _generator(device: torch.device, value: int) -> torch.Generator:
    owner = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=owner).manual_seed(int(value))


def _autocast(device: torch.device, dtype: torch.dtype):
    enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.float16,
        torch.bfloat16,
    }
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _stats(value: Tensor | None) -> dict[str, float | int | str]:
    if value is None:
        return {"present": 0, "l2": 0.0, "rms": 0.0, "max_abs": 0.0, "nonzero": 0}
    detached = value.detach().float()
    return {
        "present": 1,
        "l2": float(torch.linalg.vector_norm(detached)),
        "rms": float(detached.square().mean().sqrt()),
        "max_abs": float(detached.abs().amax()),
        "nonzero": int(torch.count_nonzero(detached)),
    }


def _vjp(loss: Tensor, values: Iterable[Tensor], *, retain_graph: bool = True):
    values = tuple(values)
    if loss.ndim != 0 or not loss.requires_grad:
        return tuple(None for _ in values)
    return torch.autograd.grad(
        loss,
        values,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _group(model: nn.Module, prefix: str) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    )


def _group_stats(
    gradients: tuple[Tensor | None, ...],
    group: tuple[tuple[str, nn.Parameter], ...],
) -> dict[str, float | int]:
    if len(gradients) != len(group):
        raise ValueError("gradient/group length mismatch")
    present = [value.detach().float() for value in gradients if value is not None]
    if present:
        square = present[0].new_zeros(())
        for value in present:
            square = square + value.square().sum()
    else:
        square = torch.zeros(())
    elements = sum(int(parameter.numel()) for _, parameter in group)
    return {
        "parameter_tensors": len(group),
        "parameter_elements": elements,
        "gradient_present_tensors": len(present),
        "gradient_l2": float(square.sqrt()),
        "gradient_rms_over_parameters": float((square / max(elements, 1)).sqrt()),
    }


def _batch_identity(batch) -> dict[str, object]:
    def values(value: Tensor | None):
        return None if value is None else value.detach().cpu().tolist()

    return {
        "batch_size": int(batch.online.batch),
        "sample_index": values(batch.audit.sample_index),
        "episode_index": values(batch.audit.episode_index),
        "frame_progress": values(batch.audit.frame_progress),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed(args.seed)

    bundle = load_deployment_checkpoint(
        args.checkpoint,
        device=device,
        t5_condition=args.t5_condition,
    )
    # Deployment ABI intentionally omits split/cache paths.  Use the formal
    # run config for data materialization while retaining the checkpoint-owned
    # graph/model and normalizer.  The graph sections are checked explicitly
    # so this cannot silently probe a different architecture.
    config = load_config(args.config)
    if deployment_graph_config(config) != deployment_graph_config(bundle.config):
        raise ValueError("formal data config graph differs from checkpoint graph")
    data = load_mainline_data(config)
    if v120_normalizer_fingerprint(data.action_normalizer) != v120_normalizer_fingerprint(
        bundle.action_normalizer
    ):
        raise ValueError("formal CALVIN action normalizer differs from checkpoint")
    if v120_normalizer_fingerprint(data.state_normalizer) != v120_normalizer_fingerprint(
        bundle.state_normalizer
    ):
        raise ValueError("formal CALVIN state normalizer differs from checkpoint")
    loader = data.loader(
        "train",
        batch_size=int(config.optimizer.batch_size),
        workers=int(args.workers),
        device=device,
        generator=torch.Generator().manual_seed(int(config.data.seed) + 101),
    )
    if int(args.batch_offset) < 0:
        raise ValueError("batch offset must be non-negative")
    iterator = iter(loader)
    raw = None
    for _ in range(int(args.batch_offset) + 1):
        raw = next(iterator)
    batch = to_training_batch(raw, goal=data.goal, config=config, device=device)

    model = bundle.model.to(device)
    model.train()
    model.set_training_step(int(bundle.global_step))
    dtype = resolve_compute_dtype(config)
    condition_generator = _generator(device, int(args.seed) + 1)
    flow_generator = _generator(device, int(args.seed) + 2)

    with _autocast(device, dtype):
        cache, training_state, _ = model.encode_online(
            batch.online,
            training_mask=True,
            geometry_supervision=False,
            collect_diagnostics=False,
            condition_generator=condition_generator,
        )
        targets, _ = model.build_training_targets(
            training_state,
            batch.future,
            collect_diagnostics=False,
        )
        flow = sample_flow_matching(
            batch.action_target.normalized,
            action_state=batch.online.history.action_state,
            codec_gripper_boundary=batch.online.history.codec_gripper_boundary,
            codec=model.outlet_adapter,
            distribution=config.bottom.flow_time_distribution,
            generator=flow_generator,
        )
        output = model.velocity(
            cache,
            noisy_action_field=flow.noisy_physical,
            time=flow.time,
            require_execution_supervision=True,
            collect_diagnostics=False,
        )
        ledger = compose_losses(
            config,
            policy_output=output,
            action_target=batch.action_target,
            history=batch.online.history,
            flow_state=flow,
            observation=training_state.observation,
            top_targets=targets,
            predicted_dynamics=cache.top.predicted_dynamics,
            action_codec=model.outlet_adapter,
            collect_diagnostics=False,
        )

    intent = training_state.top.intent
    seams = {
        "public_interval_carrier": intent.public_interval_carrier,
        "policy_interval_context": intent.policy_interval_context,
        "typed_common_value": intent.typed_common_value,
        "typed_interval_residual_value": intent.typed_interval_residual_value,
        "typed_policy_components": intent.typed_policy_components,
        "object_tokens": intent.object_tokens,
        "coarse_action_prediction": training_state.top.coarse_action.action_prediction,
        "w_semantic_delta": cache.top.predicted_dynamics.semantic_delta,
        "p2_semantic_effect": output.compiled.effect.semantic,
        "physical_velocity": output.bottom.physical_velocity,
    }

    modules = {
        "intent_typed_semantic_query": model.intent.organizer.typed_relevance_queries[0],
        "intent_typed_appearance_query": model.intent.organizer.typed_relevance_queries[1],
        "intent_typed_geometry_query": model.intent.organizer.typed_relevance_queries[2],
        "intent_typed_temperature": model.intent.organizer.typed_temperature_logit,
        "intent_object_content": model.intent.organizer.object_content,
        "intent_object_semantic": model.intent.organizer.object_semantic,
        "intent_object_appearance": model.intent.organizer.object_appearance,
        "intent_object_geometry": model.intent.organizer.object_geometry,
        "coarse_object_read": model.intent.coarse_action.object_read,
        # P2 retains only semantic and geometry typed keys.  Appearance is
        # intentionally an S-side route and has no direct P2 key owner.
        "p2_typed_semantic_key": model.policy_compiler.effect_reader.typed_intent_key[0],
        "p2_typed_geometry_key": model.policy_compiler.effect_reader.typed_intent_key[1],
        "p2_semantic_value": model.policy_compiler.effect_reader.semantic_value,
    }
    parameter_groups = {
        name: tuple(module.named_parameters())
        for name, module in modules.items()
        if isinstance(module, nn.Module)
    }
    # A scalar Parameter is wrapped in a one-item group for uniform reporting.
    parameter_groups["intent_typed_temperature"] = ((
        "intent.organizer.typed_temperature_logit",
        modules["intent_typed_temperature"],
    ),)

    losses = {
        "action": ledger.groups["action"],
        "representation": ledger.groups["representation"],
        "execution": ledger.groups["execution"],
        "total": ledger.total,
        "action_flow": ledger.contributions["action_flow"],
        "coarse_action": ledger.contributions["coarse_action"],
        "intent_online": ledger.contributions["intent_online"],
        "future_dynamics": ledger.contributions["future_dynamics"],
        "future_transition": ledger.contributions["future_transition"],
        "object_reconstruction": ledger.contributions["object_reconstruction"],
    }
    seam_report: dict[str, object] = {}
    owner_report: dict[str, object] = {}
    for loss_name, loss in losses.items():
        seam_gradients = _vjp(loss, seams.values(), retain_graph=True)
        seam_report[loss_name] = {
            name: _stats(gradient)
            for name, gradient in zip(seams, seam_gradients, strict=True)
        }
        owner_report[loss_name] = {}
        for group_name, group in parameter_groups.items():
            params = tuple(parameter for _, parameter in group)
            gradients = _vjp(loss, params, retain_graph=True)
            owner_report[loss_name][group_name] = _group_stats(gradients, group)

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only probe unexpectedly populated parameter.grad")

    result = {
        "schema": "clearvla-calvin-language-owner-vjp-v1",
        "checkpoint": {
            "path": str(Path(args.checkpoint).expanduser().resolve()),
            "epoch": int(bundle.epoch),
            "global_step": int(bundle.global_step),
            "sha256": bundle.checkpoint_sha256,
        },
        "config_profile": str(config.data.data_profile),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "batch": _batch_identity(batch),
        "loss_values": {
            name: float(value.detach().float()) for name, value in losses.items()
        },
        "seam_gradients": seam_report,
        "owner_gradients": owner_report,
        "scope": {
            "optimizer_step": False,
            "checkpoint_write": False,
            "future_path_used_only_for_targets": True,
            "interpretation": (
                "zero typed-owner VJP on action/total means the language-conditioned "
                "typed route is not trained by that surface; nonzero but tiny P2 "
                "values indicate attenuation rather than a missing tensor edge"
            ),
        },
    }
    return result


def main() -> None:
    args = _parser().parse_args()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
