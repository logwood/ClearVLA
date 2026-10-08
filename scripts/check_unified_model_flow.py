#!/usr/bin/env python3
"""Full production topology with explicitly synthetic observations and labels.

Small and production dimensions are distinct qualifications. Neither executes
pretrained DINO/T5 nor certifies real-data learning. It runs the ordinary
trainer/AdamW backward, then full proposal -> W rebuild -> refined sampling.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def configuration(
    shape: str,
    variant: str,
    typed_object_values: str = "legacy_selected_v1",
    compute_dtype: str = "fp32",
    outcome_mode: str = "before_proposal_v1",
):
    from importlib import import_module

    small_config = import_module("test_mainline_executed_world")._config

    from clearvla.mainline.config import load_config
    from clearvla.mainline.task_execution import JOINT_TASK_EXECUTION

    config = (
        small_config()
        if shape == "small"
        else load_config("configs/mainline/dinov3_causal_repair_calvin_20261006.json")
    )
    top = replace(
        config.top,
        entity_transport_gradient_mode="ordinary_bilinear_v1",
        entity_competition_scale_mode="per_observation_v1",
        observation_measurement_mode="source_consistent_v1",
        target_binding_input_mode="full_tokens_views_v1",
        observed_outcome_mode=outcome_mode,
        typed_interval_gradient_mode="ordinary_v1",
        typed_object_value_mode=typed_object_values,
        object_view_mode="per_camera_values_v1",
        task_execution_mode=JOINT_TASK_EXECUTION,
        entity_context_mode="completed_g3_v1",
        entity_ownership_mode="canonical_image_v1" if variant == "B" else "local_mixture_v1",
        identity_supervision_mode="rgbd_temporal_conditional_v2"
        if variant == "B" and shape == "production"
        else "none",
    )
    objective = replace(
        config.objectives,
        identity_correspondence=0.01 if variant == "B" and shape == "production" else 0.0,
        identity_source_prediction=0.01 if variant == "B" and shape == "production" else 0.0,
    )
    config = replace(
        config,
        top=top,
        objectives=objective,
        runtime=replace(config.runtime, compute_dtype=compute_dtype),
        data=replace(
            config.data,
            visual_feature_mode="dinov3_online_v1" if shape == "production" else "dinov2_cached_v1",
        ),
    )
    config.validate()
    return config


def run(
    shape: str,
    variant: str,
    updates: int,
    raw_side: int,
    completed_step: int = 0,
    typed_object_values: str = "legacy_selected_v1",
    compute_dtype: str = "fp32",
    batch_size: int = 1,
    reference_support: str = "full",
    outcome_mode: str = "before_proposal_v1",
) -> dict:
    import torch

    from clearvla.mainline.identity_supervision import IdentityCorrespondence
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.qualification import synthetic_batch
    from clearvla.mainline.runtime.sampling import sample_action
    from clearvla.mainline.training.engine import (
        MainlineTrainingEngine,
        validate_finite_training_batch,
    )
    from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer

    if type(completed_step) is not int or completed_step < 0:
        raise ValueError("fixture completed step must be a nonnegative integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("synthetic batch size must be a positive integer")
    if reference_support not in {"full", "alternating"}:
        raise ValueError("unknown synthetic reference support mode")
    torch.set_num_threads(1)
    torch.manual_seed(28431)
    config = configuration(shape, variant, typed_object_values, compute_dtype, outcome_mode)
    config = replace(config, optimizer=replace(config.optimizer, batch_size=batch_size))
    config.validate()
    fixture_config = replace(
        config,
        top=replace(config.top, identity_supervision_mode="none"),
        objectives=replace(
            config.objectives, identity_correspondence=0.0, identity_source_prediction=0.0
        ),
    )
    batch, normalizer = synthetic_batch(
        fixture_config, count=batch_size, raw_side=raw_side, device=torch.device("cpu")
    )
    reference = batch.online.instruction_reference
    assert reference is not None
    if reference_support == "alternating":
        # A legal partial instruction-start observation, not a learned mask.
        # Its unavailable payload is poisoned before the real S/goal paths.
        # Current/future observations keep their OWN original source support.
        support = reference.observed.clone()
        support[..., ::2] = False
        reference = replace(
            reference,
            observed=support,
            dino=torch.where(support[..., None], reference.dino, torch.nan),
        )
        batch = replace(batch, online=replace(batch.online, instruction_reference=reference))
    if config.top.identity_supervision_mode != "none":
        xy = (
            torch.tensor([[-0.5, -0.5], [0.5, 0.5], [0.1, 0.1]]).expand(batch_size, 2, 3, 2).clone()
        )
        valid = torch.ones(batch_size, 2, 3, dtype=torch.bool)
        assert batch.online.history.timing is not None
        pairs = IdentityCorrespondence(
            xy,
            xy.clone(),
            valid,
            xy.clone(),
            xy.clone(),
            valid.clone(),
            batch.online.history.timing.state_offsets[:, -2:] + 24,
        )
        batch = replace(batch, identity=pairs)
    batch.validate(config)
    validate_finite_training_batch(batch)
    print(
        json.dumps(
            {
                "event": "model_construction",
                "dimensions": config.dimensions.__dict__,
                "shape": shape,
            }
        ),
        flush=True,
    )
    model = ClearVLAMainlinePolicy(config)
    model.configure_action_normalizer(normalizer)
    print(
        json.dumps(
            {"event": "model_constructed", "parameters": sum(p.numel() for p in model.parameters())}
        ),
        flush=True,
    )
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=completed_step + max(updates, 4),
        minimum_ratio=0.1,
        update_origin=completed_step,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
    )
    # Select execution gates on randomly initialized fixture parameters.
    # This does not claim prior training or a restored learned checkpoint.
    engine.global_step = completed_step
    model.set_training_step(completed_step)
    rows = []
    parameter_ledger = []
    for step in range(updates):
        started = time.monotonic()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        result = engine.train_step(batch, collect_diagnostics=True)
        assert torch.isfinite(result.loss) and torch.isfinite(result.gradient_norm)
        gradients = {}
        for name, module in [
            ("G", model.grounding),
            ("S", model.intent),
            ("W", model.world),
            ("P", model.policy_compiler),
            ("bottom", model.execution_bottom),
        ]:
            params = [p for p in module.parameters() if p.requires_grad]
            assert all(bool(torch.isfinite(p).all()) for p in params)
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
            active = [p.grad for p in params if p.grad is not None]
            magnitude = sum(float(g.float().square().sum()) for g in active) ** 0.5
            assert active and magnitude > 0, name + " has no ordinary loss gradient"
            gradients[name] = dict(
                parameter_tensors=len(params), gradient_tensors=len(active), l2=magnitude
            )
        row = dict(
            update=step + 1,
            loss=float(result.loss),
            preclip_l2=float(result.gradient_norm),
            gradients=gradients,
            seconds=time.monotonic() - started,
        )
        # Per-parameter finite/gradient/update accounting supplements aggregate
        # module magnitudes. A dormant/zero-initialized path is not fabricated
        # into a nonzero gradient, and a missing path remains explicitly listed.
        entries = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            changed = (parameter.detach().float() - before[name].float()).square().sum().sqrt()
            entries.append(
                dict(
                    name=name,
                    shape=list(parameter.shape),
                    gradient_status="missing"
                    if grad is None
                    else ("zero" if not grad.count_nonzero() else "nonzero"),
                    gradient_l2=None if grad is None else float(grad.detach().float().norm()),
                    update_l2=float(changed),
                )
            )
        parameter_ledger.append(dict(update=step + 1, entries=entries))
        del before
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert engine.global_step == completed_step + updates and optimizer.state
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        noise = model.outlet_adapter.sample_noise(
            batch_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(9331),
        )
        sampled = sample_action(model, batch.online, config, initial_physical_noise=noise)
    assert (
        sampled.action.shape
        == (batch_size, config.dimensions.action_horizon, config.dimensions.action_dim)
        and torch.isfinite(sampled.action).all()
    )
    assert sampled.gripper_command is not None and torch.isfinite(sampled.gripper_command).all()
    return dict(
        passed=True,
        variant=variant,
        dimensions=config.dimensions.__dict__,
        raw_rgb_side=raw_side,
        reference_support_mode=reference_support,
        reference_observed_cells=int(reference.observed.sum()),
        reference_total_cells=reference.observed.numel(),
        observation_measurement_mode=config.top.observation_measurement_mode,
        observed_outcome_mode=config.top.observed_outcome_mode,
        actual_batch_size=batch_size,
        optimizer_batch_size=config.optimizer.batch_size,
        compute_dtype=config.runtime.compute_dtype,
        training_autocast_dtype=str(engine.dtype),
        sampled_action_shape=list(sampled.action.shape),
        synthetic_completed_step=completed_step,
        typed_object_value_mode=typed_object_values,
        completed_step_semantics="execution-phase fixture only; no preceding learned updates",
        input_provenance="synthetic descriptors/RGB/actions/positive-pair labels; no real dataset, pretrained encoder, or task success claim",
        scope="full production topology, ordinary training loss backward and AdamW updates, complete two-pass action generation",
        parameter_count=sum(p.numel() for p in model.parameters()),
        updates=rows,
        parameter_ledger=parameter_ledger,
        audit_caveat="Per-parameter paths are reported, not inferred from aggregate nonzero norms.",
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_data_passed=False,
        behavior_passed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("small", "production"), default="small")
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--raw-side", type=int, default=32)
    parser.add_argument(
        "--completed-step",
        type=int,
        default=0,
        help="synthetic execution-phase fixture, not an actual training history",
    )
    parser.add_argument(
        "--typed-object-values",
        choices=("legacy_selected_v1", "conditional_object_v1"),
        default="legacy_selected_v1",
    )
    parser.add_argument("--compute-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reference-support", choices=("full", "alternating"), default="full")
    parser.add_argument(
        "--outcome-mode",
        choices=("before_proposal_v1", "robot_world_before_proposal_v2"),
        default="before_proposal_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.updates < 1:
        raise ValueError("new output and positive update count required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(
            args.shape,
            args.variant,
            args.updates,
            args.raw_side,
            args.completed_step,
            args.typed_object_values,
            args.compute_dtype,
            args.batch_size,
            args.reference_support,
            args.outcome_mode,
        )
    except Exception as error:
        args.output.write_text(
            json.dumps(
                dict(passed=False, error=repr(error), shape=args.shape, variant=args.variant),
                indent=2,
            )
            + "\n"
        )
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
