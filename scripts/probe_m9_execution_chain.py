"""M9 synthetic CT-only influence audit through the actual bottom and codec.

This does not load a checkpoint, integrate a rollout, or claim learned utility.
It fixes the direct P3 route and varies one CT input source at a time. The
readout uses one velocity node's clean-field estimate at t=0.4, not a robot step.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.typed_transition import DYNAMIC_SOURCES, TransitionPlanEvidence
from clearvla.mainline.runtime.qualification import declared_runtime, synthetic_batch
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer
from clearvla.mainline.v120_core.time_domain_mmdit import EvidenceLatentMMDiTActionDecoder

ROOT = Path(__file__).resolve().parents[1]


def compact_config(config: ExperimentConfig) -> ExperimentConfig:
    """Explicit smaller dimensions, preserving the selected source/outlet modes."""
    result = replace(
        config,
        dimensions=replace(
            config.dimensions,
            hidden_size=32,
            num_heads=4,
            action_basis_tokens=2,
            visual_token_dim=16,
            goal_token_dim=16,
            patches_per_camera=64,
            goal_max_tokens=6,
        ),
        observation=replace(
            config.observation,
            feature_dim=16,
            address_route_dim=8,
            flow_iterations=2,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        top=replace(config.top, teacher_key_dim=8),
        bottom=replace(
            config.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_depth=1,
            controller_heads=4,
        ),
    )
    result.validate()
    return result


def replace_dynamic(
    evidence: TransitionPlanEvidence,
    name: str,
    value: Tensor,
) -> TransitionPlanEvidence:
    """Change the CT argument only; the actual P3 bottom argument stays fixed."""
    if name == "action":
        return replace(evidence, action=value)
    plan_fields = {
        "precision": "protected_policy_precision",
        "temporal": "temporal",
        "change": "state_change",
    }
    if name in plan_fields:
        return replace(evidence, plan=replace(evidence.plan, **{plan_fields[name]: value}))
    consequence = evidence.consequence
    if name in ("semantic", "geometry"):
        consequence = replace(consequence, effect=replace(consequence.effect, **{name: value}))
    elif name in ("semantic_interaction", "geometry_interaction"):
        consequence = replace(
            consequence,
            interaction=replace(consequence.interaction, **{name.split("_")[0]: value}),
        )
    else:
        raise ValueError(f"unknown CT source: {name}")
    protected = (
        consequence.factual_base
        + consequence.effect.combined()
        + consequence.interaction.combined()
    )
    consequence = replace(consequence, protected_consequence=protected)
    return replace(
        evidence,
        consequence=consequence,
        plan=replace(evidence.plan, protected_base=protected),
    )


def audit(
    config: ExperimentConfig, *, device: torch.device, bf16: bool, seed: int, updates: int = 0
) -> dict[str, Any]:
    """Run explicit initialization/one-update synthetic boundary checks."""
    if type(updates) is not int or updates not in (0, 1):
        raise ValueError("the bounded M9 audit permits exactly zero or one synthetic update")
    config = replace(
        config, runtime=replace(config.runtime, compute_dtype="bf16" if bf16 else "fp32")
    )
    if config.bottom.transition_condition_mode != "typed_plan_v1":
        raise ValueError("M9 CT source audit requires an explicit typed_plan_v1 config")
    torch.manual_seed(seed)
    model = ClearVLAMainlinePolicy(config).to(device).eval()
    model.set_training_step(
        0
    )  # Real initialization phase, not a fabricated completed-update clock.
    batch, normalizer = synthetic_batch(config, count=1, raw_side=48, device=device, seed=seed + 1)
    model.configure_action_normalizer(normalizer)
    dtype = torch.bfloat16 if bf16 else torch.float32
    learning = None
    if updates:
        optimizer, _ = build_optimizer(model, config)
        schedule_steps = config.optimizer.warmup_steps + 1
        engine = MainlineTrainingEngine(
            model=model,
            config=config,
            optimizer=optimizer,
            schedule=WarmupCosineSchedule(
                optimizer,
                warmup_steps=config.optimizer.warmup_steps,
                total_steps=schedule_steps,
                minimum_ratio=config.optimizer.min_lr_ratio,
            ),
            device=device,
        )
        model.train()
        result = engine.train_step(batch)
        learning = {
            "loss": float(result.loss),
            "actual_completed_updates": engine.global_step,
            "diagnostic_total_steps": schedule_steps,
            "configured_warmup_steps": config.optimizer.warmup_steps,
        }
        model.eval()
        model.zero_grad(set_to_none=True)
        model.set_training_step(engine.global_step)
    decoder = model.execution_bottom.decoder
    if not isinstance(decoder, EvidenceLatentMMDiTActionDecoder):
        raise TypeError("M9 audit requires the actual Evidence MMDiT decoder")
    command = decoder.terminal_controller.optional_command_head
    command_output_weight: Tensor | None = None
    if command is not None:
        command_output = command[-1]
        if not isinstance(command_output, nn.Linear):
            raise TypeError("binary terminal output must be the registered Linear owner")
        command_output_weight = command_output.weight
    command_output_zero = (
        None
        if command_output_weight is None
        else bool(torch.count_nonzero(command_output_weight) == 0)
    )
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=bf16):
        cache, _, _ = model.encode_online(batch.online)
    generator = torch.Generator().manual_seed(seed + 2)
    field = torch.randn(1, 24, 18, generator=generator).to(device)
    time = torch.tensor([0.4], device=device)
    direction = torch.randn(
        1,
        24,
        config.dimensions.action_basis_tokens,
        config.dimensions.hidden_size,
        generator=generator,
    ).to(device)
    current: dict[str, Any] = {"source": None}
    captured: dict[str, Any] = {}

    def isolate(_module: torch.nn.Module, args: tuple, kw: dict) -> tuple[tuple, dict]:
        evidence = TransitionPlanEvidence(
            kw["action_query"], kw["consequence"], kw["plan"], kw["seed"]
        )
        for name, value in evidence.dynamic().items():
            evidence = replace_dynamic(evidence, name, torch.zeros_like(value))
        if current["source"] is not None:
            evidence = replace_dynamic(evidence, current["source"], current["leaf"])
        return args, dict(
            kw,
            action_query=evidence.action,
            consequence=evidence.consequence,
            plan=evidence.plan,
            seed=evidence.seed,
        )

    def retain(_module: torch.nn.Module, _args: tuple, output: tuple) -> None:
        captured["transition"] = output[0]

    hooks = [
        model.transition.register_forward_pre_hook(isolate, with_kwargs=True),
        model.transition.register_forward_hook(retain),
    ]
    rng_before = torch.random.get_rng_state().clone()
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}

    def forward():
        with torch.autocast(device.type, dtype=dtype, enabled=bf16):
            step = model.velocity(cache, noisy_action_field=field, time=time)
            estimated = field + (1.0 - time[:, None, None]) * step.bottom.physical_velocity
            native = model.outlet_adapter.finalize(
                estimated,
                batch.online.history.action_state,
                codec_gripper_boundary=batch.online.history.codec_gripper_boundary,
                command_logits=step.bottom.gripper_command_logits,
            ).deployed_action
        return step, native

    records = []
    try:
        # Match autograd mode as well as weights/input; MHA may choose another
        # kernel in no_grad, which would contaminate a tiny forward difference.
        base_step, base_native = forward()
        zero_transition_exact = bool(torch.count_nonzero(captured["transition"].value) == 0)
        _, repeated_native = forward()
        baseline_repeat_exact = torch.equal(base_native, repeated_native)
        for name in DYNAMIC_SOURCES:
            leaf = direction.clone().requires_grad_()
            current.update(source=name, leaf=leaf)
            step, native = forward()
            typed_transition = model.transition.typed_transition
            if typed_transition is None:
                raise RuntimeError("typed transition disappeared during its source audit")
            value_projection = typed_transition.value_projections[name]
            if not isinstance(value_projection, nn.Linear):
                raise TypeError("typed CT source must retain its registered Linear owner")
            projection = value_projection.weight
            # A deterministic output covector, not a new optimization objective.
            output_weight = torch.linspace(0.5, 1.5, 6, device=device)
            gradient_rows = {}
            for row, label in ((0, "first"), (23, "last")):
                grads = torch.autograd.grad(
                    (native[:, row, :6] * output_weight).sum(),
                    (leaf, projection),
                    retain_graph=True,
                    allow_unused=False,
                )
                gradient_rows[label] = {
                    "input_vjp_l2": float(grads[0].float().norm()),
                    "producer_weight_vjp_l2": float(grads[1].float().norm()),
                    "finite": all(bool(torch.isfinite(g).all()) for g in grads),
                }
            # Binary argmax is not differentiable; inspect its true logit owner separately.
            logits = step.bottom.gripper_command_logits
            gripper_read = native[..., -1] if logits is None else logits[..., 1] - logits[..., 0]
            grip_owners = (
                (leaf,) if command_output_weight is None else (leaf, command_output_weight)
            )
            grip_gradients = torch.autograd.grad(gripper_read.sum(), grip_owners)
            grip_gradient = grip_gradients[0]
            direct_plan_same = all(
                torch.equal(getattr(step.compiled.plan, key), getattr(base_step.compiled.plan, key))
                for key in (
                    "protected_base",
                    "protected_policy_precision",
                    "temporal",
                    "state_change",
                )
            )
            change = native.detach().float() - base_native.detach().float()
            records.append(
                {
                    "source": name,
                    "direct_p3_plan_bit_exact": direct_plan_same,
                    "ct_value_rms": float(
                        captured["transition"].value.detach().float().square().mean().sqrt()
                    ),
                    "native_arm_delta_max_abs": float(change[..., :6].abs().max()),
                    "native_gripper_delta_max_abs": float(change[..., -1].abs().max()),
                    "gripper_gradient_readout": "continuous_native"
                    if logits is None
                    else "binary_logit_margin",
                    "gripper_input_vjp_l2": float(grip_gradient.float().norm()),
                    "command_output_weight_vjp_l2": (
                        None
                        if command_output_weight is None
                        else float(grip_gradients[1].float().norm())
                    ),
                    "gripper_vjp_finite": bool(torch.isfinite(grip_gradient).all()),
                    "native_arm_vjp": gradient_rows,
                }
            )
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "schema": "clearvla-m9-ct-execution-boundary-v1",
        "source": "synthetic_model_no_checkpoint",
        "optimizer_updates": updates,
        "learning": learning,
        "command_output_weight_exact_zero": command_output_zero,
        "readout": "one_velocity_node_clean_field_estimator_t0.4_not_rollout",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
            "dtype": str(dtype),
            "declared": declared_runtime(ROOT, platform.python_version(), str(torch.__version__)),
        },
        "config": config.as_dict(),
        "seed": seed,
        "source_digest": active_source_snapshot(ROOT).digest,
        "zero_dynamic_CT_exact_zero": zero_transition_exact,
        "zero_dynamic_native_max_abs": float(base_native.detach().abs().max()),
        "matched_context_baseline_repeat_exact": baseline_repeat_exact,
        "diagnostics_preserve_cpu_rng": torch.equal(rng_before, torch.random.get_rng_state()),
        "buffers_unchanged": all(
            torch.equal(value, buffers_before[name]) for name, value in model.named_buffers()
        ),
        "parameter_grad_buffers_unwritten": all(p.grad is None for p in model.parameters()),
        "sources": records,
        "limitations": [
            "No real-data-trained weights, real image features or task success evidence.",
            "A nonzero derivative establishes a path, not useful learned control.",
            "The zero-input origin and single-source stimuli are CT-only diagnostic interventions.",
            "FP32/AMP output rounding can hide small forward differences despite nonzero VJPs.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--compact", action="store_true", help="Explicit H32 / synthetic64-patch contract fixture"
    )
    parser.add_argument("--seed", type=int, default=9231)
    parser.add_argument(
        "--updates",
        type=int,
        choices=(0, 1),
        default=0,
        help="Optional actual one-batch update; configured warmup unchanged",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence or select a new path")
    torch.set_num_threads(1)
    config = load_config(args.config)
    if args.compact:
        config = compact_config(config)
    result = audit(
        config,
        device=torch.device(args.device),
        bf16=args.dtype == "bf16",
        seed=args.seed,
        updates=args.updates,
    )
    result["compact_fixture"] = args.compact
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "runtime", "zero_dynamic_CT_exact_zero")}
        )
    )


if __name__ == "__main__":
    main()
