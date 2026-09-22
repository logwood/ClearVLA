"""Full reduced Q5-lifecycle audit for the target-action information boundary.

The intervention changes the original goal, then fixes the single target
posterior and the directly supervised physical A0 at their producer seams.
Every descendant is rebuilt.  Any change in static P1, W, P2/P3, controlled
transition or the complete velocity output is therefore a forbidden post-A0
language path rather than a cached-value artefact.  The audit then executes
proposal Q5, one W rebuild, refined Q5 and both endpoint head reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearvla.mainline import config as config_module  # noqa: E402, I001
from clearvla.mainline import interfaces  # noqa: E402
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy  # noqa: E402
from clearvla.mainline.runtime.sampling import (  # noqa: E402
    sample_refined_cached_action_with_cache,
)
from clearvla.mainline.runtime.flow_schedule import (  # noqa: E402
    DeploymentFlowSchedule,
)
from clearvla.tools import mainline_equivalence as equivalence  # noqa: E402


MODE = "target_action_bottleneck_v1"


def _leaves(value: Any, prefix: str = "root") -> dict[str, Tensor]:
    if isinstance(value, Tensor):
        return {prefix: value}
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Tensor] = {}
        for item in fields(value):
            result.update(_leaves(getattr(value, item.name), f"{prefix}.{item.name}"))
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            result.update(_leaves(item, f"{prefix}.{key}"))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(_leaves(item, f"{prefix}.{index}"))
        return result
    return {}


def _compare(left: Any, right: Any, *, prefix: str) -> tuple[bool, float, list[str]]:
    left_leaves = _leaves(left, prefix)
    right_leaves = _leaves(right, prefix)
    mismatched: list[str] = []
    maximum = 0.0
    if set(left_leaves) != set(right_leaves):
        missing = sorted(set(left_leaves) ^ set(right_leaves))
        return False, float("inf"), [f"leaf-set:{name}" for name in missing]
    for name, left_value in left_leaves.items():
        right_value = right_leaves[name]
        if tuple(left_value.shape) != tuple(right_value.shape):
            mismatched.append(f"shape:{name}")
            continue
        if left_value.dtype != right_value.dtype:
            mismatched.append(f"dtype:{name}")
            continue
        if not torch.equal(left_value, right_value):
            mismatched.append(name)
            if left_value.is_floating_point() or left_value.is_complex():
                delta = float(
                    (left_value.detach().float() - right_value.detach().float())
                    .abs()
                    .amax()
                )
                maximum = max(maximum, delta)
            else:
                maximum = float("inf")
    return not mismatched, maximum, mismatched


def _max_delta(left: Tensor, right: Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().amax())


def run(*, fixture: Path, seed: int) -> dict[str, object]:
    modules = {"config": config_module, "interfaces": interfaces}
    base = equivalence.build_reduced_equivalence_config(modules)
    config = replace(
        base,
        top=replace(base.top, p2_spatial_intent_mode=MODE),
    )
    config.validate()
    if not fixture.is_file():
        equivalence.create_fixture(fixture, seed=seed)
    payload = equivalence._load_fixture(fixture)
    batch = equivalence._training_batch(modules, config, payload)

    torch.manual_seed(seed)
    model = ClearVLAMainlinePolicy(config).to(
        device=torch.device("cpu"), dtype=torch.float32
    )
    model.eval()
    noisy = payload["tensors"]["dynamic_physical_field"].clone()
    time = torch.tensor([0.5], dtype=torch.float32)

    transition_rows: list[Any] = []
    target_score_rows: list[Tensor] = []

    def capture_transition(_module, _inputs, output):
        transition_rows.append(output[0])

    def capture_target_score(_module, _inputs, output):
        target_score_rows.append(output.detach().clone())

    transition_handle = model.transition.register_forward_hook(capture_transition)
    if model.intent.organizer.target_score is None:
        raise RuntimeError("target-action system probe requires the target score seam")
    score_capture_handle = model.intent.organizer.target_score.register_forward_hook(
        capture_target_score
    )
    with torch.no_grad():
        base_cache, base_training, _ = model.encode_online(
            batch.online,
            training_mask=False,
            geometry_supervision=True,
            collect_diagnostics=False,
        )
        base_output = model.velocity(
            base_cache,
            noisy_action_field=noisy.clone(),
            time=time.clone(),
            collect_diagnostics=False,
        )
    score_capture_handle.remove()
    if len(target_score_rows) != 1:
        raise RuntimeError("system probe did not capture the target-score seam")
    baseline_target_score = target_score_rows[0]

    baseline_intent = base_training.top.intent
    baseline_a0 = base_cache.top.action_proposal
    changed_goal = replace(
        batch.online.goal,
        tokens=(
            -2.75 * batch.online.goal.tokens
            + torch.linspace(
                -1.0,
                1.0,
                int(batch.online.goal.tokens.shape[-1]),
                dtype=batch.online.goal.tokens.dtype,
            )[None, None]
        ),
    )
    changed_online = replace(batch.online, goal=changed_goal)
    preoverride_action: list[Tensor] = []
    def fix_target_score(_module, _inputs, output):
        if tuple(output.shape) != tuple(baseline_target_score.shape):
            raise RuntimeError("target-score intervention changed shape")
        return baseline_target_score.to(device=output.device, dtype=output.dtype)

    def fix_a0(_module, _inputs, output):
        preoverride_action.append(output.action_prediction.detach().clone())
        return replace(output, action_prediction=baseline_a0)

    score_handle = model.intent.organizer.target_score.register_forward_hook(
        fix_target_score
    )
    coarse_handle = model.intent.coarse_action.register_forward_hook(fix_a0)
    try:
        with torch.no_grad():
            changed_cache, changed_training, _ = model.encode_online(
                changed_online,
                training_mask=False,
                geometry_supervision=True,
                collect_diagnostics=False,
            )
            changed_output = model.velocity(
                changed_cache,
                noisy_action_field=noisy.clone(),
                time=time.clone(),
                collect_diagnostics=False,
            )
    finally:
        score_handle.remove()
        coarse_handle.remove()
        transition_handle.remove()

    if len(transition_rows) != 2 or len(preoverride_action) != 1:
        raise RuntimeError("system probe did not observe the expected lifecycle")

    language_deltas = {
        "protected_goal": _max_delta(
            baseline_intent.protected_goal_set,
            changed_training.top.intent.protected_goal_set,
        ),
        "public_interval": _max_delta(
            baseline_intent.public_interval_carrier,
            changed_training.top.intent.public_interval_carrier,
        ),
        "policy_interval": _max_delta(
            baseline_intent.policy_interval_context,
            changed_training.top.intent.policy_interval_context,
        ),
        "temporal": _max_delta(
            baseline_intent.temporal_queries,
            changed_training.top.intent.temporal_queries,
        ),
        "natural_a0_before_fix": _max_delta(
            baseline_a0,
            preoverride_action[0],
        ),
    }
    p_equal = torch.equal(
        base_cache.top.intent.target_posterior,
        changed_cache.top.intent.target_posterior,
    )
    a0_equal = torch.equal(base_cache.top.action_proposal, changed_cache.top.action_proposal)

    static_equal, static_delta, static_mismatch = _compare(
        {
            "factual_dock": base_cache.factual_dock,
            "candidate_world": base_cache.top.candidate_world,
            "transition_source": base_cache.transition_source,
        },
        {
            "factual_dock": changed_cache.factual_dock,
            "candidate_world": changed_cache.top.candidate_world,
            "transition_source": changed_cache.transition_source,
        },
        prefix="static",
    )
    dynamic_equal, dynamic_delta, dynamic_mismatch = _compare(
        {
            "bottom": base_output.bottom,
            "compiled": base_output.compiled,
            "transition": transition_rows[0],
        },
        {
            "bottom": changed_output.bottom,
            "compiled": changed_output.compiled,
            "transition": transition_rows[1],
        },
        prefix="dynamic",
    )
    sampler_transition_calls = [0, 0]
    sampler_world_rebuilds = [0, 0]
    active_sampler = 0

    def count_sampler_transition(_module, _inputs, _output):
        sampler_transition_calls[active_sampler] += 1

    sampler_transition_handle = model.transition.register_forward_hook(
        count_sampler_transition
    )
    q5_schedule = DeploymentFlowSchedule.same_nfe_power_five(exponent=1.25)
    original_forward_w1 = model.world.dynamics.forward_w1

    def counted_forward_w1(*args, **kwargs):
        sampler_world_rebuilds[active_sampler] += 1
        return original_forward_w1(*args, **kwargs)

    model.world.dynamics.forward_w1 = counted_forward_w1
    try:
        active_sampler = 0
        base_sample, base_refined_cache = sample_refined_cached_action_with_cache(
            model,
            base_cache,
            config,
            initial_physical_noise=noisy.clone(),
            collect_diagnostics=False,
            dtype=torch.float32,
            flow_schedule=q5_schedule,
        )
        active_sampler = 1
        changed_sample, changed_refined_cache = (
            sample_refined_cached_action_with_cache(
                model,
                changed_cache,
                config,
                initial_physical_noise=noisy.clone(),
                collect_diagnostics=False,
                dtype=torch.float32,
                flow_schedule=q5_schedule,
            )
        )
    finally:
        sampler_transition_handle.remove()
        model.world.dynamics.forward_w1 = original_forward_w1
    q5_equal, q5_delta, q5_mismatch = _compare(
        base_sample,
        changed_sample,
        prefix="q5",
    )
    refined_world_equal, refined_world_delta, refined_world_mismatch = _compare(
        {
            "candidate_world": base_refined_cache.top.candidate_world,
            "predicted_dynamics": base_refined_cache.top.predicted_dynamics,
            "action_condition": base_refined_cache.top.action_condition,
            "factual_dock": base_refined_cache.factual_dock,
            "transition_source": base_refined_cache.transition_source,
        },
        {
            "candidate_world": changed_refined_cache.top.candidate_world,
            "predicted_dynamics": changed_refined_cache.top.predicted_dynamics,
            "action_condition": changed_refined_cache.top.action_condition,
            "factual_dock": changed_refined_cache.factual_dock,
            "transition_source": changed_refined_cache.transition_source,
        },
        prefix="refined_world",
    )
    expected_velocity_calls = 2 * (int(config.runtime.inference_steps) + 1)

    target_score = model.intent.organizer.target_score
    physical_delta_head = model.transition.physical_delta_head
    if target_score is None or physical_delta_head is None:
        raise RuntimeError("nonzero Q5 audit requires target and physical CT heads")
    with torch.no_grad():
        target_score.weight.copy_(
            torch.linspace(
                -0.15,
                0.15,
                target_score.weight.numel(),
                dtype=target_score.weight.dtype,
            ).reshape_as(target_score.weight)
        )
        physical_delta_head.weight.copy_(
            torch.linspace(
                -0.02,
                0.02,
                physical_delta_head.weight.numel(),
                dtype=physical_delta_head.weight.dtype,
            ).reshape_as(physical_delta_head.weight)
        )

    nonzero_score_rows: list[Tensor] = []

    def capture_nonzero_score(_module, _inputs, output):
        nonzero_score_rows.append(output.detach().clone())

    nonzero_score_handle = target_score.register_forward_hook(capture_nonzero_score)
    with torch.no_grad():
        nonzero_base_cache, nonzero_base_training, _ = model.encode_online(
            batch.online,
            training_mask=False,
            geometry_supervision=True,
            collect_diagnostics=False,
        )
    nonzero_score_handle.remove()
    if len(nonzero_score_rows) != 1:
        raise RuntimeError("nonzero Q5 audit did not capture baseline target score")
    fixed_nonzero_score = nonzero_score_rows[0]
    nonzero_base_a0 = nonzero_base_cache.top.action_proposal
    natural_changed_scores: list[Tensor] = []
    natural_changed_a0: list[Tensor] = []

    def fix_nonzero_score(_module, _inputs, output):
        natural_changed_scores.append(output.detach().clone())
        return fixed_nonzero_score.to(device=output.device, dtype=output.dtype)

    def fix_nonzero_a0(_module, _inputs, output):
        natural_changed_a0.append(output.action_prediction.detach().clone())
        return replace(output, action_prediction=nonzero_base_a0)

    nonzero_score_handle = target_score.register_forward_hook(fix_nonzero_score)
    nonzero_a0_handle = model.intent.coarse_action.register_forward_hook(
        fix_nonzero_a0
    )
    try:
        with torch.no_grad():
            nonzero_changed_cache, nonzero_changed_training, _ = model.encode_online(
                changed_online,
                training_mask=False,
                geometry_supervision=True,
                collect_diagnostics=False,
            )
    finally:
        nonzero_score_handle.remove()
        nonzero_a0_handle.remove()
    if len(natural_changed_scores) != 1 or len(natural_changed_a0) != 1:
        raise RuntimeError("nonzero Q5 intervention lifecycle was incomplete")

    nonzero_transition_deltas: list[Tensor] = []

    def capture_nonzero_delta(_module, _inputs, output):
        transition = output[0]
        delta_v = getattr(transition, "delta_v", None)
        if not isinstance(delta_v, Tensor):
            raise RuntimeError("nonzero Q5 audit lost the physical CT ABI")
        nonzero_transition_deltas.append(delta_v.detach().clone())

    nonzero_transition_handle = model.transition.register_forward_hook(
        capture_nonzero_delta
    )
    try:
        nonzero_base_sample, nonzero_base_refined = (
            sample_refined_cached_action_with_cache(
                model,
                nonzero_base_cache,
                config,
                initial_physical_noise=noisy.clone(),
                collect_diagnostics=False,
                dtype=torch.float32,
                flow_schedule=q5_schedule,
            )
        )
        nonzero_changed_sample, nonzero_changed_refined = (
            sample_refined_cached_action_with_cache(
                model,
                nonzero_changed_cache,
                config,
                initial_physical_noise=noisy.clone(),
                collect_diagnostics=False,
                dtype=torch.float32,
                flow_schedule=q5_schedule,
            )
        )
    finally:
        nonzero_transition_handle.remove()
    nonzero_q5_equal, nonzero_q5_delta, nonzero_q5_mismatch = _compare(
        nonzero_base_sample,
        nonzero_changed_sample,
        prefix="nonzero_q5",
    )
    nonzero_descendants_equal, nonzero_descendants_delta, nonzero_descendants_mismatch = (
        _compare(
            {
                "factual": nonzero_base_cache.factual_dock,
                "candidate_world": nonzero_base_refined.top.candidate_world,
                "action_condition": nonzero_base_refined.top.action_condition,
                "transition_source": nonzero_base_refined.transition_source,
            },
            {
                "factual": nonzero_changed_cache.factual_dock,
                "candidate_world": nonzero_changed_refined.top.candidate_world,
                "action_condition": nonzero_changed_refined.top.action_condition,
                "transition_source": nonzero_changed_refined.transition_source,
            },
            prefix="nonzero_descendants",
        )
    )
    nonzero_delta_rms = [
        float(delta.float().square().mean().sqrt())
        for delta in nonzero_transition_deltas
    ]
    checks = {
        "goal_intervention_is_nontrivial": all(
            language_deltas[name] > 0.0
            for name in (
                "protected_goal",
                "public_interval",
                "policy_interval",
                "temporal",
            )
        ),
        "natural_a0_would_change_without_the_fix": (
            language_deltas["natural_a0_before_fix"] > 0.0
        ),
        "target_posterior_is_fixed_exactly": p_equal,
        "a0_is_fixed_exactly": a0_equal,
        "rebuilt_static_descendants_are_bitwise_equal": static_equal,
        "single_ode_velocity_and_transition_are_bitwise_equal": dynamic_equal,
        "proposal_q5_refined_q5_endpoint_are_bitwise_equal": q5_equal,
        "single_world_rebuild_is_bitwise_equal": refined_world_equal,
        "q5_lifecycle_call_count_is_exact": (
            sampler_transition_calls
            == [expected_velocity_calls, expected_velocity_calls]
            and sampler_world_rebuilds == [1, 1]
        ),
        "registered_q5_schedule_and_step_context_are_active": (
            base_sample.flow_schedule_identity is not None
            and base_sample.flow_schedule_identity.get("candidate_id") == "Q5/Q5"
            and base_sample.flow_schedule_identity.get("fingerprint")
            == q5_schedule.fingerprint
            and tuple(base_sample.step_sizes) == tuple(q5_schedule.refined.step_sizes)
            and torch.equal(
                base_sample.step_times,
                torch.tensor(q5_schedule.refined.query_times, dtype=torch.float32),
            )
            and float(base_sample.metrics["sampling_flow_step_context_enabled"])
            == 1.0
        ),
        "nonzero_target_score_makes_the_fixed_p_seam_nontrivial": (
            float(
                nonzero_base_training.top.intent.target_posterior.detach()
                .float()
                .std(unbiased=False)
            )
            > 0.0
            and _max_delta(fixed_nonzero_score, natural_changed_scores[0]) > 0.0
            and _max_delta(nonzero_base_a0, natural_changed_a0[0]) > 0.0
        ),
        "nonzero_ct_q5_keeps_fixed_p_a0_language_invariance": (
            nonzero_q5_equal
            and nonzero_descendants_equal
            and torch.equal(
                nonzero_base_cache.top.intent.target_posterior,
                nonzero_changed_cache.top.intent.target_posterior,
            )
            and torch.equal(
                nonzero_base_a0,
                nonzero_changed_cache.top.action_proposal,
            )
        ),
        "nonzero_ct_is_exercised_at_every_q5_velocity_call": (
            len(nonzero_delta_rms) == 2 * expected_velocity_calls
            and min(nonzero_delta_rms) > 0.0
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "target-action system contract failed: "
            f"{failed}; static={static_mismatch[:8]}; dynamic={dynamic_mismatch[:8]}; "
            f"q5={q5_mismatch[:8]}; refined={refined_world_mismatch[:8]}; "
            f"nonzero_q5={nonzero_q5_mismatch[:8]}; "
            f"nonzero_desc={nonzero_descendants_mismatch[:8]}"
        )
    return {
        "schema": "clearvla-target-action-bottleneck-q5-lifecycle-v4",
        "seed": int(seed),
        "mode": MODE,
        "fixture": str(fixture.resolve()),
        "language_deltas": language_deltas,
        "static_max_abs_delta": static_delta,
        "dynamic_max_abs_delta": dynamic_delta,
        "static_mismatch": static_mismatch,
        "dynamic_mismatch": dynamic_mismatch,
        "q5_max_abs_delta": q5_delta,
        "q5_mismatch": q5_mismatch,
        "refined_world_max_abs_delta": refined_world_delta,
        "refined_world_mismatch": refined_world_mismatch,
        "sampler_transition_calls": sampler_transition_calls,
        "sampler_world_rebuilds": sampler_world_rebuilds,
        "nonzero_target_score_delta_max": _max_delta(
            fixed_nonzero_score,
            natural_changed_scores[0],
        ),
        "nonzero_natural_a0_delta_max": _max_delta(
            nonzero_base_a0,
            natural_changed_a0[0],
        ),
        "nonzero_ct_delta_rms_min": min(nonzero_delta_rms),
        "nonzero_ct_delta_rms_max": max(nonzero_delta_rms),
        "nonzero_q5_max_abs_delta": nonzero_q5_delta,
        "nonzero_q5_mismatch": nonzero_q5_mismatch,
        "nonzero_descendants_max_abs_delta": nonzero_descendants_delta,
        "nonzero_descendants_mismatch": nonzero_descendants_mismatch,
        "checks": checks,
        "scope_note": (
            "This proves the deterministic post-A0 language boundary for the reduced "
            "proposal-Q5, one-W-rebuild, refined-Q5 and endpoint lifecycle. It does "
            "not prove that learned p identifies the correct real object or improves "
            "CALVIN behavior."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            Path(tempfile.gettempdir())
            / "clearvla_target_action_bottleneck_system_fixture.pt"
        ),
    )
    parser.add_argument("--seed", type=int, default=202_609_22)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(fixture=args.fixture, seed=args.seed)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
