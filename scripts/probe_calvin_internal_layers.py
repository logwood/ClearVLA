#!/usr/bin/env python3
"""Read-only CALVIN language probe for the active G/S/W/P path.

The public bridge probe only compares the decoded action.  This diagnostic
keeps one fixed causal observation and one fixed physical noise field while
changing the instruction, then records the tensors at the named seams:

* protected goal and S public/typed carriers;
* the four-row coarse proposal and W dynamics;
* P2 spatial and temporal posteriors (captured without changing the source);
* the resulting dynamic effect/velocity.

No model parameters, checkpoint, dataset, or simulator state are changed.
The small runtime monkey-patch only observes the already-computed masked
softmax calls and is restored immediately after each forward.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import importlib.util
import importlib.abc
import importlib.machinery
import typing
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

if sys.version_info < (3, 10) and not hasattr(typing, "TypeAlias"):
    # ``TypeAlias`` is metadata only in the frozen flow-solver protocols.  The
    # validated CALVIN Python 3.9 runtime predates its stdlib definition.
    typing.TypeAlias = object  # type: ignore[attr-defined]

# The validated CALVIN venv owns the CUDA/Torch build, while the companion
# LIBERO venv already contains the pure-Python ``transformers`` package used by
# the relocated DINO loader.  Append that site directory *after* importing
# Torch so its different Torch wheel cannot shadow the CALVIN runtime.
_extra_site = os.environ.get("CLEARVLA_EXTRA_SITE", "").strip()
if _extra_site and _extra_site not in sys.path:
    sys.path.append(_extra_site)


def _install_py39_bspine_import_compat() -> None:
    """Load the unchanged B-spine source on Python 3.9 for this probe only.

    The active source uses a PEP-604 class union as a runtime type alias.  The
    CALVIN inference venv is Python 3.9, where evaluating that alias raises at
    import time.  This diagnostic does not alter the checkout: it preloads a
    transient module whose sole textual transformation replaces the alias by
    ``object`` (the alias is used only by postponed annotations/cast).
    """

    if sys.version_info >= (3, 10):
        return
    module_name = "clearvla.mainline.v120_core.bspine"
    if module_name in sys.modules:
        return
    configured_root = os.environ.get("CLEARVLA_REPO_ROOT", "").strip()
    root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path(__file__).resolve().parents[1]
    )
    source_path = root / "clearvla" / "mainline" / "v120_core" / "bspine.py"
    source = source_path.read_text(encoding="utf-8")
    marker = re.search(
        r"(?ms:^BSpineModule\s*=\s*\(.*?^\)\s*$)"
        r"|(?m:^BSpineModule\s*=.*$)",
        source,
    )
    if marker is None:
        raise RuntimeError("the expected B-spine type-alias marker was not found")
    source = source[: marker.start()] + "BSpineModule = object" + source[marker.end() :]
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create the transient B-spine import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)


class _Py39ClearVLASourceLoader(importlib.machinery.SourceFileLoader):
    """Transiently postpone annotations in legacy modules lacking the future import."""

    def get_data(self, path: str) -> bytes:  # noqa: D401
        raw = super().get_data(path)
        if not str(path).endswith(".py"):
            return raw
        source = raw.decode("utf-8")
        if "from __future__ import annotations" not in source:
            source = "from __future__ import annotations\n" + source
        # Python 3.9 also evaluates PEP-604 expressions passed as the first
        # argument to typing.cast.  They are type-only here, so use a neutral
        # object target in this transient source view.
        source = re.sub(
            r"cast\(tuple\[object, \.\.\.\] \| list\[object\],",
            "cast(object,",
            source,
        )
        for alias in ("ParameterValue", "WorldActionCondition"):
            source = re.sub(
                rf"(?m)^{alias}\s*=.*$",
                f"{alias} = object",
                source,
            )
        return source.encode("utf-8")

    def get_code(self, fullname: str):
        # Bypass valid Python-3.9 bytecode so the transient source transform is
        # guaranteed to apply.  No .pyc file is written by this path.
        source = self.get_data(self.path)
        return self.source_to_code(source, self.path)


class _Py39ClearVLASourceFinder(importlib.abc.MetaPathFinder):
    """Use the unchanged checkout through a Python-3.9-compatible source view."""

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if not (fullname == "clearvla" or fullname.startswith("clearvla.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.origin, str) or not spec.origin.endswith(".py"):
            return spec
        spec.loader = _Py39ClearVLASourceLoader(fullname, spec.origin)
        return spec


if sys.version_info < (3, 10):
    # This is an in-memory loader only; no checkout file is rewritten.
    sys.meta_path.insert(0, _Py39ClearVLASourceFinder())


_install_py39_bspine_import_compat()

from clearvla.mainline.interfaces import (
    CurrentObservation,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
)
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import HistorySnapshot
from clearvla.mainline.model import compiler as compiler_module

from scripts.probe_calvin_language_counterfactual import (
    INSTRUCTIONS,
    _as_policy_observation,
    _environment,
    _fingerprint_observation,
    _layout_states,
    _state_for_initial_condition,
)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _summary(value: Any, *, include_array: bool = False) -> dict[str, Any]:
    arr = _array(value)
    result: dict[str, Any] = {
        "shape": list(arr.shape),
        "rms": float(np.sqrt(np.mean(np.square(arr, dtype=np.float64)))),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "std": float(np.std(arr, dtype=np.float64)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if include_array:
        result["array"] = arr.tolist()
    return result


def _rmse(left: Any, right: Any) -> float:
    a = _array(left)
    b = _array(right)
    if a.shape != b.shape:
        raise ValueError(f"cannot compare shapes {a.shape} and {b.shape}")
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64) - b.astype(np.float64)))))


def _build_online(policy: ClearVLACheckpointPolicy, history: HistorySnapshot, instruction: str) -> OnlinePolicyInput:
    """Mirror act_with_input up to (but excluding) action sampling."""

    history.validate()
    config = policy.bundle.config
    goal_tokens, goal_mask = policy._goal(instruction)
    dino, images = policy.encoder.encode(history.rgb_history, policy.preprocessing)
    raw_rgb = (
        torch.from_numpy(np.ascontiguousarray(images))
        .permute(0, 1, 4, 2, 3)
        .unsqueeze(0)
        .to(device=policy.device, dtype=torch.float32)
        .div_(255.0)
    )
    action_state = policy._normal(history.action_state[None], action=True)
    executed_action_history = policy._normal(
        history.executed_action_history[None], action=True
    )
    profile = resolve_action_state_profile(config.data.data_profile)
    codec_gripper_boundary = action_state[:, -1:]
    if profile.gripper_transition_boundary == "previous_command":
        codec_gripper_boundary = executed_action_history[:, -1, -1:]
    online = OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=dino.unsqueeze(0),
            raw_rgb=raw_rgb,
        ),
        history=ObservableHistory(
            state=policy._normal(history.state[None], action=False),
            action_state=action_state,
            codec_gripper_boundary=codec_gripper_boundary,
            state_history=policy._normal(history.state_history[None], action=False),
            executed_action_history=executed_action_history,
        ),
        goal=GoalCondition(
            tokens=goal_tokens.to(policy.device),
            mask=goal_mask.to(policy.device),
        ),
    )
    online.validate(config)
    return online


def _capture_velocity(model: Any, cache: Any, *, time_value: float, noise: torch.Tensor) -> tuple[Any, dict[str, Any]]:
    """Run one dynamic forward and observe the four P2 softmax calls."""

    captured: list[dict[str, Any]] = []
    original = compiler_module._safe_masked_softmax

    def observer(logit: torch.Tensor, support: torch.Tensor, *, dim: int) -> torch.Tensor:
        probability = original(logit, support, dim=dim)
        captured.append(
            {
                "shape": list(probability.shape),
                "dim": int(dim),
                "probability": probability.detach().float().cpu().numpy(),
                "logit": logit.detach().float().cpu().numpy(),
                "support": support.detach().cpu().numpy().astype(bool),
            }
        )
        return probability

    compiler_module._safe_masked_softmax = observer
    try:
        output = model.velocity(
            cache,
            noisy_action_field=noise,
            time=torch.full(
                (int(noise.shape[0]),),
                float(time_value),
                device=noise.device,
                dtype=torch.float32,
            ),
            collect_diagnostics=True,
        )
    finally:
        compiler_module._safe_masked_softmax = original

    if len(captured) < 4:
        raise RuntimeError(f"expected at least four P2 softmax calls, got {len(captured)}")
    # In the active compiler the first two calls are spatial semantic/geometry;
    # the next two are temporal type posterior and its neutral audit baseline.
    posterior = {
        "spatial_semantic": captured[0],
        "spatial_geometry": captured[1],
        "temporal": captured[2],
        "temporal_neutral": captured[3],
        "call_count": len(captured),
    }
    return output, posterior


def _tensor_bundle(cache: Any, output: Any, posterior: Mapping[str, Any], static_metrics: Mapping[str, Any]) -> dict[str, Any]:
    intent = cache.top.intent
    facts = cache.top.belief
    condition = cache.top.action_condition
    dynamics = cache.top.predicted_dynamics
    compiled = output.compiled
    bundle: dict[str, Any] = {
        "facts": {
            "coordinates": _summary(facts.camera_coordinates, include_array=True),
            "validity": _summary(facts.validity, include_array=True),
            "semantic": _summary(facts.semantic),
            "appearance": _summary(facts.appearance),
            "geometry": _summary(facts.geometry),
        },
        "protected_goal": _summary(intent.protected_goal_set, include_array=True),
        "goal_attention": _summary(intent.goal_attention, include_array=True),
        "interval_goal_attention": _summary(
            intent.interval_goal_attention, include_array=True
        ),
        "interval_history_attention": _summary(
            intent.interval_history_attention, include_array=True
        ),
        "interval_object_attention": _summary(
            intent.interval_object_attention, include_array=True
        ),
        "public_interval_carrier": _summary(intent.public_interval_carrier, include_array=True),
        "policy_interval_context": _summary(intent.policy_interval_context, include_array=True),
        "typed_relevance_mass": _summary(intent.typed_relevance_mass, include_array=True),
        "typed_relevance_value": _summary(intent.typed_relevance_value),
        "typed_policy_components": _summary(intent.typed_policy_components, include_array=True),
        "coarse_interval_action": _summary(condition.interval_action, include_array=True),
        "coarse_interval_delta": _summary(condition.interval_delta, include_array=True),
        "w_semantic_delta": _summary(dynamics.semantic_delta),
        "w_transport_mean": _summary(dynamics.transport_mean),
        "p2_effect_semantic": _summary(compiled.effect.semantic),
        "p2_effect_geometry": _summary(compiled.effect.geometry),
        "p2_consequence": _summary(compiled.consequence.protected_consequence),
        "p3_temporal": _summary(compiled.plan.temporal),
        "p3_state_change": _summary(compiled.plan.state_change),
        "physical_velocity": _summary(output.bottom.physical_velocity),
        "metrics": {
            str(key): float(value.detach().float().cpu().item())
            for key, value in output.metrics.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0 and bool(torch.isfinite(value).all())
        },
        "static_metrics": {
            str(key): float(value.detach().float().cpu().item())
            for key, value in static_metrics.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0 and bool(torch.isfinite(value).all())
        },
        "posterior": {
            str(name): (
                {key: value for key, value in item.items() if key not in {"probability", "logit", "support"}}
                | {
                    "probability": np.asarray(item["probability"], dtype=np.float32).tolist(),
                    "logit": np.asarray(item["logit"], dtype=np.float32).tolist(),
                    "support_count": int(np.asarray(item["support"]).sum()),
                }
            )
            for name, item in posterior.items()
            if isinstance(item, Mapping)
        },
    }
    return bundle


def _difference_bundle(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "protected_goal",
        "public_interval_carrier",
        "policy_interval_context",
        "typed_relevance_mass",
        "typed_relevance_value",
        "typed_policy_components",
        "coarse_interval_action",
        "coarse_interval_delta",
        "w_semantic_delta",
        "w_transport_mean",
        "p2_effect_semantic",
        "p2_effect_geometry",
        "p2_consequence",
        "p3_temporal",
        "p3_state_change",
        "physical_velocity",
    )
    result: dict[str, Any] = {}
    for name in names:
        a = left[name]
        b = right[name]
        # The summaries retain full arrays only for the fields where object
        # identity is most useful; for the others use scalar RMS deltas.
        if "array" in a and "array" in b:
            result[name] = {"rmse": _rmse(a["array"], b["array"])}
        else:
            result[name] = {
                "rms_difference_of_summary": float(abs(float(a["rms"]) - float(b["rms"]))),
                "mean_difference_of_summary": float(abs(float(a["mean"]) - float(b["mean"]))),
            }
    for name in ("spatial_semantic", "spatial_geometry", "temporal", "temporal_neutral"):
        a = left["posterior"][name]["probability"]
        b = right["posterior"][name]["probability"]
        result[f"posterior_{name}"] = {"rmse": _rmse(a, b)}
    return result


def run_probe(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    dataset_root: Path,
    output: Path,
    layouts: tuple[str, ...],
    instructions: tuple[str, ...],
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    time_value: float,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch_device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    env = _environment(dataset_root, show_gui=False)
    records: dict[str, Any] = {}
    try:
        all_layouts = _layout_states()
        for layout_name in layouts:
            if layout_name not in all_layouts:
                raise KeyError(f"unknown layout {layout_name!r}")
            initial_state = all_layouts[layout_name]
            robot_obs, scene_obs = _state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            observation = env.get_obs()
            policy_observation_value = _as_policy_observation(
                observation, np.zeros(7, dtype=np.float32)
            )
            from clearvla.simulation.history import CausalHistory

            timeline = CausalHistory()
            timeline.reset(policy_observation_value, reset_action=policy_observation_value.action_state)
            history = timeline.snapshot()
            layout_record: dict[str, Any] = {
                "initial_state": {str(k): v for k, v in initial_state.items()},
                "observation": _fingerprint_observation(observation),
                "instructions": {},
                "pairwise_vs_blue_right": {},
            }

            # One fixed source field is used for every instruction/layout.  It
            # is deterministic and avoids conflating language effects with
            # the ODE's generator draws.
            noise = torch.randn(
                1,
                policy.bundle.config.dimensions.action_horizon,
                policy.bundle.model.outlet_adapter.physical_dim,
                device=policy.device,
                dtype=torch.float32,
                generator=torch.Generator(device=policy.device).manual_seed(12345),
            )
            for instruction in instructions:
                online = _build_online(policy, history, instruction)
                cache, static_metrics = deployment_cache(
                    policy.bundle.model,
                    online,
                    policy.bundle.config,
                    collect_diagnostics=True,
                )
                output_value, posterior = _capture_velocity(
                    policy.bundle.model,
                    cache,
                    time_value=time_value,
                    noise=noise,
                )
                layout_record["instructions"][instruction] = _tensor_bundle(
                    cache,
                    output_value,
                    posterior,
                    static_metrics,
                )
            baseline = layout_record["instructions"][instructions[0]]
            for instruction in instructions[1:]:
                layout_record["pairwise_vs_blue_right"][instruction] = _difference_bundle(
                    layout_record["instructions"][instruction], baseline
                )
            records[layout_name] = layout_record
    finally:
        env.close()
    result = {
        "schema": "clearvla-calvin-internal-language-v1",
        "checkpoint": str(checkpoint),
        "time_value": float(time_value),
        "layouts": records,
        "instructions": list(instructions),
        "notes": {
            "noise": "fixed standard-normal physical field, seed 12345",
            "softmax_capture": "runtime observer around compiler._safe_masked_softmax; source restored after each forward",
            "interpretation": "large S typed/object differences with tiny final velocity difference indicates attenuation downstream; tiny typed differences indicates S binding failure",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=["standard", "blue_on_slider_left"])
    parser.add_argument("--instructions", nargs="+", default=list(INSTRUCTIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--time", type=float, default=0.4)
    args = parser.parse_args()
    result = run_probe(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        dataset_root=args.dataset_root,
        output=args.output,
        layouts=tuple(str(value) for value in args.layouts),
        instructions=tuple(str(value) for value in args.instructions),
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        time_value=float(args.time),
    )
    print(json.dumps({"output": str(args.output), "layouts": list(result["layouts"])}, indent=2))


if __name__ == "__main__":
    main()
