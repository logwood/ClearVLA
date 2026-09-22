"""Audit TargetFact's zero-start selector with the real training lifecycle.

This reduced CPU probe runs the ordinary optimizer, scheduler, composed loss,
backward, clipping lifecycle and ``AdamW.step`` twice.  It is deliberately not
an auxiliary selector loss: the reported gradients come from the same loss
ledger used by training.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearvla.mainline import config as config_module  # noqa: E402, I001
from clearvla.mainline import interfaces  # noqa: E402
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy  # noqa: E402
from clearvla.mainline.training.engine import MainlineTrainingEngine  # noqa: E402
from clearvla.mainline.training.optimizer import (  # noqa: E402
    WarmupCosineSchedule,
    build_optimizer,
)
from clearvla.tools import mainline_equivalence as equivalence  # noqa: E402


MODE = "target_action_bottleneck_v1"
METRIC_NAMES = (
    "gradient_parameter_p2_semantic_spatial_query_weight_rms",
    "gradient_parameter_target_fact_score_weight_rms",
    "gradient_parameter_target_fact_query_weight_rms",
    "gradient_parameter_target_fact_key_weight_rms",
    "gradient_parameter_p2_scene_semantic_value_weight_rms",
    "gradient_parameter_p2_scene_geometry_value_weight_rms",
    "gradient_tensor_s_target_posterior_rms",
    "gradient_tensor_p2_target_posterior_rms",
    "flow_jepa_target_fact_cross_forbidden_mass",
    "object_p2_target_geometry_k_marginal_error",
    "object_p2_scene_geometry_k_marginal_error",
    "object_intent_target_summary_rms",
    "object_p2_scene_semantic_selected_candidate_rms",
    "object_p2_scene_geometry_selected_physical_rms",
    "object_p2_scene_effect_precontract_rms",
)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def _parameter(model: ClearVLAMainlinePolicy, name: str) -> torch.Tensor:
    parameters = dict(model.named_parameters())
    try:
        return parameters[name]
    except KeyError as error:
        raise RuntimeError(f"probe lost parameter {name!r}") from error


def run(*, fixture: Path, seed: int, steps: int) -> dict[str, object]:
    if int(steps) < 2:
        raise ValueError("gradient staging requires at least two optimizer steps")
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
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
        train_flow_generator=torch.Generator().manual_seed(seed + 1),
        train_condition_generator=torch.Generator().manual_seed(seed + 2),
        gradient_spike_audit_threshold=None,
    )
    tracked = {
        "score": "intent.organizer.target_score.weight",
        "query": "intent.organizer.target_query.weight",
        "key": "intent.organizer.target_key.weight",
    }
    previous = {
        name: _parameter(model, path).detach().clone()
        for name, path in tracked.items()
    }
    step_rows: list[dict[str, object]] = []
    for index in range(int(steps)):
        result = engine.train_step(batch, collect_diagnostics=True)
        current = {
            name: _parameter(model, path).detach().clone()
            for name, path in tracked.items()
        }
        metrics = {
            name: _scalar(result.metrics[name])
            for name in METRIC_NAMES
        }
        step_rows.append(
            {
                "step": index + 1,
                "loss": _scalar(result.loss),
                "global_gradient_norm": _scalar(result.gradient_norm),
                "learning_rate": float(result.learning_rate),
                "metrics": metrics,
                "parameter_update_rms": {
                    name: _scalar((current[name] - previous[name]).square().mean().sqrt())
                    for name in tracked
                },
            }
        )
        previous = current

    first = step_rows[0]["metrics"]
    second = step_rows[1]["metrics"]
    assert isinstance(first, dict) and isinstance(second, dict)
    checks = {
        "retired_semantic_spatial_query_is_zero": (
            first["gradient_parameter_p2_semantic_spatial_query_weight_rms"] == 0.0
        ),
        "score_opens_on_first_task_step": (
            first["gradient_parameter_target_fact_score_weight_rms"] > 0.0
        ),
        "query_key_are_exact_zero_on_first_step": (
            first["gradient_parameter_target_fact_query_weight_rms"] == 0.0
            and first["gradient_parameter_target_fact_key_weight_rms"] == 0.0
        ),
        "query_key_receive_task_gradient_on_second_step": (
            second["gradient_parameter_target_fact_query_weight_rms"] > 0.0
            and second["gradient_parameter_target_fact_key_weight_rms"] > 0.0
        ),
        "posterior_receives_s_and_p2_gradient": (
            first["gradient_tensor_s_target_posterior_rms"] > 0.0
            and first["gradient_tensor_p2_target_posterior_rms"] > 0.0
        ),
        "uniform_target_retains_a_nonzero_factual_base": (
            first["object_intent_target_summary_rms"] > 0.0
        ),
        "p1_target_coverage_isolation_exact": (
            first["flow_jepa_target_fact_cross_forbidden_mass"] == 0.0
        ),
        "p2_geometry_k_marginal_preserved": (
            first["object_p2_target_geometry_k_marginal_error"] <= 1.0e-6
            and first["object_p2_scene_geometry_k_marginal_error"] <= 1.0e-6
        ),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"TargetFact gradient staging failed: {failed}")
    return {
        "schema": "clearvla-target-fact-gradient-staging-v1",
        "seed": int(seed),
        "mode": MODE,
        "optimizer": type(optimizer).__name__,
        "update_path": "MainlineTrainingEngine.train_step -> AdamW.step",
        "scope_note": (
            "The reduced fresh model keeps W output heads at exact zero; scene-lane "
            "opening requires the separate nonzero-W structural probe and a migrated "
            "production-checkpoint preflight."
        ),
        "fixture": str(fixture.resolve()),
        "steps": step_rows,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(tempfile.gettempdir()) / "clearvla_target_fact_fixture.pt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=91_704)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    report = run(fixture=args.fixture, seed=args.seed, steps=args.steps)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
