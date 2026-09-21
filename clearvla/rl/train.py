"""Opt-in StackCube residual-SAC runner. Use --mode base before training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clearvla.simulation.admission import (
    STACKCUBE_INSTRUCTION,
    STACKCUBE_REPAIRED_PROFILE,
    validate_stackcube_descriptor,
)
from clearvla.simulation.state_chart import CONTINUOUS_STATE_CHART

from .checkpoint import atomic_json, digest, load_adapter, save_adapter, source_digest
from .config import SACConfig
from .features import FEATURE_SCHEMA, FrozenBaseReader
from .runner import run_episode
from .sac import ResidualSAC


def validate_baseline(report: dict, identity: dict) -> list[int]:
    if (
        report.get("schema") != "clearvla-residual-evaluation-v1"
        or report.get("mode") != "base"
        or report.get("complete") is not True
        or report.get("base_identity") != identity
    ):
        raise ValueError("baseline is incomplete or belongs to another base/environment/source")
    rows = report.get("episodes", [])
    seeds = [row["seed"] for row in rows]
    if len(rows) < 20 or len(set(seeds)) != len(seeds):
        raise ValueError("baseline requires at least 20 distinct held-out reset seeds")
    if not any(row.get("success") is True for row in rows):
        raise ValueError(
            "base has no observed successes; pretrain/improve the BC base before sparse-reward residual SAC"
        )
    if any(row.get("mode") != "base" or row.get("mean_residual_native_l2") != 0 for row in rows):
        raise ValueError("baseline must be a real zero-residual evaluation")
    return seeds


def training_seed(start: int, episode_index: int, excluded: set[int]) -> int:
    # Count excluded points <= the candidate, not a mutable seed cursor;
    # continuation therefore skips exactly the same held-out seeds.
    candidate = start + episode_index
    for seed in sorted(excluded):
        if start <= seed <= candidate:
            candidate += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl/stackcube_residual_sac.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "train", "adapter"), default="base")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=10000, help="total training environment steps")
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes")
    parser.add_argument("--eval-seed", type=int, default=1000000)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=400,
        help="StackCube protocol time limit (override only for a bounded smoke)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--dinov2-model", type=Path)
    args = parser.parse_args()
    cfg = SACConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    cfg.validate()
    if args.steps <= 0 or args.episodes <= 0 or args.max_episode_steps <= 0 or args.eval_seed < 0:
        parser.error("budgets must be positive and eval-seed nonnegative")
    if args.mode == "train" and args.baseline_report is None:
        parser.error("training requires --baseline-report from --mode base")
    if args.mode == "adapter" and (args.adapter_checkpoint is None or args.baseline_report is None):
        parser.error(
            "adapter evaluation requires --adapter-checkpoint and its original --baseline-report"
        )
    if args.mode == "base" and args.adapter_checkpoint is not None:
        parser.error("base evaluation cannot load an adapter")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("use a new empty output directory")
    # Late imports keep --help and pure SAC tests free of simulator/model dependencies.
    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
    from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv

    policy = ClearVLACheckpointPolicy(
        args.base_checkpoint,
        device=torch.device(args.device),
        t5_condition=args.t5_condition,
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=True,
        seed=cfg.seed,
    )
    if (
        policy.bundle.epoch < 1
        or policy.bundle.config.runtime.max_train_batches > 0
        or policy.bundle.config.runtime.max_val_batches > 0
    ):
        raise ValueError("RL requires a formal full-epoch base, not a bounded smoke checkpoint")
    reader = FrozenBaseReader(policy)
    environment = ManiSkillStackCubeEnv(max_episode_steps=args.max_episode_steps,
                                       state_chart=CONTINUOUS_STATE_CHART)
    try:
        descriptor = environment.descriptor.to_dict()
        validate_stackcube_descriptor(descriptor, profile=STACKCUBE_REPAIRED_PROFILE)
        profile = policy.bundle.deployment_abi["action"]["data_profile"]
        if (
            profile.get("expert_admission") != "stackcube_all_source_first_action_v2"
            or profile.get("simulator_environment") != descriptor
        ):
            raise ValueError(
                "base checkpoint lacks matching replay-verified StackCube environment provenance"
            )
        expert_seeds = profile.get("expert_reset_seeds")
        if (
            not isinstance(expert_seeds, list)
            or not expert_seeds
            or any(type(seed) is not int or seed < 0 for seed in expert_seeds)
        ):
            raise ValueError("base checkpoint lacks expert reset-seed provenance")
        low, high = environment.action_bounds()
        if not np.array_equal(low, -np.ones(7)) or not np.array_equal(high, np.ones(7)):
            raise ValueError("ManiSkill native action bounds changed from [-1,1]")
        health = policy.deployment_health()
        base_identity = dict(
            base_sha256=policy.bundle.checkpoint_sha256,
            deployment_abi=dict(policy.bundle.deployment_abi),
            dino_runtime=health["observation"]["dinov2_runtime"],
            environment=descriptor,
            instruction=STACKCUBE_INSTRUCTION,
            feature_schema=FEATURE_SCHEMA,
            feature_projection_sha256=reader.projection_sha256,
            numerical_runtime={"torch": str(torch.__version__), "numpy": str(np.__version__)},
            source_sha256=source_digest(),
            base_noise_seed=cfg.seed,
            image_size=336,
            sim_backend="physx_cpu",
            render_backend="gpu",
            reward_mode="sparse",
        )
        # Canonical JSON types avoid tuple/list differences on round trip.
        base_identity = json.loads(json.dumps(base_identity))
        identity = dict(
            base=base_identity,
            sac=cfg.as_dict(),
            feature_dim=reader.feature_dim,
            execution="decoded_native_first_row_then_bounded_residual_v1",
        )
        learner = ResidualSAC(reader.feature_dim, cfg, torch.device(args.device))
        position = dict(episodes=0, env_steps=0)
        excluded_seeds = set(expert_seeds)
        if args.baseline_report:
            report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
            baseline_seeds = set(validate_baseline(report, base_identity))
            if baseline_seeds.intersection(expert_seeds):
                raise ValueError("baseline evaluation seeds overlap base demonstrations")
            excluded_seeds.update(baseline_seeds)
        if args.mode != "train" and excluded_seeds.intersection(
            range(args.eval_seed, args.eval_seed + args.episodes)
        ):
            # A supplied baseline's same seeds are allowed for matched adapter
            # evaluation, but the original demonstrations are never allowed.
            if args.mode == "base" or set(expert_seeds).intersection(
                range(args.eval_seed, args.eval_seed + args.episodes)
            ):
                raise ValueError("evaluation seeds overlap the base demonstration inventory")
        if args.mode == "train":
            identity["excluded_training_seeds"] = sorted(excluded_seeds)
        elif args.adapter_checkpoint:
            # Adapter snapshots include the training seed holdout identity.
            # Require its original baseline report for a verifiable reload.
            if args.baseline_report is None:
                raise ValueError(
                    "adapter continuation/evaluation requires its original --baseline-report"
                )
            identity["excluded_training_seeds"] = sorted(excluded_seeds)
            if not set(range(args.eval_seed, args.eval_seed + args.episodes)).issubset(
                baseline_seeds
            ):
                raise ValueError("v1 adapter evaluation must use the baseline's reserved seeds")
        if args.adapter_checkpoint:
            position = load_adapter(args.adapter_checkpoint, learner, identity)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / "run_context.json",
            {
                "identity": identity,
                "identity_sha256": digest(identity),
                "mode": args.mode,
                "step_budget": args.steps,
                "evaluation_episodes": args.episodes,
                "evaluation_seed": args.eval_seed,
                "excluded_training_seeds": sorted(excluded_seeds),
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "adapter_checkpoint": str(args.adapter_checkpoint.resolve())
                if args.adapter_checkpoint
                else None,
                "continuation": "episode_boundary_reset_v1; no mid-episode simulator resume",
            },
        )
        reports = []
        if args.mode != "train":
            for index in range(args.episodes):
                row = run_episode(
                    environment,
                    reader,
                    learner,
                    seed=args.eval_seed + index,
                    step_budget=args.max_episode_steps,
                    mode=args.mode,
                )
                reports.append(row)
                atomic_json(args.output / f"episode_{index:05d}.json", row)
                print(json.dumps(row), flush=True)
            atomic_json(
                args.output / "evaluation.json",
                {
                    "schema": "clearvla-residual-evaluation-v1",
                    "mode": args.mode,
                    "complete": True,
                    "base_identity": base_identity,
                    "episodes": reports,
                    "success_rate": sum(row["success"] for row in reports) / len(reports),
                    "official_protocol": False,
                },
            )
        else:
            start_steps = position["env_steps"]
            # New training episodes never consume baseline evaluation reset seeds.
            while position["env_steps"] - start_steps < args.steps:
                seed = training_seed(cfg.seed, position["episodes"], excluded_seeds)
                budget = min(
                    args.max_episode_steps, args.steps - (position["env_steps"] - start_steps)
                )
                row = run_episode(
                    environment, reader, learner, seed=seed, step_budget=budget, mode="train"
                )
                position["episodes"] += 1
                position["env_steps"] += row["steps"]
                atomic_json(args.output / f"episode_{position['episodes']:05d}.json", row)
                print(json.dumps(row), flush=True)
            # Snapshot includes learner/target/optimizers/replay/RNG, but never
            # frozen ClearVLA or DINO weights. No original checkpoint is touched.
            save_adapter(args.output / "adapter.pt", learner, identity, **position)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
