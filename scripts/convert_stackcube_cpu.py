"""Replay official StackCube actions with CPU physics and no captured images.

This prepares native commands only, not ClearVLA RGB experts. The later
panda_wristcam replay must independently succeed before training admission.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=73)
    parser.add_argument(
        "--render-backend",
        choices=("none", "gpu"),
        default="none",
        help="some SAPIEN builds need a GPU context even with no rendered observations",
    )
    args = parser.parse_args()
    if args.count < 10:
        parser.error("base preparation requires at least ten source demonstrations")
    if args.output.exists():
        raise FileExistsError(f"use a new controller-conversion directory: {args.output}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata["env_info"]["env_id"] != "StackCube-v1":
        raise ValueError("source is not StackCube-v1")
    if len(metadata["episodes"]) < args.count:
        raise ValueError("source inventory is smaller than requested count")
    # The vendor converter constructs both source and target environments from
    # this metadata. Disable rendering in BOTH, not just the target obs_mode.
    metadata["env_info"]["env_kwargs"].update(
        render_backend=args.render_backend, obs_mode="none", render_mode=None
    )
    args.output.mkdir(parents=True)
    copied = args.output / "trajectory.h5"
    shutil.copyfile(args.trajectory, copied)
    copied.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema": "clearvla-stackcube-cpu-conversion-v1",
        "source_trajectory": str(args.trajectory.resolve()),
        "source_trajectory_sha256": file_digest(args.trajectory),
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": file_digest(args.metadata),
        "count": args.count,
        "render_backend": args.render_backend,
        "pure_cpu": args.render_backend == "none",
        "sim_backend": "physx_cpu",
        "target_control_mode": "pd_ee_delta_pose",
        "environment_state_replay": False,
        "source_metadata_overrides": {
            "render_backend": args.render_backend,
            "obs_mode": "none",
            "render_mode": None,
        },
        "clearvla_expert_admission": False,
    }
    (args.output / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.render_backend == "none":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("GPU render context requires an explicitly selected CUDA_VISIBLE_DEVICES")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    from mani_skill.trajectory.replay_trajectory import Args
    from mani_skill.trajectory.replay_trajectory import main as replay

    replay(
        Args(
            traj_path=str(copied),
            sim_backend="physx_cpu",
            obs_mode="none",
            target_control_mode="pd_ee_delta_pose",
            save_traj=True,
            count=args.count,
            max_retry=1,
            reward_mode="sparse",
            record_rewards=True,
            num_envs=1,
            render_mode=None,
        )
    )
    converted = args.output / "trajectory.none.pd_ee_delta_pose.physx_cpu.json"
    result = json.loads(converted.read_text(encoding="utf-8"))
    rows = result["episodes"]
    report = {
        **provenance,
        "converted_episodes": len(rows),
        "all_source_successful": all(row["success"] is True for row in rows),
        "steps_min": min((row["elapsed_steps"] for row in rows), default=0),
        "steps_max": max((row["elapsed_steps"] for row in rows), default=0),
        "steps_total": sum(row["elapsed_steps"] for row in rows),
        "converted_metadata": str(converted.resolve()),
    }
    print(json.dumps(report, indent=2), flush=True)
    if len(rows) < 10 or not report["all_source_successful"]:
        raise RuntimeError("too few successful controller conversions for the base pilot")


if __name__ == "__main__":
    main()
