"""Read-only frozen arm-only checkpoint ablation on its original source tree.

This tests deployment-time detail removal, NOT a newly trained coarse-context
architecture. No optimizer is constructed, and no checkpoint is written.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

UNIFORM = tuple(index / 5.0 for index in range(6))
Q5 = tuple((index / 5.0) ** 1.25 for index in range(6))


def validate_boundaries(boundaries):
    if len(boundaries) != 6 or boundaries[0] != 0.0 or boundaries[-1] != 1.0:
        raise ValueError("replay requires five updates from 0 to 1")
    if not all(math.isfinite(t) for t in boundaries):
        raise ValueError("nonfinite schedule")
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("schedule must strictly increase")


def integrate_cache(
    model, cache, config, *, generator, initial_physical_noise,
    collect_diagnostics, dtype, static_metrics=None, execution_mode="learned",
    deployment_fastpath=False, boundaries=UNIFORM,
):
    """Same cf98238 integrator; only the explicitly selected t/dt may differ."""
    from clearvla.mainline.runtime.sampling import SamplingResult

    validate_boundaries(boundaries)
    cache.validate(config)
    batch = cache.history.batch
    device = cache.history.state.device
    amp = device.type in {"cpu", "cuda"} and dtype in {torch.bfloat16, torch.float16}
    if initial_physical_noise is None:
        value = model.outlet_adapter.sample_noise(
            batch, device=device, dtype=cache.history.action_state.dtype,
            generator=generator,
        )
    else:
        expected = (batch, config.dimensions.action_horizon, model.outlet_adapter.physical_dim)
        if tuple(initial_physical_noise.shape) != expected:
            raise ValueError("noise shape mismatch")
        value = initial_physical_noise.to(device=device)
    noise = value.clone()
    if tuple(boundaries) == UNIFORM:
        dt = 1.0 / 5.0
        steps = (dt,) * 5
        times = torch.arange(5, device=device, dtype=torch.float32) * dt
    else:
        steps = tuple(b - a for a, b in zip(boundaries, boundaries[1:]))
        times = torch.tensor(boundaries[:-1], device=device, dtype=torch.float32)
    for index, dt in enumerate(steps):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            output = model.velocity(
                cache, noisy_action_field=value, time=times[index].expand(batch),
                execution_mode=execution_mode, deployment_fastpath=deployment_fastpath,
                collect_diagnostics=False,
            )
        value = value + dt * output.bottom.physical_velocity.to(dtype=value.dtype)
    endpoint_time = torch.ones(batch, device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
        output = model.velocity(
            cache, noisy_action_field=value, time=endpoint_time,
            execution_mode=execution_mode, deployment_fastpath=deployment_fastpath,
            collect_diagnostics=False,
        )
    outlet = model.outlet_adapter.finalize(
        value, cache.history.action_state,
        codec_gripper_boundary=cache.history.codec_gripper_boundary,
        command_logits=output.bottom.gripper_command_logits,
    )
    return SamplingResult(
        action=outlet.deployed_action, physical_field=value,
        motion_logits=output.bottom.motion_logits.float(),
        initial_physical_noise=noise, step_times=times, metrics={},
        gripper_command_logits=outlet.command_logits, gripper_command=outlet.command,
        continuous_action=outlet.continuous_action,
    )


@contextmanager
def detail_removed(spine, enabled):
    """Zero detail outputs, retaining all original tensors and checkpoint keys."""
    handles = []
    try:
        if enabled:
            for lift in spine.detail_lifts.values():
                handles.append(lift.register_forward_hook(lambda _m, _args, out: torch.zeros_like(out)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def schedule_override(sampling, boundaries):
    original = sampling._integrate_cache
    try:
        sampling._integrate_cache = functools.partial(integrate_cache, boundaries=boundaries)
        yield
    finally:
        sampling._integrate_cache = original


def _tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def _writes_only_in(output):
    """Deny Python-level writes to source, dataset/cache or checkpoint paths."""
    root = output.resolve()

    def allowed(path):
        if isinstance(path, int):
            return
        path = Path(os.fsdecode(path)).resolve()
        if path != root and root not in path.parents:
            raise PermissionError(f"read-only replay attempted a write outside output: {path}")

    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                allowed(path)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"}:
            allowed(args[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            allowed(args[0])
            allowed(args[1])
    sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=0, help="0=full validation; otherwise evenly spaced batches")
    parser.add_argument("--expected-commit", default="cf98238ba774880f6cd15a7a21cd3464a9ab9cfc")
    args = parser.parse_args()
    if args.batches < 0:
        raise ValueError("batch limit must be nonnegative")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo_root.resolve()))
    os.chdir(args.repo_root)
    from clearvla.mainline.checkpoint import (
        active_source_snapshot,
        checkpoint_identity_from_mapping,
        git_commit,
    )
    from clearvla.mainline.config import config_from_mapping
    from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime import sampling
    from clearvla.mainline.runtime.evaluation import ValidationAccumulator
    from clearvla.mainline.runtime.identity import dataset_identity, language_identity
    from clearvla.mainline.train import _diagnostic_batch_indices

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("this formal replay requires an available CUDA device")
    device = torch.device("cuda:0")
    used = torch.cuda.get_device_properties(device).total_memory - torch.cuda.mem_get_info(device)[0]
    if used > 2 * 1024**3:
        raise RuntimeError("selected GPU is occupied; do not contend with a training task")
    torch.manual_seed(0)
    if git_commit(args.repo_root) != args.expected_commit:
        raise ValueError("replay checkout is not the declared original revision")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "context.json").exists() or (args.output_dir / "rows.jsonl").exists():
        raise FileExistsError("replay evidence already exists; use a new output directory")
    _writes_only_in(args.output_dir)
    # Hold one open descriptor across hashing/loading: an atomic best.pt
    # replacement by training cannot change this replay's frozen snapshot.
    with args.checkpoint.open("rb") as stream:
        file_stat = os.fstat(stream.fileno())
        sha = hashlib.sha256()
        while chunk := stream.read(8 * 1024**2):
            sha.update(chunk)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
        if file_stat.st_size != os.fstat(stream.fileno()).st_size:
            raise RuntimeError("checkpoint mutated in place during loading")
    config = config_from_mapping(payload["config"])
    if config.bottom.bspine_implementation != "fixed_bspline_arm_only_v1":
        raise ValueError("requires original trained arm-only coarse+detail checkpoint")
    if config.data.data_profile != "identity_7d_pen":
        raise ValueError("this replay is scoped to Pen continuous actions")
    identity = checkpoint_identity_from_mapping(payload["identity"])
    if identity.config_digest != config.digest(include_paths=False):
        raise ValueError("checkpoint config identity is inconsistent")
    source = active_source_snapshot(args.repo_root)
    if source != identity.source:
        drift = [p for p, h in source.files if dict(identity.source.files).get(p) != h]
        raise ValueError(f"checkpoint source drift: {drift}")
    print(json.dumps({"phase": "checkpoint_verified", "epoch": payload["epoch"],
                      "step": payload["global_step"], "sha256": sha.hexdigest()}), flush=True)
    bundle = load_mainline_data(config)
    if dataset_identity(bundle, config) != identity.dataset:
        raise ValueError("dataset/normalizer/cache identity differs")
    if language_identity(bundle, config) != identity.language:
        raise ValueError("language identity differs")
    model = ClearVLAMainlinePolicy(config).to(device)
    if model.selection.as_dict() != payload["component_selection"]:
        raise ValueError("component identity differs")
    for name, value in payload["model"].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"checkpoint has nonfinite tensor: {name}")
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    before = _tensor_digest(model.state_dict())
    epoch, step = payload["epoch"], payload["global_step"]
    del payload
    spine = model.execution_bottom.decoder.spine
    if len(spine.coarse_lifts) != 2 or len(spine.detail_lifts) != 2:
        raise ValueError("expected the original two arm groups and detail owners")
    loader = bundle.loader("val", batch_size=config.optimizer.batch_size, workers=0,
                           device=device, shuffle=False,
                           generator=torch.Generator().manual_seed(config.data.seed + 101))
    indices = sorted(_diagnostic_batch_indices(planned_batches=len(loader), budget=args.batches))
    context = {
        "schema": "clearvla-frozen-bspine-schedule-replay-v1",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha.hexdigest(),
        "checkpoint_inode": file_stat.st_ino, "checkpoint_bytes": file_stat.st_size,
        "epoch": epoch, "global_step": step,
        "source_git": args.expected_commit, "source_digest": source.digest,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loaded_state_sha256": before, "training": False, "optimizer_loaded": False,
        "checkpoint_writes": False, "new_architecture_migration": False,
        "meaning": "deployment-time zero-detail ablation of original trained arm-only checkpoint; not fresh coarse-only training",
        "data_identity": identity.as_dict()["dataset"], "splits": bundle.splits,
        "validation_samples": len(bundle.datasets["val"]), "batch_size": config.optimizer.batch_size,
        "selected_batches": indices, "total_batches": len(loader),
        "schedules": {"uniform": UNIFORM, "Q5": Q5},
        "per_cell_physical_nfe": 10, "per_cell_endpoint_calls": 2,
        "per_cell_w_rebuilds": 1, "initial_noise_seed": "37237 + one_based_validation_batch",
        "fastpath": False, "torch": torch.__version__, "device": torch.cuda.get_device_name(device),
    }
    _json(args.output_dir / "context.json", context)
    cells = (("original_uniform", "full", UNIFORM),
             ("original_Q5", "full", Q5),
             ("raw_uniform", "raw", UNIFORM), ("raw_Q5", "raw", Q5),
             ("coarse_uniform", "coarse", UNIFORM), ("coarse_Q5", "coarse", Q5))
    def accumulator():
        return ValidationAccumulator.from_action_normalizer(
            bundle.action_normalizer, device=device,
            gripper_event_threshold=config.objectives.gripper_event_threshold,
            arm_motion_threshold=config.objectives.arm_motion_threshold,
        )
    accumulators = {name: accumulator() for name, *_ in cells}
    seen_episodes = set()
    elapsed = {name: 0.0 for name, *_ in cells}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad(), (args.output_dir / "rows.jsonl").open("x", encoding="utf-8") as out:
        for index, raw in enumerate(loader, start=1):
            if index not in indices:
                continue
            batch = to_training_batch(raw, goal=bundle.goal, config=config, device=device)
            episodes = batch.audit.episode_index.tolist()
            seen_episodes.update(episodes)
            noise = model.outlet_adapter.sample_noise(
                batch.online.batch, device=device, dtype=batch.online.history.action_state.dtype,
                generator=torch.Generator(device=device).manual_seed(37237 + index),
            )
            cache, _ = sampling.deployment_cache(model, batch.online, config, dtype=torch.bfloat16)
            if index == indices[0]:
                primary = sampling.sample_refined_cached_action(model, cache, config,
                    initial_physical_noise=noise, dtype=torch.bfloat16)
                with schedule_override(sampling, UNIFORM):
                    patched = sampling.sample_refined_cached_action(model, cache, config,
                        initial_physical_noise=noise, dtype=torch.bfloat16)
                for name in ("action", "physical_field", "motion_logits", "initial_physical_noise"):
                    if not torch.equal(getattr(primary, name), getattr(patched, name)):
                        raise AssertionError(f"uniform replay does not match original: {name}")
                _json(args.output_dir / "parity.json", {"bit_exact": True, "batch": index})
                print(json.dumps({"phase": "uniform_parity_pass", "total_batches": len(indices)}), flush=True)
                del primary, patched
            for cell, branch, boundaries in cells:
                torch.cuda.synchronize(device)
                cell_started = time.perf_counter()
                with detail_removed(spine, branch == "coarse"), schedule_override(sampling, boundaries):
                    result = sampling.sample_refined_cached_action(
                        model, cache, config, initial_physical_noise=noise, dtype=torch.bfloat16,
                        execution_mode="spine_zero" if branch == "raw" else "learned",
                    )
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - cell_started
                elapsed[cell] += seconds
                if not bool(torch.isfinite(result.action).all()) or not bool(torch.isfinite(result.physical_field).all()):
                    raise FloatingPointError(f"nonfinite replay: {cell} batch {index}")
                if not torch.equal(noise, result.initial_physical_noise):
                    raise AssertionError("initial noise was mutated")
                accumulators[cell].update(result.action, batch)
                per_batch = accumulator()
                per_batch.update(result.action, batch)
                row = {"batch": index, "episodes": episodes, "cell": cell,
                       "samples": batch.online.batch, "seconds": seconds, "metrics": per_batch.means()}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                del result, per_batch
            if index == indices[0] or index % 10 == 0 or index == indices[-1]:
                print(json.dumps({"phase": "progress", "batch": index,
                    "completed": indices.index(index) + 1, "total": len(indices),
                    "elapsed_seconds": time.perf_counter() - started,
                    "seconds_per_five_cells": sum(elapsed.values()) / (indices.index(index) + 1)}), flush=True)
            del cache, batch, raw
    after = _tensor_digest(model.state_dict())
    if after != before:
        raise AssertionError("model state was modified during read-only replay")
    result = {"context": context, "cells": {key: acc.means() for key, acc in accumulators.items()},
              "seen_episodes": sorted(seen_episodes), "seconds": elapsed,
              "state_unchanged": True, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
              "total_seconds": time.perf_counter() - started}
    _json(args.output_dir / "summary.json", result)
    print(json.dumps({"phase": "complete", "total_seconds": result["total_seconds"],
                      "summary": str(args.output_dir / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
