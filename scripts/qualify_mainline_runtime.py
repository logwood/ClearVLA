#!/usr/bin/env python3
"""Bounded existing-runtime/data/model probe; never installs dependencies or launches formal training.

Example: python scripts/qualify_mainline_runtime.py --operation inference
 --source synthetic --device cuda:0 --batch-size 1 --output /tmp/clearvla-probe
One child process owns the model; timeout/OOM cannot turn a partial log into success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/mainline/structural_rebuild_m6n_calvin.json"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--operation", choices=("inspect", "data", "inference", "train-step", "fit-batch"), default="inspect"
    )
    p.add_argument("--source", choices=("synthetic", "real"))
    p.add_argument(
        "--updates", type=int,
        help="Explicit finite update budget for fit-batch only; repeats one admitted batch",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--raw-side", type=int, default=336, help="Synthetic RGB side; ignored for real data"
    )
    p.add_argument("--allow-runtime-mismatch", action="store_true")
    p.add_argument(
        "--inference-step",
        type=int,
        default=0,
        help="Explicit completed-update clock for fresh-weight inference only; not a resumed checkpoint",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.source is None:
        if a.operation == "fit-batch":
            p.error("fit-batch requires explicit --source real or synthetic")
        a.source = "synthetic"
    if a.operation == "fit-batch":
        if a.updates is None or a.updates < 1:
            p.error("fit-batch requires a positive explicit --updates budget")
    elif a.updates is not None:
        p.error("updates is valid only for fit-batch")
    if a.batch_size < 1 or not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error("batch-size and finite timeout must be positive")
    if a.inference_step < 0 or (a.inference_step and a.operation != "inference"):
        p.error("inference-step must be nonnegative and is valid only for inference")
    if a.operation == "data" and a.source != "real":
        p.error("data qualification requires --source real, never a synthetic fallback")
    a.config = a.config.resolve()
    a.output = a.output.resolve()
    if a.output == ROOT or ROOT in a.output.parents:
        p.error("output must be outside the source checkout")
    return a


def bind_worker_to_parent() -> bool:
    """Linux child dies with its owning probe, even after an external SIGKILL.

    Set this inside the child (not preexec_fn), then close the parent-death
    race before importing torch or constructing a model. Other platforms keep
    the ordinary parent timeout/signal cleanup but do not claim this guarantee.
    """
    if not sys.platform.startswith("linux"):
        return False
    import ctypes

    expected = int(os.environ.get("CLEARVLA_QUALIFICATION_PARENT", "0"))
    if expected <= 1:
        raise ValueError("internal worker requires an owning qualification parent")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot bind worker lifetime to its parent")
    if os.getppid() != expected:
        raise RuntimeError("qualification parent exited before worker startup")
    return True


def child(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    path = a.output / "report.json"
    report: dict[str, Any] = json.loads(path.read_text())
    try:
        report["parent_death_guard"] = bind_worker_to_parent()
        write_json(path, report)
        import resource

        import torch

        from clearvla.mainline.config import load_config
        from clearvla.mainline.runtime.qualification import (
            declared_runtime,
            qualification_config,
            required_data_paths,
            synthetic_batch,
        )
        from clearvla.mainline.training.engine import validate_finite_training_batch

        config = qualification_config(load_config(a.config), a.dtype)
        runtime = declared_runtime(ROOT, platform.python_version(), str(torch.__version__))
        report.update(
            runtime={
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_available": torch.cuda.is_available(),
                "declared": runtime,
            },
            config=config.as_dict(),
            config_sha256=hashlib.sha256(
                json.dumps(config.as_dict(), sort_keys=True).encode()
            ).hexdigest(),
            data_paths=required_data_paths(config),
            dimensions_unchanged=True,
            weights="fresh deterministic initialization, not a trained policy checkpoint",
            model_seed=8301,
            sampling_seed=8302,
            cgroup_memory_max=(
                Path("/sys/fs/cgroup/memory.max").read_text().strip()
                if Path("/sys/fs/cgroup/memory.max").exists()
                else None
            ),
            actual_batch_size=a.batch_size,
            supplied_raw_side=a.raw_side if a.source == "synthetic" else None,
            limitation="synthetic cached-feature input is not real DINO/T5, dataset validation or skill"
            if a.source == "synthetic"
            else "bounded recorded data; not formal training or physics",
        )

        def mark(stage: str) -> None:
            report.update(
                stage=stage,
                rss_peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            )
            if torch.cuda.is_available() and a.device.startswith("cuda"):
                report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(a.device)
                report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(a.device)
            write_json(path, report)

        def measured(stage: str, fn: Any) -> Any:
            mark(stage)
            if a.device.startswith("cuda"):
                torch.cuda.synchronize(a.device)
            start = time.perf_counter()
            result = fn()
            if a.device.startswith("cuda"):
                torch.cuda.synchronize(a.device)
            report.setdefault("timings_seconds", {})[stage] = time.perf_counter() - start
            mark(stage + "-complete")
            return result

        mark("runtime-admission")
        if not runtime["match"] and not a.allow_runtime_mismatch:
            report.update(status="blocked", reason="declared runtime mismatch; no packages changed")
            write_json(path, report)
            return 2
        device = torch.device(a.device)
        if device.type not in ("cpu", "cuda") or (
            device.type == "cuda" and not torch.cuda.is_available()
        ):
            report.update(status="blocked", reason="requested device unavailable; no CPU fallback")
            write_json(path, report)
            return 2
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        if a.operation == "inspect":
            report.update(
                status="passed", scope="environment inspection only", model_executed=False
            )
            mark("complete")
            return 0
        bundle = None
        if a.source == "real":
            missing = [k for k, v in report["data_paths"].items() if not v["exists"]]
            if missing:
                report.update(
                    status="blocked", reason="configured real inputs unavailable", missing=missing
                )
                mark("data-admission")
                return 2
            from torch.utils.data import default_collate

            from clearvla.mainline.data.loading import load_mainline_data, to_training_batch

            bundle = measured(
                "load-data", lambda: load_mainline_data(config, allow_null_goal=False)
            )
            data = bundle.datasets["train"]
            if len(data) < a.batch_size:
                raise ValueError("real training inventory is smaller than requested batch")
            # Deterministic, non-repeating samples spanning the available inventory.
            indices = [i * (len(data) - 1) // max(a.batch_size - 1, 1) for i in range(a.batch_size)]
            report["real_sample_indices"] = indices
            batch = measured(
                "read-batch",
                lambda: to_training_batch(
                    default_collate([data[i] for i in indices]),
                    goal=bundle.goal,
                    config=config,
                    device=device,
                ),
            )
            normalizer = bundle.action_normalizer
        else:
            batch, normalizer = measured(
                "synthetic-batch",
                lambda: synthetic_batch(
                    config, count=a.batch_size, raw_side=a.raw_side, device=device
                ),
            )
        batch.validate(config)
        validate_finite_training_batch(batch)
        report["actual_shapes"] = {
            "dino": list(batch.online.observation.dino_history.shape),
            "rgb": list(batch.online.observation.raw_rgb.shape),
            "goal": list(batch.online.goal.tokens.shape),
        }
        if a.operation == "data":
            report.update(status="passed", scope="real data admission only", model_executed=False)
            mark("complete")
            return 0
        from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
        from clearvla.mainline.runtime.numerics import resolve_compute_dtype
        from clearvla.mainline.runtime.sampling import (
            deployment_cache,
            sample_refined_cached_action,
        )
        from clearvla.mainline.training.engine import MainlineTrainingEngine
        from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer

        torch.manual_seed(8301)
        model = measured("construct-model", lambda: ClearVLAMainlinePolicy(config).to(device))
        report["model_constructed"] = True
        model.configure_action_normalizer(normalizer)
        report["registered_parameters"] = sum(p.numel() for p in model.parameters())
        report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        dtype = resolve_compute_dtype(config)
        if a.operation in {"train-step", "fit-batch"}:
            optimizer, _ = measured("build-optimizer", lambda: build_optimizer(model, config))
            engine = MainlineTrainingEngine(
                model=model,
                config=config,
                optimizer=optimizer,
                schedule=WarmupCosineSchedule(
                    optimizer,
                    warmup_steps=config.optimizer.warmup_steps if a.operation == "fit-batch" else 1,
                    total_steps=a.updates if a.operation == "fit-batch" else 2,
                    minimum_ratio=config.optimizer.min_lr_ratio if a.operation == "fit-batch" else 0.1,
                ),
                device=device,
                dtype=dtype,
            )
            report["model_forward_attempted"] = True
            if a.operation == "fit-batch":
                from clearvla.mainline.runtime.learning_probe import probe_fixed_batch_learning

                report["probe_schedule"] = {
                    "state": engine.schedule.state_dict(),
                    "scope": "configured warmup/minimum, requested finite horizon; not formal dataset schedule",
                    "warmup_exceeds_budget": config.optimizer.warmup_steps > a.updates,
                }

                def learning_progress(trace: dict[str, Any]) -> None:
                    report["fixed_batch_learning"] = trace
                    report["completed_optimizer_updates"] = engine.global_step
                    mark("fixed-batch-" + trace["stage"])

                report["fixed_batch_learning"] = measured(
                    "fixed-batch-learning",
                    lambda: probe_fixed_batch_learning(
                        engine, batch, updates=a.updates, on_progress=learning_progress
                    ),
                )
                report["weights_after"] = "same fresh initialization after bounded updates; no checkpoint emitted"
            else:
                result = measured(
                    "one-optimizer-update", lambda: engine.train_step(batch, collect_diagnostics=True)
                )
                if not torch.isfinite(result.loss):
                    raise FloatingPointError("non-finite one-step loss")
                report.update(loss=float(result.loss), completed_optimizer_updates=engine.global_step)
        else:
            model.eval()
            report["inference_phase_clock"] = {
                "completed_updates": a.inference_step,
                "phase": float(model.set_training_step(a.inference_step)),
                "scope": "emulated controller clock on fresh weights, not actual completed training",
            }
            report["model_forward_attempted"] = True
            cache, _ = measured(
                "online-encode", lambda: deployment_cache(model, batch.online, config, dtype=dtype)
            )
            sample = measured(
                "two-pass-sampling",
                lambda: sample_refined_cached_action(
                    model,
                    cache,
                    config,
                    dtype=dtype,
                    generator=torch.Generator(device=device).manual_seed(8302),
                    collect_diagnostics=False,
                ),
            )
            if not torch.isfinite(sample.action).all():
                raise FloatingPointError("non-finite sampled commands")
            import numpy as np

            np.save(a.output / "sampled-action.npy", sample.action.detach().float().cpu().numpy())
            report.update(
                action_artifact="sampled-action.npy",
                action_scope="model outlet result, not an executed/acknowledged controller command",
                action_shape=list(sample.action.shape),
                action_abs_max=float(sample.action.abs().max()),
            )
        report.update(
            status="passed",
            scope="bounded production computation; not closed-loop or dataset-learning acceptance",
            model_executed=True,
            model_completed=True,
        )
        mark("complete")
        return 0
    except BaseException as error:
        import traceback

        traceback.print_exc()
        report.update(status="failed", error=type(error).__name__, message=str(error))
        write_json(path, report)
        return 1


def main() -> int:
    a = arguments()
    if a.worker:
        return child(a)
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("output must be new or empty; refusing to overwrite prior evidence")
    a.output.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = {
        "schema": "clearvla-bounded-qualification-v1",
        "status": "running",
        "stage": "launch",
        "source_head": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_dirty": git("status", "--porcelain"),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "model_executed": False,
        "formal_training": False,
    }
    write_json(a.output / "report.json", status)
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    environment.pop("PYTHONPATH", None)
    environment["CLEARVLA_QUALIFICATION_PARENT"] = str(os.getpid())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(a.config),
        "--output",
        str(a.output),
        "--operation",
        a.operation,
        "--source",
        a.source,
        "--device",
        a.device,
        "--batch-size",
        str(a.batch_size),
        "--raw-side",
        str(a.raw_side),
        "--timeout",
        str(a.timeout),
        "--inference-step",
        str(a.inference_step),
        "--worker",
    ]
    if a.updates is not None:
        command += ["--updates", str(a.updates)]
    if a.dtype is not None:
        command += ["--dtype", a.dtype]
    if a.allow_runtime_mismatch:
        command.append("--allow-runtime-mismatch")
    start = time.monotonic()
    with (a.output / "process.log").open("w") as log:
        worker = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def stop_worker() -> None:
            if worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.wait(timeout=5)

        def interrupted(signum: int, frame: object) -> None:
            del frame
            stop_worker()
            interrupted_report = json.loads((a.output / "report.json").read_text())
            interrupted_report.update(
                status="failed",
                reason="qualification parent interrupted",
                returncode=128 + signum,
                model_completed=False,
            )
            write_json(a.output / "report.json", interrupted_report)
            raise SystemExit(128 + signum)

        old_signals = {
            sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        # The child owns report.json from here onward; avoid racing its atomic
        # writes just to add process metadata.
        write_json(a.output / "process.json", {"parent_pid": os.getpid(), "worker_pid": worker.pid})
        try:
            try:
                code = worker.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                stop_worker()
                code = 124
        finally:
            stop_worker()
            for sig, handler in old_signals.items():
                signal.signal(sig, handler)
    report = json.loads((a.output / "report.json").read_text())
    if report["status"] == "running" or (code != 0 and report["status"] == "passed"):
        report.update(
            status="failed",
            reason="child incomplete or killed; see last stage, not partial success",
        )
    report.update(returncode=code, wall_seconds=time.monotonic() - start)
    write_json(a.output / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "returncode": code,
                "report": str(a.output / "report.json"),
            }
        )
    )
    return code if code >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
