"""Local small-config feasibility probe for CUDA graph training capture.

This is intentionally a bounded research probe.  It uses the test-sized
architecture and synthetic tensors to expose capture blockers before the real
B8 remote benchmark is touched.
"""

from __future__ import annotations

import dataclasses
import inspect
import statistics
import sys
import time
import types
from contextlib import contextmanager

import torch

try:
    import h5py  # noqa: F401
except ModuleNotFoundError:
    h5py_stub = types.ModuleType("h5py")
    h5py_stub.File = type("File", (), {})
    h5py_stub.Dataset = type("Dataset", (), {})
    sys.modules["h5py"] = h5py_stub

sys.path.insert(0, "tests")

from test_mainline_policy import _batch, _config  # noqa: E402

from clearvla.mainline.model.policy import ClearVLAMainlinePolicy  # noqa: E402
from clearvla.mainline.training.engine import MainlineTrainingEngine, _autocast  # noqa: E402
from clearvla.mainline.training.optimizer import (  # noqa: E402
    WarmupCosineSchedule,
    build_optimizer,
)


def _move(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: _move(getattr(value, field.name), device)
                for field in dataclasses.fields(value)
            },
        )
    return value


def _engine(device: torch.device) -> tuple[MainlineTrainingEngine, object]:
    config = _config()
    torch.manual_seed(9170)
    model = ClearVLAMainlinePolicy(config).to(device).train()
    model.execution_bottom.decoder._static_neutral_owner = True
    model.execution_bottom.decoder._training_candidate_prefix_reuse = True
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=20,
        minimum_ratio=0.1,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        dtype=torch.float32,
        train_flow_generator=torch.Generator(device=device).manual_seed(9171),
        train_condition_generator=torch.Generator(device=device).manual_seed(9172),
        skip_postglobal_audit=True,
        gradient_spike_audit_threshold=None,
    )
    return engine, _move(_batch(config, batch=2), device)


@contextmanager
def _validated_capture_region():
    """Skip repeated runtime validation only while recording the static graph.

    The same tensor objects pass their complete validation in the eager warmup
    immediately before capture.  This probe restores every class method after
    recording; production integration would own an explicit checked static
    input boundary instead of using discovery.
    """

    originals: list[tuple[type, object]] = []

    def no_validate(_self: object, *_args: object, **_kwargs: object) -> None:
        return None

    seen: set[type] = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("clearvla.mainline") or module is None:
            continue
        for value in tuple(vars(module).values()):
            if (
                inspect.isclass(value)
                and value not in seen
                and "validate" in value.__dict__
            ):
                seen.add(value)
                originals.append((value, value.__dict__["validate"]))
                setattr(value, "validate", no_validate)
    try:
        yield
    finally:
        for value, original in originals:
            setattr(value, "validate", original)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    engine, batch = _engine(device)
    for _ in range(3):
        engine.train_step(batch, collect_diagnostics=False)
    torch.cuda.synchronize()
    eager_seconds: list[float] = []
    for _ in range(10):
        started = time.perf_counter()
        engine.train_step(batch, collect_diagnostics=False)
        torch.cuda.synchronize()
        eager_seconds.append(time.perf_counter() - started)

    engine, batch = _engine(device)
    engine.model.set_training_step(engine.global_step)
    # Allocate stable gradient buffers and warm libraries before capture.
    engine.optimizer.zero_grad(set_to_none=True)
    with _autocast(device, engine.dtype):
        warm_ledger, warm_metrics = engine._forward(
            batch,
            training=True,
            collect_diagnostics=False,
            generator=engine.train_flow_generator,
            condition_generator=engine.train_condition_generator,
        )
    warm_ledger.total.backward()
    engine.optimizer.zero_grad(set_to_none=False)
    # Do not retain the eager loss graph while capture creates its own
    # AccumulateGrad nodes on the private capture stream.
    del warm_ledger, warm_metrics
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(engine.train_flow_generator)
    graph.register_generator_state(engine.train_condition_generator)
    with _validated_capture_region():
        with torch.cuda.graph(graph):
            engine.optimizer.zero_grad(set_to_none=False)
            with _autocast(device, engine.dtype):
                captured_ledger, _captured_metrics = engine._forward(
                    batch,
                    training=True,
                    collect_diagnostics=False,
                    generator=engine.train_flow_generator,
                    condition_generator=engine.train_condition_generator,
                )
            captured_ledger.total.backward()

    # The output slots become meaningful on replay, not while capture records
    # their kernels.  Warm the replay/update path before measuring it.
    for _ in range(3):
        engine.model.set_training_step(engine.global_step)
        graph.replay()
        engine._gradient_lifecycle(collect_diagnostics=False)
        engine.optimizer.step()
        engine.schedule.step()
        engine.global_step += 1
    torch.cuda.synchronize()
    replay_seconds: list[float] = []
    for _ in range(10):
        started = time.perf_counter()
        engine.model.set_training_step(engine.global_step)
        graph.replay()
        engine._gradient_lifecycle(collect_diagnostics=False)
        engine.optimizer.step()
        engine.schedule.step()
        engine.global_step += 1
        torch.cuda.synchronize()
        replay_seconds.append(time.perf_counter() - started)
    eager_median = statistics.median(eager_seconds)
    replay_median = statistics.median(replay_seconds)
    print("eager_seconds", eager_seconds)
    print("replay_seconds", replay_seconds)
    print("eager_median_seconds", eager_median)
    print("replay_median_seconds", replay_median)
    print("median_speedup", eager_median / replay_median)
    print("replay_loss", float(captured_ledger.total.detach()))


if __name__ == "__main__":
    main()
