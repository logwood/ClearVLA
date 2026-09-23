"""Large-support P1 recomputation is numerical quadrature, not candidate pruning."""

from __future__ import annotations

import gc
import resource
import sys
from dataclasses import replace
from typing import Any

import pytest
import torch
from test_mainline_candidate_support import _micro_case
from test_mainline_endpoint_supervision import _config
from torch.utils.checkpoint import checkpoint

import clearvla.mainline.model.v120_p1 as p1
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.qualification import synthetic_batch
from clearvla.vision.candidate_support import posterior_microgrid_expectation


@pytest.fixture(autouse=True)
def release_graphs():
    yield
    gc.collect()
    # Return freed CPU allocator arenas between large independent test cases.
    # This cannot free live tensors or change any production numerical path.
    if sys.platform.startswith("linux"):
        import ctypes

        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)


@pytest.mark.parametrize("patches,bf16", [(100, False), (100, True)])
def test_large_native_chart_action_backward_recomputes_microgrid(patches: int, bf16: bool) -> None:
    exercise_backward(patches, bf16)


def exercise_backward(patches: int, bf16: bool) -> None:
    c = _config()
    c = replace(c, dimensions=replace(c.dimensions, patches_per_camera=patches))
    torch.manual_seed(851)
    m = ClearVLAMainlinePolicy(c).eval()
    b, n = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"))
    m.configure_action_normalizer(n)
    calls = []
    real_checkpoint = p1.checkpoint

    def checked(fn: Any, *args: Any, **kwargs: Any):
        calls.append(getattr(fn, "__name__", str(fn)))
        return real_checkpoint(fn, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(p1, "checkpoint", checked)
        print("construct", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            cache, state, _ = m.encode_online(b.online, collect_diagnostics=False)
            print("encode", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
            coarse = state.observation.progressive_state.coarse_logits
            assert coarse is not None
            coarse.retain_grad()
            out = m.velocity(
                cache,
                noisy_action_field=torch.randn(1, 24, 18),
                time=torch.full((1,), 0.4),
                collect_diagnostics=False,
            )
            loss = out.bottom.physical_velocity[:, 0].float().square().mean()
        print("velocity", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
        loss.backward()
        print("backward", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
    assert "_posterior_microgrid_expectation" in calls
    assert coarse.grad is not None and coarse.grad.shape[-1] == patches
    assert torch.isfinite(coarse.grad).all() and coarse.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_no_grad_large_native_inference_does_not_request_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _config()
    c = replace(c, dimensions=replace(c.dimensions, patches_per_camera=256))
    m = ClearVLAMainlinePolicy(c).eval()
    b, n = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"))
    m.configure_action_normalizer(n)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("inference must not allocate recomputation state")

    monkeypatch.setattr(p1, "checkpoint", forbidden)
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online, collect_diagnostics=False)
        out = m.velocity(
            cache,
            noisy_action_field=torch.randn(1, 24, 18),
            time=torch.full((1,), 0.4),
            collect_diagnostics=False,
        )
    assert torch.isfinite(out.bottom.physical_velocity).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_pure_quadrature_values_gradients_and_rng_are_exact_under_recomputation(bf16: bool) -> None:
    chart, points, logits, center = _micro_case()
    args = (
        torch.ones(1, 1, 1, 1, 1, 1, 1),
        logits,
        torch.ones(1, 1, 1, 1, 1, 2, dtype=torch.bool),
        points,
        torch.full((1, 1, 1, 1, 1), 0.1),
        chart,
        chart.clone(),
        center,
        center.clone(),
    )
    results = []
    for remat in (False, True):
        inputs = tuple(x.clone().requires_grad_(x.is_floating_point()) for x in args)
        before = torch.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            result = (
                checkpoint(posterior_microgrid_expectation, *inputs, use_reentrant=False)
                if remat
                else posterior_microgrid_expectation(*inputs)
            )
            assert isinstance(result, tuple)
            assert all(isinstance(x, torch.Tensor) for x in result)
            torch.stack([x.square().sum() for x in result]).sum().backward()
        assert torch.equal(before, torch.get_rng_state())
        results.append((tuple(x.detach() for x in result), tuple(x.grad for x in inputs)))
    for left, right in zip(*results):
        for a, b in zip(left, right):
            if a is None or b is None:
                assert a is b
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    exercise_backward(int(sys.argv[1]), bool(int(sys.argv[2])))
