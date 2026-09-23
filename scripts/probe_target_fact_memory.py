"""Measure the production TargetFact memory seam on a fixed synthetic CALVIN shape.

This is a diagnostic only.  It deliberately builds the same public online input
shape as the CALVIN target-action profile (B, 3 history frames, two cameras,
256 DINO patches, 336px RGB, 24x7 action field) without touching the dataset.
The ``microgrid_only`` variant disables only the local-refiner checkpoint helper;
the contraction implementation remains the current one.  This makes the
memory contribution of the two safe optimizations measurable on one GPU.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.interfaces import (
    CurrentObservation,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.v120_p1 import LateRawDetailPolicyReader


def _input(config, batch: int, device: torch.device) -> OnlinePolicyInput:
    dims = config.dimensions
    data = config.data
    return OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=torch.randn(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                device=device,
                dtype=torch.bfloat16,
            ),
            raw_rgb=torch.randn(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                3,
                data.cache_side,
                data.cache_side,
                device=device,
                dtype=torch.float32,
            ),
        ),
        history=ObservableHistory(
            state=torch.randn(batch, dims.state_dim, device=device),
            action_state=torch.randn(batch, dims.action_dim, device=device),
            codec_gripper_boundary=torch.randn(batch, 1, device=device),
            state_history=torch.randn(
                batch, dims.state_history_length, dims.state_dim, device=device
            ),
            executed_action_history=torch.randn(
                batch,
                dims.executed_history_length,
                dims.action_dim,
                device=device,
            ),
        ),
        goal=GoalCondition(
            tokens=torch.randn(
                batch,
                dims.goal_max_tokens,
                dims.goal_token_dim,
                device=device,
            ),
            mask=torch.ones(
                batch, dims.goal_max_tokens, device=device, dtype=torch.bool
            ),
        ),
    )


def _normalizer(action_dim: int) -> ArrayNormalizer:
    zeros = np.zeros((2, action_dim), dtype=np.float32)
    return ArrayNormalizer.fit_identity([zeros])


def _gib(value: int) -> float:
    return float(value) / float(2**30)


def run(*, batch: int, variant: str, backward: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    if variant not in {"combo", "microgrid_only"}:
        raise ValueError(f"unknown variant {variant!r}")
    config = load_config("configs/mainline/calvin_push_color6_target_fact_v1.json")
    device = torch.device("cuda")
    torch.manual_seed(17_022)
    model = ClearVLAMainlinePolicy(config).to(device)
    model.configure_action_normalizer(_normalizer(config.dimensions.action_dim))
    model.train()
    if variant == "microgrid_only":
        # Keep the current contraction exactly as-is and remove only the
        # local-refiner checkpoint wrapper.  The replacement is process-local.
        def direct(refiner, *, checkpoint_active, kwargs):
            del checkpoint_active
            return refiner(**kwargs)

        if hasattr(LateRawDetailPolicyReader, "_run_typed_local_refiner"):
            LateRawDetailPolicyReader._run_typed_local_refiner = staticmethod(direct)

    policy_input = _input(config, int(batch), device)
    noisy = torch.randn(
        batch,
        config.dimensions.action_horizon,
        model.outlet_adapter.physical_dim,
        device=device,
    )
    time_tensor = torch.full((batch,), 0.5, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    encode_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cache, state, encode_metrics = model.encode_online(
            policy_input,
            collect_diagnostics=False,
        )
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_start
    encode_alloc = torch.cuda.memory_allocated(device)
    encode_reserved = torch.cuda.memory_reserved(device)
    encode_peak = torch.cuda.max_memory_allocated(device)
    velocity_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.velocity(
            cache,
            noisy_action_field=noisy,
            time=time_tensor,
            collect_diagnostics=False,
        )
        loss = output.bottom.physical_velocity.float().square().mean()
        if backward:
            loss.backward()
    torch.cuda.synchronize(device)
    velocity_seconds = time.perf_counter() - velocity_start
    result = {
        "schema": "clearvla-target-fact-memory-v1",
        "variant": variant,
        "batch": int(batch),
        "backward": bool(backward),
        "device": str(device),
        "raw_activation_checkpoint": bool(
            model.p1.factual_reader.raw_activation_checkpoint
        ),
        "encode_allocated_gib": _gib(encode_alloc),
        "encode_reserved_gib": _gib(encode_reserved),
        "encode_peak_allocated_gib": _gib(encode_peak),
        "encode_seconds": encode_seconds,
        "velocity_allocated_gib": _gib(torch.cuda.memory_allocated(device)),
        "velocity_reserved_gib": _gib(torch.cuda.memory_reserved(device)),
        "velocity_peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
        "velocity_seconds": velocity_seconds,
        "loss": float(loss.detach().cpu()),
        "encode_metric_count": len(encode_metrics),
        "state_type": type(state).__name__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--variant", choices=("combo", "microgrid_only"), default="combo"
    )
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    run(batch=args.batch, variant=args.variant, backward=args.backward)


if __name__ == "__main__":
    main()
