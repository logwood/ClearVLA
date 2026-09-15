from __future__ import annotations

"""Probe the native arm source process without constructing the policy."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from clearvla.policy.source_process import BoundaryConditionedArmSource


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ar1", "boundary_multiscale", "all"),
        default="all",
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--arm-dim", type=int, default=6)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--innovation-weight", type=float, default=0.50)
    parser.add_argument("--velocity-weight", type=float, default=0.35)
    parser.add_argument("--acceleration-weight", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().double().cpu())


@torch.no_grad()
def _probe(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    source = BoundaryConditionedArmSource(
        horizon=args.horizon,
        arm_dim=args.arm_dim,
        mode=mode,
        temporal_rho=args.rho,
        scale=args.scale,
        innovation_weight=args.innovation_weight,
        velocity_weight=args.velocity_weight,
        acceleration_weight=args.acceleration_weight,
    ).eval()
    state = torch.zeros(args.samples, args.arm_dim)
    samples = source.sample(
        state,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed),
    )
    diagnostics = source.diagnostics(samples, state)
    stochastic = samples - source.conditional_mean(state)
    flat = stochastic.permute(1, 0, 2).reshape(args.horizon, -1).double()
    empirical_covariance = flat @ flat.T / float(flat.shape[1])
    covariance = source.covariance.double()
    covariance_relative_error = (
        empirical_covariance - covariance
    ).norm() / covariance.norm().clamp_min(1e-12)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    component_traces = source.component_covariances.double().diagonal(dim1=-2, dim2=-1).sum(
        dim=-1
    ) / float(args.horizon)
    return {
        "mode": mode,
        "horizon": args.horizon,
        "arm_dim": args.arm_dim,
        "samples": args.samples,
        "rho": args.rho,
        "scale": source.geometry.scale,
        "normalized_weights": {
            "innovation": source.geometry.innovation_weight,
            "velocity": source.geometry.velocity_weight,
            "acceleration": source.geometry.acceleration_weight,
        },
        "component_mean_variances": [float(value) for value in component_traces],
        "analytic": {
            "mean_variance": _scalar(torch.trace(covariance) / float(args.horizon)),
            "min_eigenvalue": _scalar(eigenvalues.min()),
            "max_eigenvalue": _scalar(eigenvalues.max()),
            "condition": _scalar(source.covariance_condition),
            "effective_dimension": _scalar(source.covariance_effective_dimension),
            "source_rms": _scalar(source.expected_source_rms),
            "delta_rms": _scalar(source.expected_delta_rms),
            "acceleration_rms": _scalar(source.expected_acceleration_rms),
            "first_step_std": _scalar(source.expected_first_step_std),
            "terminal_std": _scalar(source.expected_terminal_std),
        },
        "empirical": {
            key.removeprefix("arm_source_"): _scalar(value) for key, value in diagnostics.items()
        },
        "empirical_covariance_relative_frobenius_error": _scalar(covariance_relative_error),
    }


def main() -> int:
    args = _parse_args()
    modes = ("ar1", "boundary_multiscale") if args.mode == "all" else (args.mode,)
    report = {
        "schema": "clearvla-arm-source-process-probe-v1",
        "results": [_probe(args, mode) for mode in modes],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
