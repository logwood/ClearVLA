from __future__ import annotations

"""Mechanistic experiments for boundary-conditioned action source design.

The probe is deliberately independent of the training dataset and full policy.
It compares source geometry, synthetic flow-regression difficulty, condition
reader topology, and jointly learned source degeneracy using only PyTorch.
"""

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from clearvla.policy.codec import ParsevalGripperTemporalFrame
from clearvla.policy.source_process import BoundaryConditionedArmSource


HORIZON = 24
EVENT_DIM = 3
TIMING_DIM = 3
SEMANTIC_DIM = EVENT_DIM + TIMING_DIM

# Supplied by the project's 29,180-window DCT dataset probe. These are raw
# trajectory energies, not boundary-residual energies; the distinction is one
# of the questions this study audits.
GRIPPER_DCT_ENERGY = torch.tensor(
    [
        0.970336035833232,
        0.025759236157110205,
        0.0019313747799277593,
        0.0005975653016929062,
        0.00027444095918742565,
        0.00019258255323090256,
        0.0001355002094898389,
        0.00011134192925687862,
        8.562883726485639e-05,
        6.291443376816052e-05,
        4.980971405050464e-05,
        4.675692200160643e-05,
        4.065591438404202e-05,
        3.276109327237601e-05,
        2.6846464622852077e-05,
        2.5285727066086583e-05,
        2.5737385478602114e-05,
        2.875151683732913e-05,
        3.706529687007378e-05,
        5.495534852513534e-05,
        5.598580988610346e-05,
        3.555059770319234e-05,
        2.594559086076686e-05,
        2.7271624280448425e-05,
    ],
    dtype=torch.float64,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--field-steps", type=int, default=700)
    parser.add_argument("--fusion-steps", type=int, default=900)
    parser.add_argument("--collapse-steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--section",
        choices=("all", "source-learning"),
        default="all",
        help="Run the full study or only the learned-source identifiability probe.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _scalar(value: Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().double().cpu())
    return float(value)


def _mean_std(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def _orthonormal_dct(horizon: int, *, dtype: torch.dtype = torch.float64) -> Tensor:
    n = torch.arange(horizon, dtype=dtype)[None]
    k = torch.arange(horizon, dtype=dtype)[:, None]
    matrix = torch.cos(math.pi / float(horizon) * (n + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / float(horizon))
    if horizon > 1:
        matrix[1:] *= math.sqrt(2.0 / float(horizon))
    return matrix


def _difference_matrix(horizon: int, order: int = 1) -> Tensor:
    difference = torch.eye(horizon, dtype=torch.float64)
    if horizon > 1:
        difference[1:, :-1] -= torch.eye(horizon - 1, dtype=torch.float64)
    result = difference
    for _ in range(order - 1):
        result = difference @ result
    return result


def _trace_normalize(covariance: Tensor, scale: float = 1.0) -> Tensor:
    mean_variance = torch.trace(covariance) / float(covariance.shape[0])
    return covariance / mean_variance.clamp_min(1e-12) * (scale * scale)


def _shrink_covariance(covariance: Tensor, shrinkage: float, scale: float = 1.0) -> Tensor:
    normalized = _trace_normalize(covariance)
    identity = torch.eye(int(covariance.shape[0]), dtype=torch.float64)
    return _trace_normalize(
        (1.0 - shrinkage) * normalized + shrinkage * identity,
        scale=scale,
    )


def _covariance_metrics(covariance: Tensor) -> dict[str, Any]:
    covariance = covariance.double()
    horizon = int(covariance.shape[0])
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    difference = _difference_matrix(horizon, order=1)
    acceleration = _difference_matrix(horizon, order=2)
    delta_covariance = difference @ covariance @ difference.T
    acceleration_covariance = acceleration @ covariance @ acceleration.T
    dct = _orthonormal_dct(horizon)
    spectral_variance = (dct @ covariance @ dct.T).diagonal().clamp_min(0.0)
    spectral_fraction = spectral_variance / spectral_variance.sum().clamp_min(1e-12)
    trace = eigenvalues.sum()
    return {
        "mean_variance": _scalar(trace / float(horizon)),
        "source_rms": _scalar((trace / float(horizon)).sqrt()),
        "delta_rms": _scalar((torch.trace(delta_covariance) / float(horizon)).sqrt()),
        "acceleration_rms": _scalar((torch.trace(acceleration_covariance) / float(horizon)).sqrt()),
        "first_step_std": _scalar(covariance[0, 0].sqrt()),
        "terminal_std": _scalar(covariance[-1, -1].sqrt()),
        "min_eigenvalue": _scalar(eigenvalues[0]),
        "max_eigenvalue": _scalar(eigenvalues[-1]),
        "condition": _scalar(eigenvalues[-1] / eigenvalues[0].clamp_min(1e-15)),
        "effective_dimension": _scalar(
            trace.square() / eigenvalues.square().sum().clamp_min(1e-15)
        ),
        "dct_energy_first_1": _scalar(spectral_fraction[:1].sum()),
        "dct_energy_first_2": _scalar(spectral_fraction[:2].sum()),
        "dct_energy_first_4": _scalar(spectral_fraction[:4].sum()),
        "dct_energy_first_8": _scalar(spectral_fraction[:8].sum()),
        "dct_energy_tail_8": _scalar(spectral_fraction[-8:].sum()),
    }


def _source_covariances(empirical_residual_covariance: Tensor) -> dict[str, Tensor]:
    identity = torch.eye(HORIZON, dtype=torch.float64)
    residual_scale = float((torch.trace(empirical_residual_covariance) / float(HORIZON)).sqrt())
    raw_spectral = (
        _orthonormal_dct(HORIZON).T
        @ torch.diag(HORIZON * GRIPPER_DCT_ENERGY / GRIPPER_DCT_ENERGY.sum())
        @ _orthonormal_dct(HORIZON)
    )
    ar_070 = BoundaryConditionedArmSource(
        horizon=HORIZON,
        arm_dim=1,
        mode="ar1",
        temporal_rho=0.70,
        scale=1.0,
        innovation_weight=1.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
    ).covariance.double()
    ar_095 = BoundaryConditionedArmSource(
        horizon=HORIZON,
        arm_dim=1,
        mode="ar1",
        temporal_rho=0.95,
        scale=1.0,
        innovation_weight=1.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
    ).covariance.double()
    multiscale = BoundaryConditionedArmSource(
        horizon=HORIZON,
        arm_dim=1,
        mode="boundary_multiscale",
        temporal_rho=0.0,
        scale=1.0,
        innovation_weight=0.50,
        velocity_weight=0.35,
        acceleration_weight=0.15,
    ).covariance.double()
    covariances = {
        "white": identity,
        "ar1_rho_070": ar_070,
        "ar1_rho_095": ar_095,
        "arm_multiscale": multiscale,
        "raw_gripper_dct": _trace_normalize(raw_spectral),
        "raw_gripper_dct_shrink_015": _shrink_covariance(raw_spectral, 0.15),
        "raw_gripper_dct_shrink_030": _shrink_covariance(raw_spectral, 0.30),
        "raw_gripper_dct_shrink_050": _shrink_covariance(raw_spectral, 0.50),
        "raw_gripper_dct_shrink_070": _shrink_covariance(raw_spectral, 0.70),
        "residual_empirical_shrink_015": _shrink_covariance(empirical_residual_covariance, 0.15),
        "residual_empirical_shrink_030": _shrink_covariance(empirical_residual_covariance, 0.30),
        "residual_empirical_shrink_050": _shrink_covariance(empirical_residual_covariance, 0.50),
        "residual_empirical_shrink_070": _shrink_covariance(empirical_residual_covariance, 0.70),
    }
    covariances.update(
        {
            "white_scale_matched": _trace_normalize(identity, residual_scale),
            "arm_multiscale_scale_matched": _trace_normalize(multiscale, residual_scale),
            "raw_gripper_dct_shrink_030_scale_matched": _shrink_covariance(
                raw_spectral, 0.30, residual_scale
            ),
            "raw_gripper_dct_shrink_070_scale_matched": _shrink_covariance(
                raw_spectral, 0.70, residual_scale
            ),
            "residual_empirical_shrink_030_scale_matched": _shrink_covariance(
                empirical_residual_covariance, 0.30, residual_scale
            ),
            "residual_empirical_shrink_070_scale_matched": _shrink_covariance(
                empirical_residual_covariance, 0.70, residual_scale
            ),
        }
    )
    return covariances


@dataclass
class SyntheticBatch:
    state: Tensor
    semantic: Tensor
    target: Tensor
    event: Tensor
    timing: Tensor
    ood: Tensor


def _select_batch(batch: SyntheticBatch, index: Tensor) -> SyntheticBatch:
    return SyntheticBatch(
        state=batch.state[index],
        semantic=batch.semantic[index],
        target=batch.target[index],
        event=batch.event[index],
        timing=batch.timing[index],
        ood=batch.ood[index],
    )


def _smoothstep(value: Tensor) -> Tensor:
    value = value.clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def _sample_synthetic_gripper(
    batch: int,
    *,
    device: torch.device,
    split: str = "all",
) -> SyntheticBatch:
    if split not in {"all", "train", "iid", "ood"}:
        raise ValueError(f"unsupported synthetic split: {split}")
    chunks: list[SyntheticBatch] = []
    count = 0
    while count < batch:
        draw = max(batch - count, 256)
        state_class = torch.randint(0, 3, (draw,), device=device)
        state_centers = torch.tensor([-0.75, 0.0, 0.75], device=device)
        state = state_centers[state_class] + 0.04 * torch.randn(draw, device=device)
        event = torch.randint(0, EVENT_DIM, (draw,), device=device)
        timing = torch.randint(0, TIMING_DIM, (draw,), device=device)
        ood = ((state_class == 2) & (event == 1) & (timing == 2)) | (
            (state_class == 0) & (event == 2) & (timing == 0)
        )
        if split in {"train", "iid"}:
            keep = ~ood
        elif split == "ood":
            keep = ood
        else:
            keep = torch.ones_like(ood)
        if not bool(keep.any()):
            continue
        state = state[keep]
        event = event[keep]
        timing = timing[keep]
        ood = ood[keep]
        event_one_hot = torch.nn.functional.one_hot(event, EVENT_DIM).float()
        timing_one_hot = torch.nn.functional.one_hot(timing, TIMING_DIM).float()
        semantic = torch.cat([event_one_hot, timing_one_hot], dim=-1)

        destination = state.clone()
        destination[event == 1] = -0.88
        destination[event == 2] = 0.88
        destination = destination + 0.035 * torch.randn_like(destination)
        start_lookup = torch.tensor([2.0, 9.0, 15.0], device=device)
        start = start_lookup[timing] + 0.8 * (torch.rand_like(state) - 0.5)
        width = 2.5 + 1.5 * torch.rand_like(state)
        horizon_position = torch.arange(HORIZON, device=device).float()[None]
        progress = _smoothstep((horizon_position - start[:, None]) / width[:, None])
        progress[event == 0] = 0.0
        target = state[:, None] + (destination - state)[:, None] * progress

        # Small condition-unobserved low-frequency drift prevents the synthetic
        # task from becoming a deterministic lookup table.
        drift_end = 0.025 * torch.randn_like(state)
        drift = torch.linspace(0.0, 1.0, HORIZON, device=device)[None]
        target = (target + drift_end[:, None] * drift).clamp(-1.0, 1.0)
        chunks.append(
            SyntheticBatch(
                state=state,
                semantic=semantic,
                target=target,
                event=event,
                timing=timing,
                ood=ood,
            )
        )
        count += int(state.shape[0])
    return SyntheticBatch(
        state=torch.cat([row.state for row in chunks])[:batch],
        semantic=torch.cat([row.semantic for row in chunks])[:batch],
        target=torch.cat([row.target for row in chunks])[:batch],
        event=torch.cat([row.event for row in chunks])[:batch],
        timing=torch.cat([row.timing for row in chunks])[:batch],
        ood=torch.cat([row.ood for row in chunks])[:batch],
    )


def _semantic_prototype(state: Tensor, semantic: Tensor) -> Tensor:
    event = semantic[:, :EVENT_DIM].argmax(dim=-1)
    timing = semantic[:, EVENT_DIM:].argmax(dim=-1)
    destination = state.clone()
    destination[event == 1] = -0.88
    destination[event == 2] = 0.88
    start_lookup = torch.tensor([2.0, 9.0, 15.0], device=state.device)
    start = start_lookup[timing]
    position = torch.arange(HORIZON, device=state.device).float()[None]
    progress = _smoothstep((position - start[:, None]) / 3.25)
    progress[event == 0] = 0.0
    return state[:, None] + (destination - state)[:, None] * progress


class SourceSampler:
    def __init__(
        self,
        covariances: dict[str, Tensor],
        device: torch.device,
        *,
        residual_scale: float,
    ) -> None:
        self.factors = {
            key: torch.linalg.cholesky(value + 1e-9 * torch.eye(HORIZON, dtype=torch.float64))
            .float()
            .to(device)
            for key, value in covariances.items()
        }
        self.device = device
        self.residual_scale = float(residual_scale)

    def sample(self, name: str, batch: SyntheticBatch) -> Tensor:
        if name == "unanchored_white":
            return torch.randn_like(batch.target)
        if name == "semantic_shift_diagnostic":
            prototype = _semantic_prototype(batch.state, batch.semantic)
            return prototype + self.residual_scale * torch.randn_like(prototype)
        if name not in self.factors:
            raise KeyError(name)
        white = torch.randn_like(batch.target)
        residual = torch.einsum("ts,bs->bt", self.factors[name], white)
        if name.startswith("ar1_rho_"):
            rho = 0.70 if name.endswith("070") else 0.95
            gain = torch.as_tensor(rho, device=batch.state.device) ** torch.arange(
                1, HORIZON + 1, device=batch.state.device
            )
            mean = gain[None] * batch.state[:, None]
        else:
            mean = batch.state[:, None]
        return mean + residual


def _empirical_residual_covariance(samples: int, device: torch.device) -> Tensor:
    batch = _sample_synthetic_gripper(samples, device=device, split="all")
    residual = batch.target - batch.state[:, None]
    residual = residual - residual.mean(dim=0, keepdim=True)
    covariance = residual.T.double().cpu() @ residual.double().cpu()
    return covariance / float(max(1, samples - 1))


def _source_transport_probe(
    sampler: SourceSampler,
    candidates: list[str],
    *,
    samples: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(99173)
    batch = _sample_synthetic_gripper(samples, device=device, split="all")
    result: dict[str, Any] = {}
    for name in candidates:
        torch.manual_seed(73191)
        source = sampler.sample(name, batch)
        velocity = batch.target - source
        source_delta = source[:, 1:] - source[:, :-1]
        target_delta = batch.target[:, 1:] - batch.target[:, :-1]
        hold = batch.event == 0
        event = ~hold
        result[name] = {
            "source_rms_from_state": _scalar(
                (source - batch.state[:, None]).square().mean().sqrt()
            ),
            "source_delta_rms": _scalar(source_delta.square().mean().sqrt()),
            "target_delta_rms": _scalar(target_delta.square().mean().sqrt()),
            "bridge_velocity_rms": _scalar(velocity.square().mean().sqrt()),
            "bridge_first8_rms": _scalar(velocity[:, :8].square().mean().sqrt()),
            "bridge_tail8_rms": _scalar(velocity[:, -8:].square().mean().sqrt()),
            "hold_bridge_rms": _scalar(velocity[hold].square().mean().sqrt()),
            "event_bridge_rms": _scalar(velocity[event].square().mean().sqrt()),
            "source_first_state_rmse": _scalar((source[:, 0] - batch.state).square().mean().sqrt()),
            "source_terminal_state_rmse": _scalar(
                (source[:, -1] - batch.state).square().mean().sqrt()
            ),
            "source_target_cosine": _scalar(
                torch.nn.functional.cosine_similarity(
                    source - batch.state[:, None],
                    batch.target - batch.state[:, None],
                    dim=-1,
                ).mean()
            ),
        }
    return result


class FlatFlowField(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        input_dim = HORIZON + 1 + 1 + SEMANTIC_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, HORIZON),
        )

    def forward(self, noisy: Tensor, time_value: Tensor, state: Tensor, semantic: Tensor) -> Tensor:
        return self.net(torch.cat([noisy, time_value, state[:, None], semantic], dim=-1))


@torch.no_grad()
def _evaluate_flat_field(
    model: nn.Module,
    sampler: SourceSampler,
    source_name: str,
    *,
    device: torch.device,
    split: str,
    samples: int = 4096,
) -> dict[str, float | None]:
    torch.manual_seed(81003 if split == "iid" else 81007)
    batch = _sample_synthetic_gripper(samples, device=device, split=split)
    source = sampler.sample(source_name, batch)
    time_value = 0.05 + 0.90 * torch.rand(samples, 1, device=device)
    noisy = (1.0 - time_value) * source + time_value * batch.target
    target_velocity = batch.target - source
    prediction = model(noisy, time_value, batch.state, batch.semantic)
    error = prediction - target_velocity
    event = batch.event != 0

    def subset_mse(mask: Tensor) -> float | None:
        if not bool(mask.any()):
            return None
        return _scalar(error[mask].square().mean())

    return {
        "mse": _scalar(error.square().mean()),
        "rmse": _scalar(error.square().mean().sqrt()),
        "event_mse": subset_mse(event),
        "hold_mse": subset_mse(~event),
    }


def _train_flat_fields(
    sampler: SourceSampler,
    candidates: list[str],
    *,
    device: torch.device,
    steps: int,
    seeds: int,
    batch_size: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    checkpoints = sorted(set([0, 20, 50, 100, 200, steps // 2, steps]))
    torch.manual_seed(11991)
    train_pool = _sample_synthetic_gripper(
        max(32768, batch_size * 64), device=device, split="train"
    )
    pool_size = int(train_pool.state.shape[0])
    for candidate_index, source_name in enumerate(candidates):
        seed_rows: list[dict[str, Any]] = []
        for seed_index in range(seeds):
            seed = 12000 + seed_index
            torch.manual_seed(seed)
            model = FlatFlowField().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            trace: dict[str, float] = {}
            for step in range(steps + 1):
                index = torch.randint(0, pool_size, (batch_size,), device=device)
                batch = _select_batch(train_pool, index)
                source = sampler.sample(source_name, batch)
                time_value = 0.05 + 0.90 * torch.rand(batch_size, 1, device=device)
                noisy = (1.0 - time_value) * source + time_value * batch.target
                target_velocity = batch.target - source
                prediction = model(noisy, time_value, batch.state, batch.semantic)
                loss = (prediction - target_velocity).square().mean()
                if step in checkpoints:
                    trace[str(step)] = _scalar(loss)
                if step == steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            seed_rows.append(
                {
                    "seed": seed,
                    "trace": trace,
                    "iid": _evaluate_flat_field(
                        model, sampler, source_name, device=device, split="iid"
                    ),
                    "ood": _evaluate_flat_field(
                        model, sampler, source_name, device=device, split="ood"
                    ),
                }
            )
        reports[source_name] = {
            "seeds": seed_rows,
            "iid_mse": _mean_std([row["iid"]["mse"] for row in seed_rows]),
            "ood_mse": _mean_std([row["ood"]["mse"] for row in seed_rows]),
            "final_train_loss": _mean_std([row["trace"][str(steps)] for row in seed_rows]),
        }
        print(
            f"[field {candidate_index + 1}/{len(candidates)}] {source_name}: "
            f"iid={reports[source_name]['iid_mse']['mean']:.6f} "
            f"ood={reports[source_name]['ood_mse']['mean']:.6f}",
            flush=True,
        )
    return reports


def _sinusoidal_positions(horizon: int, hidden: int, device: torch.device) -> Tensor:
    position = torch.arange(horizon, device=device).float()[:, None]
    frequencies = torch.exp(
        torch.arange(0, hidden, 2, device=device).float() * (-math.log(10000.0) / float(hidden))
    )
    embedding = torch.zeros(horizon, hidden, device=device)
    embedding[:, 0::2] = torch.sin(position * frequencies)
    embedding[:, 1::2] = torch.cos(position * frequencies[: embedding[:, 1::2].shape[1]])
    return embedding


class ConditionReaderField(nn.Module):
    MODES = frozenset({"single_fused", "replicated_fused", "typed"})

    def __init__(self, *, mode: str, hidden: int = 72, heads: int = 6) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(f"unsupported condition-reader mode: {self.mode}")
        self.action_input = nn.Linear(1, hidden)
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        if self.mode == "typed":
            self.boundary = nn.Linear(1, hidden)
            self.event = nn.Linear(EVENT_DIM, hidden)
            self.timing = nn.Linear(TIMING_DIM, hidden)
            self.type_embedding = nn.Parameter(torch.randn(1, 3, hidden) * 0.02)
        else:
            self.fused = nn.Linear(1 + SEMANTIC_DIM, hidden)
            token_count = 1 if self.mode == "single_fused" else 3
            self.type_embedding = nn.Parameter(torch.randn(1, token_count, hidden) * 0.02)
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 3 * hidden), nn.GELU(), nn.Linear(3 * hidden, hidden)
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.register_buffer(
            "position",
            _sinusoidal_positions(HORIZON, hidden, torch.device("cpu")),
            persistent=False,
        )

    def _memory(self, state: Tensor, semantic: Tensor) -> Tensor:
        if self.mode != "typed":
            memory = self.fused(torch.cat([state[:, None], semantic], dim=-1))[:, None]
            if self.mode == "replicated_fused":
                memory = memory.expand(-1, 3, -1)
        else:
            memory = torch.stack(
                [
                    self.boundary(state[:, None]),
                    self.event(semantic[:, :EVENT_DIM]),
                    self.timing(semantic[:, EVENT_DIM:]),
                ],
                dim=1,
            )
        return memory + self.type_embedding.to(device=memory.device, dtype=memory.dtype)

    def forward(self, noisy: Tensor, time_value: Tensor, state: Tensor, semantic: Tensor) -> Tensor:
        action = self.action_input(noisy[..., None])
        action = action + self.position.to(device=action.device, dtype=action.dtype)[None]
        shift, scale = self.time(time_value).chunk(2, dim=-1)
        action = action * (1.0 + 0.1 * torch.tanh(scale[:, None])) + 0.1 * torch.tanh(
            shift[:, None]
        )
        memory = self._memory(state, semantic)
        attended, _ = self.cross_attention(
            self.query_norm(action),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        action = action + attended
        attended, _ = self.self_attention(
            self.self_norm(action),
            self.self_norm(action),
            self.self_norm(action),
            need_weights=False,
        )
        action = action + attended
        action = action + self.ffn(self.ffn_norm(action))
        return self.output(action)[..., 0]


@torch.no_grad()
def _evaluate_condition_reader(
    model: ConditionReaderField,
    sampler: SourceSampler,
    source_name: str,
    *,
    device: torch.device,
    split: str,
    samples: int = 4096,
) -> dict[str, float]:
    torch.manual_seed(44001 if split == "iid" else 44003)
    batch = _sample_synthetic_gripper(samples, device=device, split=split)
    source = sampler.sample(source_name, batch)
    time_value = 0.05 + 0.90 * torch.rand(samples, 1, device=device)
    noisy = (1.0 - time_value) * source + time_value * batch.target
    target_velocity = batch.target - source
    prediction = model(noisy, time_value, batch.state, batch.semantic)
    permutation = torch.randperm(samples, device=device)
    state_shuffled = model(noisy, time_value, batch.state[permutation], batch.semantic)
    semantic_shuffled = model(noisy, time_value, batch.state, batch.semantic[permutation])
    return {
        "mse": _scalar((prediction - target_velocity).square().mean()),
        "state_shuffle_mse": _scalar((state_shuffled - target_velocity).square().mean()),
        "semantic_shuffle_mse": _scalar((semantic_shuffled - target_velocity).square().mean()),
        "state_output_change_rms": _scalar((state_shuffled - prediction).square().mean().sqrt()),
        "semantic_output_change_rms": _scalar(
            (semantic_shuffled - prediction).square().mean().sqrt()
        ),
    }


def _condition_reader_experiment(
    sampler: SourceSampler,
    *,
    device: torch.device,
    steps: int,
    seeds: int,
    batch_size: int,
) -> dict[str, Any]:
    source_name = "residual_empirical_shrink_030_scale_matched"
    reports: dict[str, Any] = {}
    torch.manual_seed(32991)
    train_pool = _sample_synthetic_gripper(
        max(32768, batch_size * 64), device=device, split="train"
    )
    pool_size = int(train_pool.state.shape[0])
    modes = (
        ("single_fused", "single_fused_condition_token"),
        ("replicated_fused", "replicated_fused_condition_tokens"),
        ("typed", "typed_boundary_semantic_tokens"),
    )
    for mode, name in modes:
        rows: list[dict[str, Any]] = []
        for seed_index in range(seeds):
            seed = 33000 + seed_index
            torch.manual_seed(seed)
            model = ConditionReaderField(mode=mode).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            for _ in range(steps):
                index = torch.randint(0, pool_size, (batch_size,), device=device)
                batch = _select_batch(train_pool, index)
                source = sampler.sample(source_name, batch)
                time_value = 0.05 + 0.90 * torch.rand(batch_size, 1, device=device)
                noisy = (1.0 - time_value) * source + time_value * batch.target
                target_velocity = batch.target - source
                loss = (
                    (model(noisy, time_value, batch.state, batch.semantic) - target_velocity)
                    .square()
                    .mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            rows.append(
                {
                    "seed": seed,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "iid": _evaluate_condition_reader(
                        model, sampler, source_name, device=device, split="iid"
                    ),
                    "ood": _evaluate_condition_reader(
                        model, sampler, source_name, device=device, split="ood"
                    ),
                }
            )
        reports[name] = {
            "seeds": rows,
            "parameters": rows[0]["parameters"],
            "iid_mse": _mean_std([row["iid"]["mse"] for row in rows]),
            "ood_mse": _mean_std([row["ood"]["mse"] for row in rows]),
            "iid_state_shuffle_ratio": _mean_std(
                [row["iid"]["state_shuffle_mse"] / row["iid"]["mse"] for row in rows]
            ),
            "iid_semantic_shuffle_ratio": _mean_std(
                [row["iid"]["semantic_shuffle_mse"] / row["iid"]["mse"] for row in rows]
            ),
        }
        print(
            f"[reader] {name}: iid={reports[name]['iid_mse']['mean']:.6f} "
            f"ood={reports[name]['ood_mse']['mean']:.6f}",
            flush=True,
        )
    return reports


class LearnableSource(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {
            "fixed",
            "global_shift_only",
            "centered_shift_only",
            "shift_only",
            "affine",
        }:
            raise ValueError(mode)
        self.mode = mode
        if mode != "fixed":
            shift_dim = 1 if mode == "global_shift_only" else HORIZON
            self.shift = nn.Sequential(
                nn.Linear(1 + SEMANTIC_DIM, 64),
                nn.SiLU(),
                nn.Linear(64, shift_dim),
            )
            nn.init.zeros_(self.shift[-1].weight)
            nn.init.zeros_(self.shift[-1].bias)
        if mode == "affine":
            self.log_scale = nn.Parameter(torch.zeros(HORIZON))

    def forward(self, batch: SyntheticBatch, white: Tensor) -> Tensor:
        shift = torch.zeros_like(white)
        if self.mode != "fixed":
            shift = self.shift(torch.cat([batch.state[:, None], batch.semantic], dim=-1))
            if self.mode == "global_shift_only":
                shift = shift.expand(-1, HORIZON)
            elif self.mode == "centered_shift_only":
                shift = shift - shift.mean(dim=-1, keepdim=True)
        scale = self.log_scale.exp()[None] if self.mode == "affine" else 1.0
        return batch.state[:, None] + shift + scale * white

    def scale(self) -> Tensor:
        if self.mode == "affine":
            return self.log_scale.exp().mean()
        return torch.ones(())

    def scale_range(self) -> tuple[Tensor, Tensor]:
        if self.mode == "affine":
            scale = self.log_scale.exp()
            return scale.min(), scale.max()
        one = torch.ones(())
        return one, one


def _learned_source_collapse_experiment(
    *,
    device: torch.device,
    steps: int,
    seeds: int,
    batch_size: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    checkpoints = sorted(set([0, 20, 50, 100, 200, steps // 2, steps]))
    torch.manual_seed(54991)
    train_pool = _sample_synthetic_gripper(
        max(32768, batch_size * 64), device=device, split="train"
    )
    pool_size = int(train_pool.state.shape[0])
    for mode in (
        "fixed",
        "global_shift_only",
        "centered_shift_only",
        "shift_only",
        "affine",
    ):
        rows: list[dict[str, Any]] = []
        for seed_index in range(seeds):
            torch.manual_seed(55000 + seed_index)
            source_model = LearnableSource(mode).to(device)
            field = FlatFlowField().to(device)
            parameter_groups: list[dict[str, Any]] = [{"params": field.parameters(), "lr": 2e-3}]
            source_parameters = list(source_model.parameters())
            if source_parameters:
                parameter_groups.append({"params": source_parameters, "lr": 5e-3})
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
            trace: dict[str, Any] = {}
            for step in range(steps + 1):
                index = torch.randint(0, pool_size, (batch_size,), device=device)
                batch = _select_batch(train_pool, index)
                white = torch.randn_like(batch.target)
                source = source_model(batch, white)
                time_value = 0.05 + 0.90 * torch.rand(batch_size, 1, device=device)
                noisy = (1.0 - time_value) * source + time_value * batch.target
                target_velocity = batch.target - source
                prediction = field(noisy, time_value, batch.state, batch.semantic)
                loss = (prediction - target_velocity).square().mean()
                if step in checkpoints:
                    scale_min, scale_max = source_model.scale_range()
                    trace[str(step)] = {
                        "loss": _scalar(loss),
                        "source_scale": _scalar(source_model.scale().to(device)),
                        "source_scale_min": _scalar(scale_min.to(device)),
                        "source_scale_max": _scalar(scale_max.to(device)),
                        "source_target_rmse": _scalar(target_velocity.square().mean().sqrt()),
                    }
                if step == steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(field.parameters(), 5.0)
                optimizer.step()
            rows.append({"seed": 55000 + seed_index, "trace": trace})
        reports[mode] = {
            "seeds": rows,
            "final_loss": _mean_std([row["trace"][str(steps)]["loss"] for row in rows]),
            "final_source_scale": _mean_std(
                [row["trace"][str(steps)]["source_scale"] for row in rows]
            ),
            "final_source_target_rmse": _mean_std(
                [row["trace"][str(steps)]["source_target_rmse"] for row in rows]
            ),
        }
        print(
            f"[source-learning] {mode}: loss={reports[mode]['final_loss']['mean']:.6f} "
            f"scale={reports[mode]['final_source_scale']['mean']:.4f}",
            flush=True,
        )
    return reports


def _parseval_roundtrip(
    frame: ParsevalGripperTemporalFrame,
    native: Tensor,
    *,
    precision: str,
) -> float:
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        return _scalar((frame.synthesis(frame.analysis(native)) - native).square().mean().sqrt())
    finally:
        torch.set_float32_matmul_precision(previous)


def _parseval_probe(device: torch.device) -> dict[str, float | str]:
    frame = ParsevalGripperTemporalFrame(HORIZON, 6).to(device)
    matrix = frame.analysis_matrix.double().cpu().reshape(HORIZON * 6, HORIZON)
    gram = matrix.T @ matrix
    singular = torch.linalg.svdvals(matrix)
    native = torch.randn(8192, HORIZON, 1, device=device)
    field = frame.analysis(native)
    native_delta = native[:, 1:] - native[:, :-1]
    return {
        "ambient_matmul_precision": torch.get_float32_matmul_precision(),
        "gram_identity_max_abs": _scalar(
            (gram - torch.eye(HORIZON, dtype=torch.float64)).abs().max()
        ),
        "singular_min": _scalar(singular.min()),
        "singular_max": _scalar(singular.max()),
        "roundtrip_rmse_highest": _parseval_roundtrip(frame, native, precision="highest"),
        "roundtrip_rmse_high_tf32": _parseval_roundtrip(frame, native, precision="high"),
        "native_white_rms": _scalar(native.square().mean().sqrt()),
        "native_white_delta_rms": _scalar(native_delta.square().mean().sqrt()),
        "field_rms": _scalar(field.square().mean().sqrt()),
    }


def main() -> int:
    args = _parse_args()
    if args.quick:
        args.samples = min(args.samples, 8192)
        args.field_steps = min(args.field_steps, 80)
        args.fusion_steps = min(args.fusion_steps, 100)
        args.collapse_steps = min(args.collapse_steps, 120)
        args.seeds = min(args.seeds, 1)
        args.batch_size = min(args.batch_size, 128)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    if args.section == "source-learning":
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        report: dict[str, Any] = {
            "schema": "clearvla-learned-source-identifiability-probe-v1",
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "config": {
                "seed": args.seed,
                "collapse_steps": args.collapse_steps,
                "seeds": args.seeds,
                "batch_size": args.batch_size,
            },
            "learned_source_degeneracy": _learned_source_collapse_experiment(
                device=device,
                steps=args.collapse_steps,
                seeds=args.seeds,
                batch_size=args.batch_size,
            ),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report["environment"]["peak_cuda_memory_mib"] = round(
                torch.cuda.max_memory_allocated(device) / 2**20, 3
            )
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}", flush=True)
        print(f"elapsed={report['elapsed_seconds']:.1f}s", flush=True)
        return 0

    parseval_report = _parseval_probe(device)
    # Keep all geometry checks above on the production-default highest FP32
    # path. The remaining experiments train disposable toy networks, where
    # TF32 changes throughput but not the mechanism under test.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    empirical_covariance = _empirical_residual_covariance(args.samples, device)
    residual_scale = float((torch.trace(empirical_covariance) / float(HORIZON)).sqrt())
    covariances = _source_covariances(empirical_covariance)
    sampler = SourceSampler(covariances, device, residual_scale=residual_scale)
    source_candidates = [
        "unanchored_white",
        "white",
        "white_scale_matched",
        "ar1_rho_070",
        "ar1_rho_095",
        "arm_multiscale",
        "arm_multiscale_scale_matched",
        "raw_gripper_dct_shrink_030",
        "raw_gripper_dct_shrink_030_scale_matched",
        "raw_gripper_dct_shrink_070_scale_matched",
        "residual_empirical_shrink_030",
        "residual_empirical_shrink_030_scale_matched",
        "residual_empirical_shrink_070_scale_matched",
        "semantic_shift_diagnostic",
    ]
    field_candidates = [
        "unanchored_white",
        "white",
        "white_scale_matched",
        "ar1_rho_095",
        "arm_multiscale_scale_matched",
        "raw_gripper_dct_shrink_030_scale_matched",
        "raw_gripper_dct_shrink_070_scale_matched",
        "residual_empirical_shrink_030_scale_matched",
        "residual_empirical_shrink_070_scale_matched",
        "semantic_shift_diagnostic",
    ]
    report: dict[str, Any] = {
        "schema": "clearvla-conditioned-source-design-probe-v1",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "toy_network_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "config": {
            "seed": args.seed,
            "samples": args.samples,
            "field_steps": args.field_steps,
            "fusion_steps": args.fusion_steps,
            "collapse_steps": args.collapse_steps,
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "quick": args.quick,
            "synthetic_target_residual_rms": residual_scale,
        },
        "parseval_frame": parseval_report,
        "source_covariance_geometry": {
            name: _covariance_metrics(covariance) for name, covariance in covariances.items()
        },
        "synthetic_residual_geometry": _covariance_metrics(empirical_covariance),
        "source_transport": _source_transport_probe(
            sampler,
            source_candidates,
            samples=args.samples,
            device=device,
        ),
    }
    report["flow_regression"] = _train_flat_fields(
        sampler,
        field_candidates,
        device=device,
        steps=args.field_steps,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )
    report["condition_reader"] = _condition_reader_experiment(
        sampler,
        device=device,
        steps=args.fusion_steps,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )
    report["learned_source_degeneracy"] = _learned_source_collapse_experiment(
        device=device,
        steps=args.collapse_steps,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["environment"]["peak_cuda_memory_mib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**20, 3
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    print(f"elapsed={report['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
