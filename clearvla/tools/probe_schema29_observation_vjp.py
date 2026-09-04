"""Attribute trained Schema29/30 observation gradients on real cached batches.

This is a read-only checkpoint probe, not a training entry point.  It loads a
complete mainline checkpoint through the strict validation-replay boundary,
reproduces the Schema29/30 detached pass-zero / formal pass-one training forward,
and never constructs an optimizer or calls ``backward``/``step``.

Three modes are intentionally separate:

``--scan-batches N``
    Compute only the total-loss VJP into every observation parameter for the
    first ``N`` deterministic real training batches.  This cheaply locates a
    high-Jacobian batch and reports the exact parameter tensors that own it.

``--batch-index I``
    Replay one indexed batch and partition selected observation-owner VJPs
    into the exact action, execution, future, scaffold, coarse-flow and
    raw-flow contributions.  The report retains both component-level and
    family-level gradient angles/cancellation and closes both partitions
    against the total loss.

``--batch-index I --activation-vjp``
    Replay one indexed batch once per requested scalar and trace its VJP across
    exact activation boundaries from target-DINO/G1/G2/G3 through the object
    binder and refined W semantic prediction.  Fresh deterministic forwards
    prevent one scalar's autograd traversal from contaminating another.

Only JSON scalar summaries, names, shapes, and fingerprints are written.  No
checkpoint, activation tensor, optimizer state, or gradient tensor is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from clearvla.mainline.checkpoint import build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.manifest import ARCHITECTURE_MANIFEST
from clearvla.mainline.model import grounding as grounding_module
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.types import PhysicalActionCondition
from clearvla.mainline.runtime.checkpoints import load_checkpoint_for_validation
from clearvla.mainline.runtime.identity import dataset_identity, language_identity
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training.losses import (
    LossLedger,
    compose_losses,
    sample_flow_matching,
)
from clearvla.mainline.training.optimizer import parameter_role
from clearvla.mainline.v120_core import flow_dino_evidence as flow_dino_module

REPORT_SCHEMA = "clearvla-schema29-checkpoint-observation-vjp-v5"
_OBSERVATION_PREFIX = "observation."
_DELTA_OUTPUT_NAMES = frozenset(
    {
        "observation.encoder.flow.delta_head.2.weight",
        "observation.encoder.flow.delta_head.2.bias",
    }
)
_SHARED_FLOW_ROWS: tuple[tuple[str, str], ...] = (
    ("flow_jepa_warp_loss", "flow_warp"),
    ("flow_jepa_cycle_loss", "flow_cycle"),
    ("flow_jepa_smoothness_loss", "flow_smoothness"),
    ("flow_jepa_uncertainty_nll", "flow_uncertainty"),
    ("flow_jepa_refinement_sequence_loss", "flow_refinement_sequence"),
)
_RAW_ONLY_FLOW_ROWS: tuple[tuple[str, str], ...] = (
    ("flow_jepa_identity_advantage_loss", "flow_identity_advantage"),
    ("flow_jepa_static_identity_loss", "flow_static_identity"),
)
_COMPONENT_ORDER = (
    "action_flow",
    "decoded_action",
    "gripper_trajectory",
    "motion",
    "smooth_delta",
    "physical_delta_consistency",
    "proposal",
    "execution_value",
    "future_semantic_common",
    "future_semantic_innovation",
    "future_transport_common",
    "future_transport_innovation",
    "future_covariance",
    "future_transition",
    "object_intent_scaffold",
    "coarse_flow_geometry",
    "raw_flow_geometry",
)
_FAMILY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "action": (
        "action_flow",
        "decoded_action",
        "gripper_trajectory",
        "motion",
        "smooth_delta",
        "physical_delta_consistency",
        "proposal",
    ),
    "execution": ("execution_value",),
    "future_dynamics_transition": (
        "future_semantic_common",
        "future_semantic_innovation",
        "future_transport_common",
        "future_transport_innovation",
        "future_covariance",
        "future_transition",
    ),
    "object_intent_scaffold": ("object_intent_scaffold",),
    "coarse_flow_geometry": ("coarse_flow_geometry",),
    "raw_flow_geometry": ("raw_flow_geometry",),
}
_COMPONENT_CHUNKS: tuple[tuple[str, ...], ...] = (
    ("action_flow", "decoded_action", "gripper_trajectory"),
    ("motion", "smooth_delta", "physical_delta_consistency", "proposal"),
    (
        "execution_value",
        "future_semantic_common",
        "future_semantic_innovation",
        "future_transition",
    ),
    (
        "future_transport_common",
        "future_transport_innovation",
        "future_covariance",
        "object_intent_scaffold",
    ),
    ("coarse_flow_geometry", "raw_flow_geometry"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan or attribute trained Schema29/30 observation VJPs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mainline/object_intent_dynamics_323.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    parser.add_argument("--t5-condition", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan-batches", type=int)
    mode.add_argument("--batch-index", type=int)
    parser.add_argument(
        "--scan-seeds",
        type=int,
        help=(
            "With --batch-index, scan this many consecutive probe seeds using "
            "only the total-loss observation VJP instead of decomposing losses"
        ),
    )
    parser.add_argument(
        "--seed-axis",
        choices=(
            "joint",
            "flow",
            "condition",
            "global",
            "mask",
            "global_without_mask",
        ),
        default="joint",
        help=(
            "Random axis varied by --scan-seeds. 'global' includes the "
            "observation mask; 'global_without_mask' pins that mask and "
            "varies the remaining global RNG consumers."
        ),
    )
    parser.add_argument(
        "--activation-vjp",
        action="store_true",
        help=(
            "With --batch-index, replace the parameter-loss decomposition by "
            "an activation-level VJP trace for future semantic common, future "
            "semantic innovation, and total loss."
        ),
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        help="Pin the observation structured-mask RNG for a full/activation VJP.",
    )
    parser.add_argument("--probe-seed", type=int, default=29001)
    parser.add_argument("--top-parameters", type=int, default=12)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-checkpoint-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _owned_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _autocast(
    device: torch.device,
    dtype: torch.dtype,
    *,
    cache_enabled: bool = True,
):
    enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=bool(cache_enabled),
    )


def _torch_seed(seed: int) -> None:
    """Seed only Torch; callers own Python/NumPy and RNG restoration."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@contextmanager
def _structured_mask_seed(
    model: ClearVLAMainlinePolicy,
    *,
    seed: int | None,
    device: torch.device,
):
    """Optionally decouple observation masks from all other global RNG use.

    The active encoder builds the current-context and future-target masks via
    the same ``_structured_mask`` method.  Each invocation receives a stable
    sub-seed inside a forked RNG scope, so the caller's model/dropout stream is
    unchanged.  No tensor, parameter, or method implementation is replaced
    outside this diagnostic context.
    """

    if seed is None:
        yield
        return
    encoder = model.observation.encoder
    original = encoder._structured_mask
    call_index = 0

    def seeded_mask(score: Tensor, *, stochastic: bool) -> Tensor:
        nonlocal call_index
        current_seed = int(seed) + 7919 * call_index
        call_index += 1
        with torch.random.fork_rng(devices=_cuda_devices(device)):
            _torch_seed(current_seed)
            return original(score, stochastic=stochastic)

    had_instance_value = "_structured_mask" in encoder.__dict__
    old_instance_value = encoder.__dict__.get("_structured_mask")
    object.__setattr__(encoder, "_structured_mask", seeded_mask)
    try:
        yield
    finally:
        if had_instance_value:
            object.__setattr__(encoder, "_structured_mask", old_instance_value)
        else:
            object.__delattr__(encoder, "_structured_mask")


def _axis_seeds(
    *,
    probe_seed: int,
    batch_index: int,
    seed_axis: str,
    seed_offset: int,
) -> dict[str, int | None]:
    """Resolve independent probe RNG owners without changing batch identity."""

    anchor = int(probe_seed) + 1009 * int(batch_index)
    offset = int(seed_offset)
    seeds: dict[str, int | None] = {
        "global": anchor,
        "flow": anchor + 1,
        "condition": anchor + 2,
        "mask": None,
    }
    if seed_axis == "joint":
        seeds.update(
            {
                "global": anchor + offset,
                "flow": anchor + offset + 1,
                "condition": anchor + offset + 2,
            }
        )
    elif seed_axis == "flow":
        seeds["flow"] = anchor + 1 + offset
    elif seed_axis == "condition":
        seeds["condition"] = anchor + 2 + offset
    elif seed_axis == "global":
        seeds["global"] = anchor + offset
    elif seed_axis == "mask":
        seeds["mask"] = anchor + 3 + offset
    elif seed_axis == "global_without_mask":
        seeds["global"] = anchor + offset
        seeds["mask"] = anchor + 3
    else:
        raise ValueError(f"unsupported seed axis: {seed_axis}")
    return seeds


def _overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    data = config.data
    optimizer = config.optimizer
    runtime = config.runtime
    for field_name, value in (
        ("raw_hdf5_root", args.data_root),
        ("decoded_cache", args.decoded_cache),
        ("dino_cache", args.dino_cache),
        ("t5_condition", args.t5_condition),
    ):
        if value is not None:
            data = replace(data, **{field_name: str(value)})
    if args.num_workers is not None:
        data = replace(data, num_workers=int(args.num_workers))
    if args.batch_size is not None:
        optimizer = replace(optimizer, batch_size=int(args.batch_size))
    if args.dtype is not None:
        runtime = replace(runtime, compute_dtype=str(args.dtype))
    result = replace(config, data=data, optimizer=optimizer, runtime=runtime)
    result.validate()
    return result


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        payload = tensor.view(torch.uint16).numpy().tobytes()
    else:
        payload = tensor.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _repository_state(expected_commit: str | None) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if expected_commit is not None and commit != str(expected_commit).strip():
        raise ValueError(f"source commit {commit} does not match expected {expected_commit}")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "worktree_dirty": bool(status.strip())}


def _batch_identity(batch: TrainingBatch) -> dict[str, object]:
    def values(value: Tensor | None) -> list[float | int] | None:
        if value is None:
            return None
        raw = value.detach().cpu().tolist()
        return [item for item in raw]

    return {
        "batch_size": int(batch.online.batch),
        "sample_indices": values(batch.audit.sample_index),
        "episode_indices": values(batch.audit.episode_index),
        "frame_progress": values(batch.audit.frame_progress),
    }


def _named_trainable(
    model: ClearVLAMainlinePolicy,
    *,
    prefix: str | None = None,
) -> tuple[tuple[str, nn.Parameter], ...]:
    rows = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (prefix is None or name.startswith(prefix))
    )
    if not rows:
        raise RuntimeError("the requested trainable parameter boundary is empty")
    ids = [id(parameter) for _, parameter in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("the trainable parameter boundary contains an alias")
    for name, _ in rows:
        parameter_role(name)
    return rows


def _owner_parameters(
    observation: tuple[tuple[str, nn.Parameter], ...],
) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
    predicates = {
        "coarse_delta_output": lambda name: name in _DELTA_OUTPUT_NAMES,
        "target_dino_key": lambda name: name.startswith(
            "observation.encoder.soft_address_compiler.target_dino_key."
        ),
        "source_dino_key": lambda name: name.startswith(
            "observation.encoder.soft_address_compiler.source_dino."
        ),
        "fine_dino_key": lambda name: name.startswith(
            "observation.encoder.soft_address_compiler.fine_dino_key."
        ),
        "target_raw_key": lambda name: name.startswith(
            "observation.encoder.soft_address_compiler.raw_key."
        ),
        "source_raw_key": lambda name: name.startswith(
            "observation.encoder.soft_address_compiler.source_raw_key."
        ),
        "address_slot_query": lambda name: (
            name == "observation.encoder.soft_address_compiler.slot_query"
        ),
        "raw_pyramid_stem": lambda name: name.startswith(
            "observation.encoder.raw_flow.pyramid.stem."
        ),
        "raw_pyramid_high": lambda name: name.startswith(
            "observation.encoder.raw_flow.pyramid.high."
        ),
        "raw_mid_refiner": lambda name: name.startswith("observation.encoder.raw_flow.mid."),
        "raw_high_refiner": lambda name: name.startswith("observation.encoder.raw_flow.high."),
    }
    owners = {
        owner: tuple(row for row in observation if predicate(row[0]))
        for owner, predicate in predicates.items()
    }
    missing = [name for name, rows in owners.items() if not rows]
    if missing:
        raise RuntimeError("observation VJP owner boundary is incomplete: " + ", ".join(missing))
    ids = [id(parameter) for rows in owners.values() for _, parameter in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("observation VJP owner boundaries overlap")
    return owners


def _gradient_l2(gradients: Iterable[Tensor | None]) -> float:
    rows = [
        gradient.detach().float().square().sum() for gradient in gradients if gradient is not None
    ]
    if not rows:
        return 0.0
    return float(torch.stack(rows).sum().sqrt())


def _gradient_stats(
    named: tuple[tuple[str, nn.Parameter], ...],
    gradients: tuple[Tensor | None, ...],
) -> dict[str, float | int]:
    if len(named) != len(gradients) or not named:
        raise ValueError("gradient statistics require aligned non-empty rows")
    elements = sum(int(parameter.numel()) for _, parameter in named)
    present = tuple(value for value in gradients if value is not None)
    square_sum = (
        torch.stack([value.detach().float().square().sum() for value in present]).sum()
        if present
        else torch.zeros((), dtype=torch.float32)
    )
    l2 = float(square_sum.sqrt())
    return {
        "parameter_tensors": len(named),
        "parameter_elements": elements,
        "gradient_present_tensors": len(present),
        "gradient_l2": l2,
        "gradient_rms_over_parameter_elements": float(
            (square_sum / float(max(elements, 1))).sqrt()
        ),
    }


def _delta_channel_stats(
    named: tuple[tuple[str, nn.Parameter], ...],
    gradients: tuple[Tensor | None, ...],
) -> dict[str, float]:
    flow_rows: list[Tensor] = []
    information_rows: list[Tensor] = []
    for (name, _), gradient in zip(named, gradients, strict=True):
        if name not in _DELTA_OUTPUT_NAMES or gradient is None:
            continue
        if int(gradient.shape[0]) != 6:
            raise RuntimeError("SEA-RAFT delta output no longer owns six channels")
        flow_rows.append(gradient[:2].detach().float().square().sum())
        information_rows.append(gradient[2:].detach().float().square().sum())
    if not flow_rows or not information_rows:
        return {"flow_channels_l2": 0.0, "information_channels_l2": 0.0}
    return {
        "flow_channels_l2": float(torch.stack(flow_rows).sum().sqrt()),
        "information_channels_l2": float(torch.stack(information_rows).sum().sqrt()),
    }


def _per_parameter_stats(
    named: tuple[tuple[str, nn.Parameter], ...],
    gradients: tuple[Tensor | None, ...],
    *,
    limit: int,
) -> list[dict[str, object]]:
    rows = []
    for (name, parameter), gradient in zip(named, gradients, strict=True):
        if gradient is None:
            l2 = 0.0
            rms = 0.0
            maximum = 0.0
        else:
            value = gradient.detach().float()
            l2 = float(value.square().sum().sqrt())
            rms = float(value.square().mean().sqrt())
            maximum = float(value.abs().amax())
        rows.append(
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "elements": int(parameter.numel()),
                "gradient_l2": l2,
                "gradient_rms": rms,
                "gradient_max_abs": maximum,
            }
        )
    rows.sort(key=lambda row: float(row["gradient_l2"]), reverse=True)
    return rows[: max(int(limit), 1)]


def _cuda_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else int(device.index)]


class _ActivationCapture:
    """Capture exact live tensors without changing the active model source.

    Module hooks cover parameterized boundaries.  Three narrowly matched
    functional calls cover the pure FP32 contractions that otherwise have no
    public return value.  All patches are process-local and restored in
    ``__exit__``; the probe never stores an activation tensor in its report.
    """

    _CONTENT_EINSUM = "bcijmp,bcpr->bcijmr"
    _VARIANCE_EINSUM = "bcijmp,bcijmpd->bcijmd"
    _CANONICAL_EINSUM = "bcijmk,bcijmkr->bcijmr"

    def __init__(self, model: ClearVLAMainlinePolicy) -> None:
        self.model = model
        self.values: dict[str, Tensor] = {}
        self.aliases: dict[str, str] = {}
        self._observation_open = True
        self._einsum_rows: dict[str, list[Tensor]] = {
            self._CONTENT_EINSUM: [],
            self._VARIANCE_EINSUM: [],
            self._CANONICAL_EINSUM: [],
        }
        self._std_rows: list[tuple[Tensor, Tensor, float]] = []
        self._masked_rows: list[Tensor] = []
        self._target_key_rows: list[Tensor] = []
        self._rectifier_rows: list[Tensor] = []
        self._candidate_rows: list[Tensor] = []
        self._fine_candidate_rows: list[dict[str, Tensor]] = []
        self._delta_rows: list[tuple[Tensor, Tensor]] = []
        self._field_rows: list[tuple[Tensor, Tensor]] = []
        self._handles: list[Any] = []
        self._original_einsum: Any = None
        self._original_std: Any = None
        self._original_masked: Any = None
        self._dynamics_had_instance_field = False
        self._dynamics_instance_field: Any = None
        self._compiler_had_instance_sampler = False
        self._compiler_instance_sampler: Any = None

    @staticmethod
    def _tensor_output(result: object, *, name: str) -> Tensor:
        if not isinstance(result, Tensor):
            raise RuntimeError(f"activation hook {name} did not return one tensor")
        return result

    def __enter__(self) -> "_ActivationCapture":
        encoder = self.model.observation.encoder
        organizer = encoder.progressive_grounding_address
        if organizer is None or organizer.g2_typed_rectifier is None:
            raise RuntimeError("activation VJP requires the active typed G2 organizer")

        def target_hook(
            _module: nn.Module,
            _args: tuple[object, ...],
            result: object,
        ) -> None:
            if self._observation_open:
                self._target_key_rows.append(self._tensor_output(result, name="target_dino_key"))

        def rectifier_hook(
            _module: nn.Module,
            _args: tuple[object, ...],
            result: object,
        ) -> None:
            if self._observation_open:
                self._rectifier_rows.append(self._tensor_output(result, name="g2_typed_rectifier"))

        def candidate_hook(
            _module: nn.Module,
            _args: tuple[object, ...],
            result: object,
        ) -> None:
            if self._observation_open:
                self._candidate_rows.append(
                    self._tensor_output(result, name="grounder_candidate_norm")
                )

        def delta_hook(
            _module: nn.Module,
            args: tuple[object, ...],
            result: object,
        ) -> None:
            if not args or not isinstance(args[0], Tensor):
                raise RuntimeError("delta-head activation hook lost its input")
            if self._observation_open:
                self._delta_rows.append(
                    (
                        args[0],
                        self._tensor_output(result, name="w_delta_head"),
                    )
                )

        self._handles.extend(
            (
                encoder.soft_address_compiler.target_dino_key.register_forward_hook(target_hook),
                organizer.g2_typed_rectifier.register_forward_hook(rectifier_hook),
                self.model.grounding.grounder.candidate_norm.register_forward_hook(candidate_hook),
                self.model.world.dynamics.delta_head.register_forward_hook(delta_hook),
            )
        )

        self._original_einsum = torch.einsum

        def captured_einsum(equation: object, *operands: object) -> Tensor:
            result = self._original_einsum(equation, *operands)
            if (
                self._observation_open
                and isinstance(equation, str)
                and equation in self._einsum_rows
            ):
                self._einsum_rows[equation].append(result)
            return result

        torch.einsum = captured_einsum  # type: ignore[assignment]

        self._original_std = flow_dino_module._zero_preserving_variance_std

        def captured_std(variance: Tensor, *, epsilon: float) -> Tensor:
            result = self._original_std(variance, epsilon=epsilon)
            if self._observation_open:
                self._std_rows.append((variance, result, float(epsilon)))
            return result

        flow_dino_module._zero_preserving_variance_std = captured_std

        self._original_masked = grounding_module._masked_log_softmax

        def captured_masked(
            log_measure: Tensor,
            support: Tensor,
            *,
            dim: int,
        ) -> tuple[Tensor, Tensor, Tensor]:
            result = self._original_masked(log_measure, support, dim=dim)
            if self._observation_open and result[0].ndim == 3 and int(dim) in (-1, 2):
                self._masked_rows.append(result[0])
            return result

        grounding_module._masked_log_softmax = captured_masked

        compiler = encoder.soft_address_compiler
        original_sampler = compiler.progressive_fine_candidates
        self._compiler_had_instance_sampler = "progressive_fine_candidates" in compiler.__dict__
        self._compiler_instance_sampler = compiler.__dict__.get("progressive_fine_candidates")

        def captured_sampler(
            bank: object,
            *,
            centers: Tensor,
            support: Tensor,
            variance: Tensor,
            aligned_keys: Tensor,
            collect_diagnostics: bool = True,
        ) -> object:
            result = original_sampler(
                bank,
                centers=centers,
                support=support,
                variance=variance,
                aligned_keys=aligned_keys,
                collect_diagnostics=collect_diagnostics,
            )
            if self._observation_open:
                current_coordinates = getattr(result, "current_coordinates", None)
                if not isinstance(current_coordinates, Tensor):
                    raise RuntimeError("G2 candidate sampler lost current coordinates")
                self._fine_candidate_rows.append(
                    {
                        "centers": centers,
                        "support": support,
                        "variance": variance,
                        "aligned_keys": aligned_keys,
                        "coordinates": current_coordinates,
                    }
                )
            return result

        object.__setattr__(compiler, "progressive_fine_candidates", captured_sampler)

        dynamics = self.model.world.dynamics
        original_field = dynamics._field_with_diagnostics
        self._dynamics_had_instance_field = "_field_with_diagnostics" in dynamics.__dict__
        self._dynamics_instance_field = dynamics.__dict__.get("_field_with_diagnostics")

        def captured_field(*args: object, **kwargs: object) -> object:
            typed_common = kwargs.get("typed_common")
            typed_interval = kwargs.get("typed_interval_innovation")
            if not isinstance(typed_common, Tensor) or not isinstance(typed_interval, Tensor):
                raise RuntimeError("W field capture lost its typed owner states")
            if self._observation_open:
                self._field_rows.append((typed_common, typed_interval))
            return original_field(*args, **kwargs)

        object.__setattr__(dynamics, "_field_with_diagnostics", captured_field)
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        if self._original_einsum is not None:
            torch.einsum = self._original_einsum
        if self._original_std is not None:
            flow_dino_module._zero_preserving_variance_std = self._original_std
        if self._original_masked is not None:
            grounding_module._masked_log_softmax = self._original_masked
        compiler = self.model.observation.encoder.soft_address_compiler
        if self._compiler_had_instance_sampler:
            object.__setattr__(
                compiler,
                "progressive_fine_candidates",
                self._compiler_instance_sampler,
            )
        elif "progressive_fine_candidates" in compiler.__dict__:
            object.__delattr__(compiler, "progressive_fine_candidates")
        dynamics = self.model.world.dynamics
        if self._dynamics_had_instance_field:
            object.__setattr__(
                dynamics,
                "_field_with_diagnostics",
                self._dynamics_instance_field,
            )
        elif "_field_with_diagnostics" in dynamics.__dict__:
            object.__delattr__(dynamics, "_field_with_diagnostics")

    def _put(self, name: str, value: Tensor | None) -> None:
        if value is None:
            raise RuntimeError(f"activation boundary {name} is absent")
        if not isinstance(value, Tensor):
            raise TypeError(f"activation boundary {name} is not a tensor")
        self.values[name] = value

    def seal_online(self, training_state: object) -> None:
        """Name every observation/G/grounder tensor before Teacher executes."""

        observation = getattr(training_state, "observation", None)
        top = getattr(training_state, "top", None)
        if observation is None or top is None:
            raise RuntimeError("activation capture lost the online training state")
        state = observation.progressive_state
        facts = top.facts
        content_rows = self._einsum_rows[self._CONTENT_EINSUM]
        variance_rows = self._einsum_rows[self._VARIANCE_EINSUM]
        canonical_rows = self._einsum_rows[self._CANONICAL_EINSUM]
        expected_masked = int(self.model.grounding.grounder.iterations) + 5
        if len(self._target_key_rows) != 1:
            raise RuntimeError("target-DINO activation call count changed")
        if len(content_rows) != 2 or len(variance_rows) != 2:
            raise RuntimeError("coarse/G1 functional activation call count changed")
        if len(canonical_rows) != 4:
            raise RuntimeError("G2/G3 canonical contraction call count changed")
        if len(self._std_rows) != 3:
            raise RuntimeError("variance-to-std activation call count changed")
        if len(self._rectifier_rows) != 1 or len(self._candidate_rows) != 1:
            raise RuntimeError("G2/grounder module activation call count changed")
        if len(self._fine_candidate_rows) != 1:
            raise RuntimeError("G2 fine candidate sampler call count changed")
        if len(self._masked_rows) != expected_masked:
            raise RuntimeError(
                "grounder read activation call count changed: "
                f"expected {expected_masked}, got {len(self._masked_rows)}"
            )

        iterations = int(self.model.grounding.grounder.iterations)
        self._put("target_key", self._target_key_rows[0])
        self._put("g1_coarse_logits", state.coarse_logits)
        self._put("g1_coarse_probability", state.coarse_probability)
        self._put("address_coarse_content", content_rows[0])
        self._put("g1_aligned_keys_fp32", content_rows[1])
        self._put("address_coarse_variance_fp32", variance_rows[0])
        self._put("g1_aligned_variance_fp32", variance_rows[1])
        self._put("g1_aligned_keys", state.aligned_keys)
        self._put("g1_aligned_centers", state.aligned_centers)
        self._put("g1_aligned_variance", state.aligned_variance)
        self._put("g2_aligned_std", self._std_rows[1][1])
        self._put("g2_candidate_geometry_std", self._std_rows[2][1])
        self._put("g2_rectifier_output", self._rectifier_rows[0])
        sampler = self._fine_candidate_rows[0]
        self._put("g2_sampler_centers", sampler["centers"])
        self._put("g2_sampler_support", sampler["support"])
        self._put("g2_sampler_variance", sampler["variance"])
        self._put("g2_dynamic_coordinates", sampler["coordinates"])
        self._put("g2_fine_probability", state.fine_probability)
        self._put("g2_semantic_probability", state.g2_semantic_probability)
        self._put(
            "g2_semantic_slot_log_probability",
            state.g2_semantic_slot_log_probability,
        )
        self._put("g2_dynamic_semantic_keys", state.dynamic_semantic_keys)
        self._put("g2_dynamic_appearance_keys", state.dynamic_appearance_keys)
        self._put("g2_dynamic_geometry_keys", state.dynamic_geometry_keys)
        self._put("g2_dynamic_detail_values", state.dynamic_fine_values)
        self._put("g2_dynamic_literal_rgb", state.dynamic_literal_rgb)
        self._put("g2_rectified_keys_fp32", canonical_rows[0])
        self._put("g3_canonical_semantic_fp32", canonical_rows[1])
        self._put("g3_canonical_semantic", state.canonical_semantic_keys)
        self._put(
            "g3_semantic_slot_weights",
            state.canonical_semantic_slot_weights,
        )
        self._put("local_fact_semantic_slots", observation.local_facts.semantic_slots)
        self._put(
            "local_fact_semantic_owner_log",
            observation.local_facts.semantic_owner_log_probs,
        )
        self._put("grounder_candidate_tokens", self._candidate_rows[0])
        self._put("grounder_corrected_read", self._masked_rows[iterations + 1])
        self._put("grounder_semantic_read", self._masked_rows[iterations + 2])
        self._put("object_fact_semantic", facts.semantic)
        self._put("object_fact_validity", facts.validity)
        self._observation_open = False

    def seal_formal(self, formal_cache: object) -> None:
        """Name the refined W and prediction tensors after cache1 is built."""

        # ``seal_online`` closes observation-side hooks before the detached
        # estimator pass and the attached formal pass.  The estimator is
        # intentionally not part of the live activation trace, so only the
        # single formal W materialization and its two semantic delta-head
        # calls should be present here.
        if len(self._field_rows) != 1 or len(self._delta_rows) != 2:
            raise RuntimeError(
                "initial/refined W activation call count changed: "
                f"field={len(self._field_rows)} delta={len(self._delta_rows)}"
            )
        typed_common, typed_interval = self._field_rows[-1]
        semantic_common_state, semantic_common = self._delta_rows[-2]
        semantic_interval_state, semantic_interval = self._delta_rows[-1]
        top = getattr(formal_cache, "top", None)
        if top is None:
            raise RuntimeError("activation capture lost the formal cache")
        prediction = top.predicted_dynamics
        self._put("w_typed_common_state", typed_common)
        self._put("w_typed_interval_state", typed_interval)
        self._put("w_semantic_common_state", semantic_common_state)
        self._put("w_semantic_interval_state", semantic_interval_state)
        self._put("w_semantic_common", semantic_common)
        self._put("w_semantic_interval", semantic_interval)
        self._put("prediction_semantic_delta", prediction.semantic_delta)

        first_by_id: dict[int, str] = {}
        for name, value in self.values.items():
            identity = id(value)
            if identity in first_by_id:
                self.aliases[name] = first_by_id[identity]
            else:
                first_by_id[identity] = name

    @property
    def std_epsilon(self) -> float:
        if len(self._std_rows) != 3:
            raise RuntimeError("G2 variance epsilon is unavailable")
        return float(self._std_rows[1][2])


def _activation_forward_stats(value: Tensor) -> dict[str, object]:
    detached = value.detach().float()
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
        "requires_grad": bool(value.requires_grad),
        "elements": int(value.numel()),
        "rms": float(detached.square().mean().sqrt()),
        "mean": float(detached.mean()),
        "min": float(detached.amin()),
        "max": float(detached.amax()),
        "max_abs": float(detached.abs().amax()),
        "finite": bool(torch.isfinite(detached).all()),
    }


def _activation_gradient_stats(
    value: Tensor,
    gradient: Tensor | None,
) -> dict[str, object]:
    if gradient is None:
        return {
            "present": False,
            "l2": 0.0,
            "rms": 0.0,
            "max_abs": 0.0,
            "finite": True,
        }
    gradient_f = gradient.detach().float()
    if tuple(gradient_f.shape) != tuple(value.shape):
        raise RuntimeError("activation gradient shape does not match its tensor")
    return {
        "present": True,
        "l2": float(gradient_f.square().sum().sqrt()),
        "rms": float(gradient_f.square().mean().sqrt()),
        "max_abs": float(gradient_f.abs().amax()),
        "finite": bool(torch.isfinite(gradient_f).all()),
    }


_ACTIVATION_EDGES: tuple[tuple[str, str], ...] = (
    ("target_key", "g1_coarse_probability"),
    ("target_key", "g1_aligned_centers"),
    ("target_key", "g2_dynamic_coordinates"),
    ("target_key", "address_coarse_content"),
    ("target_key", "g1_aligned_keys_fp32"),
    ("target_key", "g1_aligned_variance_fp32"),
    ("g1_coarse_probability", "g1_aligned_centers"),
    ("g1_coarse_probability", "g1_aligned_keys_fp32"),
    ("g1_coarse_probability", "g1_aligned_variance_fp32"),
    ("g1_aligned_centers", "g2_sampler_centers"),
    ("g1_aligned_variance", "g2_aligned_std"),
    ("g1_aligned_variance", "g2_candidate_geometry_std"),
    ("g1_aligned_variance", "g2_sampler_centers"),
    ("g1_aligned_variance", "g2_sampler_support"),
    ("g2_aligned_std", "g2_rectifier_output"),
    ("g2_rectifier_output", "g2_sampler_centers"),
    ("g2_rectifier_output", "g2_sampler_support"),
    ("g2_sampler_centers", "g2_dynamic_coordinates"),
    ("g2_sampler_support", "g2_dynamic_coordinates"),
    ("g2_dynamic_coordinates", "g2_dynamic_semantic_keys"),
    ("g2_dynamic_coordinates", "g2_dynamic_appearance_keys"),
    ("g2_dynamic_coordinates", "g2_dynamic_geometry_keys"),
    ("g2_dynamic_coordinates", "g2_dynamic_detail_values"),
    ("g2_dynamic_coordinates", "g2_dynamic_literal_rgb"),
    ("g2_rectifier_output", "g2_semantic_probability"),
    ("g2_semantic_probability", "g3_canonical_semantic_fp32"),
    ("g2_dynamic_semantic_keys", "g3_canonical_semantic_fp32"),
    ("g3_canonical_semantic_fp32", "local_fact_semantic_slots"),
    ("local_fact_semantic_slots", "grounder_candidate_tokens"),
    ("grounder_candidate_tokens", "grounder_semantic_read"),
    ("grounder_semantic_read", "object_fact_semantic"),
    ("local_fact_semantic_slots", "object_fact_semantic"),
    ("local_fact_semantic_owner_log", "grounder_semantic_read"),
    ("object_fact_semantic", "w_typed_common_state"),
    ("object_fact_semantic", "w_typed_interval_state"),
    ("w_semantic_common_state", "prediction_semantic_delta"),
    ("w_semantic_interval_state", "prediction_semantic_delta"),
)


def _activation_vjp_report(
    scalar: Tensor,
    capture: _ActivationCapture,
) -> dict[str, object]:
    unique_names: list[str] = []
    unique_values: list[Tensor] = []
    index_by_id: dict[int, int] = {}
    index_by_name: dict[str, int] = {}
    for name, value in capture.values.items():
        identity = id(value)
        if identity not in index_by_id:
            index_by_id[identity] = len(unique_values)
            unique_names.append(name)
            unique_values.append(value)
        index_by_name[name] = index_by_id[identity]
    differentiable_indices = [
        index for index, value in enumerate(unique_values) if value.requires_grad
    ]
    differentiable = tuple(unique_values[index] for index in differentiable_indices)
    gradients_raw = torch.autograd.grad(
        scalar,
        differentiable,
        retain_graph=True,
        allow_unused=True,
    )
    gradients: list[Tensor | None] = [None] * len(unique_values)
    for index, gradient in zip(
        differentiable_indices,
        gradients_raw,
        strict=True,
    ):
        gradients[index] = gradient

    rows: dict[str, object] = {}
    for name, value in capture.values.items():
        gradient = gradients[index_by_name[name]]
        rows[name] = {
            "forward": _activation_forward_stats(value),
            "gradient": _activation_gradient_stats(value, gradient),
            "alias_of": capture.aliases.get(name),
        }

    edges: dict[str, object] = {}
    for parent_name, child_name in _ACTIVATION_EDGES:
        parent = capture.values[parent_name]
        child = capture.values[child_name]
        child_gradient = gradients[index_by_name[child_name]]
        key = f"{parent_name}__to__{child_name}"
        if child_gradient is None or not parent.requires_grad or not child.requires_grad:
            edges[key] = {
                "present": False,
                "reason": "missing child gradient or differentiable boundary",
            }
            continue
        contribution = torch.autograd.grad(
            child,
            parent,
            grad_outputs=child_gradient,
            retain_graph=True,
            allow_unused=True,
        )[0]
        contribution_stats = _activation_gradient_stats(parent, contribution)
        child_l2 = float(child_gradient.detach().float().square().sum().sqrt())
        contribution_l2 = float(contribution_stats["l2"])
        edges[key] = {
            "present": contribution is not None,
            "child_gradient_l2": child_l2,
            "parent_path_contribution": contribution_stats,
            "local_vjp_l2_gain": (None if child_l2 <= 0.0 else contribution_l2 / child_l2),
        }
    return {
        "scalar_value": float(scalar.detach().float()),
        "activations": rows,
        "local_edges": edges,
        "aliases": dict(capture.aliases),
        "variance_std_epsilon": capture.std_epsilon,
        "variance_std_zero_slope": 1.0 / (2.0 * capture.std_epsilon),
        "variance_std_branches": _variance_std_branch_stats(capture),
    }


def _variance_std_branch_stats(
    capture: _ActivationCapture,
) -> dict[str, object]:
    names = (
        "address_coarse",
        "g2_aligned",
        "g2_candidate_geometry",
    )
    if len(capture._std_rows) != len(names):
        raise RuntimeError("variance-to-std branch count changed")
    rows: dict[str, object] = {}
    for name, (variance, std, epsilon) in zip(
        names,
        capture._std_rows,
        strict=True,
    ):
        variance_f = variance.detach().float().clamp_min(0.0)
        gain = 0.5 / torch.sqrt(variance_f + variance_f.new_tensor(float(epsilon)).square())
        flat_gain = gain.reshape(-1)
        quantiles = torch.quantile(
            flat_gain,
            flat_gain.new_tensor((0.50, 0.90, 0.99)),
        )
        epsilon_square = float(epsilon) ** 2
        rows[name] = {
            "epsilon": float(epsilon),
            "epsilon_square": epsilon_square,
            "variance": _activation_forward_stats(variance),
            "std": _activation_forward_stats(std),
            "variance_fraction_below_epsilon_square": float(
                (variance_f < epsilon_square).float().mean()
            ),
            "variance_fraction_below_1e_8": float((variance_f < 1.0e-8).float().mean()),
            "analytic_gain": {
                "mean": float(gain.mean()),
                "rms": float(gain.square().mean().sqrt()),
                "max": float(gain.amax()),
                "p50": float(quantiles[0]),
                "p90": float(quantiles[1]),
                "p99": float(quantiles[2]),
            },
        }
    return rows


def _activation_batch(
    *,
    model: ClearVLAMainlinePolicy,
    config: ExperimentConfig,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    probe_seed: int,
    batch_index: int,
    mask_seed: int | None,
) -> dict[str, object]:
    """Trace the three useful scalar cotangents on fresh identical forwards."""

    seeds = _axis_seeds(
        probe_seed=probe_seed,
        batch_index=batch_index,
        seed_axis="joint",
        seed_offset=0,
    )
    global_seed = int(seeds["global"])
    scalar_names = (
        "future_semantic_common",
        "future_semantic_innovation",
        "total",
    )
    scalar_rows: dict[str, object] = {}
    reference_metadata: dict[str, object] | None = None
    reference_values: dict[str, float] | None = None
    for scalar_name in scalar_names:
        _seed(global_seed)
        capture = _ActivationCapture(model)
        with capture:
            ledger, _observation, _raw_losses, metadata = _formal_forward(
                model=model,
                config=config,
                batch=batch,
                device=device,
                dtype=dtype,
                flow_generator=_owned_generator(device, int(seeds["flow"])),
                condition_generator=_owned_generator(
                    device,
                    int(seeds["condition"]),
                ),
                mask_seed=mask_seed,
                activation_capture=capture,
            )
            scalar_map: dict[str, Tensor] = {
                "future_semantic_common": ledger.terms["future_semantic_common"],
                "future_semantic_innovation": ledger.terms["future_semantic_innovation"],
                "total": ledger.total,
            }
            scalar = scalar_map.get(scalar_name)
            if not isinstance(scalar, Tensor) or scalar.ndim != 0:
                raise RuntimeError(f"activation scalar {scalar_name!r} is unavailable")
            if not scalar.requires_grad:
                raise RuntimeError(f"activation scalar {scalar_name!r} is detached")
            scalar_rows[scalar_name] = _activation_vjp_report(scalar, capture)
            current_values = {
                name: float(value.detach().float()) for name, value in scalar_map.items()
            }
        if reference_metadata is None:
            reference_metadata = metadata
            reference_values = current_values
        else:
            assert reference_values is not None
            if metadata["noisy_physical_sha256"] != reference_metadata["noisy_physical_sha256"]:
                raise RuntimeError("activation fresh forwards changed flow noise")
            if metadata["context_mask_sha256"] != reference_metadata["context_mask_sha256"]:
                raise RuntimeError("activation fresh forwards changed context mask")
            for name, value in current_values.items():
                if abs(value - reference_values[name]) > 5e-7 * max(
                    abs(reference_values[name]),
                    1.0,
                ):
                    raise RuntimeError(f"activation fresh forwards changed loss scalar {name!r}")
        del ledger, _observation, _raw_losses, capture
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if reference_metadata is None or reference_values is None:
        raise RuntimeError("activation VJP did not execute a fresh forward")
    return {
        "batch_index": int(batch_index),
        "probe_seed": global_seed,
        "random_seeds": seeds,
        "mask_seed": mask_seed,
        "batch": _batch_identity(batch),
        "forward": reference_metadata,
        "loss_values": reference_values,
        "scalars": scalar_rows,
    }


def _formal_forward(
    *,
    model: ClearVLAMainlinePolicy,
    config: ExperimentConfig,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    flow_generator: torch.Generator,
    condition_generator: torch.Generator,
    mask_seed: int | None = None,
    activation_capture: _ActivationCapture | None = None,
) -> tuple[LossLedger, object, dict[str, Tensor], dict[str, object]]:
    """Reproduce the exact detached-estimator/formal-pass training graph."""

    captured_raw_losses: list[dict[str, Tensor]] = []

    def capture_raw_losses(
        _module: nn.Module,
        _args: tuple[object, ...],
        result: object,
    ) -> None:
        if not isinstance(result, tuple) or len(result) != 3 or not isinstance(result[1], Mapping):
            raise RuntimeError("raw-flow hook lost its loss dictionary")
        captured_raw_losses.append(
            {str(name): value for name, value in result[1].items() if isinstance(value, Tensor)}
        )

    raw_hook = model.observation.encoder.raw_flow.register_forward_hook(capture_raw_losses)
    try:
        with _autocast(device, dtype):
            with _structured_mask_seed(model, seed=mask_seed, device=device):
                cache0, training_state, _ = model.encode_online(
                    batch.online,
                    training_mask=True,
                    collect_diagnostics=False,
                    condition_generator=condition_generator,
                )
            if activation_capture is not None:
                activation_capture.seal_online(training_state)
            top_targets, _ = model.build_training_targets(
                training_state,
                batch.future,
                collect_diagnostics=False,
            )
            flow_state = sample_flow_matching(
                batch.action_target.normalized,
                action_state=batch.online.history.action_state,
                codec_gripper_boundary=batch.online.history.codec_gripper_boundary,
                codec=model.outlet_adapter,
                distribution=config.bottom.flow_time_distribution,
                generator=flow_generator,
            )
            with torch.random.fork_rng(devices=_cuda_devices(device)):
                with torch.no_grad():
                    with _autocast(device, dtype, cache_enabled=False):
                        pass0 = model.velocity(
                            cache0,
                            noisy_action_field=flow_state.noisy_physical,
                            time=flow_state.time,
                            require_execution_supervision=False,
                            collect_diagnostics=False,
                        )
                        remaining = (
                            1.0 - flow_state.time.to(dtype=flow_state.noisy_physical.dtype)
                        )[:, None, None]
                        pass0_clean_physical = flow_state.noisy_physical + remaining * (
                            pass0.bottom.physical_velocity.to(dtype=flow_state.noisy_physical.dtype)
                        )
                        pass0_action = model.outlet_adapter.decode(
                            pass0_clean_physical,
                            cache0.history.action_state,
                            codec_gripper_boundary=cache0.history.codec_gripper_boundary,
                        ).detach()
                        pass0_condition = PhysicalActionCondition.from_horizon_action(
                            pass0_action,
                            cache0.history.action_state.detach(),
                        )
                        del pass0, pass0_clean_physical
            refined_top, _ = model.world.refine_deployment_world(
                cache0.top,
                action_condition=pass0_condition,
                collect_diagnostics=False,
            )
            formal_cache = replace(cache0, top=refined_top)
            output = model.velocity(
                formal_cache,
                noisy_action_field=flow_state.noisy_physical,
                time=flow_state.time,
                require_execution_supervision=True,
                collect_diagnostics=False,
            )
            ledger = compose_losses(
                config,
                policy_output=output,
                action_target=batch.action_target,
                history=batch.online.history,
                flow_state=flow_state,
                observation=training_state.observation,
                top_targets=top_targets,
                predicted_dynamics=formal_cache.top.predicted_dynamics,
                action_codec=model.outlet_adapter,
                collect_diagnostics=False,
            )
            if activation_capture is not None:
                activation_capture.seal_formal(formal_cache)
    finally:
        raw_hook.remove()
    if len(captured_raw_losses) != 1:
        raise RuntimeError("formal observation forward must execute raw flow exactly once")
    metadata = {
        "flow_time_mean": float(flow_state.time.detach().float().mean()),
        "flow_time_min": float(flow_state.time.detach().float().amin()),
        "flow_time_max": float(flow_state.time.detach().float().amax()),
        "noisy_physical_rms": _rms(flow_state.noisy_physical),
        "noisy_physical_sha256": _tensor_sha256(flow_state.noisy_physical),
        "pass0_action_rms": _rms(pass0_action),
        "context_mask_sha256": _tensor_sha256(training_state.observation.grounding.context_mask),
        "context_mask_fraction": float(
            training_state.observation.grounding.context_mask.detach().float().mean()
        ),
        "formal_w_semantic_rms": _rms(formal_cache.top.predicted_dynamics.semantic_delta),
        "formal_w_transport_rms": _rms(formal_cache.top.predicted_dynamics.transport_mean),
    }
    return (
        ledger,
        training_state.observation,
        captured_raw_losses[0],
        metadata,
    )


def _loss_components(
    config: ExperimentConfig,
    ledger: LossLedger,
    observation: object,
    raw_losses: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, object]]:
    native = getattr(observation, "native_flow_losses", None)
    if not isinstance(native, Mapping):
        raise RuntimeError("checkpoint probe requires the native V120 flow ledger")
    objective = config.objectives
    coarse = ledger.total.new_zeros(())
    raw = ledger.total.new_zeros(())
    component_rows: dict[str, dict[str, float]] = {}
    for native_name, objective_name in _SHARED_FLOW_ROWS:
        combined_value = native.get(native_name)
        raw_value = raw_losses.get(native_name)
        if not isinstance(combined_value, Tensor) or not isinstance(raw_value, Tensor):
            raise RuntimeError(f"missing shared flow component {native_name!r}")
        coarse_value = 2.0 * combined_value - raw_value
        weight = float(getattr(objective, objective_name))
        coarse_contribution = 0.5 * weight * coarse_value
        raw_contribution = 0.5 * weight * raw_value
        coarse = coarse + coarse_contribution
        raw = raw + raw_contribution
        component_rows[native_name] = {
            "coarse_raw_loss": float(coarse_value.detach().float()),
            "raw_raw_loss": float(raw_value.detach().float()),
            "coarse_weighted_contribution": float(coarse_contribution.detach().float()),
            "raw_weighted_contribution": float(raw_contribution.detach().float()),
        }
    for native_name, objective_name in _RAW_ONLY_FLOW_ROWS:
        raw_value = raw_losses.get(native_name)
        if not isinstance(raw_value, Tensor):
            raise RuntimeError(f"missing raw-only flow component {native_name!r}")
        contribution = float(getattr(objective, objective_name)) * raw_value
        raw = raw + contribution
        component_rows[native_name] = {
            "coarse_raw_loss": 0.0,
            "raw_raw_loss": float(raw_value.detach().float()),
            "coarse_weighted_contribution": 0.0,
            "raw_weighted_contribution": float(contribution.detach().float()),
        }
    future_names = (
        "future_semantic_common",
        "future_semantic_innovation",
        "future_transport_common",
        "future_transport_innovation",
        "future_covariance",
    )
    missing_future = [name for name in future_names if name not in ledger.terms]
    if missing_future:
        raise RuntimeError(
            "checkpoint probe lost future component terms: " + ", ".join(missing_future)
        )
    future_weight = float(objective.future_dynamics)
    components = {
        "action_flow": ledger.contributions["action_flow"],
        "decoded_action": ledger.contributions["decoded_action"],
        "gripper_trajectory": ledger.contributions["gripper_trajectory"],
        "motion": ledger.contributions["motion"],
        "smooth_delta": ledger.contributions["smooth_delta"],
        "physical_delta_consistency": ledger.contributions["physical_delta_consistency"],
        "proposal": ledger.contributions["proposal"],
        "execution_value": ledger.contributions["execution_value"],
        "future_semantic_common": (
            future_weight * 0.55 * 0.50 * ledger.terms["future_semantic_common"]
        ),
        "future_semantic_innovation": (
            future_weight * 0.55 * 0.50 * ledger.terms["future_semantic_innovation"]
        ),
        "future_transport_common": (
            future_weight * 0.15 * 0.50 * ledger.terms["future_transport_common"]
        ),
        "future_transport_innovation": (
            future_weight * 0.15 * 0.50 * ledger.terms["future_transport_innovation"]
        ),
        "future_covariance": (future_weight * 0.05 * ledger.terms["future_covariance"]),
        "future_transition": ledger.contributions["future_transition"],
        "object_intent_scaffold": sum(
            (
                ledger.contributions[name]
                for name in (
                    "object_reconstruction",
                    "intent_online",
                    "intent_recognizer",
                    "coarse_action",
                )
            ),
            start=ledger.total.new_zeros(()),
        ),
        "coarse_flow_geometry": coarse,
        "raw_flow_geometry": raw,
    }
    if set(components) != set(_COMPONENT_ORDER):
        raise RuntimeError("observation VJP component inventory changed")
    component_sum = sum(
        (components[name] for name in _COMPONENT_ORDER),
        start=ledger.total.new_zeros(()),
    )
    family_members = tuple(name for names in _FAMILY_COMPONENTS.values() for name in names)
    if len(family_members) != len(set(family_members)) or set(family_members) != set(
        _COMPONENT_ORDER
    ):
        raise RuntimeError("observation VJP family/component map is not a partition")
    family_values = {
        family: sum(
            (components[name] for name in names),
            start=ledger.total.new_zeros(()),
        )
        for family, names in _FAMILY_COMPONENTS.items()
    }
    family_sum = sum(family_values.values(), start=ledger.total.new_zeros(()))
    loss_residual = ledger.total - component_sum
    family_residual = ledger.total - family_sum
    scale = max(abs(float(ledger.total.detach().float())), 1.0)
    if (
        abs(float(loss_residual.detach().float())) > 5e-6 * scale
        or abs(float(family_residual.detach().float())) > 5e-6 * scale
    ):
        raise RuntimeError("observation VJP loss partition does not close")
    return components, {
        "total_loss": float(ledger.total.detach().float()),
        "component_sum": float(component_sum.detach().float()),
        "family_sum": float(family_sum.detach().float()),
        "loss_partition_residual": float(loss_residual.detach().float()),
        "family_partition_residual": float(family_residual.detach().float()),
        "component_values": {
            name: float(components[name].detach().float()) for name in _COMPONENT_ORDER
        },
        "family_values": {
            name: float(value.detach().float()) for name, value in family_values.items()
        },
        "flow_components": component_rows,
    }


def _selected_parameter_boundary(
    owners: Mapping[str, tuple[tuple[str, nn.Parameter], ...]],
) -> tuple[
    tuple[tuple[str, nn.Parameter], ...],
    dict[str, tuple[int, ...]],
]:
    named: list[tuple[str, nn.Parameter]] = []
    owner_indices: dict[str, tuple[int, ...]] = {}
    for owner, rows in owners.items():
        start = len(named)
        named.extend(rows)
        owner_indices[owner] = tuple(range(start, len(named)))
    ids = [id(parameter) for _, parameter in named]
    if len(ids) != len(set(ids)):
        raise RuntimeError("selected VJP boundary contains overlapping owners")
    return tuple(named), owner_indices


def _cpu_gradients(
    loss: Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> tuple[Tensor | None, ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        None if gradient is None else gradient.detach().float().cpu() for gradient in gradients
    )


def _cpu_batched_gradients(
    losses: tuple[Tensor, ...],
    parameters: tuple[nn.Parameter, ...],
) -> tuple[tuple[Tensor | None, ...], ...]:
    """Evaluate component VJPs in one autograd engine traversal.

    Replaying ``autograd.grad`` many times over the same checkpointed graph can
    re-enter activation recomputation once per component.  The resulting
    stateful recomputation order is not a valid loss decomposition.  A batched
    identity cotangent requests every scalar row together and makes the
    linearity check against one ordinary total-loss VJP explicit.
    """

    if not losses:
        raise ValueError("batched VJP requires at least one scalar loss")
    if any(loss.ndim != 0 for loss in losses):
        raise ValueError("batched VJP losses must be scalar")
    loss_vector = torch.stack(losses)
    cotangent = torch.eye(
        len(losses),
        device=loss_vector.device,
        dtype=loss_vector.dtype,
    )
    gradients = torch.autograd.grad(
        loss_vector,
        parameters,
        grad_outputs=cotangent,
        retain_graph=False,
        allow_unused=True,
        is_grads_batched=True,
    )
    rows: list[list[Tensor | None]] = [[] for _ in losses]
    for gradient in gradients:
        if gradient is None:
            for row in rows:
                row.append(None)
            continue
        if int(gradient.shape[0]) != len(losses):
            raise RuntimeError("batched VJP lost its component axis")
        gradient_cpu = gradient.detach().float().cpu()
        for index, row in enumerate(rows):
            row.append(gradient_cpu[index])
    return tuple(tuple(row) for row in rows)


def _slice_gradients(
    gradients: tuple[Tensor | None, ...],
    indices: tuple[int, ...],
) -> tuple[Tensor | None, ...]:
    return tuple(gradients[index] for index in indices)


def _vector_inner(
    left: tuple[Tensor | None, ...],
    right: tuple[Tensor | None, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError("gradient vector boundaries do not align")
    value = 0.0
    for left_row, right_row in zip(left, right, strict=True):
        if left_row is not None and right_row is not None:
            value += float((left_row * right_row).sum())
    return value


def _vector_l2(value: tuple[Tensor | None, ...]) -> float:
    return math.sqrt(max(_vector_inner(value, value), 0.0))


def _vector_sum(
    rows: Iterable[tuple[Tensor | None, ...]],
) -> tuple[Tensor | None, ...]:
    values = tuple(rows)
    if not values:
        raise ValueError("cannot sum an empty gradient family")
    result: list[Tensor | None] = []
    for index in range(len(values[0])):
        present = [row[index] for row in values if row[index] is not None]
        result.append(None if not present else torch.stack(present).sum(dim=0))
    return tuple(result)


def _vector_difference(
    left: tuple[Tensor | None, ...],
    right: tuple[Tensor | None, ...],
) -> tuple[Tensor | None, ...]:
    if len(left) != len(right):
        raise ValueError("gradient vector boundaries do not align")
    result: list[Tensor | None] = []
    for left_row, right_row in zip(left, right, strict=True):
        if left_row is None and right_row is None:
            result.append(None)
        elif left_row is None:
            result.append(-right_row)  # type: ignore[operator]
        elif right_row is None:
            result.append(left_row)
        else:
            result.append(left_row - right_row)
    return tuple(result)


def _component_vjp_report(
    *,
    total_loss: float,
    component_values: Mapping[str, float],
    component_gradients: Mapping[str, tuple[Tensor | None, ...]],
    reference_gradient: tuple[Tensor | None, ...],
    owners: Mapping[str, tuple[tuple[str, nn.Parameter], ...]],
    top_parameters: int,
) -> dict[str, object]:
    selected, owner_indices = _selected_parameter_boundary(owners)
    if set(component_values) != set(_COMPONENT_ORDER):
        raise RuntimeError("component VJP values are incomplete")
    if set(component_gradients) != set(_COMPONENT_ORDER):
        raise RuntimeError("component VJP gradients are incomplete")
    if len(reference_gradient) != len(selected):
        raise RuntimeError("reference VJP does not align with selected parameters")
    gradients: dict[str, tuple[Tensor | None, ...]] = {
        name: component_gradients[name] for name in _COMPONENT_ORDER
    }
    gradients["total"] = _vector_sum(gradients[name] for name in _COMPONENT_ORDER)
    family_gradients = {
        family: _vector_sum(gradients[name] for name in names)
        for family, names in _FAMILY_COMPONENTS.items()
    }
    family_values = {
        family: sum(component_values[name] for name in names)
        for family, names in _FAMILY_COMPONENTS.items()
    }

    def owner_rows(value: tuple[Tensor | None, ...]) -> dict[str, object]:
        owner_rows: dict[str, object] = {}
        for owner, indices in owner_indices.items():
            owner_named = owners[owner]
            owner_gradients = _slice_gradients(value, indices)
            row: dict[str, object] = {
                **_gradient_stats(owner_named, owner_gradients),
                "top_parameters": _per_parameter_stats(
                    owner_named,
                    owner_gradients,
                    limit=top_parameters,
                ),
            }
            if owner == "coarse_delta_output":
                row["delta_output_channels"] = _delta_channel_stats(
                    owner_named,
                    owner_gradients,
                )
            owner_rows[owner] = row
        return owner_rows

    component_rows = {
        name: {
            "loss_value": float(component_values[name]),
            "owners": owner_rows(gradients[name]),
        }
        for name in _COMPONENT_ORDER
    }
    family_rows = {
        family: {
            "loss_value": float(family_values[family]),
            "owners": owner_rows(family_gradients[family]),
        }
        for family in _FAMILY_COMPONENTS
    }
    total_row = {
        "loss_value": float(total_loss),
        "owners": owner_rows(gradients["total"]),
    }
    reference_row = {
        "loss_value": float(total_loss),
        "owners": owner_rows(reference_gradient),
    }

    def geometry(
        names: tuple[str, ...],
        vectors: Mapping[str, tuple[Tensor | None, ...]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for owner, indices in owner_indices.items():
            owner_vectors = {name: _slice_gradients(vectors[name], indices) for name in names}
            total_vector = _slice_gradients(gradients["total"], indices)
            reference_vector = _slice_gradients(reference_gradient, indices)
            vector_sum = _vector_sum(owner_vectors.values())
            residual = _vector_difference(total_vector, vector_sum)
            reference_residual = _vector_difference(
                reference_vector,
                total_vector,
            )
            norms = {name: _vector_l2(value) for name, value in owner_vectors.items()}
            sum_norms = sum(norms.values())
            total_l2 = _vector_l2(total_vector)
            pairwise: dict[str, float | None] = {}
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    denominator = norms[left] * norms[right]
                    pairwise[f"{left}__{right}"] = (
                        None
                        if denominator <= 0.0
                        else _vector_inner(owner_vectors[left], owner_vectors[right]) / denominator
                    )
            result[owner] = {
                "gradient_l2": norms,
                "sum_gradient_l2": _vector_l2(vector_sum),
                "sum_individual_l2": sum_norms,
                "cancellation_ratio_sum_over_individual": (
                    None if sum_norms <= 0.0 else _vector_l2(vector_sum) / sum_norms
                ),
                "total_gradient_l2": total_l2,
                "partition_gradient_residual_l2": _vector_l2(residual),
                "partition_gradient_residual_relative": (
                    None if total_l2 <= 0.0 else _vector_l2(residual) / total_l2
                ),
                "single_total_reference_l2": _vector_l2(reference_vector),
                "single_total_reference_residual_l2": _vector_l2(reference_residual),
                "single_total_reference_residual_relative": (
                    None
                    if _vector_l2(reference_vector) <= 0.0
                    else _vector_l2(reference_residual) / _vector_l2(reference_vector)
                ),
                "pairwise_cosine": pairwise,
            }
        return result

    return {
        "engine": "fresh_forward_chunked_batched_identity_cotangent",
        "component_chunks": [list(chunk) for chunk in _COMPONENT_CHUNKS],
        "components": component_rows,
        "families": family_rows,
        "total": total_row,
        "single_total_reference": reference_row,
        "owner_component_geometry": geometry(_COMPONENT_ORDER, gradients),
        "owner_family_geometry": geometry(tuple(_FAMILY_COMPONENTS), family_gradients),
    }


def _scan_batch(
    *,
    model: ClearVLAMainlinePolicy,
    config: ExperimentConfig,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    probe_seed: int,
    batch_index: int,
    observation_parameters: tuple[tuple[str, nn.Parameter], ...],
    owners: Mapping[str, tuple[tuple[str, nn.Parameter], ...]],
    top_parameters: int,
    seed_axis: str = "joint",
    seed_offset: int = 0,
) -> dict[str, object]:
    seeds = _axis_seeds(
        probe_seed=probe_seed,
        batch_index=batch_index,
        seed_axis=seed_axis,
        seed_offset=seed_offset,
    )
    global_seed = int(seeds["global"])
    flow_seed = int(seeds["flow"])
    condition_seed = int(seeds["condition"])
    _seed(global_seed)
    flow_generator = _owned_generator(device, flow_seed)
    condition_generator = _owned_generator(device, condition_seed)
    ledger, _observation, _raw_losses, metadata = _formal_forward(
        model=model,
        config=config,
        batch=batch,
        device=device,
        dtype=dtype,
        flow_generator=flow_generator,
        condition_generator=condition_generator,
        mask_seed=(None if seeds["mask"] is None else int(seeds["mask"])),
    )
    parameters = tuple(parameter for _, parameter in observation_parameters)
    gradients = torch.autograd.grad(
        ledger.total,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    owner_index = {
        id(parameter): index for index, (_, parameter) in enumerate(observation_parameters)
    }
    owner_rows: dict[str, object] = {}
    for owner, named in owners.items():
        selected = tuple(gradients[owner_index[id(parameter)]] for _, parameter in named)
        row: dict[str, object] = _gradient_stats(named, selected)
        if owner == "coarse_delta_output":
            row["delta_output_channels"] = _delta_channel_stats(named, selected)
        owner_rows[owner] = row
    result = {
        "batch_index": int(batch_index),
        "probe_seed": global_seed,
        "seed_axis": str(seed_axis),
        "seed_offset": int(seed_offset),
        "random_seeds": seeds,
        "batch": _batch_identity(batch),
        "total_loss": float(ledger.total.detach().float()),
        "loss_groups": {
            name: float(value.detach().float()) for name, value in ledger.groups.items()
        },
        "observation": {
            **_gradient_stats(observation_parameters, gradients),
            "top_parameters": _per_parameter_stats(
                observation_parameters,
                gradients,
                limit=top_parameters,
            ),
        },
        "selected_owners": owner_rows,
        "forward": metadata,
    }
    del ledger, gradients
    return result


def _full_batch(
    *,
    model: ClearVLAMainlinePolicy,
    config: ExperimentConfig,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    probe_seed: int,
    batch_index: int,
    owners: Mapping[str, tuple[tuple[str, nn.Parameter], ...]],
    top_parameters: int,
    mask_seed: int | None = None,
) -> dict[str, object]:
    batch_seed = int(probe_seed) + 1009 * int(batch_index)
    selected, _owner_indices = _selected_parameter_boundary(owners)
    parameters = tuple(parameter for _, parameter in selected)
    chunk_names = tuple(name for chunk in _COMPONENT_CHUNKS for name in chunk)
    if len(chunk_names) != len(set(chunk_names)) or set(chunk_names) != set(_COMPONENT_ORDER):
        raise RuntimeError("component VJP chunks are not an exact partition")

    component_gradients: dict[str, tuple[Tensor | None, ...]] = {}
    component_values: dict[str, float] | None = None
    reference_gradient: tuple[Tensor | None, ...] | None = None
    reference_total: float | None = None
    reference_partition: dict[str, object] | None = None
    reference_metadata: dict[str, object] | None = None
    replay_rows: list[dict[str, object]] = []
    for chunk_index, chunk in enumerate(_COMPONENT_CHUNKS):
        _seed(batch_seed)
        ledger, observation, raw_losses, metadata = _formal_forward(
            model=model,
            config=config,
            batch=batch,
            device=device,
            dtype=dtype,
            flow_generator=_owned_generator(device, batch_seed + 1),
            condition_generator=_owned_generator(device, batch_seed + 2),
            mask_seed=mask_seed,
        )
        components, partition = _loss_components(
            config,
            ledger,
            observation,
            raw_losses,
        )
        current_values = {
            name: float(components[name].detach().float()) for name in _COMPONENT_ORDER
        }
        current_total = float(ledger.total.detach().float())
        if chunk_index == 0:
            component_values = current_values
            reference_total = current_total
            reference_partition = partition
            reference_metadata = metadata
            maximum_component_delta = 0.0
        else:
            assert component_values is not None
            assert reference_total is not None
            assert reference_metadata is not None
            maximum_component_delta = max(
                abs(current_values[name] - component_values[name]) for name in _COMPONENT_ORDER
            )
            tolerance = 5e-7 * max(abs(reference_total), 1.0)
            if abs(current_total - reference_total) > tolerance:
                raise RuntimeError("fresh-forward VJP chunks changed total loss")
            if maximum_component_delta > tolerance:
                raise RuntimeError("fresh-forward VJP chunks changed component loss")
            if metadata["noisy_physical_sha256"] != reference_metadata["noisy_physical_sha256"]:
                raise RuntimeError("fresh-forward VJP chunks changed flow noise")

        requested_losses = tuple(components[name] for name in chunk)
        requested_names = chunk
        if chunk_index == 0:
            requested_losses = (ledger.total, *requested_losses)
            requested_names = ("total", *requested_names)
        gradient_rows = _cpu_batched_gradients(requested_losses, parameters)
        for name, gradient in zip(requested_names, gradient_rows, strict=True):
            if name == "total":
                reference_gradient = gradient
            else:
                component_gradients[name] = gradient
        replay_rows.append(
            {
                "chunk_index": int(chunk_index),
                "components": list(chunk),
                "total_loss": current_total,
                "maximum_component_loss_delta_from_chunk0": float(maximum_component_delta),
                "noisy_physical_sha256": metadata["noisy_physical_sha256"],
            }
        )
        del ledger, observation, raw_losses, components, gradient_rows
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if (
        component_values is None
        or reference_gradient is None
        or reference_total is None
        or reference_partition is None
        or reference_metadata is None
        or set(component_gradients) != set(_COMPONENT_ORDER)
    ):
        raise RuntimeError("chunked component VJP did not complete")
    vjp = _component_vjp_report(
        total_loss=reference_total,
        component_values=component_values,
        component_gradients=component_gradients,
        reference_gradient=reference_gradient,
        owners=owners,
        top_parameters=top_parameters,
    )
    return {
        "batch_index": int(batch_index),
        "probe_seed": batch_seed,
        "batch": _batch_identity(batch),
        "forward": reference_metadata,
        "loss_partition": reference_partition,
        "chunk_replay": replay_rows,
        "vjp": vjp,
    }


def _write_report(path: Path, report: Mapping[str, object]) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite probe report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _raw_batch_at(train_loader: object, batch_index: int) -> object:
    """Resolve one deterministic batch without decoding every earlier batch."""

    batch_sampler = getattr(train_loader, "batch_sampler", None)
    dataset = getattr(train_loader, "dataset", None)
    collate_fn = getattr(train_loader, "collate_fn", None)
    if batch_sampler is None or dataset is None or not callable(collate_fn):
        raise RuntimeError("training loader does not expose sampler/dataset/collate")
    for index, sample_indices in enumerate(batch_sampler):
        if index != int(batch_index):
            continue
        samples = [dataset[int(sample_index)] for sample_index in sample_indices]
        return collate_fn(samples)
    raise RuntimeError(f"the deterministic sampler did not reach batch {batch_index}")


def main() -> None:
    args = _parser().parse_args()
    if args.scan_batches is not None and int(args.scan_batches) < 1:
        raise ValueError("--scan-batches must be positive")
    if args.batch_index is not None and int(args.batch_index) < 0:
        raise ValueError("--batch-index must be non-negative")
    if args.scan_seeds is not None:
        if args.batch_index is None:
            raise ValueError("--scan-seeds requires --batch-index")
        if int(args.scan_seeds) < 1:
            raise ValueError("--scan-seeds must be positive")
    if args.activation_vjp and args.batch_index is None:
        raise ValueError("--activation-vjp requires --batch-index")
    if args.activation_vjp and args.scan_seeds is not None:
        raise ValueError("--activation-vjp cannot be combined with --scan-seeds")
    if args.mask_seed is not None and args.scan_batches is not None:
        raise ValueError("--mask-seed requires --batch-index")
    if int(args.top_parameters) < 1:
        raise ValueError("--top-parameters must be positive")
    if int(ARCHITECTURE_MANIFEST.schema) < 29:
        raise RuntimeError(
            "the active architecture manifest predates detached self-conditioning"
        )

    config = _overrides(load_config(args.config), args)
    device = _device(args.device)
    dtype = resolve_compute_dtype(config)
    repository_root = Path(__file__).resolve().parents[2]
    _seed(config.data.seed)
    bundle = load_mainline_data(config)
    identity = build_checkpoint_identity(
        config,
        repo_root=repository_root,
        dataset=dataset_identity(bundle, config),
        language=language_identity(bundle, config),
    )
    model = ClearVLAMainlinePolicy(config).to(device)
    replay = load_checkpoint_for_validation(
        args.checkpoint,
        model=model,
        config=config,
        identity=identity,
    )
    if args.expected_checkpoint_step is not None and replay.global_step != int(
        args.expected_checkpoint_step
    ):
        raise ValueError(
            f"checkpoint step {replay.global_step} does not match expected "
            f"{args.expected_checkpoint_step}"
        )
    model.train()
    model.set_training_step(replay.global_step)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("checkpoint VJP probe requires pristine .grad fields")
    observation_parameters = _named_trainable(model, prefix=_OBSERVATION_PREFIX)
    owners = _owner_parameters(observation_parameters)

    loader_generator = torch.Generator().manual_seed(config.data.seed + 101)
    train_loader = bundle.loader(
        "train",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        generator=loader_generator,
    )
    requested = int(args.scan_batches) if args.scan_batches is not None else 1
    if requested > len(train_loader):
        raise ValueError(
            f"requested batch boundary {requested} exceeds loader length {len(train_loader)}"
        )
    if args.batch_index is not None and int(args.batch_index) >= len(train_loader):
        raise ValueError(
            f"requested batch index {args.batch_index} exceeds loader length {len(train_loader)}"
        )

    scan_rows: list[dict[str, object]] = []
    full_row: dict[str, object] | None = None
    activation_row: dict[str, object] | None = None
    raw_batches: Iterable[tuple[int, object]]
    if args.scan_batches is not None:
        raw_batches = enumerate(islice(train_loader, requested))
    else:
        assert args.batch_index is not None
        raw_batches = (
            (
                int(args.batch_index),
                _raw_batch_at(train_loader, int(args.batch_index)),
            ),
        )
    for batch_index, raw_batch in raw_batches:
        batch = to_training_batch(
            raw_batch,
            goal=bundle.goal,
            config=config,
            device=device,
        )
        if args.scan_batches is not None:
            row = _scan_batch(
                model=model,
                config=config,
                batch=batch,
                device=device,
                dtype=dtype,
                probe_seed=int(args.probe_seed),
                batch_index=batch_index,
                observation_parameters=observation_parameters,
                owners=owners,
                top_parameters=int(args.top_parameters),
                seed_axis=str(args.seed_axis),
            )
            scan_rows.append(row)
            print(
                "[observation-vjp-scan] "
                f"batch={batch_index} "
                f"loss={row['total_loss']:.6g} "
                f"observation_l2={row['observation']['gradient_l2']:.6g}",
                flush=True,
            )
        elif args.scan_seeds is not None:
            for seed_offset in range(int(args.scan_seeds)):
                row = _scan_batch(
                    model=model,
                    config=config,
                    batch=batch,
                    device=device,
                    dtype=dtype,
                    probe_seed=int(args.probe_seed),
                    batch_index=batch_index,
                    observation_parameters=observation_parameters,
                    owners=owners,
                    top_parameters=int(args.top_parameters),
                    seed_axis=str(args.seed_axis),
                    seed_offset=seed_offset,
                )
                row["seed_offset"] = int(seed_offset)
                scan_rows.append(row)
                print(
                    "[observation-vjp-seed-scan] "
                    f"batch={batch_index} "
                    f"seed_offset={seed_offset} "
                    f"seed={row['probe_seed']} "
                    f"loss={row['total_loss']:.6g} "
                    f"observation_l2={row['observation']['gradient_l2']:.6g}",
                    flush=True,
                )
        elif args.activation_vjp:
            activation_row = _activation_batch(
                model=model,
                config=config,
                batch=batch,
                device=device,
                dtype=dtype,
                probe_seed=int(args.probe_seed),
                batch_index=batch_index,
                mask_seed=(None if args.mask_seed is None else int(args.mask_seed)),
            )
            print(
                "[observation-vjp-activation] "
                f"batch={batch_index} "
                f"loss={activation_row['loss_values']['total']:.6g}",
                flush=True,
            )
        else:
            full_row = _full_batch(
                model=model,
                config=config,
                batch=batch,
                device=device,
                dtype=dtype,
                probe_seed=int(args.probe_seed),
                batch_index=batch_index,
                owners=owners,
                top_parameters=int(args.top_parameters),
                mask_seed=(None if args.mask_seed is None else int(args.mask_seed)),
            )
            print(
                "[observation-vjp-full] "
                f"batch={batch_index} "
                f"loss={full_row['loss_partition']['total_loss']:.6g}",
                flush=True,
            )
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.scan_batches is not None and len(scan_rows) != int(args.scan_batches):
        raise RuntimeError("scan stopped before the requested number of batches")
    if args.scan_seeds is not None and len(scan_rows) != int(args.scan_seeds):
        raise RuntimeError("seed scan stopped before the requested number of seeds")
    if (
        args.batch_index is not None
        and args.scan_seeds is None
        and full_row is None
        and activation_row is None
    ):
        raise RuntimeError("the requested full-VJP batch was not reached")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("VJP probe unexpectedly populated persistent .grad tensors")

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "manifest_schema": int(ARCHITECTURE_MANIFEST.schema),
        "manifest_sha256": ARCHITECTURE_MANIFEST.digest(),
        "source": _repository_state(args.expected_source_commit),
        "checkpoint": {
            "path": str(args.checkpoint.expanduser().resolve()),
            "epoch": int(replay.epoch),
            "global_step": int(replay.global_step),
            "best_metric": replay.best_metric,
            "saved_source_digest": replay.saved_source_digest,
            "current_source_digest": replay.current_source_digest,
            "changed_source_files": list(replay.changed_source_files),
        },
        "config": str(args.config.expanduser().resolve()),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "model_training_step": int(replay.global_step),
        "optimizer_constructed": False,
        "optimizer_step_taken": False,
        "checkpoint_written": False,
        "probe_seed": int(args.probe_seed),
        "seed_axis": str(args.seed_axis),
        "mask_seed": args.mask_seed,
        "data": {
            "train_loader_batches": len(train_loader),
            "batch_size": int(config.optimizer.batch_size),
            "workers": int(config.data.num_workers),
        },
        "mode": (
            "batch_scan"
            if args.scan_batches is not None
            else "seed_scan"
            if args.scan_seeds is not None
            else "activation_vjp"
            if args.activation_vjp
            else "full_vjp"
        ),
        "scan": scan_rows,
        "full_vjp": full_row,
        "activation_vjp": activation_row,
    }
    destination = _write_report(args.output, report)
    print(f"[observation-vjp-report] path={destination}", flush=True)


if __name__ == "__main__":
    main()
