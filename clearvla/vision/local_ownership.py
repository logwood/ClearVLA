"""One soft observation identity, multiple typed attributes (M3).

These helpers combine learned compatibility scores; they do not assert
calibrated independent sensor likelihoods or identify global physical objects.
Source validity is an input, never a predicted consequence of agreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

INDEPENDENT_LOCAL_OWNERS = "independent_typed_v1"
COUPLED_LOCAL_OWNERS = "coupled_observation_v1"
TYPES = ("semantic", "appearance", "geometry")


def local_ownership_metadata(mode: str) -> dict[str, object]:
    if mode not in {INDEPENDENT_LOCAL_OWNERS, COUPLED_LOCAL_OWNERS}:
        raise ValueError("unknown local observation ownership")
    return {
        "schema": "clearvla-local-ownership-v1",
        "mode": mode,
        "location_owner": "independent_per_type"
        if mode == INDEPENDENT_LOCAL_OWNERS
        else "one_joint_soft_observation",
        "attribute_values": "separate_semantic_appearance_geometry",
        "real_null_authority": "dense_global_binder",
    }


def _masked_log_law(
    value: Tensor, support: Tensor, *, empty_uniform: bool
) -> tuple[Tensor, Tensor]:
    """Finite log interface; an empty fine row has zero observable mass.

    An empty hypothesis row retains a neutral prior over M so the downstream
    binder routes missing candidates to null instead of destroying prior mass.
    It still has no valid observed candidates and no gradients to their values.
    """
    any_supported = support.any(dim=-1, keepdim=True)
    masked = torch.where(
        support, value.float(), torch.full_like(value.float(), torch.finfo(torch.float32).min)
    )
    safe = torch.where(any_supported, masked, torch.zeros_like(masked))
    log_probability = torch.log_softmax(safe, dim=-1)
    probability = log_probability.exp()
    if not empty_uniform:
        probability = torch.where(support, probability, torch.zeros_like(probability))
    return log_probability, probability


@dataclass(frozen=True)
class LocalObservationLaw:
    candidate_logits: Tensor
    candidate_log_probability: Tensor
    candidate_probability: Tensor
    hypothesis_log_probability: Tensor
    hypothesis_probability: Tensor


def couple_local_observation(
    typed_logits: Mapping[str, Tensor],
    parent_log_probability: Tensor,
    validity: Tensor,
) -> LocalObservationLaw:
    """Update full G1 support once, then reuse identity for every attribute."""
    if set(typed_logits) != set(TYPES):
        raise ValueError("joint local observation requires all three typed evidence sources")
    if (
        parent_log_probability.ndim < 2
        or validity.dtype != torch.bool
        or tuple(validity.shape) != tuple(parent_log_probability.shape)
    ):
        raise ValueError("local identity must preserve hypothesis/candidate support")
    shape, device = tuple(parent_log_probability.shape), parent_log_probability.device
    if validity.device != device or any(
        tuple(v.shape) != shape or v.device != device for v in typed_logits.values()
    ):
        raise ValueError("typed local evidence must use one source chart")
    # Mask before every arithmetic reduction: an invalid payload can be NaN.
    rows = [
        torch.where(
            validity, typed_logits[name].float(), torch.zeros_like(parent_log_probability.float())
        )
        for name in TYPES
    ]
    evidence = torch.stack(rows, dim=0).sum(dim=0) / math.sqrt(float(len(TYPES)))
    parent = torch.where(validity, parent_log_probability.float(), torch.zeros_like(evidence))
    logits = evidence + parent
    candidate_log, candidate = _masked_log_law(logits, validity, empty_uniform=False)
    safe_logits = logits.masked_fill(~validity, torch.finfo(torch.float32).min)
    hyp_supported = validity.any(dim=-1)
    # Integrate a normalized parent measure; no candidate-count/entropy reward.
    integrated = torch.logsumexp(safe_logits, dim=-1)
    hyp_log, hyp = _masked_log_law(integrated, hyp_supported, empty_uniform=True)
    return LocalObservationLaw(safe_logits, candidate_log, candidate, hyp_log, hyp)


def refine_local_observation(
    parent_log_probability: Tensor,
    typed_residuals: Mapping[str, Tensor],
    hypothesis_support: Tensor,
) -> Tensor:
    """G3 refines the same M identity, never three independent identities."""
    if set(typed_residuals) != set(TYPES) or hypothesis_support.dtype != torch.bool:
        raise ValueError(
            "joint local correction requires complete typed evidence and Boolean support"
        )
    if tuple(hypothesis_support.shape) != tuple(parent_log_probability.shape):
        raise ValueError("local correction lost hypothesis support")
    if hypothesis_support.device != parent_log_probability.device or any(
        tuple(v.shape) != tuple(parent_log_probability.shape)
        or v.device != parent_log_probability.device
        for v in typed_residuals.values()
    ):
        raise ValueError("local correction evidence does not align with its parent")
    residual = torch.stack(
        [
            torch.where(
                hypothesis_support,
                typed_residuals[n].float(),
                torch.zeros_like(parent_log_probability.float()),
            )
            for n in TYPES
        ]
    ).sum(0) / math.sqrt(float(len(TYPES)))
    update = 0.5 * torch.tanh(residual)
    parent = torch.where(
        hypothesis_support, parent_log_probability.float(), torch.zeros_like(update)
    )
    log_probability, _ = _masked_log_law(parent + update, hypothesis_support, empty_uniform=True)
    return log_probability
