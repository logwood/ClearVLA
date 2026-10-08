"""Source-consistent observed-feature matching, with explicit unknown mass.

This is an appearance correspondence candidate, not a physical identity oracle.
It compares frozen observed descriptors before G pooling. Identical descriptors
have an exact-match limit; unmatched rows stay unknown. No forecast/search-time
width, language, learned G content, or task label changes this measurement.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def observed_feature_correspondence(
    source: Tensor,
    target: Tensor,
    source_observed: Tensor,
    target_observed: Tensor,
    *,
    distance_scale: float = 0.05,
) -> Tensor:
    """Return [...,N,M+1], last column is unknown; all axes are declared.

    Inverse squared descriptor distance uses a null reference at distance_scale.
    The real partition is averaged over observed candidates, so increasing chart
    size cannot by itself suppress null. Multiple exact matches stay unknown;
    averaging their coordinates would invent a displacement for a static image.
    """
    if source.ndim < 2 or target.ndim < 2 or source.shape[-1] == 0:
        raise ValueError("correspondence requires explicit nonempty descriptor axes")
    if source.shape[:-2] != target.shape[:-2] or source.shape[-1] != target.shape[-1]:
        raise ValueError("correspondence requires matching batch/view/descriptor axes")
    if source_observed.shape != source.shape[:-1] or target_observed.shape != target.shape[:-1]:
        raise ValueError("correspondence source masks are misaligned")
    if (
        source_observed.dtype != torch.bool
        or target_observed.dtype != torch.bool
        or not math.isfinite(distance_scale)
        or distance_scale <= 0
    ):
        raise ValueError("correspondence requires Boolean support and positive scale")
    if any(x.device != source.device for x in (target, source_observed, target_observed)):
        raise ValueError("correspondence values and masks must share a device")
    if not source.is_floating_point() or not target.is_floating_point():
        raise ValueError("correspondence descriptors must be floating point")
    safe_source = torch.where(source_observed[..., None], source, 0.0)
    safe_target = torch.where(target_observed[..., None], target, 0.0)
    if not bool(torch.isfinite(safe_source).all()) or not bool(torch.isfinite(safe_target).all()):
        raise ValueError("observed correspondence descriptors must be finite")
    if target.shape[-2] == 0:
        # No candidate is not an empty probability vector: unknown owns one.
        # Preserve a genuine zero derivative through both admitted sources.
        return safe_source.float().sum(-1, keepdim=True) * 0 + safe_target.float().sum() * 0 + 1
    with torch.autocast(device_type=source.device.type, enabled=False):
        a = F.normalize(torch.where(source_observed[..., None], source.float(), 0.0), dim=-1)
        b = F.normalize(torch.where(target_observed[..., None], target.float(), 0.0), dim=-1)
        # Double dot products remove cancellation around identical FP32 vectors.
        # This is a small N x M comparison, not a materialized N x M x D value.
        a, b = a.double(), b.double()
        distance = (
            a.square().sum(-1)[..., None]
            + b.square().sum(-1)[..., None, :]
            - 2 * (a @ b.transpose(-1, -2))
        ).clamp_min(0.0)
        legal = source_observed[..., None] & target_observed[..., None, :]
        exact = legal & (distance <= 1e-12)
        exact_count = exact.sum(-1, keepdim=True)
        has_exact = exact_count > 0
        safe_distance = torch.where(legal & ~exact, distance, 1.0)
        score = 2 * (
            torch.log(torch.tensor(distance_scale, dtype=distance.dtype, device=distance.device))
            - safe_distance.log()
        )
        score = score - target_observed.sum(-1).clamp_min(1).double().log()[..., None, None]
        score = score.masked_fill(~legal, -torch.inf)
        null = torch.zeros_like(score[..., :1])
        law = torch.softmax(torch.cat((score, null), -1), -1)
        unique = exact_count == 1
        exact_law = exact.double() * unique
        law = torch.where(has_exact, torch.cat((exact_law, (~unique).double()), -1), law)
        return law.float()
