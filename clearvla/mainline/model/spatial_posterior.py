"""Compact current-image laws; no pixels, labels, or persistent tracker state.

Unlike a centroid, the raster retains distinct equal-mean layouts. The learned
context reads the whole law inside each K/C before consumers pool identities.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

SPATIAL_SIDE = 16


def validate_spatial_probability(probability: Tensor, coordinates: Tensor) -> None:
    if probability.shape != (*coordinates.shape[:-1], SPATIAL_SIDE, SPATIAL_SIDE):
        raise ValueError("current spatial law must retain native [B,K,C,16,16]")
    if probability.dtype != torch.float32 or probability.device != coordinates.device:
        raise ValueError("current spatial law must remain FP32 on the object device")


class SpatialPosteriorContext(nn.Module):
    """A nonlinear read of a full conditional image law, never its mean.

    The Hellinger input has fixed scale for any nonempty distribution; diffuse
    support cannot disappear just because each pixel has small probability.
    Camera support, fractional availability and K+null mass stay outside this
    conditional feature and retain their original owners.
    """
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.read = nn.Sequential(
            nn.Linear(SPATIAL_SIDE ** 2, hidden, bias=False), nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
        )

    def forward(self, probability: Tensor, support: Tensor) -> Tensor:
        if probability.shape[-2:] != (SPATIAL_SIDE, SPATIAL_SIDE):
            raise ValueError("posterior context requires the native 16x16 law")
        if support.dtype != torch.bool or support.shape != probability.shape[:-2]:
            raise ValueError("posterior context lost K/C support")
        clean = torch.where(support[..., None, None], probability.float(), 0.)
        # Clamp only protects the derivative at structural zero; it does not
        # renormalize missing views or change the producer's actual probability.
        features = clean.clamp_min(1e-12).sqrt().flatten(-2) * SPATIAL_SIDE
        features = torch.where(support[..., None], features, 0.)
        value = self.read(features.to(self.read[0].weight.dtype))
        return torch.where(support[..., None], value, 0.)


def spatial_compatibility(probability: Tensor, query: Tensor, transport: Tensor,
                          covariance: Tensor, distance) -> Tensor:
    """log E_p exp(score(query,x+transport)); no average location shortcut.

    B/T/Q/I/K/C remain distinct. Pixel chunks bound intermediate memory and
    never truncate support. The same covariance model is used by legacy P2.
    """
    from ...vision.entity_chart import current_image_grid
    grid = current_image_grid(SPATIAL_SIDE, SPATIAL_SIDE, device=probability.device).flatten(0, 1)
    p = probability.flatten(-2).float()
    pieces = []
    for start in range(0, grid.shape[0], 64):
        point = grid[start:start + 64]
        future = (transport[..., None, :].float() + point).clamp(-1., 1.)
        delta = query[..., None, :] - future[:, None, None]
        d = distance(delta, covariance[:, None, None, ..., None, :])
        score = (-.25 * d).clamp(-1., 0.)
        pieces.append((score.exp() * p[:, None, None, None, ..., start:start + 64]).sum(-1))
    mass = torch.stack(pieces).sum(0)
    return torch.where(p.sum(-1)[:, None, None, None] > 0,
                       mass.clamp_min(1e-30).log(), torch.zeros_like(mass))
