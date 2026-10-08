"""Canonical source rasterization before global object competition.

Each local source keeps its own mass until a named value expectation. This
forms m[u] and weighted typed/context fields, not the old q(k|j) p(n|j) law.
"""

import math

import torch


def rasterize_source(spatial, prior, validity, *, rows, columns):
    """Return [B,C,J,U] joint mass and [B,C,U] total source mass.

    This is the explicit FP32 bilinear reference law with all four corners.
    No K or hidden-feature expansion over atoms is materialized.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (rows, columns)):
        raise ValueError("canonical raster dimensions must be positive integers")
    if prior.ndim < 3 or any(n < 1 for n in prior.shape):
        raise ValueError("canonical source requires nonempty batch/camera/local axes")
    if (
        spatial.valid.shape != spatial.probability.shape
        or spatial.valid.dtype != torch.bool
        or spatial.coordinates.shape != (*spatial.probability.shape, 2)
    ):
        raise ValueError("canonical source lost its spatial atom support")
    if any(
        x.device != prior.device
        for x in (validity, spatial.coordinates, spatial.probability, spatial.valid)
    ):
        raise ValueError("canonical source tensors must share a device")
    if prior.shape != spatial.probability.shape[:-1] or validity.shape != (*prior.shape, 1):
        raise ValueError("canonical producer lost its prior/local-source axes")
    with torch.autocast(device_type=prior.device.type, enabled=False):
        b, c = prior.shape[:2]
        legal = validity[..., 0] > 0
        atom_legal = spatial.valid & legal[..., None]
        # Local unobserved support dominates an atom-valid placeholder. Mask
        # before products/floor, otherwise zero times NaN poisons forward/VJP.
        xy = torch.where(atom_legal[..., None], spatial.coordinates.float(), 0.0).clamp(-1, 1)
        mass = torch.where(atom_legal, spatial.probability.float(), 0.0)
        safe_prior = torch.where(legal, prior.float(), 0.0)
        safe_valid = torch.where(legal, validity[..., 0].float(), 0.0)
        mass = mass * (safe_prior * safe_valid)[..., None]
        j = math.prod(prior.shape[2:])
        x = (xy[..., 0] + 1) * (columns - 1) / 2
        y = (xy[..., 1] + 1) * (rows - 1) / 2
        x0 = x.floor()
        y0 = y.floor()
        dx = x - x0
        dy = y - y0
        source = mass.new_zeros(b, c, j, rows * columns)
        for ox, oy, weight in (
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy),
        ):
            index = (
                (y0.long() + oy).clamp_max(rows - 1) * columns
                + (x0.long() + ox).clamp_max(columns - 1)
            ).reshape(b, c, j, -1)
            source = source.scatter_add(-1, index, (mass * weight).reshape(b, c, j, -1))
        return source, source.sum(-2)


def rasterize_value(source, mass, value, valid):
    """Conditional field at u; ordinary derivatives reach pi,p,xy and values."""
    b, c, j, u = source.shape
    v = torch.where(valid > 0, value, 0.0).reshape(b, c, j, -1).float()
    with torch.autocast(device_type=source.device.type, enabled=False):
        total = source.transpose(-1, -2) @ v
        support = mass > 0
        denominator = torch.where(support, mass, 1.0)
        return torch.where(support[..., None], total / denominator[..., None], 0.0)
