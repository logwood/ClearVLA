"""Ordinary bilinear adjoints, including corners with zero forward mass.

The forward log-sum-exp remains owned by entity_chart. A log at an empty
pixel has no finite derivative; normalized probability therefore retains the
source measure and differentiates the linear bilinear law directly. This is
not a straight-through estimator. Its Jacobian is the analytic derivative of
the same normalized splat used by the forward, including zero coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


def _corners(coordinates: Tensor, rows: int, columns: int):
    xy = coordinates.float().clamp(-1, 1)
    x, y = (xy[..., 0] + 1) * ((columns - 1) / 2), (xy[..., 1] + 1) * ((rows - 1) / 2)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    sx, sy = (columns - 1) / 2, (rows - 1) / 2
    for ox, oy, w, wx, wy in (
        (0, 0, (1-dx)*(1-dy), -(1-dy)*sx, -(1-dx)*sy),
        (1, 0, dx*(1-dy), (1-dy)*sx, -dx*sy),
        (0, 1, (1-dx)*dy, -dy*sx, (1-dx)*sy),
        (1, 1, dx*dy, dy*sx, dx*sy),
    ):
        index = (y0.long()+oy).clamp_max(rows-1)*columns + (x0.long()+ox).clamp_max(columns-1)
        yield index, w, wx, wy


class _NormalizedSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base_log, coordinates, valid, destination, log_mass, supported, axes):
        observed = supported & destination
        nonempty = observed.any(axes, keepdim=True)
        masked = log_mass.masked_fill(~observed, -torch.inf)
        safe = torch.where(nonempty, masked, 0.)
        denominator = torch.logsumexp(safe, axes, keepdim=True)
        log_p = torch.where(observed, safe-denominator, 0.)
        p = torch.where(observed, log_p.exp(), 0.)
        ctx.axes = axes
        ctx.save_for_backward(base_log, coordinates, valid, destination, log_mass,
                              observed, nonempty, denominator, p)
        return p, log_p

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_p, grad_log_p):
        base, coordinates, valid, destination, lm, observed, nonempty, denominator, p = ctx.saved_tensors
        axes = ctx.axes
        gp = torch.zeros_like(p) if grad_p is None else grad_p
        gl = torch.zeros_like(p) if grad_log_p is None else torch.where(observed, grad_log_p, 0.)
        # Accumulate in double: base-logD can be large at a zero corner even
        # when all positive forward contributions are tiny. Never insert eps.
        gp, gl, p = gp.double(), gl.double(), p.double()
        center_p = (gp*p).sum(axes, keepdim=True).expand_as(p).flatten(-2)
        center_l = gl.sum(axes, keepdim=True).expand_as(p).flatten(-2)
        denom = denominator.expand_as(lm).flatten(-2).double()
        log_image = lm.flatten(-2).double()
        active_image = (destination & nonempty).flatten(-2)
        positive_image = observed.flatten(-2)
        gp, gl = gp.flatten(-2), gl.flatten(-2)
        b,k,c,h,w = lm.shape
        source = base.double()
        gbase = torch.zeros_like(source)
        gcoords = torch.zeros_like(coordinates, dtype=torch.float64)
        for index, coefficient, wx, wy in _corners(coordinates, h, w):
            idx = index[:,None].expand(b,k,c,-1)
            legal = valid & active_image.gather(-1,idx)
            coeff = coefficient[:,None].double()
            positive = legal & (coeff > 0)
            positive_pixel = positive_image.gather(-1,idx)
            log_den = denom.gather(-1,idx)
            log_num = log_image.gather(-1,idx)
            delta_p = gp.gather(-1,idx)-center_p.gather(-1,idx)
            delta_l = center_l.gather(-1,idx)
            local_l = gl.gather(-1,idx)
            # Invalid payload is removed before exp, so unavailable NaNs and
            # empty normalization groups cannot create inf*0 in backward.
            ratio_d = torch.where(legal, source-log_den, 0.).exp()
            ratio_n = torch.where(legal & positive_pixel, source-log_num, 0.).exp()
            tangent = ratio_d*(delta_p-delta_l)
            tangent = tangent + torch.where(positive_pixel, ratio_n*local_l, 0.)
            tangent = torch.where(legal, tangent, 0.)
            log_coeff = torch.where(coeff > 0, coeff, 1.).log()
            weighted_d = torch.where(positive, source+log_coeff-log_den, 0.).exp()
            weighted_n = torch.where(positive & positive_pixel, source+log_coeff-log_num, 0.).exp()
            mass_grad = weighted_d*(delta_p-delta_l) + torch.where(positive_pixel, weighted_n*local_l, 0.)
            gbase += torch.where(positive, mass_grad, 0.)
            gx = (tangent*wx[:,None].double()).sum(1)
            gy = (tangent*wy[:,None].double()).sum(1)
            gcoords += torch.stack((gx,gy),-1)
        gcoords *= ((coordinates >= -1) & (coordinates <= 1)).to(gcoords)
        return gbase.to(base.dtype), gcoords.to(coordinates.dtype), None, None, None, None, None


@dataclass(frozen=True)
class BilinearLogTransport:
    base_log: Tensor                 # [B,K,C,atoms], includes local mass and p
    coordinates: Tensor              # [B,C,atoms,2]
    valid: Tensor                    # [B,K,C,atoms]
    destination: Tensor | None = None # independent observed mask, not exp(m)>0

    def normalized(self, log_mass: Tensor, supported: Tensor, axes: tuple[int,...]):
        destination = torch.ones_like(supported) if self.destination is None else self.destination
        return _NormalizedSplat.apply(self.base_log, self.coordinates, self.valid,
                                      destination, log_mass.detach(), supported, axes)

    def restrict(self, mask: Tensor):
        return replace(self, destination=mask if self.destination is None else mask & self.destination)

    def permute(self, index: Tensor):
        return replace(self, base_log=self.base_log[:,index], valid=self.valid[:,index],
                       destination=None if self.destination is None else self.destination[:,index])
