"""Opt-in SAM-inspired address refinements; no external model or new inputs.

Shared K/null ownership is reused. Observed value tensors are never rewritten.
Both adapters start at exact zero residual and use ordinary gradients.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SlotToImageFeedback(nn.Module):
    """Shared low-rank token-to-image value feedback using existing ownership."""
    def __init__(self, hidden, rank=32):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.out = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def latent(self, x, slots, q, legal, shape):
        del shape
        q = torch.where(legal[..., None], q, 0.)
        return q @ self.down(slots)

    def forward(self, x, slots, q, legal, shape):
        safe = torch.where(legal[..., None], x, 0.)
        delta = self.out(self.latent(safe, slots, q, legal, shape))
        return torch.where(legal[..., None], safe + delta, 0.)


class RegionImageFusion(nn.Module):
    """Shared image+soft-mask local fusion, inspired by SAM's mask encoder.

    K is a batch axis, never a learned channel identity. Each camera is convolved
    independently. This is an explicit experimental adaptation, not SAM code or
    an assertion that a soft K mask denotes one physical object.
    """
    def __init__(self, hidden, rank=32):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.mask = nn.Linear(1, rank, bias=False)
        self.depthwise = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False)
        self.out = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def latent(self, x, slots, q, legal, shape):
        del slots
        b, n, _ = x.shape
        c, h, w = shape
        k = q.shape[-1]
        if n != c*h*w:
            raise ValueError('camera/spatial axes do not match real candidate layout')
        safe = torch.where(legal[..., None], x, 0.)
        q = torch.where(legal[..., None], q, 0.)
        z = self.down(safe)[:, :, None, :] + self.mask(q[..., None])
        z = torch.where(legal[..., None, None], z, 0.)
        r = z.shape[-1]
        z = z.reshape(b, c, h, w, k, r).permute(0,1,4,5,2,3).reshape(b*c*k,r,h,w)
        # Signed neighbor differences, with the same valid-neighbor weights in
        # both terms. A spatially constant valid feature must remain constant
        # even beside padding or a support hole. Plain zero-padded convolution
        # failed that control and created apparent "region contrast".
        valid = legal.reshape(b,c,1,1,h,w).expand(b,c,k,r,h,w).reshape(b*c*k,r,h,w).to(z)
        delta = self.depthwise(z) - z*self.depthwise(valid)
        z = F.silu(z + delta)
        z = z.reshape(b,c,k,r,h,w).permute(0,1,4,5,2,3).reshape(b,n,k,r)
        z = torch.where(legal[..., None, None], z, 0.)
        return (q[..., None] * z).sum(-2)

    def forward(self, x, slots, q, legal, shape):
        safe = torch.where(legal[..., None], x, 0.)
        delta = self.out(self.latent(safe, slots, q, legal, shape))
        return torch.where(legal[..., None], safe + delta, 0.)

