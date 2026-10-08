"""Probe-only SAM-inspired mechanisms; no production registration or SAM weights.

Only the existing G address features are candidates for modification. Observed
values, support, shared K/null ownership, and the action chart are not replaced.
An untrained adapter and its mechanical VJP are not behavioral evidence.
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


def mechanical_contracts(cls, hidden=48, rank=8):
    """Instrumentation checks, deliberately separate from learning evidence."""
    torch.manual_seed(1729)
    b,c,h,w,k = 2,2,4,4,4
    x = torch.randn(b,c*h*w,hidden)
    slots = torch.randn(b,k,hidden)
    q = torch.randn(b,c*h*w,k+1).softmax(-1)[...,:k]
    legal = torch.rand(b,c*h*w) > .3
    model = cls(hidden,rank)
    x = torch.where(legal[...,None],x,float('nan'))
    q = torch.where(legal[...,None],q,float('nan'))
    safe = torch.where(legal[...,None],x,0.)
    y = model(x,slots,q,legal,(c,h,w))
    assert torch.equal(y,safe), 'zero-init must preserve all observed values exactly'
    cotangent = torch.randn_like(y)
    grads = torch.autograd.grad((y*cotangent).sum(),tuple(model.parameters()),allow_unused=True)
    initial = {n:None if g is None else float(g.norm()) for (n,_),g in zip(model.named_parameters(),grads)}
    assert initial['out.weight'] > 0
    with torch.no_grad(): model.out.weight.normal_(0,.01)
    xx = x.clone().requires_grad_(); qq=q.clone().requires_grad_(); ss=slots.clone().requires_grad_()
    y = model(xx,ss,qq,legal,(c,h,w))
    perm = torch.tensor([2,0,3,1])
    equiv = float((y-model(xx,ss[:,perm],qq[...,perm],legal,(c,h,w))).abs().max())
    assert equiv < 2e-6
    empty = model(torch.full_like(x,float('nan')),slots,torch.full_like(q,float('nan')),torch.zeros_like(legal),(c,h,w))
    assert torch.isfinite(empty).all() and torch.count_nonzero(empty)==0
    vg = torch.autograd.grad((y*cotangent).sum(),(xx,qq,ss,*model.parameters()),allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in vg)
    assert torch.count_nonzero(vg[0][~legal])==0 and torch.count_nonzero(vg[1][~legal])==0
    # No camera convolution can send values from camera 0 to camera 1.
    other = safe.clone();other[:,:h*w] += 2
    changed = model(other,slots,q,legal,(c,h,w))
    # Feedback ignores x in its update; region fusion is camera-local.
    camera_error=float((changed[:,h*w:]-y[:,h*w:]).abs().max())
    assert camera_error==0
    constant_error = None
    if cls is RegionImageFusion:
        xc = torch.ones_like(x)*.7
        qc = torch.ones_like(q)*.2
        zz = model.latent(xc,slots,qc,legal,(c,h,w))
        reference=zz[legal][0]
        constant_error=float((zz[legal]-reference).abs().max().detach())
        assert constant_error<2e-6, 'padding/support must not fabricate contrast on constant fields'
    return dict(zero_init_exact=True,permutation_max_error=equiv,empty_finite_zero=True,
                invalid_input_grad_zero=True,camera_isolation_exact=True,
                initial_parameter_vjp=initial,
                nonzero_adapter_vjp_finite=True,parameters=sum(p.numel() for p in model.parameters()),
                constant_field_max_error=constant_error,
                scope='random cotangent mechanical VJP; not actual action/identity loss or trained effect')
