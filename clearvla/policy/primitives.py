from __future__ import annotations

"""Small neural primitives shared across policy generations."""

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_positions(positions: Sequence[int], hidden: int) -> Tensor:
    pos = torch.tensor(tuple(int(x) for x in positions), dtype=torch.float32)[:, None]
    half = hidden // 2
    if half == 0:
        return torch.zeros(len(positions), hidden)
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
    out = torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], dim=-1)
    if out.shape[-1] < hidden:
        out = F.pad(out, (0, hidden - out.shape[-1]))
    return out[:, :hidden]


class BiasFreeFFN(nn.Module):
    def __init__(self, hidden: int, expansion: float = 4.0) -> None:
        super().__init__()
        inner = int(round(hidden * expansion))
        self.net = nn.Sequential(
            nn.Linear(hidden, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, hidden, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden))

    def forward(self, t: Tensor) -> Tensor:
        half = self.hidden // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        phase = t[:, None] * freq[None]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.hidden:
            emb = F.pad(emb, (0, self.hidden - emb.shape[-1]))
        return self.net(emb)
