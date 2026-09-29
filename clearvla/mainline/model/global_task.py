"""P3's static causal-task compiler, separate from dynamic action coordination."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..global_task import CompiledGlobalTask


class GlobalTaskCompiler(nn.Module):
    """Transform finalized S values without introducing task-independent values.

    The four intervals remain separate. There is no learned query residual,
    phase timer, task keyword, candidate-world argument, or noisy-action input.
    Source normalization remains owned by S and the existing bottom organizer.
    """
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.values = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
        )

    def forward(self, source: Tensor) -> CompiledGlobalTask:
        if source.ndim != 3 or source.shape[1:] != (4, self.hidden):
            raise ValueError("global task compiler requires S [B,4,H]")
        if not source.is_floating_point():
            raise TypeError("global task source must be floating point")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("nonfinite supported global task source")
        # Autocast may choose a different token dtype. Preserve the source
        # reference and ordinary upstream gradients rather than detaching it.
        tokens = self.values(source.to(dtype=self.values[0].weight.dtype))
        result = CompiledGlobalTask(tokens, source, id(self))
        result.validate(source, hidden=self.hidden, strict=True)
        return result
