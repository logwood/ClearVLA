"""Pure diagnostic reductions shared by policy implementations."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor


@contextmanager
def fp32_diagnostic(value: Tensor) -> Iterator[Tensor]:
    """Yield a detached FP32 tensor outside the active autocast region.

    Diagnostic linear algebra must not inherit the model's AMP dtype or own a
    gradient path. Keeping both rules here prevents local `.float()` calls from
    being silently undone by autocast-enabled operations such as matmul.
    """

    with torch.no_grad(), torch.autocast(
        device_type=value.device.type, enabled=False
    ):
        yield value.detach().to(dtype=torch.float32)


def time_stratified_attention(
    time: Tensor,
    noisy_rows: Tensor,
    workspace_rows: Tensor,
    low_rows: Tensor | None = None,
    stage_rows: Tensor | None = None,
) -> dict[str, Tensor]:
    """Aggregate condition attention by flow-time bucket as sums and counts."""

    t = time.detach().float().reshape(-1)
    noisy_rows = noisy_rows.detach().float().reshape(-1)
    workspace_rows = workspace_rows.detach().float().reshape(-1)
    low_rows = torch.zeros_like(workspace_rows) if low_rows is None else low_rows.detach().float().reshape(-1)
    stage_rows = torch.zeros_like(workspace_rows) if stage_rows is None else stage_rows.detach().float().reshape(-1)
    out: dict[str, Tensor] = {}
    edges = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1e-6)
    for index in range(3):
        mask = ((t >= edges[index]) & (t < edges[index + 1])).float()
        out[f"mmdit_noisy_attn_t{index}_sum"] = (noisy_rows * mask).sum()
        out[f"mmdit_workspace_attn_t{index}_sum"] = (workspace_rows * mask).sum()
        out[f"mmdit_low_attn_t{index}_sum"] = (low_rows * mask).sum()
        out[f"mmdit_stage_attn_t{index}_sum"] = (stage_rows * mask).sum()
        out[f"mmdit_attn_t{index}_count"] = mask.sum()
    return out
