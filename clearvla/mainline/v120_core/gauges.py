"""Pure diagnostic reductions shared by policy implementations."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn


@contextmanager
def fp32_diagnostic(value: Tensor) -> Iterator[Tensor]:
    """Yield a detached FP32 tensor outside the active autocast region.

    Diagnostic linear algebra must not inherit the model's AMP dtype or own a
    gradient path. Keeping both rules here prevents local `.float()` calls from
    being silently undone by autocast-enabled operations such as matmul.
    """

    with torch.no_grad(), torch.autocast(device_type=value.device.type, enabled=False):
        yield value.detach().to(dtype=torch.float32)


def masked_categorical_entropy(
    logits: Tensor,
    valid: Tensor,
    *,
    dim: int = -1,
) -> Tensor:
    """Return entropy over valid categories without invalid-logit overflow.

    Filling masked logits with ``finfo.min`` is useful for softmax, but using
    the resulting log probabilities directly in ``p * log(p)`` lets masked
    categories contribute enormous finite values after probability clamping.
    This helper owns the mask and computes the reduction in FP32 so diagnostics
    and auxiliary objectives use the same categorical support.
    """

    if tuple(logits.shape) != tuple(valid.shape):
        raise ValueError(
            "masked categorical entropy expects matching logits and mask, "
            f"got {tuple(logits.shape)} and {tuple(valid.shape)}"
        )
    if logits.ndim < 1:
        raise ValueError("masked categorical entropy expects at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += logits.ndim
    if dim < 0 or dim >= logits.ndim:
        raise ValueError(f"entropy dimension {dim} is invalid for rank {logits.ndim}")

    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.float()
        mask = valid.to(device=logits.device, dtype=torch.bool)
        has_valid = mask.any(dim=dim, keepdim=True)
        # All-invalid rows are replaced by zero logits before log_softmax and
        # reduced back to zero below. This avoids NaNs without inventing a
        # categorical distribution for an inactive row.
        safe_logits = values.masked_fill(~mask, float("-inf"))
        safe_logits = torch.where(has_valid, safe_logits, torch.zeros_like(values))
        log_probability = torch.log_softmax(safe_logits, dim=dim)
        probability = log_probability.exp()
        safe_log_probability = torch.where(
            mask & has_valid,
            log_probability,
            torch.zeros_like(log_probability),
        )
        entropy = -(probability * safe_log_probability).sum(dim=dim)
        return torch.where(has_valid.squeeze(dim), entropy, torch.zeros_like(entropy))


def masked_candidate_center(
    values: Tensor,
    valid: Tensor,
    *,
    candidate_dim: int,
) -> tuple[Tensor, Tensor]:
    """Center candidate fields while keeping invalid candidates exactly zero."""

    if values.ndim < 1:
        raise ValueError("candidate centering expects at least one dimension")
    candidate_dim = int(candidate_dim)
    if candidate_dim < 0:
        candidate_dim += values.ndim
    if candidate_dim < 0 or candidate_dim >= values.ndim:
        raise ValueError(f"candidate dimension {candidate_dim} is invalid for rank {values.ndim}")
    mask = valid.to(device=values.device, dtype=torch.bool)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(values)
    except RuntimeError as error:
        raise ValueError(
            "candidate mask is not broadcastable to values: "
            f"{tuple(valid.shape)} vs {tuple(values.shape)}"
        ) from error
    mask_float = mask.to(dtype=values.dtype)
    count = mask_float.sum(dim=candidate_dim, keepdim=True).clamp_min(1.0)
    mean = (values * mask_float).sum(dim=candidate_dim, keepdim=True) / count
    centered = (values - mean) * mask_float
    return centered, mean


def select_centered_candidate(
    scores: Tensor,
    valid: Tensor,
    *,
    neutral_index: int = 0,
    tie_tolerance: float = 1e-5,
) -> Tensor:
    """Select a candidate after removing common-mode value offsets.

    ``neutral_index`` is the fixed host-operation boundary. It wins exact and
    near ties, while a genuinely lower valid candidate can still be selected.
    Keeping this policy in the gauge utilities prevents native and packaged
    decoders from inventing different tie or masking semantics.
    """
    if scores.ndim != 2 or valid.shape != scores.shape:
        raise ValueError("candidate scores and mask must share [B,C] shape")
    neutral_index = int(neutral_index)
    if neutral_index < 0 or neutral_index >= int(scores.shape[1]):
        raise ValueError("neutral candidate index is outside the candidate axis")
    valid = valid.to(device=scores.device, dtype=torch.bool)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every candidate row needs at least one valid candidate")
    if not bool(valid[:, neutral_index].all()):
        raise ValueError("neutral candidate must be valid for every row")
    masked_sum = scores.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
    masked_count = valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    centered = scores - masked_sum / masked_count
    masked_scores = centered.masked_fill(~valid, float("inf"))
    best_score, best_index = masked_scores.min(dim=-1)
    neutral_score = centered[:, neutral_index]
    return torch.where(
        neutral_score <= best_score + float(tie_tolerance),
        torch.full_like(best_index, neutral_index),
        best_index,
    )


@contextmanager
def deterministic_module_probe(*modules: nn.Module) -> Iterator[None]:
    """Run an auxiliary probe without dropout or host RNG side effects.

    ``torch.no_grad`` removes gradient ownership but does not disable dropout
    and does not restore random-number streams. Candidate probes run between
    recurrent refinement steps, so either leak would change the subsequent
    main path merely because diagnostics were enabled.
    """

    states: list[tuple[nn.Module, bool]] = []
    seen: set[int] = set()
    cuda_devices: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) not in seen:
                seen.add(id(module))
                states.append((module, bool(module.training)))
        for value in (*tuple(root.parameters()), *tuple(root.buffers())):
            if value.device.type == "cuda" and value.device.index is not None:
                cuda_devices.add(int(value.device.index))

    with torch.random.fork_rng(devices=sorted(cuda_devices), enabled=True):
        try:
            for root in modules:
                root.eval()
            yield
        finally:
            # Restore each module directly. Calling train() recursively here
            # would overwrite intentionally mixed child-module modes.
            for module, training in states:
                module.training = training


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
    low_rows = (
        torch.zeros_like(workspace_rows)
        if low_rows is None
        else low_rows.detach().float().reshape(-1)
    )
    stage_rows = (
        torch.zeros_like(workspace_rows)
        if stage_rows is None
        else stage_rows.detach().float().reshape(-1)
    )
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
