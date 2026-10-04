"""First-order binary filtering inside the existing private terminal head.

No discrete decision is fed back into the 24-row plan. Each row propagates both
previous states, so gradients traverse the complete causal recurrence. Zero new
weights reproduce existing logits; no RNG draws or fixed persistence bias.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


def binary_command_filter(logits: Tensor, transition: Tensor, boundary: Tensor) -> tuple[Tensor, Tensor]:
    """Return marginal logits and joint [previous,current] probabilities.

    Transition has unconstrained learned energies for all four pairs. Normalize
    each conditional row FIRST. This is not a CRF backwards smoothing pass;
    late horizon features cannot affect earlier messages via this recurrence.
    Use FP32 log-space arithmetic under autocast; retain all ordinary gradients.
    """
    if logits.ndim != 3 or logits.shape[-1] != 2 or logits.shape[1] < 1:
        raise ValueError("binary command emission must be [B,T,2]")
    if transition.shape != (*logits.shape, 2) or boundary.shape != (logits.shape[0], 2):
        raise ValueError("command transition or source boundary lost class/time axes")
    if any(v.device != logits.device or not v.is_floating_point() for v in (transition, boundary)):
        raise ValueError("command law tensors must be floating point on one device")
    # Boundary is validated once by the typed online-source factory. These
    # shape-only checks are reused at the numerical solver's terminal calls.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        raw, energy = logits.float(), transition.float()
        initial = torch.where(boundary > 0, boundary.float().clamp_min(torch.finfo(torch.float32).tiny).log(), -torch.inf)
        conditional_rows = raw[:, :, None, :] + energy
        normalizers = torch.logsumexp(conditional_rows, -1, keepdim=True)
        deltas = normalizers - torch.logsumexp(raw, -1)[:, :, None, None]
        log_ratios = energy - deltas
        conditional_log = conditional_rows - normalizers
        previous = initial
        rows, pairs = [], []
        for row in range(raw.shape[1]):
            emission = raw[:, row]
            # Ratio of each conditional law to the unconditioned emission law.
            # If e==0, the normalizers match and BOTH class corrections are
            # exactly equal; centering below gives bitwise zero, not a branch.
            correction = torch.logsumexp(previous[:, :, None] + log_ratios[:, row], dim=1)
            correction = correction - correction.mean(-1, keepdim=True)
            marginal_logits = emission + correction
            rows.append(marginal_logits)
            pairs.append((previous[:, :, None] + conditional_log[:, row]).exp())
            previous = torch.log_softmax(marginal_logits, -1)
        return torch.stack(rows, 1), torch.stack(pairs, 1)


class ConditionalBinaryCommand(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.transition_weight = nn.Parameter(torch.zeros(hidden, 4))

    def forward(self, state: Tensor, logits: Tensor, boundary: Tensor) -> tuple[Tensor, Tensor]:
        if state.shape != (*logits.shape[:2], self.hidden):
            raise ValueError("conditional command lost private gripper-state source")
        # A zero input cannot manufacture a transition preference. Finite source
        # validation remains owned by the model input and existing finalizer.
        transition = torch.matmul(state.to(self.transition_weight.dtype), self.transition_weight)
        return binary_command_filter(logits, transition.reshape(*logits.shape[:2], 2, 2), boundary)
