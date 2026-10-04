"""A native causal binary command distribution, not a physical holding rule."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .robot_execution import ExecutedRobotStep

INDEPENDENT_COMMANDS = "independent_rows_v1"
CONDITIONAL_COMMANDS = "conditional_binary_chain_v1"


def validate_command_sequence_mode(mode: str) -> None:
    if mode not in {INDEPENDENT_COMMANDS, CONDITIONAL_COMMANDS}:
        raise ValueError("unknown binary command sequence mode")


def command_sequence_metadata() -> dict[str, object]:
    return {
        "schema": "native-conditional-binary-command-v1",
        "classes": [-1, 1],
        "source": "existing-private-gripper-state-and-recorded-one-step-command",
        "boundary": "inverse-checkpoint-affine-chart;observed-command-onehot;absent-uniform",
        "law": "P(g_t|g_prev,h_t)=softmax(emission_t+learned_transition_t)",
        "inference": "causal-forward-marginals;native-argmax-once-no-Viterbi-or-hold-rule",
        "training": "existing-command-CE-and-selected-objectives-on-same-marginals",
        "initialization": "zero-transition-energy;exact-independent-logit-neutrality",
        "lifetime": "fresh-filter-per-terminal-call;observed-boundary-once-per-online-encoding",
        "excludes": ["future-command-labels", "predicted-unexecuted-tail-as-boundary", "contact-threshold", "dwell-timer", "new-loss"],
        "limits": "learned-first-order-law;no-guarantee-of-persistence-or-correct-contact;no-cross-call-filter-state",
    }


@dataclass(frozen=True)
class CommandChainBoundary:
    probability: Tensor  # [B,2], native negative/positive classes
    source: ExecutedRobotStep
    normalizer_fingerprint: str
    versions: tuple[int, int, int, int]

    def validate(self, source: ExecutedRobotStep | None, fingerprint: str | None = None) -> None:
        if source is not self.source:
            raise ValueError("command chain boundary belongs to another executed source")
        p = self.probability
        if p.shape != (self.source.command.shape[0], 2) or p.device != self.source.command.device or p.dtype != torch.float32:
            raise ValueError("command chain boundary lost source batch/device/FP32 probability")
        current = tuple(v._version for v in (self.source.command, self.source.observed, self.source.offsets, p))
        if current != self.versions:
            raise ValueError("command chain source or boundary was mutated after preparation")
        if fingerprint is not None and fingerprint != self.normalizer_fingerprint:
            raise ValueError("command chain boundary belongs to another action normalizer")


def prepare_command_boundary(step: ExecutedRobotStep, *, offset: Tensor, scale: Tensor,
                             fingerprint: str) -> CommandChainBoundary:
    step.validate(batch=step.command.shape[0], state_dim=step.previous_state.shape[-1],
                  action_dim=step.command.shape[-1], device=step.command.device, strict=True)
    if offset.numel() != 1 or scale.numel() != 1 or offset.device != step.command.device or scale.device != step.command.device:
        raise ValueError("command boundary needs the outlet's native gripper affine chart")
    if not bool(torch.isfinite(offset).all() and torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("command boundary affine chart must be finite and positive")
    command = torch.where(step.observed, step.command[:, -1].float(), offset.reshape(()))
    native = (command - offset.reshape(())) / scale.reshape(())
    if bool((step.observed & ((native.abs() - 1).abs() > 1e-5)).any()):
        raise ValueError("observed command boundary is not the native binary alphabet")
    positive = (native >= 0).float()
    p = torch.stack((1-positive, positive), -1)
    p = torch.where(step.observed[:, None], p, 0.5)
    versions = tuple(v._version for v in (step.command, step.observed, step.offsets, p))
    result = CommandChainBoundary(p, step, fingerprint, versions)
    result.validate(step, fingerprint)
    return result
