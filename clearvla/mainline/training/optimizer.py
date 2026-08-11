"""Optimizer ownership derived from module paths, never versioned attributes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import ExperimentConfig

ROLE_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("observation", ("observation.",)),
    ("grounding_host", ("top.grounding_host.",)),
    ("grounder", ("top.grounder.",)),
    ("intent", ("top.intent.",)),
    ("coarse_action", ("top.coarse_action.",)),
    ("plan_recognizer", ("top.recognizer.",)),
    ("history_proposal", ("history_proposal.",)),
    ("dynamics", ("top.dynamics.",)),
    ("controlled_transition", ("transition.",)),
    ("p1_factual", ("factual_reader.",)),
    ("p2_effect_reader", ("top.effect_reader.",)),
    ("consequence", ("top.consequence.",)),
    ("p3_compiler", ("top.plan_compiler.",)),
    ("bottom_query", ("bottom.query_encoder.",)),
    ("bottom_protected_reader", ("bottom.protected_reader.",)),
    ("bottom_evidence_compiler", ("bottom.evidence_compiler.",)),
    ("bottom_organizer", ("bottom.organizer.",)),
    ("bottom_mmdit", ("bottom.blocks.",)),
    ("bottom_capacity", ("bottom.capacity.",)),
    ("bottom_execution", ("bottom.execution.",)),
    (
        "bottom_heads",
        (
            "bottom.final_norm.",
            "bottom.velocity_head.",
            "bottom.event_head.",
            "bottom.motion_head.",
        ),
    ),
)

BOTTOM_DECODER_ROLES = frozenset(
    {
        "bottom_query",
        "bottom_protected_reader",
        "bottom_evidence_compiler",
        "bottom_organizer",
        "bottom_mmdit",
        "bottom_execution",
        "bottom_heads",
    }
)


def role_lr_scale(role: str, config: ExperimentConfig) -> float:
    """Return the source-resolved V120 LR scale for one active owner."""

    optimizer = config.optimizer
    if role == "history_proposal":
        return float(optimizer.history_proposal_lr_scale)
    if role == "bottom_capacity":
        return float(
            optimizer.bottom_decoder_lr_scale
            * optimizer.bottom_capacity_relative_lr_scale
        )
    if role in BOTTOM_DECODER_ROLES:
        return float(optimizer.bottom_decoder_lr_scale)
    return 1.0


def parameter_role(name: str) -> str:
    for role, prefixes in ROLE_PREFIXES:
        if any(name.startswith(prefix) for prefix in prefixes):
            return role
    if name.startswith("top.teacher."):
        return "teacher_frozen"
    raise ValueError(f"trainable parameter has no mainline owner: {name}")


@dataclass(frozen=True)
class OptimizerOwnership:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    role_counts: dict[str, int]
    group_names: tuple[str, ...]


def build_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
) -> tuple[torch.optim.AdamW, OptimizerOwnership]:
    """Put every trainable tensor in exactly one named role/decay group."""

    config.validate()
    grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
    grouped_names: dict[tuple[str, bool], list[str]] = {}
    trainable: list[str] = []
    frozen: list[str] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(name)
            continue
        role = parameter_role(name)
        if id(parameter) in seen:
            raise ValueError(f"optimizer parameter is aliased more than once: {name}")
        seen.add(id(parameter))
        # V120's contraction factor/basis was explicitly no-decay.  QR makes
        # its forward map scale-invariant, but AdamW decay still changes its
        # optimizer moments and therefore its directional learning dynamics.
        decay = (
            parameter.ndim >= 2
            and not name.endswith(".bias")
            and role != "bottom_capacity"
        )
        grouped.setdefault((role, decay), []).append(parameter)
        grouped_names.setdefault((role, decay), []).append(name)
        trainable.append(name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise ValueError("optimizer ownership did not cover every trainable parameter")
    optimizer_groups: list[dict[str, object]] = []
    names: list[str] = []
    for role, decay in sorted(grouped):
        group_name = f"{role}/{'decay' if decay else 'nodecay'}"
        names.append(group_name)
        optimizer_groups.append(
            {
                "params": grouped[(role, decay)],
                "lr": config.optimizer.learning_rate * role_lr_scale(role, config),
                "weight_decay": config.optimizer.weight_decay if decay else 0.0,
                "name": group_name,
                "parameter_names": tuple(grouped_names[(role, decay)]),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.epsilon,
    )
    role_counts = {
        role: sum(len(grouped.get((role, decay), ())) for decay in (False, True))
        for role, _ in ROLE_PREFIXES
    }
    return optimizer, OptimizerOwnership(
        trainable_names=tuple(trainable),
        frozen_names=tuple(frozen),
        role_counts=role_counts,
        group_names=tuple(names),
    )


class WarmupCosineSchedule:
    """Step-owned learning-rate schedule with serializable scalar state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        minimum_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.minimum_ratio = float(minimum_ratio)
        if self.warmup_steps <= 0 or self.total_steps <= 0:
            raise ValueError("schedule steps must be positive")
        if not 0.0 < self.minimum_ratio <= 1.0:
            raise ValueError("minimum LR ratio must be in (0,1]")
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        self.step_index = 0
        self._apply_current_ratio()

    def ratio(self, step: int) -> float:
        step = max(int(step), 0)
        if step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        progress = min(
            max(
                (step - self.warmup_steps) / float(max(self.total_steps - self.warmup_steps, 1)),
                0.0,
            ),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine

    def step(self) -> float:
        self.step_index += 1
        return self._apply_current_ratio()

    def _apply_current_ratio(self) -> float:
        ratio = self.ratio(self.step_index)
        for base, group in zip(self.base_lrs, self.optimizer.param_groups, strict=True):
            group["lr"] = base * ratio
        return ratio

    def state_dict(self) -> dict[str, object]:
        return {"step_index": self.step_index, "base_lrs": self.base_lrs}

    def load_state_dict(self, value: dict[str, object]) -> None:
        raw_base_lrs = value.get("base_lrs")
        raw_step = value.get("step_index")
        if not isinstance(raw_base_lrs, (tuple, list)):
            raise ValueError("scheduler base learning rates are invalid")
        if not isinstance(raw_step, int):
            raise ValueError("scheduler step index is invalid")
        base_lrs = tuple(float(item) for item in raw_base_lrs)
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler optimizer group count changed")
        self.base_lrs = base_lrs
        self.step_index = raw_step
        self._apply_current_ratio()


def gradient_diagnostics(
    model: nn.Module,
) -> dict[str, Tensor]:
    """Report post-clip gradients by stable role."""

    rows: dict[str, list[Tensor]] = {role: [] for role, _ in ROLE_PREFIXES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        rows[parameter_role(name)].append(parameter.grad.detach())
    reference = next(model.parameters())
    result: dict[str, Tensor] = {}
    for role, values in rows.items():
        result[f"gradient_postclip_{role}_l2"] = (
            torch.nn.utils.get_total_norm(
                values,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=True,
            )
            .detach()
            .float()
            if values
            else reference.new_zeros((), dtype=torch.float32)
        )
    return result


__all__ = [
    "OptimizerOwnership",
    "WarmupCosineSchedule",
    "build_optimizer",
    "gradient_diagnostics",
    "parameter_role",
    "role_lr_scale",
]
