"""Optimizer ownership derived from module paths, never versioned attributes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..model.component_contracts import legacy_named_parameters, modular_to_legacy_name

ROLE_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("observation", ("observation.",)),
    (
        "grounding",
        (
            "top.grounding_blocks.",
            "top.grounding_content_mod.",
            "top.grounding_content_mod_scale",
        ),
    ),
    ("grounder", ("top.grounder.",)),
    ("intent", ("top.intent.",)),
    ("coarse_action", ("top.coarse_action.",)),
    ("plan_recognizer", ("top.recognizer.",)),
    ("history_proposal", ("history_proposal.",)),
    ("dynamics", ("top.dynamics.",)),
    ("controlled_transition", ("transition.",)),
    (
        "p1_factual",
        (
            "factual_reader.",
            "bottom.p1_time.",
            "bottom.p1_content_mod.",
            "bottom.p1_content_mod_scale",
            "bottom.p1_policy_block.",
        ),
    ),
    ("p2_effect_reader", ("top.effect_reader.",)),
    ("consequence", ("top.consequence.",)),
    ("p3_compiler", ("top.plan_compiler.",)),
    ("v120_canvas_seed", ("bottom.query_encoder.",)),
    ("v120_layer_contracts", ("bottom.layer_contract_heads.",)),
    (
        "bottom_query",
        (
            "bottom.decoder.noisy_lift.",
        ),
    ),
    (
        "bottom_evidence_adapter",
        (
            "bottom.decoder.evidence_adapter.",
        ),
    ),
    (
        "bottom_policy_bridge",
        (
            "bottom.decoder.policy_delta_attnres.",
            "bottom.decoder.protected_detail_basis_attnres.",
        ),
    ),
    (
        "bottom_organizer",
        (
            "bottom.decoder.organizer.",
            "bottom.decoder.intent_seed_norm.",
        ),
    ),
    (
        "bottom_mmdit",
        (
            "bottom.decoder.blocks.",
            "bottom.decoder.horizon_query",
        ),
    ),
    ("bottom_capacity", ("bottom.decoder.operator_contractions.",)),
    ("bottom_execution", ("bottom.decoder.execution_controller.",)),
    (
        "bottom_heads",
        (
            "bottom.decoder.action_norm.",
            "bottom.decoder.velocity_head.",
            # CALVIN's discrete command-state readout is a sibling of the
            # retained physical/motion heads.  Keep it in the same bottom
            # decoder owner so it receives the existing 0.7x LR/decay policy
            # while still being covered exactly once by the optimizer audit.
            "bottom.decoder.gripper_command_head.",
            "bottom.decoder.motion_head.",
        ),
    ),
)

OPTIONAL_ROLE_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bottom_spine", ("bottom.decoder.spine.",)),
)
ALL_ROLE_PREFIXES = (*OPTIONAL_ROLE_PREFIXES, *ROLE_PREFIXES)

BOTTOM_DECODER_ROLES = frozenset(
    {
        "bottom_query",
        "bottom_spine",
        "bottom_evidence_adapter",
        "bottom_policy_bridge",
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
    # The optimizer ABI is intentionally expressed in the frozen legacy
    # logical namespace.  Accepting a final registered path here is useful for
    # diagnostics, but ownership and ordering always resolve through the same
    # explicit map.
    if not name.startswith(("top.", "bottom.", "observation.", "transition.", "history_proposal.", "factual_reader.")):
        try:
            name = modular_to_legacy_name(name)
        except KeyError:
            pass
    for role, prefixes in ALL_ROLE_PREFIXES:
        if any(name.startswith(prefix) for prefix in prefixes):
            return role
    if name.startswith("top.teacher."):
        return "teacher_frozen"
    raise ValueError(f"trainable parameter has no mainline owner: {name}")


def parameter_uses_v120_no_decay(name: str) -> bool:
    """Return the small explicit V120 scale-invariant parameter set.

    Biases and normalization affine parameters are deliberately *not*
    inferred as no-decay from rank or suffix. V120 regular optimizer groups
    inherited AdamW decay; only ordered-contraction bases/depth controls were
    separated because their forward parameterization is scale invariant.
    """

    try:
        name = modular_to_legacy_name(name)
    except KeyError:
        pass
    if not name.startswith("bottom.decoder.operator_contractions."):
        return False
    return name.endswith((".basis_raw", ".depth_weight", ".depth_bias"))


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
    for name, parameter in legacy_named_parameters(model):
        if not parameter.requires_grad:
            frozen.append(name)
            continue
        role = parameter_role(name)
        if id(parameter) in seen:
            raise ValueError(f"optimizer parameter is aliased more than once: {name}")
        seen.add(id(parameter))
        # V120 regular groups use AdamW decay for bias, LayerNorm and all top
        # modules. Only the explicitly named scale-invariant contraction
        # coordinates are no-decay; tensor rank is not an ownership rule.
        decay = not parameter_uses_v120_no_decay(name)
        grouped.setdefault((role, decay), []).append(parameter)
        grouped_names.setdefault((role, decay), []).append(name)
        trainable.append(name)
    expected = {
        id(parameter)
        for _, parameter in legacy_named_parameters(model)
        if parameter.requires_grad
    }
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
    for role, _ in OPTIONAL_ROLE_PREFIXES:
        count = sum(len(grouped.get((role, decay), ())) for decay in (False, True))
        if count:
            role_counts[role] = count
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
    *,
    stage: str,
) -> dict[str, Tensor]:
    """Report one named gradient lifecycle stage by stable role."""

    if stage not in {"raw", "postlocal", "postglobal"}:
        raise ValueError("gradient stage must be raw/postlocal/postglobal")

    named_parameters = legacy_named_parameters(model)
    rows: dict[str, list[Tensor]] = {role: [] for role, _ in ROLE_PREFIXES}
    for role, _ in OPTIONAL_ROLE_PREFIXES:
        if any(
            parameter.requires_grad and parameter_role(name) == role
            for name, parameter in named_parameters
        ):
            rows[role] = []
    spine_branches: dict[str, list[Tensor]] = {"coarse": [], "detail": []}
    for name, parameter in named_parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        role = parameter_role(name)
        rows[role].append(parameter.grad.detach())
        if role == "bottom_spine":
            for branch in spine_branches:
                if name.startswith(f"bottom.decoder.spine.{branch}_lifts."):
                    spine_branches[branch].append(parameter.grad.detach())
    reference = next(model.parameters())
    result: dict[str, Tensor] = {}
    for role, values in rows.items():
        result[f"gradient_{stage}_{role}_l2"] = (
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
    if "bottom_spine" in rows:
        for branch, values in spine_branches.items():
            result[f"gradient_{stage}_bottom_spine_{branch}_l2"] = (
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
    "parameter_uses_v120_no_decay",
    "parameter_role",
    "role_lr_scale",
]
