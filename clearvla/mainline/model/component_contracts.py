"""Typed boundaries and stable name mapping for the modular mainline graph.

The dataclasses in this module are reference-only contracts: construction does
not clone, detach, cast, reshape, reduce, or otherwise transform a tensor.  The
name map is the single checkpoint/optimizer bridge from the pre-modular
Schema30 hierarchy to the registered component hierarchy.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

from torch import Tensor, nn

from ..v120_core.bspine import (
    BSPINE0_IMPLEMENTATION,
    BSPINE_DISABLED_IMPLEMENTATION,
)
from .action_contract import V120SeedContext
from .compiler import ObjectPolicyPlanDeltaBank
from .observation_contract import ObservationEvidence
from .types import (
    CompletedP1PolicyState,
    ControlledTransitionState,
    FactualPrecisionDock,
    ObjectFactSet,
)

if TYPE_CHECKING:
    from ..config import ExperimentConfig


COMPONENT_ABI_REVISION = "mainline-modular-v1"
BASELINE_EXECUTION_BOTTOM = "v120_evidence_mmdit_v1"
BSPINE0_EXECUTION_BOTTOM = "v120_evidence_mmdit_bspine0_v1"


def _execution_bottom_selection(config: "ExperimentConfig") -> str:
    implementation = str(config.bottom.bspine_implementation)
    if implementation == BSPINE_DISABLED_IMPLEMENTATION:
        return BASELINE_EXECUTION_BOTTOM
    if implementation == BSPINE0_IMPLEMENTATION:
        return BSPINE0_EXECUTION_BOTTOM
    raise ValueError(f"unsupported B-spine implementation: {implementation!r}")


@dataclass(frozen=True)
class ComponentSelection:
    """Exactly one implementation selected for every replaceable slot."""

    conditioning: str = "observable_history_v1"
    observation: str = "restored_v120_observation_v1"
    role_query_bridge: str = "v120_shared_role_query_v1"
    grounding: str = "progressive_g123_dense_v1"
    intent: str = "stateless_object_intent_v1"
    world: str = "object_candidate_w12_v1"
    p1: str = "v120_factual_dynamic_p1_v1"
    policy_compiler: str = "object_p2_p3_v1"
    transition: str = "controlled_transition_v1"
    execution_bottom: str = BASELINE_EXECUTION_BOTTOM
    terminal_controller: str = "continuous_physical_v1"
    outlet_adapter: str = "pen_7d_continuous_v1"
    objectives: str = "schema30_objectives_v1"
    abi_revision: str = COMPONENT_ABI_REVISION

    @classmethod
    def from_config(cls, config: "ExperimentConfig") -> "ComponentSelection":
        """Resolve outlet/terminal identity before constructing any module."""

        profile = str(config.data.data_profile)
        output_mode = str(config.bottom.gripper_output_mode)
        if profile == "calvin_relative_7d_v1":
            terminal = "calvin_binary_command_v1"
            outlet = "calvin_7d_binary_v1"
        elif profile == "rdt_right_arm_action_chart_v1":
            terminal = "continuous_physical_v1"
            outlet = "rdt_right_arm_7d_v1"
        else:
            terminal = "continuous_physical_v1"
            outlet = "pen_7d_continuous_v1"
        selection = cls(
            execution_bottom=_execution_bottom_selection(config),
            terminal_controller=terminal,
            outlet_adapter=outlet,
        )
        selection.validate(config)
        if (output_mode == "calvin_binary_command") != (
            selection.terminal_controller == "calvin_binary_command_v1"
        ):
            raise ValueError("terminal selection does not match the configured outlet")
        return selection

    def validate(self, config: "ExperimentConfig") -> None:
        """Reject incompatible compositions without instantiating candidates."""

        config.validate()
        expected = type(self).from_config_without_validation(config)
        if self != expected:
            differing = tuple(
                name
                for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(expected, name)
            )
            raise ValueError(
                "component selection is incompatible with the resolved baseline: "
                + ", ".join(differing)
            )
        dims = config.dimensions
        obs = config.observation
        if (
            int(dims.action_horizon) != 24
            or int(config.top.object_slots) != 4
            or int(dims.num_cameras) != 2
            or int(obs.grid_size) != 8
            or int(dims.action_dim) != 7
            or int(config.bottom.gripper_field_dim) != 6
        ):
            raise ValueError("selected component ABI requires the complete Schema30 axes")
        if int(dims.action_basis_tokens) <= 0:
            raise ValueError("selected component ABI requires a positive basis count")

    @classmethod
    def from_config_without_validation(
        cls, config: "ExperimentConfig"
    ) -> "ComponentSelection":
        profile = str(config.data.data_profile)
        if profile == "calvin_relative_7d_v1":
            return cls(
                execution_bottom=_execution_bottom_selection(config),
                terminal_controller="calvin_binary_command_v1",
                outlet_adapter="calvin_7d_binary_v1",
            )
        if profile == "rdt_right_arm_action_chart_v1":
            return cls(
                execution_bottom=_execution_bottom_selection(config),
                outlet_adapter="rdt_right_arm_7d_v1",
            )
        return cls(execution_bottom=_execution_bottom_selection(config))

    def as_dict(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        config: ExperimentConfig | None = None,
    ) -> "ComponentSelection":
        """Restore one complete serialized selection without silent defaults."""

        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        if actual != expected:
            missing = tuple(sorted(expected.difference(actual)))
            unexpected = tuple(sorted(actual.difference(expected)))
            raise ValueError(
                "component selection fields differ: "
                f"missing={missing} unexpected={unexpected}"
            )
        selection = cls(
            **{name: str(value[name]) for name in cls.__dataclass_fields__}
        )
        if config is not None:
            selection.validate(config)
        elif selection.abi_revision != COMPONENT_ABI_REVISION:
            raise ValueError("component selection ABI revision is incompatible")
        return selection


@dataclass(frozen=True)
class SharedRoleContext:
    role_table: Tensor


@dataclass(frozen=True)
class GroundingSeed:
    canvas: Tensor
    slices: Mapping[str, slice]


@dataclass(frozen=True)
class GroundedObservationBundle:
    evidence: ObservationEvidence
    facts: ObjectFactSet
    g3_rollout: Tensor
    role_context: SharedRoleContext


@dataclass(frozen=True)
class DynamicQueryBundle:
    model_field: Tensor
    action_query: Tensor
    seed: V120SeedContext


@dataclass(frozen=True)
class PolicyCompileTrace:
    effect: object
    consequence: object


@dataclass(frozen=True)
class PolicyCompileResult:
    execution_plan: ObjectPolicyPlanDeltaBank
    trace: PolicyCompileTrace


@dataclass(frozen=True)
class TerminalHeadOutput:
    physical_velocity: Tensor
    motion_logits: Tensor
    command_logits: Tensor | None
    event_logits: Tensor | None
    diagnostics: Mapping[str, Tensor]


@dataclass(frozen=True)
class OutletActionOutput:
    deployed_action: Tensor
    world_condition_action: Tensor
    continuous_action: Tensor | None
    command_logits: Tensor | None
    command: Tensor | None


class P1StageContract(Protocol):
    def update_dynamic(
        self,
        *,
        action_query: Tensor,
        factual: FactualPrecisionDock,
        time: Tensor,
        collect_diagnostics: bool,
    ) -> tuple[CompletedP1PolicyState, dict[str, Tensor]]: ...


class ExecutionBottomStageContract(Protocol):
    def step(
        self,
        *,
        model_field: Tensor,
        time: Tensor,
        action_query: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        seed: V120SeedContext,
        transition: ControlledTransitionState,
        execution_mode: str,
        require_execution_supervision: bool,
        collect_diagnostics: bool,
    ): ...


# Longest/specialized prefixes must precede their containing decoder prefix.
MODULAR_TO_LEGACY_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "execution_bottom.decoder.terminal_controller.optional_command_head.",
        "bottom.decoder.gripper_command_head.",
    ),
    (
        "execution_bottom.decoder.terminal_controller.optional_event_head.",
        "bottom.decoder.event_head.",
    ),
    (
        "execution_bottom.decoder.terminal_controller.velocity_head.",
        "bottom.decoder.velocity_head.",
    ),
    (
        "execution_bottom.decoder.terminal_controller.motion_head.",
        "bottom.decoder.motion_head.",
    ),
    (
        "execution_bottom.decoder.terminal_controller.action_norm.",
        "bottom.decoder.action_norm.",
    ),
    ("conditioning.history_proposal.", "history_proposal."),
    ("observation.compiler.", "observation."),
    ("outlet_adapter.codec.", "action_codec."),
    ("bridge.query_encoder.", "bottom.query_encoder."),
    ("grounding.blocks.", "top.grounding_blocks."),
    ("grounding.content_mod.", "top.grounding_content_mod."),
    ("grounding.content_mod_scale", "top.grounding_content_mod_scale"),
    ("grounding.grounder.", "top.grounder."),
    ("intent.organizer.", "top.intent."),
    ("intent.coarse_action.", "top.coarse_action."),
    ("world.dynamics.", "top.dynamics."),
    ("training_targets.teacher.", "top.teacher."),
    ("training_targets.recognizer.", "top.recognizer."),
    ("p1.factual_reader.", "factual_reader."),
    ("p1.dynamic_time.", "bottom.p1_time."),
    ("p1.dynamic_content_mod.", "bottom.p1_content_mod."),
    ("p1.dynamic_content_mod_scale", "bottom.p1_content_mod_scale"),
    ("p1.dynamic_policy_block.", "bottom.p1_policy_block."),
    ("policy_compiler.effect_reader.", "top.effect_reader."),
    ("policy_compiler.consequence.", "top.consequence."),
    ("policy_compiler.plan_compiler.", "top.plan_compiler."),
    ("execution_bottom.layer_contract_heads.", "bottom.layer_contract_heads."),
    ("execution_bottom.decoder.", "bottom.decoder."),
)


def modular_to_legacy_name(name: str) -> str:
    """Map one final registered path to its unique pre-modular logical path."""

    for modular, legacy in MODULAR_TO_LEGACY_PREFIXES:
        if name == modular.rstrip(".") or name.startswith(modular):
            suffix = name[len(modular) :] if name.startswith(modular) else ""
            return legacy + suffix
    if name.startswith("transition."):
        return name
    raise KeyError(f"modular parameter/state path has no legacy map: {name}")


def legacy_named_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return parameters in the frozen pre-modular traversal order."""

    final = dict(model.named_parameters())
    ledger = getattr(model, "_legacy_parameter_order", None)
    if ledger is None:
        return tuple(model.named_parameters())
    result: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for new_name, legacy_name in ledger:
        try:
            parameter = final[new_name]
        except KeyError as error:
            raise RuntimeError(
                f"legacy parameter ledger lost final owner {new_name!r}"
            ) from error
        if id(parameter) in seen:
            raise RuntimeError(f"legacy parameter ledger duplicates {legacy_name!r}")
        seen.add(id(parameter))
        result.append((legacy_name, parameter))
    if seen != {id(parameter) for parameter in final.values()}:
        raise RuntimeError("legacy parameter ledger does not cover the final model")
    return tuple(result)


def legacy_state_dict(model: nn.Module) -> "OrderedDict[str, Tensor]":
    """Expose current tensors under the checked pre-modular logical names."""

    result: OrderedDict[str, Tensor] = OrderedDict()
    for name, value in model.state_dict().items():
        legacy = modular_to_legacy_name(name)
        if legacy in result:
            raise RuntimeError(f"legacy state map collision at {legacy!r}")
        result[legacy] = value
    return result


def map_legacy_state_dict(
    model: nn.Module, state: Mapping[str, Tensor]
) -> "OrderedDict[str, Tensor]":
    """Validate and convert one complete legacy state mapping for this model."""

    expected = model.state_dict()
    inverse: dict[str, str] = {}
    for new_name in expected:
        legacy = modular_to_legacy_name(new_name)
        if legacy in inverse:
            raise RuntimeError(f"legacy state destination collision at {legacy!r}")
        inverse[legacy] = new_name
    missing = tuple(name for name in inverse if name not in state)
    unexpected = tuple(name for name in state if name not in inverse)
    if missing or unexpected:
        raise ValueError(
            "legacy state ownership differs: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    result: OrderedDict[str, Tensor] = OrderedDict()
    for legacy, new_name in inverse.items():
        value = state[legacy]
        target = expected[new_name]
        if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
            raise ValueError(f"legacy state tensor contract differs at {legacy!r}")
        result[new_name] = value
    return result


__all__ = [
    "COMPONENT_ABI_REVISION",
    "ComponentSelection",
    "DynamicQueryBundle",
    "ExecutionBottomStageContract",
    "GroundedObservationBundle",
    "GroundingSeed",
    "MODULAR_TO_LEGACY_PREFIXES",
    "OutletActionOutput",
    "P1StageContract",
    "PolicyCompileResult",
    "PolicyCompileTrace",
    "SharedRoleContext",
    "TerminalHeadOutput",
    "legacy_named_parameters",
    "legacy_state_dict",
    "map_legacy_state_dict",
    "modular_to_legacy_name",
]
