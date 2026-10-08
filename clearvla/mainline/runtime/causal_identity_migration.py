"""Narrow, named migration from the admitted mature CALVIN identity run."""

import torch

from ..causal_identity import MODES

CAUSAL_IDENTITY_AB_V1 = "causal_identity_ab_v1"
CAUSAL_UNIFIED_SOURCE_V1 = "causal_unified_source_v1"
CAUSAL_INITIALIZATION_MODES = frozenset({CAUSAL_IDENTITY_AB_V1, CAUSAL_UNIFIED_SOURCE_V1})
SOURCE_DIGEST = "df3fd978ba754058943859a333cafd671871684c3cd4e6f55bf92576831d3f66"

SOURCE_PATHS = frozenset(
    """
clearvla/benchmarks/calvin_raw.py
clearvla/mainline/checkpoint.py
clearvla/mainline/config.py
clearvla/mainline/causal_identity.py
clearvla/mainline/data/loading.py
clearvla/mainline/data/online_visual.py
clearvla/mainline/data/identity_correspondence.py
clearvla/mainline/executed_world.py
clearvla/mainline/instruction_change.py
clearvla/mainline/identity_supervision.py
clearvla/mainline/interfaces.py
clearvla/mainline/model/dynamics.py
clearvla/mainline/model/executed_world.py
clearvla/mainline/model/grounding.py
clearvla/mainline/model/canonical_grounding.py
clearvla/mainline/model/instruction_posterior.py
clearvla/mainline/model/intent.py
clearvla/mainline/model/observation_association.py
clearvla/mainline/model/observation_contract.py
clearvla/mainline/model/observed_outcome.py
clearvla/mainline/model/policy.py
clearvla/mainline/model/restored_observation.py
clearvla/mainline/model/source_measurement.py
clearvla/mainline/model/task_execution.py
clearvla/mainline/model/teacher.py
clearvla/mainline/model/top.py
clearvla/mainline/model/types.py
clearvla/mainline/runtime/checkpoints.py
clearvla/mainline/runtime/causal_identity_migration.py
clearvla/mainline/runtime/deployment.py
clearvla/mainline/runtime/identity.py
clearvla/mainline/train.py
clearvla/mainline/training/engine.py
clearvla/mainline/training/identity.py
clearvla/mainline/training/optimizer.py
clearvla/vision/entity_chart.py
clearvla/vision/canonical_transport.py
clearvla/vision/log_transport.py
clearvla/vision/observed_correspondence.py
clearvla/vision/observed_flow.py
clearvla/vision/sensor_geometry.py
clearvla/mainline/assets/calvin_rgbd_joint_geometry_v1.json
""".split()
)


# A different source contract, not a silent extension of the admitted A/B v1.
# It uses the same explicitly checked source checkpoint and tensor migration;
# source-only boundary fixes add no policy parameters or training-only inputs.
UNIFIED_SOURCE_PATHS = SOURCE_PATHS | frozenset(
    """
clearvla/data/action_chart.py
clearvla/data/hdf5_episode.py
clearvla/data/native_contract.py
clearvla/data/physical_chart.py
clearvla/mainline/identity_pairs.py
clearvla/mainline/robot_execution.py
clearvla/mainline/model/robot_execution.py
clearvla/mainline/v120_core/codec.py
clearvla/mainline/v120_core/controller.py
clearvla/mainline/v120_core/decoder.py
clearvla/mainline/v120_core/evidence.py
clearvla/mainline/v120_core/intent.py
clearvla/mainline/v120_core/trunk_primitives.py
clearvla/vision/source_weights.py
""".split()
)


def allowed_source_paths(mode: str) -> frozenset[str]:
    if mode == CAUSAL_IDENTITY_AB_V1:
        return SOURCE_PATHS
    if mode == CAUSAL_UNIFIED_SOURCE_V1:
        return UNIFIED_SOURCE_PATHS
    raise ValueError("unknown causal initialization source contract")


def config_view(payload):
    # The caller already removed run location/optimizer/diagnostic budgets.
    result = {**payload, "top": dict(payload["top"]), "objectives": dict(payload["objectives"])}
    for name in MODES:
        result["top"].pop(name, None)
    for name in (
        "identity_correspondence",
        "identity_source_prediction",
        "gripper_command_transition",
        "calvin_frame_weight_mode",
        "calvin_frame_motion_gain",
        "calvin_frame_event_gain",
        "calvin_frame_event_radius",
        "calvin_frame_max_weight",
    ):
        result["objectives"].pop(name, None)
    return result


def validate_selection(saved, current, source_digest):
    if source_digest != SOURCE_DIGEST:
        raise ValueError("causal identity migration requires the admitted a2d597d2 source identity")
    if (
        saved.data.data_profile != "calvin_relative_7d_v1"
        or current.data.data_profile != saved.data.data_profile
    ):
        raise ValueError("causal identity migration is CALVIN-only")
    for name, pair in MODES.items():
        if getattr(saved.top, name) != pair[0]:
            raise ValueError("causal identity source already changes " + name)
    for name in (
        "entity_transport_gradient_mode",
        "entity_competition_scale_mode",
        "observation_measurement_mode",
        "target_binding_input_mode",
        "observed_outcome_mode",
    ):
        if getattr(current.top, name) != MODES[name][1]:
            raise ValueError("incomplete confirmed repair: " + name)
    is_b = current.top.entity_ownership_mode == "canonical_image_v1"
    if (
        current.top.identity_supervision_mode
        in {"rgbd_temporal_v1", "rgbd_temporal_conditional_v2"}
    ) != is_b:
        raise ValueError("B must include physical correspondence supervision, A must exclude it")
    if (
        current.objectives.calvin_frame_weight_mode != "motion_event_v1"
        or current.objectives.gripper_command_transition <= 0
    ):
        raise ValueError("causal repair must retain admitted trajectory objectives")


def migrate_state(saved, current, config):
    added = {
        f"intent.organizer.observed_outcome.{name}.weight"
        for name in ("image", "output", "semantic", "status", "view")
    }
    removed = {
        f"intent.organizer.instruction_progress.{name}"
        for name in (
            "key.weight",
            "key_position.weight",
            "null_key",
            "query.weight",
            "query_position.weight",
        )
    }
    if config.top.entity_ownership_mode == "canonical_image_v1":
        added |= {
            f"grounding.grounder.canonical_decoder.{name}.weight"
            for name in ("background", "identity", "output", "view")
        }
        added |= {
            "intent.organizer.object_identity.weight",
            "world.dynamics.object_identity.weight",
            "world.dynamics.observed_view_content.weight",
            "world.dynamics.observed_view_role.weight",
        }
        removed.add("grounding.grounder.decode_content_residual.weight")
    if set(current) - set(saved) != added or set(saved) - set(current) != removed:
        raise ValueError(
            "causal identity state inventory differs from its explicit add/remove contract"
        )
    if torch.count_nonzero(current["intent.organizer.observed_outcome.output.weight"]).item() != 0:
        raise ValueError("new observed-outcome consumer must initialize neutral")
    return {
        name: (current[name].detach().clone() if name in added else saved[name]) for name in current
    }
