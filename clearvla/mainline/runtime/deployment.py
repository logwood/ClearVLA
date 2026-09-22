"""Stable deployment ABI exported by formal mainline checkpoints.

The training config remains the exact-resume identity.  Deployment consumes a
smaller boundary contract so paths, split mechanisms, worker counts and future
data-only fields cannot make a trained graph unparsable.  Model-owning config
sections remain strict and the state dict is still loaded with exact ownership.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Mapping, cast

import numpy as np

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.state_features import NATIVE_AFFINE_STATE, state_feature_metadata
from clearvla.vision.candidate_support import MOMENT_LOCAL_SUPPORT, candidate_support_metadata
from clearvla.vision.entity_chart import CURRENT_IMAGE_CHART, QUERY_CHART, entity_chart_metadata
from clearvla.vision.local_ownership import INDEPENDENT_LOCAL_OWNERS, local_ownership_metadata
from clearvla.vision.preprocessing import (
    PreprocessConfig,
    preprocessing_identity,
)
from clearvla.vision.source_time import FIXED_VISUAL_TIME, visual_time_metadata

from ..checkpoint import CheckpointIdentity
from ..config import DataConfig, ExperimentConfig, config_from_mapping
from ..data.normalizer import ArrayNormalizer
from ..gripper_contract import (
    CALVIN_BINARY_GRIPPER_OUTPUT_MODE,
    CONTINUOUS_GRIPPER_OUTPUT_MODE,
    MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
    VALID_GRIPPER_OUTPUT_MODES,
)
from ..temporal import HISTORY_TIMING_CONTRACT, TIMED_HISTORY_ENCODING
from .flow_schedule import DeploymentFlowSchedule

DEPLOYMENT_ABI_SCHEMA = "clearvla-mainline-deployment-abi-v1"
CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE = (
    "profile_owned_full_horizon_encode_decode_loss_evaluation"
)

# LIBERO is an outlet contract, not a second shared action codec.  These
# constants are kept here only so the deployment ABI can reject a profile that
# has been relabelled or projected to a different native chart before a model
# is constructed.
_LIBERO_PROFILE_NAME = "libero_relative_7d_v1"
_LIBERO_ACTION_DIM = 7
_LIBERO_STATE_DIM = 7
_LIBERO_ACTION_HORIZON = 24
_LIBERO_GRIPPER_INDICES = (6,)
_LIBERO_ACTION_INDICES = tuple(range(_LIBERO_ACTION_DIM))
_LIBERO_ACTION_CHART = "libero_normalized_osc_pose_6d_plus_continuous_gripper"
_LIBERO_STATE_CHART = "libero_eef_6d_plus_gripper_opening_width"

_GRAPH_SECTIONS = (
    "dimensions",
    "observation",
    "top",
    "bottom",
    "objectives",
    "runtime",
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deployment_graph_config(config: ExperimentConfig) -> dict[str, object]:
    payload = config.as_dict()
    return {name: payload[name] for name in _GRAPH_SECTIONS}


def deployment_flow_schedule(config: ExperimentConfig) -> DeploymentFlowSchedule:
    """Resolve the two-pass numerical contract without changing training time."""

    raw = config.runtime.deployment_flow_schedule
    return (
        DeploymentFlowSchedule.uniform_five()
        if raw is None
        else DeploymentFlowSchedule.from_dict(raw)
    )


def _strict_abi_int(value: object, *, name: str) -> int:
    """Parse an ABI integer without accepting booleans or lossy floats."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"deployment ABI {name} must be an integer")
    return int(value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"deployment ABI {name} must be a sequence")
    return tuple(value)


def _strict_index_sequence(value: object, *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for index, item in enumerate(_sequence(value, name=name)):
        result.append(_strict_abi_int(item, name=f"{name}[{index}]"))
    return tuple(result)


def _validate_libero_profile_metadata(
    profile: Mapping[str, object],
    *,
    gripper_indices: object,
    name: str,
) -> None:
    """Validate the complete native LIBERO profile at an ABI boundary."""

    if profile.get("name") != _LIBERO_PROFILE_NAME:
        raise ValueError(f"{name} must identify {_LIBERO_PROFILE_NAME}")
    try:
        registered = resolve_action_state_profile(_LIBERO_PROFILE_NAME)
        expected_digest = registered.digest()
    except (KeyError, ValueError) as error:
        raise ValueError("registered LIBERO action profile is unavailable") from error
    if profile.get("sha256") != expected_digest:
        raise ValueError(f"{name} digest differs from the registered LIBERO profile")
    if (
        "gripper_open_direction" in profile
        and profile.get("gripper_open_direction")
        != registered.gripper_open_direction
    ):
        raise ValueError(f"{name}.gripper_open_direction differs from the LIBERO contract")
    for field, expected in (
        ("source_action_dim", _LIBERO_ACTION_DIM),
        ("source_state_dim", _LIBERO_STATE_DIM),
        ("action_chart", _LIBERO_ACTION_CHART),
        ("state_chart", _LIBERO_STATE_CHART),
        ("gripper_transition_boundary", "previous_command"),
    ):
        if field in {"source_action_dim", "source_state_dim"}:
            actual = _strict_abi_int(profile.get(field), name=f"{name}.{field}")
        else:
            actual = profile.get(field)
        if actual != expected:
            raise ValueError(
                f"{name}.{field}={actual!r} does not match LIBERO contract {expected!r}"
            )
    for field in ("action_indices", "state_indices"):
        indices = _strict_index_sequence(profile.get(field), name=f"{name}.{field}")
        if indices != _LIBERO_ACTION_INDICES:
            raise ValueError(f"{name}.{field} does not cover native LIBERO 7D order")
    profile_grippers = _strict_index_sequence(
        profile.get("gripper_indices"), name=f"{name}.gripper_indices"
    )
    if profile_grippers != _LIBERO_GRIPPER_INDICES:
        raise ValueError(f"{name}.gripper_indices must be [6]")
    action_grippers = _strict_index_sequence(
        gripper_indices, name="deployment action gripper_indices"
    )
    if action_grippers != _LIBERO_GRIPPER_INDICES:
        raise ValueError("deployment action gripper_indices must be [6] for LIBERO")
    scales = _sequence(
        profile.get("state_to_action_scale"),
        name=f"{name}.state_to_action_scale",
    )
    if len(scales) != _LIBERO_ACTION_DIM:
        raise ValueError(f"{name}.state_to_action_scale must have seven entries")
    try:
        numeric_scales = tuple(float(item) for item in scales)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}.state_to_action_scale is not numeric") from error
    if numeric_scales != (1.0,) * _LIBERO_ACTION_DIM:
        raise ValueError(f"{name}.state_to_action_scale must be all ones")


def _validate_maniskill_profile(profile: Mapping[str, object]) -> None:
    name = str(profile.get("name", ""))
    if name not in {"maniskill_pd_ee_delta_pose_7d_v1", "maniskill_pd_ee_delta_pose_7d_v2"}:
        raise ValueError("unknown ManiSkill profile identity")
    registered = resolve_action_state_profile(name)
    expected = registered.as_dict()
    actual = {key: profile.get(key) for key in expected}
    if (canonical_sha256(actual) != canonical_sha256(expected)
            or profile.get("sha256") != registered.digest()
            or profile.get("gripper_transition_boundary") != "previous_command"):
        raise ValueError("deployment ManiSkill native profile identity differs")
    if (
        "gripper_open_direction" in profile
        and profile.get("gripper_open_direction")
        != registered.gripper_open_direction
    ):
        raise ValueError("deployment ManiSkill gripper open direction differs")


def _validate_normalizer_object(normalizer: object, *, name: str, width: int) -> None:
    """Reject malformed normalizers before their hashes enter the ABI."""

    if int(width) <= 0:
        raise ValueError(f"deployment {name} normalizer width must be positive")
    if not hasattr(normalizer, "mode"):
        raise ValueError(f"deployment {name} normalizer is malformed")
    if str(getattr(normalizer, "mode")) != "zscore":
        raise ValueError(f"deployment {name} normalizer must use zscore")
    for field in ("offset", "scale", "mean", "std", "minimum", "maximum"):
        value = getattr(normalizer, field, None)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"deployment {name} normalizer {field} is not numeric"
            ) from error
        if array.shape != (1, int(width)):
            raise ValueError(
                f"deployment {name} normalizer {field} must own [1,{width}] rows"
            )
        try:
            finite = np.isfinite(array)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"deployment {name} normalizer {field} is not numeric"
            ) from error
        if not bool(finite.all()):
            raise ValueError(
                f"deployment {name} normalizer {field} contains NaN or infinity"
            )
        if field == "scale" and bool(np.any(array <= 0.0)):
            raise ValueError(
                f"deployment {name} normalizer scale must be strictly positive"
            )


def build_deployment_abi(
    config: ExperimentConfig,
    identity: CheckpointIdentity,
    *,
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    data_profile: Mapping[str, object],
    gripper_indices: tuple[int, ...],
    goal_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build the checkpoint-owned inference boundary without runtime caches."""

    config.validate()
    identity.validate()
    if not isinstance(data_profile, Mapping):
        raise ValueError("deployment data profile must be a mapping")
    profile = {str(key): value for key, value in data_profile.items()}
    profile_name = profile.get("name")
    if profile_name != config.data.data_profile:
        raise ValueError(
            "deployment data profile differs from config: "
            f"{profile_name!r} != {config.data.data_profile!r}"
        )
    if profile_name in {"maniskill_pd_ee_delta_pose_7d_v1", "maniskill_pd_ee_delta_pose_7d_v2"}:
        _validate_maniskill_profile(profile)
    gripper_codec_boundary = str(profile.get("gripper_transition_boundary", ""))
    if gripper_codec_boundary not in {
        "current_action_state",
        "previous_command",
    }:
        raise ValueError("deployment data profile lacks a gripper codec boundary")
    if profile_name == _LIBERO_PROFILE_NAME:
        _validate_libero_profile_metadata(
            profile,
            gripper_indices=gripper_indices,
            name="deployment data profile",
        )
    else:
        # Every profile still has to expose a coherent gripper projection; the
        # LIBERO branch above adds the outlet-specific exactness checks.
        profile_grippers = _strict_index_sequence(
            profile.get("gripper_indices"),
            name="deployment data profile.gripper_indices",
        )
        action_grippers = _strict_index_sequence(
            gripper_indices,
            name="deployment action gripper_indices",
        )
        if profile_grippers != action_grippers:
            raise ValueError(
                "deployment action gripper_indices differ from the data profile"
            )
    _validate_normalizer_object(
        action_normalizer,
        name="action",
        width=int(config.dimensions.action_dim),
    )
    _validate_normalizer_object(
        state_normalizer,
        name="state",
        width=len(resolve_action_state_profile(config.data.data_profile).state_indices),
    )
    graph = deployment_graph_config(config)
    flow_schedule = deployment_flow_schedule(config).to_dict()
    image_preprocess = PreprocessConfig(
        resize_hw=(int(config.data.cache_side), int(config.data.cache_side)),
        crop_hw=None,
    )
    action_contract: dict[str, object] = {
        "data_profile": profile,
        "gripper_indices": [int(value) for value in gripper_indices],
        "gripper_output_mode": str(config.bottom.gripper_output_mode),
        "arm_flow_mode": str(config.bottom.arm_flow_mode),
        "continuous_gripper_codec_boundary": gripper_codec_boundary,
        "continuous_gripper_codec_boundary_scope": (
            CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE
        ),
        "receding_horizon_execute_rows": 1,
        "prediction_horizon": int(config.dimensions.action_horizon),
    }
    if profile_name == _LIBERO_PROFILE_NAME:
        # Keep the native names and normalized bounds in the checkpoint-owned
        # outlet contract so an evaluator cannot infer a different 7D order.
        action_contract.update(
            {
                "names": [
                    "dx",
                    "dy",
                    "dz",
                    "droll",
                    "dpitch",
                    "dyaw",
                    "gripper",
                ],
                "normalized_low": [-1.0] * _LIBERO_ACTION_DIM,
                "normalized_high": [1.0] * _LIBERO_ACTION_DIM,
            }
        )
    return {
        "schema": DEPLOYMENT_ABI_SCHEMA,
        "source_config_digest": identity.config_digest,
        "architecture_manifest": dict(identity.manifest),
        "graph_config": graph,
        "graph_config_sha256": canonical_sha256(graph),
        "flow_schedule": flow_schedule,
        "flow_schedule_sha256": canonical_sha256(flow_schedule),
        "observation": {
            **({"entity_chart": entity_chart_metadata(config.top.entity_chart_mode)}
               if config.top.entity_chart_mode == CURRENT_IMAGE_CHART else {}),
            **({"entity_context": {"schema": "completed-current-g3-v1", "read": "same-full-candidate-support", "target": "observed-dino-unchanged"}}
               if config.top.entity_context_mode == "completed_g3_v1" else {}),
            "camera_names": list(config.data.camera_names),
            **(
                {"history_timing_contract": HISTORY_TIMING_CONTRACT}
                if config.top.history_encoding_mode == TIMED_HISTORY_ENCODING
                else {}
            ),
            "visual_offsets": [-8, -4, 0],
            "state_offsets": [-8, -4, 0],
            "executed_action_offsets": [-24, -16, -12, -8, -6, -4, -2, -1],
            "rgb_preprocessing": preprocessing_identity(image_preprocess),
            "dinov2": {
                "model": config.data.dinov2_model,
                "patches_per_camera": int(config.dimensions.patches_per_camera),
                "token_width": int(config.dimensions.visual_token_dim),
                "compute_dtype": str(config.runtime.compute_dtype),
                "reference_batch_size": int(config.data.dinov2_reference_batch_size),
            },
            **({"visual_source_time": visual_time_metadata(config.observation.source_time_mode)} if config.observation.source_time_mode != FIXED_VISUAL_TIME else {}),
            **({"candidate_support": candidate_support_metadata(config.observation.candidate_support_mode)}
               if config.observation.candidate_support_mode != MOMENT_LOCAL_SUPPORT else {}),
            **({"local_ownership": local_ownership_metadata(config.observation.local_ownership_mode)}
               if config.observation.local_ownership_mode != INDEPENDENT_LOCAL_OWNERS else {}),
            "state_dim": int(config.dimensions.state_dim),
            **({"state_features": state_feature_metadata(
                config.top.state_feature_mode, config.data.data_profile,
                len(resolve_action_state_profile(config.data.data_profile).state_indices),
            )} if config.top.state_feature_mode != NATIVE_AFFINE_STATE else {}),
            "action_dim": int(config.dimensions.action_dim),
        },
        "action": action_contract,
        "normalizers": {
            "action_sha256": canonical_sha256(action_normalizer.to_dict()),
            "state_sha256": canonical_sha256(state_normalizer.to_dict()),
            "mode": action_normalizer.mode,
        },
        "language": {
            "logical_name": identity.language.logical_name,
            "sha256": identity.language.sha256,
            "size_bytes": int(identity.language.size_bytes),
            "metadata": dict(goal_metadata),
        },
    }


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"deployment ABI {name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def validate_deployment_abi(value: object) -> dict[str, object]:
    abi = _mapping(value, name="root")
    if abi.get("schema") != DEPLOYMENT_ABI_SCHEMA:
        raise ValueError(
            f"formal deployment requires {DEPLOYMENT_ABI_SCHEMA}, "
            f"got {abi.get('schema')!r}"
        )
    graph = _mapping(abi.get("graph_config"), name="graph_config")
    if set(graph) != set(_GRAPH_SECTIONS):
        raise ValueError("deployment ABI graph section ownership differs")
    if str(abi.get("graph_config_sha256", "")) != canonical_sha256(graph):
        raise ValueError("deployment ABI graph digest is inconsistent")
    runtime = _mapping(graph.get("runtime"), name="graph_config.runtime")
    configured_schedule = runtime.get("deployment_flow_schedule")
    expected_schedule = (
        DeploymentFlowSchedule.uniform_five()
        if configured_schedule is None
        else DeploymentFlowSchedule.from_dict(configured_schedule)
    ).to_dict()
    if "flow_schedule" not in abi and "flow_schedule_sha256" not in abi:
        # Historical artifacts had one hard-coded uniform grid. Only that
        # unambiguous legacy case may omit the new explicit solver identity.
        if configured_schedule is not None:
            raise ValueError("explicit flow schedule requires deployment identity")
    else:
        stored_schedule = _mapping(abi.get("flow_schedule"), name="flow_schedule")
        DeploymentFlowSchedule.from_dict(stored_schedule)
        if stored_schedule != expected_schedule:
            raise ValueError("deployment flow schedule differs from graph runtime")
        if abi.get("flow_schedule_sha256") != canonical_sha256(stored_schedule):
            raise ValueError("deployment flow schedule digest is inconsistent")
    observation = _mapping(abi.get("observation"), name="observation")
    graph_top = _mapping(
        _mapping(abi.get("graph_config"), name="graph_config").get("top"), name="graph_config.top"
    )
    expected_timing = (
        HISTORY_TIMING_CONTRACT
        if graph_top.get("history_encoding_mode") == TIMED_HISTORY_ENCODING
        else None
    )
    if observation.get("history_timing_contract") != expected_timing:
        raise ValueError("deployment history timing contract differs from selected graph")
    chart_mode = graph_top.get("entity_chart_mode", QUERY_CHART)
    if chart_mode == CURRENT_IMAGE_CHART:
        actual_chart = _mapping(observation.get("entity_chart"), name="observation.entity_chart")
        if canonical_sha256(actual_chart) != canonical_sha256(entity_chart_metadata(CURRENT_IMAGE_CHART)):
            raise ValueError("deployment entity chart differs from the graph")
    elif chart_mode != QUERY_CHART or "entity_chart" in observation:
        raise ValueError("deployment entity chart is unknown or undeclared")
    context_mode = graph_top.get("entity_context_mode", "candidate_only_v1")
    if context_mode == "completed_g3_v1":
        expected_context = {"schema": "completed-current-g3-v1", "read": "same-full-candidate-support", "target": "observed-dino-unchanged"}
        actual_context = _mapping(observation.get("entity_context"), name="observation.entity_context")
        if canonical_sha256(actual_context) != canonical_sha256(expected_context):
            raise ValueError("deployment entity context differs from the graph")
    elif context_mode != "candidate_only_v1" or "entity_context" in observation:
        raise ValueError("deployment entity context is unknown or undeclared")
    action = _mapping(abi.get("action"), name="action")
    normalizers = _mapping(abi.get("normalizers"), name="normalizers")
    language = _mapping(abi.get("language"), name="language")
    profile = _mapping(action.get("data_profile"), name="action.data_profile")
    profile_name = profile.get("name")
    graph_observation = _mapping(graph.get("observation"), name="graph_config.observation")
    owner_mode = str(graph_observation.get("local_ownership_mode", INDEPENDENT_LOCAL_OWNERS))
    if owner_mode != INDEPENDENT_LOCAL_OWNERS:
        owner_metadata = _mapping(observation.get("local_ownership"), name="observation.local_ownership")
        if canonical_sha256(owner_metadata) != canonical_sha256(local_ownership_metadata(owner_mode)):
            raise ValueError("deployment local ownership differs from the graph")
    elif "local_ownership" in observation:
        raise ValueError("legacy local reader cannot acquire undeclared shared ownership")
    support_mode = str(graph_observation.get("candidate_support_mode", MOMENT_LOCAL_SUPPORT))
    if support_mode != MOMENT_LOCAL_SUPPORT:
        support_metadata = _mapping(observation.get("candidate_support"), name="observation.candidate_support")
        if canonical_sha256(support_metadata) != canonical_sha256(candidate_support_metadata(support_mode)):
            raise ValueError("deployment candidate support contract differs from graph")
    elif "candidate_support" in observation:
        raise ValueError("legacy visual graph cannot acquire an undeclared candidate support")
    visual_mode = str(graph_observation.get("source_time_mode", FIXED_VISUAL_TIME))
    if visual_mode != FIXED_VISUAL_TIME:
        visual_metadata = _mapping(observation.get("visual_source_time"), name="observation.visual_source_time")
        if canonical_sha256(visual_metadata) != canonical_sha256(visual_time_metadata(visual_mode)):
            raise ValueError("deployment visual source-time contract differs from graph")
    elif "visual_source_time" in observation:
        raise ValueError("legacy visual graph cannot acquire an undeclared source clock")
    state_mode = str(graph_top.get("state_feature_mode", NATIVE_AFFINE_STATE))
    native_width = len(resolve_action_state_profile(str(profile_name)).state_indices)
    expected_features = state_feature_metadata(state_mode, str(profile_name), native_width)
    if state_mode != NATIVE_AFFINE_STATE:
        features = _mapping(observation.get("state_features"), name="observation.state_features")
        if canonical_sha256(features) != canonical_sha256(expected_features):
            raise ValueError("deployment state feature chart differs from selected graph")
        dimensions = _mapping(graph.get("dimensions"), name="graph_config.dimensions")
        if (_strict_abi_int(observation.get("state_dim"), name="observation.state_dim")
                != expected_features["feature_state_dim"]
                or _strict_abi_int(dimensions.get("state_dim"), name="dimensions.state_dim")
                != expected_features["feature_state_dim"]):
            raise ValueError("deployment state feature width differs from selected chart")
    elif "state_features" in observation:
        raise ValueError("legacy state graph cannot silently acquire a new feature chart")
    dino = _mapping(observation.get("dinov2"), name="observation.dinov2")
    if not str(dino.get("model", "")).strip():
        raise ValueError("deployment DINO model identity is empty")
    if str(dino.get("compute_dtype", "")) not in {"bf16", "fp32"}:
        raise ValueError("deployment DINO compute_dtype must be bf16 or fp32")
    try:
        reference_batch_size = _strict_abi_int(
            dino.get("reference_batch_size", 0),
            name="observation.dinov2.reference_batch_size",
        )
    except ValueError as error:
        raise ValueError("deployment DINO reference_batch_size must be an integer") from error
    if reference_batch_size <= 0:
        raise ValueError("deployment DINO reference_batch_size must be positive")
    cameras = tuple(str(name) for name in observation.get("camera_names", ()))
    if cameras != ("top", "wrist"):
        raise ValueError("deployment ABI camera order must be top,wrist")
    if tuple(observation.get("visual_offsets", ())) != (-8, -4, 0):
        raise ValueError("deployment ABI visual history differs from the formal policy")
    if tuple(observation.get("state_offsets", ())) != (-8, -4, 0):
        raise ValueError("deployment ABI state history differs from the formal policy")
    if tuple(observation.get("executed_action_offsets", ())) != (
        -24,
        -16,
        -12,
        -8,
        -6,
        -4,
        -2,
        -1,
    ):
        raise ValueError("deployment ABI executed-action history differs")
    try:
        execute_rows = _strict_abi_int(
            action.get("receding_horizon_execute_rows", 0),
            name="action.receding_horizon_execute_rows",
        )
    except ValueError as error:
        raise ValueError("deployment execute-row field must be an integer") from error
    if execute_rows != 1:
        raise ValueError("deployment must execute exactly one predicted action row")
    output_mode = str(action.get("gripper_output_mode", "continuous"))
    if output_mode not in VALID_GRIPPER_OUTPUT_MODES:
        raise ValueError("deployment gripper_output_mode is invalid")
    bottom = _mapping(graph.get("bottom"), name="graph_config.bottom")
    if str(bottom.get("gripper_output_mode", "continuous")) != output_mode:
        raise ValueError(
            "deployment gripper output mode differs from graph bottom configuration"
        )
    arm_mode = str(action.get("arm_flow_mode", ""))
    if arm_mode not in {"legacy_independent", "relative_command_adapter"}:
        raise ValueError("deployment arm_flow_mode is invalid")
    if str(bottom.get("arm_flow_mode", "")) != arm_mode:
        raise ValueError(
            "deployment arm flow mode differs from graph bottom configuration"
        )
    if profile_name == _LIBERO_PROFILE_NAME:
        dimensions = _mapping(graph.get("dimensions"), name="graph_config.dimensions")
        for field, expected in (
            ("action_dim", _LIBERO_ACTION_DIM),
            ("state_dim", _LIBERO_STATE_DIM),
            ("action_horizon", _LIBERO_ACTION_HORIZON),
        ):
            try:
                actual = _strict_abi_int(
                    dimensions.get(field),
                    name=f"graph_config.dimensions.{field}",
                )
            except ValueError as error:
                raise ValueError(f"LIBERO graph dimension {field} is invalid") from error
            if actual != expected:
                raise ValueError(
                    f"LIBERO graph dimension {field} must be {expected}, got {actual}"
                )
    profile_gripper_boundary = str(profile.get("gripper_transition_boundary", ""))
    if profile_gripper_boundary not in {
        "current_action_state",
        "previous_command",
    }:
        raise ValueError("deployment data profile gripper codec boundary is invalid")
    if (
        str(action.get("continuous_gripper_codec_boundary", ""))
        != profile_gripper_boundary
    ):
        raise ValueError("deployment gripper codec boundary differs from its data profile")
    if (
        str(action.get("continuous_gripper_codec_boundary_scope", ""))
        != CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE
    ):
        raise ValueError("deployment gripper codec boundary scope is stale")
    if "gripper_indices" in action and "gripper_indices" in profile:
        action_grippers = _strict_index_sequence(
            action.get("gripper_indices"), name="action.gripper_indices"
        )
        profile_grippers = _strict_index_sequence(
            profile.get("gripper_indices"), name="action.data_profile.gripper_indices"
        )
        if action_grippers != profile_grippers:
            raise ValueError(
                "deployment action gripper_indices differ from the data profile"
            )
    if profile_name == "calvin_relative_7d_v1":
        if arm_mode != "relative_command_adapter":
            raise ValueError("CALVIN deployment requires its relative-command adapter")
        if output_mode != CALVIN_BINARY_GRIPPER_OUTPUT_MODE:
            raise ValueError("CALVIN deployment requires its binary command outlet")
    elif profile_name == _LIBERO_PROFILE_NAME:
        _validate_libero_profile_metadata(
            profile,
            gripper_indices=action.get("gripper_indices"),
            name="deployment data profile",
        )
        if arm_mode != "relative_command_adapter":
            raise ValueError("LIBERO deployment requires its relative-command adapter")
        if output_mode != "continuous":
            raise ValueError("LIBERO deployment requires continuous gripper output")
        if profile_gripper_boundary != "previous_command":
            raise ValueError(
                "LIBERO deployment requires a previous-command gripper boundary"
            )
        for field, expected in (
            ("state_dim", _LIBERO_STATE_DIM),
            ("action_dim", _LIBERO_ACTION_DIM),
        ):
            try:
                actual = _strict_abi_int(
                    observation.get(field),
                    name=f"observation.{field}",
                )
            except ValueError as error:
                raise ValueError(f"LIBERO deployment {field} is invalid") from error
            if actual != expected:
                raise ValueError(
                    f"LIBERO deployment {field} must be {expected}, got {actual}"
                )
        try:
            horizon = _strict_abi_int(
                action.get("prediction_horizon"),
                name="action.prediction_horizon",
            )
        except ValueError as error:
            raise ValueError("LIBERO deployment prediction horizon is invalid") from error
        if horizon != _LIBERO_ACTION_HORIZON:
            raise ValueError("LIBERO deployment prediction horizon must be 24")
        names = _sequence(action.get("names"), name="action.names")
        if names != (
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
            "gripper",
        ):
            raise ValueError("LIBERO deployment action names/order is stale")
        for field, expected in (
            ("normalized_low", (-1.0,) * _LIBERO_ACTION_DIM),
            ("normalized_high", (1.0,) * _LIBERO_ACTION_DIM),
        ):
            values = _sequence(action.get(field), name=f"action.{field}")
            try:
                numeric = tuple(float(item) for item in values)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"LIBERO deployment {field} is not numeric") from error
            if numeric != expected:
                raise ValueError(f"LIBERO deployment {field} differs from [-1,1]")
    elif profile_name in {"maniskill_pd_ee_delta_pose_7d_v1", "maniskill_pd_ee_delta_pose_7d_v2"}:
        _validate_maniskill_profile(profile)
        if (arm_mode != "relative_command_adapter" or output_mode not in {
                    CONTINUOUS_GRIPPER_OUTPUT_MODE,
                    MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
                }
                or profile_gripper_boundary != "previous_command"
                or action.get("prediction_horizon") != 24
                or observation.get("state_dim") != 7 or observation.get("action_dim") != 7
                or tuple(action.get("gripper_indices", ())) != (6,)):
            raise ValueError(
                "ManiSkill deployment requires its explicit relative 24x7 boundary"
            )
        if (
            output_mode == MANISKILL_BINARY_GRIPPER_OUTPUT_MODE
            and profile_name != "maniskill_pd_ee_delta_pose_7d_v2"
        ):
            raise ValueError(
                "ManiSkill binary deployment requires the repaired v2 profile"
            )
        if (
            profile_name == "maniskill_pd_ee_delta_pose_7d_v1"
            and output_mode != CONTINUOUS_GRIPPER_OUTPUT_MODE
        ):
            raise ValueError("legacy ManiSkill v1 deployment requires continuous gripper output")
    elif arm_mode != "legacy_independent":
        raise ValueError("the relative-command action adapter is CALVIN/LIBERO/ManiSkill-only")
    if normalizers.get("mode") != "zscore":
        raise ValueError("deployment normalizers must use zscore")
    for owner, digest in (
        ("action normalizer", normalizers.get("action_sha256")),
        ("state normalizer", normalizers.get("state_sha256")),
        ("language", language.get("sha256")),
    ):
        text = str(digest).lower()
        if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise ValueError(f"deployment {owner} identity must be SHA-256")
    return abi


def deployment_config_from_checkpoint(
    raw_config: object,
    raw_abi: object,
) -> ExperimentConfig:
    """Rebuild only the graph and deployment-facing data fields.

    Data conversion/split/cache path fields are intentionally not parsed from
    the training checkpoint.  This is the compatibility boundary that prevents
    a future data-only ``DataConfig`` field from breaking an otherwise exact
    model state.
    """

    if not isinstance(raw_config, Mapping):
        raise ValueError("deployment checkpoint config must be a mapping")
    abi = validate_deployment_abi(raw_abi)
    graph = _mapping(abi["graph_config"], name="graph_config")
    for name in _GRAPH_SECTIONS:
        if raw_config.get(name) != graph[name]:
            raise ValueError(f"checkpoint {name} differs from its deployment ABI")
    observation = _mapping(abi["observation"], name="observation")
    action = _mapping(abi["action"], name="action")
    dino = _mapping(observation.get("dinov2"), name="observation.dinov2")
    profile = _mapping(action.get("data_profile"), name="action.data_profile")
    objective = _mapping(graph["objectives"], name="graph_config.objectives")

    # Start from current benign data defaults, then inject only fields that
    # affect the deployed input chart.  Split paths and counts never cross this
    # boundary.
    data = cast(dict[str, object], asdict(DataConfig()))
    data.update(
        {
            "camera_names": tuple(str(name) for name in observation["camera_names"]),
            "data_profile": str(profile.get("name", "")),
            "cache_side": int(
                _mapping(
                    _mapping(observation.get("rgb_preprocessing"), name="rgb_preprocessing").get(
                        "config"
                    ),
                    name="rgb_preprocessing.config",
                )["resize_hw"][0]  # type: ignore[index]
            ),
            "dinov2_model": str(dino.get("model", "")),
            "dinov2_reference_batch_size": int(dino.get("reference_batch_size", 0)),
            "split_mode": "ordered-counts",
            "split_manifest": "",
            "task_selection_manifest": "",
            "normalizer_artifact": "",
            "task_filter": "",
            "train_episodes": 63,
            "val_episodes": 5,
            "test_episodes": 5,
            "sampling_gripper_event_threshold": float(
                objective.get("gripper_event_threshold", 0.0)
            ),
        }
    )
    payload = {
        "data": data,
        **graph,
        # Optimizer state is not instantiated for deployment.  Current formal
        # defaults satisfy ExperimentConfig validation without importing stale
        # optimizer-only checkpoint fields.
        "optimizer": {},
    }
    config = config_from_mapping(payload)
    if int(config.dimensions.patches_per_camera) != int(dino.get("patches_per_camera", -1)):
        raise ValueError("deployment DINO patch count differs from graph dimensions")
    if int(config.dimensions.visual_token_dim) != int(dino.get("token_width", -1)):
        raise ValueError("deployment DINO width differs from graph dimensions")
    if str(config.runtime.compute_dtype) != str(dino.get("compute_dtype", "")):
        raise ValueError("deployment DINO compute dtype differs from graph runtime")
    return config


__all__ = [
    "DEPLOYMENT_ABI_SCHEMA",
    "build_deployment_abi",
    "canonical_sha256",
    "deployment_config_from_checkpoint",
    "deployment_graph_config",
    "validate_deployment_abi",
]
