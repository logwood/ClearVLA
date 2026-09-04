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

from clearvla.vision.preprocessing import (
    PreprocessConfig,
    preprocessing_identity,
)

from ..checkpoint import CheckpointIdentity
from ..config import DataConfig, ExperimentConfig, config_from_mapping
from ..data.normalizer import ArrayNormalizer

DEPLOYMENT_ABI_SCHEMA = "clearvla-mainline-deployment-abi-v1"
CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE = (
    "profile_owned_full_horizon_encode_decode_loss_evaluation"
)

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
    gripper_codec_boundary = str(data_profile.get("gripper_transition_boundary", ""))
    if gripper_codec_boundary not in {
        "current_action_state",
        "previous_command",
    }:
        raise ValueError("deployment data profile lacks a gripper codec boundary")
    graph = deployment_graph_config(config)
    image_preprocess = PreprocessConfig(
        resize_hw=(int(config.data.cache_side), int(config.data.cache_side)),
        crop_hw=None,
    )
    return {
        "schema": DEPLOYMENT_ABI_SCHEMA,
        "source_config_digest": identity.config_digest,
        "architecture_manifest": dict(identity.manifest),
        "graph_config": graph,
        "graph_config_sha256": canonical_sha256(graph),
        "observation": {
            "camera_names": list(config.data.camera_names),
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
            "state_dim": int(config.dimensions.state_dim),
            "action_dim": int(config.dimensions.action_dim),
        },
        "action": {
            "data_profile": dict(data_profile),
            "gripper_indices": [int(value) for value in gripper_indices],
            "gripper_output_mode": str(config.bottom.gripper_output_mode),
            "arm_flow_mode": str(config.bottom.arm_flow_mode),
            "continuous_gripper_codec_boundary": gripper_codec_boundary,
            "continuous_gripper_codec_boundary_scope": (
                CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE
            ),
            "receding_horizon_execute_rows": 1,
            "prediction_horizon": int(config.dimensions.action_horizon),
        },
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
    observation = _mapping(abi.get("observation"), name="observation")
    action = _mapping(abi.get("action"), name="action")
    normalizers = _mapping(abi.get("normalizers"), name="normalizers")
    language = _mapping(abi.get("language"), name="language")
    dino = _mapping(observation.get("dinov2"), name="observation.dinov2")
    if not str(dino.get("model", "")).strip():
        raise ValueError("deployment DINO model identity is empty")
    if str(dino.get("compute_dtype", "")) not in {"bf16", "fp32"}:
        raise ValueError("deployment DINO compute_dtype must be bf16 or fp32")
    try:
        reference_batch_size = int(dino.get("reference_batch_size", 0))
    except (TypeError, ValueError) as error:
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
    if int(action.get("receding_horizon_execute_rows", 0)) != 1:
        raise ValueError("deployment must execute exactly one predicted action row")
    output_mode = str(action.get("gripper_output_mode", "continuous"))
    if output_mode not in {"continuous", "calvin_binary_command"}:
        raise ValueError("deployment gripper_output_mode is invalid")
    bottom = _mapping(graph.get("bottom"), name="graph_config.bottom")
    if str(bottom.get("gripper_output_mode", "continuous")) != output_mode:
        raise ValueError(
            "deployment gripper output mode differs from graph bottom configuration"
        )
    arm_mode = str(action.get("arm_flow_mode", ""))
    if arm_mode not in {"legacy_independent", "relative_command_direct"}:
        raise ValueError("deployment arm_flow_mode is invalid")
    if str(bottom.get("arm_flow_mode", "")) != arm_mode:
        raise ValueError(
            "deployment arm flow mode differs from graph bottom configuration"
        )
    profile = _mapping(action.get("data_profile"), name="action.data_profile")
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
    if profile.get("name") == "calvin_relative_7d_v1":
        if arm_mode != "relative_command_direct":
            raise ValueError("CALVIN deployment requires direct relative-command arms")
    elif arm_mode != "legacy_independent":
        raise ValueError("direct relative-command arms are CALVIN-only")
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
