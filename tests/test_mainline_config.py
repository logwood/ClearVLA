from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_IMPLEMENTATION,
    BSPINE0_SPEC_FINGERPRINT,
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_IMPLEMENTATION,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    BSPINE_DISABLED_IMPLEMENTATION,
)

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_config_loads_one_flat_active_preset() -> None:
    config = load_config(ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json")
    config.validate()
    assert config.optimizer.batch_size == 8
    assert config.optimizer.history_proposal_lr_scale == 0.625
    assert config.optimizer.bottom_decoder_lr_scale == 0.7
    assert config.optimizer.bottom_capacity_relative_lr_scale == 2.0
    assert config.dimensions.hidden_size == 512
    assert config.dimensions.goal_token_dim == 4096
    # The formal ``dinov2_cache_336`` metadata is a 16x16 token chart.  Do not
    # infer token count from the decoded 336-pixel image side.
    assert config.dimensions.patches_per_camera == 256
    assert config.observation.grid_size == 8
    assert config.observation.flow_reference_frames == 4
    assert config.bottom.max_dwell == 2
    assert config.bottom.execution_warmup_steps == 200
    assert config.bottom.execution_transition_steps == 1000
    assert config.bottom.execution_eval_policy == "soft"
    assert config.objectives.execution_value == 0.05
    assert config.objectives.execution_value_huber_delta == 0.10
    assert config.data.information_uniform_fraction == 0.50
    assert config.data.information_event_fraction == 0.125
    assert config.data.information_motion_quantile == 0.70
    assert config.bottom.gripper_field_dim == 6
    assert config.bottom.physical_decode_delta_blend == 0.25
    assert config.objectives.horizon_tail_emphasis == 0.20
    assert config.objectives.horizon_first_step_protection == 0.05
    assert config.objectives.gripper_event_threshold == 0.10
    assert len(config.digest()) == 64
    assert config.digest() != config.digest(include_paths=True)


def test_bspine_config_is_explicit_and_baseline_payload_stays_schema30() -> None:
    baseline = load_config(ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json")
    enabled = load_config(
        ROOT / "configs" / "mainline" / "object_intent_dynamics_323_pen_bspine0.json"
    )
    arm_only = load_config(
        ROOT
        / "configs"
        / "mainline"
        / "object_intent_dynamics_323_pen_bspine_arm_only.json"
    )
    assert baseline.bottom.bspine_implementation == BSPINE_DISABLED_IMPLEMENTATION
    assert not any(name.startswith("bspine_") for name in baseline.as_dict()["bottom"])
    assert enabled.bottom.bspine_implementation == BSPINE0_IMPLEMENTATION
    assert enabled.bottom.bspine_degree == 3
    assert enabled.bottom.bspine_control_points == 12
    assert enabled.bottom.bspine_basis_digest == BSPINE0_BASIS_DIGEST
    assert enabled.bottom.bspine_spec_fingerprint == BSPINE0_SPEC_FINGERPRINT
    assert "bspine_action_group_mask" not in enabled.as_dict()["bottom"]
    assert arm_only.bottom.bspine_implementation == BSPINE_ARM_ONLY_IMPLEMENTATION
    assert arm_only.bottom.bspine_action_group_mask == BSPINE_ARM_ONLY_ACTION_GROUP_MASK
    assert arm_only.bottom.bspine_basis_digest == BSPINE0_BASIS_DIGEST
    assert arm_only.bottom.bspine_spec_fingerprint == BSPINE_ARM_ONLY_SPEC_FINGERPRINT
    assert (
        arm_only.as_dict()["bottom"]["bspine_action_group_mask"]
        == BSPINE_ARM_ONLY_ACTION_GROUP_MASK
    )
    assert enabled.digest() != baseline.digest()
    assert arm_only.digest() not in {baseline.digest(), enabled.digest()}

    incomplete = replace(
        baseline,
        bottom=replace(
            baseline.bottom,
            bspine_implementation=BSPINE0_IMPLEMENTATION,
        ),
    )
    try:
        incomplete.validate()
    except ValueError as error:
        assert "frozen cubic K=12" in str(error)
    else:
        raise AssertionError("B-spine must not run without its serialized basis identity")

    wrong_mask = replace(
        arm_only,
        bottom=replace(arm_only.bottom, bspine_action_group_mask="11111"),
    )
    try:
        wrong_mask.validate()
    except ValueError as error:
        assert "11000 group mask" in str(error)
    else:
        raise AssertionError("arm-only B-spine accepted a non-arm action-group mask")


def test_mainline_config_rejects_legacy_or_unknown_switches() -> None:
    payload = ExperimentConfig().as_dict()
    payload["flow_jepa_object_intent_dynamics_mainline"] = 1
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown config sections" in str(error)
    else:
        raise AssertionError("legacy graph selectors must not enter the mainline spec")

    payload = json.loads(
        (ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json").read_text(
            encoding="utf-8"
        )
    )
    payload["optimizer"] = {"batch_size": 8, "midcut_aux_loss_weight": 0.1}
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown optimizer fields" in str(error)
    else:
        raise AssertionError("inactive historical losses must not be configurable")

    payload = ExperimentConfig().as_dict()
    payload["observation"]["raw_mid_radius"] = 2
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown observation fields" in str(error)
    else:
        raise AssertionError("dead active-looking architecture fields must be rejected")

    payload = ExperimentConfig().as_dict()
    payload["top"]["proposal_condition_dropout"] = 0.25
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown top fields" in str(error)
    else:
        raise AssertionError("a dropout mask without a forward consumer must be rejected")


def test_mainline_config_enforces_fixed_graph_boundaries() -> None:
    config = ExperimentConfig()
    broken = replace(
        config,
        observation=replace(config.observation, microgrid_side=2),
    )
    try:
        broken.validate()
    except ValueError as error:
        assert "four 3x3 factual glimpses" in str(error)
    else:
        raise AssertionError("P1 ownership cannot be changed by a loose flag")
    broken_history = replace(
        config,
        dimensions=replace(config.dimensions, visual_history_length=2),
    )
    try:
        broken_history.validate()
    except ValueError as error:
        assert "causal visual history" in str(error)
    else:
        raise AssertionError("fixed dataset history cannot masquerade as configurable")
    broken_flow_span = replace(
        config,
        observation=replace(config.observation, flow_reference_frames=1),
    )
    try:
        broken_flow_span.validate()
    except ValueError as error:
        assert "four-frame raw pair" in str(error)
    else:
        raise AssertionError("flow temporal units cannot drift from the raw-pair dataset")
    broken_physical_field = replace(
        config,
        bottom=replace(config.bottom, gripper_field_dim=7),
    )
    try:
        broken_physical_field.validate()
    except ValueError as error:
        assert "six gripper channels" in str(error)
    else:
        raise AssertionError("the formal 18-D physical field cannot drift")


def test_config_identity_ignores_relocation_but_not_data_semantics() -> None:
    config = ExperimentConfig()
    relocated = replace(
        config,
        data=replace(
            config.data,
            raw_hdf5_root="/relocated/raw",
            decoded_cache="/relocated/decoded",
            dino_cache="/relocated/dino",
            t5_condition="/relocated/goal.pt",
            output_dir="/relocated/run",
        ),
    )
    assert config.digest() == relocated.digest()
    changed_stride = replace(config, data=replace(config.data, stride=2))
    changed_camera_key = replace(
        config,
        data=replace(config.data, top_camera_key="observations/images/other"),
    )
    changed_dtype = replace(
        config,
        runtime=replace(config.runtime, compute_dtype="fp32"),
    )
    assert config.digest() != changed_stride.digest()
    assert config.digest() != changed_camera_key.digest()
    assert config.digest() != changed_dtype.digest()


def test_pread_cache_backend_is_opt_in_and_serialized_explicitly() -> None:
    baseline = ExperimentConfig()
    baseline_data = baseline.as_dict()["data"]
    assert "visual_cache_read_backend" not in baseline_data
    assert "visual_pread_max_open_files" not in baseline_data

    pread = replace(
        baseline,
        data=replace(
            baseline.data,
            visual_cache_read_backend="pread",
            visual_pread_max_open_files=3,
        ),
    )
    pread.validate()
    pread_data = pread.as_dict()["data"]
    assert pread_data["visual_cache_read_backend"] == "pread"
    assert pread_data["visual_pread_max_open_files"] == 3
    assert pread.digest() != baseline.digest()

    with pytest.raises(ValueError, match="custom visual_pread_max_open_files"):
        replace(
            baseline,
            data=replace(baseline.data, visual_pread_max_open_files=3),
        ).validate()


def test_gripper_compatibility_weight_is_explicit_only_when_nondefault() -> None:
    baseline = ExperimentConfig()
    legacy = replace(
        baseline,
        objectives=replace(
            baseline.objectives,
            gripper_compatibility_weight=1.0,
        ),
    )
    deployed_only = replace(
        baseline,
        objectives=replace(
            baseline.objectives,
            gripper_compatibility_weight=0.0,
        ),
    )

    # The default/explicit-one objective remains byte- and digest-compatible
    # with old Schema30 configs, while the controlled zero-weight experiment is
    # impossible to confuse with that baseline.
    assert legacy.as_dict() == baseline.as_dict()
    assert legacy.digest() == baseline.digest()
    payload = deployed_only.as_dict()
    assert payload["objectives"]["gripper_compatibility_weight"] == 0.0
    restored = config_from_mapping(json.loads(json.dumps(payload)))
    assert restored.objectives.gripper_compatibility_weight == 0.0
    assert restored.digest() == deployed_only.digest()
    assert restored.digest() != baseline.digest()


def test_release_first_repair_is_explicit_and_libero_scoped() -> None:
    baseline = ExperimentConfig()
    assert "release_first_action_fraction" not in baseline.as_dict()["data"]
    assert "gripper_first_step_release" not in baseline.as_dict()["objectives"]
    libero = replace(
        baseline,
        data=replace(
            baseline.data,
            data_profile="libero_relative_7d_v1",
            split_mode="episode-manifest",
            split_manifest="/data/libero/splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
            window_boundary_contract="causal_prefix_terminal_suffix_v2",
            release_first_action_fraction=1.0,
        ),
        objectives=replace(
            baseline.objectives,
            gripper_first_step_release=0.05,
        ),
        bottom=replace(
            baseline.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="continuous",
        ),
    )
    libero.validate()
    payload = libero.as_dict()
    assert payload["data"]["release_first_action_fraction"] == 1.0
    assert payload["objectives"]["gripper_first_step_release"] == 0.05
    restored = config_from_mapping(json.loads(json.dumps(payload)))
    assert restored.digest() == libero.digest()
    invalid = replace(
        baseline,
        data=replace(baseline.data, release_first_action_fraction=1.0),
    )
    try:
        invalid.validate()
    except ValueError as error:
        assert "LIBERO terminal-suffix" in str(error)
    else:
        raise AssertionError("release-first sampling escaped its LIBERO boundary")


def test_information_sampler_epoch_budget_is_explicit_and_round_trips() -> None:
    baseline = ExperimentConfig()
    assert "information_batches_per_epoch" not in baseline.as_dict()["data"]
    libero = replace(
        baseline,
        data=replace(
            baseline.data,
            data_profile="libero_relative_7d_v1",
            split_mode="episode-manifest",
            split_manifest="/data/libero/splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
            window_boundary_contract="causal_prefix_terminal_suffix_v2",
            information_batches_per_epoch=562,
        ),
        bottom=replace(
            baseline.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="continuous",
        ),
    )
    libero.validate()
    payload = libero.as_dict()
    assert payload["data"]["information_batches_per_epoch"] == 562
    restored = config_from_mapping(json.loads(json.dumps(payload)))
    assert restored.data.information_batches_per_epoch == 562
    assert restored.digest() == libero.digest()
    with pytest.raises(ValueError, match="positive integer"):
        replace(
            libero,
            data=replace(libero.data, information_batches_per_epoch=0),
        ).validate()


def test_stackcube_first_step_hold_is_explicit_and_maniskill_scoped() -> None:
    baseline = ExperimentConfig()
    assert "gripper_first_step_hold" not in baseline.as_dict()["objectives"]
    stackcube = replace(
        baseline,
        data=replace(
            baseline.data,
            data_profile="maniskill_pd_ee_delta_pose_7d_v2",
            split_mode="episode-manifest",
            split_manifest="/data/stackcube/splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
        ),
        bottom=replace(baseline.bottom, arm_flow_mode="relative_command_adapter"),
        objectives=replace(baseline.objectives, gripper_first_step_hold=0.05),
    )
    stackcube.validate()
    payload = stackcube.as_dict()
    assert payload["objectives"]["gripper_first_step_hold"] == 0.05
    assert config_from_mapping(json.loads(json.dumps(payload))).digest() == stackcube.digest()
    with pytest.raises(ValueError, match="ManiSkill"):
        replace(
            baseline,
            objectives=replace(baseline.objectives, gripper_first_step_hold=0.05),
        ).validate()


def test_libero_first_step_hold_is_explicit_and_continuous_scoped() -> None:
    baseline = ExperimentConfig()
    libero = replace(
        baseline,
        data=replace(
            baseline.data,
            data_profile="libero_relative_7d_v1",
            split_mode="episode-manifest",
            split_manifest="/data/libero/splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
            window_boundary_contract="causal_prefix_terminal_suffix_v2",
        ),
        bottom=replace(
            baseline.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="continuous",
        ),
        objectives=replace(baseline.objectives, gripper_first_step_hold=0.05),
    )
    libero.validate()
    payload = libero.as_dict()
    assert payload["objectives"]["gripper_first_step_hold"] == 0.05
    assert config_from_mapping(json.loads(json.dumps(payload))).digest() == libero.digest()


def test_libero_retarget_overlay_is_explicit_without_changing_legacy_payload() -> None:
    baseline = ExperimentConfig()
    assert "libero_retarget_overlay_root" not in baseline.as_dict()["data"]
    assert "libero_retarget_overlay_manifest" not in baseline.as_dict()["data"]

    overlay = replace(
        baseline,
        data=replace(
            baseline.data,
            data_profile="libero_relative_7d_v1",
            split_mode="episode-manifest",
            split_manifest="/data/libero/base/splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
            window_boundary_contract="causal_prefix_terminal_suffix_v2",
            libero_retarget_overlay_root="/data/libero/overlay",
            libero_retarget_overlay_manifest=(
                "/data/libero/overlay/overlay_manifest.json"
            ),
        ),
        bottom=replace(
            baseline.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="continuous",
        ),
    )
    overlay.validate()
    payload = overlay.as_dict()
    assert payload["data"]["libero_retarget_overlay_root"].endswith("/overlay")
    restored = config_from_mapping(json.loads(json.dumps(payload)))
    assert restored.as_dict() == payload

    relocated = replace(
        overlay,
        data=replace(
            overlay.data,
            libero_retarget_overlay_root="/relocated/overlay",
            libero_retarget_overlay_manifest="/relocated/overlay/manifest.json",
        ),
    )
    assert relocated.digest() == overlay.digest()

    incomplete = replace(
        overlay,
        data=replace(overlay.data, libero_retarget_overlay_manifest=""),
    )
    try:
        incomplete.validate()
    except ValueError as error:
        assert "configured together" in str(error)
    else:
        raise AssertionError("a half-configured LIBERO retarget overlay was accepted")
