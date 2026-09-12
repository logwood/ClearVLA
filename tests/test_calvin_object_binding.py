from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.calvin_object_binding import (
    CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
    CalvinObjectBindingSidecar,
    calvin_episode_inventory_digest,
)
from clearvla.mainline.interfaces import CalvinObjectBindingTarget
from clearvla.mainline.model.calvin_object_binding import (
    CALVIN_OBJECT_BINDING_COMPONENT,
    CalvinObjectBindingBridge,
)
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.intent import CoarseActionIntent
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.types import ActionIntentDock, DenseFactChart, ObjectFactSet


def _facts(
    *,
    batch: int = 2,
    objects: int = 4,
    hidden: int = 8,
    route: int = 3,
    content: int = 5,
    validity: torch.Tensor | None = None,
    coordinates: torch.Tensor | None = None,
) -> ObjectFactSet:
    cameras = 2
    if validity is None:
        validity = torch.ones(batch, objects, 1)
    if coordinates is None:
        coordinates = torch.tensor(
            [[[-0.8, -0.4], [-0.2, 0.6], [0.4, -0.1], [0.9, 0.7]]],
            dtype=torch.float32,
        ).expand(batch, -1, -1).clone()
    candidate_prefix = (batch, cameras, 1, 1, 1)
    dense = DenseFactChart(
        public_scene_base=torch.zeros(batch, cameras, 1, 1, hidden),
        dino_content=torch.zeros(batch, cameras, 1, 1, content),
        cell_observed=torch.ones(batch, cameras, 1, 1, 1, dtype=torch.bool),
        candidate_content=torch.zeros(*candidate_prefix, content),
        candidate_semantic=torch.zeros(*candidate_prefix, route),
        candidate_appearance=torch.zeros(*candidate_prefix, route),
        candidate_geometry=torch.zeros(*candidate_prefix, route),
        candidate_coordinates=torch.zeros(*candidate_prefix, 2),
        candidate_support=torch.ones(*candidate_prefix),
        candidate_validity=torch.ones(*candidate_prefix, 1),
        candidate_owner_prior=torch.ones(*candidate_prefix),
        candidate_owner_log_prior=torch.zeros(*candidate_prefix),
        candidate_semantic_prior=torch.ones(*candidate_prefix),
        candidate_appearance_prior=torch.ones(*candidate_prefix),
        candidate_geometry_prior=torch.ones(*candidate_prefix),
        candidate_semantic_log_prior=torch.zeros(*candidate_prefix),
        candidate_appearance_log_prior=torch.zeros(*candidate_prefix),
        candidate_geometry_log_prior=torch.zeros(*candidate_prefix),
        candidate_transport_prior=torch.zeros(*candidate_prefix, 2),
    )
    camera_coordinates = coordinates[:, :, None].expand(-1, -1, cameras, -1).clone()
    assignment = torch.full((batch, objects, cameras, 1, 1, 1), 1.0 / objects)
    return ObjectFactSet(
        dense_chart=dense,
        content=torch.zeros(batch, objects, content),
        semantic=torch.randn(batch, objects, route),
        appearance=torch.randn(batch, objects, route),
        geometry=torch.randn(batch, objects, route),
        camera_coordinates=camera_coordinates,
        camera_transport_prior=torch.zeros_like(camera_coordinates),
        camera_support=torch.ones(batch, objects, cameras, 1),
        camera_validity=torch.ones(batch, objects, cameras, 1),
        log_camera_validity=torch.zeros(batch, objects, cameras, 1),
        support=torch.ones(batch, objects, 1),
        existence=torch.ones(batch, objects, 1),
        validity=validity.float(),
        log_validity=torch.zeros(batch, objects, 1),
        object_to_chart=torch.full((batch, objects, cameras, 1, 1), 1.0 / objects),
        candidate_assignment=assignment,
        semantic_candidate_assignment=assignment.clone(),
        appearance_candidate_assignment=assignment.clone(),
        geometry_candidate_assignment=assignment.clone(),
        null_assignment=torch.zeros(*candidate_prefix),
        reconstructed_dino=torch.zeros(batch, cameras, 1, 1, content),
        reconstruction_error=torch.zeros(()),
    )


def _binding_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(
            base.data,
            data_profile="calvin_relative_7d_v1",
            split_mode="episode-manifest",
            split_manifest="splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
            calvin_object_binding_sidecar="targets.npz",
        ),
        top=replace(base.top, calvin_object_binding="calvin_primary_v1"),
        bottom=replace(
            base.bottom,
            arm_flow_mode="relative_command_direct",
            gripper_output_mode="calvin_binary_command",
        ),
        objectives=replace(
            base.objectives,
            gripper_command=0.1,
            calvin_object_binding=0.05,
            calvin_object_binding_action_compatibility=0.05,
        ),
    )
    config.validate()
    return config


def test_masked_pointer_has_exact_null_fallback_and_masks_invalid_slots():
    logits = torch.tensor([[1.0, 5.0, -2.0]])
    null = torch.tensor([-1.0])
    valid = torch.tensor([[True, False, True]])
    pointer = CalvinObjectBindingBridge._masked_pointer(logits, null, valid)
    assert pointer.shape == (1, 4)
    assert pointer[0, 1].item() == 0.0
    assert torch.allclose(pointer.sum(-1), torch.ones(1))

    none = CalvinObjectBindingBridge._masked_pointer(logits, null, torch.zeros_like(valid))
    assert torch.equal(none, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_forward_masks_invalid_and_all_invalid_has_zero_context():
    torch.manual_seed(3)
    bridge = CalvinObjectBindingBridge(hidden=8, route_dim=3)
    validity = torch.tensor(
        [[[1.0], [0.0], [1.0], [1.0]], [[0.0], [0.0], [0.0], [0.0]]]
    )
    facts = _facts(validity=validity)
    result, _ = bridge(
        protected_goal=torch.randn(2, 3, 8),
        history_tokens=torch.randn(2, 2, 8),
        object_tokens=torch.randn(2, 4, 8),
        facts=facts,
    )
    assert result.pointer[0, 1].item() == 0.0
    assert torch.equal(result.pointer[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(result.selected_context[1], torch.zeros_like(result.selected_context[1]))
    assert torch.equal(result.selected_geometry[1], torch.zeros_like(result.selected_geometry[1]))


def test_binding_is_k_permutation_equivariant_and_context_invariant():
    torch.manual_seed(4)
    bridge = CalvinObjectBindingBridge(hidden=8, route_dim=3)
    facts = _facts()
    goal = torch.randn(2, 3, 8)
    history = torch.randn(2, 2, 8)
    objects = torch.randn(2, 4, 8)
    result, _ = bridge(
        protected_goal=goal,
        history_tokens=history,
        object_tokens=objects,
        facts=facts,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _ = bridge(
        protected_goal=goal,
        history_tokens=history,
        object_tokens=objects[:, permutation],
        facts=facts.permute(permutation),
    )
    assert torch.allclose(permuted.pointer[:, :4], result.pointer[:, permutation], atol=1e-6)
    assert torch.allclose(permuted.pointer[:, 4], result.pointer[:, 4], atol=1e-6)
    assert torch.allclose(permuted.selected_context, result.selected_context, atol=1e-6)
    assert torch.allclose(permuted.selected_geometry, result.selected_geometry, atol=1e-6)


def test_language_and_position_interventions_change_pointer_and_context():
    torch.manual_seed(5)
    bridge = CalvinObjectBindingBridge(hidden=8, route_dim=3)
    facts = _facts()
    objects = torch.randn(2, 4, 8)
    history = torch.zeros(2, 2, 8)
    goal_a = torch.randn(2, 3, 8)
    result_a, _ = bridge(
        protected_goal=goal_a,
        history_tokens=history,
        object_tokens=objects,
        facts=facts,
    )
    result_b, _ = bridge(
        protected_goal=-goal_a,
        history_tokens=history,
        object_tokens=objects,
        facts=facts,
    )
    assert not torch.allclose(result_a.pointer, result_b.pointer)
    assert not torch.allclose(result_a.selected_context, result_b.selected_context)

    moved_coordinates = facts.camera_coordinates.clone()
    moved_coordinates[:, [0, 3]] = moved_coordinates[:, [3, 0]]
    moved_facts = replace(facts, camera_coordinates=moved_coordinates)
    moved, _ = bridge(
        protected_goal=goal_a,
        history_tokens=history,
        object_tokens=objects,
        facts=moved_facts,
    )
    assert not torch.allclose(result_a.pointer, moved.pointer)


def test_action_boundary_backpropagates_to_every_binding_owner():
    torch.manual_seed(6)
    bridge = CalvinObjectBindingBridge(hidden=8, route_dim=3)
    coarse = CoarseActionIntent(hidden=8, action_dim=7, heads=2)
    facts = _facts(batch=2, hidden=8, route=3)
    object_tokens = torch.randn(2, 4, 8)
    result, _ = bridge(
        protected_goal=torch.randn(2, 3, 8),
        history_tokens=torch.randn(2, 2, 8),
        object_tokens=object_tokens,
        facts=facts,
    )
    dock = ActionIntentDock(
        public_interval_carrier=torch.randn(2, 4, 8),
        history_memory=torch.randn(2, 2, 8),
        public_object_memory=object_tokens,
        selected_object_context=result.selected_context,
        object_binding_pointer=result.pointer,
        object_binding_selected_geometry=result.selected_geometry,
    )
    coarse(dock).action_prediction.float().square().mean().backward()
    missing = []
    for name, parameter in bridge.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            missing.append(name)
    assert not missing, f"binding owners missing action-level VJP: {missing}"
    assert bridge.context_gate.grad is not None
    assert bridge.context_gate.grad.abs().item() > 0.0


def test_binding_supervised_loss_accepts_soft_correspondence_and_coverage():
    pointer = torch.tensor([[0.6, 0.2, 0.1, 0.05, 0.05]], requires_grad=True)
    target = torch.tensor([[0.2, 0.6, 0.1, 0.05, 0.05]])
    loss = CalvinObjectBindingBridge.supervised_loss(pointer, target, torch.tensor([[0.8]]))
    assert loss.item() > 0.0
    loss.backward()
    assert pointer.grad is not None and torch.isfinite(pointer.grad).all()


def test_sidecar_loads_and_rejects_missing_or_duplicate_window_keys(tmp_path):
    path = tmp_path / "targets.npz"
    np.savez(
        path,
        schema=np.asarray(CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA),
        episode_ids=np.asarray(["episode_0", "episode_0"]),
        centers=np.asarray([24, 25], dtype=np.int64),
        pointer_targets=np.asarray(
            [[1, 0, 0, 0, 0], [0.1, 0.8, 0, 0, 0.1]], dtype=np.float32
        ),
        coverage=np.asarray([1.0, 0.8], dtype=np.float32),
        ambiguity=np.asarray([0.0, 0.25], dtype=np.float32),
        source_digest=np.asarray(calvin_episode_inventory_digest(["episode_0"])),
        manifest_digest=np.asarray("manifest-v1"),
    )
    sidecar = CalvinObjectBindingSidecar.load(path)
    assert CalvinObjectBindingSidecar.load(
        path,
        expected_source_digest=calvin_episode_inventory_digest(["episode_0"]),
        expected_manifest_digest="manifest-v1",
    ).rows == 2
    sidecar.require_complete([("episode_0", 24), ("episode_0", 25)])
    pointer, coverage, ambiguity = sidecar.lookup("episode_0", 25)
    assert pointer.shape == (5,)
    assert coverage.item() == pytest.approx(0.8)
    assert ambiguity.item() == pytest.approx(0.25)
    with pytest.raises(KeyError, match="missing target rows"):
        sidecar.require_complete([("episode_0", 26)])

    duplicate = tmp_path / "duplicate.npz"
    np.savez(
        duplicate,
        schema=np.asarray(CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA),
        episode_ids=np.asarray(["episode_0", "episode_0"]),
        centers=np.asarray([24, 24], dtype=np.int64),
        pointer_targets=np.asarray([[1, 0, 0, 0, 0]] * 2, dtype=np.float32),
        coverage=np.ones(2, dtype=np.float32),
        ambiguity=np.zeros(2, dtype=np.float32),
        source_digest=np.asarray(calvin_episode_inventory_digest(["episode_0"])),
        manifest_digest=np.asarray("manifest-v1"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        CalvinObjectBindingSidecar.load(duplicate)


def test_training_target_is_calvin_only_and_discounts_ambiguity():
    config = _binding_config()
    target = CalvinObjectBindingTarget(
        pointer=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        coverage=torch.tensor([[0.8]]),
        ambiguity=torch.tensor([[0.25]]),
        episode_index=torch.tensor([0]),
        center=torch.tensor([24]),
    )
    target.validate(config, device=torch.device("cpu"))
    assert target.effective_weight().item() == pytest.approx(0.6)
    with pytest.raises(ValueError, match="legal only"):
        target.validate(ExperimentConfig(), device=torch.device("cpu"))


def test_calvin_config_constructs_binding_child_and_non_calvin_has_no_parameters():
    config = load_config("configs/mainline/calvin_object_binding_formal_v1.json")
    policy = ClearVLAMainlinePolicy(config)
    assert policy.selection.intent == CALVIN_OBJECT_BINDING_COMPONENT
    assert policy.intent.calvin_object_binding is not None
    assert any("calvin_object_binding" in name for name, _ in policy.named_parameters())

    baseline = ClearVLAMainlinePolicy(ExperimentConfig())
    assert baseline.intent.calvin_object_binding is None
    assert not any("calvin_object_binding" in name for name, _ in baseline.named_parameters())
    assert not any("calvin_object_binding" in name for name in baseline.state_dict())
    assert ComponentSelection.from_config(ExperimentConfig()).intent == "stateless_object_intent_v1"


def test_binding_config_fails_closed_without_sidecar_or_both_objectives():
    config = _binding_config()
    with pytest.raises(ValueError, match="requires data.calvin_object_binding_sidecar"):
        replace(config, data=replace(config.data, calvin_object_binding_sidecar="")).validate()
    with pytest.raises(ValueError, match="pointer and action-compatibility"):
        replace(
            config,
            objectives=replace(config.objectives, calvin_object_binding_action_compatibility=0.0),
        ).validate()
    with pytest.raises(ValueError, match="only for the CALVIN"):
        replace(
            config,
            data=replace(
                config.data,
                data_profile="identity_7d_pen",
                split_mode="ordered-counts",
                split_manifest="",
                train_episodes=63,
                val_episodes=5,
                test_episodes=5,
                sampling_gripper_event_threshold=None,
            ),
            bottom=replace(
                config.bottom,
                arm_flow_mode="legacy_independent",
                gripper_output_mode="continuous",
            ),
            objectives=replace(config.objectives, gripper_command=0.0),
        ).validate()
