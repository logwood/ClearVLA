"""One training-only semantic change; deployment and unrelated owners stay fixed."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.interfaces import ActionSupervision
from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
from clearvla.mainline.training.losses import FlowMatchingState, action_terms

ROOT = Path(__file__).resolve().parents[1]


def _terms(mode, field, target, state):
    base = ExperimentConfig()
    config = replace(base, objectives=replace(
        base.objectives, gripper_persistence_mode=mode,
        action_frame_weight_mode="event_motion_v1",
        action_frame_event_gain=0.75, action_frame_motion_gain=0.35,
    ))
    codec = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    encoded = codec.encode(target, state)
    zero = torch.zeros_like(encoded)
    return action_terms(
        config, codec,
        SimpleNamespace(bottom=SimpleNamespace(
            physical_velocity=field, motion_logits=field[..., 0] * 0,
            decoder_tensors={},
        )),
        ActionSupervision(normalized=target, raw_units=target,
                          current_raw_units=state,
                          gripper_transition_boundary=state,
                          gripper_transition_boundary_raw_units=state),
        SimpleNamespace(action_state=state),
        FlowMatchingState(time=state[:, 0] * 0, source_physical_noise=zero,
                          noisy_physical=zero, target_physical=encoded,
                          target_physical_velocity=encoded),
    )


def test_preset_changes_only_persistence_and_output_path():
    config_dir = ROOT / "configs/mainline"
    base = load_config(config_dir / "object_intent_dynamics_323_pen_shared_v1.json")
    ablation = load_config(config_dir / "object_intent_dynamics_323_pen_anchored_v1.json")
    expected = replace(base,
        objectives=replace(base.objectives, gripper_persistence_mode="anchored_training"),
        data=replace(base.data, output_dir=ablation.data.output_dir),
    )
    assert ablation == expected
    assert base.objectives.gripper_persistence_mode == "deployment_cumulative"
    assert base.digest() != ablation.digest()
    assert config_from_mapping(ablation.as_dict()) == ablation
    rdt = load_config(config_dir / "object_intent_dynamics_323_rdt_shared_v1.json")
    assert rdt.objectives.gripper_persistence_mode == "deployment_cumulative"
    with pytest.raises(ValueError, match="Pen-only"):
        replace(rdt, objectives=replace(rdt.objectives,
            gripper_persistence_mode="anchored_training")).validate()
    with pytest.raises(ValueError, match="gripper_persistence_mode"):
        replace(base, objectives=replace(base.objectives,
            gripper_persistence_mode="typo")).validate()


def test_only_auxiliary_persistence_changes_and_pre_event_error_is_not_hidden_in_decode():
    codec = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    state = torch.zeros(1, 7)
    target = torch.zeros(1, 24, 7)
    target[:, 2:, -1] = 1.0
    field = codec.encode(target, state)
    field[:, 0, 13] += 0.5
    cumulative = _terms("deployment_cumulative", field, target, state)
    anchored = _terms("anchored_training", field, target, state)
    for key in cumulative:
        if key.startswith("gripper_trajectory") or key == "gripper_persistence_mode_code":
            continue
        assert torch.equal(cumulative[key], anchored[key]), key
    assert cumulative["gripper_trajectory_persistence"] > 0
    assert anchored["gripper_trajectory_persistence"] == 0
    assert cumulative["decoded_action"] == anchored["decoded_action"] > 0
    assert cumulative["gripper_trajectory_transition"] == anchored["gripper_trajectory_transition"]
    # The target event is NOT a runtime reset. Both modes decode the same error.
    decoded = codec.decode(field, state)
    torch.testing.assert_close(decoded[:, 2:, -1], target[:, 2:, -1] + 0.125)
    assert cumulative["gripper_persistence_mode_code"] == 0
    assert anchored["gripper_persistence_mode_code"] == 1


@pytest.mark.parametrize("mode", ["deployment_cumulative", "anchored_training"])
@pytest.mark.parametrize("events", [False, True])
def test_finite_loss_and_value_delta_only_backward(mode, events):
    torch.manual_seed(9141)
    codec = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    state = torch.zeros(2, 7)
    target = torch.zeros(2, 24, 7)
    if events:
        target[:, 3:13, -1] = 1.0
        target[:, 13:, -1] = -0.5
    field = (codec.encode(target, state) + 0.03 * torch.randn(2, 24, 18)).requires_grad_()
    terms = _terms(mode, field, target, state)
    assert all(torch.isfinite(value).all() for value in terms.values())
    terms["gripper_trajectory"].backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert torch.count_nonzero(field.grad[..., :12]) == 0
    assert torch.count_nonzero(field.grad[..., 14:]) == 0
    if events:
        assert field.grad[..., 12].abs().sum() > 0
        assert field.grad[..., 13].abs().sum() > 0
    else:
        assert terms["gripper_trajectory"] == 0
        assert torch.count_nonzero(field.grad) == 0
        # Hold rows remain trained by the unchanged main action objective.
        assert terms["action_flow"] > 0
        assert terms["decoded_action"] > 0


def test_mode_does_not_change_model_state_rng_or_two_pass_sampling(tmp_path):
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.sampling import sample_action
    from tests.test_mainline_policy import _batch, _config

    config = _config()
    anchored = replace(config, objectives=replace(config.objectives,
        gripper_persistence_mode="anchored_training"))
    torch.manual_seed(9142)
    base = ClearVLAMainlinePolicy(config).eval()
    rng_base = torch.get_rng_state().clone()
    torch.manual_seed(9142)
    candidate = ClearVLAMainlinePolicy(anchored).eval()
    assert torch.equal(rng_base, torch.get_rng_state())
    assert base.state_dict().keys() == candidate.state_dict().keys()
    for key, value in base.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    batch = _batch(config)
    with torch.no_grad():
        a = sample_action(base, batch.online, config,
                          generator=torch.Generator().manual_seed(9143))
        b = sample_action(candidate, batch.online, anchored,
                          generator=torch.Generator().manual_seed(9143))
    assert torch.equal(a.action, b.action)
    assert torch.equal(a.physical_field, b.physical_field)
    assert torch.equal(a.motion_logits, b.motion_logits)
    path = tmp_path / "model-roundtrip.pt"
    torch.save(candidate.state_dict(), path)
    restored = ClearVLAMainlinePolicy(anchored).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        c = sample_action(restored, batch.online, anchored,
                          generator=torch.Generator().manual_seed(9143))
    assert torch.equal(c.action, b.action)


def test_anchored_full_forward_ledger_and_backward():
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.training.engine import MainlineTrainingEngine
    from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer
    from tests.test_mainline_policy import _batch, _config

    config = _config()
    config = replace(config, objectives=replace(config.objectives,
        gripper_persistence_mode="anchored_training"))
    torch.manual_seed(9144)
    model = ClearVLAMainlinePolicy(config).train()
    optimizer, _ = build_optimizer(model, config)
    engine = MainlineTrainingEngine(model=model, config=config, optimizer=optimizer,
        schedule=WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1),
        device=torch.device("cpu"), dtype=torch.float32)
    ledger, _ = engine._forward(_batch(config), training=True,
        collect_diagnostics=True, generator=torch.Generator().manual_seed(9145))
    assert ledger.terms["gripper_persistence_mode_code"] == 1
    torch.testing.assert_close(sum(ledger.contributions.values()), ledger.total)
    torch.testing.assert_close(sum(ledger.groups.values()), ledger.total)
    ledger.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    head = model.bottom.decoder.velocity_head
    for layer in (head.grip_value, head.grip_delta, head.gripper_gate):
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0
