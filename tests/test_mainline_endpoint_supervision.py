"""Endpoint head training uses the deployed numerical context, not a sixth ODE step."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_mainline_controller_values import _config as base_config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.endpoint_supervision import (
    CLEAN_ENDPOINT_SUPERVISION,
    EndpointHeadSupervision,
    clean_endpoint_field,
    endpoint_condition,
    endpoint_supervision_metadata,
)
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.flow_schedule import DeploymentFlowSchedule
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.losses import FlowMatchingState, action_terms
from clearvla.mainline.v120_core.controller import EvidenceExecutionController

Production = tuple[ExperimentConfig, ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch]


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(
        c,
        bottom=replace(c.bottom, endpoint_supervision_mode=CLEAN_ENDPOINT_SUPERVISION),
        runtime=replace(
            c.runtime,
            deployment_flow_schedule=DeploymentFlowSchedule.same_nfe_power_five().to_dict(),
        ),
    )


def _flow(rows: int = 24) -> FlowMatchingState:
    target = torch.randn(2, 24, 18, requires_grad=True)
    noise = torch.randn_like(target, requires_grad=True)
    mask = (torch.arange(24)[None] < rows).expand(2, -1)
    return FlowMatchingState(
        torch.full((2,), 0.3), noise, noise, target, target - noise, row_valid=mask
    )


def test_config_roundtrip_and_default_omission() -> None:
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    defaults = base_config().as_dict()["bottom"]
    assert isinstance(defaults, dict)
    assert "endpoint_supervision_mode" not in defaults
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m6m_calvin.json"
        ).bottom.endpoint_supervision_mode
        == CLEAN_ENDPOINT_SUPERVISION
    )


@pytest.mark.parametrize("change", ["unknown", "continuous"])
def test_invalid_endpoint_graph_rejected(change: str) -> None:
    c = _config()
    b = (
        replace(c.bottom, endpoint_supervision_mode="guess")
        if change == "unknown"
        else replace(c.bottom, gripper_output_mode="continuous")
    )
    with pytest.raises(ValueError):
        replace(c, bottom=b).validate()


@pytest.mark.parametrize("context", [True, False])
def test_endpoint_definition_is_exact(context: bool) -> None:
    t, ctx = endpoint_condition(torch.randn(2, 24, 18), context_enabled=context)
    assert t.dtype == torch.float32 and torch.equal(t, torch.ones(2))
    if context:
        assert ctx is not None
        assert torch.equal(ctx.step_size, torch.zeros(2))
        assert torch.equal(ctx.normalized_index, torch.ones(2)) and torch.equal(
            ctx.endpoint, torch.ones(2)
        )
    else:
        assert ctx is None


@pytest.mark.parametrize("rows", [0, 1, 7, 24])
def test_clean_endpoint_keeps_noise_for_unknown_and_binary_lanes(rows: int) -> None:
    f = _flow(rows)
    out = clean_endpoint_field(f, arm_dim=6)
    assert not out.requires_grad
    torch.testing.assert_close(out[:, :rows, :12], f.target_physical[:, :rows, :12], rtol=0, atol=0)
    torch.testing.assert_close(
        out[:, rows:, :12], f.source_physical_noise[:, rows:, :12], rtol=0, atol=0
    )
    torch.testing.assert_close(out[..., 12:], f.source_physical_noise[..., 12:], rtol=0, atol=0)


def test_unknown_labels_and_gripper_labels_cannot_poison_endpoint_input() -> None:
    f = _flow(3)
    t = f.target_physical.detach().clone()
    t[:, 3:] = float("nan")
    t[..., 12:] = float("nan")
    out = clean_endpoint_field(replace(f, target_physical=t), arm_dim=6)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, clean_endpoint_field(f, arm_dim=6), rtol=0, atol=0)


@pytest.fixture(scope="module")
def production() -> Production:
    torch.manual_seed(931)
    c = _config()
    m, e = _model_engine(c)
    b = _batch()
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    return c, m, e, b


def test_engine_has_one_online_encode_two_velocity_roles(
    production: Production, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, m, e, b = production
    records = []
    encodes = []
    original = m.velocity
    original_encode = m.encode_online

    def record(cache, **kw):
        result = original(cache, **kw)
        records.append((cache, kw, result))
        return result

    def encode(*a, **kw):
        encodes.append(1)
        return original_encode(*a, **kw)

    monkeypatch.setattr(m, "velocity", record)
    monkeypatch.setattr(m, "encode_online", encode)
    ledger, _ = e._forward(
        b, training=False, collect_diagnostics=False, generator=torch.Generator().manual_seed(6)
    )
    assert len(records) == 2 and len(encodes) == 1 and records[0][0] is records[1][0]
    assert records[0][1]["require_execution_supervision"] is True
    assert records[1][1]["require_execution_supervision"] is False
    kw = records[1][1]
    t, ctx = endpoint_condition(kw["noisy_action_field"], context_enabled=True)
    assert torch.equal(kw["time"], t) and kw["flow_step_context"] is not None and ctx is not None
    for n in ("time", "step_size", "normalized_index", "endpoint"):
        torch.testing.assert_close(
            getattr(kw["flow_step_context"], n), getattr(ctx, n), rtol=0, atol=0
        )
    assert not kw["noisy_action_field"].requires_grad and torch.isfinite(ledger.total)
    # Directly differentiate: command objective belongs to endpoint logits,
    # not interior command logits; physical flow remains the interior output.
    interior, terminal = records[0][2], records[1][2]
    assert (
        terminal.bottom.gripper_command_logits is not None
        and interior.bottom.gripper_command_logits is not None
    )
    g = torch.autograd.grad(
        ledger.terms["gripper_command"],
        (interior.bottom.gripper_command_logits, terminal.bottom.gripper_command_logits),
        allow_unused=True,
        retain_graph=True,
    )
    assert g[0] is None and g[1] is not None and g[1].abs().sum() > 0
    g = torch.autograd.grad(
        ledger.terms["action_flow"],
        (interior.bottom.physical_velocity, terminal.bottom.physical_velocity),
        allow_unused=True,
    )
    assert g[0] is not None and g[1] is None


@pytest.mark.parametrize("center", [0, 79])
def test_endpoint_train_step_covers_real_tail_and_reset(center: int) -> None:
    torch.manual_seed(932)
    m, e = _model_engine(_config())
    out = e.train_step(_batch(center), collect_diagnostics=True)
    assert torch.isfinite(out.loss)
    # Source and label streams remain strict despite a short tail.
    assert out.metrics["training_endpoint_head_calls"] == 1


def test_diagnostics_do_not_switch_endpoint_values_or_loss(production: Production) -> None:
    c, m, e, b = production
    before = torch.random.get_rng_state().clone()
    rows = []
    for diagnostics in (False, True):
        torch.random.set_rng_state(before)
        ledger, _ = e._forward(
            b,
            training=False,
            collect_diagnostics=diagnostics,
            generator=torch.Generator().manual_seed(7),
        )
        rows.append({k: v.detach().clone() for k, v in ledger.contributions.items()})
    assert rows[0].keys() == rows[1].keys()
    for key in rows[0]:
        torch.testing.assert_close(rows[0][key], rows[1][key], rtol=0, atol=0)


def test_sampler_still_uses_two_endpoint_calls_no_extra_velocity(
    production: Production, monkeypatch: pytest.MonkeyPatch
) -> None:
    c, m, e, b = production
    records = []
    original = m.velocity

    def record(cache, **kw):
        records.append(kw)
        return original(cache, **kw)

    monkeypatch.setattr(m, "velocity", record)
    result = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(10))
    assert len(records) == 12 and torch.isfinite(result.action).all()
    endpoints = [x for x in records if torch.equal(x["time"], torch.ones_like(x["time"]))]
    assert len(endpoints) == 2
    for x in endpoints:
        assert torch.equal(x["flow_step_context"].endpoint, torch.ones(1))


def test_cpu_bf16_real_backward() -> None:
    c = _config()
    c = replace(c, runtime=replace(c.runtime, compute_dtype="bf16"))
    m, e = _model_engine(c)
    result = e.train_step(_batch(), collect_diagnostics=True)
    assert torch.isfinite(result.loss)


def test_endpoint_loss_requires_selected_record(production: Production) -> None:
    c, m, e, b = production
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    f = _flow()
    output = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
    with pytest.raises(ValueError, match="endpoint supervision"):
        action_terms(c, m.outlet_adapter, output, b.action_target, b.online.history, f)
    with pytest.raises(ValueError, match="exact clean endpoint"):
        EndpointHeadSupervision(output, torch.tensor([0.4]), None).validate()


def test_checkpoint_and_ABI(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import test_mainline_target_binding as original

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    build = original.build_deployment_abi

    def checked(*args, **kwargs):
        abi = build(*args, **kwargs)
        assert abi["endpoint_supervision"] == endpoint_supervision_metadata()
        bad = copy.deepcopy(abi)
        del bad["endpoint_supervision"]
        with pytest.raises(ValueError, match="endpoint supervision"):
            validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(original, "_config", _config)
    monkeypatch.setattr(original, "_batch", _batch)
    monkeypatch.setattr(original, "build_deployment_abi", checked)
    original.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_unsupported_endpoint_logits_are_quarantined_before_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Masking loss values after CE/BCE is too late for NaN-safe gradients."""
    torch.manual_seed(941)
    m, e = _model_engine(_config())
    b = _batch(79)
    valid = b.action_target.row_valid
    assert valid is not None and valid.any() and (~valid).any()
    original = m.velocity
    leaves: list[torch.Tensor] = []

    def record(cache, **kw):
        out = original(cache, **kw)
        if torch.equal(kw["time"], torch.ones_like(kw["time"])):
            assert out.bottom.gripper_command_logits is not None
            cmd = out.bottom.gripper_command_logits.detach().clone()
            motion = out.bottom.motion_logits.detach().clone()
            cmd[~valid] = float("nan")
            motion[~valid] = float("nan")
            cmd.requires_grad_()
            motion.requires_grad_()
            leaves.extend([cmd, motion])
            return replace(
                out, bottom=replace(out.bottom, gripper_command_logits=cmd, motion_logits=motion)
            )
        return out

    monkeypatch.setattr(m, "velocity", record)
    ledger, _ = e._forward(
        b, training=False, collect_diagnostics=True, generator=torch.Generator().manual_seed(12)
    )
    assert torch.isfinite(ledger.total)
    loss = ledger.terms["gripper_command"] + ledger.terms["motion"]
    gradients = torch.autograd.grad(loss, leaves)
    for grad in gradients:
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad[~valid]) == 0
        assert grad[valid].abs().sum() > 0


def test_endpoint_diagnostics_gradients_and_global_rng_match(production: Production) -> None:
    _, m, e, b = production
    results = []
    state = torch.random.get_rng_state().clone()
    params = tuple(p for p in m.parameters() if p.requires_grad)
    for diagnostics in (False, True):
        torch.random.set_rng_state(state)
        ledger, _ = e._forward(
            b,
            training=False,
            collect_diagnostics=diagnostics,
            generator=torch.Generator().manual_seed(14),
        )
        grads = torch.autograd.grad(ledger.total, params, allow_unused=True)
        results.append(
            (
                tuple(None if g is None else g.detach().clone() for g in grads),
                torch.random.get_rng_state().clone(),
            )
        )
    assert torch.equal(results[0][1], results[1][1])
    for a, b_grad in zip(results[0][0], results[1][0]):
        if a is None:
            assert b_grad is None
        else:
            assert b_grad is not None
            torch.testing.assert_close(a, b_grad, rtol=0, atol=0)


def test_combined_endpoint_and_mature_controller_have_live_parameter_gradients() -> None:
    """Cross the existing capacity warm-up, without editing parameters by hand."""
    torch.manual_seed(942)
    m, e = _model_engine(_config())
    e.global_step = 201
    for _ in range(2):
        result = e.train_step(_batch(), collect_diagnostics=True)
        assert torch.isfinite(result.loss)
    controller = m.execution_bottom.execution
    assert isinstance(controller, EvidenceExecutionController)
    groups = {
        "source_value": controller.value_proj,
        "capacity_read": controller.operation_value,
        "capacity_head": controller.capacity_head,
        "candidate_memory": controller.value_reader.memory_attention,
        "candidate_score": controller.value_reader.value_head,
    }
    for name, module in groups.items():
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.abs().sum()) for g in gradients if g is not None) > 0, name
