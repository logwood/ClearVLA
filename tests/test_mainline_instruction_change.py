"""Typed causal S/P3 comparison; external image/T5 transport alone is synthetic."""

from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_robot_execution import _batch
from test_mainline_robot_execution import _config as base_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.instruction_change import (
    TYPED_REFERENCE_CHANGE,
    InstructionChangeEvidence,
    instruction_change_metadata,
)
from clearvla.mainline.instruction_reference import InstructionReference
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.instruction_change import (
    InstructionChangePlanRead,
    TypedChangeValueRead,
    TypedInstructionReferenceRead,
)
from clearvla.mainline.model.policy import (
    ClearVLAMainlinePolicy,
    OnlinePolicyCache,
    OnlineTrainingState,
)
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import MainlineTrainingEngine


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, top=replace(c.top, instruction_change_mode=TYPED_REFERENCE_CHANGE))


def _evidence() -> InstructionChangeEvidence:
    torch.manual_seed(6792)
    binding = TargetBinding.from_logits(
        torch.randn(2, 3), torch.randn(2, 1), torch.ones(2, 3, dtype=torch.bool)
    )
    state = torch.randn(2, 10)
    ref = InstructionReference(
        torch.randn(2, 2, 64, 16),
        state - 0.1,
        torch.ones(2, 2, 64, dtype=torch.bool),
        torch.tensor([10, 20]),
    )
    return InstructionChangeEvidence(
        torch.randn(2, 3, 2, 16),
        torch.randn(2, 3, 2, 2),
        state - ref.state,
        torch.randn(2, 3, 2, 6),
        torch.ones(2, 3, 2, dtype=torch.bool),
        torch.full((2, 3, 2), 0.5),
        binding,
        state,
        ref,
        ("top", "wrist"),
    )


def _values(names: tuple[str, ...] = ("top", "wrist")) -> TypedChangeValueRead:
    return TypedChangeValueRead(hidden=16, content_dim=16, state_dim=10, camera_names=names)


def _plan() -> InstructionChangePlanRead:
    return InstructionChangePlanRead(
        hidden=16, content_dim=16, state_dim=10, camera_names=("top", "wrist")
    )


Updated = tuple[
    ClearVLAMainlinePolicy,
    MainlineTrainingEngine,
    TrainingBatch,
    OnlinePolicyCache,
    OnlineTrainingState,
]


@pytest.fixture(scope="module")
def updated() -> Updated:
    torch.manual_seed(6790)
    m, e = _model_engine(_config())
    b = _batch()
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
    return m, e, b, cache, state


def test_explicit_config_and_source_closure():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "instruction_change_mode" not in cast(dict, base_config().as_dict()["top"])
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m6h_calvin.json"
        ).top.instruction_change_mode
        == TYPED_REFERENCE_CHANGE
    )
    paths = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/model/instruction_change.py" in paths
    assert "clearvla/mainline/instruction_change.py" in paths


@pytest.mark.parametrize("change", ["unknown", "no_reference", "local_target", "pointwise"])
def test_config_rejects_incoherent_consumers(change: str) -> None:
    c = _config()
    fields = {
        "unknown": {"instruction_change_mode": "guess"},
        "no_reference": {"instruction_reference_mode": "none"},
        "local_target": {"target_binding_mode": "reader_local_v1"},
        "pointwise": {"p3_coordination_mode": "pointwise_legacy_v1"},
    }
    with pytest.raises(ValueError):
        replace(c, top=replace(c.top, **fields[change])).validate()


@pytest.mark.parametrize(
    "bad", ["image_axis", "status_axis", "mask_type", "view_mass", "named_view", "dtype"]
)
def test_evidence_rejects_chart_corruption(bad: str) -> None:
    e = _evidence()
    fields = {
        "image_axis": {"image_delta": e.image_delta[:, :, 0]},
        "status_axis": {"match_status": e.match_status[..., :4]},
        "mask_type": {"view_observed": e.view_observed.float()},
        "view_mass": {"view_weight": e.view_weight * 2},
        "named_view": {"camera_names": ("top", "top")},
        "dtype": {"content_delta": e.content_delta.bfloat16()},
    }
    with pytest.raises((ValueError, TypeError)):
        replace(e, **fields[bad]).validate(strict=True)


def test_matcher_has_no_direction_or_elapsed_time_input():
    assert set(inspect.signature(TypedInstructionReferenceRead.forward).parameters) == {
        "self",
        "reference",
        "current_dino",
        "current_state",
        "facts",
        "binding",
    }
    assert "age_steps" not in inspect.getsource(TypedInstructionReferenceRead.forward)


@pytest.mark.parametrize("case", ["identical", "missing_pixels", "robot_only"])
def test_observation_types_remain_separate(updated: Updated, case: str) -> None:
    m, _, b, cache, st = updated
    reader = m.intent.organizer.instruction_progress
    assert isinstance(reader, TypedInstructionReferenceRead)
    ref = b.online.instruction_reference
    binding = cache.top.intent.target_binding
    assert ref is not None and binding is not None
    dino = b.online.observation.dino_history[:, -1]
    observed = ref.observed.clone()
    if case == "missing_pixels":
        observed[..., ::2] = False
    start = torch.where(observed[..., None], dino, torch.full_like(dino, float("nan")))
    state = b.online.history.state
    ref = replace(
        ref,
        dino=start,
        observed=observed,
        state=state.clone() if case != "robot_only" else state - 0.2,
    )
    _, ev = reader(
        reference=ref, current_dino=dino, current_state=state, facts=st.top.facts, binding=binding
    )
    assert ev.content_delta.count_nonzero() == ev.image_delta.count_nonzero() == 0
    assert ev.match_status.count_nonzero() > 0
    if case == "robot_only":
        assert ev.robot_delta.count_nonzero() > 0
    else:
        assert ev.robot_delta.count_nonzero() == 0
        plan = m.policy_compiler.plan_compiler.instruction_change_read
        assert plan is not None
        q = torch.randn(1, 24, 3, m.config.dimensions.hidden_size, requires_grad=True)
        out = plan(ev, q)
        out.sum().backward()
        assert out.count_nonzero() == 0 and q.grad is not None and q.grad.count_nonzero() == 0
    if case == "missing_pixels":
        assert not torch.equal(ev.match_status[..., 3], ev.match_status[..., 4])


def test_location_change_does_not_require_appearance_change():
    r = TypedInstructionReferenceRead(
        hidden=16, content_dim=16, state_dim=10, camera_names=("top", "wrist")
    )
    with torch.no_grad():
        r.key.weight.copy_(torch.eye(16))
    image = torch.zeros(1, 2, 64, 16)
    image[:, :, 0, :] = 8
    after = torch.roll(image, shifts=7, dims=2)
    q = torch.ones(1, 1, 16)
    support = torch.ones(1, 2, 64, dtype=torch.bool)
    c0, x0, _ = r._read(q, image, support)
    c1, x1, _ = r._read(q, after, support)
    torch.testing.assert_close(c0, c1, rtol=0, atol=0)
    assert (x1 - x0)[..., 0].min() > 1.9 and (x1 - x0)[..., 1].abs().max() < 1e-5


def test_target_binding_changes_read_not_measurement(updated: Updated) -> None:
    m, _, b, cache, st = updated
    r = m.intent.organizer.instruction_progress
    assert isinstance(r, TypedInstructionReferenceRead)
    binding = cache.top.intent.target_binding
    ref = b.online.instruction_reference
    assert binding is not None and ref is not None
    new = TargetBinding.from_logits(
        torch.tensor([[9.0, -9.0, -9.0, -9.0]]), torch.tensor([[-9.0]]), binding.supported
    )
    kw = dict(
        reference=ref,
        current_dino=b.online.observation.dino_history[:, -1],
        current_state=b.online.history.state,
        facts=st.top.facts,
    )
    x, a = r(**kw, binding=binding)
    y, z = r(**kw, binding=new)
    for n in ("content_delta", "image_delta", "robot_delta", "match_status"):
        torch.testing.assert_close(getattr(a, n), getattr(z, n), rtol=0, atol=0)
    assert not torch.equal(x, y)


def test_view_interpretation_precedes_raw_cancellation():
    e, r = _evidence(), _values()
    with torch.no_grad():
        r.image.weight.fill_(1)
        r.view_role.weight[:, 0].fill_(torch.atanh(torch.tensor(0.5)))
        r.view_role.weight[:, 1].fill_(-torch.atanh(torch.tensor(0.5)))
    xy = torch.zeros_like(e.image_delta)
    xy[:, :, 0, 0] = 0.25
    xy[:, :, 1, 0] = -0.25
    assert (xy * e.view_weight[..., None]).sum(2).count_nonzero() == 0
    assert r(replace(e, image_delta=xy))[1].abs().min() > 0


def test_null_mass_is_not_removed_by_value_read():
    e, r = _evidence(), _values()
    original = r(e)
    mass = e.binding.mass * 0.5
    law = TargetBinding(
        torch.cat((mass, 1 - mass.sum(-1, keepdim=True)), -1).log(), e.binding.supported
    )
    for a, b in zip(original, r(replace(e, binding=law))):
        torch.testing.assert_close(b, a * 0.5, rtol=1e-6, atol=1e-6)
    null = TargetBinding.from_logits(
        torch.zeros_like(mass), torch.zeros(2, 1), torch.zeros_like(e.binding.supported)
    )
    empty = replace(
        e,
        binding=null,
        view_observed=torch.zeros_like(e.view_observed),
        view_weight=torch.zeros_like(e.view_weight),
    )
    assert all(x.count_nonzero() == 0 for x in r(empty))


@pytest.mark.parametrize("bf16", [False, True])
def test_status_cannot_create_P3_change_or_context_gradient(bf16: bool) -> None:
    p, e = _plan(), _evidence()
    q = torch.randn(2, 24, 3, 16, requires_grad=True)
    status = e.match_status.clone().requires_grad_()
    e = replace(
        e,
        content_delta=torch.zeros_like(e.content_delta),
        image_delta=torch.zeros_like(e.image_delta),
        robot_delta=torch.zeros_like(e.robot_delta),
        match_status=status,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out = p(e, q)
    out.float().sum().backward()
    assert out.count_nonzero() == 0 and q.grad is not None and q.grad.count_nonzero() == 0
    assert status.grad is not None and status.grad.count_nonzero() == 0


def test_invalid_view_NaN_quarantine():
    e, r = _evidence(), _values()
    mask = e.view_observed.clone()
    mask[:, :, 1] = False
    content = e.content_delta.clone()
    content[:, :, 1] = float("nan")
    xy = e.image_delta.clone()
    xy[:, :, 1] = float("nan")
    out = r(
        replace(
            e, content_delta=content, image_delta=xy, view_observed=mask, view_weight=mask.float()
        )
    )
    assert all(torch.isfinite(x).all() for x in out)
    torch.stack([x.square().sum() for x in out]).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())


def test_camera_and_object_permutation():
    e, r = _evidence(), _values()
    expected = r(e)
    k = torch.tensor([2, 0, 1])
    p = e.permute(k, e.binding.permute(k))
    for a, b in zip(expected, r(p)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    other = _values(("wrist", "top"))
    other.load_state_dict(r.state_dict(), strict=True)
    c = torch.tensor([1, 0])
    p = replace(
        e,
        content_delta=e.content_delta[:, :, c],
        image_delta=e.image_delta[:, :, c],
        match_status=e.match_status[:, :, c],
        view_observed=e.view_observed[:, :, c],
        view_weight=e.view_weight[:, :, c],
        camera_names=("wrist", "top"),
    )
    for a, b in zip(expected, other(p)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="camera"):
        r(p)


def test_dock_and_permutation_keep_same_target_law(updated: Updated) -> None:
    _, _, _, cache, _ = updated
    intent = cache.top.intent
    ev = intent.instruction_change
    assert (
        ev is not None
        and ev.binding is intent.target_binding
        and ev.current_state is cache.history.state
    )
    assert intent.policy_dock().instruction_change is ev
    p = intent.permute(torch.tensor([2, 0, 3, 1]))
    p.validate(horizon=24, hidden=intent.object_tokens.shape[-1])
    assert p.instruction_change is not None and p.instruction_change.binding is p.target_binding


@pytest.mark.parametrize("kind", ["missing", "binding", "current", "camera", "reference"])
def test_cache_rejects_wrong_provenance(updated: Updated, kind: str) -> None:
    _, _, _, cache, _ = updated
    ev = cache.top.intent.instruction_change
    assert ev is not None
    if kind == "missing":
        ev = None
    elif kind == "binding":
        ev = replace(ev, binding=replace(ev.binding))
    elif kind == "current":
        ev = replace(ev, current_state=ev.current_state.clone())
    elif kind == "reference":
        ev = replace(ev, reference=replace(ev.reference))
    else:
        ev = replace(ev, camera_names=("wrist", "top"))
    bad = replace(
        cache, top=replace(cache.top, intent=replace(cache.top.intent, instruction_change=ev))
    )
    with pytest.raises(ValueError):
        bad.validate(_config())


@pytest.mark.parametrize("bf16", [False, True])
def test_real_update_and_row_zero_gradients_reach_all_new_parameters(bf16: bool) -> None:
    torch.manual_seed(6791)
    m, e = _model_engine(_config())
    b = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        e.train_step(b, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, _, _ = m.encode_online(b.online)
        out = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
        loss = out.bottom.physical_velocity[:, 0].float().square().mean()
    loss.backward()
    for module in (
        m.intent.organizer.instruction_progress,
        m.policy_compiler.plan_compiler.instruction_change_read,
    ):
        assert module is not None
        for name, p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
                name
            )
    observer = m.policy_compiler.plan_compiler.robot_observer
    assert observer is not None
    assert all(p.grad is None for p in observer.response.parameters())
    assert all(p.grad is None for p in m.training_targets.parameters())


def test_each_type_has_direct_P3_effect(updated: Updated) -> None:
    m, _, _, cache, _ = updated
    ev = cache.top.intent.instruction_change
    p = m.policy_compiler.plan_compiler.instruction_change_read
    assert ev is not None and p is not None
    q = torch.randn(1, 24, 3, m.config.dimensions.hidden_size)
    original = p(ev, q)
    for n in ("content_delta", "image_delta", "robot_delta"):
        assert not torch.equal(original, p(replace(ev, **{n: getattr(ev, n) + 0.5}), q)), n


def test_comparison_runs_once_not_per_ODE_and_no_state_mutation(
    updated: Updated, monkeypatch: pytest.MonkeyPatch
) -> None:
    m, _, b, _, _ = updated
    r = m.intent.organizer.instruction_progress
    assert isinstance(r, TypedInstructionReferenceRead)
    calls: list[int] = []
    old = r.forward

    def count(**kw):
        calls.append(1)
        return old(**kw)

    monkeypatch.setattr(r, "forward", count)
    with torch.no_grad():
        buffers = {k: v.clone() for k, v in m.named_buffers()}
        a = sample_action(m, b.online, m.config, generator=torch.Generator().manual_seed(0))
        assert len(calls) == 1
        z = sample_action(m, b.online, m.config, generator=torch.Generator().manual_seed(0))
    assert len(calls) == 2
    torch.testing.assert_close(a.action, z.action, rtol=0, atol=0)
    assert all(torch.equal(v, buffers[k]) for k, v in m.named_buffers())


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostics_preserve_value_gradient_RNG(bf16: bool) -> None:
    torch.manual_seed(6795)
    m, _ = _model_engine(_config())
    m.eval()
    b = _batch()
    results: list[tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]] = []
    for diag in (False, True):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            cache, _, _ = m.encode_online(b.online, collect_diagnostics=diag)
            ev = cache.top.intent.instruction_change
            r = m.policy_compiler.plan_compiler.instruction_change_read
            assert ev is not None and r is not None
            out = r(ev, cache.top.intent.temporal_queries[:, :, None])
            grad = torch.autograd.grad(out.float().square().sum(), tuple(r.parameters()))
        results.append((out.detach(), grad, torch.get_rng_state()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    for ga, gb in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_future_labels_cannot_mutate_observed_change(updated: Updated) -> None:
    m, _, b, cache, st = updated
    ev = cache.top.intent.instruction_change
    assert ev is not None
    names = ("content_delta", "image_delta", "robot_delta", "match_status")
    saved = {n: getattr(ev, n).clone() for n in names}
    with torch.no_grad():
        m.build_training_targets(
            st, replace(b.future, action_sequence=b.future.action_sequence + 2)
        )
    assert all(torch.equal(getattr(ev, n), saved[n]) for n in names)


def test_checkpoint_exact_reload_and_ABI_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert abi["instruction_change"] == instruction_change_metadata(("top", "wrist"))
        for kind in ("missing", "reinterpret"):
            bad = copy.deepcopy(abi)
            if kind == "missing":
                del bad["instruction_change"]
            else:
                cast(dict, bad["instruction_change"])["limits"] = "physical success"
            with pytest.raises(ValueError, match="instruction change"):
                validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)
