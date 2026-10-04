"""Joint task execution: real modules, explicitly synthetic source fixtures."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.future_time import CONTROL_INTERVALS
from clearvla.mainline.model.target_binding import TargetBinding, TargetEvidence
from clearvla.mainline.model.task_execution import (
    JointTaskRelationEncoder,
    ObjectRoleRead,
    TaskAwareFactualRead,
    TaskOutcomePlanRead,
    TaskRelationRead,
)
from clearvla.mainline.model.types import FutureObjectDynamics
from clearvla.mainline.operation_expectation import OperationExpectation
from clearvla.mainline.task_execution import JOINT_TASK_EXECUTION
from clearvla.mainline.world_control import CandidateControlDomain


def fixture(seed=287, cameras=("top", "wrist")):
    torch.manual_seed(seed)
    b, k, c, h, d, s = 2, 4, len(cameras), 16, 12, 10
    law = TargetBinding.from_logits(
        torch.randn(b, k), torch.randn(b, 1), torch.ones(b, k, dtype=torch.bool)
    )
    content = torch.randn(b, k, d)
    state = torch.randn(b, s)
    from clearvla.vision.entity_chart import current_image_grid

    position_probability = torch.softmax(torch.randn(b, k, c, 9), -1)
    position_grid = current_image_grid(3, 3, device=state.device).reshape(9, 2)
    kw = dict(
        task_intervals=torch.randn(b, 4, h),
        attributes=torch.randn(b, k, 4, h),
        position_probability=position_probability,
        position_grid=position_grid,
        state=state,
        history_context=torch.randn(b, h),
        view_observed=torch.ones(b, k, c, dtype=torch.bool),
        binding=law,
        current_content=content,
    )
    enc = JointTaskRelationEncoder(hidden=h, state_dim=s, camera_names=cameras)
    relation = enc(**kw)
    expected = OperationExpectation(
        torch.randn(b, 4, k, d),
        torch.randn(b, 4, k, c, 2),
        torch.randn(b, 4, s),
        kw["view_observed"],
        law,
        state,
        content,
        cameras,
    )
    semantic = torch.randn(b, 4, k, d)
    predicted = FutureObjectDynamics(
        current_reference=content.detach(),
        source_content=content,
        successor_content=content[:, None] + semantic,
        semantic_delta=semantic,
        transport_mean=torch.randn(b, 4, k, c, 2),
        transport_covariance=torch.ones(b, 4, k, c, 3) * 0.01,
        chart_availability=torch.ones(b, k, 1),
        log_chart_availability=torch.zeros(b, k, 1),
        camera_coordinates=torch.einsum("bkcn,nd->bkcd", position_probability, position_grid),
        camera_chart_availability=torch.ones(b, k, c, 1),
        log_camera_chart_availability=torch.zeros(b, k, c, 1),
        camera_names=cameras,
        control_domain=CandidateControlDomain(24, CONTROL_INTERVALS),
        time_grid_mode=relation.time_grid_mode,
    )
    return enc, kw, relation, expected, predicted


def test_selection_is_explicit_with_unchanged_default_serialization():
    old = load_config("configs/mainline/structural_rebuild_m6n_calvin.json")
    assert "task_execution_mode" not in old.as_dict()["top"]
    for name in ["m6n", "cumulative"]:
        c = load_config(f"configs/mainline/task_grounded_execution_{name}_calvin.json")
        assert c.top.task_execution_mode == JOINT_TASK_EXECUTION
        assert config_from_mapping(c.as_dict()) == c
        assert (
            c.objectives
            == load_config(
                "configs/mainline/structural_rebuild_"
                + ("m6n" if name == "m6n" else "annotated_goal")
                + "_calvin.json"
            ).objectives
        )
    for values in [
        dict(task_execution_mode="guess"),
        dict(world_control_mode="legacy_extrapolation_v1"),
        dict(world_supervision_mode="legacy_candidate_v1"),
        dict(target_binding_mode="reader_local_v1"),
    ]:
        with pytest.raises(ValueError):
            replace(c, top=replace(c.top, **values)).validate()


@pytest.mark.parametrize("amp", [False, True])
def test_encoder_joint_task_geometry_robot_owner_gradients(amp):
    enc, kw, _, _, _ = fixture()
    for name in [
        "task_intervals",
        "attributes",
        "position_probability",
        "state",
        "history_context",
    ]:
        kw[name] = kw[name].detach().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        r = enc(**kw)
        r.values.float().square().sum().backward()
    for name in [
        "task_intervals",
        "attributes",
        "position_probability",
        "state",
        "history_context",
    ]:
        grad = kw[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    for name, p in enc.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_task_cannot_be_replaced_by_relation_only_value_bypass():
    enc, kw, _, _, _ = fixture()
    zero = enc(**{**kw, "task_intervals": torch.zeros_like(kw["task_intervals"])})
    assert torch.count_nonzero(zero.values) == 0
    a = enc(**kw).values
    b = enc(**{**kw, "task_intervals": -kw["task_intervals"]}).values
    assert not torch.allclose(a, b)
    c = enc(**{**kw, "position_probability": kw["position_probability"].roll(1, -1)}).values
    assert not torch.allclose(a, c)


def test_encoder_object_equivariance_and_named_camera_relabeling():
    enc, kw, r, _, _ = fixture()
    p = torch.tensor([2, 0, 3, 1])
    law = kw["binding"].permute(p)
    k2 = {**kw, "binding": law}
    for n in ["attributes", "position_probability", "view_observed", "current_content"]:
        k2[n] = kw[n][:, p]
    torch.testing.assert_close(enc(**k2).values, r.values[:, :, p])
    other = JointTaskRelationEncoder(hidden=16, state_dim=10, camera_names=("wrist", "top"))
    other.load_state_dict(enc.state_dict())
    reverse = {
        **kw,
        "position_probability": kw["position_probability"][:, :, [1, 0]],
        "view_observed": kw["view_observed"][:, :, [1, 0]],
    }
    torch.testing.assert_close(other(**reverse).values, r.values[:, :, :, [1, 0]])


def test_invalid_view_values_mask_before_projection_and_backward():
    enc, kw, _, _, _ = fixture()
    valid = kw["view_observed"].clone()
    valid[:, -1] = False
    valid[:, 0, -1] = False
    coords = kw["position_probability"].masked_fill(~valid[..., None], torch.nan).requires_grad_()
    attrs = (
        kw["attributes"].masked_fill(~valid.any(-1)[..., None, None], torch.nan).requires_grad_()
    )
    r = enc(**{**kw, "position_probability": coords, "attributes": attrs, "view_observed": valid})
    assert torch.isfinite(r.values).all()
    assert torch.count_nonzero(r.values.masked_select(~valid[:, None, :, :, None])) == 0
    r.values.square().sum().backward()
    assert torch.isfinite(coords.grad).all() and torch.isfinite(attrs.grad).all()
    assert torch.count_nonzero(coords.grad[~valid]) == 0


@pytest.mark.parametrize("amp", [False, True])
def test_role_read_no_K_competition_preserves_mass_and_value_zero(amp):
    _, _, r, _, _ = fixture()
    read = ObjectRoleRead(16, 4)
    q = torch.randn(2, 24, 4, 16)
    support = r.view_observed[:, None].expand(-1, 4, -1, -1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        a = read(q, r.values, r.values, support, r.binding.mass)
        b = read(q, r.values, r.values, support, r.binding.mass * 0.1)
        z = read(q, r.values, torch.zeros_like(r.values), support, r.binding.mass)
        empty = read(q, r.values, r.values, torch.zeros_like(support), r.binding.mass)
    torch.testing.assert_close(b, a * 0.1, rtol=0.02 if amp else 1e-5, atol=1e-6)
    assert torch.count_nonzero(z) == torch.count_nonzero(empty) == 0


def test_null_target_keeps_scene_without_assigning_target_to_distractor():
    _, _, r, _, _ = fixture()
    law = TargetBinding(
        torch.cat((torch.full((2, 4), -torch.inf), torch.zeros(2, 1)), -1), r.binding.supported
    )
    rr = replace(r, binding=law)
    read = TaskRelationRead(16, 4)
    out = read(torch.randn(2, 24, 16), rr)
    assert torch.count_nonzero(law.mass) == 0
    assert out.norm() > 0


def test_P1_relation_is_query_only_and_cannot_fabricate_current_facts():
    _, _, r, _, _ = fixture()
    read = TaskAwareFactualRead(16, 4)
    q = torch.randn(2, 24, 4, 16)
    e = TargetEvidence(torch.zeros(2, 4, 6, 16), torch.ones(2, 4, 6, dtype=torch.bool))
    out = read(q, e, r.binding, relation=r)
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("amp", [False, True])
def test_joint_gap_zero_when_predictions_agree_but_relation_remains(amp):
    _, _, r, e, w = fixture()
    read = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=r.camera_names)
    w = replace(w, semantic_delta=e.semantic_delta, transport_mean=e.image_delta)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        target, scene = read(torch.randn(2, 24, 4, 16), r, e, w)
    assert torch.count_nonzero(scene) == 0
    assert target.norm() > 0


def test_non_target_consequences_are_not_erased_by_target_selection():
    _, _, r, e, w = fixture()
    log = torch.full((2, 5), -torch.inf)
    log[:, 0] = 0
    law = TargetBinding(log, r.binding.supported)
    r = replace(r, binding=law)
    e = replace(e, binding=law)
    read = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=r.camera_names)
    q = torch.randn(2, 24, 4, 16)
    a, sa = read(q, r, e, w)
    delta = w.transport_mean.clone()
    delta[:, :, 1] += 0.5
    b, sb = read(q, r, e, replace(w, transport_mean=delta))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(sa, sb)
    # Full null disables the target role, not scene evidence or the whole policy.
    log[:, :] = -torch.inf
    log[:, -1] = 0
    null = TargetBinding(log, law.supported)
    rn = replace(r, binding=null)
    en = replace(e, binding=null)
    tn, sn = read(q, rn, en, w)
    assert torch.count_nonzero(tn) == 0 and sn.norm() > 0


def test_comparison_requires_exact_source_not_just_equal_tensor_shape():
    _, _, r, e, w = fixture()
    read = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=r.camera_names)
    with pytest.raises(ValueError, match="exact current object"):
        read(torch.randn(2, 24, 4, 16), r, e, replace(w, source_content=w.source_content.clone()))
    with pytest.raises(ValueError, match="camera roles"):
        read(torch.randn(2, 24, 4, 16), r, e, replace(w, camera_names=("wrist", "top")))


def test_unknown_control_intervals_are_quarantined_before_joint_arithmetic():
    _, _, r, e, w = fixture()
    read = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=r.camera_names)
    q = torch.randn(2, 24, 4, 16)
    d = CandidateControlDomain(8, CONTROL_INTERVALS)
    w = replace(w, control_domain=d)
    a = read(q, r, e, w)
    sem = w.semantic_delta.clone()
    sem[:, 2:] = torch.nan
    img = w.transport_mean.clone()
    img[:, 2:] = torch.nan
    b = read(q, r, e, replace(w, semantic_delta=sem, transport_mean=img))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_balanced_pooled_effects_do_not_erase_objectwise_mismatch():
    _, _, r, e, w = fixture()
    read = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=r.camera_names)
    q = torch.randn(2, 24, 4, 16)
    # Different object/position pairing with identical scene-average motion.
    wm = w.transport_mean.clone()
    wm[:, :, 0, :, 0] += 0.4
    wm[:, :, 1, :, 0] -= 0.4
    torch.testing.assert_close(wm.mean(2), w.transport_mean.mean(2))
    a = read(q, r, e, w)
    b = read(q, r, e, replace(w, transport_mean=wm))
    assert not torch.allclose(a[0] + a[1], b[0] + b[1])


def test_role_object_permutation_invariance():
    _, _, r, _, _ = fixture()
    read = ObjectRoleRead(16, 4)
    q = torch.randn(2, 24, 4, 16)
    valid = r.view_observed[:, None].expand(-1, 4, -1, -1)
    p = torch.tensor([3, 1, 0, 2])
    a = read(q, r.values, r.values, valid, r.binding.mass)
    b = read(q, r.values[:, :, p], r.values[:, :, p], valid[:, :, p], r.binding.mass[:, p])
    torch.testing.assert_close(a, b)


def test_shared_binder_history_is_context_not_identity_value():
    from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder

    torch.manual_seed(875)
    m = TaskConditionedTargetBinder(16, 4)
    x = torch.randn(2, 4, 16)
    v = torch.ones(2, 4, dtype=torch.bool)
    no_task = torch.zeros(2, 4, 16)
    a = m(no_task, x, v, history=torch.randn(2, 16))
    b = m(no_task, x, v, history=torch.randn(2, 16) * 100)
    torch.testing.assert_close(a.log_probability, b.log_probability, atol=0, rtol=0)
    task = torch.randn(2, 4, 16)
    history = torch.randn(2, 16)
    c = m(task, x, v, history=history)
    d = m(-task, x, v, history=history)
    assert not torch.allclose(c.mass, d.mass)
    p = torch.tensor([1, 3, 0, 2])
    torch.testing.assert_close(m(task, x[:, p], v[:, p], history=history).mass, c.mass[:, p])


def test_source_quarantines_all_missing_targets_in_contextual_binder():
    from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder

    m = TaskConditionedTargetBinder(16, 4)
    x = torch.full((2, 4, 16), torch.nan, requires_grad=True)
    a = m(torch.randn(2, 4, 16), x, torch.zeros(2, 4, dtype=torch.bool), history=torch.randn(2, 16))
    assert torch.equal(a.null_mass, torch.ones(2, 1)) and torch.count_nonzero(a.mass) == 0
    a.mass.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) == 0


def test_same_centroid_distinct_native_support_is_not_forced_to_alias():
    enc, kw, _, _, _ = fixture()
    pa = torch.zeros_like(kw["position_probability"])
    pb = torch.zeros_like(pa)
    pa[..., 3] = 0.5
    pa[..., 5] = 0.5  # horizontal support about image center
    pb[..., 1] = 0.5
    pb[..., 7] = 0.5  # vertical support with identical centroid
    grid = kw["position_grid"]
    torch.testing.assert_close(
        torch.einsum("bkcn,nd->bkcd", pa, grid),
        torch.einsum("bkcn,nd->bkcd", pb, grid),
        atol=0,
        rtol=0,
    )
    a = enc(**{**kw, "position_probability": pa}).values
    b = enc(**{**kw, "position_probability": pb}).values
    assert not torch.allclose(a, b)


def test_interval_context_modulates_relation_values_without_changing_support():
    enc, kw, relation, _, _ = fixture()
    zero_context = torch.zeros_like(kw["task_intervals"])
    zero = enc(**kw, interval_context=zero_context)
    torch.testing.assert_close(zero.values, relation.values, rtol=0, atol=0)
    context = torch.randn_like(kw["task_intervals"])
    changed = enc(**kw, interval_context=context)
    torch.testing.assert_close(changed.view_observed, relation.view_observed)
    torch.testing.assert_close(changed.binding.mass, relation.binding.mass)
    assert (changed.values - relation.values).abs().max() > 1e-7
    assert changed.values[:, 0].sub(changed.values[:, 1]).abs().mean() > 1e-7
