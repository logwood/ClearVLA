"""Native S/P2 v2 tests; all observations here are explicitly synthetic.

No robot success, metric pose, equivariance to arbitrary camera movement, or
pretrained-encoder claim follows from these representation/contract tests.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from test_mainline_task_execution import fixture as source_fixture
from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.future_time import CONTROL_INTERVALS
from clearvla.mainline.model.task_execution import JointTaskRelationEncoder, TaskOutcomePlanRead
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.task_execution import (
    JOINT_SPATIAL_TASK_EXECUTION as V2, JOINT_TASK_EXECUTION as V1,
    task_execution_metadata,
)
from clearvla.mainline.world_control import CandidateControlDomain


def fixture(cameras=("top", "wrist")):
    _, kw, _, expected, predicted = source_fixture(cameras=cameras)
    # Actual per-view values, not broadcast copies of the object summary.
    torch.manual_seed(73013)
    kw["attributes"] = torch.randn(2, 4, len(cameras), 4, 16)
    enc = JointTaskRelationEncoder(hidden=16, state_dim=10, camera_names=cameras,
                                   task_execution_mode=V2)
    relation = enc(**kw)
    reader = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=cameras,
                                task_execution_mode=V2)
    return enc, kw, relation, expected, predicted, reader


def test_explicit_selection_no_new_loss_and_legacy_metadata():
    old = load_config("configs/mainline/dinov3_online_task_global_calvin.json")
    new = load_config("configs/mainline/dinov3_online_spatial_effect_calvin.json")
    assert new.top.task_execution_mode == V2 and old.top.task_execution_mode == V1
    assert old.objectives == new.objectives
    assert old.optimizer == new.optimizer and old.runtime == new.runtime
    assert old.bottom == new.bottom and old.observation == new.observation
    assert config_from_mapping(new.as_dict()) == new
    a, b = old.as_dict(), new.as_dict()
    b["top"]["task_execution_mode"] = V1
    b["data"]["output_dir"] = a["data"]["output_dir"]
    assert a == b
    assert task_execution_metadata() == task_execution_metadata(V1)
    assert task_execution_metadata(V2) != task_execution_metadata(V1)
    selection = ComponentSelection.from_config_without_validation(new)
    assert selection.intent == "joint_spatial_effect_intent_v2"
    assert selection.policy_compiler == "joint_spatial_effect_global_p2_p3_v2"
    assert selection.execution_bottom.endswith("_p3_task_global")
    with pytest.raises(ValueError):
        replace(new, top=replace(new.top, task_execution_mode="unknown")).validate()
    with pytest.raises(ValueError):
        task_execution_metadata("unknown")


@pytest.mark.parametrize("amp", [False, True])
def test_both_native_owners_have_finite_input_and_parameter_gradients(amp):
    enc, kw, _, e, w, reader = fixture()
    for n in ("task_intervals", "attributes", "position_probability", "state", "history_context"):
        kw[n] = kw[n].detach().requires_grad_()
    e = replace(e, current_state=kw["state"],
                semantic_delta=e.semantic_delta.detach().requires_grad_(),
                image_delta=e.image_delta.detach().requires_grad_())
    w = replace(w, semantic_delta=w.semantic_delta.detach().requires_grad_(),
                transport_mean=w.transport_mean.detach().requires_grad_())
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        relation = enc(**kw)
        target, scene = reader(torch.randn(2, 24, 2, 16), relation, e, w)
        (target.float().square().sum() + scene.float().square().sum()).backward()
    for n in ("task_intervals", "attributes", "position_probability", "state", "history_context"):
        grad = kw[n].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, n
    for value in (e.semantic_delta, e.image_delta, w.semantic_delta, w.transport_mean):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    for owner in (enc, reader):
        for name, p in owner.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert relation.binding is kw["binding"] and relation.execution_mode == V2


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("tile", [1, 7, 16])
def test_spatial_tiling_and_recomputation_match_dense_values_and_gradients(amp, tile):
    torch.manual_seed(791)
    enc = JointTaskRelationEncoder(hidden=8, state_dim=7, camera_names=("top", "wrist"),
                                   task_execution_mode=V2, spatial_tile_size=tile)
    features = torch.randn(19, 8, requires_grad=True)
    context = torch.randn(1, 4, 3, 2, 8, requires_grad=True)
    p = torch.softmax(torch.randn(1, 3, 2, 19), -1).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        out = enc._spatial_expectation(features, context, p)
    with torch.autocast("cpu", enabled=False):
        local = F.silu(context[..., None, :] + features[None, None, None, None])
        local = local - F.silu(context)[..., None, :]
        reference = torch.einsum("bkcn,bikcnh->bikch", p, local)
    torch.testing.assert_close(out, reference, atol=4e-7, rtol=3e-6)
    adjoint = torch.randn_like(out)
    grad = torch.autograd.grad((out * adjoint).sum(), (features, context, p), retain_graph=True)
    refgrad = torch.autograd.grad((reference * adjoint).sum(), (features, context, p))
    for a, b in zip(grad, refgrad, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=3e-5)


def test_context_enters_before_expectation_and_zero_coordinates_stay_zero():
    enc, _, _, _, _, _ = fixture()
    f = torch.zeros(3, 16)
    f[:, 0] = torch.tensor([-2., 0., 2.])
    a = torch.zeros(1, 4, 1, 2, 16)
    b = a + 1.5
    p = torch.tensor([.5, 0., .5])[None, None, None].expand(1, 1, 2, -1)
    assert not torch.allclose(enc._spatial_expectation(f, a, p), enc._spatial_expectation(f, b, p))
    assert torch.count_nonzero(enc._spatial_expectation(torch.zeros_like(f), b, p)) == 0
    # Same image-coordinate mean, different support distribution; no centroid shortcut.
    centered = torch.tensor([0., 1., 0.])[None, None, None].expand_as(p)
    assert torch.equal((p * f[:, 0]).sum(-1), (centered * f[:, 0]).sum(-1))
    assert not torch.allclose(enc._spatial_expectation(f, a, p), enc._spatial_expectation(f, a, centered))


@pytest.mark.parametrize("amp", [False, True])
def test_object_camera_permutations_preserve_source_law(amp):
    enc, kw, _, _, _, _ = fixture()
    korder = torch.tensor([3, 1, 0, 2])
    ko = {**kw, "binding": kw["binding"].permute(korder)}
    for key in ("attributes", "position_probability", "view_observed", "current_content"):
        ko[key] = kw[key][:, korder]
    other = JointTaskRelationEncoder(hidden=16, state_dim=10, camera_names=("wrist", "top"),
                                     task_execution_mode=V2)
    other.load_state_dict(enc.state_dict(), strict=True)
    co = {**kw}
    for key in ("attributes", "position_probability", "view_observed"):
        co[key] = kw[key][:, :, [1, 0]]
    original_probability = kw["position_probability"].clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        baseline = enc(**kw).values
        torch.testing.assert_close(enc(**ko).values, baseline[:, :, korder])
        torch.testing.assert_close(other(**co).values, baseline[:, :, :, [1, 0]])
    torch.testing.assert_close(kw["position_probability"], original_probability, atol=0, rtol=0)


@pytest.mark.parametrize("amp", [False, True])
def test_absent_views_and_unknown_control_are_quarantined_before_gradients(amp):
    enc, kw, _, e, w, reader = fixture()
    valid = kw["view_observed"].clone()
    valid[:, 0, -1] = False
    valid[:, -1] = False
    kw["view_observed"] = valid
    for key, mask in (("attributes", ~valid[..., None, None]),
                      ("position_probability", ~valid[..., None])):
        kw[key] = kw[key].masked_fill(mask, torch.nan).requires_grad_()
    e = replace(e, view_observed=valid)
    d = CandidateControlDomain(8, CONTROL_INTERVALS)
    sem, image = w.semantic_delta.clone(), w.transport_mean.clone()
    sem[:, 2:] = torch.nan
    image[:, 2:] = torch.nan
    w = replace(w, semantic_delta=sem.requires_grad_(), transport_mean=image.requires_grad_(), control_domain=d)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        relation = enc(**kw)
        target, scene = reader(torch.randn(2, 24, 2, 16), relation, e, w)
        (target.float().square().sum() + scene.float().square().sum()).backward()
    assert torch.isfinite(relation.values).all() and torch.isfinite(target).all() and torch.isfinite(scene).all()
    assert torch.count_nonzero(relation.values.masked_select(~valid[:, None, :, :, None])) == 0
    for key in ("attributes", "position_probability"):
        assert torch.isfinite(kw[key].grad).all()
        assert torch.count_nonzero(kw[key].grad[~valid]) == 0
    for owner in (enc, reader):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in owner.parameters())
    assert torch.count_nonzero(w.semantic_delta.grad[:, 2:]) == 0
    assert torch.count_nonzero(w.transport_mean.grad[:, 2:]) == 0


@pytest.mark.parametrize("amp", [False, True])
def test_paired_comparison_zero_antisymmetry_and_distinct_effect_context(amp):
    _, _, _, _, _, reader = fixture()
    a, b = torch.randn(2, 4, 4, 2, 32), torch.randn(2, 4, 4, 2, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        same, _ = reader._effect_comparison(a, a)
        ab, key = reader._effect_comparison(a, b)
        ba, _ = reader._effect_comparison(b, a)
        shifted, shifted_key = reader._effect_comparison(a + 3., b + 3.)
    assert torch.count_nonzero(same) == 0
    torch.testing.assert_close(ab, -ba, atol=0, rtol=0)
    assert not torch.allclose(ab, shifted)
    assert not torch.allclose(key, shifted_key)
    assert ab.dtype == torch.float32


def test_bf16_context_does_not_pre_round_the_two_effects_to_equality():
    _, _, _, _, _, reader = fixture()
    a = torch.full((1, 4, 1, 2, 32), 1.0)
    b = a + 2e-4
    assert torch.equal(a.bfloat16(), b.bfloat16())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gap, _ = reader._effect_comparison(a, b)
    assert torch.count_nonzero(gap) > 0 and torch.isfinite(gap).all()


@pytest.mark.parametrize("amp", [False, True])
def test_prediction_agreement_is_not_a_forced_stop(amp):
    _, _, r, e, w, reader = fixture()
    w = replace(w, semantic_delta=e.semantic_delta, transport_mean=e.image_delta)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        target, scene = reader(torch.randn(2, 24, 2, 16), r, e, w)
    assert torch.count_nonzero(scene) == 0 and torch.count_nonzero(target) > 0


def test_zero_task_no_new_coordinate_value_bypass_and_strict_graph_identity():
    enc, kw, r, e, w, reader = fixture()
    zero = enc(**{**kw, "task_intervals": torch.zeros_like(kw["task_intervals"])})
    assert torch.count_nonzero(zero.values) == 0
    with pytest.raises(ValueError, match="mix execution graphs"):
        reader(torch.randn(2, 24, 2, 16), replace(r, execution_mode=V1), e, w)
    old = JointTaskRelationEncoder(hidden=16, state_dim=10, camera_names=("top", "wrist"))
    assert not any("spatial_context" in n for n in old.state_dict())
    with pytest.raises(RuntimeError):
        enc.load_state_dict(old.state_dict(), strict=True)
    with pytest.raises(ValueError):
        JointTaskRelationEncoder(hidden=16, state_dim=10, camera_names=("top",), spatial_tile_size=0)


def test_full_null_keeps_scene_and_does_not_select_a_substitute_target():
    _, _, r, e, w, reader = fixture()
    from clearvla.mainline.model.target_binding import TargetBinding
    log = torch.full((2, 5), -torch.inf)
    log[:, -1] = 0.
    law = TargetBinding(log, r.binding.supported)
    target, scene = reader(torch.randn(2, 24, 2, 16), replace(r, binding=law), replace(e, binding=law), w)
    assert torch.count_nonzero(target) == 0 and torch.count_nonzero(scene) > 0
