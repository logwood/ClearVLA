"""Explicit camera correspondence interfaces and real identity-loss dispatch."""

from dataclasses import replace

import pytest
import torch
from test_identity_supervision import labels as legacy_labels
from test_source_consistent_measurement import production as _production

from clearvla.mainline.identity_pairs import IdentityPairBatch, IdentityPairGroup
from clearvla.mainline.training.identity import identity_terms

production = _production


def group(cameras=3, n=4):
    source = torch.zeros(1, n, 2)
    return IdentityPairGroup(
        "cross",
        1,
        0,
        cameras - 1,
        source,
        source.clone(),
        torch.ones(1, n, dtype=torch.bool),
        "sensor-test-fixture-v1",
    )


def test_nonstandard_camera_names_and_single_camera_temporal_labels():
    pair = group()
    batch = IdentityPairBatch(("external_left", "head", "tool"), torch.tensor([[2, 4]]), (pair,))
    batch.validate(batch=1, device=torch.device("cpu"))
    single = replace(pair, kind="temporal", source_time=0, source_camera=0, target_camera=0)
    IdentityPairBatch(("head",), torch.tensor([[2, 4]]), (single,)).validate(
        batch=1, device=torch.device("cpu")
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(target_camera=3),
        dict(source_time=2),
        dict(provenance=""),
        dict(source_camera=2),
        dict(kind="inferred-object-oracle"),
    ],
)
def test_invalid_sources_fail_closed(change):
    with pytest.raises(ValueError):
        IdentityPairBatch(
            ("a", "b", "c"), torch.tensor([[2, 4]]), (replace(group(), **change),)
        ).validate(batch=1, device=torch.device("cpu"))


def test_unobserved_coordinate_payload_never_becomes_a_label():
    p = replace(
        group(),
        valid=torch.zeros(1, 4, dtype=torch.bool),
        source=torch.full((1, 4, 2), float("nan")),
    )
    IdentityPairBatch(("a", "b", "c"), torch.tensor([[2, 4]]), (p,)).validate(
        batch=1, device=torch.device("cpu")
    )
    with pytest.raises(ValueError):
        IdentityPairBatch(
            ("a", "b", "c"), torch.tensor([[2, 4]]), (replace(p, valid=torch.ones_like(p.valid)),)
        ).validate(batch=1, device=torch.device("cpu"))


def test_generic_pairs_match_actual_legacy_objective_and_backward(production):
    model, _, batch, _, _ = production
    if model.config.top.entity_ownership_mode != "canonical_image_v1":
        pytest.skip("this supervised branch is canonical-only")
    model.zero_grad(set_to_none=True)
    _, state, _ = model.encode_online(batch.online)
    old = legacy_labels(batch.online)
    generic = IdentityPairBatch(old.camera_names, old.source_frames, old.pair_groups())
    a = identity_terms(model, batch.online, state.top.facts, old)
    b = identity_terms(model, batch.online, state.top.facts, generic)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    (b["identity_correspondence"] + b["identity_source_prediction"]).backward()
    assert model.grounding.grounder.content_key[1].weight.grad.abs().max() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_pairs_reports_no_labels_without_nan_or_optimization_target(production):
    model, _, batch, _, state = production
    if model.config.top.entity_ownership_mode != "canonical_image_v1":
        pytest.skip("canonical-only")
    labels = IdentityPairBatch(
        ("top", "wrist"), batch.online.history.timing.state_offsets[:, -2:] + 16, ()
    )
    result = identity_terms(model, batch.online, state.top.facts, labels)
    assert (
        result["identity_cross_admitted_pairs"] == 0
        and result["identity_temporal_admitted_pairs"] == 0
    )
    assert result["identity_correspondence"] == 0 and result["identity_source_prediction"] == 0


def test_training_loss_rejects_label_clock_from_another_history(production):
    model, _, batch, _, state = production
    if model.config.top.entity_ownership_mode != "canonical_image_v1":
        pytest.skip("canonical-only")
    source = legacy_labels(batch.online)
    bad = replace(source, source_frames=source.source_frames + torch.tensor([[-1, 0]]))
    with pytest.raises(ValueError, match="source span"):
        identity_terms(model, batch.online, state.top.facts, bad)


def test_unobserved_group_does_not_dilute_real_label_loss_or_backward(production):
    model, _, batch, _, _ = production
    if model.config.top.entity_ownership_mode != "canonical_image_v1":
        pytest.skip("canonical-only identity branch")
    model.zero_grad(set_to_none=True)
    _, state, _ = model.encode_online(batch.online)
    raw = legacy_labels(batch.online)
    one = raw.pair_groups()[0]
    empty = replace(
        raw.pair_groups()[1],
        valid=torch.zeros_like(one.valid),
        source=torch.full_like(one.source, float("nan")),
        target=torch.full_like(one.target, float("nan")),
    )
    one_batch = IdentityPairBatch(raw.camera_names, raw.source_frames, (one,))
    padded_batch = replace(one_batch, groups=(one, empty))
    a = identity_terms(model, batch.online, state.top.facts, one_batch)
    b = identity_terms(model, batch.online, state.top.facts, padded_batch)
    for name in (
        "identity_correspondence",
        "identity_source_prediction",
        "identity_source_removed_prediction_mse",
    ):
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    key = model.grounding.grounder.content_key[1].weight
    ga = torch.autograd.grad(
        a["identity_correspondence"] + a["identity_source_prediction"], key, retain_graph=True
    )[0]
    gb = torch.autograd.grad(b["identity_correspondence"] + b["identity_source_prediction"], key)[0]
    torch.testing.assert_close(ga, gb, atol=2e-7, rtol=2e-5)
    assert ga.abs().sum() > 0
