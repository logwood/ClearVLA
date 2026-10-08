"""Source support belongs to its producer, including in the integrated graph.

These are ordinary mathematical/typed-graph regressions. They neither qualify
a physical matcher nor supply instance labels from simulator oracles.
"""

from dataclasses import replace

import pytest
import torch
from test_mainline_structural_contracts import _local_facts

from clearvla.mainline.model.canonical_grounding import canonical_grounding
from clearvla.mainline.model.grounding import DenseObjectGrounder, dense_chart_from_local_facts
from clearvla.mainline.model.observation_association import ObjectObservationAssociation
from clearvla.vision.entity_chart import (
    CanonicalImageReadSource,
    CurrentImageSupport,
    current_image_grid,
)


def tiny_case():
    torch.manual_seed(7120)
    local = _local_facts(cameras=1, side=2, content=8, route=4, hidden=16)
    valid = torch.ones(1, 1, 2, 2, 4, 1, dtype=torch.bool)
    valid[..., 0, 0] = False
    grid = current_image_grid(2, 2, device=torch.device("cpu"))
    xy = grid[None, None, :, :, None, None].expand(1, 1, 2, 2, 4, 1, 2).clone()
    spatial = CurrentImageSupport(xy, valid.float(), valid, torch.zeros_like(valid).float())
    local = replace(
        local,
        current_image_support=spatial,
        slot_validity=valid[..., 0, None].float(),
        current_observed_content=local.target_dino_content,
    )
    module = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=2,
        iterations=1,
        entity_chart_mode="current_image_support_v1",
        object_view_mode="per_camera_values_v1",
        entity_ownership_mode="canonical_image_v1",
        camera_names=("top",),
        entity_transport_gradient_mode="ordinary_bilinear_v1",
        entity_competition_scale_mode="per_observation_v1",
    )
    return module, local, dense_chart_from_local_facts(local)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_owner_prior",
        "candidate_semantic_prior",
        "candidate_appearance_prior",
        "candidate_geometry_prior",
    ],
)
def test_integrated_canonical_prior_cannot_resurrect_invalid_payload(field):
    module, local, chart = tiny_case()
    legal = chart.candidate_validity[..., 0] > 0
    clean_prior = torch.where(legal, getattr(chart, field), 0.0).detach().requires_grad_()
    bad_prior = torch.where(legal, clean_prior.detach(), float("nan")).requires_grad_()
    control, _ = canonical_grounding(
        module, local, replace(chart, **{field: clean_prior}), None, collect_diagnostics=False
    )
    actual, _ = canonical_grounding(
        module, local, replace(chart, **{field: bad_prior}), None, collect_diagnostics=False
    )
    for name in ("content", "identity_state", "object_to_chart", "reconstructed_dino"):
        torch.testing.assert_close(getattr(actual, name), getattr(control, name), atol=0, rtol=0)
    loss = actual.reconstruction_error + actual.content.square().mean()
    grad = torch.autograd.grad(loss, bad_prior)[0]
    assert torch.isfinite(grad).all()
    assert grad[~legal].count_nonzero() == 0
    assert grad[legal].abs().sum() > 0


def measured_case():
    module, local, chart = tiny_case()
    facts, _ = canonical_grounding(module, local, chart, None, collect_diagnostics=False)
    current = torch.eye(4, 8).reshape(1, 1, 2, 2, 8)
    observed = torch.ones(1, 1, 2, 2, 1, dtype=torch.bool)
    observed[:, :, :, 1] = False  # This is CURRENT support, not successor support.
    chart = replace(chart, dino_content=current, cell_observed=observed)
    support = torch.zeros(1, 2, 1, 2, 2, dtype=torch.bool)
    support[..., 0, 0] = True
    assert chart.current_image_support is not None
    src = CanonicalImageReadSource(
        torch.zeros_like(support).float(), support, chart.current_image_support
    )
    image = src.on_image(rows=2, columns=2)
    probability, _ = image.normalized((2, 3, 4))
    facts = replace(
        facts,
        dense_chart=chart,
        current_image_source=src,
        current_image_measure=image,
        object_to_chart=probability,
    )
    facts.validate()
    association = ObjectObservationAssociation(
        content_dim=8,
        key_dim=4,
        camera_names=("top",),
        observation_measurement_mode="source_consistent_v1",
    )
    return association, facts, current


def test_successor_full_image_does_not_inherit_current_training_mask():
    association, facts, current = measured_case()
    successor = current.flip(-2)  # known rightward correspondence -1 -> +1
    result = association.measure_observations(
        facts=facts, observations=successor[:, None], relative_offsets=torch.tensor([4])
    )
    torch.testing.assert_close(
        result.transport_per_support[..., 0],
        torch.full_like(result.transport_per_support[..., 0], 2.0),
    )
    torch.testing.assert_close(
        result.transport_per_support[..., 1], torch.zeros_like(result.transport_per_support[..., 1])
    )
    assert result.null_probability.count_nonzero() == 0


def test_successor_support_is_explicit_and_masked_before_finite_check():
    association, facts, current = measured_case()
    support = torch.zeros(1, 1, 1, 2, 2, dtype=torch.bool)
    result = association.measure_observations(
        facts=facts,
        observations=torch.full_like(current[:, None], float("nan")),
        relative_offsets=torch.tensor([4]),
        target_cell_observed=support,
    )
    torch.testing.assert_close(result.null_probability, torch.ones_like(result.null_probability))
    assert result.transport_per_support.count_nonzero() == 0
    torch.testing.assert_close(result.successor_per_support, result.current_reference)


@pytest.mark.parametrize("bad", [torch.ones(1, 1), torch.ones(1, 1, 1, 2, 2)])
def test_successor_support_rejects_undeclared_axes_and_nonboolean_mask(bad):
    association, facts, current = measured_case()
    with pytest.raises((TypeError, ValueError)):
        association.measure_observations(
            facts=facts,
            observations=current[:, None],
            relative_offsets=torch.tensor([4]),
            target_cell_observed=bad,
        )
