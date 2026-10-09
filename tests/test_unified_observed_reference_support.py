"""The current W reference must read the same observed source domain as Teacher.

These CPU checks use the production training-mask producer on artificial online
inputs. They distinguish the G-owned source law from unavailable reconstruction
payload, without changing any successor/label support or claiming real training.
"""

from dataclasses import replace

import pytest
import torch
from test_source_consistent_measurement import production as _production

from clearvla.mainline.model.observation_association import ObjectObservationAssociation

production = _production


@pytest.fixture(scope="module")
def masked_production(production):
    model, _, batch, _, _ = production
    model.train()
    try:
        torch.manual_seed(810)
        with torch.no_grad():
            cache, state, _ = model.encode_online(batch.online, training_mask=True)
    finally:
        model.eval()
    assert (~state.observation.local_facts.cell_observed).any()
    return model, batch, cache, state


def _measure(facts):
    association = ObjectObservationAssociation(
        content_dim=facts.content.shape[-1],
        camera_names=("top", "wrist"),
        observation_measurement_mode="source_consistent_v1",
    )
    return association.measure_observations(
        facts=facts,
        observations=facts.dense_chart.dino_content[:, None],
        relative_offsets=torch.tensor([4]),
        target_cell_observed=facts.dense_chart.cell_observed[..., 0][:, None],
    )


def test_actual_training_mask_has_one_current_reference_for_G_W_and_teacher(masked_production):
    _, _, cache, state = masked_production
    facts = state.top.facts
    measured = _measure(facts)
    torch.testing.assert_close(facts.observed_content, measured.current_reference[:, 0], atol=2e-6, rtol=2e-6)
    assert facts.world_belief().observed_content is facts.observed_content
    torch.testing.assert_close(cache.top.predicted_dynamics.current_reference, measured.current_reference[:, 0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(measured.successor_per_support, measured.current_reference, atol=2e-6, rtol=0)
    assert measured.transport_per_support.abs().max() < 2e-6


@pytest.mark.parametrize("poison", [100.0, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bf16", [False, True])
def test_unobserved_reconstruction_values_cannot_enter_current_reference(masked_production, poison, bf16):
    model, _, _, state = masked_production
    local = state.observation.local_facts
    observed = local.cell_observed
    clean = replace(local, target_dino_content=torch.where(observed, local.target_dino_content, 0.0))
    dirty = replace(local, target_dino_content=torch.where(observed, local.target_dino_content, poison))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        before, _ = model.grounding.materialize_facts(clean)
        actual, _ = model.grounding.materialize_facts(dirty)
    assert torch.isfinite(actual.observed_content).all()
    torch.testing.assert_close(actual.observed_content, before.observed_content, atol=0, rtol=0)
    for name in ("content", "semantic", "appearance", "geometry", "camera_coordinates", "object_to_chart"):
        torch.testing.assert_close(getattr(actual, name), getattr(before, name), atol=0, rtol=0)
    torch.testing.assert_close(actual.current_image_source.log_measure, before.current_image_source.log_measure, atol=0, rtol=0)
    assert actual.current_image_source.spatial is local.current_image_support


@pytest.mark.parametrize("bf16", [False, True])
def test_missing_payload_cannot_change_ordinary_G_source_or_parameter_vjp(masked_production, bf16):
    model, _, _, state = masked_production
    local = state.observation.local_facts
    grounder = model.grounding.grounder
    parameters = (grounder.slot_seed, grounder.content_key[-1].weight, grounder.semantic_key.weight)
    values = []
    gradients = []
    for poison in (0.0, float("nan")):
        changed = replace(local, target_dino_content=torch.where(local.cell_observed, local.target_dino_content, poison))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            facts, _ = model.grounding.materialize_facts(changed)
            loss = facts.observed_content.float().square().sum()
        gradient = torch.autograd.grad(loss, parameters)
        assert all(torch.isfinite(g).all() for g in gradient)
        assert all(g.count_nonzero() > 0 for g in gradient)
        values.append(facts.observed_content)
        gradients.append(gradient)
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
    for clean, dirty in zip(*gradients, strict=True):
        torch.testing.assert_close(clean, dirty, atol=0, rtol=0)


def test_supported_current_values_remain_in_reference(masked_production):
    model, _, _, state = masked_production
    local = state.observation.local_facts
    history = local.observed_history
    assert history is not None
    observed_content = history.content.clone()
    observed_content[:, -1] += 0.25
    shifted = replace(local, target_dino_content=torch.where(local.cell_observed, local.target_dino_content + 0.25, torch.nan),
                      observed_history=replace(history, content=observed_content))
    with torch.no_grad():
        before, _ = model.grounding.materialize_facts(local)
        after, _ = model.grounding.materialize_facts(shifted)
    torch.testing.assert_close(after.observed_content - before.observed_content, torch.full_like(before.observed_content, 0.25), atol=2e-6, rtol=0)
    torch.testing.assert_close(after.observed_content, _measure(after).current_reference[:, 0], atol=2e-6, rtol=2e-6)


def test_empty_current_reference_is_finite_zero_with_zero_owner_vjp(masked_production):
    model, _, _, state = masked_production
    local = state.observation.local_facts
    history = local.observed_history
    assert history is not None
    observed = history.observed.clone()
    observed[:, -1] = False
    empty = replace(local, cell_observed=torch.zeros_like(local.cell_observed), target_dino_content=torch.full_like(local.target_dino_content, torch.nan),
                    observed_history=replace(history, observed=observed))
    facts, _ = model.grounding.materialize_facts(empty)
    assert torch.isfinite(facts.observed_content).all()
    assert facts.observed_content.count_nonzero() == 0
    torch.testing.assert_close(facts.observed_content, _measure(facts).current_reference[:, 0], atol=0, rtol=0)
    gradient = torch.autograd.grad(facts.observed_content.sum(), model.grounding.grounder.slot_seed)[0]
    assert torch.isfinite(gradient).all() and gradient.count_nonzero() == 0
