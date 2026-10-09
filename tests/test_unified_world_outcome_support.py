"""S must ignore unavailable past-K null payloads before outcome arithmetic.

The packets/facts come from the actual online encoder on artificial inputs.
Non-neutral reader weights are test-only; no pretrained/behavior claim follows.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.model.observed_outcome import ObservedOutcomeRead
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.runtime.qualification import synthetic_batch
from scripts.check_unified_model_flow import configuration


@pytest.fixture(scope="module", params=["A", "B"])
def world_sources(request):
    torch.manual_seed(7109)
    config = configuration(
        "small",
        "A" if request.param == "A" else "B",
        "conditional_object_v1",
        outcome_mode="robot_world_before_proposal_v2",
    )
    batch, normalizer = synthetic_batch(config, count=2, raw_side=32, device=torch.device("cpu"))
    model = ClearVLAMainlinePolicy(config).eval()
    model.configure_action_normalizer(normalizer)
    with torch.no_grad():
        cache, state, _ = model.encode_online(batch.online)
    assert cache.world_feedback is not None and state.top.intent.target_binding is not None
    feedback = cache.world_feedback.feedback
    assert feedback.window.observed.all() and feedback.view_observed.all()
    return feedback, state.top.facts, state.top.intent.target_binding, config.dimensions.hidden_size


def missing_past(feedback, *, all_past=False, window_observed=True):
    """Change only declared past support; the second batch row stays untouched."""
    support = feedback.view_observed.clone()
    support[0, slice(None) if all_past else -1] = False
    window = feedback.window.observed.clone()
    window[0] = window_observed
    if not window_observed:
        support[0] = False
    null = torch.where(support.any(-1)[..., None], feedback.null, 1.0)
    return replace(
        feedback, view_observed=support, null=null, window=replace(feedback.window, observed=window)
    )


def poisoned_null(feedback, bad):
    return replace(
        feedback,
        null=torch.where(
            feedback.view_observed.any(-1)[..., None],
            feedback.null,
            bad,
        ),
    )


def active_reader(world_sources):
    feedback, facts, _, hidden = world_sources
    torch.manual_seed(7110)
    reader = ObservedOutcomeRead(hidden, facts.content.shape[-1], feedback.camera_names)
    # Production retains its zero-output initialization. Open only this test
    # instance to measure source/binding and all reader-parameter derivatives.
    with torch.no_grad():
        reader.output.weight.normal_(0.0, 0.1)
    return reader


def live_inputs(world_sources):
    _, facts, binding, _ = world_sources
    source = facts.current_image_source
    assert source is not None
    image_logits = source.log_measure.detach().clone().requires_grad_()
    target_logits = binding.log_probability.detach().clone().requires_grad_()
    facts = replace(facts, current_image_source=replace(source, log_measure=image_logits))
    binding = TargetBinding.from_logits(
        target_logits[:, :-1],
        target_logits[:, -1:],
        binding.supported,
    )
    return facts, binding, (image_logits, target_logits)


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
@pytest.mark.parametrize("bf16", [False, True])
def test_unavailable_null_preserves_output_and_ordinary_source_binding_parameter_vjps(
    world_sources,
    bad,
    bf16,
):
    clean = missing_past(world_sources[0])
    dirty = poisoned_null(clean, bad)
    # This is admitted missing payload, not corruption of an observed value.
    clean.validate(strict=True)
    dirty.validate(strict=True)
    reader = active_reader(world_sources)
    control = deepcopy(reader)
    facts, binding, leaves = live_inputs(world_sources)
    clean_facts, clean_binding, clean_leaves = live_inputs(world_sources)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        actual = reader(dirty, facts, binding)
        expected = control(clean, clean_facts, clean_binding)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    got = torch.autograd.grad(actual.float().square().sum(), (*reader.parameters(), *leaves))
    want = torch.autograd.grad(
        expected.float().square().sum(), (*control.parameters(), *clean_leaves)
    )
    for actual_grad, expected_grad in zip(got, want, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    if not bf16:
        # Ordinary current-image and shared K+null binding routes must remain live.
        assert got[-2][0].abs().sum() > 0 and got[-1][0].abs().sum() > 0


def test_missing_null_cannot_poison_shared_gradients_from_another_batch_row(world_sources):
    clean = missing_past(world_sources[0])
    dirty = poisoned_null(clean, torch.nan)
    reader = active_reader(world_sources)
    control = deepcopy(reader)
    _, facts, binding, _ = world_sources
    actual = reader(dirty, facts, binding)[1]
    expected = control(clean, facts, binding)[1]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for name, parameter in reader.named_parameters():
        expected_grad = dict(control.named_parameters())[name].grad
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize("window_observed", [False, True])
def test_all_missing_past_preserves_reset_versus_unknown(world_sources, window_observed):
    clean = missing_past(world_sources[0], all_past=True, window_observed=window_observed)
    dirty = poisoned_null(clean, torch.nan)
    dirty.validate(strict=True)
    reader = active_reader(world_sources)
    _, facts, binding, _ = world_sources
    actual = reader(dirty, facts, binding)
    expected = reader(clean, facts, binding)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if window_observed:
        # An observed current window with no past match retains unknown status.
        assert actual[0].abs().sum() > 0
    else:
        assert actual[0].count_nonzero() == 0
        actual[0].sum().backward()
        for parameter in reader.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() == 0


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_observed_null_is_still_rejected(world_sources, bad):
    feedback, facts, binding, _ = world_sources
    null = feedback.null.clone()
    null[0, 0] = bad
    with pytest.raises(ValueError, match="nonfinite supported"):
        active_reader(world_sources)(replace(feedback, null=null), facts, binding)
