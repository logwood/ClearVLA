"""S must retain a camera's source log mass until task compatibility is added.

The input facts come from the actual artificial A/B online G path. Interventions
below change only the source law to expose numerical ownership, not trained
object identity. No optimizer, rollout or physical-performance claim follows.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_source_consistent_measurement import production as _production

from clearvla.vision.entity_chart import CanonicalImageReadSource

production = _production


class _BindingCaptured(Exception):
    pass


def _log_view(source):
    valid = source.supported.flatten(3)
    has = valid.any(-1, keepdim=True)
    logs = source.log_measure.flatten(3).masked_fill(~valid, -torch.inf)
    result = torch.logsumexp(torch.where(has, logs, 0.0), -1)
    return torch.where(has[..., 0], result, -torch.inf)


def _source_case(facts, *, log_gap, missing=False):
    source = facts.current_image_source
    assert source is not None and facts.current_image_measure is not None
    dimensions = (None,) * (source.log_measure.ndim - 3)
    old = _log_view(source)
    wanted = torch.zeros_like(old)
    wanted[..., 1] = -log_gap
    logs = source.log_measure - old[(..., *dimensions)] + wanted[(..., *dimensions)]
    support = source.supported.clone()
    if missing:
        if missing == "all":
            support.zero_()
        else:
            support[:, 0] = False
            support[:, 1, 1] = False
        logs = torch.where(support, logs, torch.nan)
    logs = logs.detach().requires_grad_()
    source = replace(source, log_measure=logs, supported=support)
    image = source.on_image(rows=facts.object_to_chart.shape[-2], columns=facts.object_to_chart.shape[-1])
    joint, _ = image.normalized((2, 3, 4))
    camera_valid = source.supported.flatten(3).any(-1)
    object_valid = camera_valid.any(-1, keepdim=True)
    overrides = dict(
        current_image_source=source,
        current_image_measure=image,
        object_to_chart=joint,
        validity=object_valid.float(),
        log_validity=torch.where(object_valid, 0.0, -torch.inf),
        camera_validity=camera_valid[..., None].float(),
        log_camera_validity=torch.where(camera_valid[..., None], 0.0, -torch.inf),
    )
    if isinstance(source, CanonicalImageReadSource):
        log_view = _log_view(source)
        overrides.update(view_log_mass=log_view, view_mass=log_view.exp())
    return replace(facts, **overrides), logs


def _capture(production, facts, *, bf16=False):
    model, _, batch, cache, _ = production
    organizer = model.intent.organizer
    assert organizer.shared_binder is not None
    captured = []

    def before(_module, args, kwargs):
        captured.append((args, kwargs))
        raise _BindingCaptured

    hook = organizer.shared_binder.register_forward_pre_hook(before, with_kwargs=True)
    history = cache.history
    try:
        with pytest.raises(_BindingCaptured), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            organizer(
                goal_tokens=batch.online.goal.tokens,
                goal_mask=batch.online.goal.mask,
                state_history=history.state_history,
                state=history.state,
                executed_history=history.executed_action_history,
                facts=facts,
                collect_diagnostics=False,
                history_timing=history.timing,
                instruction_reference=batch.online.instruction_reference,
                current_dino=batch.online.observation.dino_history[:, -1],
                executed_feedback=cache.world_feedback.feedback,
            )
    finally:
        hook.remove()
    return deepcopy(organizer.shared_binder), *captured[0]


def _open_discriminative_view(binder, args, kwargs):
    """One finite source-dependent compatibility offsets a -120 log prior.

    Solve the existing linear interaction's small design matrix. This changes
    test parameters, not the production score function or any evidence mask.
    """
    task, objects, _ = args
    with torch.no_grad():
        binder.score.weight.zero_()
        binder.null.weight.zero_()
        binder.null.bias.zero_()
        task_value = binder.task_value(task)
        mask = kwargs["task_mask"]
        mean = (task_value * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        values = binder.objects(objects.flatten(1, 2))
        context = mean[:, None].expand_as(values)
        design = torch.cat((values * context, values * torch.tanh(context)), -1)
        design = design.double().flatten(0, 1)
        wanted = torch.zeros(design.shape[0], 1, dtype=torch.double)
        wanted[1] = 120.0  # K0, wrist; every other K/view has zero score.
        weight = torch.linalg.lstsq(design, wanted, rcond=1e-10).solution[:, 0]
        torch.testing.assert_close(design @ weight, wanted[:, 0], atol=1e-6, rtol=0)
        binder.task_object_score.weight.copy_(weight.float()[None])


def test_actual_s_keeps_rare_discriminative_camera_until_compatibility(production):
    facts, source_log = _source_case(production[4].top.facts, log_gap=120.0)
    binder, args, kwargs = _capture(production, facts)
    _open_discriminative_view(binder, args, kwargs)
    # The public probability view really has underflowed; finite log mass has not.
    assert kwargs["view_mass"][..., 1].count_nonzero() == 0
    actual = binder(*args, **kwargs)
    expected = binder(*args, **dict(kwargs, view_log_mass=_log_view(facts.current_image_source)))
    assert expected.mass[0, 0] > 0.30
    torch.testing.assert_close(actual.mass, expected.mass, atol=2e-6, rtol=2e-5)
    parameters = (source_log, binder.task_object_score.weight)
    ga = torch.autograd.grad(actual.mass[0, 0], parameters, retain_graph=True)
    ge = torch.autograd.grad(expected.mass[0, 0], parameters)
    for actual_grad, expected_grad in zip(ga, ge, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
    assert ga[0][:, 0, 1].abs().sum() > 0.01


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("missing", [False, True, "all"])
def test_source_log_marginal_support_and_ordinary_vjp(production, bf16, missing):
    facts, source_log = _source_case(production[4].top.facts, log_gap=120.0, missing=missing)
    binder, args, kwargs = _capture(production, facts, bf16=bf16)
    actual_log = kwargs["view_log_mass"]
    assert actual_log is not None, "S discarded the producer's finite log allocation"
    expected_log = _log_view(facts.current_image_source)
    torch.testing.assert_close(actual_log, expected_log, atol=2e-5, rtol=0)
    legal = torch.isfinite(expected_log)
    coefficients = torch.arange(expected_log.numel()).reshape_as(expected_log).float() + 1
    actual_loss = (torch.where(legal, actual_log, 0.0) * coefficients).sum()
    expected_loss = (torch.where(legal, expected_log, 0.0) * coefficients).sum()
    actual_grad = torch.autograd.grad(actual_loss, source_log, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_loss, source_log, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=2e-5)
    assert torch.isfinite(actual_grad).all()
    if facts.current_image_source.supported.any():
        assert actual_grad[facts.current_image_source.supported].abs().sum() > 0
    assert actual_grad[~facts.current_image_source.supported].count_nonzero() == 0
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        bound = binder(*args, **kwargs)
    if missing:
        assert bound.mass[:, 0].count_nonzero() == 0
    assert torch.isfinite(bound.mass).all()
    if missing == "all":
        assert bound.mass.count_nonzero() == 0
        torch.testing.assert_close(bound.null_mass, torch.ones_like(bound.null_mass))


def test_representable_camera_mass_keeps_existing_binding_and_parameter_vjp(production):
    facts, _ = _source_case(production[4].top.facts, log_gap=2.0)
    binder, args, kwargs = _capture(production, facts)
    actual = binder(*args, **kwargs)
    probability_control = binder(*args, **dict(kwargs, view_log_mass=None))
    torch.testing.assert_close(actual.log_probability, probability_control.log_probability, atol=2e-6, rtol=2e-6)
    parameters = tuple(binder.parameters())
    ga = torch.autograd.grad(actual.mass.square().sum(), parameters, retain_graph=True)
    gc = torch.autograd.grad(probability_control.mass.square().sum(), parameters)
    for actual_grad, control_grad in zip(ga, gc, strict=True):
        torch.testing.assert_close(actual_grad, control_grad, atol=2e-6, rtol=2e-5)


def test_view_marginal_has_no_spurious_pixel_knot_coordinate_gradient(production):
    facts, source_log = _source_case(production[4].top.facts, log_gap=2.0)
    source = facts.current_image_source
    assert source is not None
    if isinstance(source, CanonicalImageReadSource):
        # B already owns canonical-cell mass; its original local coordinates
        # do not participate in re-reading that immutable canonical measure.
        return
    coordinates = torch.ones_like(source.spatial.coordinates).requires_grad_()
    spatial = replace(source.spatial, coordinates=coordinates)
    source = replace(source, spatial=spatial)
    image = source.on_image(rows=facts.object_to_chart.shape[-2], columns=facts.object_to_chart.shape[-1])
    probability, _ = image.normalized((2, 3, 4))
    facts = replace(facts, current_image_source=source, current_image_measure=image,
                    dense_chart=replace(facts.dense_chart, current_image_support=spatial),
                    object_to_chart=probability)
    _, _, kwargs = _capture(production, facts)
    log_view = kwargs["view_log_mass"]
    assert log_view is not None
    actual = log_view.log_softmax(-1)
    # Independent linear bilinear reference keeps all four corner derivatives.
    from clearvla.vision.entity_chart import pushforward_to_current_image
    linear = pushforward_to_current_image(source_log.exp(), spatial,
                rows=facts.object_to_chart.shape[-2], columns=facts.object_to_chart.shape[-1])
    camera_mass = linear.sum((-2, -1))
    expected = (camera_mass / camera_mass.sum(-1, keepdim=True)).log()
    weight = torch.arange(actual.numel()).reshape_as(actual).float()
    ga = torch.autograd.grad((actual * weight).sum(), (source_log, coordinates),
                             retain_graph=True, allow_unused=True)
    ge = torch.autograd.grad((expected * weight).sum(), (source_log, coordinates))
    for a, e in zip(ga, ge, strict=True):
        if a is None:
            a = torch.zeros_like(e)
        torch.testing.assert_close(a, e, atol=5e-6, rtol=3e-5)
    assert ga[1] is None or ga[1].abs().max() < 5e-6
