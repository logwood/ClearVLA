"""Independent current/reference source masks, through actual S and P3.

Exact-descriptor/chart-coordinate examples are synthetic mathematical checks,
not certification of long-gap physical correspondence or learned task success.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_unified_source_support_ownership import measured_case

from clearvla.mainline.instruction_reference import InstructionReference
from clearvla.mainline.model.instruction_posterior import (
    PosteriorInstructionChangePlanRead,
    PosteriorInstructionReferenceRead,
)
from clearvla.mainline.model.target_binding import TargetBinding


def case(*, legacy=False):
    torch.manual_seed(44321)
    _, facts, current = measured_case()
    current = current.flatten(2, 3)
    mask = torch.tensor([[[False, True, False, True]]])
    ref = current.reshape(1, 1, 2, 2, 8).flip(-2).flatten(2, 3)
    reference = InstructionReference(
        torch.where(mask[..., None], ref, torch.nan),
        torch.zeros(1, 3),
        mask,
        torch.tensor([4]),
    )
    binding = TargetBinding(
        torch.tensor([[0.4, 0.5, 0.1]]).log(), torch.ones(1, 2, dtype=torch.bool)
    )
    reader = PosteriorInstructionReferenceRead(
        hidden=16,
        content_dim=8,
        state_dim=3,
        camera_names=("top",),
        observation_measurement_mode="legacy_v1" if legacy else "source_consistent_v1",
    )
    return reader, facts, current, reference, binding


def read(items):
    reader, facts, current, reference, binding = items
    return reader(
        reference=reference,
        current_dino=current,
        current_state=reference.state,
        facts=facts,
        binding=binding,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disjoint_image_masks_do_not_delete_known_displacement(dtype):
    values = case()
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        output, e = read(values)
        p = e.posterior
        assert p is not None
        p.validate(strict=True)
        assert p.observed.count_nonzero() == 0  # no overlap at the SAME coordinates
        torch.testing.assert_close(e.image_delta, torch.tensor([-2.0, 0.0]).expand(1, 2, 1, 2))
        torch.testing.assert_close(e.content_delta, torch.zeros_like(e.content_delta))
        assert p.current_null.count_nonzero() == p.reference_null.count_nonzero() == 0
        plan = PosteriorInstructionChangePlanRead(
            hidden=16, content_dim=8, state_dim=3, camera_names=("top",)
        )
        prepared = plan.prepare(e)
        assert prepared.image.abs().sum() > 0
        query = torch.randn(1, 4, 2, 16, requires_grad=True)
        action_context = plan(e, query, prepared)
        loss = action_context.square().sum() + output.square().sum()
    loss.backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
    for module in (values[0], plan):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_missing_reference_is_unknown_correspondence_not_missing_current_source():
    r, f, now, ref, b = case()
    ref = replace(
        ref, observed=torch.zeros_like(ref.observed), dino=torch.full_like(ref.dino, torch.nan)
    )
    _, e = read((r, f, now, ref, b))
    p = e.posterior
    assert p is not None
    p.validate(strict=True)
    torch.testing.assert_close(p.source_probability.sum(-1), torch.ones(1, 2, 1))
    torch.testing.assert_close(p.reference_null, torch.ones(1, 2, 1))
    assert e.view_observed.count_nonzero() == e.image_delta.count_nonzero() == 0
    assert p.current_probability.count_nonzero() == p.reference_probability.count_nonzero() == 0
    values = r.values.prepare(e)
    assert values.joint is not None
    assert (
        values.content.count_nonzero()
        == values.image.count_nonzero()
        == values.joint.count_nonzero()
        == 0
    )


def test_missing_current_source_stays_absent_and_cannot_use_reference_visibility():
    r, f, now, ref, b = case()
    chart = replace(f.dense_chart, cell_observed=torch.zeros_like(f.dense_chart.cell_observed))
    _, e = read((r, replace(f, dense_chart=chart), torch.full_like(now, torch.nan), ref, b))
    p = e.posterior
    assert p is not None
    assert p.source_probability.count_nonzero() == p.reference_probability.count_nonzero() == 0
    assert e.image_delta.count_nonzero() == 0
    p.validate(strict=True)


def test_independent_masks_quarantine_before_nonlinear_projection_and_backward():
    r, f, now, ref, b = case()
    own = f.dense_chart.cell_observed[..., 0].flatten(2)
    now = torch.where(own[..., None], now, torch.nan).requires_grad_()
    ref = replace(ref, dino=ref.dino.clone().requires_grad_())
    out, e = read((r, f, now, ref, b))
    vals = r.values.prepare(e)
    assert vals.joint is not None
    gradients = torch.autograd.grad(out.square().sum() + vals.joint.square().sum(), (now, ref.dino))
    for gradient, support in zip(gradients, (own, ref.observed), strict=True):
        assert torch.isfinite(gradient).all()
        assert gradient[~support].count_nonzero() == 0
        assert gradient[support].abs().sum() > 0
    with pytest.raises(ValueError, match="finite"):
        read((r, f, now.detach().fill_(torch.nan), ref, b))


def test_independent_support_type_rejects_cross_domain_probability():
    _, e = read(case())
    p = e.posterior
    assert p is not None
    with pytest.raises(ValueError, match="both source masks"):
        replace(p, reference_observed=None).validate(strict=True)
    with pytest.raises(ValueError, match="producer support"):
        replace(p, reference_probability=p.current_probability).validate(strict=True)
    with pytest.raises(ValueError, match="diagnostic intersection"):
        replace(p, observed=p.current_support).validate(strict=True)
    index = torch.tensor([1, 0])
    moved = e.permute(index, e.binding.permute(index))
    moved.validate(strict=True)
    assert moved.posterior is not None
    assert moved.posterior.current_observed is p.current_observed
    torch.testing.assert_close(moved.image_delta, e.image_delta[:, index])


def test_legacy_shared_mask_path_keeps_its_counterexample_and_layout():
    _, e = read(case(legacy=True))
    assert e.image_delta.count_nonzero() == 0
    p = e.posterior
    assert p is not None and p.current_observed is p.reference_observed is None
    assert p.current_support is p.observed and p.reference_support is p.observed
    p.validate(strict=True)


def test_zero_change_with_same_partial_observation():
    r, f, now, ref, b = case()
    own = f.dense_chart.cell_observed[..., 0].flatten(2)
    same = replace(ref, observed=own, dino=torch.where(own[..., None], now, torch.nan))
    _, e = read((r, f, now, same, b))
    assert e.image_delta.count_nonzero() == e.content_delta.count_nonzero() == 0
    v = r.values.prepare(e)
    assert v.joint is not None
    assert v.image.count_nonzero() == v.content.count_nonzero() == v.joint.count_nonzero() == 0


def test_endpoint_comparison_uses_current_support_not_same_coordinate_intersection():
    from clearvla.mainline.model.annotation_goal import (
        AnnotatedGoalValueRead,
        EndpointGoalPredictor,
        compare_goal_to_current,
    )

    r, f, now, ref, b = case()
    _, change = read((r, f, now, ref, b))
    predictor = EndpointGoalPredictor(
        hidden=16, content_dim=8, state_dim=3, camera_names=("top",), heads=2
    )
    prediction = predictor(reference=ref, language=torch.randn(1, 4, 16))
    evidence = compare_goal_to_current(prediction, change)
    assert change.posterior is not None
    assert evidence.current_observed is change.posterior.current_support
    assert evidence.current_observed.count_nonzero() > 0
    values = AnnotatedGoalValueRead(
        hidden=16, content_dim=8, state_dim=3, camera_names=("top",)
    ).prepare(evidence)
    (values.content.square().sum() + values.image.square().sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in predictor.parameters())


def test_named_migration_and_ABI_do_not_relabel_old_support_contract(tmp_path):
    from test_mainline_global_task import abi_for

    from clearvla.mainline.config import load_config
    from clearvla.mainline.runtime.causal_identity_migration import (
        CAUSAL_IDENTITY_AB_V1,
        CAUSAL_UNIFIED_REFERENCE_V1,
        CAUSAL_UNIFIED_SOURCE_V1,
        CAUSAL_UNIFIED_VALUES_V1,
        SOURCE_DIGEST,
        allowed_source_paths,
        validate_selection,
    )
    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    from clearvla.mainline.train import _parser
    from scripts.check_unified_model_flow import configuration

    saved = load_config("configs/mainline/dinov3_causal_repair_calvin_20261006.json")
    new = load_config("configs/mainline/unified_values_a_calvin_check.json")
    validate_selection(saved, new, SOURCE_DIGEST, mode=CAUSAL_UNIFIED_REFERENCE_V1)
    for old in (CAUSAL_IDENTITY_AB_V1, CAUSAL_UNIFIED_SOURCE_V1, CAUSAL_UNIFIED_VALUES_V1):
        assert "clearvla/mainline/instruction_posterior.py" not in allowed_source_paths(old)
    assert "clearvla/mainline/instruction_posterior.py" in allowed_source_paths(
        CAUSAL_UNIFIED_REFERENCE_V1
    )
    args = _parser().parse_args(
        [
            "--config",
            "configs/mainline/unified_values_a_calvin_check.json",
            "--init-model-contract-migration",
            CAUSAL_UNIFIED_REFERENCE_V1,
        ]
    )
    assert args.init_model_contract_migration == CAUSAL_UNIFIED_REFERENCE_V1
    _, abi, _ = abi_for(configuration("small", "A"), tmp_path)
    validate_deployment_abi(abi)
    corrupt = deepcopy(abi)
    identity = corrupt["causal_identity"]
    assert isinstance(identity, dict)
    identity.pop("instruction_observation_support")
    with pytest.raises(ValueError):
        validate_deployment_abi(corrupt)


def test_actual_source_probability_has_ordinary_nonzero_coordinate_moment_gradient():
    from clearvla.vision.entity_chart import CanonicalImageReadSource

    r, f, now, ref, binding = case()
    base_source = f.current_image_source
    assert base_source is not None
    support = base_source.supported.clone()
    support[..., 1, 0] = True
    log_measure = torch.zeros_like(support, dtype=torch.float32, requires_grad=True)
    target_mask = torch.tensor([[[False, True, True, False]]])
    target = now.clone()
    target[..., 1, :] = now[..., 0, :]
    ref = replace(
        ref, observed=target_mask, dino=torch.where(target_mask[..., None], target, torch.nan)
    )

    def evaluate(logits):
        source = CanonicalImageReadSource(logits, support, base_source.spatial)
        facts = replace(f, current_image_source=source)
        _, e = read((r, facts, now, ref, binding))
        return e.image_delta[..., 0].sum()

    gradient = torch.autograd.grad(evaluate(log_measure), log_measure)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[..., 0, 0], torch.full((1, 2, 1), -0.5))
    torch.testing.assert_close(gradient[..., 1, 0], torch.full((1, 2, 1), 0.5))
    assert gradient[~support].count_nonzero() == 0
    direction = torch.zeros_like(log_measure)
    direction[..., 0, 0] = 1
    eps = 1e-3
    numerical = (
        evaluate(log_measure.detach() + eps * direction)
        - evaluate(log_measure.detach() - eps * direction)
    ) / (2 * eps)
    torch.testing.assert_close((gradient * direction).sum(), numerical, rtol=2e-4, atol=2e-4)
