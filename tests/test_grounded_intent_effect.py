from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from clearvla.policy.flow_dino_evidence import (
    _intervene_grounded_fact_slots,
)
from clearvla.policy.grounded_intent_effect import (
    GROUNDING_MANIFEST,
    BoundedFutureEffectReader,
    ConsequenceConditionedPolicyPlanCompiler,
    FutureEffectField,
    GroundedFactSet,
    GroundedWorldEffectCompiler,
    StatelessIntentOrganizer,
    StatelessIntentState,
    ZeroPreservingConsequenceOrganizer,
    bounded_owner_update,
    manifest_from_mapping,
    sample_spatial_slots,
)
from clearvla.policy.trunk_primitives import TemporalDynamicsBoundDiTBlock


def test_grounded_probe_launcher_uses_public_intervention_names() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "scripts"
        / "run_grounded_intent_effect_323_model_path_probe.sh"
    ).read_text(encoding="utf-8")
    for mode in (
        "address_g3_episode_shuffle",
        "address_g3_slot_permute",
        "address_g3_slot_mean",
        "intent_interval_h4_8_shuffle",
        "intent_interval_h32_48_shuffle",
        "future_effect_h4_8_shuffle",
        "future_effect_h32_48_shuffle",
        "protected_detail_episode_shuffle",
        "p3_precision_delta_episode_shuffle",
        "p3_temporal_delta_episode_shuffle",
        "future_effect_reliability_one",
    ):
        assert mode in launcher
    for internal_or_invalid_name in (
        "address_g3_shuffle",
        "intent_interval_h4_8_episode_shuffle",
        "future_effect_h4_8_episode_shuffle",
        "protected_detail_shuffle ",
    ):
        assert internal_or_invalid_name not in launcher
    full_modes = next(
        line
        for line in launcher.splitlines()
        if line.startswith("FULL_MODEL_PATH_MODES=")
    )
    assert "goal_zero" in full_modes
    assert "goal_episode_shuffle" in full_modes
    assert "intent_goal_set_zero" not in full_modes
    assert "intent_goal_set_episode_shuffle" not in full_modes
    followup_modes = next(
        line
        for line in launcher.splitlines()
        if line.startswith("INTENT_EFFECT_FOLLOWUP_MODEL_PATH_MODES=")
    )
    for mode in (
        "intent_goal_set_zero",
        "intent_achieved_zero",
        "intent_remaining_zero",
        "intent_temporal_zero",
        "future_effect_reliability_one",
        "address_g3_slot_permute",
        "address_g3_slot_mean",
    ):
        assert mode in followup_modes
    assert 'PROBE_PROFILE="${PROBE_PROFILE:-full}"' in launcher

    training_launcher = (
        root / "scripts" / "current_grounded_intent_effect_323.sh"
    ).read_text(encoding="utf-8")
    for script in (training_launcher, launcher):
        assert "/data/senwang/data" in script
        assert "/data/senwang/checkpoint" in script
        assert (
            "${DATA_ROOT:-/data/liang.zhang/dataset/grab_pen_single/grab_pen_single}"
            in script
        )
        assert "${CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/cache_336}" in script
        assert (
            "${DINO_CACHE_DIR:-${CLEARVLA_DATA_CACHE_ROOT}/dinov2_cache_336}"
            in script
        )
    assert "${T5_CONDITION_PATH:-${CLEARVLA_CHECKPOINT_ROOT}/grasp_pen_embed.pt}" in (
        training_launcher
    )
    assert "STAGE1_CHECKPOINT=" not in training_launcher
    assert (
        "${CHECKPOINT:-${CLEARVLA_CHECKPOINT_ROOT}/"
        "v119_grounded_intent_effect_323_b8/checkpoints/latest.pt}"
    ) in launcher


def _facts(
    *,
    batch: int = 2,
    cameras: int = 2,
    grid: int = 3,
    slots: int = 4,
    hidden: int = 32,
    route: int = 16,
) -> GroundedFactSet:
    generator = torch.Generator().manual_seed(101)
    owner_logits = {
        name: torch.randn(
            batch,
            cameras,
            grid,
            grid,
            slots,
            generator=generator,
        )
        for name in ("semantic", "appearance", "geometry")
    }
    axis = torch.linspace(-1.0, 1.0, grid)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).reshape(
        1,
        1,
        grid,
        grid,
        1,
        2,
    ).expand(batch, cameras, -1, -1, slots, -1)
    coordinates = coordinates + 0.05 * torch.randn(
        coordinates.shape,
        generator=generator,
    )
    facts = GroundedFactSet(
        public_scene_base=torch.randn(
            batch,
            cameras,
            grid,
            grid,
            hidden,
            generator=generator,
        ),
        content_slots=torch.randn(
            batch,
            cameras,
            grid,
            grid,
            slots,
            hidden,
            generator=generator,
        ),
        semantic_slots=torch.randn(
            batch,
            cameras,
            grid,
            grid,
            slots,
            route,
            generator=generator,
        ),
        appearance_slots=torch.randn(
            batch,
            cameras,
            grid,
            grid,
            slots,
            route,
            generator=generator,
        ),
        geometry_slots=torch.randn(
            batch,
            cameras,
            grid,
            grid,
            slots,
            route,
            generator=generator,
        ),
        semantic_owner_probs=torch.softmax(owner_logits["semantic"], dim=-1),
        appearance_owner_probs=torch.softmax(owner_logits["appearance"], dim=-1),
        geometry_owner_probs=torch.softmax(owner_logits["geometry"], dim=-1),
        slot_coordinates=coordinates.clamp(-1.0, 1.0),
        slot_support=torch.full(
            (batch, cameras, grid, grid, slots),
            0.25,
        ),
        slot_validity=torch.ones(
            batch,
            cameras,
            grid,
            grid,
            slots,
            1,
        ),
    )
    facts.validate()
    return facts


@pytest.mark.parametrize(
    "mode",
    ("address_g3_slot_permute", "address_g3_slot_mean"),
)
def test_grounded_g3_slot_intervention_preserves_public_p1_base(
    mode: str,
) -> None:
    facts = _facts()
    changed, delta = _intervene_grounded_fact_slots(facts, mode)
    torch.testing.assert_close(
        changed.public_scene_base,
        facts.public_scene_base,
        rtol=0.0,
        atol=0.0,
    )
    assert float(delta) > 0.0
    if mode == "address_g3_slot_permute":
        torch.testing.assert_close(
            changed.semantic_slots,
            facts.semantic_slots.roll(shifts=1, dims=-2),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            changed.semantic_owner_probs,
            facts.semantic_owner_probs.roll(shifts=1, dims=-1),
            rtol=0.0,
            atol=0.0,
        )
    else:
        for value in (
            changed.content_slots,
            changed.semantic_slots,
            changed.appearance_slots,
            changed.geometry_slots,
            changed.slot_coordinates,
            changed.slot_validity,
        ):
            torch.testing.assert_close(
                value,
                value[..., :1, :].expand_as(value),
                rtol=0.0,
                atol=0.0,
            )
        for value in (
            changed.semantic_owner_probs,
            changed.appearance_owner_probs,
            changed.geometry_owner_probs,
            changed.slot_support,
        ):
            torch.testing.assert_close(
                value,
                value[..., :1].expand_as(value),
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            changed.semantic_slots.mean(dim=-2),
            facts.semantic_slots.mean(dim=-2),
        )


def test_grounded_g3_single_slot_intervention_is_an_exact_noop() -> None:
    facts = _facts(slots=1)
    for mode in ("address_g3_slot_permute", "address_g3_slot_mean"):
        changed, delta = _intervene_grounded_fact_slots(facts, mode)
        torch.testing.assert_close(
            changed.semantic_slots,
            facts.semantic_slots,
            rtol=0.0,
            atol=0.0,
        )
        assert float(delta) == 0.0


def _intent_inputs(
    *,
    batch: int = 2,
    hidden: int = 32,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(103)
    return {
        "goal_tokens": torch.randn(batch, 9, hidden, generator=generator),
        "state_history_tokens": torch.randn(
            batch,
            5,
            8,
            generator=generator,
        ),
        "action_history_tokens": torch.randn(
            batch,
            4,
            7,
            generator=generator,
        ),
    }


def _intent(
    facts: GroundedFactSet,
    *,
    hidden: int = 32,
    horizon: int = 12,
) -> tuple[StatelessIntentOrganizer, StatelessIntentState]:
    module = StatelessIntentOrganizer(
        hidden=hidden,
        state_dim=8,
        action_dim=7,
        fact_dim=facts.route_dim,
        action_horizon=horizon,
        heads=4,
    )
    state, _ = module(
        facts=facts,
        **_intent_inputs(batch=facts.batch, hidden=hidden),
    )
    return module, state


def _completed_effect(
    facts: GroundedFactSet,
    *,
    hidden: int = 32,
    horizon: int = 12,
) -> tuple[StatelessIntentState, GroundedWorldEffectCompiler, FutureEffectField]:
    _, intent = _intent(facts, hidden=hidden, horizon=horizon)
    compiler = GroundedWorldEffectCompiler(
        hidden=hidden,
        fact_dim=facts.route_dim,
        route_dim=16,
        heads=4,
    )
    generator = torch.Generator().manual_seed(107)
    world = torch.randn(
        facts.batch,
        4,
        int(facts.semantic_slots.shape[1]),
        int(facts.semantic_slots.shape[2]),
        int(facts.semantic_slots.shape[3]),
        hidden,
        generator=generator,
    )
    proposal = torch.randn(
        facts.batch,
        4,
        hidden,
        generator=generator,
    )
    working = compiler.initialize(proposal)
    working, _ = compiler.forward_w1(
        world_tokens=world,
        facts=facts,
        intent=intent,
        working=working,
        output_dtype=world.dtype,
    )
    w1 = working.effect_w1
    assert w1 is not None
    working, _ = compiler.forward_w2(
        world_tokens=world,
        facts=facts,
        intent=intent,
        working=working,
        output_dtype=world.dtype,
    )
    effect = working.effect
    assert effect is not None
    torch.testing.assert_close(effect.semantic_delta[:, :2], w1.semantic_delta)
    return intent, compiler, effect


def test_manifest_is_small_capability_identity() -> None:
    GROUNDING_MANIFEST.validate()
    manifest = GROUNDING_MANIFEST.as_dict()
    assert manifest["capability"] == "grounded_intent_effect_323"
    assert manifest["topology"] == (3, 2, 3)
    assert manifest["intervals"] == ((4, 8), (8, 16), (16, 32), (32, 48))
    assert not any(str(key).startswith("v") for key in manifest)
    assert manifest_from_mapping(manifest).as_dict() == manifest
    with pytest.raises(ValueError, match="topology must have three"):
        manifest_from_mapping({**manifest, "topology": (3, 2)})
    with pytest.raises(ValueError, match="intervals must contain pairs"):
        manifest_from_mapping(
            {
                **manifest,
                "intervals": ((4, 8), (8, 16), (16,), (32, 48)),
            }
        )


def test_g2_to_g3_zero_residual_is_exact_identity() -> None:
    probability = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    inherited = bounded_owner_update(
        probability,
        torch.zeros_like(probability),
    )
    torch.testing.assert_close(inherited, probability, rtol=1e-6, atol=1e-7)


def test_slot_sampler_preserves_object_permutation() -> None:
    chart = torch.arange(2 * 3 * 3 * 5, dtype=torch.float32).reshape(
        1,
        2,
        3,
        3,
        5,
    )
    facts = _facts(batch=1, cameras=2, grid=3, slots=4, hidden=5)
    sampled = sample_spatial_slots(chart, facts.slot_coordinates)
    permutation = torch.tensor((2, 0, 3, 1))
    permuted = sample_spatial_slots(
        chart,
        facts.slot_coordinates[..., permutation, :],
    )
    torch.testing.assert_close(permuted, sampled[..., permutation, :])


def test_s_has_no_frame_progress_input_and_uses_typed_observations() -> None:
    facts = _facts()
    module = StatelessIntentOrganizer(
        hidden=32,
        state_dim=8,
        action_dim=7,
        fact_dim=facts.route_dim,
        action_horizon=12,
        heads=4,
    )
    state, metrics = module(
        facts=facts,
        **_intent_inputs(batch=facts.batch, hidden=32),
    )
    names = tuple(name for name, _ in module.named_parameters())
    assert not any("progress" in name or "phase" in name for name in names)
    assert state.interval_intents.shape == (2, 4, 32)
    assert state.temporal_control.shape == (2, 12, 32)
    assert "grounded_s_interval_goal_attention_entropy" in metrics
    for interval in ("h4_8", "h8_16", "h16_32", "h32_48"):
        assert f"grounded_s_{interval}_goal_attention_entropy" in metrics
        assert f"grounded_s_{interval}_source_attention_entropy" in metrics
        for head in range(4):
            assert f"grounded_s_{interval}_goal_head_{head}_entropy" in metrics

    changed_facts = replace(
        facts,
        geometry_slots=facts.geometry_slots.roll(shifts=1, dims=-2),
    )
    with torch.no_grad():
        changed, _ = module(facts=changed_facts, **_intent_inputs())
    assert not torch.allclose(state.interval_intents, changed.interval_intents)


def test_s_diagnostics_do_not_change_the_bf16_value_path() -> None:
    torch.manual_seed(119)
    facts = _facts(
        batch=2,
        cameras=1,
        grid=2,
        slots=2,
        hidden=16,
        route=8,
    )
    module = StatelessIntentOrganizer(
        hidden=16,
        state_dim=8,
        action_dim=7,
        fact_dim=8,
        action_horizon=6,
        heads=4,
    ).eval()
    inputs = _intent_inputs(batch=2, hidden=16)
    with torch.no_grad(), torch.autocast(
        device_type="cpu",
        dtype=torch.bfloat16,
    ):
        lean, lean_metrics = module(
            facts=facts,
            collect_diagnostics=False,
            **inputs,
        )
        audited, audited_metrics = module(
            facts=facts,
            collect_diagnostics=True,
            **inputs,
        )
    assert lean_metrics == {}
    assert "grounded_s_interval_goal_attention_entropy" in audited_metrics
    for name in (
        "protected_goal_tokens",
        "achieved_evidence",
        "remaining_goal",
        "interval_intents",
        "temporal_control",
        "completion_evidence",
        "completion_probability",
        "completion_uncertainty",
        "goal_attention",
        "interval_source_attention",
    ):
        torch.testing.assert_close(
            getattr(lean, name),
            getattr(audited, name),
            rtol=0.0,
            atol=0.0,
        )


def test_s_does_not_invent_cross_modality_history_order() -> None:
    facts = _facts()
    module, _ = _intent(facts)
    inputs = _intent_inputs()
    with torch.no_grad():
        baseline = module._history(
            inputs["state_history_tokens"],
            inputs["action_history_tokens"],
        )
        changed_action = module._history(
            inputs["state_history_tokens"],
            inputs["action_history_tokens"].roll(shifts=1, dims=1),
        )
    state_tokens = int(inputs["state_history_tokens"].shape[1])
    torch.testing.assert_close(
        baseline[:, :state_tokens],
        changed_action[:, :state_tokens],
    )
    assert not torch.allclose(
        baseline[:, state_tokens:],
        changed_action[:, state_tokens:],
    )


def test_grounded_g_mask_excludes_intent_and_history_sources() -> None:
    lengths = {
        "task": 4,
        "state": 1,
        "state_history": 3,
        "executed": 2,
        "proposal": 4,
        "trajectory": 6,
        "stage": 1,
        "rollout": 8,
        "registers": 2,
    }
    offset = 0
    slices: dict[str, slice] = {}
    for name, length in lengths.items():
        slices[name] = slice(offset, offset + length)
        offset += length
    mask = TemporalDynamicsBoundDiTBlock._directed_attention_mask(
        offset,
        slices,
        device=torch.device("cpu"),
        role="grounding",
        grounded_fact_only=True,
    )
    for query_name in ("stage", "rollout"):
        for source_name in (
            "task",
            "state_history",
            "executed",
            "proposal",
            "trajectory",
        ):
            assert bool(mask[slices[query_name], slices[source_name]].all())
        assert not bool(mask[slices[query_name], slices["state"]].any())


def test_grounded_p1_mask_accepts_only_the_explicit_trajectory_query() -> None:
    lengths = {
        "task": 4,
        "state": 1,
        "state_history": 3,
        "executed": 2,
        "proposal": 4,
        "trajectory": 6,
        "stage": 1,
        "rollout": 8,
        "registers": 2,
    }
    offset = 0
    slices: dict[str, slice] = {}
    for name, length in lengths.items():
        slices[name] = slice(offset, offset + length)
        offset += length
    mask = TemporalDynamicsBoundDiTBlock._directed_attention_mask(
        offset,
        slices,
        device=torch.device("cpu"),
        role="policy",
        policy_explicit_handoff_only=True,
        grounded_policy_explicit_only=True,
    )
    for source_name in (
        "task",
        "state",
        "state_history",
        "executed",
        "proposal",
        "registers",
        "stage",
        "rollout",
    ):
        assert bool(mask[slices["trajectory"], slices[source_name]].all())
    assert not bool(
        mask[slices["trajectory"], slices["trajectory"]].any()
    )


def test_w_preserves_four_intervals_and_object_identity() -> None:
    facts = _facts()
    _, _, field = _completed_effect(facts)
    field.validate()
    assert field.semantic_delta.shape[:2] == (2, 4)
    assert field.semantic_delta.shape[-2] == 4
    assert field.interval_names == ("h4_8", "h8_16", "h16_32", "h32_48")
    object_variation = field.semantic_delta.float().std(
        dim=-2,
        unbiased=False,
    ).mean()
    assert float(object_variation.detach()) > 0.0


def test_w2_diagnostics_describe_only_its_late_intervals() -> None:
    facts = _facts()
    _, intent = _intent(facts)
    compiler = GroundedWorldEffectCompiler(
        hidden=32,
        fact_dim=facts.route_dim,
        route_dim=16,
        heads=4,
    )
    world = torch.randn(2, 4, 2, 3, 3, 32)
    working = compiler.initialize(torch.randn(2, 4, 32))
    working, _ = compiler.forward_w1(
        world_tokens=world,
        facts=facts,
        intent=intent,
        working=working,
        output_dtype=world.dtype,
    )
    working, metrics = compiler.forward_w2(
        world_tokens=world,
        facts=facts,
        intent=intent,
        working=working,
        output_dtype=world.dtype,
    )

    assert working.effect is not None
    assert "grounded_w2_h16_32_semantic_rms" in metrics
    assert "grounded_w2_h32_48_semantic_rms" in metrics
    assert "grounded_w2_h4_8_semantic_rms" not in metrics
    assert "grounded_w2_h8_16_semantic_rms" not in metrics


def test_w_is_equivariant_to_object_slot_permutation() -> None:
    facts = _facts(batch=1)
    intent, compiler, _ = _completed_effect(facts)
    generator = torch.Generator().manual_seed(109)
    world = torch.randn(
        1,
        4,
        2,
        3,
        3,
        32,
        generator=generator,
    )
    proposal = torch.randn(1, 4, 32, generator=generator)

    def run(current_facts: GroundedFactSet) -> FutureEffectField:
        working = compiler.initialize(proposal)
        working, _ = compiler.forward_w1(
            world_tokens=world,
            facts=current_facts,
            intent=intent,
            working=working,
            output_dtype=world.dtype,
            collect_diagnostics=False,
        )
        working, _ = compiler.forward_w2(
            world_tokens=world,
            facts=current_facts,
            intent=intent,
            working=working,
            output_dtype=world.dtype,
            collect_diagnostics=False,
        )
        assert working.effect is not None
        return working.effect

    baseline = run(facts)
    permutation = torch.tensor((3, 1, 0, 2))
    permuted_facts = replace(
        facts,
        content_slots=facts.content_slots[..., permutation, :],
        semantic_slots=facts.semantic_slots[..., permutation, :],
        appearance_slots=facts.appearance_slots[..., permutation, :],
        geometry_slots=facts.geometry_slots[..., permutation, :],
        semantic_owner_probs=facts.semantic_owner_probs[..., permutation],
        appearance_owner_probs=facts.appearance_owner_probs[..., permutation],
        geometry_owner_probs=facts.geometry_owner_probs[..., permutation],
        slot_coordinates=facts.slot_coordinates[..., permutation, :],
        slot_support=facts.slot_support[..., permutation],
        slot_validity=facts.slot_validity[..., permutation, :],
    )
    permuted = run(permuted_facts)
    torch.testing.assert_close(
        permuted.semantic_delta,
        baseline.semantic_delta[..., permutation, :],
        rtol=2e-5,
        atol=2e-6,
    )


def test_p2_read_is_equivariant_to_object_slot_permutation() -> None:
    facts = _facts()
    intent, _, field = _completed_effect(facts)
    reader = BoundedFutureEffectReader(hidden=32, horizon=12, basis=3)
    query = torch.randn(2, 12, 3, 32)
    baseline, _ = reader(query, field, intent, collect_diagnostics=False)
    permutation = torch.tensor((2, 0, 3, 1))
    permuted = replace(
        field,
        current_reference=field.current_reference[..., permutation, :],
        semantic_delta=field.semantic_delta[..., permutation, :],
        transport_delta=field.transport_delta[..., permutation, :],
        covariance_delta=field.covariance_delta[..., permutation, :],
        visibility_change=field.visibility_change[..., permutation, :],
        persistence_change=field.persistence_change[..., permutation, :],
        reliability=field.reliability[..., permutation, :],
        validity=field.validity[..., permutation, :],
        uncertainty=field.uncertainty[..., permutation, :],
        source_coordinates=field.source_coordinates[..., permutation, :],
    )
    changed, _ = reader(query, permuted, intent, collect_diagnostics=False)
    torch.testing.assert_close(changed, baseline, rtol=2e-5, atol=2e-6)


def test_p2_coordinate_score_is_grounded_in_post_p1_query() -> None:
    facts = _facts()
    intent, _, field = _completed_effect(facts)
    reader = BoundedFutureEffectReader(hidden=32, horizon=12, basis=3)
    query = torch.randn(2, 12, 3, 32)
    with torch.no_grad():
        reader.action_coordinate.weight.zero_()
        reader.action_coordinate.weight[0, 0] = 1.0
        reader.action_coordinate.weight[1, 1] = 1.0
    _, baseline_metrics = reader(query, field, intent)
    changed_query = query.clone()
    changed_query[..., :2] = 0.0
    _, changed_metrics = reader(changed_query, field, intent)
    assert float(baseline_metrics["grounded_p2_query_coordinate_std"]) > 0.0
    assert not torch.equal(
        baseline_metrics["grounded_p2_query_coordinate_std"],
        changed_metrics["grounded_p2_query_coordinate_std"],
    )


def test_neutral_effect_produces_exact_zero_p2_and_identity_consequence() -> None:
    facts = _facts()
    intent, _, field = _completed_effect(facts)
    neutral = FutureEffectField.neutral_from(field)
    reader = BoundedFutureEffectReader(hidden=32, horizon=12, basis=3)
    query = torch.randn(2, 12, 3, 32)
    read, metrics = reader(query, neutral, intent)
    assert torch.count_nonzero(read) == 0
    for key in (
        "grounded_p2_content_score_abs_max",
        "grounded_p2_intent_score_abs_max",
        "grounded_p2_coordinate_score_abs_max",
    ):
        assert float(metrics[key]) <= 1.0 + 1e-6
    assert bool(((reader.temperatures >= 0.25) & (reader.temperatures <= 4.0)).all())

    organizer = ZeroPreservingConsequenceOrganizer(32)
    factual = torch.randn_like(read)
    consequence, _ = organizer(factual_base=factual, effect_read=read)
    assert torch.count_nonzero(consequence.effect) == 0
    assert torch.count_nonzero(consequence.interaction) == 0
    assert torch.equal(consequence.protected_consequence, factual)


def test_p2_all_invalid_slots_degrade_to_exact_zero_without_nan() -> None:
    facts = _facts()
    intent, _, field = _completed_effect(facts)
    invalid = replace(
        field,
        validity=torch.zeros_like(field.validity),
    )
    reader = BoundedFutureEffectReader(hidden=32, horizon=12, basis=3)
    read, _ = reader(torch.randn(2, 12, 3, 32), invalid, intent)
    assert bool(torch.isfinite(read).all())
    assert torch.count_nonzero(read) == 0


def test_p3_precision_and_temporal_both_depend_on_consequence() -> None:
    facts = _facts()
    intent, _, field = _completed_effect(facts)
    reader = BoundedFutureEffectReader(hidden=32, horizon=12, basis=3)
    query = torch.randn(2, 12, 3, 32)
    read, _ = reader(query, field, intent)
    organizer = ZeroPreservingConsequenceOrganizer(32)
    factual = torch.randn_like(read)
    consequence, _ = organizer(factual_base=factual, effect_read=read)
    compiler = ConsequenceConditionedPolicyPlanCompiler(
        hidden=32,
        horizon=12,
        basis=3,
    )
    p1 = torch.randn_like(read)
    detail = torch.randn_like(read)
    base, _ = compiler(
        p1_delta=p1,
        protected_detail=detail,
        consequence=consequence,
        intent=intent,
        action_query=query,
    )
    changed_consequence = replace(
        consequence,
        protected_consequence=consequence.protected_consequence.roll(
            shifts=1,
            dims=1,
        ),
    )
    changed, _ = compiler(
        p1_delta=p1,
        protected_detail=detail,
        consequence=changed_consequence,
        intent=intent,
        action_query=query,
    )
    assert not torch.allclose(base.precision, changed.precision)
    assert not torch.allclose(base.temporal, changed.temporal)


def test_p2_reports_effect_before_and_after_reliability_masking() -> None:
    facts = _facts()
    intent, _, field = _completed_effect(facts)
    reader = BoundedFutureEffectReader(hidden=32, horizon=12, basis=3)
    query = torch.randn(2, 12, 3, 32)
    baseline, metrics = reader(
        query,
        field,
        intent,
        collect_diagnostics=True,
    )
    reliability_zero = replace(
        field,
        reliability=torch.zeros_like(field.reliability),
    )
    reliability_one = replace(
        field,
        reliability=torch.ones_like(field.reliability),
    )
    zeroed, zero_metrics = reader(
        query,
        reliability_zero,
        intent,
        collect_diagnostics=True,
    )
    bypassed, bypass_metrics = reader(
        query,
        reliability_one,
        intent,
        collect_diagnostics=True,
    )
    torch.testing.assert_close(
        metrics["grounded_p2_effect_value_pre_mask_rms"],
        zero_metrics["grounded_p2_effect_value_pre_mask_rms"],
        rtol=0.0,
        atol=0.0,
    )
    assert float(
        metrics["grounded_p2_effect_value_post_validity_rms"]
    ) >= float(metrics["grounded_p2_effect_value_post_reliability_rms"])
    assert float(
        zero_metrics["grounded_p2_effect_value_post_reliability_rms"]
    ) == 0.0
    assert float(
        zero_metrics["grounded_p2_effect_reliability_valid_mean"]
    ) == 0.0
    torch.testing.assert_close(
        bypass_metrics["grounded_p2_effect_value_pre_mask_rms"],
        metrics["grounded_p2_effect_value_pre_mask_rms"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        bypass_metrics["grounded_p2_effect_value_post_reliability_rms"],
        bypass_metrics["grounded_p2_effect_value_post_validity_rms"],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(bypassed) > 0
    torch.testing.assert_close(zeroed, torch.zeros_like(zeroed))
    assert torch.count_nonzero(baseline) > 0


def test_grounded_graph_bf16_forward_backward_is_finite() -> None:
    torch.manual_seed(113)
    facts = _facts(batch=1, cameras=1, grid=2, slots=2, hidden=16, route=8)
    intent_module = StatelessIntentOrganizer(
        hidden=16,
        state_dim=8,
        action_dim=7,
        fact_dim=8,
        action_horizon=6,
        heads=4,
    )
    world_module = GroundedWorldEffectCompiler(
        hidden=16,
        fact_dim=8,
        route_dim=8,
        heads=4,
    )
    reader = BoundedFutureEffectReader(hidden=16, horizon=6, basis=2)
    consequence_module = ZeroPreservingConsequenceOrganizer(16)
    plan_module = ConsequenceConditionedPolicyPlanCompiler(
        hidden=16,
        horizon=6,
        basis=2,
    )
    world = torch.randn(1, 4, 1, 2, 2, 16)
    proposal = torch.randn(1, 4, 16)
    query = torch.randn(1, 6, 2, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        intent, _ = intent_module(
            facts=facts,
            **_intent_inputs(batch=1, hidden=16),
        )
        working = world_module.initialize(proposal)
        working, _ = world_module.forward_w1(
            world_tokens=world,
            facts=facts,
            intent=intent,
            working=working,
            output_dtype=torch.bfloat16,
        )
        working, _ = world_module.forward_w2(
            world_tokens=world,
            facts=facts,
            intent=intent,
            working=working,
            output_dtype=torch.bfloat16,
        )
        assert working.effect is not None
        effect_read, _ = reader(query, working.effect, intent)
        consequence, _ = consequence_module(
            factual_base=query,
            effect_read=effect_read,
        )
        plan, _ = plan_module(
            p1_delta=query,
            protected_detail=query,
            consequence=consequence,
            intent=intent,
            action_query=query,
        )
    tensors = (
        intent.interval_intents,
        working.effect.semantic_delta,
        effect_read,
        consequence.protected_consequence,
        plan.precision,
        plan.temporal,
    )
    assert all(bool(torch.isfinite(value).all()) for value in tensors)
    loss = torch.stack(
        tuple(value.float().square().mean() for value in tensors)
    ).sum()
    loss.backward()
    for module in (
        intent_module,
        world_module,
        reader,
        consequence_module,
        plan_module,
    ):
        assert any(
            parameter.grad is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        )
