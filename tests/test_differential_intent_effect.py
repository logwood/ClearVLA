from __future__ import annotations

from dataclasses import replace

import torch

from clearvla.policy.differential_intent_effect import (
    ConsequencePlanOrganizer,
    DifferentialFutureEffectReader,
    DifferentialPolicyPlanCompiler,
    DifferentialStatelessIntentController,
    DifferentialWindowEffectBank,
    DifferentialWindowRouteCompiler,
    IntentStateBank,
)


def _intent_inputs(
    *,
    batch: int = 2,
    hidden: int = 64,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "goal_tokens": torch.randn(batch, 7, hidden, generator=generator),
        "state_history_tokens": torch.randn(
            batch,
            5,
            8,
            generator=generator,
        ),
        "history_tokens": torch.randn(
            batch,
            4,
            8,
            generator=generator,
        ),
        "grounding_tokens": torch.randn(
            batch,
            11,
            hidden,
            generator=generator,
        ),
    }


def _controller(hidden: int = 64) -> DifferentialStatelessIntentController:
    return DifferentialStatelessIntentController(
        hidden,
        4,
        4,
        24,
        state_dim=8,
        action_dim=8,
        heads=4,
    )


def test_differential_intent_uses_one_canonical_state_bank() -> None:
    module = _controller()
    bank, metrics = module(**_intent_inputs())
    assert isinstance(bank, IntentStateBank)
    assert bank.intent_state.shape == (2, 4, 64)
    assert bank.window_view.tokens.shape == (2, 3, 64)
    assert bank.window_view.program_attention.shape == (2, 3, 4)
    assert bank.temporal_control.shape == (2, 24, 64)
    assert torch.allclose(
        bank.window_view.program_attention.sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )
    assert metrics["flow_jepa_intent_program_attention_entropy"].isfinite()

    forbidden = ("progress_head", "window_score", "phase_output")
    names = tuple(name for name, _ in module.named_parameters())
    assert not any(token in name for token in forbidden for name in names)


def test_differential_intent_predictive_and_action_reads_train_same_state() -> None:
    module = _controller()
    bank, _ = module(**_intent_inputs())
    loss = (
        bank.window_view.predictive_effect.square().mean()
        + bank.temporal_control.square().mean()
    )
    loss.backward()

    assert module.goal_block.attention.in_proj_weight.grad is not None
    assert module.history_blocks[0].attention.in_proj_weight.grad is not None
    assert module.history_blocks[1].attention.in_proj_weight.grad is not None
    assert module.grounding_to_program.attention.in_proj_weight.grad is not None
    assert module.window_read.in_proj_weight.grad is not None


def test_differential_intent_is_observable_and_stateless() -> None:
    module = _controller().eval()
    inputs = _intent_inputs()
    with torch.no_grad():
        first, _ = module(**inputs)
        repeated, _ = module(**inputs)
        changed_inputs = dict(inputs)
        changed_inputs["history_tokens"] = inputs["history_tokens"].roll(
            shifts=1,
            dims=1,
        )
        changed, _ = module(**changed_inputs)

    assert torch.equal(first.intent_state, repeated.intent_state)
    assert torch.equal(first.window_view.tokens, repeated.window_view.tokens)
    assert not torch.allclose(
        first.window_view.tokens,
        changed.window_view.tokens,
    )


def _completed_effect_graph() -> tuple[
    DifferentialStatelessIntentController,
    IntentStateBank,
    DifferentialWindowRouteCompiler,
    DifferentialWindowEffectBank,
]:
    intent_module = _controller()
    intent, _ = intent_module(**_intent_inputs())
    compiler = DifferentialWindowRouteCompiler(
        route_dim=32,
        hidden=64,
        heads=4,
        slots_per_cell=4,
    )
    generator = torch.Generator().manual_seed(23)
    selected_route = torch.randn(
        2,
        4,
        2,
        3,
        3,
        32,
        generator=generator,
    )
    current_keys = torch.randn(
        2,
        2,
        3,
        3,
        4,
        32,
        generator=generator,
    )
    w1, route_state, _ = compiler.forward_w1(
        selected_route,
        current_keys,
        intent.window_view,
        output_dtype=selected_route.dtype,
    )
    completed, _, _ = compiler.forward_w2(
        selected_route,
        current_keys,
        intent.window_view,
        w1_bank=w1,
        w1_route_state=route_state,
        output_dtype=selected_route.dtype,
    )
    assert torch.equal(completed.semantic_delta[:, :2], w1.semantic_delta)
    assert torch.equal(completed.current_reference, w1.current_reference)
    return intent_module, intent, compiler, completed


def test_differential_w_preserves_w1_ownership_and_uses_one_current_fact() -> None:
    _, intent, _, bank = _completed_effect_graph()
    bank.validate(expected_slots=3)
    assert bank.current_reference.ndim == 6
    assert bank.semantic_delta.ndim == 7
    assert bank.successor_content.shape == bank.semantic_delta.shape
    assert intent.window_view.tokens.shape[1] == bank.slots


def test_differential_w_reads_the_canonical_window_view_not_parallel_heads() -> None:
    intent_module = _controller().eval()
    intent, _ = intent_module(**_intent_inputs())
    compiler = DifferentialWindowRouteCompiler(
        route_dim=32,
        hidden=64,
        heads=4,
        slots_per_cell=4,
    ).eval()
    generator = torch.Generator().manual_seed(29)
    selected_route = torch.randn(
        2,
        4,
        2,
        3,
        3,
        32,
        generator=generator,
    )
    current_keys = torch.randn(
        2,
        2,
        3,
        3,
        4,
        32,
        generator=generator,
    )
    changed_view = replace(
        intent.window_view,
        tokens=intent.window_view.tokens.roll(shifts=1, dims=1),
    )
    with torch.no_grad():
        base, _, _ = compiler.forward_w1(
            selected_route,
            current_keys,
            intent.window_view,
            output_dtype=selected_route.dtype,
        )
        changed, _, _ = compiler.forward_w1(
            selected_route,
            current_keys,
            changed_view,
            output_dtype=selected_route.dtype,
        )
    assert not torch.allclose(base.semantic_delta, changed.semantic_delta)


def test_differential_p2_effect_read_is_exact_zero_for_zero_effect() -> None:
    _, intent, _, bank = _completed_effect_graph()
    zero_bank = DifferentialWindowEffectBank(
        current_reference=bank.current_reference,
        semantic_delta=torch.zeros_like(bank.semantic_delta),
        transport_mean=torch.zeros_like(bank.transport_mean),
        transport_covariance=torch.zeros_like(bank.transport_covariance),
        persistence=torch.zeros_like(bank.persistence),
        visibility=torch.zeros_like(bank.visibility),
        uncertainty=torch.zeros_like(bank.uncertainty),
        slot_valid=torch.ones_like(bank.slot_valid),
        slot_names=bank.slot_names,
    )
    reader = DifferentialFutureEffectReader(
        hidden=64,
        horizon=6,
        basis=4,
    )
    read, metrics = reader(
        torch.randn(2, 6, 4, 64),
        zero_bank,
        intent.window_view,
    )
    assert torch.count_nonzero(read) == 0
    assert metrics["flow_jepa_p2_effect_coordinate_score_rms"].isfinite()


def test_differential_p2_learns_effect_identity_without_fixed_slot_mass() -> None:
    controller = _controller()
    intent, _ = controller(**_intent_inputs())
    common_token = intent.window_view.tokens[:, :1].expand(-1, 3, -1)
    common_coordinate = intent.window_view.support_coordinates[:, :1].expand(
        -1,
        3,
        -1,
    )
    controlled_intent = replace(
        intent.window_view,
        tokens=common_token,
        support_coordinates=common_coordinate,
    )
    batch, slots, cameras, rows, columns, owners, hidden = (
        2,
        3,
        2,
        2,
        2,
        2,
        64,
    )
    generator = torch.Generator().manual_seed(37)
    base_effect = torch.randn(
        batch,
        1,
        cameras,
        rows,
        columns,
        owners,
        hidden,
        generator=generator,
    )
    distinct_effect = torch.cat(
        (
            base_effect,
            0.5 * base_effect.roll(shifts=1, dims=-1),
            -0.75 * base_effect,
        ),
        dim=1,
    )
    common_geometry = torch.zeros(
        batch,
        slots,
        cameras,
        rows,
        columns,
        owners,
        1,
    )

    def bank(semantic_delta: torch.Tensor) -> DifferentialWindowEffectBank:
        return DifferentialWindowEffectBank(
            current_reference=torch.zeros(
                batch,
                cameras,
                rows,
                columns,
                owners,
                hidden,
            ),
            semantic_delta=semantic_delta,
            transport_mean=torch.zeros(
                batch,
                slots,
                cameras,
                rows,
                columns,
                owners,
                2,
            ),
            transport_covariance=torch.zeros(
                batch,
                slots,
                cameras,
                rows,
                columns,
                owners,
                3,
            ),
            persistence=common_geometry,
            visibility=common_geometry,
            uncertainty=common_geometry,
            slot_valid=torch.ones_like(common_geometry),
            slot_names=("near", "mid", "late"),
        )

    reader = DifferentialFutureEffectReader(
        hidden=hidden,
        horizon=6,
        basis=2,
    ).eval()
    query = torch.randn(batch, 6, 2, hidden, generator=generator)
    _, distinct_metrics = reader(
        query,
        bank(distinct_effect),
        controlled_intent,
    )
    distinct_mass = torch.stack(
        tuple(
            distinct_metrics[f"flow_jepa_p2_effect_{name}_mass"]
            for name in ("near", "mid", "late")
        )
    )
    assert float(distinct_mass.std(unbiased=False)) > 1e-5

    identical_effect = base_effect.expand(-1, slots, -1, -1, -1, -1, -1)
    _, identical_metrics = reader(
        query,
        bank(identical_effect),
        controlled_intent,
    )
    identical_mass = torch.stack(
        tuple(
            identical_metrics[f"flow_jepa_p2_effect_{name}_mass"]
            for name in ("near", "mid", "late")
        )
    )
    torch.testing.assert_close(
        identical_mass,
        torch.full_like(identical_mass, 1.0 / 3.0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_differential_effect_and_intent_reach_protected_plan_base() -> None:
    intent_module, intent, w_compiler, bank = _completed_effect_graph()
    reader = DifferentialFutureEffectReader(
        hidden=64,
        horizon=6,
        basis=4,
    )
    query = torch.randn(2, 6, 4, 64)
    effect_read, _ = reader(query, bank, intent.window_view)
    organizer = ConsequencePlanOrganizer(64)
    factual = torch.randn(2, 6, 4, 64)
    p1_delta = torch.randn(2, 6, 4, 64)
    p2_delta = torch.randn(2, 6, 4, 64)
    consequence, _ = organizer(
        factual_base=factual,
        effect_read=effect_read,
        p2_delta=p2_delta,
    )
    plan_compiler = DifferentialPolicyPlanCompiler(
        hidden=64,
        horizon=6,
        basis=4,
    )
    # The controller was built for horizon 24; create the action-time view that
    # the small P test owns without altering the three canonical window tokens.
    intent = IntentStateBank(
        **{
            **intent.__dict__,
            "temporal_control": torch.nn.functional.interpolate(
                intent.window_view.tokens.float().transpose(1, 2),
                size=6,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2),
        }
    )
    plan, _ = plan_compiler(
        p1_delta=p1_delta,
        protected_detail=factual,
        consequence=consequence,
        intent=intent,
    )
    role_bank = plan.as_policy_role_bank(source_depth=7)
    assert role_bank.values.shape == (2, 2, 6, 4, 64)
    assert plan.source_names == ("p3_precision", "p3_temporal")
    assert not hasattr(plan, "effect")

    loss = (
        plan.protected_base.square().mean()
        + plan.precision.square().mean()
        + plan.temporal.square().mean()
    )
    loss.backward()
    assert intent_module.window_read.in_proj_weight.grad is not None
    assert w_compiler.effect_semantic[-1].weight.grad is not None
    assert reader.effect_value[-1].weight.grad is not None
    assert organizer.organizer[-1].weight.grad is not None


def test_differential_graph_bf16_has_bounded_finite_large_input_jacobian() -> None:
    torch.manual_seed(43)
    hidden = 32
    controller = _controller(hidden)
    inputs = {
        name: value * 1_000.0
        for name, value in _intent_inputs(hidden=hidden).items()
    }
    compiler = DifferentialWindowRouteCompiler(
        route_dim=16,
        hidden=hidden,
        heads=4,
        slots_per_cell=2,
    )
    reader = DifferentialFutureEffectReader(
        hidden=hidden,
        horizon=24,
        basis=2,
    )
    organizer = ConsequencePlanOrganizer(hidden)
    plan_compiler = DifferentialPolicyPlanCompiler(
        hidden=hidden,
        horizon=24,
        basis=2,
    )
    selected_route = torch.randn(2, 4, 2, 2, 2, 16) * 1_000.0
    current_keys = torch.randn(2, 2, 2, 2, 2, 16) * 1_000.0
    query = torch.randn(2, 24, 2, hidden) * 1_000.0
    factual = torch.randn_like(query) * 1_000.0
    p1_delta = torch.randn_like(query) * 1_000.0
    p2_delta = torch.randn_like(query) * 1_000.0

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        intent, _ = controller(**inputs)
        w1, route_state, _ = compiler.forward_w1(
            selected_route,
            current_keys,
            intent.window_view,
            output_dtype=torch.bfloat16,
        )
        effect, _, _ = compiler.forward_w2(
            selected_route,
            current_keys,
            intent.window_view,
            w1_bank=w1,
            w1_route_state=route_state,
            output_dtype=torch.bfloat16,
        )
        effect_read, _ = reader(query, effect, intent.window_view)
        consequence, _ = organizer(
            factual_base=factual,
            effect_read=effect_read,
            p2_delta=p2_delta,
        )
        plan, _ = plan_compiler(
            p1_delta=p1_delta,
            protected_detail=factual,
            consequence=consequence,
            intent=intent,
        )

    tensors = (
        intent.intent_state,
        intent.window_view.tokens,
        effect.semantic_delta,
        effect.transport_mean,
        effect.transport_covariance,
        effect_read,
        consequence.organized_delta,
        plan.protected_base,
        plan.precision,
        plan.temporal,
    )
    assert all(bool(torch.isfinite(value).all()) for value in tensors)
    assert float(
        effect.semantic_delta.detach().float().square().mean(dim=-1).sqrt().amax()
    ) <= 0.501
    assert float(
        consequence.organized_delta.detach().float().square().mean(dim=-1).sqrt().amax()
    ) <= 0.251
    assert float(
        plan.precision.detach().float().square().mean(dim=-1).sqrt().amax()
    ) <= 0.351
    assert float(
        plan.temporal.detach().float().square().mean(dim=-1).sqrt().amax()
    ) <= 0.351

    loss = (
        intent.window_view.predictive_effect.float().square().mean()
        + effect.semantic_delta.float().square().mean()
        + effect_read.float().square().mean()
        + consequence.organized_delta.float().square().mean()
        + plan.precision.float().square().mean()
        + plan.temporal.float().square().mean()
        + plan.execution_terminal.probability.float().mean()
    )
    loss.backward()
    for module in (
        controller,
        compiler,
        reader,
        organizer,
        plan_compiler,
    ):
        for parameter in module.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
