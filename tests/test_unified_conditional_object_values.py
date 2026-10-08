"""S -> P2 conditional values: real-source and ordinary-derivative contracts.

Inputs and weights are synthetic. These are numerical/integration tests, not
proof of learned physical identity, real-data training or CALVIN success.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.model.typed_object_values import (
    CONDITIONAL_OBJECT,
    conditional_typed_values,
)


def test_double_precision_ordinary_jacobian_matches_forward():
    torch.manual_seed(29031)
    x = torch.randn(2, 3, 3, 4, dtype=torch.double, requires_grad=True)
    q = torch.randn(2, 4, 3, 4, dtype=torch.double, requires_grad=True)
    g = torch.rand(2, 4, 3, dtype=torch.double, requires_grad=True)
    valid = torch.ones(2, 3, dtype=torch.bool)

    def fn(a, b, c):
        return conditional_typed_values(a, b, c, valid)

    assert torch.autograd.gradcheck(fn, (x, q, g))
    assert torch.autograd.gradgradcheck(fn, (x, q, g))


def test_source_zero_and_unavailable_nan_stay_zero_in_forward_and_backward():
    x = torch.randn(1, 3, 3, 4)
    x[:, 0] = 0
    x[:, 2] = torch.nan
    x.requires_grad_()
    q = torch.randn(1, 4, 3, 4, requires_grad=True)
    g = torch.full((1, 4, 3), 0.6, requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    value = conditional_typed_values(x, q, g, valid)
    assert torch.isfinite(value).all()
    assert value[:, :, [0, 2]].count_nonzero() == 0
    gradients = torch.autograd.grad(value.square().sum(), (x, q, g))
    assert all(torch.isfinite(v).all() for v in gradients)
    assert gradients[0][:, 2].count_nonzero() == 0
    assert gradients[0][:, 1].count_nonzero() > 0
    assert gradients[1].count_nonzero() > 0


def test_mass_applied_once_and_equal_entropy_identity_swap_changes_value():
    # Two opposite physical source values under the same task value.
    x = torch.tensor([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
    q = torch.tensor([[[[0.5, -0.2]], [[0.2, 0.3]]]])
    value = conditional_typed_values(x, q, torch.ones(1, 2, 1), torch.ones(1, 2, dtype=torch.bool))
    b = torch.tensor([[0.7, 0.2]], requires_grad=True)

    def read(m):
        return torch.einsum("bk,biktr->bitr", m, value)

    assert not torch.allclose(read(b), read(b.flip(-1)))
    torch.testing.assert_close(read(b / 10), read(b) / 10)
    assert read(torch.zeros_like(b)).count_nonzero() == 0
    grad = torch.autograd.grad(read(b).sum(), b)[0]
    torch.testing.assert_close(grad, value.sum((1, 3, 4)))


def test_simultaneous_K_permutation_and_no_batch_partner_coupling():
    torch.manual_seed(29033)
    x = torch.randn(2, 4, 3, 7)
    q = torch.randn(2, 4, 3, 7)
    g = torch.rand(2, 4, 3)
    s = torch.ones(2, 4, dtype=torch.bool)
    y = conditional_typed_values(x, q, g, s)
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(
        conditional_typed_values(x[:, order], q, g, s[:, order]), y[:, :, order]
    )
    torch.testing.assert_close(conditional_typed_values(x[:1], q[:1], g[:1], s[:1]), y[:1])


@pytest.fixture(scope="module")
def actual():
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.qualification import synthetic_batch
    from scripts.check_unified_model_flow import configuration

    torch.set_num_threads(1)
    torch.manual_seed(29034)
    c = configuration("small", "B")
    c = replace(c, top=replace(c.top, typed_object_value_mode=CONDITIONAL_OBJECT))
    c.validate()
    batch, normalizer = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"))
    model = ClearVLAMainlinePolicy(c)
    model.configure_action_normalizer(normalizer)
    model.eval()
    cache, state, _ = model.encode_online(batch.online, geometry_supervision=False)
    return model, cache, state, c, batch


def isolated_dock(dock, mass):
    supported = dock.target_binding.supported
    total = mass.sum(-1, keepdim=True)
    law = TargetBinding(torch.cat((mass, 1 - total), -1).log(), supported)
    return replace(
        dock,
        target_binding=law,
        task_relation=None,
        instruction_change=None,
        instruction_plan_values=None,
        operation_expectation=None,
        annotated_goal=None,
        annotated_goal_values=None,
    )


def test_actual_online_S_exports_conditional_values_and_dock_identity(actual):
    model, cache, state, config, _ = actual
    intent = cache.top.intent
    assert intent.typed_object_value_mode == CONDITIONAL_OBJECT
    assert intent.policy_dock().typed_object_value_mode == CONDITIONAL_OBJECT
    p = intent.permute(torch.tensor([2, 0, 3, 1]))
    assert p.typed_object_value_mode == CONDITIONAL_OBJECT
    torch.testing.assert_close(
        p.typed_relevance_value, intent.typed_relevance_value[:, :, [2, 0, 3, 1]]
    )
    assert model.intent.organizer.typed_object_value_mode == CONDITIONAL_OBJECT
    facts = state.top.facts
    # Only the operation gate's task posterior changes; no pointwise binding
    # is hidden inside a conditional exported value.
    output = model.intent.organizer._typed_relevance(
        public_interval_carrier=intent.public_interval_carrier,
        facts=facts,
        binding=intent.target_binding,
    )
    torch.testing.assert_close(output[1], intent.typed_relevance_value, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("null_fraction", [0.0, 0.5, 0.999, 1.0])
def test_actual_P2_null_support_and_single_measure_for_both_types(actual, null_fraction):
    model, cache, _, _, _ = actual
    dock = cache.top.intent.policy_dock()
    reader = model.policy_compiler.effect_reader
    dynamics = cache.top.predicted_dynamics
    query = torch.randn_like(cache.factual_dock.protected_detail)
    mass = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    full, _ = reader.spatial_select(
        query, dynamics, isolated_dock(dock, mass), collect_diagnostics=False
    )
    value, _ = reader.spatial_select(
        query, dynamics, isolated_dock(dock, mass * (1 - null_fraction)), collect_diagnostics=False
    )
    # Geometry may choose a view inside each K, but cannot reassign K mass.
    torch.testing.assert_close(
        value.selected_target_value,
        full.selected_target_value * (1 - null_fraction),
        rtol=2e-5,
        atol=2e-6,
    )
    if null_fraction == 1:
        assert value.selected_target_value.count_nonzero() == 0
        effect, _ = reader.temporal_terminal(query, value, collect_diagnostics=False)
        assert effect.semantic.count_nonzero() == 0 and effect.geometry.count_nonzero() == 0


def test_no_query_only_value_when_S_has_no_real_target(actual):
    model, cache, state, _, _ = actual
    intent = cache.top.intent
    null = isolated_dock(intent.policy_dock(), torch.zeros(1, 4)).target_binding
    values = model.intent.organizer._typed_relevance(
        public_interval_carrier=intent.public_interval_carrier,
        facts=state.top.facts,
        binding=null,
    )
    assert values[2].count_nonzero() == 0  # S's own consumption obeys null too.


def test_mode_serialization_ABI_and_old_mode_not_silently_relabelled(tmp_path):
    from test_mainline_global_task import abi_for

    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    from scripts.check_unified_model_flow import configuration

    old = load_config("configs/mainline/dinov3_causal_repair_calvin_20261006.json")
    old_top = old.as_dict()["top"]
    assert isinstance(old_top, dict)
    assert "typed_object_value_mode" not in old_top
    c = configuration("small", "A")
    c = replace(c, top=replace(c.top, typed_object_value_mode=CONDITIONAL_OBJECT))
    assert config_from_mapping(c.as_dict()) == c
    _, abi, _ = abi_for(c, tmp_path)
    validate_deployment_abi(abi)
    bad = deepcopy(abi)
    identity = bad["causal_identity"]
    assert isinstance(identity, dict)
    identity.pop("typed_object_values")
    with pytest.raises(ValueError):
        validate_deployment_abi(bad)
    for values in (
        {"typed_object_value_mode": "guess"},
        {"target_binding_mode": "reader_local_v1"},
        {"typed_interval_gradient_mode": "legacy_common_surrogate_v1"},
    ):
        with pytest.raises(ValueError):
            replace(c, top=replace(c.top, **values)).validate()


def test_new_value_migration_does_not_broaden_earlier_contracts():
    from clearvla.mainline.runtime.causal_identity_migration import (
        CAUSAL_IDENTITY_AB_V1,
        CAUSAL_UNIFIED_SOURCE_V1,
        CAUSAL_UNIFIED_VALUES_V1,
        SOURCE_DIGEST,
        allowed_source_paths,
        config_view,
        validate_selection,
    )

    saved = load_config("configs/mainline/dinov3_causal_repair_calvin_20261006.json")
    new = load_config("configs/mainline/unified_source_a_calvin_check.json")
    new = replace(new, top=replace(new.top, typed_object_value_mode=CONDITIONAL_OBJECT))
    for old_mode in (CAUSAL_IDENTITY_AB_V1, CAUSAL_UNIFIED_SOURCE_V1):
        with pytest.raises(ValueError, match="explicit"):
            validate_selection(saved, new, SOURCE_DIGEST, mode=old_mode)
        assert "clearvla/mainline/model/typed_object_values.py" not in allowed_source_paths(
            old_mode
        )
    validate_selection(saved, new, SOURCE_DIGEST, mode=CAUSAL_UNIFIED_VALUES_V1)
    assert "clearvla/mainline/model/typed_object_values.py" in allowed_source_paths(
        CAUSAL_UNIFIED_VALUES_V1
    )
    assert (
        "typed_object_value_mode"
        not in config_view(new.as_dict(), mode=CAUSAL_UNIFIED_VALUES_V1)["top"]
    )


def test_new_mode_does_not_add_parameters_or_change_constructor_rng():
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from scripts.check_unified_model_flow import configuration

    c = configuration("small", "A")
    torch.manual_seed(29037)
    old = ClearVLAMainlinePolicy(c)
    old_rng = torch.get_rng_state()
    torch.manual_seed(29037)
    new = ClearVLAMainlinePolicy(
        replace(c, top=replace(c.top, typed_object_value_mode=CONDITIONAL_OBJECT))
    )
    assert torch.equal(old_rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_complete_model_ordinary_updates_and_two_pass_sampling(variant):
    from scripts.check_unified_model_flow import run

    out = run(
        "small",
        variant,
        updates=2,
        raw_side=32,
        completed_step=1200,
        typed_object_values=CONDITIONAL_OBJECT,
    )
    assert out["typed_object_value_mode"] == CONDITIONAL_OBJECT
    assert not out["real_data_passed"] and not out["behavior_passed"]
    final = out["parameter_ledger"][-1]["entries"]
    assert all(row["gradient_status"] != "missing" for row in final)
    assert any(row["update_l2"] > 0 for row in final if "typed_relevance_queries" in row["name"])


def test_actual_training_parser_admits_only_explicit_new_migration():
    from clearvla.mainline.runtime.causal_identity_migration import CAUSAL_UNIFIED_VALUES_V1
    from clearvla.mainline.train import _parser

    args = _parser().parse_args(
        [
            "--config",
            "configs/mainline/unified_values_a_calvin_check.json",
            "--init-model-contract-migration",
            CAUSAL_UNIFIED_VALUES_V1,
        ]
    )
    assert args.init_model_contract_migration == CAUSAL_UNIFIED_VALUES_V1


def test_old_public_residual_null_bypass_is_reproduced_not_hidden(actual):
    model, cache, _, _, _ = actual
    dock = isolated_dock(cache.top.intent.policy_dock(), torch.zeros(1, 4))
    q = torch.randn_like(cache.factual_dock.protected_detail)
    reader = model.policy_compiler.effect_reader
    old, _ = reader.spatial_select(
        q,
        cache.top.predicted_dynamics,
        replace(dock, typed_object_value_mode="legacy_selected_v1"),
        collect_diagnostics=False,
    )
    new, _ = reader.spatial_select(q, cache.top.predicted_dynamics, dock, collect_diagnostics=False)
    assert old.selected_target_value.count_nonzero() > 0
    assert new.selected_target_value.count_nonzero() == 0
