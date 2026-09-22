"""M6j: production adapter/host values preserve magnitude; not zero whole action.

External observations are synthetic. Neural components, optimizer, sampling and
checkpoint paths are the actual production implementations.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_mainline_operation_expectation import _batch
from test_mainline_operation_expectation import _config as base_config
from test_mainline_state_features import _model_engine
from torch import Tensor, nn

from clearvla.mainline.bottom_evidence import MAGNITUDE_EVIDENCE, bottom_evidence_metadata
from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.model.restored_bottom import _build_decoder_config
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
    EvidenceView,
    EvidenceViewAdapter,
    TimeDomainMMDiTBlock,
)


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, bottom=replace(c.bottom, evidence_value_mode=MAGNITUDE_EVIDENCE))


def _adapter() -> EvidenceViewAdapter:
    return EvidenceViewAdapter(_build_decoder_config(_config())).eval()


def _view_inputs(h: int = 32) -> dict[str, Any]:
    return {
        "trajectory_tokens": torch.zeros(1, 24, h),
        "rollout_tokens": torch.randn(1, 512, h),
        "transition_memory": [torch.randn(1, 512, h), torch.randn(1, 24, h)],
        "event_evidence": torch.randn(1, 24, 3),
        "state_memory": [torch.randn(1, 1, h), torch.randn(1, 1, h)],
        "layer_contracts": [
            {
                "rollout_tokens": torch.randn(1, 512, h),
                "state_tokens": torch.randn(1, 1, h),
                "state_history_tokens": torch.randn(1, 1, h),
            }
        ],
        "intent_memory": {"state": torch.randn(1, 1, h), "executed": torch.randn(1, 1, h)},
    }


def _block() -> TimeDomainMMDiTBlock:
    cfg = _build_decoder_config(_config())
    return TimeDomainMMDiTBlock(replace(cfg, dropout=0.0)).eval()


def _block_inputs() -> tuple[Tensor, Tensor, Tensor]:
    return torch.randn(1, 24, 32), torch.randn(1, 19, 32), torch.randn(1, 32)


def test_config_roundtrip_and_source_closure() -> None:
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "evidence_value_mode" not in cast(dict, base_config().as_dict()["bottom"])
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m6j_calvin.json"
        ).bottom.evidence_value_mode
        == MAGNITUDE_EVIDENCE
    )
    assert _build_decoder_config(c).evidence_value_mode == MAGNITUDE_EVIDENCE
    assert "clearvla/mainline/bottom_evidence.py" in dict(
        active_source_snapshot(Path(__file__).resolve().parents[1]).files
    )


@pytest.mark.parametrize("where", ["experiment", "adapter", "block", "core"])
def test_unknown_semantics_rejected(where: str) -> None:
    c = _config()
    with pytest.raises(ValueError, match="evidence value mode"):
        if where == "experiment":
            replace(c, bottom=replace(c.bottom, evidence_value_mode="guess")).validate()
        else:
            core = replace(_build_decoder_config(c), evidence_value_mode="guess")
            if where == "adapter":
                EvidenceViewAdapter(core)
            elif where == "block":
                TimeDomainMMDiTBlock(core)
            else:
                core.validate()


@pytest.mark.parametrize("kind", ["transition", "event", "state", "rollout", "trajectory"])
def test_zero_source_has_zero_values_but_valid_selector(kind: str) -> None:
    a, args = _adapter(), _view_inputs()
    names = {
        "transition": "transition_memory",
        "event": "event_evidence",
        "state": "state_memory",
        "rollout": "rollout_tokens",
        "trajectory": "trajectory_tokens",
    }
    x = args[names[kind]]
    args[names[kind]] = (
        [torch.zeros_like(t) for t in x] if isinstance(x, list) else torch.zeros_like(x)
    )
    v = a(**args)
    lo, hi = v.ranges[kind]
    assert torch.count_nonzero(v.value_tokens[:, lo:hi]) == 0
    assert torch.isfinite(v.tokens).all() and v.tokens[:, lo:hi].abs().sum() > 0
    assert v.value_mode == MAGNITUDE_EVIDENCE


@pytest.mark.parametrize("scale", [0.0, 1e-7, 0.125, 1.0, 3.0])
def test_transition_value_scales_before_any_key_based_read(scale: float) -> None:
    torch.manual_seed(632)
    a, args = _adapter(), _view_inputs()
    # A diagonal test map makes homogeneity exact in FP32, independent of
    # cancellation/roundoff in two separately ordered dense matrix products.
    with torch.no_grad():
        cast(nn.Linear, a.source_proj["transition"]).weight.copy_(torch.eye(32))
    ref = a(**args)
    lo, hi = ref.ranges["transition"]
    args["transition_memory"] = [x * scale for x in args["transition_memory"]]
    actual = a(**args)
    torch.testing.assert_close(
        actual.value_tokens[:, lo:hi], ref.value_tokens[:, lo:hi] * scale, rtol=1e-5, atol=1e-8
    )


def test_type_and_depth_selector_embeddings_never_become_value_content() -> None:
    a, args = _adapter(), _view_inputs()
    ref = a(**args)
    with torch.no_grad():
        a.bank.type_embed.add_(torch.randn_like(a.bank.type_embed) * 5)
        a.layer_depth_embed.add_(torch.randn_like(a.layer_depth_embed) * 5)
    new = a(**args)
    torch.testing.assert_close(new.value_tokens, ref.value_tokens, rtol=0, atol=0)
    assert not torch.equal(new.tokens, ref.tokens) and new.ranges == ref.ranges
    torch.testing.assert_close(new.key_bias, ref.key_bias, rtol=0, atol=0)


def test_transition_value_gradients_do_not_train_selector_type_embeddings() -> None:
    a, args = _adapter(), _view_inputs()
    v = a(**args)
    lo, hi = v.ranges["transition"]
    v.value_tokens[:, lo:hi].square().sum().backward()
    assert a.bank.type_embed.grad is None
    weight = cast(nn.Linear, a.source_proj["transition"]).weight
    assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_neutral_trajectory_has_no_dead_value_parameters() -> None:
    a = _adapter()
    assert isinstance(a.source_proj["trajectory"], nn.Identity)
    assert not list(a.source_proj["trajectory"].parameters())
    for name in ("transition", "rollout", "state"):
        assert isinstance(a.source_proj[name], nn.Linear) and a.source_proj[name].bias is None


@pytest.mark.parametrize("bf16", [False, True])
def test_zero_values_have_zero_cross_write_and_no_selector_gradient(bf16: bool) -> None:
    b = _block()
    action, keys, global_c = _block_inputs()
    keys.requires_grad_(True)
    values = torch.zeros_like(keys, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out, metrics = b(action, keys, global_c, evidence_value_tokens=values)
        baseline, _ = b(action, keys, global_c, evidence_value_tokens=values, evidence_scale=0.0)
    assert metrics["evidence_update_norm"] == 0
    torch.testing.assert_close(out, baseline, rtol=0, atol=0)
    out.square().sum().backward()
    assert keys.grad is not None and torch.count_nonzero(keys.grad) == 0
    for layer in (b.evidence_query, b.evidence_key):
        for p in layer.parameters():
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0
    assert (
        values.grad is not None
        and torch.isfinite(values.grad).all()
        and values.grad.abs().sum() > 0
    )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("scale", [1e-7, 1e-5, 1e-3])
def test_small_value_is_not_promoted_to_unit_update(scale: float) -> None:
    torch.manual_seed(633)
    b = _block()
    action, keys, global_c = _block_inputs()
    values = torch.randn_like(keys)
    _, ref = b(action, keys, global_c, evidence_value_tokens=values * 0.01)
    _, small = b(action, keys, global_c, evidence_value_tokens=values * scale)
    torch.testing.assert_close(
        small["evidence_update_norm"],
        ref["evidence_update_norm"] * (scale / 0.01),
        rtol=2e-5,
        atol=1e-12,
    )


def test_constant_nonzero_feature_survives_value_projection() -> None:
    b = _block()
    action, keys, global_c = _block_inputs()
    seen: list[Tensor] = []

    def capture(_module: nn.Module, _args: object, out: object) -> None:
        assert isinstance(out, Tensor)
        seen.append(out)

    handle = b.evidence_value.register_forward_hook(capture)
    b(action, keys, global_c, evidence_value_tokens=torch.ones_like(keys) * 0.02)
    handle.remove()
    assert seen[0].abs().sum() > 0


def test_large_write_is_bounded_without_creating_minimum_amplitude() -> None:
    b = _block()
    action, keys, global_c = _block_inputs()
    _, m = b(action, keys, global_c, evidence_value_tokens=torch.randn_like(keys) * 1e5)
    assert torch.isfinite(m["evidence_update_norm"]) and m["evidence_update_norm"] <= 0.25 * (
        32**0.5
    )


def test_reader_parameter_inventory_includes_new_K_V_once() -> None:
    b = _block()
    ids = [id(p) for p in b.evidence_reader_parameters()]
    assert len(ids) == len(set(ids)) and not hasattr(b, "evidence_kv")
    for mod in (b.evidence_key, b.evidence_value, b.evidence_out):
        assert all(ids.count(id(p)) == 1 for p in mod.parameters())


@pytest.mark.parametrize("bf16", [False, True])
def test_production_optimizer_and_row_zero_gradient(bf16: bool) -> None:
    torch.manual_seed(635)
    m, engine = _model_engine(_config())
    batch = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        engine.train_step(batch, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, _, _ = m.encode_online(batch.online)
        v = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
        loss = v.bottom.physical_velocity[:, 0].float().square().mean()
    loss.backward()
    decoder = m.execution_bottom.decoder
    assert isinstance(decoder, EvidenceLatentMMDiTActionDecoder)
    modules: list[nn.Module] = [
        decoder.evidence_adapter.source_proj[n] for n in ("transition", "rollout", "state")
    ]
    modules += [decoder.evidence_adapter.event_proj]
    for block in decoder.blocks:
        assert isinstance(block, TimeDomainMMDiTBlock)
        modules.extend((block.evidence_key, block.evidence_value, block.evidence_out))
    optimizer_ids = [id(p) for group in engine.optimizer.param_groups for p in group["params"]]
    for mod in modules:
        for name, p in mod.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
                name
            )
            assert optimizer_ids.count(id(p)) == 1


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostics_do_not_change_block_value_gradient_or_RNG(bf16: bool) -> None:
    torch.manual_seed(636)
    b = _block()
    action, keys, global_c = _block_inputs()
    values = torch.randn_like(keys) * 0.03
    results = []
    for collect in (False, True):
        b.zero_grad(set_to_none=True)
        rng = torch.random.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            out, metrics = b(action, keys, global_c, evidence_value_tokens=values)
            if collect:
                assert all(not x.requires_grad for x in metrics.values())
            grads = torch.autograd.grad(out.float().square().mean(), tuple(b.parameters()))
        assert torch.equal(rng, torch.random.get_rng_state())
        results.append((out, grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, c in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(a, c, rtol=0, atol=0)


def test_real_sampler_repeatable_without_new_mutable_state() -> None:
    torch.manual_seed(638)
    c = _config()
    m, e = _model_engine(c)
    batch = _batch()
    e.train_step(batch)
    m.eval()
    before = {n: v.clone() for n, v in m.named_buffers()}
    a = sample_action(m, batch.online, c, generator=torch.Generator().manual_seed(9))
    b = sample_action(m, batch.online, c, generator=torch.Generator().manual_seed(9))
    assert torch.isfinite(a.action).all()
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
    for n, v in m.named_buffers():
        torch.testing.assert_close(v, before[n], rtol=0, atol=0)


def test_checkpoint_reload_and_reject_wrong_value_ABI(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert abi["bottom_evidence"] == bottom_evidence_metadata()
        for kind in ("missing", "value", "zero"):
            bad = copy.deepcopy(abi)
            if kind == "missing":
                del bad["bottom_evidence"]
            else:
                cast(dict[str, object], bad["bottom_evidence"])[kind] = "wrong"
            with pytest.raises(ValueError, match="bottom evidence"):
                validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_random_transition_projection_is_exactly_the_direct_unscaled_linear_map() -> None:
    a, args = _adapter(), _view_inputs()
    view = a(**args)
    lo, hi = view.ranges["transition"]
    expected = a.source_proj["transition"](torch.cat(args["transition_memory"], dim=1))
    torch.testing.assert_close(view.value_tokens[:, lo:hi], expected, rtol=0, atol=0)


def test_host_rejects_implicit_selector_as_value_fallback() -> None:
    b = _block()
    action, keys, g = _block_inputs()
    with pytest.raises(ValueError, match="explicit value stream"):
        b(action, keys, g)
    assert b.evidence_key.bias is None


def test_real_CT_neutral_intervention_stays_zero_through_bottom_bank() -> None:
    torch.manual_seed(639)
    c = _config()
    m, _ = _model_engine(c)
    m.eval()
    batch = _batch()
    captured: list[EvidenceView] = []

    def capture(_m: nn.Module, _args: object, out: object) -> None:
        assert isinstance(out, EvidenceView)
        captured.append(out)

    decoder = m.execution_bottom.decoder
    assert isinstance(decoder, EvidenceLatentMMDiTActionDecoder)
    handle = decoder.evidence_adapter.register_forward_hook(capture)
    m.transition.set_eval_intervention("delta_neutral")
    try:
        with torch.no_grad():
            cache, _, _ = m.encode_online(batch.online)
            v = m.velocity(
                cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4])
            )
        assert torch.isfinite(v.bottom.physical_velocity).all()
        assert captured
        for view in captured:
            for name in ("transition", "trajectory"):
                lo, hi = view.ranges[name]
                assert torch.count_nonzero(view.value_tokens[:, lo:hi]) == 0
        assert (
            v.bottom.evidence_tokens is not None
            and torch.count_nonzero(v.bottom.evidence_tokens) == 0
        )
    finally:
        handle.remove()
        m.transition.clear_eval_intervention()


def test_decoder_rejects_mismatched_value_semantics_before_host_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m, _ = _model_engine(_config())
    m.eval()
    b = _batch()
    decoder = m.execution_bottom.decoder
    assert isinstance(decoder, EvidenceLatentMMDiTActionDecoder)
    adapter = decoder.evidence_adapter
    original = adapter.forward

    def wrong(*args, **kwargs):
        return replace(original(*args, **kwargs), value_mode="normalized_legacy_v1")

    monkeypatch.setattr(adapter, "forward", wrong)
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
        with pytest.raises(ValueError, match="value semantics"):
            m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))


@pytest.mark.parametrize("bf16", [False, True])
def test_full_velocity_and_reader_gradients_are_logging_invariant(bf16: bool) -> None:
    torch.manual_seed(640)
    m, e = _model_engine(_config())
    batch = _batch()
    e.train_step(batch)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(batch.online)
    decoder = m.execution_bottom.decoder
    assert isinstance(decoder, EvidenceLatentMMDiTActionDecoder)
    block = decoder.blocks[0]
    assert isinstance(block, TimeDomainMMDiTBlock)
    params = block.evidence_reader_parameters()
    field = torch.randn(1, 24, 18)
    results = []
    for collect in (False, True):
        before = torch.random.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            v = m.velocity(
                cache,
                noisy_action_field=field,
                time=torch.tensor([0.4]),
                collect_diagnostics=collect,
            )
            out = v.bottom.physical_velocity
            grads = torch.autograd.grad(out.float().square().mean(), params)
        assert torch.equal(before, torch.random.get_rng_state())
        results.append((out, grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, b in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
