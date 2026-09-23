"""Real native execution controller boundaries, not robot actuator guarantees."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine
from test_mainline_typed_transition import _config as base_config
from torch import Tensor

from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.controller_values import RAW_CONTROLLER_VALUES, controller_value_metadata
from clearvla.mainline.model.restored_bottom import _build_decoder_config
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.v120_core.controller import EvidenceExecutionController


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, bottom=replace(c.bottom, controller_value_mode=RAW_CONTROLLER_VALUES))


def _controller() -> EvidenceExecutionController:
    return EvidenceExecutionController(_build_decoder_config(_config()), block_count=3).eval()


def _inputs() -> dict[str, Tensor]:
    return dict(
        global_condition=torch.randn(1, 32),
        time_context=torch.randn(1, 32),
        evidence_tokens=torch.randn(1, 7, 32),
        evidence_value_tokens=torch.randn(1, 7, 32),
        action_tokens=torch.randn(1, 24, 32),
        feedback=torch.randn(1, 24, 32),
    )


def test_config_and_components() -> None:
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    defaults = base_config().as_dict()["bottom"]
    assert isinstance(defaults, dict)
    assert "controller_value_mode" not in defaults
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m6l_calvin.json"
        ).bottom.controller_value_mode
        == RAW_CONTROLLER_VALUES
    )
    m = _controller()
    assert m.preserve_values and m.value_reader.preserve_values
    for a in (
        m.value_reader.memory_attention,
        m.value_reader.action_attention,
        m.value_reader.temporal_attention,
    ):
        assert a.in_proj_bias is None and a.out_proj.bias is None


@pytest.mark.parametrize("mode", ["guess", "normalized_evidence"])
def test_wrong_config_rejected(mode: str) -> None:
    c = _config()
    b = (
        replace(c.bottom, controller_value_mode="guess")
        if mode == "guess"
        else replace(c.bottom, evidence_value_mode="normalized_legacy_v1")
    )
    with pytest.raises(ValueError):
        replace(c, bottom=b).validate()


@pytest.mark.parametrize("owner", ["controller", "reader", "lanes"])
def test_missing_values_fail_closed(owner: str) -> None:
    m = _controller()
    kw = _inputs()
    del kw["evidence_value_tokens"]
    with pytest.raises(ValueError, match="explicit evidence values"):
        if owner == "lanes":
            m._source_lanes(**kw)
        elif owner == "controller":
            m(state=None, block_index=0, **kw)
        else:
            del kw["feedback"]
            m.value_reader(state=m.initial_state(kw["global_condition"]), block_index=0, **kw)


@pytest.mark.parametrize("scale", [0.0, 1e-4, 0.1, 2.0])
def test_source_values_retain_scale_without_changing_keys(scale: float) -> None:
    m = _controller()
    kw = _inputs()
    kw["global_condition"].zero_()
    kw["time_context"].zero_()
    k, v = m._source_lanes(**kw)
    k2, v2 = m._source_lanes(**{**kw, "evidence_value_tokens": kw["evidence_value_tokens"] * scale})
    torch.testing.assert_close(k2, k, rtol=0, atol=0)
    torch.testing.assert_close(v2, v * scale, rtol=2e-6, atol=1e-7)
    assert torch.count_nonzero(v[:, 9:]) == 0


def test_selector_roles_reach_candidate_keys_not_values() -> None:
    torch.manual_seed(912)
    m = _controller()
    reader = m.value_reader
    kw = _inputs()
    del kw["feedback"]
    torch.nn.init.normal_(reader.value_head.weight, std=0.1)
    records = []
    handle = reader.memory_attention.register_forward_pre_hook(
        lambda module, args: records.append(tuple(x.detach().clone() for x in args))
    )
    try:
        state = m.initial_state(kw["global_condition"])
        a = reader(state=state, block_index=0, **kw)
        b = reader(
            state=state, block_index=0, **{**kw, "evidence_tokens": kw["evidence_tokens"].flip(1)}
        )
    finally:
        handle.remove()
    assert not torch.equal(records[0][1], records[1][1])
    torch.testing.assert_close(records[0][2], records[1][2], rtol=0, atol=0)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("bf16", [False, True])
def test_real_controller_reader_gradients_and_no_state_mutation(bf16: bool) -> None:
    torch.manual_seed(913)
    m = _controller()
    kw = _inputs()
    torch.nn.init.normal_(m.capacity_head.weight, std=0.05)
    torch.nn.init.normal_(m.value_reader.value_head.weight, std=0.05)
    kw["evidence_value_tokens"].requires_grad_()
    kw["evidence_tokens"].requires_grad_()
    before = {n: x.clone() for n, x in m.named_buffers()}
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out = m(state=None, block_index=0, **kw)
        val = m.predict_execution_value(
            state=out.state, block_index=0, **{n: x for n, x in kw.items() if n != "feedback"}
        )
        (val.square().sum() + out.capacity_ratios.sum()).backward()
    for name in ("evidence_value_tokens", "evidence_tokens"):
        g = kw[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    for n, x in m.named_buffers():
        torch.testing.assert_close(x, before[n], rtol=0, atol=0)


def test_production_optimizer_and_sampling() -> None:
    torch.manual_seed(914)
    c = _config()
    m, engine = _model_engine(c)
    b = _batch()
    out = engine.train_step(b, collect_diagnostics=True)
    assert torch.isfinite(out.loss)
    m.eval()
    a = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(1))
    again = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a.action, again.action, rtol=0, atol=0)
    assert torch.isfinite(a.action).all()


def test_checkpoint_and_ABI(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import test_mainline_target_binding as original

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    build = original.build_deployment_abi

    def checked(*args, **kwargs):
        abi = build(*args, **kwargs)
        assert abi["controller_values"] == controller_value_metadata()
        bad = copy.deepcopy(abi)
        del bad["controller_values"]
        with pytest.raises(ValueError, match="controller value"):
            validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(original, "_config", _config)
    monkeypatch.setattr(original, "_batch", _batch)
    monkeypatch.setattr(original, "build_deployment_abi", checked)
    original.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)
