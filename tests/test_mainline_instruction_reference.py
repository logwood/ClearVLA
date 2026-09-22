"""Causal instruction anchor: actual dataset, S, optimizer and online adapter.

Only the external image/DINO/T5 transport is synthetic. Scene/task evaluator
state, physical object IDs, future observations and scripted phases are absent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_mainline_entity_history import _batch as unanchored_batch
from test_mainline_state_features import _data, _model_engine
from test_mainline_target_binding import _config as m5a_config
from test_mainline_timed_history import _ImageStore
from torch.utils.data import default_collate

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.instruction_reference import (
    INSTRUCTION_START_REFERENCE,
    InstructionReference,
)
from clearvla.mainline.model.instruction_progress import InstructionReferenceRead
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import validate_finite_training_batch
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory


def _config():
    base = m5a_config()
    return replace(
        base, top=replace(base.top, instruction_reference_mode=INSTRUCTION_START_REFERENCE)
    )


def _batch(center=16):
    b = unanchored_batch(center)
    o = b.online
    d = o.observation.dino_history[:, -1]
    ref = InstructionReference(
        d.clone()
        if center == 0
        else torch.randn(d.shape, generator=torch.Generator().manual_seed(1471)),
        o.history.state.clone() if center == 0 else o.history.state.clone() * 0.7,
        torch.ones(d.shape[:-1], dtype=torch.bool),
        torch.tensor([center]),
    )
    return replace(b, online=replace(o, instruction_reference=ref))


class _Tokens:
    def __init__(self):
        self.keys = None

    def load_batch(self, keys):
        self.keys = keys.clone()
        return keys[:, 1, None, None, None].expand(-1, 2, 64, 16).float().contiguous()


def _dataset(start=3):
    old = _data()
    episode = replace(
        old.episodes[0],
        source_start=1000 + start,
        context_start=1000,
        valid_center_start=start,
        instruction="push the block",
    )
    return ObservedStateWindowDataset(
        [episode],
        [0],
        image_store=cast(Any, _ImageStore()),
        camera_names=old.camera_names,
        action_normalizer=old.action_normalizer,
        state_normalizer=old.state_normalizer,
        config=replace(old.config, instruction_reference_mode=INSTRUCTION_START_REFERENCE),
    )


def test_mode_is_explicit_and_old_config_identity_unchanged():
    base, cfg = m5a_config(), _config()
    cfg.validate()
    top_mapping = base.as_dict()["top"]
    assert isinstance(top_mapping, dict)
    assert "instruction_reference_mode" not in top_mapping
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m5b_calvin.json"
        ).top.instruction_reference_mode
        == INSTRUCTION_START_REFERENCE
    )
    for changes in (
        {"instruction_reference_mode": "guess"},
        {"target_binding_mode": "reader_local_v1"},
    ):
        with pytest.raises(ValueError):
            replace(cfg, top=replace(cfg.top, **changes)).validate()


@pytest.mark.parametrize(
    "kind",
    ["missing", "state_shape", "dino_shape", "mask_type", "age_type", "future", "unexpected"],
)
def test_typed_causal_admission(kind):
    o, cfg = _batch().online, _config()
    ref = o.instruction_reference
    assert ref is not None
    if kind == "missing":
        o = replace(o, instruction_reference=None)
    elif kind == "unexpected":
        cfg = m5a_config()
    else:
        fields = {
            "state_shape": {"state": ref.state[:, :-1]},
            "dino_shape": {"dino": ref.dino[:, :, :-1]},
            "mask_type": {"observed": ref.observed.float()},
            "age_type": {"age_steps": ref.age_steps.float()},
            "future": {"age_steps": -ref.age_steps},
        }
        o = replace(o, instruction_reference=replace(ref, **fields[kind]))
    with pytest.raises((ValueError, TypeError)):
        o.validate(cfg)


@pytest.mark.parametrize("index", [0, 15, 76])
def test_dataset_anchor_is_declared_instruction_start_not_sliding_history(index):
    data = _dataset(3)
    raw = data[index]
    assert raw["instruction_reference_key"].tolist() == [[0, 3]]
    assert raw["instruction_reference_age"] == index
    assert raw["center_index"] == index + 3
    start_state = data[0]["state"]
    torch.testing.assert_close(raw["instruction_reference_state"], start_state, rtol=0, atol=0)
    tokens = _Tokens()
    cache = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, tokens))
    cached = cache[index]
    assert tokens.keys is not None and tokens.keys[-1].tolist() == [0, 3]
    assert torch.all(cached["instruction_reference_dino"] == 3)
    assert cached["target_future_dinov2_tokens"].shape[0] == 12
    assert cached["history_dinov2_tokens"].shape[0] == 3
    batch = to_training_batch(
        default_collate([cached]),
        goal=GoalTemplate(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {}),
        config=_config(),
        device=torch.device("cpu"),
    )
    assert batch.online.instruction_reference is not None
    batch.validate(_config())
    torch.testing.assert_close(
        batch.online.instruction_reference.state[0], start_state, rtol=0, atol=0
    )
    assert batch.future.dino_supports.shape[1] == 12


def test_dataset_refuses_missing_origin_instead_of_guessing():
    data = _data()
    with pytest.raises(ValueError, match="source_start/context_start"):
        ObservedStateWindowDataset(
            data.episodes,
            [0],
            image_store=data.image_store,
            camera_names=data.camera_names,
            action_normalizer=data.action_normalizer,
            state_normalizer=data.state_normalizer,
            config=replace(data.config, instruction_reference_mode=INSTRUCTION_START_REFERENCE),
        )


def _encoded(center=16):
    cfg, b = _config(), _batch(center)
    model, _ = _model_engine(cfg)
    model.eval()
    cache, state, _ = model.encode_online(b.online)
    assert state is not None
    module = model.intent.organizer.instruction_progress
    assert isinstance(module, InstructionReferenceRead)
    return model, module, cache, state.top.facts, b


def _read(module, cache, facts, b, reference=None):
    ref = b.online.instruction_reference if reference is None else reference
    assert ref is not None and cache.top.intent.target_binding is not None
    return module(
        reference=ref,
        current_dino=b.online.observation.dino_history[:, -1],
        current_state=b.online.history.state,
        facts=facts,
        binding=cache.top.intent.target_binding,
        task_context=cache.top.intent.protected_goal_set.mean(1),
    )


def test_same_observation_has_exact_zero_change_but_retains_match_status():
    _, module, cache, facts, b = _encoded(0)
    output, info = _read(module, cache, facts, b)
    torch.testing.assert_close(
        info["reference_current_read"], info["reference_start_read"], rtol=0, atol=0
    )
    assert info["reference_view_available"].all() and torch.isfinite(output).all()


def test_source_age_is_not_a_neural_phase_feature():
    model, _, _, _, b = _encoded()
    ref = b.online.instruction_reference
    assert ref is not None
    newer = replace(b.online, instruction_reference=replace(ref, age_steps=ref.age_steps + 10000))
    a, _, _ = model.encode_online(b.online)
    c, _, _ = model.encode_online(newer)
    torch.testing.assert_close(
        a.top.intent.public_interval_carrier, c.top.intent.public_interval_carrier, rtol=0, atol=0
    )


def test_invalid_reference_payload_is_quarantined_not_stationary_confidence():
    _, module, cache, facts, b = _encoded()
    ref = b.online.instruction_reference
    assert ref is not None
    values = torch.full_like(ref.dino, torch.nan, requires_grad=True)
    missing = replace(ref, dino=values, observed=torch.zeros_like(ref.observed))
    out, info = _read(module, cache, facts, b, missing)
    assert torch.isfinite(out).all() and not info["reference_view_available"].any()
    assert torch.count_nonzero(info["reference_start_weights"]) == 0
    out.square().sum().backward()
    assert values.grad is not None and torch.count_nonzero(values.grad) == 0
    known, _ = _read(module, cache, facts, b)
    assert not torch.equal(known, out)


def test_reference_object_permutation_and_null_mass():
    _, module, cache, facts, b = _encoded()
    law = cache.top.intent.target_binding
    ref = b.online.instruction_reference
    assert law is not None and ref is not None
    a, _ = _read(module, cache, facts, b)
    index = torch.tensor([3, 0, 2, 1])
    permuted = replace(cache, top=replace(cache.top, intent=cache.top.intent.permute(index)))
    z, _ = _read(module, permuted, facts.permute(index), b)
    torch.testing.assert_close(a, z, rtol=1e-5, atol=1e-7)
    from clearvla.mainline.model.target_binding import TargetBinding

    null = TargetBinding(torch.tensor([[-torch.inf] * 4 + [0.0]]), law.supported)
    neutral = replace(
        cache, top=replace(cache.top, intent=replace(cache.top.intent, target_binding=null))
    )
    z, _ = _read(module, neutral, facts, b)
    assert torch.count_nonzero(z) == 0


@pytest.mark.parametrize("center", [0, 16, 79])
def test_actual_production_optimizer_uses_the_new_reference_reader(center):
    torch.manual_seed(1481)
    cfg, b = _config(), _batch(center)
    model, engine = _model_engine(cfg)
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if "instruction_progress" in n
    }
    validate_finite_training_batch(b)
    engine.train_step(b)
    assert engine.global_step == 1 and before
    changed = []
    for n, p in model.named_parameters():
        if n in before:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            changed.append(not torch.equal(before[n], p))
    assert any(changed)
    if center > 0:
        assert all(changed)


@pytest.mark.parametrize("bf16", [False, True])
def test_action_only_gradient_reaches_reference_without_teacher(bf16):
    torch.manual_seed(1487)
    cfg, b = _config(), _batch(16)
    model, _ = _model_engine(cfg)
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, _, _ = model.encode_online(b.online)
        out = model.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4])
        )
        loss = out.bottom.physical_velocity.float().square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        if "instruction_progress" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
                name
            )
    assert all(p.grad is None for p in model.training_targets.parameters())


def test_one_reference_comparison_per_observation_not_per_ode(monkeypatch):
    cfg, b = _config(), _batch(16)
    model, _ = _model_engine(cfg)
    model.eval()
    seen = []
    module = model.intent.organizer.instruction_progress
    assert module is not None
    hook = module.register_forward_pre_hook(
        lambda _, args, kwargs: seen.append(kwargs["reference"]), with_kwargs=True
    )
    a = sample_action(model, b.online, cfg, generator=torch.Generator().manual_seed(1493))
    assert len(seen) == 1 and seen[0] is b.online.instruction_reference
    assert a.action.shape == (1, 24, 7) and torch.isfinite(a.action).all()
    hook.remove()


def test_exact_checkpoint_and_native_outlet_with_reference(tmp_path: Path, monkeypatch):
    import copy

    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        for kind in ("missing", "changed"):
            wrong = copy.deepcopy(abi)
            observation = cast(dict[str, object], wrong["observation"])
            if kind == "missing":
                del observation["instruction_reference"]
            else:
                chart = cast(dict[str, object], observation["instruction_reference"])
                chart["age"] = "neural-phase-clock"
            with pytest.raises(ValueError, match="reference"):
                validate_deployment_abi(wrong)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def _adapter(monkeypatch):
    import clearvla.simulation.clearvla_policy as module

    data, cfg = _data(), _config()
    policy = object.__new__(module.ClearVLACheckpointPolicy)
    monkeypatch.setattr(
        policy,
        "bundle",
        SimpleNamespace(
            config=cfg,
            model=None,
            action_normalizer=data.action_normalizer,
            state_normalizer=data.state_normalizer,
            language=SimpleNamespace(
                is_instruction_bank=False,
                tokens=torch.zeros(1, 4, 16),
                mask=torch.ones(1, 4, dtype=torch.bool),
            ),
        ),
        raising=False,
    )
    calls = []

    def encode(rgb, _):
        calls.append(1)
        # Declared fixture maps the first image pixel to frozen cached tokens.
        frame_values = np.stack([rgb[c][:, 0, 0, 0] for c in data.camera_names], 1)
        tokens = torch.tensor(frame_values).float()[:, :, None, None].expand(3, 2, 64, 16).clone()
        return tokens, np.zeros((3, 2, 32, 32, 3), np.uint8)

    monkeypatch.setattr(policy, "encoder", SimpleNamespace(encode=encode), raising=False)
    monkeypatch.setattr(policy, "preprocessing", None, raising=False)
    policy.device = torch.device("cpu")
    policy._seed = 123
    policy._generator = torch.Generator()
    policy.reset()
    monkeypatch.setattr(
        module,
        "sample_action",
        lambda *_, **__: SimpleNamespace(
            action=torch.zeros(1, 24, 7), gripper_command=torch.ones(1, 24)
        ),
    )
    return policy, data, calls


def _history(data, end):
    h = CausalHistory()
    e = data.episodes[0]
    for step in range(end + 1):
        rgb = {name: np.full((32, 32, 3), step, np.uint8) for name in data.camera_names}
        observation = PolicyObservation(rgb, e.states_raw[step], e.action_states_raw[step])
        if step == 0:
            h.reset(observation)
        else:
            h.append(e.actions_raw[step - 1], observation)
    return h.snapshot()


def test_online_reference_lifetime_and_explicit_repeat_instruction(monkeypatch):
    policy, data, calls = _adapter(monkeypatch)
    _, first = policy.act_with_input(_history(data, 3), "push block")
    _, later = policy.act_with_input(_history(data, 16), "push block")
    assert first.instruction_reference is not None and later.instruction_reference is not None
    assert later.instruction_reference.age_steps.item() == 13
    torch.testing.assert_close(
        first.instruction_reference.dino, later.instruction_reference.dino, rtol=0, atol=0
    )
    # Returned diagnostic tensors are not mutable aliases of the stored anchor.
    later.instruction_reference.dino.fill_(999)
    _, repeat = policy.act_with_input(_history(data, 16), "push block")
    assert repeat.instruction_reference is not None
    assert torch.all(repeat.instruction_reference.dino == 3)
    rng = policy._generator.get_state().clone()
    policy.begin_instruction("push block")
    assert torch.equal(rng, policy._generator.get_state())
    _, restarted = policy.act_with_input(_history(data, 16), "push block")
    assert restarted.instruction_reference is not None
    assert restarted.instruction_reference.age_steps.item() == 0
    assert torch.all(restarted.instruction_reference.dino == 16)
    assert len(calls) == 4  # no extra DINO encoder for the anchor
    with pytest.raises(ValueError, match="restarted"):
        policy.act_with_input(_history(data, 1), "push block")
    policy.reset()
    _, fresh = policy.act_with_input(_history(data, 0), "push block")
    assert (
        fresh.instruction_reference is not None
        and fresh.instruction_reference.age_steps.item() == 0
    )


def test_online_offline_instruction_origin_and_feature_state_agree(monkeypatch):
    policy, data, _ = _adapter(monkeypatch)
    policy.act_with_input(_history(data, 3), "push block")
    _, online = policy.act_with_input(_history(data, 18), "push block")
    source = CachedTokenPolicyWindowDataset(_dataset(3), token_store=cast(Any, _Tokens()))
    raw = source[15]
    assert online.instruction_reference is not None
    torch.testing.assert_close(
        online.instruction_reference.dino[0], raw["instruction_reference_dino"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        online.instruction_reference.state[0], raw["instruction_reference_state"], rtol=0, atol=0
    )
    assert online.instruction_reference.age_steps[0] == raw["instruction_reference_age"]


def test_cached_half_precision_source_reaches_fp32_policy():
    b = _batch(16)
    ref = b.online.instruction_reference
    assert ref is not None
    o = replace(
        b.online,
        observation=replace(
            b.online.observation, dino_history=b.online.observation.dino_history.half()
        ),
        instruction_reference=replace(ref, dino=ref.dino.half()),
    )
    model, _ = _model_engine(_config())
    cache, _, _ = model.encode_online(o)
    assert torch.isfinite(cache.top.intent.public_interval_carrier).all()
    cache.top.intent.public_interval_carrier.square().mean().backward()
    reader = model.intent.organizer.instruction_progress
    assert reader is not None and reader.key.weight.grad is not None


@pytest.mark.parametrize("where", ["state", "observed_dino", "masked_dino"])
def test_reference_finite_admission_respects_physical_source_mask(where):
    cfg, b = _config(), _batch(16)
    ref = b.online.instruction_reference
    assert ref is not None
    if where == "state":
        ref = replace(ref, state=torch.full_like(ref.state, float("nan")))
    else:
        ref = replace(ref, dino=torch.full_like(ref.dino, float("nan")))
        if where == "masked_dino":
            ref = replace(ref, observed=torch.zeros_like(ref.observed))
    online = replace(b.online, instruction_reference=ref)
    if where == "masked_dino":
        online.validate(cfg)
    else:
        with pytest.raises(ValueError, match="finite"):
            online.validate(cfg)


@pytest.mark.parametrize("center", [0, 79])
def test_half_cache_current_reference_and_future_share_the_training_ingress(center):
    cfg, b = _config(), _batch(center)
    reference = b.online.instruction_reference
    assert reference is not None
    b = replace(
        b,
        online=replace(
            b.online,
            observation=replace(
                b.online.observation, dino_history=b.online.observation.dino_history.half()
            ),
            instruction_reference=replace(reference, dino=reference.dino.half()),
        ),
        future=replace(b.future, dino_supports=b.future.dino_supports.half()),
    )
    original = b.online.observation.dino_history.clone()
    model, engine = _model_engine(cfg)
    engine.train_step(b)
    assert engine.global_step == 1
    assert b.online.observation.dino_history.dtype == torch.float16
    torch.testing.assert_close(original, b.online.observation.dino_history, rtol=0, atol=0)
    for name, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


@torch.no_grad()
def test_reference_and_policy_are_isolated_from_changed_future_supervision():
    torch.manual_seed(1499)
    cfg, b = _config(), _batch(16)
    model, _ = _model_engine(cfg)
    model.eval()
    changed = replace(b, future=replace(b.future, dino_supports=17 * b.future.dino_supports + 9))
    assert changed.online.instruction_reference is b.online.instruction_reference
    left = sample_action(model, b.online, cfg, generator=torch.Generator().manual_seed(5))
    right = sample_action(model, changed.online, cfg, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(left.action, right.action, rtol=0, atol=0)


@pytest.mark.parametrize("reference_enabled", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_real_training_diagnostics_are_executable_and_do_not_change_loss(
    reference_enabled: bool, grad_enabled: bool
):
    cfg = _config() if reference_enabled else m5a_config()
    batch = _batch() if reference_enabled else unanchored_batch(16)
    torch.manual_seed(873)
    model, engine = _model_engine(cfg)
    model.eval()
    rng = torch.random.get_rng_state()
    # Hold autograd mode fixed: eval/no-grad and eval/grad may dispatch different
    # attention kernels. This test isolates logging, not cross-kernel roundoff.
    # Both deployed no-grad diagnostics and differentiable training diagnostics
    # retain exact equality; no numerical tolerance is relaxed.
    with torch.set_grad_enabled(grad_enabled):
        plain, _ = engine._forward(
            batch, training=False, collect_diagnostics=False,
            generator=torch.Generator().manual_seed(873),
        )
    plain_rng = torch.random.get_rng_state().clone()
    plain_gradients: dict[str, torch.Tensor | None] = {}
    if grad_enabled:
        plain.total.backward()
        plain_gradients = {
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(rng)
    with torch.set_grad_enabled(grad_enabled):
        traced, metrics = engine._forward(
            batch, training=False, collect_diagnostics=True,
            generator=torch.Generator().manual_seed(873),
        )
    torch.testing.assert_close(plain.total, traced.total, rtol=0, atol=0)
    torch.testing.assert_close(plain_rng, torch.random.get_rng_state(), rtol=0, atol=0)
    for plain_values, traced_values in (
        (plain.groups, traced.groups), (plain.contributions, traced.contributions)
    ):
        assert plain_values.keys() == traced_values.keys()
        for name in plain_values:
            torch.testing.assert_close(plain_values[name], traced_values[name], rtol=0, atol=0)
    key = "object_intent_observed_transport_per_view_rms"
    assert key in metrics and torch.isfinite(metrics[key])
    assert "object_intent_observed_transport_rms" not in metrics
    assert metrics["object_p2_semantic_spatial_query_present"] == 0
    assert "object_p2_semantic_terminal_query_delta_rms" not in metrics
    assert torch.isfinite(metrics["object_p2_semantic_terminal_query_weight_rms"])
    if grad_enabled:
        traced.total.backward()
        assert any(value is not None for value in plain_gradients.values())
        for name, p in model.named_parameters():
            expected = plain_gradients[name]
            if expected is None:
                assert p.grad is None, name
            else:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                torch.testing.assert_close(expected, p.grad, rtol=0, atol=0)
    else:
        assert not plain.total.requires_grad and not traced.total.requires_grad
        assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("reference_enabled", [False, True])
def test_actual_optimizer_diagnostic_batch_uses_selected_parameter_owners(reference_enabled):
    cfg = _config() if reference_enabled else m5a_config()
    batch = _batch() if reference_enabled else unanchored_batch(16)
    torch.manual_seed(877)
    model, engine = _model_engine(cfg)
    before = {n: p.detach().clone() for n,p in model.named_parameters() if "shared_binder" in n}
    result = engine.train_step(batch, collect_diagnostics=True)
    assert engine.global_step == 1 and result is not None
    assert "policy_compiler.effect_reader.source_query.0.weight" not in dict(model.named_parameters())
    metrics = engine._r2_parameter_gradient_metrics()
    assert metrics["gradient_parameter_p2_semantic_spatial_query_present"] == 0
    assert "gradient_parameter_p2_semantic_spatial_query_weight_rms" not in metrics
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert before and any(not torch.equal(value, dict(model.named_parameters())[name]) for name,value in before.items())
