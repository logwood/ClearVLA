"""P3/static/bottom source contracts, using actual compiler and bottom classes.

Synthetic boundary values do not establish pretrained DINO or learned skill.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from test_mainline_task_execution_integration import config as task_config

from clearvla.mainline.bottom_evidence import NORMALIZED_EVIDENCE
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.global_task import (
    COMPILED_TASK_GLOBAL, PROPRIOCEPTIVE_GLOBAL, global_intent_memory, global_task_metadata,
)
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.components import ExecutionBottomStage
from clearvla.mainline.model.global_task import GlobalTaskCompiler
from clearvla.mainline.model.restored_bottom import RestoredV120EvidenceBottom, _build_decoder_config
from clearvla.mainline.runtime.deployment import build_deployment_abi, validate_deployment_abi
from clearvla.mainline.v120_core.time_domain_mmdit import EvidenceConditionOrganizer, EvidenceViewAdapter


def config(amp=False):
    c = task_config(amp)
    return replace(c, bottom=replace(c.bottom, global_condition_mode=COMPILED_TASK_GLOBAL))


def test_default_and_candidate_config_identity():
    old = task_config()
    assert 'global_condition_mode' not in old.as_dict()['bottom']
    assert 'global_condition_mode' not in ExperimentConfig().as_dict()['bottom']
    new = config()
    assert config_from_mapping(new.as_dict()) == new
    assert ComponentSelection.from_config(old) != ComponentSelection.from_config(new)
    preset = load_config('configs/mainline/dinov3_online_task_global_calvin.json')
    reference = load_config('configs/mainline/dinov3_online_cumulative_calvin.json')
    assert replace(preset, bottom=reference.bottom, data=reference.data) == reference
    assert replace(preset.bottom, global_condition_mode=PROPRIOCEPTIVE_GLOBAL) == reference.bottom
    assert replace(preset.data, output_dir=reference.data.output_dir) == reference.data
    assert preset.top.object_view_mode == 'per_camera_values_v1'
    assert preset.data.visual_feature_mode == 'dinov3_online_v1'


@pytest.mark.parametrize('change', ['unknown', 'no_joint', 'no_typed', 'no_magnitude'])
def test_incompatible_compositions_are_rejected(change):
    c = config()
    if change == 'unknown':
        c = replace(c, bottom=replace(c.bottom, global_condition_mode='task_guess'))
    elif change == 'no_joint':
        c = replace(c, top=replace(c.top, task_execution_mode='none'))
    elif change == 'no_typed':
        c = replace(c, top=replace(c.top, p3_coordination_mode='pointwise_legacy_v1'))
    else:
        c = replace(c, bottom=replace(c.bottom, evidence_value_mode=NORMALIZED_EVIDENCE))
    with pytest.raises(ValueError):
        c.validate()


@pytest.mark.parametrize('amp', [False, True])
def test_compiler_keeps_zero_and_ordinary_source_parameter_gradients(amp):
    torch.manual_seed(450)
    compiler = GlobalTaskCompiler(32)
    source = torch.randn(2, 4, 32, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
        record = compiler(source)
        z = compiler(torch.zeros_like(source))
    assert torch.count_nonzero(z.tokens) == 0
    assert record.source_intervals is source and record.compiler_identity == id(compiler)
    record.tokens.float().square().mean().backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    for p in compiler.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['shape', 'integer', 'nan', 'inf'])
def test_invalid_source_fails_before_value_projection(kind):
    compiler = GlobalTaskCompiler(32)
    source = torch.ones(1, 4, 32)
    if kind == 'shape': source = source[:, :3]
    elif kind == 'integer': source = source.long()
    else: source[0, 0, 0] = float(kind)
    calls = []
    handle = compiler.values.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        with pytest.raises((ValueError, TypeError)):
            compiler(source)
        assert not calls
    finally:
        handle.remove()


def test_adapter_ownership_and_unselected_source_rejection():
    source = torch.randn(2, 4, 32)
    record = GlobalTaskCompiler(32)(source)
    state, executed = torch.randn(2, 1, 32), torch.randn(2, 1, 32)
    kw = dict(mode=COMPILED_TASK_GLOBAL, compiled=record, source=source, state=state, executed=executed)
    memory = global_intent_memory(**kw)
    assert memory['task'] is record.tokens and memory['state'] is state and memory['executed'] is executed
    with pytest.raises(ValueError, match='another observation'):
        global_intent_memory(**dict(kw, source=source.clone()))
    with pytest.raises(ValueError, match='requires P3'):
        global_intent_memory(**dict(kw, compiled=None))
    with pytest.raises(ValueError, match='unselected'):
        global_intent_memory(**dict(kw, mode=PROPRIOCEPTIVE_GLOBAL))
    old = global_intent_memory(**dict(kw, mode=PROPRIOCEPTIVE_GLOBAL, compiled=None))
    assert set(old) == {'state', 'executed'}


def abi_for(c, tmp_path):
    from test_mainline_checkpoint import _dataset as identity_fixture
    from test_mainline_state_features import _data
    from clearvla.data.action_chart import resolve_action_state_profile
    from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
    from clearvla.mainline.runtime.deployment import canonical_sha256
    data = _data()
    lang = tmp_path / 't5.pt'
    torch.save({'tokens': torch.zeros(1, 2, c.dimensions.goal_token_dim)}, lang)
    dataset = replace(identity_fixture(),
        action_normalizer_sha256=canonical_sha256(data.action_normalizer.to_dict()),
        state_normalizer_sha256=canonical_sha256(data.state_normalizer.to_dict()))
    identity = build_checkpoint_identity(c, repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset, language=ArtifactIdentity.from_file('t5_goal', lang))
    profile = resolve_action_state_profile('calvin_relative_7d_v1')
    abi = build_deployment_abi(c, identity,
        action_normalizer=data.action_normalizer, state_normalizer=data.state_normalizer,
        data_profile={**profile.as_dict(), 'gripper_transition_boundary': profile.gripper_transition_boundary},
        gripper_indices=(6,), goal_metadata={})
    return identity, abi, data


@pytest.mark.parametrize('mutation', ['remove', 'wrong_source', 'inject_old'])
def test_deployment_metadata_rejects_semantic_mismatch(mutation, tmp_path):
    c = task_config() if mutation == 'inject_old' else config()
    _, abi, _ = abi_for(c, tmp_path)
    validate_deployment_abi(abi)
    abi = copy.deepcopy(abi)
    if mutation == 'remove': abi.pop('global_task')
    elif mutation == 'wrong_source': abi['global_task']['source'] = 'raw_language'
    else: abi['global_task'] = global_task_metadata()
    with pytest.raises(ValueError, match='global task'):
        validate_deployment_abi(abi)


def boundary(amp=False):
    """Full production adapter/bank/organizer, compact hidden size only."""
    core = _build_decoder_config(config(amp))
    adapter, organizer = EvidenceViewAdapter(core), EvidenceConditionOrganizer(core)
    h = core.hidden_size
    state, executed = torch.randn(2, 1, h), torch.randn(2, 1, h)
    kw = dict(
        trajectory_tokens=torch.zeros(2, 24, h),
        rollout_tokens=torch.randn(2, 16, h),
        transition_memory=[torch.randn(2, 16, h)],
        event_evidence=torch.randn(2, 24, 3),
        state_memory=[state, torch.randn(2, 1, h)],
        layer_contracts=[{'rollout_tokens': torch.randn(2, 16, h),
                          'state_tokens': state, 'state_history_tokens': torch.randn(2, 1, h)}],
    )
    return core, adapter, organizer, state, executed, kw


@pytest.mark.parametrize('amp', [False, True])
@pytest.mark.parametrize('seed', [40, 71])
def test_fixed_state_different_task_changes_actual_global_and_retains_state(amp, seed):
    torch.manual_seed(seed)
    core, adapter, organizer, state, executed, kw = boundary(amp)
    compiler = GlobalTaskCompiler(core.hidden_size)
    source = torch.randn(2, 4, core.hidden_size, requires_grad=True)
    time = torch.full((2,), .4)
    owner = SimpleNamespace(core_config=core)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
        compiled = compiler(source)
        intent = SimpleNamespace(compiled_global_task=compiled, public_interval_carrier=source)
        memory = ExecutionBottomStage._intent_memory(owner, intent, state, executed)
        other_adapter_memory = RestoredV120EvidenceBottom._intent_memory(owner, intent, state, executed)
        assert all(memory[k] is other_adapter_memory[k] for k in memory)
        view = adapter(**kw, intent_memory=memory)
        out = organizer(view, time)['global_condition']
        memory2 = dict(memory, task=compiler(-source).tokens)
        different = organizer(adapter(**kw, intent_memory=memory2), time)['global_condition']
        assert not torch.equal(out, different)
        different_state = organizer(adapter(**kw, intent_memory=dict(memory, state=-state)), time)['global_condition']
        assert not torch.equal(out, different_state)
        different_exec = organizer(adapter(**kw, intent_memory=dict(memory, executed=-executed)), time)['global_condition']
        assert not torch.equal(out, different_exec)
        zero_view = adapter(**kw, intent_memory=dict(memory, task=torch.zeros_like(compiled.tokens)))
        assert torch.count_nonzero(zero_view.intent_tokens[:, :4]) == 0
        # Upstream action-conditioned sources remain local evidence, not writers
        # to the static global task condition.
        altered = dict(kw, rollout_tokens=kw['rollout_tokens'] * 4,
                       transition_memory=[kw['transition_memory'][0] * -3],
                       event_evidence=kw['event_evidence'] + 2)
        same = organizer(adapter(**altered, intent_memory=memory), time)['global_condition']
        torch.testing.assert_close(out, same, rtol=0, atol=0)
    out.float().square().sum().backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    paths = list(compiler.parameters()) + list(adapter.intent_proj['task'].parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in paths)


def test_selected_bottom_does_not_fallback_without_compiled_task():
    core, adapter, _, state, executed, kw = boundary()
    with pytest.raises(ValueError, match='compiled task memory'):
        adapter(**kw, intent_memory={'state': state, 'executed': executed})
    with pytest.raises(ValueError, match='four interval'):
        adapter(**kw, intent_memory={'task': torch.randn(2, 3, core.hidden_size),
                                   'state': state, 'executed': executed})
