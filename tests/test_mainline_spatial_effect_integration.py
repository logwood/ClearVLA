"""V2 native engine/codec/cache integration on synthetic compact transports.

No simulator, pretrained visual encoder, or behavioral success is tested.
This reuses the unchanged baseline test contracts with the explicitly selected
V2 and compiled-global, per-camera graph.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from test_mainline_endpoint_supervision import _config as endpoint_config
from test_mainline_operation_expectation import _batch as _transport_batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.runtime.deployment import build_deployment_abi, validate_deployment_abi
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.task_execution import (JOINT_SPATIAL_TASK_EXECUTION, JOINT_TASK_EXECUTION, task_execution_metadata)
from clearvla.mainline.global_task import COMPILED_TASK_GLOBAL


@pytest.fixture(autouse=True)
def release_completed_graphs():
    import gc

    gc.collect()
    yield
    gc.collect()


def _batch():
    """Real engine fixture with explicit nonzero synthetic task values.

    The inherited transport fixture uses zero T5 placeholders. Those are kept
    in their original tests; task-dependent positive controls must not obtain
    artificial task content from learned read queries instead.
    """
    batch = _transport_batch()
    goal = batch.online.goal
    tokens = torch.randn(goal.tokens.shape, generator=torch.Generator().manual_seed(43971),
                         dtype=goal.tokens.dtype, device=goal.tokens.device)
    tokens = torch.where(goal.mask[..., None], tokens, 0.0)
    return replace(batch, online=replace(batch.online, goal=replace(goal, tokens=tokens)))


def config(amp=False):
    c = endpoint_config()
    return replace(
        c,
        top=replace(c.top, task_execution_mode=JOINT_SPATIAL_TASK_EXECUTION, object_view_mode="per_camera_values_v1"),
        bottom=replace(c.bottom, global_condition_mode=COMPILED_TASK_GLOBAL),
        runtime=replace(c.runtime, compute_dtype="bf16" if amp else "fp32"),
    )


@pytest.mark.parametrize("amp", [False, True])
def test_actual_training_step_all_new_parameter_owners_and_two_pass_sampling(amp):
    torch.manual_seed(9321)
    c = config(amp)
    m, e = _model_engine(c)
    b = _batch()
    tracked = {
        n: p
        for n, p in m.named_parameters()
        if any(
            x in n
            for x in [
                "shared_binder",
                "task_relation_encoder",
                "intent.organizer.interval_object.",
                "intent.coarse_action.object_read.",
                "p1.target_read.",
                "effect_reader.task_execution.",
                "global_task_compiler.",
                ".evidence_adapter.intent_proj.task.",
            ]
        )
    }
    before = {n: p.detach().clone() for n, p in tracked.items()}
    assert tracked
    parameters = [p for g in e.optimizer.param_groups for p in g["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert all(any(p is x for x in parameters) for p in tracked.values())
    e.train_step(b, collect_diagnostics=True)
    assert e.global_step == 1
    for n, p in tracked.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert all(not torch.equal(p, before[n]) for n, p in tracked.items()), [
        n for n, p in tracked.items() if torch.equal(p, before[n])
    ]
    m.eval()
    with torch.no_grad():
        a = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(9322))
        bb = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(9322))
    assert a.action.shape == (1, 24, 7) and torch.isfinite(a.action).all()
    torch.testing.assert_close(a.action, bb.action, rtol=0, atol=0)


def test_real_cache_rejects_dropped_relation():
    m, e = _model_engine(config())
    b = _batch()
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    c2 = replace(
        cache, top=replace(cache.top, intent=replace(cache.top.intent, task_relation=None))
    )
    with pytest.raises(ValueError, match="task relation"):
        c2.validate(config())
    assert config().as_dict()["top"]["task_execution_mode"] == JOINT_SPATIAL_TASK_EXECUTION


def test_exact_state_dict_reload_and_future_label_isolation():
    torch.manual_seed(9323)
    c = config()
    m, e = _model_engine(c)
    b = _batch()
    e.train_step(b)
    other, _ = _model_engine(c)
    other.load_state_dict(copy.deepcopy(m.state_dict()), strict=True)
    other.set_training_step(1)
    m.eval()
    other.eval()
    with torch.no_grad():
        a = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(7))
        a2 = sample_action(other, b.online, c, generator=torch.Generator().manual_seed(7))
        cache, training, _ = m.encode_online(b.online)
        snapshot = cache.top.intent.task_relation.values.clone()
        m.build_training_targets(training, b.future, collect_diagnostics=False)
    torch.testing.assert_close(a.action, a2.action, rtol=0, atol=0)
    torch.testing.assert_close(cache.top.intent.task_relation.values, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("amp", [False, True])
def test_action_only_gradient_reaches_new_owners_without_teacher(amp):
    torch.manual_seed(9343)
    c = config(amp)
    m, e = _model_engine(c)
    b = _batch()
    # Existing W heads initialize at zero, so the indirect coarse -> W ->
    # action path starts with zero VJP. Use a real update, not edited weights.
    e.train_step(b)
    e.optimizer.zero_grad(set_to_none=True)
    m.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache, _, _ = m.encode_online(b.online)
        out = m.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
        )
        loss = out.bottom.physical_velocity.float().square().mean()
    loss.backward()
    paths = (
        "shared_binder",
        "task_relation_encoder",
        "intent.organizer.interval_object.",
        "intent.coarse_action.object_read.",
        "p1.target_read.",
        "effect_reader.task_execution.",
        "global_task_compiler.",
        ".evidence_adapter.intent_proj.task.",
    )
    for path in paths:
        ps = [(n, p) for n, p in m.named_parameters() if path in n]
        assert ps, path
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in ps), path
        assert sum(p.grad.float().abs().sum().item() for n, p in ps) > 0, path
    assert all(p.grad is None for p in m.training_targets.parameters())


def test_checkpoint_deployment_abi_and_exact_restore(tmp_path):
    from pathlib import Path

    from test_mainline_checkpoint import _dataset as identity_fixture
    from test_mainline_state_features import _data

    from clearvla.data.action_chart import resolve_action_state_profile
    from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
    from clearvla.mainline.model.component_contracts import ComponentSelection
    from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
    from clearvla.mainline.runtime.deployment import canonical_sha256
    from clearvla.simulation.checkpoint import load_deployment_checkpoint

    c = config()
    data = _data()
    b = _batch()
    m, e = _model_engine(c)
    m.configure_action_normalizer(data.action_normalizer)
    e.train_step(b)
    m.eval()
    lang = tmp_path / "language.pt"
    torch.save({"tokens": b.online.goal.tokens, "mask": b.online.goal.mask}, lang)
    dataset = replace(
        identity_fixture(),
        action_normalizer_sha256=canonical_sha256(data.action_normalizer.to_dict()),
        state_normalizer_sha256=canonical_sha256(data.state_normalizer.to_dict()),
    )
    identity = build_checkpoint_identity(
        c,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset,
        language=ArtifactIdentity.from_file("t5_goal", lang),
    )
    assert "clearvla/mainline/model/task_execution.py" in dict(identity.source.files)
    profile = resolve_action_state_profile("calvin_relative_7d_v1")
    abi = build_deployment_abi(
        c,
        identity,
        action_normalizer=data.action_normalizer,
        state_normalizer=data.state_normalizer,
        data_profile={
            **profile.as_dict(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        gripper_indices=(6,),
        goal_metadata={},
    )
    validate_deployment_abi(abi)
    assert abi["task_execution"] == task_execution_metadata(JOINT_SPATIAL_TASK_EXECUTION)
    old_semantics = copy.deepcopy(abi)
    old_semantics["task_execution"] = task_execution_metadata(JOINT_TASK_EXECUTION)
    with pytest.raises(ValueError, match="task execution"):
        validate_deployment_abi(old_semantics)
    for key, value in [("missing", None), ("changed", {"schema": "wrong"})]:
        bad = copy.deepcopy(abi)
        if key == "missing":
            del bad["task_execution"]
        else:
            bad["task_execution"] = value
        with pytest.raises(ValueError, match="task"):
            validate_deployment_abi(bad)
    assert ComponentSelection.from_config(c).intent == "joint_spatial_effect_intent_v2"
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=m,
        optimizer=e.optimizer,
        schedule=e.schedule,
        config=c,
        identity=identity,
        epoch=0,
        global_step=1,
        best_metric=None,
        data_state={
            "action_normalizer": data.action_normalizer.to_dict(),
            "state_normalizer": data.state_normalizer.to_dict(),
            "deployment_abi": abi,
        },
    )
    other, oe = _model_engine(c)
    load_checkpoint_exact(
        path, model=other, optimizer=oe.optimizer, schedule=oe.schedule, config=c, identity=identity
    )
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            path,
            model=other,
            optimizer=oe.optimizer,
            schedule=oe.schedule,
            config=endpoint_config(),
            identity=identity,
        )
    with torch.no_grad():
        a = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(65))
    del other, oe, m, e
    import gc

    gc.collect()
    dep = load_deployment_checkpoint(path, device=torch.device("cpu"))
    with torch.no_grad():
        aa = sample_action(
            dep.model, b.online, dep.config, generator=torch.Generator().manual_seed(65)
        )
    torch.testing.assert_close(a.action, aa.action, rtol=0, atol=0)


def test_world_and_consequence_ablations_are_declared_different_consumers():
    from clearvla.mainline.train import _world_dynamic_neutral_cache

    torch.manual_seed(93151)
    c = config()
    m, e = _model_engine(c)
    b = _batch()
    e.train_step(b)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
        neutral, _ = _world_dynamic_neutral_cache(cache, c)
        reader = m.policy_compiler.effect_reader
        q = torch.randn(1, 24, 4, c.dimensions.hidden_size)
        original = reader.compile_task_execution(
            q, cache.top.candidate_world, cache.top.intent.policy_dock()
        )
        changed = reader.compile_task_execution(
            q, neutral.top.candidate_world, neutral.top.intent.policy_dock()
        )
    assert original is not None and changed is not None
    assert not torch.equal(original.scene, changed.scene)


def test_coarse_task_value_has_no_history_or_query_only_bypass():
    torch.manual_seed(94651)
    c = config()
    m, e = _model_engine(c)
    m.eval()
    b = _batch()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
        dock = cache.top.intent.action_dock()
        relation = replace(dock.task_relation, values=torch.zeros_like(dock.task_relation.values))
        dock = replace(dock, task_relation=relation)
        first = m.intent.coarse_action(dock).action_prediction
        # History remains a query input, but cannot itself supply motion when
        # the joint task value is neutralized. This is NOT a robot stop rule.
        changed = replace(
            dock, history_memory=dock.history_memory + torch.randn_like(dock.history_memory) * 5
        )
        second = m.intent.coarse_action(changed).action_prediction
    torch.testing.assert_close(first, torch.zeros_like(first), rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_compiled_task_single_owner_ode_lifetime_and_changed_observation():
    torch.manual_seed(714)
    c = config()
    m, _ = _model_engine(c)
    m.eval()
    b = _batch()
    compiler_calls, consumer_tasks = [], []
    compiler = m.policy_compiler.plan_compiler.global_task_compiler
    h1 = compiler.register_forward_hook(lambda module, args, output: compiler_calls.append(output))
    def receive(module, args, kwargs):
        consumer_tasks.append(kwargs["intent_memory"]["task"])
    h2 = m.execution_bottom.decoder.evidence_adapter.register_forward_pre_hook(receive, with_kwargs=True)
    try:
        with torch.no_grad():
            first = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(22))
        assert torch.isfinite(first.action).all()
        assert len(compiler_calls) == 1 and len(consumer_tasks) == 12
        assert all(v is compiler_calls[0].tokens for v in consumer_tasks)
        old_record = compiler_calls[0]
        changed_goal = replace(b.online.goal, tokens=b.online.goal.tokens * 1.3)
        changed_online = replace(b.online, goal=changed_goal)
        with torch.no_grad():
            second = sample_action(m, changed_online, c, generator=torch.Generator().manual_seed(22))
        assert len(compiler_calls) == 2
        assert compiler_calls[1] is not old_record
        assert compiler_calls[1].source_intervals is not old_record.source_intervals
        assert all(v is compiler_calls[1].tokens for v in consumer_tasks[12:])
        assert not torch.equal(first.action, second.action)
    finally:
        h1.remove()
        h2.remove()


def test_v2_cache_cannot_silently_consume_v1_relation():
    m, _ = _model_engine(config())
    b = _batch()
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    relation = replace(cache.top.intent.task_relation, execution_mode=JOINT_TASK_EXECUTION)
    altered = replace(cache, top=replace(cache.top, intent=replace(cache.top.intent, task_relation=relation)))
    with pytest.raises(ValueError, match="execution|semantics"):
        altered.validate(config())


@pytest.mark.parametrize("mode", [JOINT_TASK_EXECUTION, JOINT_SPATIAL_TASK_EXECUTION])
def test_object_relabeling_preserves_relation_and_exact_paired_source(mode):
    torch.manual_seed(756)
    c = config()
    c = replace(c, top=replace(c.top, task_execution_mode=mode))
    model, _ = _model_engine(c)
    model.eval()
    b = _batch()
    with torch.no_grad():
        cache, _, _ = model.encode_online(b.online)
    original = cache.top.intent
    index = torch.tensor([2, 0, 3, 1])
    moved = original.permute(index)
    moved.validate(hidden=c.dimensions.hidden_size, horizon=24)
    assert moved.task_relation is not None
    assert moved.task_relation.execution_mode == mode
    assert moved.task_relation.binding is moved.target_binding
    assert moved.task_relation.current_content is moved.operation_expectation.current_content
    assert moved.task_relation.current_state is original.task_relation.current_state
    assert moved.compiled_global_task is original.compiled_global_task
    assert moved.public_interval_carrier is original.public_interval_carrier
    torch.testing.assert_close(moved.task_relation.values,
        original.task_relation.values[:, :, index], rtol=0, atol=0)
    torch.testing.assert_close(moved.task_relation.view_observed,
        original.task_relation.view_observed[:, index], rtol=0, atol=0)
