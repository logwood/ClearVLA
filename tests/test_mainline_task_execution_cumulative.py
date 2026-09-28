"""The new execution graph with the retained endpoint and executed-world paths.

The fixture uses synthetic transport. This verifies composition and causality,
not physical success, learned object identities, or real-data convergence.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_annotation_goal import _config as cumulative_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.runtime.qualification import synthetic_batch
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.task_execution import JOINT_TASK_EXECUTION


@pytest.mark.parametrize("amp", [False, True])
def test_cumulative_goal_feedback_and_task_relation_train_and_sample_together(amp):
    torch.manual_seed(67109)
    base = cumulative_config()
    config = replace(base, top=replace(base.top, task_execution_mode=JOINT_TASK_EXECUTION),
                     runtime=replace(base.runtime, compute_dtype="bf16" if amp else "fp32"))
    config.validate()
    batch, normalizer = synthetic_batch(config, count=1, raw_side=32,
                                        device=torch.device("cpu"), seed=67110)
    model, engine = _model_engine(config)
    model.configure_action_normalizer(normalizer)
    result = engine.train_step(batch, collect_diagnostics=True)
    assert engine.global_step == 1 and torch.isfinite(result.loss)
    assert abs(float(result.metrics["loss_ledger_gap"])) < 1e-6
    model.eval()
    with torch.no_grad():
        cache, _, _ = model.encode_online(batch.online)
        assert cache.top.intent.task_relation is not None
        assert cache.top.intent.annotated_goal is not None
        assert cache.world_feedback is not None
        assert cache.top.intent.instruction_change is not None
        reference = cache.top.intent.annotated_goal.prediction.reference
        assert reference is cache.instruction_reference
        a = sample_action(model, batch.online, config, generator=torch.Generator().manual_seed(5))
        b = sample_action(model, batch.online, config, generator=torch.Generator().manual_seed(5))
    assert torch.isfinite(a.action).all() and a.action.shape == (1, 24, 7)
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
