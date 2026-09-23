"""Curve/owner admission and real AdamW continuation, not skill qualification."""
from __future__ import annotations

import copy
import math
from typing import Any

import pytest
import torch
from test_mainline_takeover_admission import (
    _checkpoint,
    _reject_without_mutation,
    _tree_equal,
)
from test_mainline_takeover_admission import (
    checkpoint_identity as checkpoint_identity,
)

from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.training.optimizer import WarmupCosineSchedule


@pytest.mark.parametrize("kw", [{"total_steps": 80}, {"warmup_steps": 4}, {"minimum_ratio": .2}])
def test_curve_drift_is_rejected_before_state_mutation(tmp_path, checkpoint_identity, kw):
    path, args = _checkpoint(tmp_path, checkpoint_identity)
    options: dict[str, Any] = dict(warmup_steps=2, total_steps=8, minimum_ratio=.1)
    options.update(kw)
    args["schedule"] = WarmupCosineSchedule(args["optimizer"], **options)
    args["schedule"].base_lrs = (1e-3,)
    args["schedule"]._apply_current_ratio()
    _reject_without_mutation(path, args)

@pytest.mark.parametrize("op", ["save", "load"])
def test_scheduler_must_belong_to_actual_optimizer(tmp_path, checkpoint_identity, op):
    path, args = _checkpoint(tmp_path, checkpoint_identity)
    other = torch.optim.AdamW(args["model"].parameters(), lr=1e-3)
    args["schedule"] = WarmupCosineSchedule(other, warmup_steps=2, total_steps=8, minimum_ratio=.1)
    args["schedule"].step()
    if op == "load":
        _reject_without_mutation(path, args)
    else:
        dest = tmp_path / "invalid-owner" / "bad.pt"
        with pytest.raises(ValueError):
            save_checkpoint(dest, **args, epoch=1, global_step=1, best_metric=None)
        assert not dest.parent.exists()

@pytest.mark.parametrize("field,bad", [
    ("step_index", True), ("step_index", -1), ("step_index", "1"),
    ("base_lrs", (float("nan"),)), ("base_lrs", (float("inf"),)),
    ("base_lrs", (-.1,)), ("base_lrs", (True,)),
])
def test_standalone_rejected_load_is_atomic(field, bad):
    m = torch.nn.Linear(2, 1)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    s = WarmupCosineSchedule(o, warmup_steps=2, total_steps=8, minimum_ratio=.1)
    value = copy.deepcopy(s.state_dict())
    value[field] = bad
    before = copy.deepcopy((s.state_dict(), o.state_dict()))
    with pytest.raises(ValueError):
        s.load_state_dict(value)
    _tree_equal(before, (s.state_dict(), o.state_dict()))

def test_legacy_two_field_state_is_not_silently_upgraded(tmp_path, checkpoint_identity):
    path, args = _checkpoint(tmp_path, checkpoint_identity)
    p = torch.load(path, weights_only=False)
    p["schedule"] = {k: p["schedule"][k] for k in ("step_index", "base_lrs")}
    torch.save(p, path)
    _reject_without_mutation(path, args)

@pytest.mark.parametrize("field,bad", [("warmup_steps", True), ("warmup_steps", 1.5),
    ("total_steps", "8"), ("total_steps", 0), ("minimum_ratio", True), ("minimum_ratio", float("nan"))])
def test_constructor_does_not_coerce_or_mutate_invalid_curve(field, bad):
    m = torch.nn.Linear(2,1)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = copy.deepcopy(o.state_dict())
    kw: dict[str, Any] = dict(warmup_steps=2,total_steps=8,minimum_ratio=.1)
    kw[field]=bad
    with pytest.raises(ValueError):
        WarmupCosineSchedule(o, **kw)
    _tree_equal(before, o.state_dict())

def test_group_reordering_rejected_before_lr_or_clock_change():
    m = torch.nn.Linear(2,1)
    o = torch.optim.AdamW([{"params":[m.weight],"lr":.001},{"params":[m.bias],"lr":.002}])
    s = WarmupCosineSchedule(o,warmup_steps=2,total_steps=8,minimum_ratio=.1)
    o.param_groups.reverse()
    lr = [g["lr"] for g in o.param_groups]
    with pytest.raises(ValueError):
        s.step()
    assert s.step_index == 0 and [g["lr"] for g in o.param_groups] == lr

def test_multistep_exact_resume_preserves_weights_moments_rng_curve(tmp_path, checkpoint_identity):
    path, args = _checkpoint(tmp_path, checkpoint_identity)
    m,o,s = args["model"], args["optimizer"], args["schedule"]
    # Saving already captured this global RNG; no unrelated draws precede run.
    def run():
        trace=[]
        for _ in range(11):
            x=torch.randn(4,3)
            y=torch.randn(4,2)
            o.zero_grad(set_to_none=True)
            loss=(m(x)-y).square().mean()
            loss.backward()
            o.step()
            s.step()
            trace.append(copy.deepcopy((m.state_dict(),o.state_dict(),s.state_dict(),torch.get_rng_state(),loss.detach())))
        return trace
    expected=run()
    load_checkpoint_exact(path, **args)
    actual=run()
    _tree_equal(expected,actual)
    assert s.step_index == 12 and s.ratio(12) == pytest.approx(.1)

def test_finite_formula_unchanged_including_degenerate_duration():
    for warmup,total in [(2,8),(8,2),(2,2)]:
        o=torch.optim.AdamW(torch.nn.Linear(2,1).parameters(),lr=.001)
        s=WarmupCosineSchedule(o,warmup_steps=warmup,total_steps=total,minimum_ratio=.1)
        for step in range(20):
            progress=min(max((step-warmup)/max(total-warmup,1),0.),1.)
            expected=(step+1)/warmup if step<warmup else .1+.9*(.5*(1+math.cos(math.pi*progress)))
            assert s.ratio(step)==expected
