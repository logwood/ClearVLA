"""Named AdamW continuation must preserve its next update and reject mismatches."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from clearvla.mainline.train import _initialize_optimizer_moments, _initialize_training_clock
from clearvla.mainline.runtime.checkpoints import CALVIN_ENDPOINT_TRAJECTORY_REPAIR_V1_MIGRATION as MIGRATION
from clearvla.mainline.training.optimizer import WarmupCosineSchedule


class Tiny(torch.nn.Linear):
    def set_training_step(self, step):
        self.completed = step


def fixture(tmp_path):
    torch.manual_seed(7)
    original = Tiny(2, 2)
    def optimizer(model):
        return torch.optim.AdamW([{
            "params": list(model.parameters()), "parameter_names": ("weight", "bias"),
            "name": "test/decay",
        }], lr=1e-3, weight_decay=.01)
    opt = optimizer(original)
    for index in range(3):
        opt.zero_grad(set_to_none=True)
        original(torch.tensor([[1., 2.], [-1., .5]])).square().mean().backward()
        opt.step()
    path = tmp_path / "source.pt"
    payload = {"global_step": 3, "identity": {"source": {"digest": "source-digest"}},
               "model": deepcopy(original.state_dict()), "optimizer": deepcopy(opt.state_dict())}
    torch.save(payload, path)
    target = Tiny(2, 2)
    target.load_state_dict(original.state_dict())
    new_opt = optimizer(target)
    schedule = WarmupCosineSchedule(new_opt, warmup_steps=2, total_steps=20, minimum_ratio=.1)
    engine = SimpleNamespace(model=target, optimizer=new_opt, schedule=schedule, global_step=0)
    admitted = SimpleNamespace(global_step=3, saved_source_digest="source-digest", model_contract_migration=MIGRATION)
    _initialize_training_clock(engine, schedule, admitted, mode="checkpoint")
    return original, opt, engine, admitted, path, payload


def test_retained_named_moments_produce_same_next_adam_update_and_keep_current_lr(tmp_path):
    original, opt, engine, admitted, path, _ = fixture(tmp_path)
    lr = engine.optimizer.param_groups[0]["lr"]
    assert lr != opt.param_groups[0]["lr"]
    report = _initialize_optimizer_moments(engine, admitted, path, mode="checkpoint")
    assert report["loaded"] and report["state_tensors"] == 2
    assert engine.optimizer.param_groups[0]["lr"] == lr
    assert engine.global_step == engine.schedule.step_index == 3
    # A new objective/gradient can be used after retaining the old moments.
    for source, restored in zip(original.parameters(), engine.model.parameters(), strict=True):
        source.grad = torch.full_like(source, .037)
        restored.grad = source.grad.clone()
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(opt.state[source][key], engine.optimizer.state[restored][key])
    opt.param_groups[0]["lr"] = lr
    opt.step()
    engine.optimizer.step()
    for source, restored in zip(original.parameters(), engine.model.parameters(), strict=True):
        assert torch.equal(source, restored)
    # Future CPU updates cannot mutate the saved checkpoint's moment tensors.
    saved = torch.load(path, weights_only=False)
    assert float(saved["optimizer"]["state"][0]["step"]) == 3


@pytest.mark.parametrize("fault", [
    "parameter_order", "epsilon", "moment_shape", "moment_nan", "negative_variance",
    "future_step", "unknown_state", "duplicate_id", "weight_identity", "source_identity",
])
def test_bad_continuation_is_rejected_before_optimizer_mutation(tmp_path, fault):
    _, _, engine, admitted, path, payload = fixture(tmp_path)
    group = payload["optimizer"]["param_groups"][0]
    state = payload["optimizer"]["state"][group["params"][0]]
    if fault == "parameter_order":
        group["parameter_names"] = tuple(reversed(group["parameter_names"]))
    elif fault == "epsilon":
        group["eps"] *= 2
    elif fault == "moment_shape":
        state["exp_avg"] = state["exp_avg"].flatten()
    elif fault == "moment_nan":
        state["exp_avg"].flatten()[0] = float("nan")
    elif fault == "negative_variance":
        state["exp_avg_sq"].flatten()[0] = -1
    elif fault == "future_step":
        state["step"].fill_(4)
    elif fault == "unknown_state":
        state["unrecognized"] = torch.zeros(())
    elif fault == "duplicate_id":
        group["params"][1] = group["params"][0]
    elif fault == "weight_identity":
        payload["model"]["weight"].add_(1)
    elif fault == "source_identity":
        payload["identity"]["source"]["digest"] = "another"
    torch.save(payload, path)
    groups = deepcopy(engine.optimizer.state_dict()["param_groups"])
    with pytest.raises(ValueError):
        _initialize_optimizer_moments(engine, admitted, path, mode="checkpoint")
    assert not engine.optimizer.state
    assert engine.optimizer.state_dict()["param_groups"] == groups


def test_opt_in_requires_admitted_clock_and_graph(tmp_path):
    _, _, engine, admitted, path, _ = fixture(tmp_path)
    bad = SimpleNamespace(**{**admitted.__dict__, "model_contract_migration": "different_graph"})
    with pytest.raises(ValueError, match="parameter-preserving"):
        _initialize_optimizer_moments(engine, bad, path, mode="checkpoint")
    engine.global_step = 0
    with pytest.raises(ValueError, match="retained clock"):
        _initialize_optimizer_moments(engine, admitted, path, mode="checkpoint")
    assert not engine.optimizer.state
    report = _initialize_optimizer_moments(engine, admitted, tmp_path / "missing.pt", mode="fresh")
    assert not report["loaded"]


def test_cli_requires_retained_clock_for_moment_initialization():
    from clearvla.mainline.train import _parser, _overrides
    from test_mainline_checkpoint import _reduced_joint_calvin_config
    args = _parser().parse_args([
        "--init-checkpoint", "source.pt", "--init-model-contract-migration", MIGRATION,
        "--init-optimizer-state", "checkpoint",
    ])
    with pytest.raises(ValueError, match="retained training clock"):
        _overrides(_reduced_joint_calvin_config(), args)
    args.init_training_clock = "checkpoint"
    _overrides(_reduced_joint_calvin_config(), args)
