from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal

from clearvla.rl.checkpoint import load_adapter, save_adapter
from clearvla.rl.config import SACConfig
from clearvla.rl.features import Decision, FrozenBaseReader
from clearvla.rl.networks import ResidualActionMap, ResidualActor
from clearvla.rl.replay import ReplayBuffer
from clearvla.rl.runner import check_cuda_memory, run_episode
from clearvla.rl.sac import ResidualSAC, bellman_target
from clearvla.rl.train import validate_baseline
from clearvla.simulation.admission import STACKCUBE_REPAIRED_PROFILE as STACKCUBE_PROFILE
from clearvla.simulation.contracts import (
    EvaluationState,
    PolicyObservation,
    ResetResult,
    StepResult,
)
from clearvla.simulation.history import CausalHistory


def config(**kwargs):
    return replace(
        SACConfig(hidden=16, batch_size=4, replay_capacity=16, learning_starts=4), **kwargs
    )


def fill(learner):
    rng = np.random.default_rng(4)
    for i in range(8):
        learner.replay.add(
            features=rng.normal(size=8),
            next_features=rng.normal(size=8),
            residual=rng.uniform(-1, 1, 7),
            executed=rng.uniform(-1, 1, 7),
            reward=float(i % 2),
            terminated=i == 0,
            truncated=i == 1,
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"gamma": 1},
        {"tau": float("nan")},
        {"seed": True},
        {"residual_fraction": 0.5},
        {"learning_starts": 1},
        {"batch_size": 99},
        {"initial_log_std": -99},
    ],
)
def test_config_fail_closed(bad):
    with pytest.raises(ValueError):
        config(**bad).validate()


def test_zero_mean_exact_and_logprob_is_unit_tanh_density():
    actor = ResidualActor(8, 16, -1)
    obs = torch.zeros(5, 8)
    rng = torch.Generator().manual_seed(11)
    before = rng.get_state().clone()
    zero, _ = actor.sample(obs, generator=rng, deterministic=True)
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
    assert torch.equal(before, rng.get_state())
    action, logprob = actor.sample(obs, generator=rng)
    raw = torch.atanh(action)
    expected = (
        Normal(torch.zeros_like(raw), torch.full_like(raw, np.exp(-1))).log_prob(raw)
        - torch.log1p(-action.square())
    ).sum(-1, keepdim=True)
    torch.testing.assert_close(logprob, expected)
    with torch.no_grad():
        actor.net[-1].bias[:7].fill_(100)
    _, saturated_logprob = actor.sample(obs, generator=rng)
    assert torch.isfinite(saturated_logprob).all()
    assert all(p.grad is None for p in actor.parameters())


def test_execution_bounds_are_native_not_zscore_and_zero_is_baseline():
    mapper = ResidualActionMap(-np.ones(7), np.ones(7), 0.1)
    base = np.array([9, -9, 0, 0.95, -0.95, 0.5, -1], np.float32)
    action, _ = mapper.execute(base, np.zeros(7))
    np.testing.assert_array_equal(action, base.clip(-1, 1))
    action, diag = mapper.execute(base, np.ones(7))
    assert np.max(np.abs(action - base.clip(-1, 1))) <= 0.200001
    assert diag["residual_clip_fraction"] > 0
    with pytest.raises(ValueError):
        mapper.execute(base, np.full(7, np.nan))
    with pytest.raises(ValueError):
        mapper.execute(base, np.full(7, 1.01))


def test_terminal_vs_timeout_bellman_target():
    reward = torch.tensor([[1.0], [2.0]])
    result = bellman_target(
        reward,
        torch.tensor([[1.0], [0.0]]),
        torch.full((2, 1), 3.0),
        torch.full((2, 1), -2.0),
        gamma=0.9,
        alpha=0.1,
    )
    torch.testing.assert_close(result, torch.tensor([[1.0], [4.88]]))


def test_full_process_memory_gate_does_not_report_learner_only(monkeypatch):
    assert check_cuda_memory(torch.device("cpu")) == 0
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (gib, 24 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 3 * gib)
    with pytest.raises(RuntimeError, match="22 GiB"):
        check_cuda_memory(torch.device("cuda"))


def test_replay_ring_copies_and_rejects_bad_rows_without_mutation():
    replay = ReplayBuffer(4, 8, seed=3)
    feature = np.ones(8)

    def row():
        replay.add(
            features=feature,
            next_features=feature + 1,
            residual=np.zeros(7),
            executed=np.ones(7),
            reward=1,
            terminated=False,
            truncated=True,
        )

    for _ in range(6):
        row()
    feature[:] = 8
    assert replay.size == 4 and replay.cursor == 2
    assert np.all(replay.arrays["features"] == 1)
    feature[:] = np.nan
    with pytest.raises(ValueError):
        row()
    assert replay.cursor == 2
    restored = ReplayBuffer(4, 8, seed=9)
    restored.load_state_dict(replay.state_dict())
    for name, tensor in replay.sample(3, torch.device("cpu")).items():
        assert torch.isfinite(tensor).all()


def test_learning_updates_actor_critics_not_targets_by_grad_and_keeps_caller_rng():
    caller = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    learner = ResidualSAC(8, config())
    assert torch.equal(caller, torch.get_rng_state())
    if cuda_states:
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_states, torch.cuda.get_rng_state_all(), strict=True)
        )
    fill(learner)
    old_actor = deepcopy(learner.actor.state_dict())
    old_target = deepcopy(learner.target.state_dict())
    report = learner.update()
    assert all(np.isfinite(value) for value in report.values())
    assert any(not torch.equal(old_actor[n], v) for n, v in learner.actor.state_dict().items())
    for name, target in learner.target.state_dict().items():
        torch.testing.assert_close(
            target, old_target[name].lerp(learner.critic.state_dict()[name], config().tau)
        )
    assert all(p.grad is None and not p.requires_grad for p in learner.target.parameters())
    assert all(p.grad is None for p in learner.critic.parameters())


def test_adapter_checkpoint_roundtrip_continues_next_update(tmp_path):
    learner = ResidualSAC(8, config())
    fill(learner)
    learner.update()
    identity = {"base_sha256": "a" * 64, "environment": "fixture"}
    path = tmp_path / "adapter.pt"
    save_adapter(path, learner, identity, episodes=2, env_steps=8)
    with pytest.raises(FileExistsError):
        save_adapter(path, learner, identity, episodes=2, env_steps=8)
    other = ResidualSAC(8, config())
    position = load_adapter(path, other, identity)
    assert position == {"episodes": 2, "env_steps": 8}
    assert other.update() == learner.update()
    for name, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, other.actor.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        load_adapter(path, other, {"base_sha256": "b" * 64})


def test_nonfinite_optimizer_checkpoint_rejected_before_model_mutation():
    learner = ResidualSAC(8, config())
    fill(learner)
    learner.update()
    state = deepcopy(learner.state_dict())
    other = ResidualSAC(8, config())
    before = deepcopy(other.actor.state_dict())
    owner = next(iter(state["actor_optimizer"]["state"].values()))
    owner["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite optimizer"):
        other.load_state_dict(state)
    for name, value in before.items():
        assert torch.equal(value, other.actor.state_dict()[name])


class Environment:
    def __init__(self, *, terminal=False):
        self.actions, self.terminal = [], terminal

    def observation(self):
        return PolicyObservation(
            rgb={
                name: np.full((8, 8, 3), len(self.actions), np.uint8) for name in ("top", "wrist")
            },
            state=np.full(7, len(self.actions), np.float32),
            action_state=self.actions[-1].copy() if self.actions else np.zeros(7, np.float32),
        )

    def reset(self, *, seed):
        self.actions = []
        return ResetResult(self.observation(), EvaluationState({"secret_goal_pose": 999}))

    def action_bounds(self):
        return -np.ones(7, np.float32), np.ones(7, np.float32)

    def clip_action(self, action):
        return action.clip(-1, 1)

    def step(self, action):
        self.actions.append(action.copy())
        return StepResult(
            self.observation(),
            0.1,
            self.terminal,
            False,
            EvaluationState({"success": False, "secret_goal_pose": -99}),
        )


class Reader:
    def reset(self):
        self.calls, self.history = 0, []

    def decide(self, history):
        assert not hasattr(history, "evaluation") and not hasattr(history, "reward")
        self.calls += 1
        self.history.append(history)
        # Deliberately different on each draw to catch resampled next actions.
        return Decision(
            np.full(8, self.calls, np.float32), np.full(7, 0.1 * self.calls, np.float32)
        )


def test_collector_real_executed_history_cached_next_base_and_timeout():
    env, reader = Environment(), Reader()
    learner = ResidualSAC(8, config(learning_starts=8))
    learner.act = lambda features, deterministic=False: np.full(7, 0.5, np.float32)
    run_episode(env, reader, learner, seed=0, step_budget=3, mode="train")
    assert reader.calls == 4  # includes final timeout observation, no redraw
    np.testing.assert_allclose(env.actions, np.array([[0.2] * 7, [0.3] * 7, [0.4] * 7]))
    np.testing.assert_array_equal(reader.history[1].executed_action_history[-1], env.actions[0])
    replay = learner.replay.arrays
    np.testing.assert_array_equal(replay["features"][:3, 0], [1, 2, 3])
    np.testing.assert_array_equal(replay["next_features"][:3, 0], [2, 3, 4])
    assert replay["truncated"][2] == 1 and replay["terminated"].sum() == 0
    np.testing.assert_array_equal(replay["executed"][:3], env.actions)


def test_base_eval_no_learning_no_residual_and_true_terminal_no_next_base():
    env, reader = Environment(terminal=True), Reader()
    learner = ResidualSAC(8, config())
    before = deepcopy(learner.actor.state_dict())
    report = run_episode(env, reader, learner, seed=0, step_budget=3, mode="base")
    assert reader.calls == 1 and learner.replay.size == 0 and learner.updates == 0
    assert report["terminated"] and not report["truncated"]
    assert report["mean_residual_native_l2"] == 0
    for name, value in before.items():
        assert torch.equal(value, learner.actor.state_dict()[name])


def test_reader_frozen_causal_features_single_encoder_call():
    from clearvla.mainline.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg = replace(
        cfg,
        data=replace(cfg.data, data_profile=STACKCUBE_PROFILE),
        bottom=replace(cfg.bottom, arm_flow_mode="relative_command_adapter"),
    )
    policy = SimpleNamespace(
        bundle=SimpleNamespace(
            config=cfg,
            model=nn.Linear(2, 2),
            language=SimpleNamespace(
                is_instruction_bank=True,
                instructions=("stack the red cube on top of the green cube",),
            ),
        ),
        encoder=nn.Linear(2, 2),
        reset=lambda: None,
    )
    policy.encoder.expected_width = 8
    calls = []

    def act(history, instruction):
        calls.append(instruction)
        return np.zeros((24, 7), np.float32), SimpleNamespace(
            observation=SimpleNamespace(dino_history=torch.ones(1, 3, 2, 16, 8)),
            history=SimpleNamespace(
                state=torch.zeros(1, 7),
                action_state=torch.zeros(1, 7),
                state_history=torch.zeros(1, 3, 7),
                executed_action_history=torch.zeros(1, 8, 7),
            ),
        )

    policy.act_with_input = act
    reader = FrozenBaseReader(policy)
    history = CausalHistory()
    history.reset(Environment().observation())
    decision = reader.decide(history.snapshot())
    assert len(calls) == 1 and decision.features.shape == (1027,)
    assert all(not p.requires_grad for p in policy.bundle.model.parameters())
    assert all(not p.requires_grad for p in policy.encoder.parameters())


def test_baseline_gate_rejects_no_success_wrong_identity_and_smoke():
    identity = {"checkpoint": "abc"}
    rows = [dict(seed=i, mode="base", success=i == 0, mean_residual_native_l2=0) for i in range(20)]
    report = dict(
        schema="clearvla-residual-evaluation-v1",
        mode="base",
        complete=True,
        base_identity=identity,
        episodes=rows,
    )
    assert validate_baseline(report, identity) == list(range(20))
    with pytest.raises(ValueError):
        validate_baseline({**report, "complete": False}, identity)
    with pytest.raises(ValueError):
        validate_baseline(report, {"checkpoint": "def"})
    rows[0]["success"] = False
    with pytest.raises(ValueError, match="pretrain"):
        validate_baseline(report, identity)


def test_entrypoint_base_train_adapter_and_resume_roundtrip(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from clearvla.mainline.config import ExperimentConfig
    from clearvla.rl.train import main
    from clearvla.simulation.admission import STACKCUBE_INSTRUCTION
    from clearvla.simulation.contracts import EnvironmentDescriptor

    descriptor = EnvironmentDescriptor(
        backend="maniskill3",
        benchmark="ManiSkill3",
        task="StackCube-v1",
        robot="panda_wristcam",
        simulator_version="3.0.1",
        control_hz=20,
        max_episode_steps=200,
        state_semantics="[tcp xyz metres, causal rotation-vector of Rx(pi)^-1 R_tcp radians, two-finger opening width metres]",
        action_semantics="ManiSkill normalized pd_ee_delta_pose: translation xyz, axis-angle rotation xyz, gripper target",
        action_state_semantics="previous clipped native 7D action",
        camera_semantics={
            "top": "StackCube base_camera external view",
            "wrist": "panda_wristcam hand_camera",
        },
    )

    class FakeEnv(Environment):
        def __init__(self, max_episode_steps, state_chart):
            super().__init__(terminal=True)
            self.descriptor = descriptor

        def step(self, action):
            result = super().step(action)
            return replace(result, reward=1.0, evaluation=EvaluationState({"success": True}))

        def close(self):
            pass

    class FakeBase:
        def __init__(self, path, **kwargs):
            cfg = ExperimentConfig()
            cfg = replace(
                cfg,
                data=replace(cfg.data, data_profile=STACKCUBE_PROFILE),
                bottom=replace(cfg.bottom, arm_flow_mode="relative_command_adapter"),
            )
            self.encoder = nn.Linear(2, 2)
            self.encoder.expected_width = 8
            self.bundle = SimpleNamespace(
                config=cfg,
                epoch=1,
                model=nn.Linear(2, 2),
                checkpoint_sha256="a" * 64,
                language=SimpleNamespace(
                    is_instruction_bank=True, instructions=[STACKCUBE_INSTRUCTION]
                ),
                deployment_abi={
                    "action": {
                        "data_profile": {
                    "expert_admission": "stackcube_all_source_first_action_v2",
                            "simulator_environment": descriptor.to_dict(),
                            "expert_reset_seeds": list(range(10)),
                        }
                    }
                },
            )

        def reset(self):
            pass

        def deployment_health(self):
            return {"observation": {"dinov2_runtime": {"model": "fixture"}}}

        def act_with_input(self, history, instruction):
            return np.zeros((24, 7), np.float32), SimpleNamespace(
                observation=SimpleNamespace(dino_history=torch.ones(1, 3, 2, 16, 8)),
                history=SimpleNamespace(
                    state=torch.zeros(1, 7),
                    action_state=torch.zeros(1, 7),
                    state_history=torch.zeros(1, 3, 7),
                    executed_action_history=torch.zeros(1, 8, 7),
                ),
            )

    module = types.ModuleType("clearvla.simulation.clearvla_policy")
    module.ClearVLACheckpointPolicy = FakeBase
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module = types.ModuleType("clearvla.simulation.maniskill_adapter")
    module.ManiSkillStackCubeEnv = FakeEnv
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config().as_dict()))

    def run(name, args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rl",
                "--config",
                str(config_path),
                "--base-checkpoint",
                str(tmp_path / "fixture.pt"),
                "--device",
                "cpu",
                "--output",
                str(tmp_path / name),
                *args,
            ],
        )
        main()

    run("base", ["--mode", "base", "--episodes", "20"])
    baseline = str(tmp_path / "base" / "evaluation.json")
    run("train", ["--mode", "train", "--steps", "8", "--baseline-report", baseline])
    adapter = str(tmp_path / "train" / "adapter.pt")
    run(
        "eval",
        [
            "--mode",
            "adapter",
            "--episodes",
            "20",
            "--baseline-report",
            baseline,
            "--adapter-checkpoint",
            adapter,
        ],
    )
    run(
        "resume",
        [
            "--mode",
            "train",
            "--steps",
            "4",
            "--baseline-report",
            baseline,
            "--adapter-checkpoint",
            adapter,
        ],
    )
    saved = torch.load(tmp_path / "resume" / "adapter.pt", weights_only=False)
    assert saved["episodes"] == 12 and saved["env_steps"] == 12
    assert saved["learner"]["updates"] == 9
    row = json.loads((tmp_path / "resume" / "episode_00009.json").read_text())
    assert row["seed"] == 18
    assert json.loads((tmp_path / "eval" / "evaluation.json").read_text())["complete"] is True
    with pytest.raises(FileExistsError):
        run("train", ["--mode", "train", "--steps", "8", "--baseline-report", baseline])
