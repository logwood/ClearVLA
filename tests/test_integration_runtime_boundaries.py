"""Regression coverage for the 2026-09-17 integration, without real assets."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clearvla.mainline.training.engine import validate_finite_training_batch
from clearvla.simulation.history import EXECUTED_ACTION_OFFSETS, VISUAL_OFFSETS, CausalHistory
from tests.test_mainline_policy import _batch, _config
from tests.test_simulation_policy_runtime import _observation

ROOT = Path(__file__).resolve().parents[1]


def test_long_rollout_history_is_bounded_and_preserves_absolute_offsets():
    history = CausalHistory()
    reset_action = np.full(7, -3.0, dtype=np.float32)
    history.reset(_observation(0, np.zeros(7)), reset_action=reset_action)
    for now in range(200):
        if now:
            history.append(np.full(7, now - 1), _observation(now, np.full(7, now)))
        snapshot = history.snapshot()
        assert snapshot.time_index == now
        expected_obs = [max(0, now + offset) for offset in VISUAL_OFFSETS]
        np.testing.assert_array_equal(snapshot.rgb_history['top'][:, 0, 0, 0], expected_obs)
        np.testing.assert_array_equal(snapshot.state_history[:, 0], expected_obs)
        expected_actions = [now + off if now + off >= 0 else -3 for off in EXECUTED_ACTION_OFFSETS]
        np.testing.assert_array_equal(snapshot.executed_action_history[:, 0], expected_actions)
        assert len(history._observations) <= 9
        assert len(history._actions) <= 24
    with pytest.raises(IndexError):
        history._observation_at(0)
    with pytest.raises(IndexError):
        history._action_at(0)
    with pytest.raises(IndexError):
        history._action_at(history.time_index)
    history.reset(_observation(5, np.zeros(7)), reset_action=reset_action)
    assert history.snapshot().time_index == 0
    assert len(history._observations) == 1 and len(history._actions) == 0
    np.testing.assert_array_equal(history.snapshot().rgb_history['top'][:, 0, 0, 0], [5]*3)


@pytest.mark.parametrize('owner', ['action_state', 'codec_gripper_boundary'])
def test_preflight_rejects_nonfinite_online_action_anchors(owner):
    batch = _batch(_config())
    validate_finite_training_batch(batch)
    broken = getattr(batch.online.history, owner).clone()
    broken.flatten()[0] = float('nan')
    history = replace(batch.online.history, **{owner: broken})
    batch = replace(batch, online=replace(batch.online, history=history))
    with pytest.raises(ValueError, match='online.' + owner):
        validate_finite_training_batch(batch)


def _launch(tmp_path, script, *args, overrides=None):
    # A recording interpreter ensures no data loading or training can happen.
    spy = tmp_path / 'python spy'
    spy.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                   'with open(os.environ["ARGV_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    spy.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MAINLINE_', 'CALVIN_', 'RDT_'))
           and k not in {'DATA_ROOT','CACHE_DIR','DINO_CACHE_DIR','T5_CONDITION_PATH','OUT_DIR'}}
    log = tmp_path / 'argv.jsonl'
    env.update(CLEARVLA_PYTHON=str(spy), ARGV_LOG=str(log))
    env.update(overrides or {})
    subprocess.run(['bash', str(ROOT / script), *args], env=env, check=True, capture_output=True, text=True, timeout=15)
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize('script', ['scripts/train_mainline.sh', 'run_current_policy.sh', 'scripts/train_rdt_multitask.sh'])
def test_launchers_preserve_serialized_config_and_custom_interpreter(tmp_path, script):
    calls = _launch(tmp_path, script, '--help', overrides={'MAINLINE_CONFIG': 'custom config.json'})
    args = calls[-1]
    assert args[:3] == ['-u', '-m', 'clearvla.mainline.train']
    assert args[args.index('--config')+1] == 'custom config.json'
    assert args[-1] == '--help'
    for option in ('--dtype', '--batch-size', '--num-workers', '--data-root', '--decoded-cache', '--dino-cache', '--t5-condition', '--output-dir'):
        assert option not in args


def test_launcher_preserves_explicit_overrides_and_cli_priority(tmp_path):
    args = _launch(tmp_path, 'scripts/train_mainline.sh', '--batch-size', '3', overrides={
        'MAINLINE_BATCH_SIZE': '5', 'MAINLINE_DTYPE': 'fp32', 'MAINLINE_NUM_WORKERS': '0',
        'DATA_ROOT': '/data/with spaces', 'OUT_DIR': '/new run',
    })[-1]
    assert args[-2:] == ['--batch-size', '3']
    assert args[args.index('--batch-size')+1] == '5'
    assert args[args.index('--data-root')+1] == '/data/with spaces'
    assert args[args.index('--output-dir')+1] == '/new run'


def test_calvin_v1_launcher_no_longer_drops_help_or_smoke(tmp_path):
    calls = _launch(tmp_path, 'scripts/train_calvin_object_binding_formal_v1.sh', '--smoke', '--max-train-batches', '1')
    assert len(calls) == 2  # profile preflight and training interpreter
    assert calls[-1][-3:] == ['--smoke', '--max-train-batches', '1']


def test_checkpoint_snapshot_includes_online_ingress_and_egress_owners():
    from clearvla.mainline.checkpoint import active_source_snapshot
    paths = dict(active_source_snapshot(ROOT).files)
    for name in ('history.py', 'vision.py', 'contracts.py', 'clearvla_policy.py', 'checkpoint.py'):
        assert 'clearvla/simulation/' + name in paths


@pytest.mark.parametrize('path', ['history.py', 'vision.py', 'clearvla_policy.py', 'checkpoint.py'])
def test_deployment_source_gate_rejects_online_adapter_drift(path):
    from clearvla.mainline.checkpoint import SourceSnapshot
    from clearvla.simulation.checkpoint import _validate_deployment_source_compatibility
    owner = 'clearvla/simulation/' + path
    common = ('clearvla/mainline/model/policy.py', 'a'*64)
    saved = SourceSnapshot(files=(common, (owner, 'b'*64)), digest='c'*64)
    changed = SourceSnapshot(files=(common, (owner, 'd'*64)), digest='e'*64)
    with pytest.raises(ValueError, match='model/runtime source differs'):
        _validate_deployment_source_compatibility(saved, changed)
