from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from clearvla.data.samplers import ColdStartInformationBatchSampler, InformationBalancedBatchSampler, InformationBalancedSamplerConfig
from clearvla.mainline.config import load_config
from clearvla.mainline.runtime.checkpoints import _validate_maniskill_cold_start_repair


def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/mainline/maniskill_stackcube_dinov3_latest_20261007.json")


@pytest.mark.parametrize("size", (4, 8, 16))
def test_cold_sampler_preserves_uniform_event_and_explicit_cold_quotas(size):
    # Disjoint pools make the actual reserved event/cold quotas observable.
    scores = np.arange(160, dtype=float)
    events = np.arange(160) >= 120
    cold = np.arange(160) < 24
    cfg = InformationBalancedSamplerConfig(batch_size=size, batches_per_epoch=20, seed=7)
    sampler = ColdStartInformationBatchSampler(scores, events, cold, cfg, fraction=.25)
    batches = list(sampler)
    assert len(batches) == 20 and all(len(x) == size for x in batches)
    assert all(len(set(x)) == size for x in batches)
    assert all(cold[x].sum() >= size // 4 for x in batches)
    for i, batch in enumerate(batches):
        quota = int((i+1)*size*.125) - int(i*size*.125)
        assert events[batch].sum() >= quota
    assert batches == list(sampler)
    sampler.set_epoch(1)
    assert batches != list(sampler)
    assert sampler.summary["uniform_fraction"] == .5
    assert sampler.summary["event_fraction"] == .125
    assert sampler.summary["motion_fraction"] == .125


def test_legacy_sampler_and_serialized_config_remain_unchanged_when_off():
    cfg = config()
    assert cfg.data.cold_start_fraction == 0
    assert "cold_start_fraction" not in cfg.as_dict()["data"]
    a = InformationBalancedBatchSampler(np.arange(100), np.zeros(100, bool), InformationBalancedSamplerConfig(batch_size=8, seed=9))
    assert list(a) == list(a)
    assert "cold_start_fraction" not in a.summary


def test_migration_admits_only_baseline_control_or_declared_training_selectors():
    old = config()
    _validate_maniskill_cold_start_repair(old, old)
    new = replace(old, data=replace(old.data, cold_start_fraction=.25),
                  top=replace(old.top, action_history_condition_dropout=.5),
                  optimizer=replace(old.optimizer, batch_size=4, epochs=1))
    new.validate()
    _validate_maniskill_cold_start_repair(old, new)
    with pytest.raises(ValueError, match="outside sampler/history"):
        _validate_maniskill_cold_start_repair(old, replace(new, top=replace(new.top, object_slots=5)))
    with pytest.raises(ValueError, match="outside sampler/history"):
        _validate_maniskill_cold_start_repair(old, replace(new, objectives=replace(new.objectives, gripper_event_threshold=.3)))
    with pytest.raises(ValueError, match="audited baseline"):
        _validate_maniskill_cold_start_repair(new, new)
    with pytest.raises(ValueError, match="divisible by four"):
        replace(new, optimizer=replace(new.optimizer, batch_size=2)).validate()


def test_empty_or_incompatible_cold_pool_fails_closed():
    cfg = InformationBalancedSamplerConfig(batch_size=8)
    for cold in (np.zeros(20, bool), np.ones(19, bool)):
        with pytest.raises(ValueError):
            ColdStartInformationBatchSampler(np.arange(20), np.zeros(20, bool), cold, cfg, fraction=.25)
    with pytest.raises(ValueError, match="ManiSkill"):
        replace(config().data, data_profile="identity_7d_pen", cold_start_fraction=.25).validate()


def test_maniskill_can_retain_verified_optimizer_moments(tmp_path):
    from test_calvin_optimizer_moment_initialization import fixture
    from clearvla.mainline.runtime.checkpoints import MANISKILL_COLD_START_REPAIR_V1_MIGRATION
    from clearvla.mainline.train import _initialize_optimizer_moments, _parser, _overrides
    _, _, engine, admitted, path, _ = fixture(tmp_path)
    admitted.model_contract_migration = MANISKILL_COLD_START_REPAIR_V1_MIGRATION
    report = _initialize_optimizer_moments(engine, admitted, path, mode="checkpoint")
    assert report["loaded"] and engine.global_step == 3
    args = _parser().parse_args(["--init-checkpoint", "source.pt", "--init-model-contract-migration", MANISKILL_COLD_START_REPAIR_V1_MIGRATION,
                                "--init-optimizer-state", "checkpoint", "--init-training-clock", "checkpoint"])
    _overrides(config(), args)
