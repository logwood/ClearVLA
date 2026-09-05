from __future__ import annotations

from dataclasses import replace
from unittest import mock

import torch
from test_mainline_bspine_arm_coarse_context import _candidate_config
from test_mainline_policy import _batch

from clearvla.mainline.config import load_config
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.deployment import deployment_flow_schedule
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def test_paired_formal_configs_are_matched_except_routing_and_output() -> None:
    paths = [
        "configs/mainline/object_intent_dynamics_323_pen_bspine_arm_coarse_context.json",
        "configs/mainline/object_intent_dynamics_323_pen_bspine_arm_private_reader.json",
    ]
    configs = [load_config(path) for path in paths]
    payloads = [config.as_dict() for config in configs]
    for payload in payloads:
        payload["data"].pop("output_dir")
        payload["bottom"].pop("bspine_implementation")
    assert payloads[0] == payloads[1]
    for config in configs:
        assert config.optimizer.batch_size == 8
        assert config.optimizer.epochs == 8
        assert config.data.seed == 0
        assert config.runtime.max_train_batches == config.runtime.max_val_batches == 0
        assert config.bottom.flow_time_distribution == "v120_mirrored_beta_1_5_1"
        assert deployment_flow_schedule(config).candidate_id == "Q5/Q5"


def test_gradient_probe_does_not_change_optimizer_update_or_rng() -> None:
    base = _candidate_config()
    for implementation in ("fixed_bspline_arm_coarse_context_v1", "fixed_bspline_arm_private_reader_v1"):
        config = replace(base, bottom=replace(base.bottom, bspine_implementation=implementation))
        torch.manual_seed(1600)
        batch = _batch(config)
        results = []
        states = []
        rng = []
        for enable in (False, True):
            torch.manual_seed(1601)
            model = ClearVLAMainlinePolicy(config)
            optimizer, _ = build_optimizer(model, config)
            schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1)
            engine = MainlineTrainingEngine(
                model=model, config=config, optimizer=optimizer, schedule=schedule,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            probe = engine._bspine_gradient_direction_probe
            with mock.patch.object(engine, "_bspine_gradient_direction_probe", side_effect=probe if enable else lambda _ledger: {}):
                results.append(engine.train_step(batch, collect_diagnostics=True))
            states.append({key: value.clone() for key, value in model.state_dict().items()})
            rng.append(torch.get_rng_state().clone())
        assert torch.equal(results[0].loss, results[1].loss)
        assert torch.equal(results[0].gradient_norm, results[1].gradient_norm)
        assert torch.equal(rng[0], rng[1])
        assert all(torch.equal(states[0][key], value) for key, value in states[1].items())
