"""M9 CT-only native-readout contracts, explicitly not trained task evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.model.typed_transition import DYNAMIC_SOURCES
from scripts.probe_m9_execution_chain import audit, compact_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    scope="module",
    params=[
        ("structural_rebuild_pen_current.json", False, 0),
        ("structural_rebuild_pen_current.json", True, 0),
        ("structural_rebuild_m6n_calvin.json", False, 0),
        ("structural_rebuild_m6n_calvin.json", True, 0),
        ("structural_rebuild_m6n_calvin.json", False, 1),
        ("structural_rebuild_m6n_calvin.json", True, 1),
    ],
)
def chain_report(request):
    filename, bf16, updates = request.param
    config = compact_config(load_config(ROOT / "configs/mainline" / filename))
    return audit(config, device=torch.device("cpu"), bf16=bf16, seed=9231, updates=updates)


@pytest.mark.parametrize("source", DYNAMIC_SOURCES)
def test_ct_source_reaches_native_readout_without_direct_p3_shortcut(chain_report, source):
    row = next(item for item in chain_report["sources"] if item["source"] == source)
    assert row["direct_p3_plan_bit_exact"]
    assert row["ct_value_rms"] > 0.0
    for name in ("first", "last"):
        endpoint = row["native_arm_vjp"][name]
        assert endpoint["finite"]
        assert endpoint["input_vjp_l2"] > 0.0
        assert endpoint["producer_weight_vjp_l2"] > 0.0
    assert row["gripper_vjp_finite"]
    if chain_report["config"]["bottom"]["gripper_output_mode"] == "continuous":
        assert row["gripper_gradient_readout"] == "continuous_native"
        assert row["gripper_input_vjp_l2"] > 0.0
    else:
        # Binary native decisions can stay unchanged despite a live logit path.
        assert row["gripper_gradient_readout"] == "binary_logit_margin"
        assert row["command_output_weight_vjp_l2"] > 0.0
        if chain_report["optimizer_updates"] == 0:
            # The real binary command head is exactly zero-initialized. It must
            # receive parameter gradients before its input gradients can open.
            assert chain_report["command_output_weight_exact_zero"]
            assert row["gripper_input_vjp_l2"] == 0.0
        else:
            assert not chain_report["command_output_weight_exact_zero"]
            assert row["gripper_input_vjp_l2"] > 0.0


def test_probe_does_not_turn_zero_CT_features_into_a_robot_stop(chain_report):
    assert chain_report["zero_dynamic_CT_exact_zero"]
    assert chain_report["zero_dynamic_native_max_abs"] > 0.0


def test_probe_holds_execution_context_fixed(chain_report):
    assert chain_report["matched_context_baseline_repeat_exact"]
    assert chain_report["diagnostics_preserve_cpu_rng"]
    assert chain_report["buffers_unchanged"]
    assert chain_report["parameter_grad_buffers_unwritten"]
    assert chain_report["optimizer_updates"] in (0, 1)
    assert chain_report["source"] == "synthetic_model_no_checkpoint"


def test_binary_value_isolation_reaches_actual_two_pass_native_actions():
    from test_mainline_controller_values import _config
    from test_mainline_operation_expectation import _batch
    from test_mainline_state_features import _model_engine

    from clearvla.mainline.runtime.sampling import sample_action
    from clearvla.mainline.v120_core.time_domain_mmdit import EvidenceLatentMMDiTActionDecoder

    torch.manual_seed(9210)
    config = _config()
    model, engine = _model_engine(config)
    batch = _batch()
    engine.train_step(batch)
    model.eval()
    # Explicit fully-open diagnostic phase; completed optimizer updates remain 1.
    model.set_training_step(10000)
    decoder = model.execution_bottom.decoder
    assert isinstance(decoder, EvidenceLatentMMDiTActionDecoder)
    assert decoder.execution_controller is not None
    reader = decoder.execution_controller.value_reader
    calls = []

    def change_unused(_module, _args, output):
        calls.append(1)
        delta = torch.arange(output.shape[1], dtype=output.dtype)[None, :, None] * 2
        return torch.stack((output[..., 0], output[..., 1] + delta), dim=-1)

    with torch.no_grad():
        before = sample_action(
            model, batch.online, config, generator=torch.Generator().manual_seed(9211)
        )
        hook = reader.register_forward_hook(change_unused)
        try:
            after = sample_action(
                model, batch.online, config, generator=torch.Generator().manual_seed(9211)
            )
        finally:
            hook.remove()
    assert engine.global_step == 1 and len(calls) > 0
    torch.testing.assert_close(before.action, after.action, rtol=0, atol=0)
