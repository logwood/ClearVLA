from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch
import torch.nn.functional as F

from clearvla.policy.refinement import NestedLowRankContractionBank
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.decoder import HierarchicalMMDiTActionDecoder
from clearvla.policy.evidence import HierarchicalEvidenceWorkspace, OwnedEvidenceMemoryBank
from clearvla.policy.intent import IntentContractCompiler, PolicyConditionOrganizer
from clearvla.policy.legacy.residual import LayeredV37StyleResidualActionFlowDenoiser
from clearvla.policy.system import V39PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    V39PolicyTrainerConfig,
    _accumulate_metric_tensors,
    _oracle_exit_supervision,
    _optimizer_groups,
    _sync_loss_row,
)


class HierarchicalMMDiTActionDecoderTest(unittest.TestCase):
    @staticmethod
    def _config() -> V39PolicyConfig:
        return V39PolicyConfig(
            hidden_size=32,
            num_heads=4,
            depth=2,
            midcut_layer=1,
            action_horizon=4,
            first_execution_steps=2,
            mid_execution_steps=4,
            latent_action_near_steps=2,
            latent_action_mid_steps=4,
            dropout=0.0,
            final_action_decoder="hierarchical_mmdit_action",
            layer_contract_adapters=1,
            hierarchical_mmdit_depth=2,
            hierarchical_mmdit_refine_steps=2,
            hierarchical_mmdit_low_slots=5,
            hierarchical_mmdit_stage_slots=4,
            hierarchical_mmdit_operator_stages=4,
            hierarchical_mmdit_operator_rank=16,
            hierarchical_mmdit_operator_groups=16,
            hierarchical_mmdit_ffn_expansion=2.0,
        )

    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = self._config()
        self.batch = 2

    def test_metric_boundaries_reject_non_scalar_tensors(self) -> None:
        losses = {"hierarchical_mmdit_exit_logits": torch.zeros(2, 3)}
        message = r"hierarchical_mmdit_exit_logits.*shape=\(2, 3\)"
        with self.assertRaisesRegex(ValueError, message):
            _sync_loss_row(losses)
        with self.assertRaisesRegex(ValueError, message):
            _accumulate_metric_tensors({}, losses)

    def test_intent_compiler_api_has_no_oracle_or_diffusion_inputs(self) -> None:
        parameters = set(inspect.signature(IntentContractCompiler.forward).parameters)
        forbidden = {"target", "target_physical", "noisy_action", "noisy_physical", "time", "z"}
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_dynamic_summaries_only_change_read_selector_contract(self) -> None:
        h = self.config.hidden_size
        compiler = IntentContractCompiler(self.config).eval()
        stable = {
            "geom_summary": torch.randn(self.batch, h),
            "global_summary": torch.randn(self.batch, h),
            "state_summary": torch.randn(self.batch, h),
        }
        dynamic_a = {
            "layer_scan": torch.randn(self.batch, h),
            "transition_summary": torch.randn(self.batch, h),
            "event_summary": torch.randn(self.batch, h),
        }
        dynamic_b = {key: value + 10.0 * torch.randn_like(value) for key, value in dynamic_a.items()}
        output_a = compiler(**stable, **dynamic_a)
        output_b = compiler(**stable, **dynamic_b)
        torch.testing.assert_close(output_a["global_intent"], output_b["global_intent"])
        torch.testing.assert_close(output_a["stage_contract"], output_b["stage_contract"])
        self.assertFalse(torch.allclose(output_a["read_contract"], output_b["read_contract"]))

    def test_owned_bank_assigns_equal_prior_mass_per_role(self) -> None:
        h = self.config.hidden_size
        bank = OwnedEvidenceMemoryBank(self.config)
        sources = {
            "layer": torch.randn(self.batch, 6, h),
            "trajectory": torch.randn(self.batch, 2, h),
            "rollout": torch.randn(self.batch, 3, h),
            "transition": torch.randn(self.batch, 4, h),
            "event": torch.randn(self.batch, 1, h),
            "state": torch.randn(self.batch, 2, h),
        }
        _, bias, ranges = bank.prepare_sources(
            sources,
            batch_size=self.batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
            allow_empty=False,
        )
        for role in bank.ROLE_NAMES:
            role_mass = torch.zeros(())
            for name, (start, stop) in ranges.items():
                if bank._source_role(name) == role:
                    role_mass = role_mass + bias[start:stop].exp().sum()
            torch.testing.assert_close(role_mass, torch.ones(()), atol=1e-6, rtol=1e-6)

    def test_role_index_buffer_does_not_change_legacy_workspace_checkpoints(self) -> None:
        legacy = HierarchicalEvidenceWorkspace(self.config, stratified_roles=False)
        self.assertNotIn("low_slot_role_ids", legacy.state_dict())
        clean = HierarchicalMMDiTActionDecoder(self.config)
        self.assertIn("low_slot_role_ids", clean.workspace.state_dict())

    def test_organizer_excludes_noisy_trajectory_summary_from_owned_sources(self) -> None:
        cfg = self.config
        h = cfg.hidden_size
        organizer = PolicyConditionOrganizer(cfg).eval()
        layers_a = []
        layers_b = []
        for _ in range(cfg.depth):
            entry = {
                key: torch.randn(self.batch, 3, h)
                for key in LayeredV37StyleResidualActionFlowDenoiser._LAYER_KEYS
            }
            changed = dict(entry)
            changed["trajectory_pooled"] = entry["trajectory_pooled"] + 1000.0
            layers_a.append(entry)
            layers_b.append(changed)
        shared = {
            "trajectory_tokens": torch.randn(self.batch, cfg.action_horizon, h),
            "trajectory_workspace_tokens": torch.randn(self.batch, cfg.action_horizon, h),
            "rollout_tokens": torch.randn(self.batch, 5, h),
            "transition_memory": [torch.randn(self.batch, 4, h)],
            "event_evidence": torch.randn(self.batch, cfg.action_horizon, 3),
            "state_memory": [torch.randn(self.batch, 2, h)],
            "intent_memory": {
                name: torch.randn(self.batch, 2, h)
                for name in PolicyConditionOrganizer._INTENT_SOURCE_NAMES
            },
        }
        output_a = organizer(layer_contracts=layers_a, **shared)
        output_b = organizer(layer_contracts=layers_b, **shared)
        torch.testing.assert_close(output_a["layer_scan"], output_b["layer_scan"])
        evidence_a = output_a["evidence_sources"]
        evidence_b = output_b["evidence_sources"]
        self.assertNotIn("intent", evidence_a)
        torch.testing.assert_close(evidence_a["layer"], evidence_b["layer"])

    def test_complete_forward_has_one_action_path_and_live_contract_gradients(self) -> None:
        cfg = self.config
        h = cfg.hidden_size
        decoder = HierarchicalMMDiTActionDecoder(cfg).train()
        decoder.set_operator_contraction_training_step(10_000)
        layer_contracts = []
        for _ in range(cfg.depth):
            layer_contracts.append({
                key: torch.randn(self.batch, 3, h)
                for key in LayeredV37StyleResidualActionFlowDenoiser._LAYER_KEYS
            })
        noisy = torch.randn(self.batch, cfg.action_horizon, cfg.physical_action_dim)
        output = decoder(
            noisy_physical=noisy,
            time=torch.rand(self.batch),
            trajectory_tokens=torch.randn(self.batch, cfg.action_horizon, h),
            trajectory_workspace_tokens=torch.randn(self.batch, cfg.action_horizon, h),
            rollout_tokens=torch.randn(self.batch, 8, h),
            transition_memory=[torch.randn(self.batch, 6, h)],
            event_evidence=torch.randn(self.batch, cfg.action_horizon, 3),
            state_memory=[torch.randn(self.batch, 1, h), torch.randn(self.batch, 3, h)],
            intent_memory={
                name: torch.randn(self.batch, 2, h)
                for name in PolicyConditionOrganizer._INTENT_SOURCE_NAMES
            },
            layer_contracts=layer_contracts,
        )
        self.assertEqual(
            tuple(output["pred_velocity"].shape),
            (self.batch, cfg.action_horizon, cfg.physical_action_dim),
        )
        self.assertFalse(any("cvae" in key or "posterior" in key for key in output))
        self.assertEqual(float(output["intent_contract_deterministic"]), 1.0)
        self.assertEqual(float(output["owned_hierarchical_manager_fixed_output_prior"]), 1.0)
        self.assertEqual(float(output["owned_hierarchical_manager_fixed_role_prior"]), 1.0)
        self.assertEqual(float(output["owned_hierarchical_low_role_stratified"]), 1.0)
        self.assertEqual(float(output["owned_hierarchical_low_causal_attention"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_serial_composition"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_competitive_market"]), 0.0)
        self.assertEqual(
            cfg.hierarchical_mmdit_architecture_version,
            "post_gate_contraction_sidecar_v11_oracle_router",
        )
        self.assertFalse(hasattr(decoder, "shared_block"))
        self.assertNotIn("noisy_market_bias", dict(decoder.named_parameters()))
        self.assertNotIn("noisy_market_bias", dict(decoder.named_buffers()))
        self.assertEqual(float(output["hierarchical_mmdit_shared_core_count"]), 0.0)
        self.assertEqual(
            float(output["hierarchical_mmdit_distinct_blocks"]),
            float(cfg.hierarchical_mmdit_depth),
        )
        self.assertEqual(
            float(output["hierarchical_mmdit_full_rank_block_count"]),
            float(cfg.hierarchical_mmdit_depth),
        )
        self.assertEqual(
            float(output["hierarchical_mmdit_operator_stage_count"]),
            float(cfg.hierarchical_mmdit_operator_stages),
        )
        self.assertEqual(
            float(output["hierarchical_mmdit_refine_block_count"]),
            float(cfg.hierarchical_mmdit_depth),
        )
        self.assertEqual(float(output["hierarchical_mmdit_mandatory_low_rank_writer"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_shared_full_rank_path"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_distinct_full_rank_path"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_step_conditioned_full_rank"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_shared_base_scale_identifiable"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_shared_base_bias_free"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_stage_nested_contraction"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_contraction_sidecar"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_post_gate_sidecar"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_shared_amplitude_owner"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_duplicate_amplitude_owner"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_geometry_identifiable"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_boundary_identity"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_nested_path"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_continuous_depth"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_nonexpansive"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_post_contraction_renorm"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_operator_stage_local_selection"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_block_state_normalized"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_factor_cache_per_forward"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_stage_selector_dynamic_state"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_stage_selector_read_only_inputs"]), 1.0)
        self.assertGreaterEqual(
            float(output["hierarchical_mmdit_stage_selector_query_change"]), 0.0
        )
        for block in decoder.blocks:
            for projection in (
                block.self_out,
                block.noisy_out,
                block.stage_out,
                block.low_out,
            ):
                self.assertIsNotNone(projection.bias)
            self.assertIsNone(block.ffn.net[2].bias)
        self.assertEqual(
            float(output["hierarchical_mmdit_operator_rank"]),
            float(cfg.hierarchical_mmdit_operator_rank),
        )
        for role in decoder.workspace.memory_bank.ROLE_NAMES:
            torch.testing.assert_close(
                output[f"owned_workspace_role_{role}_attention"],
                torch.tensor(0.2),
                atol=1e-5,
                rtol=1e-5,
            )
        action_role_mass = sum(
            output[f"hierarchical_mmdit_action_low_role_{role}_attention"]
            for role in decoder.workspace.memory_bank.ROLE_NAMES
        )
        torch.testing.assert_close(action_role_mass, torch.ones(()), atol=1e-5, rtol=1e-5)
        expected_host_gates = {
            "self": 0.02,
            "noisy": 0.08,
            "stage": 0.04,
            "low": 0.06,
            "ffn": 0.02,
        }
        for branch in ("self", "noisy", "stage", "low", "ffn"):
            self.assertGreater(
                float(output[f"hierarchical_mmdit_action_{branch}_depth_ratio"]), 0.0
            )
            self.assertLessEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_depth_ratio"]), 1.0
            )
            self.assertLess(
                float(output[f"hierarchical_mmdit_action_{branch}_basis_norm_error"]),
                2e-5,
            )
            self.assertLess(
                float(output[f"hierarchical_mmdit_action_{branch}_basis_orthogonality_error"]),
                2e-5,
            )
            self.assertLessEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_effective_depth"]),
                float(cfg.hierarchical_mmdit_operator_rank),
            )
            self.assertLessEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_contraction_ratio"]),
                1.0 + 2e-6,
            )
            self.assertGreaterEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_subspace_energy_fraction"]),
                0.0,
            )
            self.assertLessEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_subspace_energy_fraction"]),
                1.0,
            )
            self.assertLessEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_nonexpansive_violation"]),
                2e-6,
            )
            self.assertEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_boundary_identity_error"]),
                0.0,
            )
            self.assertEqual(
                float(output[f"hierarchical_mmdit_action_{branch}_nested_order_violation"]),
                0.0,
            )
            self.assertGreater(float(output[f"hierarchical_mmdit_action_{branch}_base_rms"]), 0.0)
            torch.testing.assert_close(
                output[f"hierarchical_mmdit_action_{branch}_base_gate"],
                torch.tensor(expected_host_gates[branch]),
                atol=1e-6,
                rtol=0.0,
            )
            self.assertGreater(float(output[f"hierarchical_mmdit_action_{branch}_basis_raw_norm"]), 0.0)
        for stage_index in range(cfg.hierarchical_mmdit_operator_stages):
            for branch in ("self", "noisy", "stage", "low", "ffn"):
                self.assertIn(
                    f"hierarchical_mmdit_stage_{stage_index}_{branch}_effective_depth",
                    output,
                )
        for block_index in range(cfg.hierarchical_mmdit_depth):
            for branch in ("self", "noisy", "stage", "low", "ffn"):
                self.assertIn(
                    f"hierarchical_mmdit_block_{block_index}_{branch}_effective_depth",
                    output,
                )
        self.assertGreaterEqual(
            float(output["hierarchical_mmdit_action_serial_cancellation_orthogonal_baseline"]), 0.0
        )
        self.assertLessEqual(
            abs(float(output["hierarchical_mmdit_action_branch_weighted_cosine"])), 1.0
        )
        torch.testing.assert_close(
            output["owned_hierarchical_manager_low_output_strength"], torch.ones(()), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            output["owned_hierarchical_manager_stage_output_strength"], torch.ones(()), atol=1e-6, rtol=0.0
        )
        loss = output["pred_velocity"].square().mean()
        loss = loss + output["event_logits"].square().mean() + output["motion_logits"].square().mean()
        loss.backward()

        def grad_norm(module: torch.nn.Module) -> float:
            return float(sum(
                parameter.grad.detach().float().square().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            ).sqrt())

        self.assertGreater(grad_norm(decoder.intent_compiler), 0.0)
        self.assertGreater(grad_norm(decoder.workspace), 0.0)
        for block, contractions in zip(
            decoder.blocks, decoder.operator_contractions, strict=True
        ):
            self.assertGreater(grad_norm(block), 0.0)
            self.assertGreater(grad_norm(contractions), 0.0)
            self.assertGreater(grad_norm(block.self_out), 0.0)
            self.assertGreater(grad_norm(block.noisy_kv), 0.0)
            self.assertGreater(grad_norm(block.stage_kv), 0.0)
            self.assertGreater(grad_norm(block.low_kv), 0.0)
        self.assertEqual(
            tuple(output["refinement_probe_pred_velocity"].shape),
            (self.batch, cfg.hierarchical_mmdit_refine_steps + 1, cfg.action_horizon, cfg.physical_action_dim),
        )
        self.assertEqual(
            tuple(output["refinement_probe_block_ids"].shape),
            (self.batch, cfg.hierarchical_mmdit_refine_steps),
        )
        self.assertEqual(
            tuple(output["hierarchical_mmdit_exit_logits"].shape),
            (self.batch, cfg.hierarchical_mmdit_refine_steps),
        )
        self.assertTrue(output["hierarchical_mmdit_exit_logits"].requires_grad)
        self.assertEqual(
            tuple(output["refinement_probe_exit_candidates"].shape),
            (self.batch, cfg.hierarchical_mmdit_refine_steps),
        )
        block_usage = sum(
            output[f"hierarchical_mmdit_block_{index}_usage"]
            for index in range(cfg.hierarchical_mmdit_depth)
        )
        torch.testing.assert_close(block_usage, torch.ones(()), atol=1e-6, rtol=1e-6)
        for key in (
            "hierarchical_mmdit_action_response_arm",
            "hierarchical_mmdit_action_response_gripper",
            "hierarchical_mmdit_action_response_arm_null",
            "hierarchical_mmdit_action_response_gripper_null",
        ):
            self.assertGreaterEqual(float(output[key]), 0.0)
        for time_bin in range(3):
            for prefix in (
                "hierarchical_mmdit_action_response",
                "hierarchical_mmdit_stage_pressure",
            ):
                for quantile in ("p25", "p50", "p75"):
                    key = f"{prefix}_t{time_bin}_{quantile}"
                    self.assertIn(key, output)
                    self.assertTrue(bool(torch.isfinite(output[key])))
                    self.assertGreaterEqual(float(output[key]), 0.0)

    def test_exhaustion_thresholds_use_the_same_three_time_bins_as_diagnostics(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config)
        actual = decoder._threshold_rows(
            torch.tensor([0.0, 0.33, 0.34, 0.66, 0.67, 1.0]),
            (1.0, 2.0, 3.0),
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        )

    def test_branch_geometry_uses_unequal_norm_orthogonal_baseline(self) -> None:
        block = HierarchicalMMDiTActionDecoder(self.config).blocks[0]
        updates = []
        for index in range(5):
            update = torch.zeros(1, 1, 5)
            update[..., index] = 1.0
            updates.append(update)
        _, _, cancellation, baseline, excess, weighted_cosine = block._branch_geometry(tuple(updates))
        expected = torch.tensor(1.0 - 1.0 / (5.0 ** 0.5))
        torch.testing.assert_close(cancellation, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(baseline, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(excess, torch.zeros(()), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(weighted_cosine, torch.zeros(()), atol=1e-6, rtol=0.0)

        unequal = tuple(update * float(index + 1) for index, update in enumerate(updates))
        _, _, cancellation, baseline, excess, weighted_cosine = block._branch_geometry(unequal)
        expected_unequal = 1.0 - (sum(float(i * i) for i in range(1, 6)) ** 0.5) / 15.0
        torch.testing.assert_close(cancellation, torch.tensor(expected_unequal), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(baseline, torch.tensor(expected_unequal), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(excess, torch.zeros(()), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(weighted_cosine, torch.zeros(()), atol=1e-6, rtol=0.0)

        aligned = tuple(updates[0].clone() for _ in range(5))
        _, _, cancellation, baseline, excess, weighted_cosine = block._branch_geometry(aligned)
        torch.testing.assert_close(cancellation, torch.zeros(()), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(baseline, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(excess, -expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weighted_cosine, torch.ones(()), atol=1e-6, rtol=1e-6)

    def test_maximum_depth_recovers_the_ordinary_full_rank_projection(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        block = decoder.blocks[0]
        h = self.config.hidden_size
        projection_input = torch.randn(self.batch, self.config.action_horizon, h)
        base_gate = torch.full((self.batch,), 0.05)
        operator_cond = torch.randn(self.batch, h)
        stage_index = torch.zeros(self.batch, dtype=torch.long)
        update, metrics = block._compose_update(
            branch="noisy",
            projection_input=projection_input,
            base_projection=block.noisy_out,
            contraction=decoder.operator_contractions[0]["noisy"],
            contraction_progress=decoder.contraction_progress,
            operator_cond=operator_cond,
            stage_index=stage_index,
            base_gate=base_gate,
        )
        # Compute the V77 path independently.  Calling the block helper here
        # would let an accidental boundary rewrite make the test self-fulfilling.
        projected = block.noisy_out(projection_input)
        denominator = projected.float().square().mean(
            dim=(1, 2), keepdim=True
        ).add(1e-6).sqrt()
        expected_direction = (projected.float() / denominator).to(projected.dtype)
        torch.testing.assert_close(
            update,
            base_gate[:, None, None] * expected_direction,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            metrics["realized_scale"], metrics["host_update_rms"], atol=0.0, rtol=0.0
        )
        self.assertEqual(float(metrics["depth_ratio"]), 1.0)
        self.assertEqual(float(metrics["effective_depth"]), 16.0)
        self.assertEqual(float(metrics["boundary_identity_error"]), 0.0)
        self.assertLessEqual(float(metrics["nonexpansive_violation"]), 2e-6)

    def test_identity_sidecar_preserves_base_projection_gradients(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        block = decoder.blocks[0]
        h = self.config.hidden_size
        actual_input = torch.randn(
            self.batch, self.config.action_horizon, h, requires_grad=True
        )
        reference_input = actual_input.detach().clone().requires_grad_(True)
        reference_projection = torch.nn.Linear(h, h)
        reference_projection.load_state_dict(block.noisy_out.state_dict())
        actual_gate = torch.tensor([0.03, 0.07], requires_grad=True)
        reference_gate = actual_gate.detach().clone().requires_grad_(True)
        probe = torch.randn_like(actual_input)

        actual, _ = block._compose_update(
            branch="noisy",
            projection_input=actual_input,
            base_projection=block.noisy_out,
            contraction=decoder.operator_contractions[0]["noisy"],
            contraction_progress=decoder.contraction_progress,
            operator_cond=torch.randn(self.batch, h),
            stage_index=torch.zeros(self.batch, dtype=torch.long),
            base_gate=actual_gate,
        )
        projected = reference_projection(reference_input)
        denominator = projected.float().square().mean(
            dim=(1, 2), keepdim=True
        ).add(1e-6).sqrt()
        reference = reference_gate[:, None, None] * (
            projected.float() / denominator
        ).to(projected.dtype)
        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)

        (actual * probe).sum().backward()
        (reference * probe).sum().backward()
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            block.noisy_out.weight.grad,
            reference_projection.weight.grad,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            block.noisy_out.bias.grad,
            reference_projection.bias.grad,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            actual_gate.grad, reference_gate.grad, atol=0.0, rtol=0.0
        )
        for parameter in decoder.operator_contractions[0]["noisy"].parameters():
            if parameter.grad is not None:
                self.assertEqual(float(parameter.grad.abs().sum()), 0.0)

    def test_detached_velocity_prediction_remains_outside_the_objective(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).train()
        action = torch.randn(
            self.batch, self.config.action_horizon, self.config.hidden_size,
            requires_grad=True,
        )
        prediction = decoder._detached_velocity_prediction(action)
        self.assertFalse(prediction.requires_grad)

        main_prediction = decoder.velocity_head(decoder.action_norm(action))
        main_prediction.square().mean().backward()
        self.assertIsNotNone(action.grad)
        self.assertGreater(float(action.grad.norm()), 0.0)
        self.assertTrue(all(
            layer.weight.grad is not None and float(layer.weight.grad.norm()) > 0.0
            for layer in decoder.velocity_head.output_layers()
        ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP regression requires CUDA")
    def test_detached_velocity_prediction_preserves_amp_head_gradients(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).cuda().train()
        action = torch.randn(
            self.batch,
            self.config.action_horizon,
            self.config.hidden_size,
            device="cuda",
            requires_grad=True,
        )
        decoder.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = decoder._detached_velocity_prediction(action)
            main_prediction = decoder.velocity_head(decoder.action_norm(action))
            target = torch.randn_like(main_prediction)
            loss = (main_prediction.float() - target.float()).square().mean()
        self.assertFalse(prediction.requires_grad)
        loss.backward()
        self.assertTrue(all(
            layer.weight.grad is not None and float(layer.weight.grad.float().norm()) > 0.0
            for layer in decoder.velocity_head.output_layers()
        ))

    def test_contraction_changes_direction_without_owning_residual_amplitude(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        decoder.set_operator_contraction_training_step(10_000)
        block = decoder.blocks[0]
        contraction = decoder.operator_contractions[0]["noisy"]
        with torch.no_grad():
            contraction.depth_weight.zero_()
            contraction.depth_bias.fill_(-20.0)
        h = self.config.hidden_size
        base_gate = torch.full((self.batch,), 0.05)
        _, metrics = block._compose_update(
            branch="noisy",
            projection_input=torch.randn(
                self.batch, self.config.action_horizon, h
            ),
            base_projection=block.noisy_out,
            contraction=contraction,
            contraction_progress=decoder.contraction_progress,
            operator_cond=torch.randn(self.batch, h),
            stage_index=torch.zeros(self.batch, dtype=torch.long),
            base_gate=base_gate,
        )
        self.assertLess(
            float(metrics["contracted_rms"]),
            float(metrics["host_update_rms"]),
        )
        torch.testing.assert_close(
            metrics["realized_scale"],
            metrics["contracted_rms"],
            atol=0.0,
            rtol=0.0,
        )

    def test_host_gates_train_while_sidecar_waits_for_warmup(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).train()
        block = decoder.blocks[0]
        h = self.config.hidden_size
        output, _ = block(
            torch.randn(self.batch, self.config.action_horizon, h),
            noisy_tokens=torch.randn(self.batch, self.config.action_horizon, h),
            stage_tokens=torch.randn(
                self.batch, self.config.hierarchical_mmdit_stage_slots, h
            ),
            low_tokens=torch.randn(
                self.batch, self.config.hierarchical_mmdit_low_slots, h
            ),
            shared_cond=torch.randn(self.batch, h),
            operator_cond=torch.randn(self.batch, h),
            contractions=decoder.operator_contractions[0],
            contraction_progress=decoder.contraction_progress,
            stage_index=torch.zeros(self.batch, dtype=torch.long),
        )
        (output * torch.randn_like(output)).mean().backward()
        self.assertGreater(float(block.noisy_out.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(block.mod.weight.grad[2 * h :].abs().sum()), 0.0
        )
        for parameter in decoder.operator_contractions[0].parameters():
            if parameter.grad is not None:
                self.assertEqual(float(parameter.grad.abs().sum()), 0.0)

        decoder.zero_grad(set_to_none=True)
        decoder.set_operator_contraction_training_step(10_000)
        output, _ = block(
            torch.randn(self.batch, self.config.action_horizon, h),
            noisy_tokens=torch.randn(self.batch, self.config.action_horizon, h),
            stage_tokens=torch.randn(
                self.batch, self.config.hierarchical_mmdit_stage_slots, h
            ),
            low_tokens=torch.randn(
                self.batch, self.config.hierarchical_mmdit_low_slots, h
            ),
            shared_cond=torch.randn(self.batch, h),
            operator_cond=torch.randn(self.batch, h),
            contractions=decoder.operator_contractions[0],
            contraction_progress=decoder.contraction_progress,
            stage_index=torch.zeros(self.batch, dtype=torch.long),
        )
        (output * torch.randn_like(output)).mean().backward()
        self.assertGreater(
            float(block.mod.weight.grad[2 * h :].abs().sum()), 0.0
        )
        self.assertGreater(float(sum(
            parameter.grad.detach().float().square().sum()
            for parameter in decoder.operator_contractions[0].parameters()
            if parameter.grad is not None
        ).sqrt()), 0.0)

    def test_contraction_sidecars_do_not_own_base_block_parameters(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config)
        self.assertFalse(any(
            name.startswith("blocks.") and ".contractions." in name
            for name, _ in decoder.named_parameters()
        ))
        self.assertTrue(any(
            name.startswith("operator_contractions.")
            for name, _ in decoder.named_parameters()
        ))
        self.assertTrue(all(hasattr(block, "mod") for block in decoder.blocks))
        self.assertTrue(all(
            not hasattr(block, "residual_mod") for block in decoder.blocks
        ))

    def test_sidecar_capacity_does_not_reinitialize_v77_host(self) -> None:
        small = replace(
            self.config,
            hierarchical_mmdit_operator_rank=8,
            hierarchical_mmdit_operator_groups=8,
        )
        large = replace(
            self.config,
            hierarchical_mmdit_operator_rank=16,
            hierarchical_mmdit_operator_groups=16,
        )
        torch.manual_seed(1234)
        small_decoder = HierarchicalMMDiTActionDecoder(small)
        torch.manual_seed(1234)
        large_decoder = HierarchicalMMDiTActionDecoder(large)
        sidecar_prefixes = (
            "operator_contractions.",
            "operator_stage_identity",
            "stage_selector_control.",
            "stage_selector_query.",
            "exit_controller.",
            "operator_condition.",
            "contraction_progress",
        )
        small_state = {
            key: value
            for key, value in small_decoder.state_dict().items()
            if not key.startswith(sidecar_prefixes)
        }
        large_state = {
            key: value
            for key, value in large_decoder.state_dict().items()
            if not key.startswith(sidecar_prefixes)
        }
        self.assertEqual(small_state.keys(), large_state.keys())
        for key in small_state:
            torch.testing.assert_close(
                small_state[key], large_state[key], atol=0.0, rtol=0.0
            )

    def test_boundary_stage_selection_cannot_change_owned_block_forward(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        block = decoder.blocks[0]
        h = self.config.hidden_size
        inputs = {
            "action": torch.randn(self.batch, self.config.action_horizon, h),
            "noisy_tokens": torch.randn(self.batch, self.config.action_horizon, h),
            "stage_tokens": torch.randn(
                self.batch, self.config.hierarchical_mmdit_stage_slots, h
            ),
            "low_tokens": torch.randn(
                self.batch, self.config.hierarchical_mmdit_low_slots, h
            ),
            "shared_cond": torch.randn(self.batch, h),
            "operator_cond": torch.randn(self.batch, h),
            "contractions": decoder.operator_contractions[0],
            "contraction_progress": decoder.contraction_progress,
        }
        stage_zero, _ = block(
            **inputs,
            stage_index=torch.zeros(self.batch, dtype=torch.long),
        )
        stage_one, _ = block(
            **inputs,
            stage_index=torch.ones(self.batch, dtype=torch.long),
        )
        torch.testing.assert_close(stage_one, stage_zero, atol=0.0, rtol=0.0)

    def test_depth_path_is_continuous_nested_and_nonexpansive(self) -> None:
        writer = NestedLowRankContractionBank(
            hidden_size=16,
            condition_size=16,
            stage_count=1,
            rank=8,
            group_count=8,
            depth_logit_init=2.0,
        ).eval()
        base = torch.randn(1, 4, 16)
        depths = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        repeated = base.expand(len(depths), -1, -1).clone()
        output, metrics = writer(
            repeated,
            torch.randn(1, 16).expand(len(depths), -1).clone(),
            torch.zeros(len(depths), dtype=torch.long),
            contraction_progress=1.0,
            depth_ratio_override=depths,
        )
        torch.testing.assert_close(output[0], base[0], atol=0.0, rtol=0.0)
        output_norms = output.flatten(1).norm(dim=-1)
        self.assertTrue(bool((output_norms[:-1] >= output_norms[1:] - 2e-6).all()))
        torch.testing.assert_close(
            metrics["effective_depth_rows"], depths * 8.0, atol=1e-6, rtol=0.0
        )
        base_energy = repeated[-1].float().square().sum()
        retained_energy = output[-1].float().square().sum() / base_energy
        torch.testing.assert_close(
            retained_energy + metrics["subspace_energy_fraction_rows"][-1],
            torch.ones(()),
            atol=3e-6,
            rtol=3e-6,
        )
        nearby, _ = writer(
            base.expand(2, -1, -1).clone(),
            torch.randn(1, 16).expand(2, -1).clone(),
            torch.zeros(2, dtype=torch.long),
            depth_ratio_override=torch.tensor([0.5, 0.5001]),
        )
        self.assertLess(
            float((nearby[1] - nearby[0]).norm() / base.norm().clamp_min(1e-8)),
            1e-3,
        )
        self.assertLessEqual(float(metrics["nonexpansive_violation"]), 2e-6)
        self.assertEqual(float(metrics["nested_order_violation"]), 0.0)

    def test_basis_rescaling_cannot_change_contraction_function(self) -> None:
        writer = NestedLowRankContractionBank(
            hidden_size=16,
            condition_size=16,
            stage_count=2,
            rank=8,
            group_count=8,
            depth_logit_init=2.0,
        ).eval()
        base = torch.randn(3, 4, 16)
        condition = torch.randn(3, 16)
        stage_index = torch.tensor([0, 1, 0])
        baseline, _ = writer(
            base,
            condition,
            stage_index,
            depth_ratio_override=0.5,
        )
        with torch.no_grad():
            writer.basis_raw.mul_(100.0)
        rescaled, _ = writer(
            base,
            condition,
            stage_index,
            depth_ratio_override=0.5,
        )
        torch.testing.assert_close(rescaled, baseline, atol=2e-6, rtol=2e-6)

    def test_identity_warmup_is_exact_and_has_no_false_contraction_gradient(self) -> None:
        writer = NestedLowRankContractionBank(
            hidden_size=16,
            condition_size=16,
            stage_count=2,
            rank=8,
            group_count=8,
            depth_logit_init=2.0,
        ).train()
        base = torch.randn(3, 4, 16, requires_grad=True)
        output, metrics = writer(
            base,
            torch.randn(3, 16),
            torch.tensor([0, 1, 0]),
            contraction_progress=0.0,
        )
        torch.testing.assert_close(output, base, atol=0.0, rtol=0.0)
        self.assertEqual(float(metrics["depth_ratio"]), 1.0)
        self.assertEqual(float(metrics["effective_depth"]), 8.0)
        output.sum().backward()
        torch.testing.assert_close(base.grad, torch.ones_like(base), atol=0.0, rtol=0.0)
        for parameter in (writer.basis_raw, writer.depth_weight, writer.depth_bias):
            if parameter.grad is not None:
                self.assertEqual(float(parameter.grad.abs().sum()), 0.0)

    def test_depth_usage_cost_only_owns_depth_controls(self) -> None:
        writer = NestedLowRankContractionBank(
            hidden_size=16,
            condition_size=16,
            stage_count=1,
            rank=8,
            group_count=8,
            depth_logit_init=2.0,
        ).train()
        _, metrics = writer(
            torch.randn(2, 4, 16),
            torch.randn(2, 16),
            torch.zeros(2, dtype=torch.long),
            contraction_progress=1.0,
        )
        metrics["depth_usage_cost"].backward()
        self.assertIsNone(writer.basis_raw.grad)
        self.assertIsNotNone(writer.depth_weight.grad)
        self.assertIsNotNone(writer.depth_bias.grad)
        self.assertGreater(float(writer.depth_bias.grad.abs().sum()), 0.0)

    def test_three_refine_blocks_partition_six_local_operator_stages(self) -> None:
        cfg = replace(
            self._config(),
            hierarchical_mmdit_depth=3,
            hierarchical_mmdit_refine_steps=3,
            hierarchical_mmdit_stage_slots=6,
            hierarchical_mmdit_operator_stages=6,
        )
        cfg.validate()
        decoder = HierarchicalMMDiTActionDecoder(cfg)
        shelves = [
            decoder._fixed_stage_candidates(step, device=torch.device("cpu")).tolist()
            for step in range(3)
        ]
        self.assertEqual(shelves, [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(sorted(value for shelf in shelves for value in shelf), list(range(6)))
        torch.testing.assert_close(
            decoder._operator_stage_to_block(torch.arange(6)),
            torch.tensor([0, 0, 1, 1, 2, 2]),
        )
        self.assertEqual([block.operator_stage_count for block in decoder.blocks], [2, 2, 2])
        self.assertEqual(
            len({id(block.self_out.weight) for block in decoder.blocks}),
            3,
        )

    def test_three_four_and_five_block_depths_keep_two_local_stages_each(self) -> None:
        for depth in (3, 4, 5):
            with self.subTest(depth=depth):
                cfg = replace(
                    self._config(),
                    hierarchical_mmdit_depth=depth,
                    hierarchical_mmdit_refine_steps=depth,
                    hierarchical_mmdit_stage_slots=2 * depth,
                    hierarchical_mmdit_operator_stages=2 * depth,
                )
                cfg.validate()
                decoder = HierarchicalMMDiTActionDecoder(cfg)
                self.assertEqual(len(decoder.blocks), depth)
                self.assertEqual(
                    [block.operator_stage_count for block in decoder.blocks],
                    [2] * depth,
                )
                shelves = [
                    decoder._fixed_stage_candidates(index, device=torch.device("cpu")).tolist()
                    for index in range(depth)
                ]
                self.assertEqual(
                    shelves,
                    [[2 * index, 2 * index + 1] for index in range(depth)],
                )
                self.assertEqual(
                    len({id(block.self_out.weight) for block in decoder.blocks}),
                    depth,
                )

    def test_nondivisible_stage_partition_round_trips_block_ownership(self) -> None:
        cfg = replace(
            self._config(),
            hierarchical_mmdit_depth=3,
            hierarchical_mmdit_refine_steps=3,
            hierarchical_mmdit_stage_slots=4,
            hierarchical_mmdit_operator_stages=4,
        )
        cfg.validate()
        decoder = HierarchicalMMDiTActionDecoder(cfg)
        self.assertEqual(
            [decoder._fixed_stage_candidates(i, device=torch.device("cpu")).tolist() for i in range(3)],
            [[0], [1], [2, 3]],
        )
        torch.testing.assert_close(
            decoder._operator_stage_to_block(torch.arange(4)),
            torch.tensor([0, 1, 2, 2]),
        )

    def test_contraction_schedule_has_exact_identity_warmup_then_continuous_budget(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config)
        self.assertEqual(decoder.set_operator_contraction_training_step(199), 0.0)
        self.assertEqual(float(decoder.contraction_progress), 0.0)
        self.assertAlmostEqual(
            decoder.set_operator_contraction_training_step(950), 0.5
        )
        self.assertEqual(float(decoder.contraction_progress), 0.5)
        restored = HierarchicalMMDiTActionDecoder(self.config)
        restored.load_state_dict(decoder.state_dict())
        self.assertEqual(restored._contraction_progress_value, 0.5)
        self.assertEqual(decoder.set_operator_contraction_training_step(5000), 1.0)
        self.assertEqual(float(decoder.contraction_progress), 1.0)

    def test_noisy_condition_has_no_external_amplitude_gate(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        block = decoder.blocks[0]
        h = self.config.hidden_size
        _, metrics = block(
            torch.randn(self.batch, self.config.action_horizon, h),
            noisy_tokens=torch.randn(self.batch, self.config.action_horizon, h),
            stage_tokens=torch.randn(self.batch, self.config.hierarchical_mmdit_stage_slots, h),
            low_tokens=torch.randn(self.batch, self.config.hierarchical_mmdit_low_slots, h),
            shared_cond=torch.randn(self.batch, h),
            operator_cond=torch.randn(self.batch, h),
            contractions=decoder.operator_contractions[0],
            contraction_progress=decoder.contraction_progress,
            stage_index=torch.zeros(self.batch, dtype=torch.long),
        )
        self.assertNotIn("action_noisy_external_modulation_mean", metrics)

    def test_prepared_contraction_factors_preserve_block_output(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        decoder.set_operator_contraction_training_step(10_000)
        block = decoder.blocks[0]
        h = self.config.hidden_size
        inputs = {
            "action": torch.randn(self.batch, self.config.action_horizon, h),
            "noisy_tokens": torch.randn(self.batch, self.config.action_horizon, h),
            "stage_tokens": torch.randn(
                self.batch, self.config.hierarchical_mmdit_stage_slots, h
            ),
            "low_tokens": torch.randn(
                self.batch, self.config.hierarchical_mmdit_low_slots, h
            ),
            "shared_cond": torch.randn(self.batch, h),
            "operator_cond": torch.randn(self.batch, h),
            "contractions": decoder.operator_contractions[0],
            "contraction_progress": decoder.contraction_progress,
            "stage_index": torch.tensor([0, 1]),
        }
        uncached, _ = block(**inputs)
        cached, _ = block(
            **inputs,
            contraction_factors=decoder.prepare_contraction_factors()[0],
        )
        torch.testing.assert_close(cached, uncached, atol=2e-6, rtol=2e-6)

    def test_step_condition_reaches_full_rank_path_at_identity_boundary(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        block = decoder.blocks[0]
        with torch.no_grad():
            block.mod.weight[: 2 * self.config.hidden_size].normal_(
                mean=0.0, std=0.05
            )
        h = self.config.hidden_size
        common = {
            "action": torch.randn(self.batch, self.config.action_horizon, h),
            "noisy_tokens": torch.randn(self.batch, self.config.action_horizon, h),
            "stage_tokens": torch.randn(
                self.batch, self.config.hierarchical_mmdit_stage_slots, h
            ),
            "low_tokens": torch.randn(
                self.batch, self.config.hierarchical_mmdit_low_slots, h
            ),
            "operator_cond": torch.randn(self.batch, h),
            "contractions": decoder.operator_contractions[0],
            "contraction_progress": decoder.contraction_progress,
            "stage_index": torch.zeros(self.batch, dtype=torch.long),
        }
        baseline, _ = block(
            **common,
            shared_cond=torch.zeros(self.batch, h),
        )
        conditioned, _ = block(
            **common,
            shared_cond=torch.randn(self.batch, h),
        )
        self.assertFalse(torch.allclose(conditioned, baseline))

    def test_stage_bank_has_no_additive_or_independent_mask_shortcut(self) -> None:
        writer = NestedLowRankContractionBank(
            hidden_size=16,
            condition_size=16,
            stage_count=2,
            rank=8,
            group_count=8,
            depth_logit_init=2.0,
        ).eval()
        self.assertFalse(hasattr(writer, "channel_coefficient"))
        self.assertFalse(hasattr(writer, "mask_weight"))
        base = torch.randn(1, 4, 16).expand(2, -1, -1).clone()
        condition = torch.randn(1, 16).expand(2, -1).clone()
        output, _ = writer(
            base,
            condition,
            torch.tensor([0, 1]),
            depth_ratio_override=0.0,
        )
        self.assertFalse(torch.allclose(output[1], output[0]))

    def test_stage_probability_is_backward_surrogate_not_forward_amplitude_mix(self) -> None:
        writer = NestedLowRankContractionBank(
            hidden_size=16,
            condition_size=16,
            stage_count=2,
            rank=8,
            group_count=8,
            depth_logit_init=2.0,
        )
        base = torch.randn(2, 4, 16)
        condition = torch.randn(2, 16)
        selected = torch.zeros(2, dtype=torch.long)
        writer.eval()
        expected, _ = writer(
            base, condition, selected, depth_ratio_override=0.5
        )
        writer.train()
        actual, _ = writer(
            base,
            condition,
            selected,
            stage_candidates=torch.tensor([[0, 1], [0, 1]]),
            stage_probabilities=torch.tensor([[0.01, 0.99], [0.25, 0.75]]),
            depth_ratio_override=0.5,
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_random_dwell_is_monotonic_and_covers_all_blocks_without_prefix_exit(self) -> None:
        cfg = replace(
            self._config(),
            hierarchical_mmdit_refine_steps=6,
            hierarchical_mmdit_schedule_mode="random_dwell",
            hierarchical_mmdit_random_prefix_probability=0.0,
        )
        cfg.validate()
        decoder = HierarchicalMMDiTActionDecoder(cfg)
        for _ in range(32):
            schedule, active, active_count = decoder._random_dwell_schedule(device=torch.device("cpu"))
            self.assertEqual(active_count, int(active.sum()))
            self.assertGreaterEqual(active_count, cfg.hierarchical_mmdit_depth)
            self.assertLessEqual(active_count, cfg.hierarchical_mmdit_refine_steps)
            self.assertTrue(bool(active[:active_count].all()))
            self.assertFalse(bool(active[active_count:].any()))
            active_schedule = schedule[:active_count]
            self.assertTrue(bool((active_schedule[1:] >= active_schedule[:-1]).all()))
            self.assertEqual(
                set(active_schedule.tolist()),
                set(range(cfg.hierarchical_mmdit_depth)),
            )

    def test_adaptive_block_owners_only_select_their_local_semantic_stages(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        h = self.config.hidden_size
        selected, candidates, probabilities, _, _, _, _ = decoder._select_adaptive_stages(
            block_index=torch.tensor([0, 1]),
            stage_content=torch.randn(
                self.batch, self.config.hierarchical_mmdit_stage_slots, h
            ),
            global_intent=torch.randn(self.batch, h),
            time_state=torch.randn(self.batch, h),
            step_state=torch.randn(self.batch, h),
            action=torch.randn(self.batch, self.config.action_horizon, h),
            control_state=torch.randn(self.batch, 4),
        )
        self.assertIn(int(selected[0]), (0, 1))
        self.assertIn(int(selected[1]), (2, 3))
        torch.testing.assert_close(candidates[:, 0], selected)
        torch.testing.assert_close(probabilities, torch.ones_like(probabilities))

    def test_stage_selector_query_reads_live_refinement_state(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        h = self.config.hidden_size
        shared = {
            "block_index": 0,
            "stage_content": torch.randn(
                self.batch, self.config.hierarchical_mmdit_operator_stages, h
            ),
            "global_intent": torch.randn(self.batch, h),
            "time_state": torch.randn(self.batch, h),
        }
        step = torch.randn(self.batch, h)
        action = torch.randn(self.batch, self.config.action_horizon, h)
        control = torch.randn(self.batch, 4)
        *_, base_query, _ = decoder._select_owned_stage(
            **shared,
            step_state=step,
            action=action,
            control_state=control,
        )
        *_, step_query, _ = decoder._select_owned_stage(
            **shared,
            step_state=step + torch.randn_like(step),
            action=action,
            control_state=control,
        )
        *_, action_query, _ = decoder._select_owned_stage(
            **shared,
            step_state=step,
            action=action + torch.randn_like(action),
            control_state=control,
        )
        *_, control_query, _ = decoder._select_owned_stage(
            **shared,
            step_state=step,
            action=action,
            control_state=control + torch.randn_like(control),
        )
        self.assertFalse(torch.allclose(base_query, step_query))
        self.assertFalse(torch.allclose(base_query, action_query))
        self.assertFalse(torch.allclose(base_query, control_query))

    def test_stage_selector_inputs_are_read_only_but_selector_parameters_train(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).train()
        decoder.set_operator_contraction_training_step(10_000)
        h = self.config.hidden_size
        stage_content = torch.randn(
            self.batch,
            self.config.hierarchical_mmdit_operator_stages,
            h,
            requires_grad=True,
        )
        global_intent = torch.randn(self.batch, h, requires_grad=True)
        time_state = torch.randn(self.batch, h, requires_grad=True)
        step_state = torch.randn(self.batch, h, requires_grad=True)
        action = torch.randn(
            self.batch, self.config.action_horizon, h, requires_grad=True
        )
        control_state = torch.randn(self.batch, 4, requires_grad=True)
        _, _, probabilities, _, _, _, _ = decoder._select_owned_stage(
            block_index=0,
            stage_content=stage_content,
            global_intent=global_intent,
            time_state=time_state,
            step_state=step_state,
            action=action,
            control_state=control_state,
        )
        probabilities.square().sum().backward()
        for value in (
            stage_content, global_intent, time_state, step_state, action, control_state,
        ):
            self.assertIsNone(value.grad)
        selector_grad = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in decoder.stage_selector_parameters()
            if parameter.grad is not None
        ).sqrt()
        self.assertGreater(float(selector_grad), 0.0)

    def test_stage_selector_exploration_anneals_without_warmup_rng_drift(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).train()
        h = self.config.hidden_size
        inputs = {
            "block_index": 0,
            "stage_content": torch.randn(
                64, self.config.hierarchical_mmdit_operator_stages, h
            ),
            "global_intent": torch.randn(64, h),
            "time_state": torch.randn(64, h),
            "step_state": torch.randn(64, h),
            "action": torch.randn(64, self.config.action_horizon, h),
            "control_state": torch.zeros(64, 4),
        }
        decoder.set_operator_contraction_training_step(0)
        before = torch.random.get_rng_state().clone()
        *_, warmup_exploration = decoder._select_owned_stage(**inputs)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        torch.testing.assert_close(warmup_exploration, torch.ones_like(warmup_exploration))

        warmup = self.config.hierarchical_mmdit_operator_contraction_warmup_steps
        transition = self.config.hierarchical_mmdit_operator_contraction_transition_steps
        decoder.set_operator_contraction_training_step(warmup + transition // 2)
        selected, _, probabilities, _, _, _, exploration = decoder._select_owned_stage(
            **inputs
        )
        self.assertEqual(set(selected.tolist()), {0, 1})
        torch.testing.assert_close(
            exploration,
            torch.full_like(exploration, 0.5),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertGreaterEqual(float(probabilities.min()), 0.25 - 1e-6)

        decoder.set_operator_contraction_training_step(warmup + transition)
        *_, learned_exploration = decoder._select_owned_stage(**inputs)
        torch.testing.assert_close(
            learned_exploration, torch.zeros_like(learned_exploration)
        )

    def test_oracle_exit_supervision_uses_earliest_near_best_boundary(self) -> None:
        exit_logits = torch.tensor(
            [[-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
            requires_grad=True,
        )
        candidate_error = torch.tensor(
            [[0.80, 0.50, 0.49], [0.60, 0.40, 0.20]],
            requires_grad=True,
        )
        initial_error = torch.ones(2, requires_grad=True)
        result = _oracle_exit_supervision(
            exit_logits=exit_logits,
            candidate_error=candidate_error,
            initial_error=initial_error,
            candidate_mask=torch.ones(2, 3, dtype=torch.bool),
            relative_tolerance=0.02,
        )
        torch.testing.assert_close(result["target_depth"], torch.tensor(2.5))
        torch.testing.assert_close(result["predicted_depth"], torch.tensor(2.5))
        torch.testing.assert_close(result["depth_accuracy"], torch.tensor(1.0))
        result["loss"].backward()
        self.assertIsNotNone(exit_logits.grad)
        self.assertGreater(float(exit_logits.grad.abs().sum()), 0.0)
        self.assertIsNone(candidate_error.grad)
        self.assertIsNone(initial_error.grad)

    def test_learned_exhaustion_does_not_require_threshold_heuristics(self) -> None:
        cfg = replace(self._config(), hierarchical_mmdit_exhaustion_mode="learned")
        cfg.validate()

    def test_oracle_exit_depth_counts_block_boundaries_not_refine_steps(self) -> None:
        exit_logits = torch.tensor([
            [-10.0, -10.0, -10.0, 10.0, -10.0, -10.0]
        ], requires_grad=True)
        candidate_error = torch.tensor([[9.0, 0.8, 9.0, 0.2, 9.0, 0.3]])
        candidate_mask = torch.tensor([[False, True, False, True, False, True]])
        result = _oracle_exit_supervision(
            exit_logits=exit_logits,
            candidate_error=candidate_error,
            initial_error=torch.tensor([1.0]),
            candidate_mask=candidate_mask,
            relative_tolerance=0.0,
        )
        torch.testing.assert_close(result["target_depth"], torch.tensor(2.0))
        torch.testing.assert_close(result["predicted_depth"], torch.tensor(2.0))

    def test_exit_loss_updates_only_exit_controller(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).train()
        h = self.config.hidden_size
        inputs = {
            "global_intent": torch.randn(self.batch, h, requires_grad=True),
            "time_state": torch.randn(self.batch, h, requires_grad=True),
            "step_state": torch.randn(self.batch, h, requires_grad=True),
            "action": torch.randn(
                self.batch, self.config.action_horizon, h, requires_grad=True
            ),
            "control_state": torch.randn(self.batch, 4, requires_grad=True),
        }
        query = decoder._controller_query(**inputs)
        logits = decoder._exit_logit(query)
        F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits)
        ).backward()
        for value in inputs.values():
            self.assertIsNone(value.grad)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in decoder.stage_selector_parameters()
        ))
        exit_grad = sum(
            parameter.grad.detach().float().square().sum()
            for parameter in decoder.exit_controller_parameters()
            if parameter.grad is not None
        ).sqrt()
        self.assertGreater(float(exit_grad), 0.0)

    def test_mixed_adaptive_dispatch_preserves_sample_order_and_block_ownership(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        h = self.config.hidden_size
        action = torch.randn(self.batch, self.config.action_horizon, h)
        noisy = torch.randn_like(action)
        stage = torch.randn(
            self.batch, self.config.hierarchical_mmdit_stage_slots, h
        )
        low = torch.randn(
            self.batch, self.config.hierarchical_mmdit_low_slots, h
        )
        shared = torch.randn(self.batch, h)
        operator = torch.randn(self.batch, h)
        block_index = torch.tensor([1, 0])
        global_stage = torch.tensor([2, 0])
        factors = decoder.prepare_contraction_factors()
        mixed, mixed_metrics = decoder._run_owned_blocks(
            action,
            block_index=block_index,
            noisy_tokens=noisy,
            stage_tokens=stage,
            low_tokens=low,
            shared_cond=shared,
            operator_cond=operator,
            stage_index=global_stage,
            stage_candidates=global_stage[:, None],
            stage_probabilities=torch.ones(self.batch, 1),
            contraction_factors=factors,
        )
        for row, owner in enumerate((1, 0)):
            start, _ = decoder._stage_bounds(owner)
            expected, expected_metrics = decoder.blocks[owner](
                action[row : row + 1],
                noisy_tokens=noisy[row : row + 1],
                stage_tokens=stage[row : row + 1],
                low_tokens=low[row : row + 1],
                shared_cond=shared[row : row + 1],
                operator_cond=operator[row : row + 1],
                contractions=decoder.operator_contractions[owner],
                contraction_progress=decoder.contraction_progress,
                stage_index=(global_stage[row : row + 1] - start),
                contraction_factors=factors[owner],
                low_role_ids=decoder.workspace.low_slot_role_ids,
                low_role_names=decoder.workspace.memory_bank.ROLE_NAMES,
            )
            torch.testing.assert_close(mixed[row], expected[0], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(
                mixed_metrics["action_noisy_update_fraction_rows"][row],
                expected_metrics["action_noisy_update_fraction_rows"][0],
                atol=2e-6,
                rtol=2e-6,
            )

    def test_adaptive_execution_rejects_uncalibrated_zero_thresholds(self) -> None:
        cfg = replace(self._config(), hierarchical_mmdit_exhaustion_mode="shadow")
        with self.assertRaisesRegex(ValueError, "calibrated positive action-response"):
            cfg.validate()

    def test_shadow_exhaustion_exits_after_confirmation_without_changing_main_path(self) -> None:
        cfg = replace(
            self._config(),
            hierarchical_mmdit_refine_steps=4,
            hierarchical_mmdit_exhaustion_mode="shadow",
            hierarchical_mmdit_action_response_thresholds=(1e6, 1e6, 1e6),
            hierarchical_mmdit_stage_pressure_thresholds=(1e6, 1e6, 1e6),
            hierarchical_mmdit_exhaustion_confirm_steps=2,
        )
        cfg.validate()
        h = cfg.hidden_size
        decoder = HierarchicalMMDiTActionDecoder(cfg).eval()
        layer_contracts = [{
            key: torch.randn(self.batch, 3, h)
            for key in LayeredV37StyleResidualActionFlowDenoiser._LAYER_KEYS
        } for _ in range(cfg.depth)]
        with torch.no_grad():
            output = decoder(
                noisy_physical=torch.randn(
                    self.batch, cfg.action_horizon, cfg.physical_action_dim
                ),
                time=torch.rand(self.batch),
                trajectory_tokens=torch.randn(self.batch, cfg.action_horizon, h),
                trajectory_workspace_tokens=torch.randn(self.batch, cfg.action_horizon, h),
                rollout_tokens=torch.randn(self.batch, 8, h),
                transition_memory=[torch.randn(self.batch, 6, h)],
                event_evidence=torch.randn(self.batch, cfg.action_horizon, 3),
                state_memory=[torch.randn(self.batch, 2, h)],
                intent_memory={
                    name: torch.randn(self.batch, 2, h)
                    for name in PolicyConditionOrganizer._INTENT_SOURCE_NAMES
                },
                layer_contracts=layer_contracts,
            )
        self.assertEqual(float(output["hierarchical_mmdit_executed_steps"]), 4.0)
        self.assertEqual(float(output["hierarchical_mmdit_shadow_executed_steps"]), 2.0)
        self.assertEqual(float(output["hierarchical_mmdit_shadow_early_exit_rate"]), 1.0)
        self.assertEqual(float(output["hierarchical_mmdit_shadow_unresolved_rate"]), 0.0)
        self.assertEqual(float(output["hierarchical_mmdit_shadow_budget_exhausted_rate"]), 0.0)
        self.assertEqual(
            tuple(output["refinement_shadow_probe_pred_velocity"].shape),
            (self.batch, cfg.hierarchical_mmdit_refine_steps + 1, cfg.action_horizon, cfg.physical_action_dim),
        )

    def test_optimizer_keeps_full_rank_blocks_regular_and_contraction_specialized(self) -> None:
        system = V39PolicySystem(self._config())
        trainer = V39PolicyTrainerConfig(training_stage="policy")
        groups = _optimizer_groups(system, trainer)
        owner_by_parameter: dict[int, dict[str, object]] = {}
        for group in groups:
            for parameter in group["params"]:
                identity = id(parameter)
                self.assertNotIn(identity, owner_by_parameter)
                owner_by_parameter[identity] = group
        decoder = system.planner.hierarchical_mmdit_action_decoder
        self.assertIsNotNone(decoder)
        assert decoder is not None
        factor_parameters = decoder.factor_parameters()
        self.assertGreater(len(factor_parameters), 0)
        for parameter in factor_parameters:
            owner = owner_by_parameter[id(parameter)]
            self.assertEqual(float(owner.get("weight_decay", trainer.weight_decay)), 0.0)
            self.assertIn("contraction_basis_no_decay", str(owner["name"]))
            self.assertAlmostEqual(
                float(owner["lr"]),
                trainer.lr
                * trainer.latent_cvae_action_decoder_lr_scale
                * trainer.hierarchical_mmdit_contraction_lr_scale,
            )
        base_parameters = decoder.scale_invariant_base_parameters()
        self.assertEqual(base_parameters, ())
        regular_owner = owner_by_parameter[id(decoder.blocks[0].self_out.weight)]
        self.assertEqual(
            float(regular_owner.get("weight_decay", trainer.weight_decay)),
            float(trainer.weight_decay),
        )
        self.assertAlmostEqual(
            float(regular_owner["lr"]),
            trainer.lr * trainer.latent_cvae_action_decoder_lr_scale,
        )
        for parameter in decoder.contraction_control_parameters():
            owner = owner_by_parameter[id(parameter)]
            self.assertEqual(float(owner.get("weight_decay", trainer.weight_decay)), 0.0)
            self.assertIn("contraction_depth_no_decay", str(owner["name"]))
            self.assertAlmostEqual(
                float(owner["lr"]),
                trainer.lr
                * trainer.latent_cvae_action_decoder_lr_scale
                * trainer.hierarchical_mmdit_contraction_lr_scale,
            )
        mod_owner = owner_by_parameter[id(decoder.blocks[0].mod.weight)]
        self.assertIs(mod_owner, regular_owner)
        self.assertFalse(any(
            str(group["name"]).endswith("residual_control") for group in groups
        ))


if __name__ == "__main__":
    unittest.main()
