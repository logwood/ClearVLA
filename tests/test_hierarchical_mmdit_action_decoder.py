from __future__ import annotations

import inspect
import unittest

import torch

from clearvla.experiments.observed_state_lab.policy_v39 import (
    ActionOnlyPhysicalVelocityHead,
    ConditionNeutralActionInitializer,
    HierarchicalMMDiTActionDecoder,
    HierarchicalEvidenceWorkspace,
    IntentContractCompiler,
    LayeredV37StyleResidualActionFlowDenoiser,
    OwnedEvidenceMemoryBank,
    OwnedHierarchicalActionBlock,
    PolicyConditionOrganizer,
    V39PolicyConfig,
)
from clearvla.policy.decoder import (
    ActionOnlyPhysicalVelocityHead as PackagedActionOnlyPhysicalVelocityHead,
    ConditionNeutralActionInitializer as PackagedConditionNeutralActionInitializer,
    HierarchicalMMDiTActionDecoder as PackagedHierarchicalMMDiTActionDecoder,
    OwnedHierarchicalActionBlock as PackagedOwnedHierarchicalActionBlock,
)
from clearvla.policy.intent import (
    IntentContractCompiler as PackagedIntentContractCompiler,
    PolicyConditionOrganizer as PackagedPolicyConditionOrganizer,
)
from clearvla.policy.evidence import (
    HierarchicalEvidenceWorkspace as PackagedHierarchicalEvidenceWorkspace,
    OwnedEvidenceMemoryBank as PackagedOwnedEvidenceMemoryBank,
)


class HierarchicalMMDiTActionDecoderTest(unittest.TestCase):
    def test_intent_legacy_imports_are_packaged_classes(self) -> None:
        self.assertIs(IntentContractCompiler, PackagedIntentContractCompiler)
        self.assertIs(PolicyConditionOrganizer, PackagedPolicyConditionOrganizer)

    def test_evidence_legacy_imports_are_packaged_classes(self) -> None:
        self.assertIs(HierarchicalEvidenceWorkspace, PackagedHierarchicalEvidenceWorkspace)
        self.assertIs(OwnedEvidenceMemoryBank, PackagedOwnedEvidenceMemoryBank)

    def test_decoder_legacy_imports_are_packaged_classes(self) -> None:
        self.assertIs(ActionOnlyPhysicalVelocityHead, PackagedActionOnlyPhysicalVelocityHead)
        self.assertIs(ConditionNeutralActionInitializer, PackagedConditionNeutralActionInitializer)
        self.assertIs(HierarchicalMMDiTActionDecoder, PackagedHierarchicalMMDiTActionDecoder)
        self.assertIs(OwnedHierarchicalActionBlock, PackagedOwnedHierarchicalActionBlock)

    @staticmethod
    def _config() -> V39PolicyConfig:
        return V39PolicyConfig(
            hidden_size=32,
            num_heads=4,
            depth=2,
            midcut_layer=1,
            action_horizon=4,
            dropout=0.0,
            final_action_decoder="hierarchical_mmdit_action",
            layer_contract_adapters=1,
            hierarchical_mmdit_depth=2,
            hierarchical_mmdit_refine_steps=2,
            hierarchical_mmdit_low_slots=5,
            hierarchical_mmdit_stage_slots=3,
            hierarchical_mmdit_ffn_expansion=2.0,
        )

    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = self._config()
        self.batch = 2

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
        self.assertEqual(cfg.hierarchical_mmdit_architecture_version, "serial_owned_rms_v3")
        self.assertNotIn("noisy_market_bias", dict(decoder.named_parameters()))
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
        for branch in ("self", "noisy", "stage", "low", "ffn"):
            self.assertLessEqual(
                abs(float(output[f"hierarchical_mmdit_action_{branch}_gate"])),
                cfg.hierarchical_mmdit_residual_scale_max + 1e-6,
            )
            torch.testing.assert_close(
                output[f"hierarchical_mmdit_action_{branch}_normalized_rms"],
                torch.ones(()),
                atol=5e-4,
                rtol=5e-4,
            )
            self.assertLess(
                float(output[f"hierarchical_mmdit_action_{branch}_gate_scale_error"]),
                2e-6,
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
        self.assertGreater(grad_norm(decoder.blocks), 0.0)
        for block in decoder.blocks:
            self.assertGreater(grad_norm(block.noisy_kv), 0.0)
            self.assertGreater(grad_norm(block.stage_kv), 0.0)
            self.assertGreater(grad_norm(block.low_kv), 0.0)

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

    def test_projection_scale_cannot_bypass_any_residual_gate(self) -> None:
        decoder = HierarchicalMMDiTActionDecoder(self.config).eval()
        block = decoder.blocks[0]
        h = self.config.hidden_size
        inputs = {
            "action": torch.randn(self.batch, self.config.action_horizon, h),
            "noisy_tokens": torch.randn(self.batch, self.config.action_horizon, h),
            "stage_tokens": torch.randn(self.batch, self.config.hierarchical_mmdit_stage_slots, h),
            "low_tokens": torch.randn(self.batch, self.config.hierarchical_mmdit_low_slots, h),
            "global_cond": torch.randn(self.batch, h),
        }
        state = {name: value.detach().clone() for name, value in block.state_dict().items()}
        baseline_action, baseline_metrics = block(**inputs)
        scaled_modules = {
            "self": block.self_out,
            "noisy": block.noisy_out,
            "stage": block.stage_out,
            "low": block.low_out,
            "ffn": block.ffn.net[-1],
        }
        for branch, module in scaled_modules.items():
            block.load_state_dict(state)
            with torch.no_grad():
                module.weight.mul_(100.0)
                if module.bias is not None:
                    module.bias.mul_(100.0)
            scaled_action, scaled_metrics = block(**inputs)
            torch.testing.assert_close(scaled_action, baseline_action, atol=5e-4, rtol=5e-4)
            torch.testing.assert_close(
                scaled_metrics[f"action_{branch}_realized_scale"],
                baseline_metrics[f"action_{branch}_realized_scale"],
                atol=2e-6,
                rtol=2e-5,
            )
            self.assertGreater(
                float(scaled_metrics[f"action_{branch}_projected_rms"]),
                50.0 * float(baseline_metrics[f"action_{branch}_projected_rms"]),
            )
            torch.testing.assert_close(
                scaled_metrics[f"action_{branch}_normalized_rms"],
                torch.ones(()),
                atol=5e-4,
                rtol=5e-4,
            )
        block.load_state_dict(state)

    def test_noisy_value_gate_owns_post_normalization_amplitude(self) -> None:
        block = HierarchicalMMDiTActionDecoder(self.config).blocks[0].eval()
        h = self.config.hidden_size
        global_cond = torch.randn(self.batch, h)
        value_gate = torch.tensor([0.25, 0.75])
        _, metrics = block(
            torch.randn(self.batch, self.config.action_horizon, h),
            noisy_tokens=torch.randn(self.batch, self.config.action_horizon, h),
            stage_tokens=torch.randn(self.batch, self.config.hierarchical_mmdit_stage_slots, h),
            low_tokens=torch.randn(self.batch, self.config.hierarchical_mmdit_low_slots, h),
            global_cond=global_cond,
            noisy_value_gate=value_gate,
        )
        mod = block.mod(block.global_norm(global_cond))
        gates = block.scale_max * torch.tanh(mod[:, 2 * h:])
        expected = (gates[:, 1].abs() * value_gate).mean()
        torch.testing.assert_close(
            metrics["action_noisy_realized_scale"], expected, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            metrics["action_noisy_gate_scale_error"], torch.zeros(()), atol=2e-6, rtol=0.0
        )


if __name__ == "__main__":
    unittest.main()
