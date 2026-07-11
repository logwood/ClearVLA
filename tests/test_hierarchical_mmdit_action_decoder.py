from __future__ import annotations

import inspect
import unittest

import torch

from clearvla.experiments.observed_state_lab.policy_v39 import (
    HierarchicalMMDiTActionDecoder,
    HierarchicalEvidenceWorkspace,
    IntentContractCompiler,
    LayeredV37StyleResidualActionFlowDenoiser,
    OwnedEvidenceMemoryBank,
    PolicyConditionOrganizer,
    V39PolicyConfig,
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


if __name__ == "__main__":
    unittest.main()
