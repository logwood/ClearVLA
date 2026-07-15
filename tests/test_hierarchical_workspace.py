from __future__ import annotations

import unittest

import torch

from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.evidence import HierarchicalEvidenceWorkspace
from clearvla.policy.legacy.cvae import LatentCVAEMMDiTBlock


class HierarchicalWorkspaceTest(unittest.TestCase):
    @staticmethod
    def _config() -> V39PolicyConfig:
        return V39PolicyConfig(
            hidden_size=32,
            num_heads=4,
            depth=2,
            action_horizon=4,
            dropout=0.0,
            final_action_decoder="adaptive_recurrent_cvae_action",
            latent_cvae_mmdit_decoder=1,
            latent_cvae_horizon_tokens=4,
            latent_cvae_hierarchical_workspace=1,
            latent_cvae_stage_slots=3,
            latent_cvae_workspace_noisy_query=0,
            latent_cvae_workspace_time_state=0,
            latent_cvae_workspace_controller=0,
            latent_cvae_workspace_progress_value=0,
            adaptive_cvae_route_time_query=0,
            adaptive_cvae_progress_memory=0,
            adaptive_cvae_layer_routing=0,
            adaptive_cvae_context_capsules=0,
            adaptive_cvae_refine_steps=2,
        )

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = self._config()
        self.workspace = HierarchicalEvidenceWorkspace(self.config).eval()
        self.batch = 2
        self.primary = torch.randn(self.batch, self.config.hidden_size)

    def _prepared(self, token_count: int):
        return self.workspace.prepare_evidence(
            {"trajectory": torch.randn(self.batch, token_count, self.config.hidden_size)},
            batch_size=self.batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_stage_cannot_enter_low_value_when_selection_is_degenerate(self) -> None:
        prepared = self._prepared(token_count=1)
        stage_a = torch.randn(
            self.batch, self.config.latent_cvae_stage_slots, self.config.hidden_size,
            requires_grad=True,
        )
        stage_b = torch.randn_like(stage_a)

        low_a, _, _, low_bias_a, _, metrics = self.workspace.step(
            prepared_evidence=prepared,
            stage_content=stage_a,
            primary_cond=self.primary,
            step_index=0,
        )
        low_b, _, _, low_bias_b, _, _ = self.workspace.step(
            prepared_evidence=prepared,
            stage_content=stage_b,
            primary_cond=self.primary,
            step_index=0,
        )

        # With one evidence value, every selector has weight one. Any remaining
        # stage dependence would therefore be a forbidden stage -> low-value path.
        torch.testing.assert_close(low_a, low_b, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(low_bias_a, low_bias_b, atol=1e-7, rtol=0.0)
        probe = torch.linspace(-1.0, 1.0, self.config.hidden_size).reshape(1, 1, -1)
        stage_grad = torch.autograd.grad((low_a * probe).sum(), stage_a, allow_unused=True)[0]
        if stage_grad is not None:
            torch.testing.assert_close(stage_grad, torch.zeros_like(stage_grad), atol=1e-7, rtol=0.0)
        self.assertIn("hierarchical_stage_role_norm", metrics)
        self.assertIn("hierarchical_stage_content_norm", metrics)
        self.assertIn("hierarchical_stage_role_content_cosine", metrics)

    def test_stage_still_controls_multi_value_selection(self) -> None:
        prepared = self._prepared(token_count=3)
        stage = torch.randn(
            self.batch, self.config.latent_cvae_stage_slots, self.config.hidden_size,
            requires_grad=True,
        )
        low, _, _, _, _, _ = self.workspace.step(
            prepared_evidence=prepared,
            stage_content=stage,
            primary_cond=self.primary,
            step_index=1,
        )
        probe = torch.linspace(-1.0, 1.0, self.config.hidden_size).reshape(1, 1, -1)
        stage_grad = torch.autograd.grad((low * probe).sum(), stage)[0]
        self.assertGreater(float(stage_grad.abs().max()), 1e-9)

    def test_stage_role_is_persistent_and_content_is_recurrent(self) -> None:
        prepared = self._prepared(token_count=3)
        role_before = self.workspace.stage_role.detach().clone()
        content = self.workspace.init_stage(self.primary)
        _, content_next, stage_tokens, low_bias, stage_bias, metrics = self.workspace.step(
            prepared_evidence=prepared,
            stage_content=content,
            primary_cond=self.primary,
            step_index=0,
        )
        torch.testing.assert_close(self.workspace.stage_role.detach(), role_before)
        self.assertEqual(tuple(content_next.shape), tuple(content.shape))
        self.assertEqual(tuple(stage_tokens.shape), tuple(content.shape))
        torch.testing.assert_close(low_bias.exp(), torch.ones_like(low_bias), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(stage_bias.exp(), torch.full_like(stage_bias, 0.1), atol=1e-6, rtol=1e-6)
        self.assertGreater(float(metrics["hierarchical_stage_update_norm"]), 0.0)
        self.assertGreater(float(metrics["hierarchical_stage_role_diversity"]), 0.0)

    def test_promotion_projection_scale_cannot_bypass_manager_gate(self) -> None:
        prepared = self._prepared(token_count=3)
        content = self.workspace.init_stage(self.primary)
        baseline = self.workspace.step(
            prepared_evidence=prepared,
            stage_content=content,
            primary_cond=self.primary,
            step_index=0,
        )
        state = {
            name: value.detach().clone()
            for name, value in self.workspace.stage_promote_out.state_dict().items()
        }
        with torch.no_grad():
            self.workspace.stage_promote_out.weight.mul_(100.0)
            self.workspace.stage_promote_out.bias.mul_(100.0)
        scaled = self.workspace.step(
            prepared_evidence=prepared,
            stage_content=content,
            primary_cond=self.primary,
            step_index=0,
        )
        torch.testing.assert_close(scaled[1], baseline[1], atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(scaled[2], baseline[2], atol=5e-4, rtol=5e-4)
        baseline_metrics = baseline[-1]
        scaled_metrics = scaled[-1]
        self.assertGreater(
            float(scaled_metrics["hierarchical_stage_promoted_norm"]),
            50.0 * float(baseline_metrics["hierarchical_stage_promoted_norm"]),
        )
        self.assertGreater(
            float(scaled_metrics["hierarchical_stage_promoted_projected_rms"]),
            50.0 * float(baseline_metrics["hierarchical_stage_promoted_projected_rms"]),
        )
        torch.testing.assert_close(
            scaled_metrics["hierarchical_stage_promoted_normalized_rms"],
            baseline_metrics["hierarchical_stage_promoted_normalized_rms"],
            atol=5e-4,
            rtol=5e-4,
        )
        torch.testing.assert_close(
            scaled_metrics["hierarchical_stage_promoted_realized_scale"],
            baseline_metrics["hierarchical_stage_promoted_realized_scale"],
            atol=5e-4,
            rtol=5e-4,
        )
        self.assertLess(
            float(scaled_metrics["hierarchical_stage_promote_gate_scale_error"]), 2e-5
        )
        self.workspace.stage_promote_out.load_state_dict(state)

    def test_mmdit_condition_groups_have_length_fair_prior_mass(self) -> None:
        block = LatentCVAEMMDiTBlock(self.config)
        bias = block._hierarchical_key_bias(
            action_len=4,
            cond_len=11,
            low_start=0,
            low_len=4,
            stage_start=4,
            stage_len=3,
            noisy_start=7,
            noisy_len=4,
            device=torch.device("cpu"),
        )
        condition_bias = bias[4:].exp()
        low_mass = condition_bias[0:4].sum()
        stage_mass = condition_bias[4:7].sum()
        noisy_mass = condition_bias[7:11].sum()
        torch.testing.assert_close(low_mass, torch.tensor(4.0))
        torch.testing.assert_close(stage_mass, torch.tensor(4.0))
        torch.testing.assert_close(noisy_mass, torch.tensor(4.0))


if __name__ == "__main__":
    unittest.main()
