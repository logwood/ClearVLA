from __future__ import annotations

import unittest

import torch

from clearvla.experiments.observed_state_lab.policy_v36_2 import (
    ParsevalGripperTemporalFrame,
    PhysicalActionCodec,
    PhysicalActionTokenLift,
    PhysicalVelocityHead,
    V362PolicyConfig,
)
from clearvla.policy.codec import (
    ParsevalGripperTemporalFrame as PackagedParsevalGripperTemporalFrame,
    PhysicalActionCodec as PackagedPhysicalActionCodec,
    PhysicalActionTokenLift as PackagedPhysicalActionTokenLift,
    PhysicalVelocityHead as PackagedPhysicalVelocityHead,
    TransitionAwarePhysicalVelocityHead as PackagedTransitionAwarePhysicalVelocityHead,
)
from clearvla.experiments.observed_state_lab.policy_v36_3 import TransitionAwarePhysicalVelocityHead


class PhysicalActionCodecManifoldTest(unittest.TestCase):
    def test_legacy_import_is_packaged_class(self) -> None:
        self.assertIs(ParsevalGripperTemporalFrame, PackagedParsevalGripperTemporalFrame)
        self.assertIs(PhysicalActionCodec, PackagedPhysicalActionCodec)
        self.assertIs(PhysicalActionTokenLift, PackagedPhysicalActionTokenLift)
        self.assertIs(PhysicalVelocityHead, PackagedPhysicalVelocityHead)
        self.assertIs(TransitionAwarePhysicalVelocityHead, PackagedTransitionAwarePhysicalVelocityHead)

    def setUp(self) -> None:
        self.config = V362PolicyConfig(
            arm_flow_mode="manifold_native",
            arm_noise_temporal_rho=0.7,
            gripper_field_mode="parseval_temporal",
            gripper_field_dim=6,
        )
        self.codec = PhysicalActionCodec(self.config)
        self.state = torch.randn(16, self.config.action_dim)

    def test_encode_is_fixed_by_projection(self) -> None:
        action = torch.randn(16, self.config.action_horizon, self.config.action_dim)
        physical = self.codec.encode(action, self.state)
        projected = self.codec.project_physical(physical, self.state)
        torch.testing.assert_close(projected, physical, atol=2e-6, rtol=2e-6)

    def test_projection_is_idempotent(self) -> None:
        physical = torch.randn(16, self.config.action_horizon, self.config.physical_action_dim)
        once = self.codec.project_physical(physical, self.state)
        twice = self.codec.project_physical(once, self.state)
        torch.testing.assert_close(twice, once, atol=2e-6, rtol=2e-6)

    def test_noise_and_bridge_velocity_stay_on_manifold(self) -> None:
        action = torch.randn(16, self.config.action_horizon, self.config.action_dim)
        target = self.codec.encode(action, self.state)
        noise = self.codec.sample_noise(
            16, device=torch.device("cpu"), dtype=torch.float32, action_state=self.state,
        )
        torch.testing.assert_close(
            self.codec.project_physical(noise, self.state), noise, atol=2e-6, rtol=2e-6,
        )
        _, _, tangent_null = self.codec.project_arm_tangent(
            (noise - target)[..., : 2 * self.config.arm_dim]
        )
        torch.testing.assert_close(tangent_null, torch.zeros_like(tangent_null), atol=2e-6, rtol=0.0)

    def test_conditioned_ar_variance_schedule(self) -> None:
        batch = 4096
        state = torch.zeros(batch, self.config.action_dim)
        noise = self.codec.sample_noise(
            batch, device=torch.device("cpu"), dtype=torch.float32, action_state=state,
        )
        arm_abs = noise[..., : self.config.arm_dim]
        arm_delta = noise[..., self.config.arm_dim : 2 * self.config.arm_dim]
        abs_variance = arm_abs.var(dim=(0, 2), unbiased=False)
        delta_variance = arm_delta.var(dim=(0, 2), unbiased=False)

        rho = float(self.config.arm_noise_temporal_rho)
        step = torch.arange(1, self.config.action_horizon + 1, dtype=torch.float32)
        expected_abs = 1.0 - rho ** (2.0 * step)
        previous_step = torch.arange(self.config.action_horizon, dtype=torch.float32)
        expected_delta = (1.0 - rho * rho) + (1.0 - rho) ** 2 * (
            1.0 - rho ** (2.0 * previous_step)
        )
        torch.testing.assert_close(abs_variance, expected_abs, atol=0.025, rtol=0.05)
        torch.testing.assert_close(delta_variance, expected_delta, atol=0.025, rtol=0.05)

    def test_unit_gaussian_state_has_stationary_population_marginals(self) -> None:
        batch = 4096
        state = torch.randn(batch, self.config.action_dim)
        noise = self.codec.sample_noise(
            batch, device=torch.device("cpu"), dtype=torch.float32, action_state=state,
        )
        arm_abs = noise[..., : self.config.arm_dim]
        arm_delta = noise[..., self.config.arm_dim : 2 * self.config.arm_dim]
        abs_variance = arm_abs.var(dim=(0, 2), unbiased=False)
        delta_variance = arm_delta.var(dim=(0, 2), unbiased=False)

        expected_abs = torch.ones_like(abs_variance)
        expected_delta = torch.full_like(
            delta_variance, 2.0 * (1.0 - self.config.arm_noise_temporal_rho)
        )
        torch.testing.assert_close(abs_variance, expected_abs, atol=0.025, rtol=0.05)
        torch.testing.assert_close(delta_variance, expected_delta, atol=0.025, rtol=0.05)

    def test_manifold_noise_requires_action_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires action_state"):
            self.codec.sample_noise(2, device=torch.device("cpu"), dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
