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
    NativeTimePhysicalActionTokenLift,
    ParsevalGripperTemporalFrame as PackagedParsevalGripperTemporalFrame,
    PhysicalActionCodec as PackagedPhysicalActionCodec,
    PhysicalActionTokenLift as PackagedPhysicalActionTokenLift,
    PhysicalVelocityHead as PackagedPhysicalVelocityHead,
    TransitionAwarePhysicalVelocityHead as PackagedTransitionAwarePhysicalVelocityHead,
)
from clearvla.policy.decoder import ActionOnlyPhysicalVelocityHead
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

    def test_native_velocity_expands_exactly_into_both_tangent_spaces(self) -> None:
        arm_velocity = torch.randn(
            3, self.config.action_horizon, self.config.arm_dim
        )
        arm_field = self.codec.encode_arm_tangent(arm_velocity)
        recovered_arm, projected_arm, arm_null = self.codec.project_arm_tangent(
            arm_field
        )
        torch.testing.assert_close(recovered_arm, arm_velocity, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(projected_arm, arm_field, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(arm_null, torch.zeros_like(arm_null), atol=2e-6, rtol=0.0)

        gripper_velocity = torch.randn(3, self.config.action_horizon, 1)
        gripper_field = self.codec.encode_gripper_tangent(gripper_velocity)
        torch.testing.assert_close(
            self.codec.decode_gripper_field(gripper_field),
            gripper_velocity,
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            self.codec.project_gripper_field(gripper_field),
            gripper_field,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_native_time_lift_gathers_delayed_field_views_before_projection(self) -> None:
        source_step = 5
        action_a = torch.zeros(1, self.config.action_horizon, self.config.action_dim)
        action_b = action_a.clone()
        action_b[:, source_step, self.config.gripper_index] = 1.0
        state = torch.zeros(1, self.config.action_dim)
        physical_a = self.codec.encode(action_a, state)
        physical_b = self.codec.encode(action_b, state)
        field_delta = (
            physical_b[..., 2 * self.config.arm_dim :]
            - physical_a[..., 2 * self.config.arm_dim :]
        )
        field_time_support = field_delta.abs().sum(dim=-1)[0] > 1e-6
        self.assertGreater(int(field_time_support.sum()), 1)

        lift = NativeTimePhysicalActionTokenLift(self.config).eval()
        token_delta = lift(physical_b) - lift(physical_a)
        token_time_support = token_delta.abs().sum(dim=-1)[0] > 1e-5
        expected_support = torch.zeros_like(token_time_support)
        expected_support[source_step] = True
        torch.testing.assert_close(token_time_support, expected_support)
        torch.testing.assert_close(
            token_delta[:, ~expected_support],
            torch.zeros_like(token_delta[:, ~expected_support]),
            atol=2e-6,
            rtol=0.0,
        )

        arm_action = action_a.clone()
        arm_action[:, source_step, 0] = 1.0
        arm_physical = self.codec.encode(arm_action, state)
        arm_field_delta = arm_physical[..., : 2 * self.config.arm_dim]
        arm_field_support = arm_field_delta.abs().sum(dim=-1)[0] > 1e-6
        self.assertGreater(int(arm_field_support.sum()), 1)
        arm_token_delta = lift(arm_physical) - lift(physical_a)
        torch.testing.assert_close(
            arm_token_delta[:, ~expected_support],
            torch.zeros_like(arm_token_delta[:, ~expected_support]),
            atol=2e-6,
            rtol=0.0,
        )

    def test_action_only_head_emits_native_time_tangent_velocity(self) -> None:
        head = ActionOnlyPhysicalVelocityHead(self.config).eval()
        tokens = torch.randn(
            3, self.config.action_horizon, self.config.hidden_size
        )
        physical_velocity = head(tokens)
        self.assertEqual(
            tuple(physical_velocity.shape),
            (3, self.config.action_horizon, self.config.physical_action_dim),
        )

        arm_field = physical_velocity[..., : 2 * self.config.arm_dim]
        _, projected_arm, arm_null = self.codec.project_arm_tangent(arm_field)
        torch.testing.assert_close(projected_arm, arm_field, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(arm_null, torch.zeros_like(arm_null), atol=2e-6, rtol=0.0)

        gripper_field = physical_velocity[..., 2 * self.config.arm_dim :]
        torch.testing.assert_close(
            self.codec.project_gripper_field(gripper_field),
            gripper_field,
            atol=2e-6,
            rtol=2e-6,
        )

        source_step = 2
        base_tokens = torch.zeros_like(tokens)
        changed_tokens = base_tokens.clone()
        changed_tokens[:, source_step] = torch.randn_like(
            changed_tokens[:, source_step]
        )
        velocity_delta = head(changed_tokens) - head(base_tokens)
        arm_delta, _, _ = self.codec.project_arm_tangent(
            velocity_delta[..., : 2 * self.config.arm_dim]
        )
        gripper_delta = self.codec.decode_gripper_field(
            velocity_delta[..., 2 * self.config.arm_dim :]
        )
        native_delta = torch.cat([arm_delta, gripper_delta], dim=-1)
        outside_source = torch.ones(
            self.config.action_horizon, dtype=torch.bool
        )
        outside_source[source_step] = False
        torch.testing.assert_close(
            native_delta[:, outside_source],
            torch.zeros_like(native_delta[:, outside_source]),
            atol=2e-6,
            rtol=0.0,
        )
        physical_velocity.square().mean().backward()
        self.assertIsNotNone(head.arm_native)
        self.assertIsNotNone(head.grip_native)
        self.assertIsNotNone(head.arm_native.weight.grad)
        self.assertIsNotNone(head.grip_native.weight.grad)
        self.assertGreater(float(head.arm_native.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.grip_native.weight.grad.abs().sum()), 0.0)

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
