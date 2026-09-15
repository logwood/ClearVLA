from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from clearvla.policy.codec import (
    ActionTemporalDCT,
    DCTFlowCodec,
    FrequencyPhysicalActionTokenLift,
    SoftSpectralAperture,
    TemporalDCT,
)
from clearvla.policy.config import V362PolicyConfig


class TemporalDCTTest(unittest.TestCase):
    def setUp(self) -> None:
        self.horizon = 24
        self.chart = TemporalDCT(self.horizon).eval()
        self.action_chart = ActionTemporalDCT(
            self.horizon,
            arm_dims=(0, 1, 2, 3, 4, 5),
            gripper_index=6,
        ).eval()

    def test_matrix_is_orthonormal(self) -> None:
        matrix = self.chart.matrix.double()
        torch.testing.assert_close(
            matrix @ matrix.transpose(0, 1),
            torch.eye(self.horizon, dtype=torch.float64),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_roundtrip_is_exact_for_float32_and_float64(self) -> None:
        for dtype in (torch.float32, torch.float64):
            values = torch.randn(3, self.horizon, 7, dtype=dtype)
            reconstructed = self.chart.decode(self.chart.encode(values))
            torch.testing.assert_close(reconstructed, values, atol=3e-6, rtol=3e-6)

    def test_parseval_energy_is_preserved(self) -> None:
        values = torch.randn(5, self.horizon, 7)
        coefficients = self.chart.encode(values)
        torch.testing.assert_close(
            values.square().sum(),
            coefficients.square().sum(),
            atol=3e-5,
            rtol=3e-6,
        )

    def test_arm_and_gripper_share_chart_but_remain_separate_groups(self) -> None:
        values = torch.randn(2, self.horizon, 7)
        coefficients = self.action_chart.encode(values)
        groups = self.action_chart.groups(coefficients)
        self.assertEqual(tuple(groups["arm"].shape), (2, self.horizon, 6))
        self.assertEqual(tuple(groups["gripper"].shape), (2, self.horizon, 1))
        energy = self.action_chart.group_frequency_energy(coefficients)
        self.assertEqual(tuple(energy["arm"].shape), (self.horizon,))
        self.assertEqual(tuple(energy["gripper"].shape), (self.horizon,))

    def test_full_frequency_copy_is_identity(self) -> None:
        values = torch.randn(2, self.horizon, 7)
        coefficients = self.chart.encode(values)
        full = self.chart.low_frequency(coefficients, self.horizon)
        torch.testing.assert_close(full, coefficients)

    def test_arm_truncation_does_not_touch_gripper(self) -> None:
        values = torch.randn(2, self.horizon, 7)
        coefficients = self.action_chart.encode(values)
        truncated = self.action_chart.low_frequency(
            coefficients,
            arm_keep=8,
            gripper_keep=self.horizon,
        )
        torch.testing.assert_close(
            truncated[..., 6],
            coefficients[..., 6],
        )
        torch.testing.assert_close(
            truncated[..., 8:, :6],
            torch.zeros_like(truncated[..., 8:, :6]),
        )

    def test_gradient_crosses_both_directions(self) -> None:
        values = torch.randn(2, self.horizon, 7, requires_grad=True)
        coefficients = self.chart.encode(values)
        loss = coefficients.square().mean() + self.chart.decode(coefficients).square().mean()
        loss.backward()
        self.assertIsNotNone(values.grad)
        self.assertGreater(float(values.grad.abs().sum()), 0.0)

    def test_group_boundary_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            ActionTemporalDCT(self.horizon, arm_dims=(0, 6), gripper_index=6).encode(
                torch.randn(1, self.horizon, 7)
            )
        with self.assertRaises(ValueError):
            ActionTemporalDCT(self.horizon, arm_dims=(0, 1), gripper_index=9).encode(
                torch.randn(1, self.horizon, 7)
            )

    def test_soft_aperture_grows_without_hard_frequency_slicing(self) -> None:
        aperture = SoftSpectralAperture(
            self.horizon,
            arm_channels=12,
            gripper_channels=2,
            arm_start_fraction=0.16,
            gripper_start_fraction=0.33,
        )
        early = aperture(torch.tensor([0.0]))
        late = aperture(torch.tensor([0.5]))
        final = aperture(torch.tensor([1.0]))
        self.assertTrue(bool((early["coefficient_mask"] > 0.0).all()))
        self.assertTrue(bool((late["coefficient_mask"] >= early["coefficient_mask"]).all()))
        torch.testing.assert_close(
            final["coefficient_mask"],
            torch.ones_like(final["coefficient_mask"]),
        )
        self.assertGreater(
            float(early["gripper_mask"].mean()),
            0.0,
        )
        self.assertLess(
            float(early["gripper_mask"].mean()),
            float(final["gripper_mask"].mean()),
        )

    def test_soft_aperture_controller_preserves_frequency_order(self) -> None:
        aperture = SoftSpectralAperture(
            self.horizon,
            arm_channels=12,
            gripper_channels=6,
            controller_shift_limit=2.0,
        )
        controller_shift = torch.randn(4, self.horizon, 2) * 3.0
        early = aperture(
            torch.full((4,), 0.2),
            controller_shift=controller_shift,
        )
        late = aperture(
            torch.full((4,), 0.7),
            controller_shift=controller_shift,
        )
        for name in ("arm_mask", "gripper_mask"):
            self.assertTrue(bool((early[name][:, 1:] <= early[name][:, :-1] + 1e-7).all()))
            self.assertTrue(bool((late[name] >= early[name] - 1e-7).all()))
        self.assertTrue(bool((early["frequency_spacing_min"] > 0.0).all()))
        self.assertTrue(
            bool((early["frequency_spacing_max"] >= early["frequency_spacing_min"]).all())
        )

    def test_zero_controller_is_exactly_the_native_frequency_chart(self) -> None:
        aperture = SoftSpectralAperture(
            self.horizon,
            arm_channels=12,
            gripper_channels=6,
        )
        progress = torch.tensor([0.15, 0.55, 0.9])
        baseline = aperture(progress)
        controlled = aperture(
            progress,
            controller_shift=torch.zeros(3, self.horizon, 2),
        )
        for name in (
            "coefficient_mask",
            "token_mask",
            "arm_mask",
            "gripper_mask",
            "arm_cutoff",
            "gripper_cutoff",
            "frequency_warp_rms",
        ):
            torch.testing.assert_close(controlled[name], baseline[name])
        torch.testing.assert_close(
            controlled["frequency_warp_rms"],
            torch.zeros_like(controlled["frequency_warp_rms"]),
            atol=1e-7,
            rtol=0.0,
        )

    def test_frequency_lift_keeps_full_coefficient_state(self) -> None:
        config = SimpleNamespace(
            hidden_size=16,
            action_horizon=self.horizon,
            arm_dim=6,
            physical_action_dim=14,
        )
        lift = FrequencyPhysicalActionTokenLift(config)
        coefficients = torch.randn(3, self.horizon, 14)
        tokens = lift(coefficients)
        self.assertEqual(tuple(tokens.shape), (3, self.horizon, 16))

    def test_complete_flow_chart_keeps_bridge_and_velocity_on_the_same_manifold(self) -> None:
        config = V362PolicyConfig(
            action_horizon=self.horizon,
            arm_flow_mode="manifold_native",
            gripper_field_mode="parseval_temporal",
            gripper_field_dim=6,
        )
        codec = DCTFlowCodec(config).eval()
        state = torch.randn(4, config.action_dim)
        action = torch.randn(4, self.horizon, config.action_dim)
        target_physical = codec.physical.encode(action, state)
        target_coefficients = codec.encode_physical(target_physical)
        noise_coefficients = codec.sample_noise(
            4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_state=state,
        )
        midpoint = 0.35 * target_coefficients + 0.65 * noise_coefficients
        torch.testing.assert_close(
            codec.project_state(midpoint, state),
            midpoint,
            atol=3e-5,
            rtol=3e-5,
        )

        arm_native = torch.randn(4, self.horizon, config.arm_dim)
        grip_native = torch.randn(4, self.horizon, 1)
        tangent = codec.expand_tangent_velocity(
            codec.temporal.encode(arm_native),
            codec.temporal.encode(grip_native),
        )
        torch.testing.assert_close(
            codec.project_tangent(tangent),
            tangent,
            atol=3e-5,
            rtol=3e-5,
        )

    def test_boundary_multiscale_source_enters_the_complete_dct_chart_directly(self) -> None:
        config = V362PolicyConfig(
            action_horizon=self.horizon,
            arm_flow_mode="manifold_native",
            arm_source_mode="boundary_multiscale",
            arm_source_scale=0.8,
            arm_source_innovation_weight=0.2,
            arm_source_velocity_weight=0.5,
            arm_source_acceleration_weight=0.3,
            gripper_field_mode="parseval_temporal",
            gripper_field_dim=6,
        )
        codec = DCTFlowCodec(config).eval()
        state = torch.randn(4, config.action_dim)
        coefficients = codec.sample_noise(
            4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_state=state,
            generator=torch.Generator().manual_seed(112),
        )
        torch.testing.assert_close(
            codec.project_state(coefficients, state),
            coefficients,
            atol=3e-5,
            rtol=3e-5,
        )
        physical = codec.decode_coefficients(coefficients)
        arm = physical[..., : 2 * config.arm_dim]
        expected_delta = codec.physical.encode_arm_coordinates(arm[..., : config.arm_dim], state)[1]
        torch.testing.assert_close(
            arm[..., config.arm_dim :],
            expected_delta,
            atol=3e-5,
            rtol=3e-5,
        )

    def test_tangent_expansion_owns_fp32_geometry_under_bfloat16_autocast(self) -> None:
        config = V362PolicyConfig(
            action_horizon=self.horizon,
            arm_flow_mode="manifold_native",
            gripper_field_mode="parseval_temporal",
            gripper_field_dim=6,
        )
        codec = DCTFlowCodec(config).eval()
        arm = torch.randn(3, self.horizon, config.arm_dim).to(torch.bfloat16)
        gripper = torch.randn(3, self.horizon, 1).to(torch.bfloat16)
        reference = codec.expand_tangent_velocity(arm.float(), gripper.float()).to(torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = codec.expand_tangent_velocity(arm, gripper)
        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            codec.project_tangent(actual),
            actual,
            atol=2e-2,
            rtol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
