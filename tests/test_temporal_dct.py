from __future__ import annotations

import unittest

import torch

from clearvla.policy.codec import ActionTemporalDCT, TemporalDCT


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


if __name__ == "__main__":
    unittest.main()
