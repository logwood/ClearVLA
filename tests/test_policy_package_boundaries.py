from __future__ import annotations

import ast
from pathlib import Path
import unittest

import torch

from clearvla.experiments.observed_state_lab.policy import TimeEmbedding
from clearvla.experiments.observed_state_lab.policy_v39 import LatentCVAEActionDecoder
from clearvla.experiments.observed_state_lab.world_model import BiasFreeFFN, sinusoidal_positions
from clearvla.policy.gauges import time_stratified_attention
from clearvla.policy.primitives import (
    BiasFreeFFN as PackagedBiasFreeFFN,
    TimeEmbedding as PackagedTimeEmbedding,
    sinusoidal_positions as packaged_sinusoidal_positions,
)


class PolicyPackageBoundaryTest(unittest.TestCase):
    def test_shared_primitive_facades_preserve_identity(self) -> None:
        self.assertIs(BiasFreeFFN, PackagedBiasFreeFFN)
        self.assertIs(TimeEmbedding, PackagedTimeEmbedding)
        self.assertIs(sinusoidal_positions, packaged_sinusoidal_positions)

    def test_packaged_policy_has_no_experiment_imports(self) -> None:
        root = Path(__file__).resolve().parents[1] / "clearvla" / "policy"
        violations: list[str] = []
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name.startswith("clearvla.experiments") for name in names):
                    violations.append(f"{path.name}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_time_attention_gauge_matches_legacy_definition(self) -> None:
        time = torch.tensor([0.0, 0.2, 0.4, 0.7, 1.0])
        noisy = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        workspace = 1.0 - noisy
        low = noisy.square()
        stage = workspace.square()
        expected = LatentCVAEActionDecoder._time_stratified_attention(
            time, noisy, workspace, low, stage,
        )
        actual = time_stratified_attention(time, noisy, workspace, low, stage)
        self.assertEqual(list(expected), list(actual))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
