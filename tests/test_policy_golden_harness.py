from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from clearvla.tools.policy_golden import (
    ArtifactBuilder,
    FIXTURE_SCHEMA,
    _optimizer_manifest,
    _support_tree_metadata,
    compare,
    create_fixture,
)


class PolicyGoldenHarnessTest(unittest.TestCase):
    def test_optimizer_manifest_finds_intra_and_inter_group_duplicates(self) -> None:
        module = torch.nn.Linear(2, 1)
        weight = module.weight
        manifest = _optimizer_manifest(
            module,
            [
                {"name": "first", "lr": 1.0e-3, "params": [weight, weight]},
                {"name": "second", "lr": 1.0e-3, "params": [weight]},
            ],
        )
        self.assertEqual(manifest[0]["duplicate_names"], ["weight"])
        self.assertEqual(manifest[1]["duplicate_names"], ["weight"])

    def test_support_tree_fingerprint_is_path_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (second / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            left = _support_tree_metadata([f"clearvla/data={first}"])
            right = _support_tree_metadata([f"clearvla/data={second}"])
            self.assertEqual(left, right)
            (second / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
            changed = _support_tree_metadata([f"clearvla/data={second}"])
            self.assertNotEqual(left[0]["sha256"], changed[0]["sha256"])

    def test_fixture_is_reproducible_and_event_rich(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            create_fixture(first, seed=19)
            create_fixture(second, seed=19)
            a = torch.load(first, map_location="cpu", weights_only=False)
            b = torch.load(second, map_location="cpu", weights_only=False)
            self.assertEqual(a["schema"], FIXTURE_SCHEMA)
            self.assertEqual(a["spec"]["horizon"], 24)
            self.assertEqual(a["spec"]["depth"], 8)
            self.assertEqual(tuple(a["tensors"]["sample_noise_native"].shape), (2, 24, 7))
            self.assertEqual(list(a["tensors"]), list(b["tensors"]))
            for key in a["tensors"]:
                torch.testing.assert_close(a["tensors"][key], b["tensors"][key])
            gripper = a["tensors"]["policy_action"][..., -1]
            self.assertEqual(tuple(gripper.shape), (2, 24))
            self.assertGreater(float((gripper[:, 1:] - gripper[:, :-1]).abs().max()), 0.3)

    def test_artifact_builder_rejects_nonfinite_tensors_and_scalars(self) -> None:
        builder = ArtifactBuilder()
        with self.assertRaises(FloatingPointError):
            builder.add("bad/tensor", torch.tensor(float("nan")))
        with self.assertRaises(FloatingPointError):
            builder.add("bad/scalar", float("inf"))

    def test_compare_detects_values_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = ArtifactBuilder()
            left.add("phase", {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
            left.write(root / "left", {"variant": "v77"})
            right = ArtifactBuilder()
            right.add("phase", {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])})
            right.write(root / "right", {"variant": "v77"})
            report = root / "comparison.json"
            self.assertFalse(compare(root / "left", root / "right", report_path=report))
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["pass"])
            self.assertEqual(payload["differences"][0]["path"], "phase/b")
            self.assertEqual(payload["comparison_coverage"]["shared_tensor_count"], 2)
            self.assertEqual(payload["comparison_coverage"]["shared_tensor_numel"], 2)
            self.assertTrue(payload["comparison_coverage"]["exact_tensor_values"])
            self.assertFalse(payload["comparison_coverage"]["nonfinite_allowed"])

    def test_tolerant_compare_keeps_integral_tensors_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = ArtifactBuilder()
            left.add("rng", torch.tensor([1, 2], dtype=torch.uint8))
            left.write(root / "left", {"variant": "v77"})
            right = ArtifactBuilder()
            right.add("rng", torch.tensor([1, 3], dtype=torch.uint8))
            right.write(root / "right", {"variant": "v77"})
            self.assertFalse(compare(root / "left", root / "right", atol=1.0, rtol=1.0))

    def test_ignore_prefix_applies_to_values_and_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = ArtifactBuilder()
            left.add("structure/modules", [{"class": "Legacy"}])
            left.add("behavior", torch.tensor([4.0]))
            left.write(root / "left", {"variant": "v77"})
            right = ArtifactBuilder()
            right.add("structure/modules", [{"class": "Extracted"}])
            right.add("behavior", torch.tensor([4.0]))
            right.write(root / "right", {"variant": "v77"})
            self.assertTrue(
                compare(
                    root / "left",
                    root / "right",
                    ignore_prefixes=("structure/modules",),
                )
            )


if __name__ == "__main__":
    unittest.main()
