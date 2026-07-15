from __future__ import annotations

import ast
import builtins
from dataclasses import is_dataclass
from pathlib import Path
import symtable
import unittest

import torch

from clearvla.experiments.observed_state_lab.policy import TimeEmbedding
from clearvla.experiments.observed_state_lab.policy_v39 import (
    AdaptiveRecurrentCVAEActionDecoder,
    HierarchicalMMDiTActionDecoder,
    HierarchicalLatentMainActionDecoder,
    LatentCVAEActionDecoder,
    LayeredV37StyleResidualActionFlowDenoiser,
    SemanticEvidenceWorkspace,
    TemporalMidcutWorldActionDiT,
    V39PolicyConfig,
    V39PolicySystem,
    _parse_layer_pair_schedule,
)
from clearvla.experiments.observed_state_lab.world_model import BiasFreeFFN, sinusoidal_positions
from clearvla.experiments.observed_state_lab.policy_v38 import (
    ControlledResidualLatentDynamics,
    DenseVisualMemory,
)
from clearvla.policy import NestedLowRankContractionBank as ExportedNestedLowRankContractionBank
from clearvla.policy.gauges import time_stratified_attention
from clearvla.policy.config import V39PolicyConfig as PackagedV39PolicyConfig
from clearvla.policy.decoder import (
    HierarchicalMMDiTActionDecoder as PackagedHierarchicalMMDiTActionDecoder,
)
from clearvla.policy.legacy import (
    AdaptiveRecurrentCVAEActionDecoder as PackagedAdaptiveRecurrentCVAEActionDecoder,
    HierarchicalLatentMainActionDecoder as PackagedHierarchicalLatentMainActionDecoder,
    LatentCVAEActionDecoder as PackagedLatentCVAEActionDecoder,
    LayeredV37StyleResidualActionFlowDenoiser as PackagedLayeredResidualDecoder,
    SemanticEvidenceWorkspace as PackagedSemanticEvidenceWorkspace,
)
from clearvla.policy.legacy.residual import (
    _parse_layer_pair_schedule as packaged_parse_layer_pair_schedule,
)
from clearvla.policy.primitives import (
    BiasFreeFFN as PackagedBiasFreeFFN,
    TimeEmbedding as PackagedTimeEmbedding,
    sinusoidal_positions as packaged_sinusoidal_positions,
)
from clearvla.policy.refinement import NestedLowRankContractionBank
from clearvla.policy.trunk_primitives import (
    ControlledResidualLatentDynamics as PackagedControlledResidualLatentDynamics,
    DenseVisualMemory as PackagedDenseVisualMemory,
)
from clearvla.policy.trunk import TemporalMidcutWorldActionDiT as PackagedTemporalMidcutWorldActionDiT
from clearvla.policy.system import V39PolicySystem as PackagedV39PolicySystem


class PolicyPackageBoundaryTest(unittest.TestCase):
    def test_shared_primitive_facades_preserve_identity(self) -> None:
        self.assertIs(BiasFreeFFN, PackagedBiasFreeFFN)
        self.assertIs(TimeEmbedding, PackagedTimeEmbedding)
        self.assertIs(sinusoidal_positions, packaged_sinusoidal_positions)
        self.assertIs(V39PolicyConfig, PackagedV39PolicyConfig)
        self.assertTrue(is_dataclass(PackagedV39PolicyConfig))
        self.assertTrue(PackagedV39PolicyConfig.__dataclass_params__.frozen)

    def test_v39_module_is_a_definition_free_facade(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "clearvla"
            / "experiments"
            / "observed_state_lab"
            / "policy_v39.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(definitions, [])

    def test_legacy_decoder_facades_preserve_identity(self) -> None:
        self.assertIs(
            AdaptiveRecurrentCVAEActionDecoder,
            PackagedAdaptiveRecurrentCVAEActionDecoder,
        )
        self.assertIs(HierarchicalLatentMainActionDecoder, PackagedHierarchicalLatentMainActionDecoder)
        self.assertIs(LatentCVAEActionDecoder, PackagedLatentCVAEActionDecoder)
        self.assertIs(LayeredV37StyleResidualActionFlowDenoiser, PackagedLayeredResidualDecoder)
        self.assertIs(SemanticEvidenceWorkspace, PackagedSemanticEvidenceWorkspace)
        self.assertIs(_parse_layer_pair_schedule, packaged_parse_layer_pair_schedule)

    def test_trunk_primitive_facades_preserve_identity(self) -> None:
        self.assertIs(ControlledResidualLatentDynamics, PackagedControlledResidualLatentDynamics)
        self.assertIs(DenseVisualMemory, PackagedDenseVisualMemory)
        self.assertIs(TemporalMidcutWorldActionDiT, PackagedTemporalMidcutWorldActionDiT)
        self.assertIs(V39PolicySystem, PackagedV39PolicySystem)

    def test_current_decoder_and_refinement_operator_are_packaged(self) -> None:
        self.assertIs(HierarchicalMMDiTActionDecoder, PackagedHierarchicalMMDiTActionDecoder)
        self.assertIs(ExportedNestedLowRankContractionBank, NestedLowRankContractionBank)

    def test_packaged_policy_has_no_experiment_imports(self) -> None:
        root = Path(__file__).resolve().parents[1] / "clearvla" / "policy"
        violations: list[str] = []
        for path in sorted(root.rglob("*.py")):
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

    def test_packaged_policy_has_no_new_unresolved_global_references(self) -> None:
        root = Path(__file__).resolve().parents[1] / "clearvla" / "policy"
        builtins_set = set(dir(builtins))
        # These two defects predate the package extraction and live only in
        # disabled legacy branches. Keep them visible without allowing the
        # refactor to introduce any additional unresolved module dependency.
        frozen_legacy_defects = {
            ("legacy/cvae.py", "_decode_with_z", "hierarchical_refine"),
            ("legacy/residual.py", "forward", "action_update_means"),
        }
        violations: list[tuple[str, str, str]] = []
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
            module_bindings = {
                symbol.get_name()
                for symbol in table.get_symbols()
                if symbol.is_assigned()
                or symbol.is_imported()
                or symbol.is_namespace()
                or symbol.is_parameter()
            }
            pending = [table]
            while pending:
                current = pending.pop()
                for symbol in current.get_symbols():
                    reference = (relative, current.get_name(), symbol.get_name())
                    if (
                        symbol.is_referenced()
                        and symbol.is_global()
                        and symbol.get_name() not in module_bindings
                        and symbol.get_name() not in builtins_set
                        and reference not in frozen_legacy_defects
                    ):
                        violations.append(reference)
                pending.extend(current.get_children())
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
