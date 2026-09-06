from __future__ import annotations

import copy

import pytest
from test_mainline_flow_schedule_identity import _abi

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.runtime.deployment import validate_deployment_abi


@pytest.mark.parametrize("component", ["observation", "top", "bottom", "training", "runtime"])
def test_deployment_rejects_same_shape_old_component_semantics(component: str) -> None:
    abi = _abi(ExperimentConfig())
    old = copy.deepcopy(abi)
    old["architecture_manifest"]["components"][component] += "_old_behavior"
    # The saved graph is self-consistent and all shapes are unchanged.
    assert old["graph_config"] == abi["graph_config"]
    with pytest.raises(ValueError, match="active implementation"):
        validate_deployment_abi(old)
    assert validate_deployment_abi(abi) == abi


def test_deployment_cannot_omit_the_architecture_manifest() -> None:
    abi = _abi(ExperimentConfig())
    del abi["architecture_manifest"]
    with pytest.raises(ValueError, match="architecture_manifest"):
        validate_deployment_abi(abi)
