"""Serialized deployment integer fields must not silently coerce inputs."""
from __future__ import annotations

import pytest

from clearvla.mainline.runtime.deployment import validate_deployment_abi
from tests.test_libero_deployment_contract import _abi


@pytest.mark.parametrize("value", [True, False, 1.0, 1.9, "32", None, float("inf")])
def test_reference_batch_size_requires_a_json_integer(value: object) -> None:
    _, abi = _abi()
    abi["observation"]["dinov2"]["reference_batch_size"] = value
    with pytest.raises(ValueError, match="integer"):
        validate_deployment_abi(abi)


@pytest.mark.parametrize("value", [True, False, 1.0, 1.9, "1", None, float("inf")])
def test_executed_rows_requires_a_json_integer(value: object) -> None:
    _, abi = _abi()
    abi["action"]["receding_horizon_execute_rows"] = value
    with pytest.raises(ValueError, match="integer"):
        validate_deployment_abi(abi)


def test_exported_integer_contract_is_unchanged() -> None:
    _, abi = _abi()
    assert validate_deployment_abi(abi) == abi
