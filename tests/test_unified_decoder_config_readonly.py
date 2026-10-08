"""Frozen native configurations satisfy the decoder's actual read-only contract."""

from dataclasses import FrozenInstanceError
from importlib import import_module

import pytest

from clearvla.mainline.v120_core.config import V39PolicyConfig as MainlineConfig
from clearvla.mainline.v120_core.decoder import PolicyDecoderConfig as MainlineRead
from clearvla.policy.config import V39PolicyConfig as LegacyConfig
from clearvla.policy.decoder import PolicyDecoderConfig as LegacyRead


def legacy_dimensions(config: LegacyRead) -> tuple[int, int, int]:
    return config.hidden_size, config.arm_dim, config.physical_action_dim


def mainline_dimensions(config: MainlineRead) -> tuple[int, int, int]:
    return config.hidden_size, config.arm_dim, config.physical_action_dim


def test_native_frozen_configs_are_admitted_without_cast_or_mutation():
    legacy, mainline = LegacyConfig(), MainlineConfig()
    assert legacy_dimensions(legacy) == (
        legacy.hidden_size,
        legacy.arm_dim,
        legacy.physical_action_dim,
    )
    assert mainline_dimensions(mainline) == (
        mainline.hidden_size,
        mainline.arm_dim,
        mainline.physical_action_dim,
    )
    for config in (legacy, mainline):
        with pytest.raises(FrozenInstanceError):
            setattr(config, "hidden_size", config.hidden_size + 1)


def test_both_read_contracts_keep_every_declared_field_without_setters():
    old = {name for name, value in vars(LegacyRead).items() if isinstance(value, property)}
    current = {name for name, value in vars(MainlineRead).items() if isinstance(value, property)}
    assert old == current and len(current) == 52
    for interface, config in ((LegacyRead, LegacyConfig()), (MainlineRead, MainlineConfig())):
        for name in current:
            value = vars(interface)[name]
            assert isinstance(value, property) and value.fset is None
            assert getattr(config, name) is not None


@pytest.mark.parametrize(
    "family", ["codec", "controller", "evidence", "intent", "trunk_primitives"]
)
@pytest.mark.parametrize("prefix", ["clearvla.policy", "clearvla.mainline.v120_core"])
def test_downstream_config_contracts_do_not_reintroduce_mutation(family, prefix):
    module = import_module(prefix + "." + family)
    native = LegacyConfig() if prefix == "clearvla.policy" else MainlineConfig()
    interfaces = [
        value
        for name, value in vars(module).items()
        if name.endswith("Config")
        and isinstance(value, type)
        and value.__module__ == module.__name__
    ]
    assert len(interfaces) == 1
    properties = {
        name: value for name, value in vars(interfaces[0]).items() if isinstance(value, property)
    }
    assert properties
    for name, prop in properties.items():
        assert prop.fset is None
        assert hasattr(native, name)
