"""Frozen native configurations satisfy the decoder's actual read-only contract."""

from dataclasses import FrozenInstanceError
from importlib import import_module

import pytest

from clearvla.mainline.v120_core.config import V39PolicyConfig as MainlineConfig
from clearvla.mainline.v120_core.decoder import PolicyDecoderConfig as MainlineRead
from clearvla.mainline.v120_core.evidence import PolicyEvidenceConfig as MainlineEvidence
from clearvla.mainline.v120_core.intent import PolicyIntentConfig as MainlineIntent
from clearvla.policy.config import V39PolicyConfig as LegacyConfig
from clearvla.policy.decoder import PolicyDecoderConfig as LegacyRead
from clearvla.policy.evidence import PolicyEvidenceConfig as LegacyEvidence
from clearvla.policy.intent import PolicyIntentConfig as LegacyIntent


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


@pytest.mark.parametrize("prefix", ["clearvla.policy", "clearvla.mainline.v120_core"])
def test_decoder_admits_its_downstream_read_requirements_transitively(prefix):
    decoder = import_module(prefix + ".decoder").PolicyDecoderConfig
    native = LegacyConfig() if prefix == "clearvla.policy" else MainlineConfig()
    for family, name in (("evidence", "PolicyEvidenceConfig"), ("intent", "PolicyIntentConfig")):
        requirement = getattr(import_module(prefix + "." + family), name)
        assert requirement in decoder.__mro__
        for member in vars(requirement):
            value = getattr(requirement, member)
            if isinstance(value, property):
                assert isinstance(getattr(decoder, member), property)
                assert getattr(native, member) is not None


def legacy_downstream(config: LegacyRead) -> tuple[LegacyEvidence, LegacyIntent]:
    return config, config


def mainline_downstream(config: MainlineRead) -> tuple[MainlineEvidence, MainlineIntent]:
    return config, config


def test_decoder_downstream_protocol_assignment_is_not_a_cast():
    native_legacy = LegacyConfig()
    a, b = legacy_downstream(native_legacy)
    assert a is native_legacy and b is native_legacy
    native_mainline = MainlineConfig()
    c, d = mainline_downstream(native_mainline)
    assert c is native_mainline and d is native_mainline
