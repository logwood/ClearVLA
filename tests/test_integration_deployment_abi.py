"""Fail-closed deployment contracts for all four active outlets, not only LIBERO."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    deployment_config_from_checkpoint,
    validate_deployment_abi,
)

PROFILES = ('identity_7d_pen', 'rdt_right_arm_action_chart_v1', 'calvin_relative_7d_v1', 'libero_relative_7d_v1')


def _abi(name):
    config = ExperimentConfig()
    if name == PROFILES[1]:
        config = load_config('configs/mainline/rdt_multitask8_data_v1.json')
    elif name == PROFILES[2]:
        config = load_config('configs/mainline/calvin_object_binding_formal_v1.json')
    elif name == PROFILES[3]:
        config = replace(config, data=replace(config.data, data_profile=name),
                         bottom=replace(config.bottom, arm_flow_mode='relative_command_direct'))
    profile = resolve_action_state_profile(name)
    identity = SimpleNamespace(validate=lambda: None, config_digest='a'*64, manifest={'test':'manifest'},
                               language=SimpleNamespace(logical_name='goal', sha256='b'*64, size_bytes=1))
    normalizer = SimpleNamespace(mode='zscore', to_dict=lambda: {'mode':'zscore', 'offset':[[0.]*7], 'scale':[[1.]*7]})
    abi = build_deployment_abi(config, identity, action_normalizer=normalizer, state_normalizer=normalizer,
                              data_profile={**profile.as_dict(), 'sha256':profile.digest(),
                                            'gripper_transition_boundary':profile.gripper_transition_boundary},
                              gripper_indices=profile.gripper_indices, goal_metadata={})
    return config, abi


@pytest.mark.parametrize('name', PROFILES)
def test_every_active_outlet_exports_a_readable_deployment_contract(name):
    config, abi = _abi(name)
    validate_deployment_abi(abi)
    restored = deployment_config_from_checkpoint(config.as_dict(), abi)
    assert restored.data.data_profile == name
    assert restored.data.camera_names == config.data.camera_names


@pytest.mark.parametrize('name', PROFILES)
def test_profile_digest_cannot_be_reused_with_a_different_gripper_boundary(name):
    _, abi = _abi(name)
    # Keep any unrelated camera limitation out of this independent mutation.
    abi['observation']['camera_names'] = ['top', 'wrist']
    other = 'previous_command' if name in {PROFILES[0], PROFILES[2]} else 'current_action_state'
    abi['action']['data_profile']['gripper_transition_boundary'] = other
    abi['action']['continuous_gripper_codec_boundary'] = other
    with pytest.raises(ValueError, match='profile|boundary'):
        validate_deployment_abi(abi)


@pytest.mark.parametrize('field,value', [('action_indices', [6,1,2,3,4,5,0]), ('sha256','f'*64),
                                        ('state_to_action_scale',[2.]*7), ('source_state_dim',True)])
def test_pen_profile_registry_is_authoritative(field, value):
    _, abi = _abi(PROFILES[0])
    abi['action']['data_profile'][field] = value
    with pytest.raises(ValueError, match='profile|digest'):
        validate_deployment_abi(abi)


@pytest.mark.parametrize('where,field,value', [
    ('dino','reference_batch_size', 1.5), ('dino','reference_batch_size', True),
    ('dino','patches_per_camera', 576.5), ('dino','token_width', True),
    ('observation','state_dim', 8), ('observation','action_dim', 7.0),
    ('observation','visual_offsets', [-8., -4., 0.]),
    ('action','prediction_horizon', 23), ('action','receding_horizon_execute_rows', True),
])
def test_non_libero_dimensions_and_offsets_are_not_silently_coerced(where,field,value):
    _, abi = _abi(PROFILES[0])
    section = abi['observation']['dinov2'] if where == 'dino' else abi[where]
    section[field] = value
    with pytest.raises(ValueError):
        validate_deployment_abi(abi)
