"""Versioned candidate graph semantics; historical ABI is unchanged by default."""
from collections.abc import Mapping


MODES = {
    'entity_transport_gradient_mode': ('positive_corners_v1', 'ordinary_bilinear_v1'),
    'entity_competition_scale_mode': ('batch_global_v1', 'per_observation_v1'),
    'observation_measurement_mode': ('legacy_v1', 'source_consistent_v1'),
    'target_binding_input_mode': ('protected_pooled_v1', 'full_tokens_views_v1'),
    'observed_outcome_mode': ('none', 'before_proposal_v1'),
    'entity_ownership_mode': ('local_mixture_v1', 'canonical_image_v1'),
    'identity_supervision_mode': ('none', 'rgbd_temporal_v1', 'rgbd_temporal_conditional_v2', 'rgbd_temporal_regions_v3'),
}


def causal_identity_metadata(top):
    modes={key:(top.get(key,pair[0]) if isinstance(top,Mapping) else getattr(top,key)) for key,pair in MODES.items()}
    if any(modes[key] not in pair for key,pair in MODES.items()):
        raise ValueError('unknown causal identity graph selector')
    if all(modes[key]==pair[0] for key,pair in MODES.items()):return None
    result = {
        'schema':'clearvla-causal-identity-candidate-v1','selectors':modes,
        'normalized_transport':'ordinary-bilinear-coordinate-adjoint' if modes['entity_transport_gradient_mode']=='ordinary_bilinear_v1' else 'legacy-positive-corners',
        'competition_scope':modes['entity_competition_scale_mode'],
        'binding':'one-K-plus-null-full-masked-language-and-view-allocation' if modes['target_binding_input_mode']=='full_tokens_views_v1' else 'protected-pooled',
        'measurement':modes['observation_measurement_mode'],
        'outcome':modes['observed_outcome_mode'],
        'ownership':modes['entity_ownership_mode'],
        'identity_supervision':{'mode':modes['identity_supervision_mode'],'plane':'training-labels-only','inputs':['sensor-depth','camera-robot-calibration','observed-joints','causal-RGB-flow'],'excludes':['scene-state','object-labels','future-observations','learned-G-matches']},
        'policy_inputs':'unchanged-RGB-language-observed-proprioception-and-executed-controls',
        'identity_claim':'candidate-learned-correspondence-not-certified-physical-objects',
    }
    if modes['identity_supervision_mode'] in {'rgbd_temporal_conditional_v2','rgbd_temporal_regions_v3'}:
        result['identity_supervision']['correspondence']='real-K-conditional-before-interpolation;producer-support-only;null-is-not-an-identity-label'
    if modes['identity_supervision_mode']=='rgbd_temporal_regions_v3':
        result['identity_supervision']['inputs']+=['frozen-RGB-mask-proposals','independent-rigid-motion-evidence']
        result['identity_supervision'].update(regions='sam-surface-motion-negative-only;unknown-retained',
            objectives='retain-all-original-positives-and-source-MSE;add-negative-JS-margin-and-equal-region-source-MSE',
            annotations='pinned-compact-training-labels;no-RGB-or-DINO-cache;no-online-policy-input')
    return result
