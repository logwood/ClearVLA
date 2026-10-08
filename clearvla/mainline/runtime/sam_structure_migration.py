"""Narrow B-v2 checkpoint initialization for two explicit structural trials."""
import torch

SAM_STRUCTURE_V1 = "sam_structure_v1"
SOURCE_DIGEST = "882ec91396f0611c5160805090fed14bc7a4bf511162bee2f11ec84222087815"
SOURCE_PATHS = frozenset(['clearvla/mainline/causal_identity.py', 'clearvla/mainline/config.py', 'clearvla/mainline/data/identity_correspondence.py', 'clearvla/mainline/data/identity_region_annotations.py', 'clearvla/mainline/data/loading.py', 'clearvla/mainline/data/online_visual.py', 'clearvla/mainline/executed_world.py', 'clearvla/mainline/identity_supervision.py', 'clearvla/mainline/instruction_change.py', 'clearvla/mainline/interfaces.py', 'clearvla/mainline/model/canonical_grounding.py', 'clearvla/mainline/model/grounding.py', 'clearvla/mainline/model/image_feedback.py', 'clearvla/mainline/model/instruction_posterior.py', 'clearvla/mainline/model/observation_association.py', 'clearvla/mainline/model/policy.py', 'clearvla/mainline/model/source_measurement.py', 'clearvla/mainline/model/top.py', 'clearvla/mainline/runtime/causal_identity_migration.py', 'clearvla/mainline/runtime/checkpoints.py', 'clearvla/mainline/runtime/sam_structure_migration.py', 'clearvla/mainline/train.py', 'clearvla/mainline/training/engine.py', 'clearvla/mainline/training/identity.py', 'clearvla/mainline/training/identity_regions.py'])

def config_view(payload):
    result={**payload, "top":dict(payload["top"])}
    result["top"].pop("entity_image_feedback_mode",None)
    return result

def validate_selection(saved,current,source_digest):
    if source_digest != SOURCE_DIGEST:
        raise ValueError("SAM structure exploration requires the pinned B-v2 source identity")
    if saved.top.entity_image_feedback_mode != "none" or current.top.entity_image_feedback_mode not in {"slot_to_image_v1","region_fusion_v1"}:
        raise ValueError("SAM structure migration must introduce exactly one declared adapter")
    if saved.top.identity_supervision_mode != "rgbd_temporal_conditional_v2" or saved.top.entity_ownership_mode != "canonical_image_v1":
        raise ValueError("SAM structure exploration requires unchanged B-v2 ownership/supervision")
    if saved.data.data_profile != "calvin_relative_7d_v1" or current.data.data_profile != saved.data.data_profile:
        raise ValueError("SAM exploration source profile mismatch")

def migrate_state(saved,current,config):
    prefix="grounding.grounder.image_feedback."
    names={"down.weight","out.weight"}
    if config.top.entity_image_feedback_mode == "region_fusion_v1":
        names|={"mask.weight","depthwise.weight"}
    added={prefix+n for n in names}
    if set(current)-set(saved) != added or set(saved)-set(current):
        raise ValueError("SAM migration differs from the exact adapter-only state inventory")
    if torch.count_nonzero(current[prefix+"out.weight"]).item() != 0:
        raise ValueError("new image feedback must initialize at exact zero output")
    for name in saved:
        if saved[name].shape != current[name].shape or saved[name].dtype != current[name].dtype:
            raise ValueError("inherited tensor contract changed: "+name)
    return {name:current[name].detach().clone() if name in added else saved[name] for name in current}
