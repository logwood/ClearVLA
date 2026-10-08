"""Explicit B-v1 continuation: address memory or its matched no-memory control."""
import torch

B_V1_ADDRESS_MEMORY_V1 = "b_v1_address_memory_v1"
SOURCE_DIGEST = '15eb1669daa33fab4559ee9f88a978bc572b65033d336af809da6ea8b628e328'
SOURCE_PATHS = frozenset(['clearvla/mainline/config.py', 'clearvla/mainline/causal_identity.py', 'clearvla/mainline/model/grounding.py', 'clearvla/mainline/model/top.py', 'clearvla/mainline/model/policy.py', 'clearvla/mainline/model/canonical_grounding.py', 'clearvla/mainline/model/address_memory.py', 'clearvla/mainline/runtime/address_memory_migration.py', 'clearvla/mainline/runtime/checkpoints.py', 'clearvla/mainline/train.py'])
KEY = "grounding.grounder.address_memory_gain"


def config_view(payload):
    result = {**payload, "top":dict(payload["top"])}
    result["top"].pop("entity_address_memory_mode", None)
    return result


def validate_selection(saved, current, source_digest):
    if source_digest != SOURCE_DIGEST:
        raise ValueError("address-memory initialization requires pinned B-v1 source")
    if saved.top.entity_address_memory_mode != "none" or current.top.entity_address_memory_mode not in {"none","conditional_logits_v1"}:
        raise ValueError("migration must be the declared address-only trial/control")
    if saved.top.identity_supervision_mode != "rgbd_temporal_v1" or saved.top.entity_ownership_mode != "canonical_image_v1":
        raise ValueError("address-memory trial requires original B-v1 graph/objectives")
    if saved.data.data_profile != "calvin_relative_7d_v1" or current.data.data_profile != saved.data.data_profile:
        raise ValueError("address-memory source profile differs")


def migrate_state(saved, current, config):
    added = {KEY} if config.top.entity_address_memory_mode == "conditional_logits_v1" else set()
    if set(current)-set(saved) != added or set(saved)-set(current):
        raise ValueError("migration changed state outside declared scalar gains")
    if added:
        gain = current[KEY]
        if gain.ndim != 1 or gain.dtype != torch.float32 or not bool(torch.isfinite(gain).all()) or torch.count_nonzero(gain).item() != 0:
            raise ValueError("address gains must initialize finite exact-zero FP32")
    for name, value in saved.items():
        if value.shape != current[name].shape or value.dtype != current[name].dtype:
            raise ValueError("inherited tensor changed: "+name)
    return {n:current[n].detach().clone() if n in added else saved[n] for n in current}
