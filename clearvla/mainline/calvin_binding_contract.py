"""Lightweight, dependency-free CALVIN binding ABI constants.

Keeping these values outside the model/data modules prevents import cycles and
ensures that the sidecar, typed batch boundary and bridge cannot silently
drift to different role widths or component identities.
"""

CALVIN_OBJECT_BINDING_COMPONENT = "calvin_primary_object_binding_v2"
CALVIN_OBJECT_BINDING_INTENT = CALVIN_OBJECT_BINDING_COMPONENT
CALVIN_OBJECT_BINDING_ROLE_NAMES = (
    "red_block",
    "blue_block",
    "pink_block",
)
CALVIN_OBJECT_BINDING_ROLE_COUNT = len(CALVIN_OBJECT_BINDING_ROLE_NAMES)
CALVIN_OBJECT_BINDING_TARGET_COUNT = CALVIN_OBJECT_BINDING_ROLE_COUNT + 1
CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA = "clearvla-calvin-object-binding-v2"
CALVIN_OBJECT_BINDING_CONFIG = "calvin_primary_v2"
