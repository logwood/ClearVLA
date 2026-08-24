"""Compact serialized identity for the capability-named mainline.

Experiment labels and host paths are deliberately absent.  The manifest says
which mathematical components a checkpoint owns; executable tensor contracts
remain in :mod:`clearvla.mainline.interfaces` and the typed top modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, cast

CAPABILITY_NAME = "object_intent_dynamics_323"
CAPABILITY_SCHEMA = 36
LAYOUT_NAME = "clearvla_mainline"
LAYOUT_SCHEMA = 1
TOPOLOGY = (3, 2, 3)
INTERVALS = ((4, 8), (8, 16), (16, 32), (32, 48))


@dataclass(frozen=True)
class ComponentABI:
    """Stable component identities used for explicit checkpoint migration."""

    observation: str = "restored_v120_three_frame_flow_dino_progressive_g123_bank"
    top: str = "single_content_k_identity_incremental_stateless_intent_causal_w_near_far_camera_specific_effect_matched_semantic_geometry_p2_static_fact_single_precision_p3"
    bottom: str = "restored_v120_shared_seed_typed_bounded_dynamic_p1_query_only_four_active_plan_lanes_exact_g3_anchor_transition_evidence_mmdit_dense512_execution"
    training: str = "v120_mirrored_physical_flow_observed_current_grounding_partial_ot_neutral_status_camera_specific_future_loss_support_event_boost_v120_decay_three_owner_clip"
    runtime: str = "cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated_active_ablations_only"

    def validate(self) -> None:
        for name, value in self.as_dict().items():
            if not value or value.strip() != value or " " in value:
                raise ValueError(f"component ABI {name} is not a stable identifier")

    def as_dict(self) -> dict[str, str]:
        return {
            "observation": self.observation,
            "top": self.top,
            "bottom": self.bottom,
            "training": self.training,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class ArchitectureManifest:
    """Architecture identity without experiment or legacy launcher state."""

    capability: str = CAPABILITY_NAME
    schema: int = CAPABILITY_SCHEMA
    layout: str = LAYOUT_NAME
    layout_schema: int = LAYOUT_SCHEMA
    topology: tuple[int, int, int] = TOPOLOGY
    intervals: tuple[tuple[int, int], ...] = INTERVALS
    object_slots: int = 4
    language_required: bool = True
    components: ComponentABI = ComponentABI()

    def validate(self, *, require_current_schema: bool = True) -> None:
        """Validate the stable graph boundary.

        Stored mainline checkpoints may carry an older top/schema while still
        owning a byte-for-byte compatible bottom ABI.  Such a manifest is not
        a legal current graph or exact-resume target, but it must remain
        parseable for the explicit bottom-only migration path.  The relaxed
        mode therefore relaxes only the positive capability schema number; all
        structural identities and component ABIs remain validated.
        """

        if self.capability != CAPABILITY_NAME:
            raise ValueError("mainline capability identity is incompatible")
        if int(self.schema) <= 0 or (
            require_current_schema and int(self.schema) != CAPABILITY_SCHEMA
        ):
            raise ValueError("mainline capability identity/schema is incompatible")
        if self.layout != LAYOUT_NAME or int(self.layout_schema) != LAYOUT_SCHEMA:
            raise ValueError("mainline code-layout identity is incompatible")
        if tuple(self.topology) != TOPOLOGY:
            raise ValueError("mainline topology must be G3/W2/P3")
        if tuple(self.intervals) != INTERVALS:
            raise ValueError("mainline requires the four canonical future intervals")
        if int(self.object_slots) != 4:
            raise ValueError("mainline requires four global object slots")
        if not bool(self.language_required):
            raise ValueError("formal mainline training requires language")
        self.components.validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "schema": int(self.schema),
            "layout": self.layout,
            "layout_schema": int(self.layout_schema),
            "topology": list(self.topology),
            "intervals": [list(interval) for interval in self.intervals],
            "object_slots": int(self.object_slots),
            "language_required": bool(self.language_required),
            "components": self.components.as_dict(),
        }

    def digest(self) -> str:
        """Return the canonical manifest digest stored in a checkpoint."""

        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ARCHITECTURE_MANIFEST = ArchitectureManifest()


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"manifest {name} must be an integer")
    return int(value)


def _boolean(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"manifest {name} must be a boolean")


def manifest_from_mapping(
    value: Mapping[str, object],
    *,
    require_current_schema: bool = True,
) -> ArchitectureManifest:
    """Restore and validate one compact mainline architecture manifest."""

    topology_value = value.get("topology")
    intervals_value = value.get("intervals")
    components_value = value.get("components")
    if not isinstance(topology_value, (tuple, list)):
        raise ValueError("manifest topology must be a sequence")
    if not isinstance(intervals_value, (tuple, list)):
        raise ValueError("manifest intervals must be a sequence")
    if not isinstance(components_value, Mapping):
        raise ValueError("manifest components must be a mapping")

    topology_items = cast(tuple[object, ...] | list[object], topology_value)
    topology = tuple(
        _integer(item, name=f"topology[{index}]") for index, item in enumerate(topology_items)
    )
    if len(topology) != 3:
        raise ValueError("manifest topology must have three entries")

    interval_rows: list[tuple[int, int]] = []
    for index, raw_interval in enumerate(intervals_value):
        if not isinstance(raw_interval, (tuple, list)) or len(raw_interval) != 2:
            raise ValueError(f"manifest interval {index} must be a pair")
        interval_rows.append(
            (
                _integer(raw_interval[0], name=f"intervals[{index}][0]"),
                _integer(raw_interval[1], name=f"intervals[{index}][1]"),
            )
        )

    component_mapping = cast(Mapping[str, object], components_value)
    components = ComponentABI(
        observation=str(component_mapping.get("observation", "")),
        top=str(component_mapping.get("top", "")),
        bottom=str(component_mapping.get("bottom", "")),
        training=str(component_mapping.get("training", "")),
        runtime=str(component_mapping.get("runtime", "")),
    )
    manifest = ArchitectureManifest(
        capability=str(value.get("capability", "")),
        schema=_integer(value.get("schema", -1), name="schema"),
        layout=str(value.get("layout", "")),
        layout_schema=_integer(value.get("layout_schema", -1), name="layout_schema"),
        topology=cast(tuple[int, int, int], topology),
        intervals=tuple(interval_rows),
        object_slots=_integer(value.get("object_slots", -1), name="object_slots"),
        language_required=_boolean(value.get("language_required", False), name="language_required"),
        components=components,
    )
    manifest.validate(require_current_schema=require_current_schema)
    return manifest
