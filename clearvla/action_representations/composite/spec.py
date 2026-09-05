"""Versioned role-wise action representation contracts.

This module describes how one continuous flow state is partitioned into
semantic actuator roles and how non-continuous endpoint payloads remain
outside that state. It deliberately owns no robot codec or solver logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping, Sequence, cast

TemporalViewKind = Literal["identity", "bspline"]
EndpointPayloadKind = Literal["continuous", "logits", "labels"]
EndpointDistributionKind = Literal[
    "continuous",
    "categorical",
    "independent_binary",
    "deterministic",
]
EndpointUsage = Literal["action_owner", "conditioning", "auxiliary"]
TemporalAlignment = Literal["action_horizon", "clean_endpoint", "none"]
OwnerKind = Literal["role", "codec", "outlet"]


def _strict_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    keys = set(value)
    optional_keys = set() if optional is None else optional
    unknown = keys.difference(required | optional_keys)
    missing = required.difference(keys)
    if unknown or missing:
        raise ValueError(
            f"invalid {name} keys; missing={sorted(missing)}, extra={sorted(unknown)}"
        )


def _label(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _optional_label(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _label(value, name=name)


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    sequence = _sequence(value, name=name)
    return tuple(_integer(item, name=f"{name} entry") for item in sequence)


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    sequence = _sequence(value, name=name)
    result: list[float] = []
    for item in sequence:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} entries must be real numbers")
        result.append(float(item))
    return tuple(result)


@dataclass(frozen=True)
class ContinuousRoleSpec:
    """One ordered subset of the floating-point ODE state.

    ``retain_raw`` records whether exact sampled rows travel beside the
    structured temporal view. This is the safe migration mode for a lossy
    chart and mirrors the current mainline's raw-plus-B-spine principle.
    """

    role_id: str
    state_indices: tuple[int, ...]
    semantic_quantity: str
    geometry_id: str
    decode_group_id: str
    temporal_view_kind: TemporalViewKind
    view_spec_fingerprint: str
    retain_raw: bool = True
    allow_lossy_chart: bool = False
    constraint_policy_id: str = "caller_owned"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_id", _label(self.role_id, name="role_id"))
        object.__setattr__(
            self,
            "state_indices",
            _integer_tuple(self.state_indices, name="state_indices"),
        )
        object.__setattr__(
            self,
            "semantic_quantity",
            _label(self.semantic_quantity, name="semantic_quantity"),
        )
        object.__setattr__(self, "geometry_id", _label(self.geometry_id, name="geometry_id"))
        object.__setattr__(
            self,
            "decode_group_id",
            _label(self.decode_group_id, name="decode_group_id"),
        )
        object.__setattr__(
            self,
            "view_spec_fingerprint",
            _label(self.view_spec_fingerprint, name="view_spec_fingerprint"),
        )
        object.__setattr__(
            self,
            "constraint_policy_id",
            _label(self.constraint_policy_id, name="constraint_policy_id"),
        )
        object.__setattr__(
            self,
            "retain_raw",
            _boolean(self.retain_raw, name="retain_raw"),
        )
        object.__setattr__(
            self,
            "allow_lossy_chart",
            _boolean(self.allow_lossy_chart, name="allow_lossy_chart"),
        )
        if not self.state_indices:
            raise ValueError("continuous role state_indices cannot be empty")
        if any(index < 0 for index in self.state_indices):
            raise ValueError("continuous role state_indices must be non-negative")
        if len(set(self.state_indices)) != len(self.state_indices):
            raise ValueError("continuous role state_indices must be unique")
        if self.temporal_view_kind not in {"identity", "bspline"}:
            raise ValueError(
                f"unsupported temporal_view_kind: {self.temporal_view_kind!r}"
            )

    @property
    def width(self) -> int:
        return len(self.state_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "state_indices": list(self.state_indices),
            "semantic_quantity": self.semantic_quantity,
            "geometry_id": self.geometry_id,
            "decode_group_id": self.decode_group_id,
            "temporal_view_kind": self.temporal_view_kind,
            "view_spec_fingerprint": self.view_spec_fingerprint,
            "retain_raw": self.retain_raw,
            "allow_lossy_chart": self.allow_lossy_chart,
            "constraint_policy_id": self.constraint_policy_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContinuousRoleSpec:
        _strict_keys(
            value,
            name="continuous role spec",
            required={
                "role_id",
                "state_indices",
                "semantic_quantity",
                "geometry_id",
                "decode_group_id",
                "temporal_view_kind",
                "view_spec_fingerprint",
            },
            optional={"retain_raw", "allow_lossy_chart", "constraint_policy_id"},
        )
        return cls(
            role_id=value["role_id"],
            state_indices=_integer_tuple(value["state_indices"], name="state_indices"),
            semantic_quantity=value["semantic_quantity"],
            geometry_id=value["geometry_id"],
            decode_group_id=value["decode_group_id"],
            temporal_view_kind=value["temporal_view_kind"],
            view_spec_fingerprint=value["view_spec_fingerprint"],
            retain_raw=_boolean(value.get("retain_raw", True), name="retain_raw"),
            allow_lossy_chart=_boolean(
                value.get("allow_lossy_chart", False),
                name="allow_lossy_chart",
            ),
            constraint_policy_id=value.get("constraint_policy_id", "caller_owned"),
        )


@dataclass(frozen=True)
class EndpointSpec:
    """Typed non-ODE payload produced at the clean flow endpoint.

    Floating logits are legal here: ownership and update semantics, not dtype
    alone, distinguish them from continuous ODE coordinates.
    """

    role_id: str
    decode_group_id: str | None
    semantic_kind: str
    payload_kind: EndpointPayloadKind
    payload_shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    temporal_alignment: TemporalAlignment
    distribution_kind: EndpointDistributionKind
    vocabulary_id: str | None
    usage: EndpointUsage
    producer_id: str
    action_mapping: str
    boundary_policy: str
    interpolation_policy: str = "none"
    label_values: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_id", _label(self.role_id, name="role_id"))
        object.__setattr__(
            self,
            "decode_group_id",
            _optional_label(self.decode_group_id, name="decode_group_id"),
        )
        for name in (
            "semantic_kind",
            "producer_id",
            "action_mapping",
            "boundary_policy",
            "interpolation_policy",
        ):
            object.__setattr__(self, name, _label(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "payload_shape",
            _integer_tuple(self.payload_shape, name="payload_shape"),
        )
        axes = _sequence(self.axis_names, name="axis_names")
        object.__setattr__(
            self,
            "axis_names",
            tuple(_label(value, name="axis_names entry") for value in axes),
        )
        object.__setattr__(
            self,
            "vocabulary_id",
            _optional_label(self.vocabulary_id, name="vocabulary_id"),
        )
        object.__setattr__(
            self,
            "label_values",
            _integer_tuple(self.label_values, name="label_values"),
        )
        if self.payload_kind not in {"continuous", "logits", "labels"}:
            raise ValueError(f"unsupported endpoint payload_kind: {self.payload_kind!r}")
        if self.temporal_alignment not in {"action_horizon", "clean_endpoint", "none"}:
            raise ValueError(
                f"unsupported endpoint temporal_alignment: {self.temporal_alignment!r}"
            )
        if self.distribution_kind not in {
            "continuous",
            "categorical",
            "independent_binary",
            "deterministic",
        }:
            raise ValueError(
                f"unsupported endpoint distribution_kind: {self.distribution_kind!r}"
            )
        if self.usage not in {"action_owner", "conditioning", "auxiliary"}:
            raise ValueError(f"unsupported endpoint usage: {self.usage!r}")
        if any(value < 1 for value in self.payload_shape):
            raise ValueError("endpoint payload_shape entries must be positive")
        if len(self.axis_names) != len(self.payload_shape):
            raise ValueError("endpoint axis_names must name every payload axis")
        if len(set(self.axis_names)) != len(self.axis_names):
            raise ValueError("endpoint axis_names must be unique")
        if self.usage == "action_owner" and self.decode_group_id is None:
            raise ValueError("an action-owner endpoint requires a decode_group_id")
        if self.payload_kind in {"logits", "labels"} and self.vocabulary_id is None:
            raise ValueError("logit/label endpoints require a vocabulary_id")
        if self.payload_kind == "labels" and not self.label_values:
            raise ValueError("label endpoints require declared label_values")
        if self.payload_kind != "labels" and self.label_values:
            raise ValueError("label_values are valid only for label endpoints")
        if len(set(self.label_values)) != len(self.label_values):
            raise ValueError("endpoint label_values must be unique")
        if self.payload_kind == "continuous" and self.distribution_kind not in {
            "continuous",
            "deterministic",
        }:
            raise ValueError("continuous endpoint payload has an incompatible distribution")
        if self.payload_kind == "logits" and self.distribution_kind not in {
            "categorical",
            "independent_binary",
        }:
            raise ValueError("logit endpoint payload has an incompatible distribution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "decode_group_id": self.decode_group_id,
            "semantic_kind": self.semantic_kind,
            "payload_kind": self.payload_kind,
            "payload_shape": list(self.payload_shape),
            "axis_names": list(self.axis_names),
            "temporal_alignment": self.temporal_alignment,
            "distribution_kind": self.distribution_kind,
            "vocabulary_id": self.vocabulary_id,
            "usage": self.usage,
            "producer_id": self.producer_id,
            "action_mapping": self.action_mapping,
            "boundary_policy": self.boundary_policy,
            "interpolation_policy": self.interpolation_policy,
            "label_values": list(self.label_values),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EndpointSpec:
        _strict_keys(
            value,
            name="endpoint spec",
            required={
                "role_id",
                "decode_group_id",
                "semantic_kind",
                "payload_kind",
                "payload_shape",
                "axis_names",
                "temporal_alignment",
                "distribution_kind",
                "vocabulary_id",
                "usage",
                "producer_id",
                "action_mapping",
                "boundary_policy",
            },
            optional={"interpolation_policy", "label_values"},
        )
        return cls(
            role_id=value["role_id"],
            decode_group_id=value["decode_group_id"],
            semantic_kind=value["semantic_kind"],
            payload_kind=value["payload_kind"],
            payload_shape=_integer_tuple(value["payload_shape"], name="payload_shape"),
            axis_names=tuple(_sequence(value["axis_names"], name="axis_names")),
            temporal_alignment=value["temporal_alignment"],
            distribution_kind=value["distribution_kind"],
            vocabulary_id=value["vocabulary_id"],
            usage=value["usage"],
            producer_id=value["producer_id"],
            action_mapping=value["action_mapping"],
            boundary_policy=value["boundary_policy"],
            interpolation_policy=value.get("interpolation_policy", "none"),
            label_values=_integer_tuple(value.get("label_values", ()), name="label_values"),
        )


@dataclass(frozen=True)
class OwnerRef:
    """Structured reference to the sole deployed-action owner for one group."""

    kind: OwnerKind
    owner_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"role", "codec", "outlet"}:
            raise ValueError(f"unsupported owner kind: {self.kind!r}")
        object.__setattr__(self, "owner_id", _label(self.owner_id, name="owner_id"))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "owner_id": self.owner_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OwnerRef:
        _strict_keys(
            value,
            name="owner reference",
            required={"kind", "owner_id"},
        )
        return cls(kind=value["kind"], owner_id=value["owner_id"])


@dataclass(frozen=True)
class DecodeGroupSpec:
    """One disjoint slice of the deployed native action and its final owner."""

    group_id: str
    action_indices: tuple[int, ...]
    final_owner: OwnerRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _label(self.group_id, name="group_id"))
        object.__setattr__(
            self,
            "action_indices",
            _integer_tuple(self.action_indices, name="action_indices"),
        )
        if not isinstance(self.final_owner, OwnerRef):
            raise TypeError("final_owner must be an OwnerRef")
        if not self.action_indices:
            raise ValueError("decode group action_indices cannot be empty")
        if any(index < 0 for index in self.action_indices):
            raise ValueError("decode group action_indices must be non-negative")
        if len(set(self.action_indices)) != len(self.action_indices):
            raise ValueError("decode group action_indices must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "action_indices": list(self.action_indices),
            "final_owner": self.final_owner.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecodeGroupSpec:
        _strict_keys(
            value,
            name="decode group spec",
            required={"group_id", "action_indices", "final_owner"},
        )
        final_owner = value["final_owner"]
        if not isinstance(final_owner, Mapping):
            raise TypeError("decode group final_owner must be a mapping")
        return cls(
            group_id=value["group_id"],
            action_indices=_integer_tuple(value["action_indices"], name="action_indices"),
            final_owner=OwnerRef.from_dict(final_owner),
        )


@dataclass(frozen=True)
class CompositeActionSpec:
    """Immutable identity for a role-wise continuous/endpoint action bundle."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    REPRESENTATION_NAME: ClassVar[str] = "clearvla.composite_action"

    sample_times: tuple[float, ...]
    state_dim: int
    action_dim: int
    continuous_roles: tuple[ContinuousRoleSpec, ...]
    endpoint_specs: tuple[EndpointSpec, ...]
    decode_groups: tuple[DecodeGroupSpec, ...]
    codec_id: str
    normalizer_id: str
    causal_boundary_id: str
    outlet_id: str | None = None
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_times",
            _float_tuple(self.sample_times, name="sample_times"),
        )
        object.__setattr__(self, "state_dim", _integer(self.state_dim, name="state_dim"))
        object.__setattr__(self, "action_dim", _integer(self.action_dim, name="action_dim"))
        object.__setattr__(self, "continuous_roles", tuple(self.continuous_roles))
        object.__setattr__(self, "endpoint_specs", tuple(self.endpoint_specs))
        object.__setattr__(self, "decode_groups", tuple(self.decode_groups))
        object.__setattr__(self, "codec_id", _label(self.codec_id, name="codec_id"))
        object.__setattr__(
            self,
            "normalizer_id",
            _label(self.normalizer_id, name="normalizer_id"),
        )
        object.__setattr__(
            self,
            "causal_boundary_id",
            _label(self.causal_boundary_id, name="causal_boundary_id"),
        )
        object.__setattr__(
            self,
            "outlet_id",
            _optional_label(self.outlet_id, name="outlet_id"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _integer(self.schema_version, name="schema_version"),
        )
        self.validate()

    @property
    def horizon(self) -> int:
        return len(self.sample_times)

    def validate(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported composite schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )
        if self.horizon < 1:
            raise ValueError("sample_times cannot be empty")
        if not all(math.isfinite(value) for value in self.sample_times):
            raise ValueError("sample_times must be finite")
        if any(b <= a for a, b in zip(self.sample_times, self.sample_times[1:])):
            raise ValueError("sample_times must be strictly increasing")
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        if not self.continuous_roles:
            raise ValueError("at least one continuous role is required")
        if not self.decode_groups:
            raise ValueError("at least one decode group is required")
        if not all(isinstance(role, ContinuousRoleSpec) for role in self.continuous_roles):
            raise TypeError("continuous_roles must contain ContinuousRoleSpec values")
        if not all(isinstance(role, EndpointSpec) for role in self.endpoint_specs):
            raise TypeError("endpoint_specs must contain EndpointSpec values")
        if not all(isinstance(group, DecodeGroupSpec) for group in self.decode_groups):
            raise TypeError("decode_groups must contain DecodeGroupSpec values")

        continuous_ids = [role.role_id for role in self.continuous_roles]
        endpoint_ids = [role.role_id for role in self.endpoint_specs]
        all_ids = continuous_ids + endpoint_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("continuous and endpoint role_id values must be globally unique")

        state_indices = [
            index for role in self.continuous_roles for index in role.state_indices
        ]
        if sorted(state_indices) != list(range(self.state_dim)):
            raise ValueError(
                "continuous role state_indices must partition [0,state_dim) exactly"
            )

        group_ids = [group.group_id for group in self.decode_groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("decode group ids must be unique")
        action_indices = [
            index for group in self.decode_groups for index in group.action_indices
        ]
        if sorted(action_indices) != list(range(self.action_dim)):
            raise ValueError(
                "decode group action_indices must partition [0,action_dim) exactly"
            )
        known_groups = set(group_ids)
        for role in self.continuous_roles:
            if role.decode_group_id not in known_groups:
                raise ValueError(
                    f"role {role.role_id!r} references unknown decode group "
                    f"{role.decode_group_id!r}"
                )
        for endpoint in self.endpoint_specs:
            if (
                endpoint.decode_group_id is not None
                and endpoint.decode_group_id not in known_groups
            ):
                raise ValueError(
                    f"endpoint {endpoint.role_id!r} references unknown decode group "
                    f"{endpoint.decode_group_id!r}"
                )
            if endpoint.temporal_alignment == "action_horizon":
                if not endpoint.payload_shape or endpoint.payload_shape[0] != self.horizon:
                    raise ValueError(
                        f"endpoint {endpoint.role_id!r} must align its first payload "
                        "axis with the action horizon"
                    )

        roles_by_group = {
            group_id: {
                role.role_id
                for role in (*self.continuous_roles, *self.endpoint_specs)
                if role.decode_group_id == group_id
            }
            for group_id in group_ids
        }
        endpoint_by_id = {endpoint.role_id: endpoint for endpoint in self.endpoint_specs}
        for group in self.decode_groups:
            members = roles_by_group[group.group_id]
            if not members:
                raise ValueError(f"decode group {group.group_id!r} has no attached role")
            owner = group.final_owner
            if owner.kind == "role":
                if owner.owner_id not in members:
                    raise ValueError(
                        f"decode group {group.group_id!r} final role owner is not attached"
                    )
                endpoint_owner = endpoint_by_id.get(owner.owner_id)
                if endpoint_owner is not None and endpoint_owner.usage != "action_owner":
                    raise ValueError("an auxiliary/conditioning endpoint cannot own action output")
            elif owner.kind == "codec" and owner.owner_id != self.codec_id:
                raise ValueError("decode group codec owner must match codec_id")
            elif owner.kind == "outlet":
                if self.outlet_id is None or owner.owner_id != self.outlet_id:
                    raise ValueError("decode group outlet owner must match outlet_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.REPRESENTATION_NAME,
            "schema_version": self.schema_version,
            "sample_times": list(self.sample_times),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "continuous_roles": [role.to_dict() for role in self.continuous_roles],
            "endpoint_specs": [role.to_dict() for role in self.endpoint_specs],
            "decode_groups": [group.to_dict() for group in self.decode_groups],
            "codec_id": self.codec_id,
            "normalizer_id": self.normalizer_id,
            "causal_boundary_id": self.causal_boundary_id,
            "outlet_id": self.outlet_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompositeActionSpec:
        _strict_keys(
            value,
            name="composite action spec",
            required={
                "state_dim",
                "action_dim",
                "sample_times",
                "continuous_roles",
                "endpoint_specs",
                "decode_groups",
                "codec_id",
                "normalizer_id",
                "causal_boundary_id",
            },
            optional={"representation", "schema_version", "outlet_id"},
        )
        representation = value.get("representation", cls.REPRESENTATION_NAME)
        if representation != cls.REPRESENTATION_NAME:
            raise ValueError(f"not a {cls.REPRESENTATION_NAME} specification")
        continuous_roles = _sequence(value["continuous_roles"], name="continuous_roles")
        endpoint_specs = _sequence(value["endpoint_specs"], name="endpoint_specs")
        decode_groups = _sequence(value["decode_groups"], name="decode_groups")
        if not all(isinstance(role, Mapping) for role in continuous_roles):
            raise TypeError("serialized continuous roles must be mappings")
        if not all(isinstance(role, Mapping) for role in endpoint_specs):
            raise TypeError("serialized endpoint specs must be mappings")
        if not all(isinstance(group, Mapping) for group in decode_groups):
            raise TypeError("serialized decode groups must be mappings")
        return cls(
            sample_times=_float_tuple(value["sample_times"], name="sample_times"),
            state_dim=_integer(value["state_dim"], name="state_dim"),
            action_dim=_integer(value["action_dim"], name="action_dim"),
            continuous_roles=tuple(
                ContinuousRoleSpec.from_dict(cast(Mapping[str, Any], role))
                for role in continuous_roles
            ),
            endpoint_specs=tuple(
                EndpointSpec.from_dict(cast(Mapping[str, Any], role))
                for role in endpoint_specs
            ),
            decode_groups=tuple(
                DecodeGroupSpec.from_dict(cast(Mapping[str, Any], group))
                for group in decode_groups
            ),
            codec_id=value["codec_id"],
            normalizer_id=value["normalizer_id"],
            causal_boundary_id=value["causal_boundary_id"],
            outlet_id=value.get("outlet_id"),
            schema_version=_integer(
                value.get("schema_version", cls.CURRENT_SCHEMA_VERSION),
                name="schema_version",
            ),
        )

    def identity_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value["sample_times_hex"] = [number.hex() for number in self.sample_times]
        del value["sample_times"]
        return value

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.identity_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CompositeActionSpec",
    "ContinuousRoleSpec",
    "DecodeGroupSpec",
    "EndpointDistributionKind",
    "EndpointPayloadKind",
    "EndpointSpec",
    "EndpointUsage",
    "OwnerKind",
    "OwnerRef",
    "TemporalAlignment",
    "TemporalViewKind",
]


