"""Role-wise composition of continuous temporal charts and typed endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .charts import RolePayload, TemporalRoleChart, _validate_supplied_grid
from .spec import (
    CompositeActionSpec,
    ContinuousRoleSpec,
    EndpointPayloadKind,
    EndpointSpec,
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class EndpointPayload:
    """One endpoint-owned payload that never enters the continuous ODE state."""

    role_id: str
    payload_kind: EndpointPayloadKind
    value: Tensor

    def to(self, *args: Any, **kwargs: Any) -> EndpointPayload:
        requested = self.value.to(*args, **kwargs)
        if self.payload_kind == "labels":
            # Labels are symbols, not numerical coordinates.  Going through a
            # requested floating or narrower-integer dtype and then casting back
            # can silently change a legal value (for example 257 -> BF16 -> 256).
            # Use the requested result only to resolve the destination device;
            # copy the original integer values directly to that device.
            copy_on_same_device = (
                requested.device == self.value.device and requested is not self.value
            )
            moved = self.value.to(
                device=requested.device,
                dtype=self.value.dtype,
                copy=copy_on_same_device,
            )
        else:
            moved = requested
        return EndpointPayload(self.role_id, self.payload_kind, moved)

    def detach(self) -> EndpointPayload:
        return EndpointPayload(self.role_id, self.payload_kind, self.value.detach())

    def as_state_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "payload_kind": self.payload_kind,
            "value": self.value,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> EndpointPayload:
        if set(value) != {"role_id", "payload_kind", "value"} or not isinstance(
            value["value"], Tensor
        ):
            raise ValueError("invalid endpoint payload")
        role_id = value["role_id"]
        payload_kind = value["payload_kind"]
        if not isinstance(role_id, str) or not role_id.strip():
            raise TypeError("endpoint payload role_id must be a non-empty string")
        if payload_kind not in {"continuous", "logits", "labels"}:
            raise ValueError("endpoint payload has an invalid payload_kind")
        return cls(role_id=role_id, payload_kind=payload_kind, value=value["value"])


@dataclass(frozen=True)
class CompositeActionPayload:
    """Serializable result of encoding one continuous state and its sidecars."""

    roles: tuple[RolePayload, ...]
    endpoints: tuple[EndpointPayload, ...]
    schema_version: int
    spec_fingerprint: str
    source_state_values_preserved: bool

    def to(self, *args: Any, **kwargs: Any) -> CompositeActionPayload:
        moved_roles = tuple(role.to(*args, **kwargs) for role in self.roles)

        def role_dtypes_preserved(before: RolePayload, after: RolePayload) -> bool:
            if before.raw is not None:
                if after.raw is None or before.raw.dtype != after.raw.dtype:
                    return False
            for key, value in before.chart_state.items():
                if isinstance(value, Tensor):
                    moved_value = after.chart_state.get(key)
                    if not isinstance(moved_value, Tensor) or value.dtype != moved_value.dtype:
                        return False
            return True

        return CompositeActionPayload(
            roles=moved_roles,
            endpoints=tuple(endpoint.to(*args, **kwargs) for endpoint in self.endpoints),
            schema_version=self.schema_version,
            spec_fingerprint=self.spec_fingerprint,
            source_state_values_preserved=(
                self.source_state_values_preserved
                and all(
                    role_dtypes_preserved(before, after)
                    for before, after in zip(self.roles, moved_roles)
                )
            ),
        )

    def detach(self) -> CompositeActionPayload:
        return CompositeActionPayload(
            roles=tuple(role.detach() for role in self.roles),
            endpoints=tuple(endpoint.detach() for endpoint in self.endpoints),
            schema_version=self.schema_version,
            spec_fingerprint=self.spec_fingerprint,
            source_state_values_preserved=self.source_state_values_preserved,
        )

    def as_state_dict(self) -> dict[str, Any]:
        return {
            "roles": [role.as_state_dict() for role in self.roles],
            "endpoints": [endpoint.as_state_dict() for endpoint in self.endpoints],
            "schema_version": self.schema_version,
            "spec_fingerprint": self.spec_fingerprint,
            "source_state_values_preserved": self.source_state_values_preserved,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> CompositeActionPayload:
        expected = {
            "roles",
            "endpoints",
            "schema_version",
            "spec_fingerprint",
            "source_state_values_preserved",
        }
        if set(value) != expected:
            raise ValueError("invalid composite action payload keys")
        roles = value["roles"]
        endpoints = value["endpoints"]
        if (
            isinstance(roles, (str, bytes))
            or isinstance(endpoints, (str, bytes))
            or not isinstance(roles, Sequence)
            or not isinstance(endpoints, Sequence)
        ):
            raise TypeError("serialized composite roles/endpoints must be sequences")
        if not all(isinstance(role, Mapping) for role in roles):
            raise TypeError("serialized composite roles must be mappings")
        if not all(isinstance(endpoint, Mapping) for endpoint in endpoints):
            raise TypeError("serialized composite endpoints must be mappings")
        schema_version = value["schema_version"]
        spec_fingerprint = value["spec_fingerprint"]
        source_state_values_preserved = value["source_state_values_preserved"]
        if type(schema_version) is not int:
            raise TypeError("payload schema_version must be an integer")
        if not isinstance(spec_fingerprint, str) or not spec_fingerprint.strip():
            raise TypeError("payload spec_fingerprint must be a non-empty string")
        if type(source_state_values_preserved) is not bool:
            raise TypeError("source_state_values_preserved must be a boolean")
        return cls(
            roles=tuple(RolePayload.from_state_dict(role) for role in roles),
            endpoints=tuple(
                EndpointPayload.from_state_dict(endpoint) for endpoint in endpoints
            ),
            schema_version=schema_version,
            spec_fingerprint=spec_fingerprint,
            source_state_values_preserved=source_state_values_preserved,
        )


@dataclass(frozen=True)
class DecodedCompositeState:
    """One selected continuous-state view and its endpoint validity status."""

    continuous_state: Tensor
    endpoints: Mapping[str, Tensor]
    view: str
    role_lossless: Mapping[str, bool]
    role_source_equal: Mapping[str, bool]
    endpoint_binding: str
    requires_endpoint_refresh: bool


class CompositeActionRepresentation(nn.Module):
    """Compose independent temporal views without teaching a solver role semantics.

    This is an outer-boundary representation tool.  The wrapped B-spline chart
    performs validation that may synchronize an accelerator, so this module is
    intentionally not suitable for calls inside an ODE stage loop.
    """

    def __init__(
        self,
        spec: CompositeActionSpec,
        charts: Mapping[str, nn.Module],
    ) -> None:
        super().__init__()
        self.spec = spec
        supplied = dict(charts)
        expected_ids = {role.role_id for role in spec.continuous_roles}
        if set(supplied) != expected_ids:
            raise ValueError(
                "charts must contain exactly the continuous role ids; "
                f"expected={sorted(expected_ids)}, got={sorted(supplied)}"
            )
        self.charts = nn.ModuleDict(supplied)
        for role in spec.continuous_roles:
            chart = self._chart(role)
            if chart.chart_kind != role.temporal_view_kind:
                raise ValueError(f"role {role.role_id!r} chart kind does not match its spec")
            if chart.chart_fingerprint != role.view_spec_fingerprint:
                raise ValueError(
                    f"role {role.role_id!r} chart fingerprint does not match its spec"
                )
            if int(chart.horizon) != spec.horizon or int(chart.width) != role.width:
                raise ValueError(
                    f"role {role.role_id!r} chart horizon/width does not match its spec"
                )
            chart_grid = tuple(float(value).hex() for value in chart.sample_times)
            spec_grid = tuple(value.hex() for value in spec.sample_times)
            if chart_grid != spec_grid:
                raise ValueError(
                    f"role {role.role_id!r} chart sample grid does not match the "
                    "composite spec"
                )
            if not bool(chart.is_lossless) and not (
                role.retain_raw or role.allow_lossy_chart
            ):
                raise ValueError(
                    f"lossy role {role.role_id!r} must retain raw rows or explicitly "
                    "allow a lossy chart"
                )

    def _chart(self, role: ContinuousRoleSpec) -> TemporalRoleChart:
        chart = self.charts[role.role_id]
        if not isinstance(chart, TemporalRoleChart):
            raise TypeError(f"role {role.role_id!r} chart does not implement the chart ABI")
        return chart

    def _validate_continuous(self, value: Tensor) -> None:
        if value.ndim != 3 or tuple(value.shape[1:]) != (
            self.spec.horizon,
            self.spec.state_dim,
        ):
            raise ValueError(
                "continuous state must be "
                f"[B,{self.spec.horizon},{self.spec.state_dim}], got {tuple(value.shape)}"
            )
        if int(value.shape[0]) < 1:
            raise ValueError("continuous state batch cannot be empty")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("continuous state must be finite and floating point")

    def _authoritative_times(
        self,
        supplied: Tensor | Sequence[float] | None,
    ) -> tuple[float, ...]:
        _validate_supplied_grid(
            supplied,
            expected=self.spec.sample_times,
            mismatch_message=(
                "times do not match the immutable composite sample grid"
            ),
        )
        return self.spec.sample_times

    @staticmethod
    def _validate_endpoint(
        endpoint: EndpointSpec,
        value: Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"endpoint {endpoint.role_id!r} must be a tensor")
        if tuple(value.shape) != (batch, *endpoint.payload_shape):
            raise ValueError(
                f"endpoint {endpoint.role_id!r} must have shape "
                f"{(batch, *endpoint.payload_shape)}, got {tuple(value.shape)}"
            )
        if value.device != device:
            raise ValueError("continuous state and endpoints must share a device")
        if endpoint.payload_kind in {"continuous", "logits"}:
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"endpoint {endpoint.role_id!r} must be finite floating point"
                )
        elif value.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"label endpoint {endpoint.role_id!r} must use an integer tensor"
            )
        else:
            representable_labels = tuple(
                label
                for label in endpoint.label_values
                if _INT64_MIN <= label <= _INT64_MAX
            )
            allowed = torch.tensor(
                representable_labels,
                device=value.device,
                dtype=torch.int64,
            )
            # All accepted input integer dtypes embed exactly into int64.  This
            # prevents a negative vocabulary symbol from wrapping to uint8 and
            # aliasing an unrelated label (for example -1 -> 255).
            canonical_value = value.to(dtype=torch.int64)
            if not bool(torch.isin(canonical_value, allowed).all()):
                raise ValueError(
                    f"label endpoint {endpoint.role_id!r} contains a value outside "
                    "its declared vocabulary"
                )

    def encode(
        self,
        continuous_state: Tensor,
        *,
        endpoints: Mapping[str, Tensor] | None = None,
        times: Tensor | Sequence[float] | None = None,
        origins: Mapping[str, Tensor] | None = None,
    ) -> CompositeActionPayload:
        """Encode continuous roles while keeping endpoint payloads out-of-band."""

        self._validate_continuous(continuous_state)
        endpoint_values = {} if endpoints is None else dict(endpoints)
        expected_endpoints = {endpoint.role_id for endpoint in self.spec.endpoint_specs}
        if set(endpoint_values) != expected_endpoints:
            raise ValueError(
                "endpoints must contain exactly the declared endpoint role ids; "
                f"expected={sorted(expected_endpoints)}, got={sorted(endpoint_values)}"
            )
        origin_values = {} if origins is None else dict(origins)
        known_roles = {role.role_id for role in self.spec.continuous_roles}
        if not set(origin_values).issubset(known_roles):
            raise ValueError("origins contain an unknown continuous role id")
        authoritative_times = self._authoritative_times(times)

        role_payloads: list[RolePayload] = []
        for role in self.spec.continuous_roles:
            index = torch.tensor(
                role.state_indices,
                device=continuous_state.device,
                dtype=torch.long,
            )
            role_value = continuous_state.index_select(-1, index)
            chart = self._chart(role)
            chart_state = chart.encode(
                role_value,
                times=authoritative_times,
                origin=origin_values.get(role.role_id),
            )
            role_payloads.append(
                RolePayload(
                    role_id=role.role_id,
                    chart_kind=chart.chart_kind,
                    chart_fingerprint=chart.chart_fingerprint,
                    chart_state=dict(chart_state),
                    raw=role_value.clone() if role.retain_raw else None,
                )
            )

        endpoint_payloads: list[EndpointPayload] = []
        for endpoint in self.spec.endpoint_specs:
            value = endpoint_values[endpoint.role_id]
            self._validate_endpoint(
                endpoint,
                value,
                batch=int(continuous_state.shape[0]),
                device=continuous_state.device,
            )
            endpoint_payloads.append(
                EndpointPayload(endpoint.role_id, endpoint.payload_kind, value.clone())
            )
        return CompositeActionPayload(
            roles=tuple(role_payloads),
            endpoints=tuple(endpoint_payloads),
            schema_version=self.spec.schema_version,
            spec_fingerprint=self.spec.fingerprint,
            source_state_values_preserved=True,
        )

    def _validate_payload(self, payload: CompositeActionPayload) -> None:
        if not isinstance(payload, CompositeActionPayload):
            raise TypeError("payload must be a CompositeActionPayload")
        if payload.schema_version != self.spec.schema_version:
            raise ValueError("payload schema version does not match the representation")
        if payload.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("payload fingerprint does not match the representation")
        if type(payload.source_state_values_preserved) is not bool:
            raise TypeError("payload source_state_values_preserved must be a boolean")
        expected_roles = [role.role_id for role in self.spec.continuous_roles]
        actual_roles = [role.role_id for role in payload.roles]
        if actual_roles != expected_roles:
            raise ValueError("payload continuous role order/identity does not match the spec")
        expected_endpoints = [endpoint.role_id for endpoint in self.spec.endpoint_specs]
        actual_endpoints = [endpoint.role_id for endpoint in payload.endpoints]
        if actual_endpoints != expected_endpoints:
            raise ValueError("payload endpoint order/identity does not match the spec")
        payload_batch: int | None = None
        payload_device: torch.device | None = None
        for role, role_payload in zip(self.spec.continuous_roles, payload.roles):
            chart = self._chart(role)
            if role_payload.chart_kind != chart.chart_kind:
                raise ValueError(f"payload chart kind differs for role {role.role_id!r}")
            if role_payload.chart_fingerprint != chart.chart_fingerprint:
                raise ValueError(
                    f"payload chart fingerprint differs for role {role.role_id!r}"
                )
            has_raw = role_payload.raw is not None
            if has_raw != role.retain_raw:
                raise ValueError(
                    f"payload raw-bypass presence differs for role {role.role_id!r}"
                )
            if role_payload.raw is not None:
                raw = role_payload.raw
                if raw.ndim != 3 or tuple(raw.shape[1:]) != (
                    self.spec.horizon,
                    role.width,
                ):
                    raise ValueError(f"payload raw bypass has wrong shape for {role.role_id!r}")
                if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
                    raise ValueError(f"payload raw bypass must be finite for {role.role_id!r}")
            chart_value = chart.decode(role_payload.chart_state)
            if chart_value.ndim != 3 or tuple(chart_value.shape[1:]) != (
                self.spec.horizon,
                role.width,
            ):
                raise ValueError(f"payload chart state has wrong shape for {role.role_id!r}")
            if not chart_value.is_floating_point() or not bool(
                torch.isfinite(chart_value).all()
            ):
                raise ValueError(f"payload chart state must be finite for {role.role_id!r}")
            reference = role_payload.raw if role_payload.raw is not None else chart_value
            assert reference is not None
            if (
                int(chart_value.shape[0]) != int(reference.shape[0])
                or chart_value.device != reference.device
            ):
                raise ValueError(
                    f"payload chart/raw batch or device differs for role {role.role_id!r}"
                )
            if payload_batch is None:
                payload_batch = int(reference.shape[0])
                payload_device = reference.device
            elif (
                int(reference.shape[0]) != payload_batch
                or reference.device != payload_device
            ):
                raise ValueError("payload roles must share batch size and device")
        assert payload_batch is not None and payload_device is not None
        for endpoint, endpoint_payload in zip(self.spec.endpoint_specs, payload.endpoints):
            if endpoint_payload.payload_kind != endpoint.payload_kind:
                raise ValueError(
                    f"payload kind differs for endpoint {endpoint.role_id!r}"
                )
            self._validate_endpoint(
                endpoint,
                endpoint_payload.value,
                batch=payload_batch,
                device=payload_device,
            )

    def decode(
        self,
        payload: CompositeActionPayload,
        *,
        view: str = "retained",
        output_dtype: torch.dtype | None = None,
        allow_stale_endpoints: bool = False,
    ) -> DecodedCompositeState:
        """Decode either the retained-row or structured chart state view.

        ``retained`` uses declared raw rows. ``chart`` always uses
        the temporal chart and therefore exposes an explicitly lossy role when
        that role opted into a compact representation. A changed state requires
        its endpoint head to be rerun; returning source-bound endpoint payloads
        in that case requires explicit audit-only acknowledgement.
        """

        self._validate_payload(payload)
        if view not in {"retained", "chart"}:
            raise ValueError("view must be 'retained' or 'chart'")
        if type(allow_stale_endpoints) is not bool:
            raise TypeError("allow_stale_endpoints must be a boolean")
        decoded_roles: list[tuple[ContinuousRoleSpec, Tensor]] = []
        role_lossless: dict[str, bool] = {}
        role_source_equal: dict[str, bool] = {}
        batch: int | None = None
        device: torch.device | None = None
        dtype = output_dtype
        for role, role_payload in zip(self.spec.continuous_roles, payload.roles):
            chart = self._chart(role)
            if view == "retained" and role.retain_raw:
                assert role_payload.raw is not None
                value = role_payload.raw
                role_lossless[role.role_id] = True
                if dtype is not None:
                    value = value.to(dtype=dtype)
            else:
                value = chart.decode(role_payload.chart_state, output_dtype=dtype)
                role_lossless[role.role_id] = bool(chart.is_lossless)
            raw_source = role_payload.raw
            role_source_equal[role.role_id] = bool(
                raw_source is not None
                and value.dtype == raw_source.dtype
                and torch.equal(value, raw_source)
            )
            if value.ndim != 3 or tuple(value.shape[1:]) != (
                self.spec.horizon,
                role.width,
            ):
                raise ValueError(f"decoded role {role.role_id!r} has an invalid shape")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"decoded role {role.role_id!r} must be finite")
            if batch is None:
                batch = int(value.shape[0])
                device = value.device
                dtype = value.dtype if dtype is None else dtype
            elif int(value.shape[0]) != batch or value.device != device:
                raise ValueError("decoded roles must share batch size and device")
            decoded_roles.append((role, value))
        assert batch is not None and device is not None and dtype is not None
        continuous = torch.zeros(
            batch,
            self.spec.horizon,
            self.spec.state_dim,
            device=device,
            dtype=dtype,
        )
        for role, value in decoded_roles:
            index = torch.tensor(role.state_indices, device=device, dtype=torch.long)
            continuous = continuous.index_copy(-1, index, value.to(dtype=dtype))
        self._validate_continuous(continuous)

        endpoint_values: dict[str, Tensor] = {}
        for endpoint, endpoint_payload in zip(self.spec.endpoint_specs, payload.endpoints):
            value = endpoint_payload.value
            self._validate_endpoint(
                endpoint,
                value,
                batch=batch,
                device=device,
            )
            endpoint_values[endpoint.role_id] = value
        requires_endpoint_refresh = bool(endpoint_values) and not (
            payload.source_state_values_preserved
            and all(role_source_equal.values())
        )
        if requires_endpoint_refresh and not allow_stale_endpoints:
            raise ValueError(
                "the selected state is not proven identical to the endpoint source; "
                "rerun endpoint producers on the selected clean state or set "
                "allow_stale_endpoints=True for an audit-only comparison"
            )
        return DecodedCompositeState(
            continuous_state=continuous,
            endpoints=endpoint_values,
            view=view,
            role_lossless=role_lossless,
            role_source_equal=role_source_equal,
            endpoint_binding="encoded_source_state",
            requires_endpoint_refresh=requires_endpoint_refresh,
        )

    def integration_metadata(self) -> dict[str, Any]:
        """Return the minimal serialized identity needed by an experiment owner."""

        return {
            "representation": self.spec.REPRESENTATION_NAME,
            "schema_version": self.spec.schema_version,
            "spec_fingerprint": self.spec.fingerprint,
            "spec": self.spec.to_dict(),
            "charts": {
                role.role_id: dict(self._chart(role).metadata())
                for role in self.spec.continuous_roles
            },
            "solver_role_awareness": "none",
            "endpoint_update_semantics": "typed_out_of_band_at_clean_endpoint",
            "selected_view_is_replay_identity": False,
            "nonidentical_view_requires_endpoint_refresh": True,
            "ode_loop_safe": False,
        }


__all__ = [
    "CompositeActionPayload",
    "CompositeActionRepresentation",
    "DecodedCompositeState",
    "EndpointPayload",
]


