"""Typed delta routing at named boundaries of the active mainline.

The modules in this file route *block deltas*, never cumulative hidden states.
The current carrier stays outside the softmax at every call site.  This keeps
the bridge close to identity, preserves role ownership, and lets ordinary task
losses train the query, key, and value-producing upstream blocks without
straight-through estimators or auxiliary routing losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def register_gradient_rms_metric(
    value: Tensor,
    metrics: dict[str, Tensor],
    name: str,
) -> None:
    """Observe one tensor's ordinary backward RMS without changing it."""

    slot = value.new_zeros((), dtype=torch.float32)
    metrics[name] = slot
    if not torch.is_grad_enabled() or not value.requires_grad:
        return

    def capture(gradient: Tensor) -> Tensor:
        with torch.no_grad():
            slot.copy_(gradient.detach().float().square().mean().sqrt())
        return gradient

    value.register_hook(capture)


def register_gradient_axis_rms_metrics(
    value: Tensor,
    metrics: dict[str, Tensor],
    names: tuple[str, ...],
    *,
    dim: int,
) -> None:
    """Observe one backward RMS per owner on an existing typed axis."""

    axis = int(dim)
    if axis < 0:
        axis += value.ndim
    if not 0 <= axis < value.ndim:
        raise ValueError("gradient diagnostic axis is out of range")
    if int(value.shape[axis]) != len(names):
        raise ValueError("gradient diagnostic names do not match the owner axis")
    slots = tuple(value.new_zeros((), dtype=torch.float32) for _ in names)
    metrics.update(dict(zip(names, slots, strict=True)))
    if not torch.is_grad_enabled() or not value.requires_grad:
        return

    def capture(gradient: Tensor) -> Tensor:
        with torch.no_grad():
            for index, slot in enumerate(slots):
                owned = gradient.select(axis, index)
                slot.copy_(owned.detach().float().square().mean().sqrt())
        return gradient

    value.register_hook(capture)


def smooth_rms_contract(
    value: Tensor,
    max_rms: float,
) -> tuple[Tensor, Tensor]:
    """Smoothly bound per-token RMS without clipping values or gradients.

    Hidden streams in this repository are pre-normalized before producing
    updates, so a fixed normalized-chart RMS is a meaningful interface
    contract.  The fourth-order soft saturation is almost identity for normal
    small residuals, approaches ``max_rms`` for arbitrarily large inputs, and
    has a finite derivative at exactly zero.  It cannot be evaded by inflating
    the carrier because the carrier amplitude is not part of the limit.
    """

    maximum = float(max_rms)
    if maximum <= 0.0:
        raise ValueError("smooth RMS contract requires a positive maximum")
    mean_square = value.float().square().mean(dim=-1, keepdim=True)
    normalized_square = mean_square / (maximum * maximum)
    scale = (1.0 + normalized_square.square()).pow(-0.25)
    return value * scale.to(dtype=value.dtype), scale


def variance_floored_centered_norm(
    value: Tensor,
    floor: float,
) -> tuple[Tensor, Tensor]:
    """Zero-preserving selector normalization with a bounded Jacobian.

    Ordinary LayerNorm expands every non-constant input to unit variance.  If
    several signed residual values nearly cancel, that creates an
    inverse-standard-deviation backward gain while the forward activation can
    still look harmless.  This normalization keeps the useful centered
    direction but adds a fixed normalized-chart variance floor:

        (x - mean(x)) / sqrt(var(x) + floor**2)

    Exact zero/constant inputs remain exact zero, small inputs are never
    expanded by more than ``1 / floor``, and no affine term can synthesize a
    value that was absent from the source.
    """

    minimum = float(floor)
    if minimum <= 0.0:
        raise ValueError("variance-floored normalization requires a positive floor")
    value_f = value.float()
    centered = value_f - value_f.mean(dim=-1, keepdim=True)
    # A representable constant does not necessarily survive the first
    # subtraction as bit-exact zero: the reduction can round its mean by one
    # ULP.  Re-centering the already tiny residual is numerically cheap and
    # preserves the advertised exact-zero contract for constant inputs without
    # introducing a threshold or a detached branch.
    centered = centered - centered.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    denominator = torch.sqrt(variance + minimum * minimum)
    normalized = centered / denominator
    return normalized.to(dtype=value.dtype), denominator


def rms_floored_l2_normalize(
    value: Tensor,
    floor: float,
    *,
    dim: int,
) -> tuple[Tensor, Tensor]:
    """L2 normalization with an explicit floor expressed in RMS units.

    ``F.normalize`` protects a learnable feature only after its complete
    vector norm has fallen below a usually tiny absolute epsilon.  Its forward
    cosine is scale invariant long before that point, while its backward gain
    grows as ``1 / ||x||``.  This helper keeps the historical unit-L2
    convention for ordinary features, but translates one dimension-independent
    RMS floor into the matching L2 denominator:

        x / sqrt(sum(x**2) + width * floor**2)

    A normally scaled feature (RMS close to one) is changed only by the smooth
    floor term.  A cancelled or near-zero feature remains near zero instead of
    being expanded into a confident cosine direction.
    """

    minimum = float(floor)
    if minimum <= 0.0:
        raise ValueError("RMS-floored L2 normalization requires a positive floor")
    ndim = int(value.ndim)
    axis = int(dim)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError("RMS-floored L2 normalization dimension is out of range")
    width = int(value.shape[axis])
    if width <= 0:
        raise ValueError("RMS-floored L2 normalization requires a non-empty axis")
    value_f = value.float()
    mean_square = value_f.square().mean(dim=axis, keepdim=True)
    rms_denominator = torch.sqrt(mean_square + minimum * minimum)
    l2_denominator = rms_denominator * math.sqrt(float(width))
    normalized = value_f / l2_denominator
    return normalized.to(dtype=value.dtype), rms_denominator


def smooth_absolute_contract(value: Tensor, maximum: float) -> Tensor:
    """Smoothly bound each scalar while remaining identity near zero."""

    limit = float(maximum)
    if limit <= 0.0:
        raise ValueError("smooth absolute contract requires a positive maximum")
    value_f = value.float()
    scale = (1.0 + (value_f.abs() / limit).pow(8.0)).pow(-1.0 / 8.0)
    return (value_f * scale).to(dtype=value.dtype)


class VarianceFlooredCenteredNorm(nn.Module):
    """Module wrapper for zero-preserving normalization in value pipelines."""

    def __init__(self, floor: float) -> None:
        super().__init__()
        self.floor = float(floor)
        if self.floor <= 0.0:
            raise ValueError("variance-floored normalization requires a positive floor")

    def forward(self, value: Tensor) -> Tensor:
        normalized, _ = variance_floored_centered_norm(value, self.floor)
        return normalized

    def forward_with_denominator(self, value: Tensor) -> tuple[Tensor, Tensor]:
        return variance_floored_centered_norm(value, self.floor)


class AffineVarianceFlooredCenteredNorm(nn.Module):
    """LayerNorm-compatible affine chart with an explicit Jacobian bound."""

    def __init__(
        self,
        width: int,
        floor: float,
        *,
        affine_maximum: float = 4.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.floor = float(floor)
        self.affine_maximum = float(affine_maximum)
        if self.width < 1 or self.floor <= 0.0 or self.affine_maximum <= 0.0:
            raise ValueError("affine variance-floor normalization contract is invalid")
        # Names intentionally match nn.LayerNorm for checkpoint compatibility.
        self.weight = nn.Parameter(torch.ones(self.width))
        self.bias = nn.Parameter(torch.zeros(self.width))

    def forward_with_denominator(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized, denominator = variance_floored_centered_norm(value, self.floor)
        bounded_weight = smooth_absolute_contract(self.weight, self.affine_maximum)
        output = normalized.float() * bounded_weight.to(
            device=value.device, dtype=torch.float32
        ) + self.bias.to(device=value.device, dtype=torch.float32)
        gain = bounded_weight.detach().float().abs().amax() / self.floor
        return output.to(dtype=value.dtype), denominator, gain

    def forward(self, value: Tensor) -> Tensor:
        normalized, _, _ = self.forward_with_denominator(value)
        return normalized


@dataclass(frozen=True)
class PolicyRoleDeltaBank:
    """Policy-approved top-to-bottom delta values.

    ``values`` keeps source depth and action basis explicit:
    ``[batch, source, horizon, basis, hidden]``.  ``protected_detail`` is the
    already world-conditioned high-resolution write made at the W->P boundary.
    ``protected_policy_precision`` is the live per-step P1 policy residual.
    Both remain separate additive lanes and therefore cannot lose a
    source-survival softmax competition against coarser world/policy values.
    """

    values: Tensor
    source_names: tuple[str, ...]
    source_depths: tuple[int, ...]
    protected_detail: Tensor | None = None
    protected_policy_precision: Tensor | None = None

    def validate(self, *, hidden_size: int, horizon: int) -> None:
        if self.values.ndim != 5:
            raise ValueError("policy role-delta values must be [B,source,horizon,basis,H]")
        if int(self.values.shape[1]) != len(self.source_names):
            raise ValueError("policy role-delta names do not match the source axis")
        if int(self.values.shape[1]) != len(self.source_depths):
            raise ValueError("policy role-delta depths do not match the source axis")
        if int(self.values.shape[2]) != int(horizon):
            raise ValueError("policy role-delta horizon does not match the action horizon")
        if int(self.values.shape[3]) <= 0:
            raise ValueError("policy role-delta bank must retain at least one basis token")
        if int(self.values.shape[4]) != int(hidden_size):
            raise ValueError("policy role-delta hidden size is invalid")
        if len(self.source_names) <= 0:
            raise ValueError("policy role-delta bank cannot be empty")
        expected = (
            int(self.values.shape[0]),
            int(horizon),
            int(self.values.shape[3]),
            int(hidden_size),
        )
        for name in ("protected_detail", "protected_policy_precision"):
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != expected:
                raise ValueError(f"{name} must be [B,horizon,basis,H]")


class RoleDeltaAttnRes(nn.Module):
    """Low-rank selector over full-width, schema-aligned role deltas.

    Queries have shape ``[B,...,H]`` and values ``[B,...,source,H]``.  A
    learned identity key distinguishes legal source/depth/camera or
    source/depth/basis slots without projecting the full-width value.  The
    explicit null candidate has a zero value.  It can suppress an unnecessary
    *extra bridge*, but it can never suppress the caller's main carrier because
    that carrier is added outside this module.
    """

    def __init__(
        self,
        hidden_size: int,
        route_dim: int,
        *,
        max_sources: int,
        include_null: bool = True,
        max_value_rms: float | None = None,
        normalization_floor: float | None = None,
    ) -> None:
        super().__init__()
        if min(int(hidden_size), int(route_dim), int(max_sources)) < 1:
            raise ValueError("RoleDeltaAttnRes dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.route_dim = int(route_dim)
        self.max_sources = int(max_sources)
        self.include_null = bool(include_null)
        self.max_value_rms = None if max_value_rms is None else float(max_value_rms)
        if self.max_value_rms is not None and self.max_value_rms <= 0.0:
            raise ValueError("role-delta value RMS maximum must be positive")
        self.normalization_floor = (
            None if normalization_floor is None else float(normalization_floor)
        )
        if self.normalization_floor is not None and self.normalization_floor <= 0.0:
            raise ValueError("role-delta normalization floor must be positive")
        self.query_proj = nn.Linear(self.hidden_size, self.route_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.route_dim, bias=False)
        self.source_key = nn.Parameter(torch.randn(self.max_sources, self.route_dim) * 0.02)
        if self.include_null:
            self.null_key = nn.Parameter(torch.zeros(1, self.route_dim))
        else:
            self.register_parameter("null_key", None)
        nn.init.normal_(self.query_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        query: Tensor,
        delta_values: Tensor,
        *,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if query.ndim < 3 or int(query.shape[-1]) != self.hidden_size:
            raise ValueError("role-delta query must be [B,...,H]")
        if delta_values.ndim != query.ndim + 1:
            raise ValueError("role-delta values must add one source axis before H")
        if tuple(delta_values.shape[:-2]) != tuple(query.shape[:-1]):
            raise ValueError("role-delta query and value schemas are not aligned")
        if int(delta_values.shape[-1]) != self.hidden_size:
            raise ValueError("role-delta values have the wrong hidden size")
        source_count = int(delta_values.shape[-2])
        if source_count <= 0 or source_count > self.max_sources:
            raise ValueError(f"role-delta source count must be in [1,{self.max_sources}]")

        raw_delta_values = delta_values
        if self.max_value_rms is None:
            routed_values = raw_delta_values
            value_scale = raw_delta_values.new_ones(
                (*raw_delta_values.shape[:-1], 1), dtype=torch.float32
            )
        else:
            routed_values, value_scale = smooth_rms_contract(
                raw_delta_values,
                self.max_value_rms,
            )
        if self.normalization_floor is None:
            query_unit = F.layer_norm(query.float(), (self.hidden_size,)).to(dtype=query.dtype)
            value_unit = F.layer_norm(routed_values.float(), (self.hidden_size,)).to(
                dtype=routed_values.dtype
            )
            query_denominator = query.new_ones((*query.shape[:-1], 1), dtype=torch.float32)
            value_denominator = routed_values.new_ones(
                (*routed_values.shape[:-1], 1), dtype=torch.float32
            )
        else:
            query_unit, query_denominator = variance_floored_centered_norm(
                query,
                self.normalization_floor,
            )
            value_unit, value_denominator = variance_floored_centered_norm(
                routed_values,
                self.normalization_floor,
            )
        route_query = self.query_proj(query_unit)
        route_keys = self.key_proj(value_unit)
        identity = self.source_key[:source_count].to(
            device=route_keys.device, dtype=route_keys.dtype
        )
        identity_shape = (1,) * (route_keys.ndim - 2) + tuple(identity.shape)
        route_keys = route_keys + identity.reshape(identity_shape)
        logits = torch.einsum(
            "...r,...sr->...s",
            route_query.float(),
            route_keys.float(),
        ) / math.sqrt(float(self.route_dim))
        if self.include_null:
            if self.null_key is None:
                raise RuntimeError("null-enabled role route has no null key")
            null_key = self.null_key.to(device=route_query.device, dtype=route_query.dtype)
            null_logits = torch.einsum(
                "...r,kr->...k", route_query.float(), null_key.float()
            ) / math.sqrt(float(self.route_dim))
            probabilities = torch.softmax(torch.cat((logits, null_logits), dim=-1), dim=-1)
            source_probability = probabilities[..., :source_count].to(dtype=routed_values.dtype)
        else:
            probabilities = torch.softmax(logits, dim=-1)
            source_probability = probabilities.to(dtype=routed_values.dtype)
        routed = torch.einsum("...s,...sh->...h", source_probability, routed_values)
        if not collect_diagnostics:
            return routed, {}

        probability_f = probabilities.detach().float()
        entropy_raw = -(probability_f * probability_f.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_denominator = math.log(
            float(source_count + 1 if self.include_null else max(source_count, 2))
        )
        entropy = (
            entropy_raw / entropy_denominator
            if entropy_denominator > 0.0
            else torch.zeros_like(entropy_raw)
        )
        reduce_dims = tuple(range(probability_f.ndim - 1))
        source_probability_f = probability_f[..., :source_count]
        source_mass = source_probability_f.mean(dim=reduce_dims)
        # Keep "how many legal sources are used" separate from "how many
        # candidates including null are active".  Otherwise a route split
        # between one real source and null would look like two useful source
        # roles, even though all non-null mass has collapsed onto one depth.
        conditional_source_probability = source_probability_f / (
            source_probability_f.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        source_entropy = -(
            conditional_source_probability * conditional_source_probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        query_rms = query.detach().float().square().mean().sqrt()
        raw_delta_rms = raw_delta_values.detach().float().square().mean().sqrt()
        delta_rms = routed_values.detach().float().square().mean().sqrt()
        routed_rms = routed.detach().float().square().mean().sqrt()
        metrics = {
            "entropy": entropy.mean(),
            "max": probability_f.amax(dim=-1).mean(),
            "null_mass": (
                probability_f[..., -1].mean() if self.include_null else probability_f.new_zeros(())
            ),
            "source_mass": source_mass,
            "source_mass_max": source_mass.max(),
            "source_effective_count": source_entropy.exp().mean(),
            "candidate_effective_count": entropy_raw.exp().mean(),
            "sample_route_std": probability_f.std(dim=0, unbiased=False).mean(),
            "query_rms": query_rms,
            "value_rms": delta_rms,
            "raw_value_rms": raw_delta_rms,
            "value_compression": (1.0 - value_scale.detach().float()).mean(),
            "value_contract_enabled": routed.new_tensor(
                float(self.max_value_rms is not None), dtype=torch.float32
            ),
            "variance_safe_norm": routed.new_tensor(
                float(self.normalization_floor is not None), dtype=torch.float32
            ),
            "query_norm_denominator_min": query_denominator.detach().float().amin(),
            "value_norm_denominator_min": value_denominator.detach().float().amin(),
            "update_rms": routed_rms,
            "carrier_ratio": routed_rms / query_rms.clamp_min(1e-8),
        }
        for source_index in range(source_count):
            metrics[f"source_{source_index}_mass"] = source_mass[source_index]
        # Preserve factual route differentiation instead of reporting only a
        # global source mean.  Call sites know the semantics of these axes:
        # G->W uses anchor/camera, W->P uses horizon/basis, and P->MMDiT uses
        # horizon.  A zero value means that axis receives the same routing
        # distribution everywhere, even when the global average looks healthy.
        for axis in range(1, probability_f.ndim - 1):
            metrics[f"query_axis_{axis}_route_std"] = probability_f.std(
                dim=axis,
                unbiased=False,
            ).mean()
        return routed, metrics
