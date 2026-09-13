"""Permutation-safe CALVIN language-to-object binding.

The shared mainline deliberately keeps the global object axis anonymous: G
may relabel its K rows whenever the scene changes.  A supervision target that
calls row 0 ``red`` therefore teaches a slot-index shortcut, not grounding.
This adapter uses two separate, typed views at the S/coarse seam:

* a language-conditioned pointer over the valid object rows plus an explicit
  null row; and
* a *language-free* visual role scorer over the same rows.

The training target is a role distribution (red/blue/pink/null),
not a K-slot pointer.  The loss combines the two views by a K-invariant
object-role joint distribution.  Permuting object rows consequently permutes
only the pointer's K columns and leaves the role distribution and selected
context unchanged.  No privileged ``scene_obs`` enters this module.

This is an outlet-scoped CALVIN component.  It is intentionally a new ABI
(``v2``): checkpoints or sidecars produced for the rejected fixed-slot v1
design must not be resumed or silently reinterpreted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..calvin_binding_contract import (
    CALVIN_OBJECT_BINDING_COMPONENT,
    CALVIN_OBJECT_BINDING_INTENT,
    CALVIN_OBJECT_BINDING_ROLE_COUNT,
    CALVIN_OBJECT_BINDING_ROLE_NAMES,
    CALVIN_OBJECT_BINDING_TARGET_COUNT,
)
from .types import ObjectFactSet


def _normalized_entropy(probability: Tensor) -> Tensor:
    """Return entropy normalized by the number of choices."""

    if probability.ndim != 2 or int(probability.shape[-1]) < 2:
        raise ValueError("probabilities must be [B,N] with at least two choices")
    choices = int(probability.shape[-1])
    value = probability.float().clamp_min(1e-8)
    entropy = -(probability.float() * value.log()).sum(dim=-1)
    return entropy / math.log(float(choices))


@dataclass(frozen=True)
class CalvinObjectBindingResult:
    """Runtime output of the CALVIN role bridge.

    ``pointer`` is retained for diagnostics and for the action-compatible
    selected context.  ``role_distribution`` is the only supervised semantic
    output; its last class is null.  ``selected_geometry`` is a diagnostic
    summary and deliberately does not cross the ActionIntentDock boundary.
    """

    pointer: Tensor  # [B,K+1], final column is null
    role_distribution: Tensor  # [B,R+1], final column is null
    visual_role_logits: Tensor  # [B,K,R], language-free diagnostic view
    selected_context: Tensor  # [B,1,H], zero for an exact null decision
    selected_geometry: Tensor  # [B,2], diagnostic only
    confidence: Tensor  # [B,1]
    entropy: Tensor  # [B,1]
    valid_mass: Tensor  # [B,1]

    def validate(
        self,
        *,
        batch: int,
        objects: int,
        hidden: int,
        role_count: int = CALVIN_OBJECT_BINDING_ROLE_COUNT,
    ) -> None:
        if tuple(self.pointer.shape) != (batch, objects + 1):
            raise ValueError("CALVIN binding pointer must be [B,K+1]")
        if tuple(self.role_distribution.shape) != (batch, role_count + 1):
            raise ValueError("CALVIN binding role distribution must be [B,R+1]")
        if tuple(self.visual_role_logits.shape) != (batch, objects, role_count):
            raise ValueError("CALVIN visual role logits must be [B,K,R]")
        if tuple(self.selected_context.shape) != (batch, 1, hidden):
            raise ValueError("CALVIN selected object context must be [B,1,H]")
        if tuple(self.selected_geometry.shape) != (batch, 2):
            raise ValueError("CALVIN selected geometry must be [B,2]")
        for name in ("confidence", "entropy", "valid_mass"):
            if tuple(getattr(self, name).shape) != (batch, 1):
                raise ValueError(f"CALVIN binding {name} must be [B,1]")
        for name in (
            "pointer",
            "role_distribution",
            "visual_role_logits",
            "selected_context",
            "selected_geometry",
            "confidence",
            "entropy",
            "valid_mass",
        ):
            value = getattr(self, name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"CALVIN binding {name} is non-finite")
        for name in ("pointer", "role_distribution"):
            value = getattr(self, name).float()
            if bool((value < -1e-6).any()):
                raise ValueError(f"CALVIN binding {name} must be non-negative")
            sums = value.sum(dim=-1)
            if not bool(torch.allclose(sums, torch.ones_like(sums), atol=2e-2, rtol=2e-2)):
                raise ValueError(f"CALVIN binding {name} must sum to one")


class CalvinObjectBindingBridge(nn.Module):
    """Validity-aware, permutation-equivariant language/role bridge.

    The pointer scorer reads protected language/history and visual object
    features.  The role scorer reads only shared, language-free fact routes;
    it has no slot-index embedding or role-specific K head.  A role target is
    trained through the normalized joint distribution over valid object-role
    pairs and null.  This closes the forward path without assigning semantic
    meaning to a particular K row.
    """

    def __init__(
        self,
        *,
        hidden: int,
        route_dim: int,
        objects: int = 4,
        role_count: int = CALVIN_OBJECT_BINDING_ROLE_COUNT,
    ) -> None:
        super().__init__()
        if int(objects) != 4:
            raise ValueError("CALVIN object binding currently requires K=4")
        if int(role_count) != CALVIN_OBJECT_BINDING_ROLE_COUNT:
            raise ValueError(
                "CALVIN role ABI requires three named roles plus the explicit null class"
            )
        if int(hidden) <= 0 or int(route_dim) <= 0:
            raise ValueError("CALVIN object binding dimensions must be positive")
        self.hidden = int(hidden)
        self.route_dim = int(route_dim)
        self.objects = int(objects)
        self.role_count = int(role_count)
        self.goal_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.object_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.route_projection = nn.Linear(3 * route_dim, hidden, bias=False)
        self.coordinate_projection = nn.Linear(2, hidden, bias=False)
        self.scorer = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.null_scorer = nn.Linear(hidden, 1, bias=False)

        # The role view is intentionally language-free and slot-shared.  It
        # uses observable semantic/appearance/geometry routes and coordinates,
        # rather than a fixed row label or a learned per-slot seed.
        self.role_input = nn.Linear(3 * route_dim + 2, hidden, bias=False)
        self.role_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.role_head = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, self.role_count, bias=False),
        )

        # Keep the selected action-context route bounded at initialization;
        # this scalar is part of the bridge owner and is audited by VJP tests.
        self.context_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def _object_features(self, object_tokens: Tensor, facts: ObjectFactSet) -> Tensor:
        routes = torch.cat((facts.semantic, facts.appearance, facts.geometry), dim=-1)
        route_context = self.route_projection(routes.to(dtype=object_tokens.dtype))
        coordinate_context = self.coordinate_projection(
            facts.coordinates.to(device=object_tokens.device, dtype=object_tokens.dtype)
        )
        return self.object_norm(
            object_tokens + 0.25 * route_context + 0.10 * coordinate_context
        )

    def _visual_role_logits(self, facts: ObjectFactSet, *, dtype: torch.dtype) -> Tensor:
        routes = torch.cat((facts.semantic, facts.appearance, facts.geometry), dim=-1)
        value = torch.cat(
            (routes.float(), facts.coordinates.float()), dim=-1
        ).to(dtype=dtype)
        value = self.role_input(value)
        # Subtract a *valid-object* set mean.  Invalid/padded rows must not
        # perturb a valid role logit, while a common scene offset should still
        # cancel.  The validity mask is evidence metadata, not a trainable
        # route, so detach it at this boundary.
        valid = facts.validity[..., 0].detach().float().clamp(0.0, 1.0)
        valid_bool = valid > 0.0
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        # ``where`` (rather than ``0 * value``) quarantines a malformed NaN
        # in a padded row before the set reduction.  The valid rows retain
        # their exact values and the output for invalid rows is zeroed below.
        safe_value = torch.where(valid_bool[..., None], value, torch.zeros_like(value))
        mean = safe_value.float().sum(dim=1, keepdim=True) / denominator[..., None]
        value = self.role_norm(value - mean.to(dtype=value.dtype))
        logits = self.role_head(value)
        return torch.where(valid_bool[..., None], logits, torch.zeros_like(logits))

    @staticmethod
    def _masked_pointer(logits: Tensor, null_logits: Tensor, valid: Tensor) -> Tensor:
        """Normalize K logits plus null with an exact all-invalid fallback."""

        if logits.ndim != 2 or valid.ndim != 2:
            raise ValueError("binding logits/validity must be [B,K]")
        if tuple(logits.shape) != tuple(valid.shape):
            raise ValueError("binding logits and validity do not align")
        batch, objects = logits.shape
        if tuple(null_logits.shape) != (batch,):
            raise ValueError("binding null logits must be [B]")
        valid_bool = valid.to(device=logits.device, dtype=torch.bool)
        masked = logits.float().masked_fill(~valid_bool, -torch.inf)
        masked = torch.cat((masked, null_logits.float()[:, None]), dim=-1)
        has_valid = valid_bool.any(dim=-1, keepdim=True)
        safe = torch.where(has_valid, masked, torch.zeros_like(masked))
        pointer = torch.softmax(safe, dim=-1).to(dtype=logits.dtype)
        null_only = torch.zeros_like(pointer)
        null_only[:, objects] = 1.0
        return torch.where(has_valid, pointer, null_only)

    def _joint_role_distribution(
        self,
        *,
        logits: Tensor,
        null_logits: Tensor,
        role_logits: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Return a normalized [B,R+1] distribution over role or null.

        The softmax is over valid K-by-R pairs and one null event.  A K-row
        permutation only reorders terms in that sum, which is the key
        structural invariant missing from the rejected fixed-slot sidecar.
        """

        batch, objects = logits.shape
        valid_bool = valid.to(device=logits.device, dtype=torch.bool)
        pair = logits.float()[:, :, None] + role_logits.float()
        pair_mask = valid_bool[:, :, None].expand(-1, -1, self.role_count)
        flat = pair.reshape(batch, objects * self.role_count).masked_fill(
            ~pair_mask.reshape(batch, objects * self.role_count), -torch.inf
        )
        combined = torch.cat((flat, null_logits.float()[:, None]), dim=-1)
        has_valid = valid_bool.any(dim=-1, keepdim=True)
        safe = torch.where(has_valid, combined, torch.zeros_like(combined))
        joint = torch.softmax(safe, dim=-1)
        null_only = torch.zeros_like(joint)
        null_only[:, -1] = 1.0
        joint = torch.where(has_valid, joint, null_only)
        role_mass = joint[:, :-1].reshape(batch, objects, self.role_count).sum(dim=1)
        return torch.cat((role_mass, joint[:, -1:]), dim=-1)

    def forward(
        self,
        *,
        protected_goal: Tensor,
        history_tokens: Tensor,
        object_tokens: Tensor,
        facts: ObjectFactSet,
    ) -> tuple[CalvinObjectBindingResult, dict[str, Tensor]]:
        facts.validate()
        if protected_goal.ndim != 3 or history_tokens.ndim != 3:
            raise ValueError("CALVIN binding goal/history must be rank-3 token arrays")
        if object_tokens.ndim != 3:
            raise ValueError("CALVIN binding object tokens must be [B,K,H]")
        batch, objects, hidden = object_tokens.shape
        if objects != self.objects or hidden != self.hidden:
            raise ValueError("CALVIN binding object token shape does not match the bridge")
        if int(protected_goal.shape[0]) != batch or int(history_tokens.shape[0]) != batch:
            raise ValueError("CALVIN binding batch axes do not align")
        if int(facts.content.shape[0]) != batch or int(facts.content.shape[1]) != objects:
            raise ValueError("CALVIN binding facts do not align with object tokens")

        goal = protected_goal.float().mean(dim=1)
        if int(history_tokens.shape[1]) > 0:
            goal = goal + 0.25 * history_tokens[:, -1].float()
        query = self.goal_norm(goal.to(dtype=object_tokens.dtype))
        valid = facts.validity[..., 0].detach() > 0.0
        features = self._object_features(object_tokens, facts)
        # Keep invalid/padded rows out of every downstream reduction, even if
        # an upstream cache contains a non-finite placeholder in that row.
        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        pair = torch.cat((features, query[:, None].expand(-1, objects, -1)), dim=-1)
        logits = self.scorer(pair).squeeze(-1)
        null_logit = self.null_scorer(query).squeeze(-1)
        pointer = self._masked_pointer(logits, null_logit, valid)

        role_logits = self._visual_role_logits(facts, dtype=object_tokens.dtype)
        role_distribution = self._joint_role_distribution(
            logits=logits,
            null_logits=null_logit,
            role_logits=role_logits,
            valid=valid,
        )

        real_pointer = pointer[:, :objects]
        valid_mass = real_pointer.sum(dim=-1, keepdim=True)
        selected = (real_pointer[..., None] * features).sum(dim=1, keepdim=True)
        selected = torch.where(
            valid_mass[..., None] > 1e-6,
            selected,
            torch.zeros_like(selected),
        )
        gate = torch.tanh(self.context_gate).to(
            device=object_tokens.device, dtype=object_tokens.dtype
        )
        selected_context = gate * selected

        geometry = torch.where(
            valid[..., None], facts.coordinates.float(), torch.zeros_like(facts.coordinates.float())
        )
        selected_geometry = (real_pointer.float()[..., None] * geometry).sum(dim=1)
        selected_geometry = torch.where(
            valid_mass.float() > 1e-6,
            selected_geometry,
            torch.zeros_like(selected_geometry),
        )
        result = CalvinObjectBindingResult(
            pointer=pointer,
            role_distribution=role_distribution,
            visual_role_logits=role_logits,
            selected_context=selected_context,
            selected_geometry=selected_geometry,
            confidence=pointer.max(dim=-1, keepdim=True).values,
            entropy=_normalized_entropy(pointer).unsqueeze(-1),
            valid_mass=valid_mass,
        )
        result.validate(
            batch=batch,
            objects=objects,
            hidden=hidden,
            role_count=self.role_count,
        )
        metrics = {
            "calvin_binding_pointer_entropy": result.entropy.detach().float().mean(),
            "calvin_binding_pointer_confidence": result.confidence.detach().float().mean(),
            "calvin_binding_valid_mass": result.valid_mass.detach().float().mean(),
            "calvin_binding_context_gate": gate.detach().float(),
            "calvin_binding_null_mass": pointer[:, objects].detach().float().mean(),
            "calvin_binding_role_entropy": _normalized_entropy(
                result.role_distribution
            ).detach().float().mean(),
            "calvin_binding_role_null_mass": result.role_distribution[:, -1]
            .detach()
            .float()
            .mean(),
            "calvin_binding_selected_geometry_rms": selected_geometry.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        return result, metrics

    @staticmethod
    def supervised_loss(
        role_distribution: Tensor,
        target: Tensor,
        coverage: Tensor | None = None,
    ) -> Tensor:
        """Soft-label CE on semantic roles, invariant to K-row permutation."""

        if (
            role_distribution.ndim != 2
            or target.ndim != 2
            or tuple(role_distribution.shape) != tuple(target.shape)
        ):
            raise ValueError("CALVIN role distribution/target must share [B,R+1]")
        target = target.to(device=role_distribution.device, dtype=torch.float32)
        if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
            raise ValueError("CALVIN role targets must be finite and non-negative")
        target_sum = target.sum(dim=-1, keepdim=True)
        if not bool(torch.allclose(target_sum, torch.ones_like(target_sum), atol=2e-4, rtol=2e-4)):
            raise ValueError("CALVIN role targets must sum to one")
        row_loss = -(
            target * role_distribution.float().clamp_min(1e-8).log()
        ).sum(dim=-1)
        if coverage is None:
            return row_loss.mean()
        if coverage.ndim == 2 and int(coverage.shape[-1]) == 1:
            coverage = coverage[:, 0]
        if coverage.ndim != 1 or int(coverage.shape[0]) != int(role_distribution.shape[0]):
            raise ValueError("CALVIN binding coverage must be [B] or [B,1]")
        weight = coverage.to(device=role_distribution.device, dtype=torch.float32).clamp(
            0.0, 1.0
        )
        return (row_loss * weight).sum() / weight.sum().clamp_min(1.0)

    # Explicit name for callers/audits that want to distinguish role loss from
    # the diagnostic K-pointer.  Keep ``supervised_loss`` as the stable hook.
    role_supervised_loss = supervised_loss


__all__ = [
    "CALVIN_OBJECT_BINDING_COMPONENT",
    "CALVIN_OBJECT_BINDING_INTENT",
    "CALVIN_OBJECT_BINDING_ROLE_NAMES",
    "CALVIN_OBJECT_BINDING_ROLE_COUNT",
    "CALVIN_OBJECT_BINDING_TARGET_COUNT",
    "CalvinObjectBindingBridge",
    "CalvinObjectBindingResult",
]
