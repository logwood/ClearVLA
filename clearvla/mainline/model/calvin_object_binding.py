"""CALVIN-only language-to-object binding at the S/coarse-action seam.

The shared G/W/P graph deliberately keeps object slots and modality routes
separate.  CALVIN nevertheless needs a task-role pointer (for example, the
blue block rather than the red block).  This module provides that pointer as
an outlet-scoped adapter.  It never reads privileged ``scene_obs`` and it is
not constructed for Pen/RDT/LIBERO.

The adapter starts with a small bounded context route.  This makes the frozen
action-boundary intervention and owner VJP observable before short training;
it is still an explicitly new CALVIN component identity rather than an exact
resume alias for an older checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .types import ObjectFactSet

CALVIN_OBJECT_BINDING_COMPONENT = "calvin_primary_object_binding_v1"
# ComponentSelection.intent serializes the outlet-scoped implementation ID.
# Keep one canonical value so policy construction cannot compare against a
# different alias and silently omit the bridge.
CALVIN_OBJECT_BINDING_INTENT = CALVIN_OBJECT_BINDING_COMPONENT


def _normalized_entropy(probability: Tensor) -> Tensor:
    """Return entropy normalized by the number of pointer choices."""

    if probability.ndim != 2 or int(probability.shape[-1]) < 2:
        raise ValueError("pointer probabilities must be [B,K+1] with K>=1")
    choices = int(probability.shape[-1])
    entropy = -(probability.float() * probability.float().clamp_min(1e-8).log()).sum(
        dim=-1
    )
    return entropy / math.log(float(choices))


@dataclass(frozen=True)
class CalvinObjectBindingResult:
    """The pointer and selected context consumed by the coarse proposal."""

    pointer: Tensor  # [B,K+1], last column is null
    selected_context: Tensor  # [B,1,H], zero for an exact null decision
    selected_geometry: Tensor  # [B,2]
    confidence: Tensor  # [B,1]
    entropy: Tensor  # [B,1]
    valid_mass: Tensor  # [B,1]

    def validate(self, *, batch: int, objects: int, hidden: int) -> None:
        if tuple(self.pointer.shape) != (batch, objects + 1):
            raise ValueError("CALVIN binding pointer must be [B,K+1]")
        if tuple(self.selected_context.shape) != (batch, 1, hidden):
            raise ValueError("CALVIN selected object context must be [B,1,H]")
        if tuple(self.selected_geometry.shape) != (batch, 2):
            raise ValueError("CALVIN selected geometry must be [B,2]")
        for name in ("confidence", "entropy", "valid_mass"):
            if tuple(getattr(self, name).shape) != (batch, 1):
                raise ValueError(f"CALVIN binding {name} must be [B,1]")
        if not bool(torch.isfinite(self.pointer).all()):
            raise ValueError("CALVIN binding pointer is non-finite")
        if not bool(torch.isfinite(self.selected_context).all()):
            raise ValueError("CALVIN selected object context is non-finite")
        sums = self.pointer.float().sum(dim=-1)
        # The pointer is consumed in BF16 on the active path; softmax roundoff
        # can exceed the FP32 contract tolerance while remaining numerically
        # normalized.  Keep this a finite-boundary check, not an exact-sum
        # requirement.
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=2e-2, rtol=2e-2)):
            raise ValueError("CALVIN binding pointer must sum to one")


class CalvinObjectBindingBridge(nn.Module):
    """Validity-aware, permutation-equivariant K-way object pointer.

    The bridge owns task-role selection only.  It consumes the already
    encoded protected language/history and the observable ``ObjectFactSet``;
    it does not introduce a second visual encoder or a privileged simulator
    input.  Every object row uses the same scorer and no slot-index embedding
    is present, so permuting K rows permutes the pointer and nothing else.
    """

    def __init__(self, *, hidden: int, route_dim: int, objects: int = 4) -> None:
        super().__init__()
        if int(objects) != 4:
            raise ValueError("CALVIN object binding currently requires K=4")
        if int(hidden) <= 0 or int(route_dim) <= 0:
            raise ValueError("CALVIN object binding dimensions must be positive")
        self.hidden = int(hidden)
        self.route_dim = int(route_dim)
        self.objects = int(objects)
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
        # Keep a small, bounded route at initialization so an action-level
        # language/position intervention is observable in the frozen probe.
        # The short adapter run may still learn this scalar jointly with its
        # coarse adapter; it is never used as an execution gate.
        self.context_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def _object_features(self, object_tokens: Tensor, facts: ObjectFactSet) -> Tensor:
        routes = torch.cat((facts.semantic, facts.appearance, facts.geometry), dim=-1)
        route_context = self.route_projection(routes.to(dtype=object_tokens.dtype))
        coordinate_context = self.coordinate_projection(
            facts.coordinates.to(device=object_tokens.device, dtype=object_tokens.dtype)
        )
        # Keep the route sidecar bounded relative to the learned visual token;
        # it is a selector feature, not a replacement for G's content value.
        return self.object_norm(
            object_tokens + 0.25 * route_context + 0.10 * coordinate_context
        )

    @staticmethod
    def _masked_pointer(logits: Tensor, null_logits: Tensor, valid: Tensor) -> Tensor:
        if logits.ndim != 2 or valid.ndim != 2:
            raise ValueError("binding logits/validity must be [B,K]")
        if tuple(logits.shape) != tuple(valid.shape):
            raise ValueError("binding logits and validity do not align")
        batch, objects = logits.shape
        if tuple(null_logits.shape) != (batch,):
            raise ValueError("binding null logits must be [B]")
        valid_bool = valid.to(device=logits.device, dtype=torch.bool)
        masked = logits.masked_fill(~valid_bool, torch.finfo(logits.dtype).min)
        masked = torch.cat((masked, null_logits[:, None]), dim=-1)
        pointer = torch.softmax(masked.float(), dim=-1).to(dtype=logits.dtype)
        has_valid = valid_bool.any(dim=-1, keepdim=True)
        null_only = torch.zeros_like(pointer)
        null_only[:, objects] = 1.0
        return torch.where(has_valid, pointer, null_only)

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
        features = self._object_features(object_tokens, facts)
        pair = torch.cat((features, query[:, None].expand(-1, objects, -1)), dim=-1)
        logits = self.scorer(pair).squeeze(-1)
        # The null score is query-conditioned but has no object-slot axis.
        null_logit = self.null_scorer(query).squeeze(-1)
        valid = facts.validity[..., 0].detach() > 0.0
        pointer = self._masked_pointer(logits, null_logit, valid)

        real_pointer = pointer[:, :objects]
        valid_mass = real_pointer.sum(dim=-1, keepdim=True)
        selected = (real_pointer[..., None] * features).sum(dim=1, keepdim=True)
        selected = torch.where(
            valid_mass[..., None] > 1e-6,
            selected,
            torch.zeros_like(selected),
        )
        gate = torch.tanh(self.context_gate).to(device=object_tokens.device, dtype=object_tokens.dtype)
        selected_context = gate * selected

        geometry = facts.coordinates.float()
        selected_geometry = (
            real_pointer.float()[..., None] * geometry
        ).sum(dim=1)
        selected_geometry = torch.where(
            valid_mass.float() > 1e-6,
            selected_geometry,
            torch.zeros_like(selected_geometry),
        )
        result = CalvinObjectBindingResult(
            pointer=pointer,
            selected_context=selected_context,
            selected_geometry=selected_geometry,
            confidence=pointer.max(dim=-1, keepdim=True).values,
            entropy=_normalized_entropy(pointer).unsqueeze(-1),
            valid_mass=valid_mass,
        )
        result.validate(batch=batch, objects=objects, hidden=hidden)
        metrics = {
            "calvin_binding_pointer_entropy": result.entropy.detach().float().mean(),
            "calvin_binding_pointer_confidence": result.confidence.detach().float().mean(),
            "calvin_binding_valid_mass": result.valid_mass.detach().float().mean(),
            "calvin_binding_context_gate": gate.detach().float(),
            "calvin_binding_null_mass": pointer[:, objects].detach().float().mean(),
            "calvin_binding_selected_geometry_rms": selected_geometry.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        return result, metrics

    @staticmethod
    def supervised_loss(
        pointer: Tensor,
        target: Tensor,
        coverage: Tensor | None = None,
    ) -> Tensor:
        """Soft-label KL/CE for a training-only sidecar target."""

        if pointer.ndim != 2 or target.ndim != 2 or tuple(pointer.shape) != tuple(target.shape):
            raise ValueError("CALVIN binding pointer/target must share [B,K+1]")
        target = target.to(device=pointer.device, dtype=torch.float32)
        if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
            raise ValueError("CALVIN binding targets must be finite and non-negative")
        target_sum = target.sum(dim=-1, keepdim=True)
        if not bool(torch.allclose(target_sum, torch.ones_like(target_sum), atol=2e-4, rtol=2e-4)):
            raise ValueError("CALVIN binding targets must sum to one")
        row_loss = -(target * pointer.float().clamp_min(1e-8).log()).sum(dim=-1)
        if coverage is None:
            return row_loss.mean()
        if coverage.ndim == 2 and int(coverage.shape[-1]) == 1:
            coverage = coverage[:, 0]
        if coverage.ndim != 1 or int(coverage.shape[0]) != int(pointer.shape[0]):
            raise ValueError("CALVIN binding coverage must be [B] or [B,1]")
        weight = coverage.to(device=pointer.device, dtype=torch.float32).clamp(0.0, 1.0)
        return (row_loss * weight).sum() / weight.sum().clamp_min(1.0)

__all__ = [
    "CALVIN_OBJECT_BINDING_COMPONENT",
    "CALVIN_OBJECT_BINDING_INTENT",
    "CalvinObjectBindingBridge",
    "CalvinObjectBindingResult",
]
