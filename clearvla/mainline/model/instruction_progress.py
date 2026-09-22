"""Current/reference evidence comparison, not an oracle progress estimator.

A fixed current entity/query reads BOTH observation charts using the same keys
and value mapping. Their change is exactly zero for identical observations,
but the full per-camera distribution is retained until each value read. No
pixel is subtracted from a world coordinate. An uncertain visual match remains
an uncertain read, not an invented physical object ID or success label.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..instruction_reference import InstructionReference
from .routing import smooth_rms_contract
from .target_binding import TargetBinding, masked_probability
from .types import ObjectFactSet


class InstructionReferenceRead(nn.Module):
    def __init__(self, *, hidden: int, content_dim: int, state_dim: int, cameras: int) -> None:
        super().__init__()
        self.query = nn.Linear(content_dim, hidden, bias=False)
        self.task_query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(content_dim, hidden, bias=False)
        self.value = nn.Linear(content_dim, hidden, bias=False)
        self.position_value = nn.Linear(2, hidden, bias=False)
        self.state_delta = nn.Linear(state_dim, hidden, bias=False)
        self.view_role = nn.Embedding(cameras, hidden)
        self.read_output = nn.Linear(hidden, hidden, bias=False)
        self.match_status = nn.Linear(4, hidden, bias=False)
        self.scale = hidden**-0.5
        self.cameras = int(cameras)

    def _read(self, query: Tensor, content: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
        batch, cameras, patches, _ = content.shape
        side = math.isqrt(patches)
        if side * side != patches or cameras != self.cameras:
            raise ValueError("instruction comparison requires the declared square patch chart")
        safe = torch.where(observed[..., None], content, torch.zeros_like(content)).to(query.dtype)
        # Keys exclude coordinate embeddings: matching may follow appearance
        # across positions; image positions remain values; read entropy remains explicit.
        logits = torch.einsum("bkh,bcph->bkcp", query.float(), self.key(safe).float()) * self.scale
        valid = observed[:, None].expand_as(logits)
        weights = masked_probability(logits, valid)
        axis = torch.linspace(-1.0, 1.0, side, device=content.device, dtype=torch.float32)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((x, y), -1).reshape(patches, 2)
        values = self.value(safe) + self.position_value(grid.to(safe.dtype))[None, None]
        values = torch.where(observed[..., None], values, torch.zeros_like(values))
        read = torch.einsum("bkcp,bcph->bkch", weights, values.float())
        return read, weights

    def forward(
        self,
        *,
        reference: InstructionReference,
        current_dino: Tensor,
        current_state: Tensor,
        facts: ObjectFactSet,
        binding: TargetBinding,
        task_context: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if current_dino.shape != reference.dino.shape:
            raise ValueError("current/reference patch charts disagree")
        if current_state.shape != reference.state.shape:
            raise ValueError("current/reference state charts disagree")
        if facts.camera_validity.shape[2] != self.cameras:
            raise ValueError("instruction comparison lost declared camera identity")
        object_ok = binding.supported
        safe_objects = torch.where(
            object_ok[..., None], facts.content, torch.zeros_like(facts.content)
        )
        query = self.query(safe_objects) + self.task_query(task_context)[:, None]
        # Raw input contains the real current image; G's physical view support
        # restricts which entity/view comparisons are legal. No historical
        # confidence or future visibility is used as a source mask.
        camera_observed = (facts.camera_validity[..., 0] > 0).any(1)
        current_observed = camera_observed[..., None].expand(current_dino.shape[:-1])
        now, now_weights = self._read(query, current_dino, current_observed)
        start, start_weights = self._read(query, reference.dino, reference.observed)
        view_ok = (
            (facts.camera_validity[..., 0] > 0)
            & reference.observed.any(-1)[:, None]
            & object_ok[..., None]
        )
        delta = torch.where(view_ok[..., None], now - start, torch.zeros_like(now))
        view_ids = torch.arange(self.cameras, device=delta.device)
        # Role-aware mapping before view reduction; role alone cannot invent a
        # change when observations agree. The null law is applied last.
        role = (1 + torch.tanh(self.view_role(view_ids)).float())[None, None]
        delta = delta * role
        view_weights = masked_probability(facts.log_camera_validity[..., 0], view_ok)
        per_object = (delta * view_weights[..., None]).sum(2)
        visual_delta = (per_object * binding.mass[..., None]).sum(1)

        def entropy(weights: Tensor) -> Tensor:
            safe = torch.where(weights > 0, weights, torch.ones_like(weights))
            return -(weights * safe.log()).sum(-1) / math.log(max(weights.shape[-1], 2))

        # Correspondence ambiguity and source availability stay separate from
        # the change. Missing reference is NOT a confidently stationary target.
        status = torch.stack(
            (
                entropy(now_weights),
                entropy(start_weights),
                view_ok.float(),
                reference.observed.float().mean(-1)[:, None].expand_as(view_weights),
            ),
            -1,
        )
        status = torch.where(object_ok[..., None, None], status, torch.zeros_like(status))
        status_features = self.match_status(status.to(query.dtype)).float() * role
        # Fixed declared camera mean keeps missing-view status; do not normalize
        # a learned match confidence into a physical source-validity mask.
        status_context = (status_features.mean(2) * binding.mass[..., None]).sum(1)
        robot_delta = self.state_delta((current_state - reference.state).to(query.dtype)).float()
        combined = visual_delta + status_context + robot_delta * binding.mass.sum(-1, keepdim=True)
        output, _ = smooth_rms_contract(self.read_output(combined.to(query.dtype)), 0.35)
        return output, {
            "reference_current_read": now,
            "reference_start_read": start,
            "reference_current_weights": now_weights,
            "reference_start_weights": start_weights,
            "reference_view_available": view_ok,
        }
