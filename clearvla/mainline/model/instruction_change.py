"""Task-independent soft correspondence, typed causal changes, distinct S/P3 reads.

The matcher compares both raw image charts with the SAME current object query.
Language chooses the object through S's binding; it cannot alter this measurement
again through a direction-conditioned matching query. No physical ID is assumed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..instruction_change import InstructionChangeEvidence, instruction_change_metadata
from ..instruction_reference import InstructionReference
from .routing import smooth_rms_contract
from .target_binding import TargetBinding, masked_probability
from .types import ObjectFactSet


class TypedChangeValueRead(nn.Module):
    """Interpret source kinds and named views before reduction; never normalize values."""

    role_basis: Tensor

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        instruction_change_metadata(camera_names)
        self.hidden, self.content_dim, self.state_dim = hidden, content_dim, state_dim
        self.camera_names = camera_names
        self.content = nn.Linear(content_dim, hidden, bias=False)
        self.image = nn.Linear(2, hidden, bias=False)
        self.robot = nn.Linear(state_dim, hidden, bias=False)
        self.status = nn.Linear(6, hidden, bias=False)
        self.view_role = nn.Linear(len(camera_names), hidden, bias=False)
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(n) for n in camera_names]],
            persistent=False,
        )

    def forward(self, evidence: InstructionChangeEvidence) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        evidence.validate(hidden_content=self.content_dim)
        if (
            evidence.camera_names != self.camera_names
            or evidence.robot_delta.shape[-1] != self.state_dim
        ):
            raise ValueError("instruction read uses another camera/state chart")
        dtype = self.content.weight.dtype
        # Sanitize BEFORE projections. Match ambiguity is not source validity.
        valid = evidence.view_observed[..., None]
        content = self.content(torch.where(valid, evidence.content_delta, 0.0).to(dtype))
        image = self.image(torch.where(valid, evidence.image_delta, 0.0).to(dtype))
        role = (1.0 + torch.tanh(self.view_role(self.role_basis.to(dtype)))).float()[None, None]
        weights = evidence.view_weight[..., None]
        mass = evidence.binding.mass[..., None]

        def reduce_views(value: Tensor) -> Tensor:
            per_object = (value.float() * role * weights).sum(2)
            # Preserve null mass; no subsequent real-object renormalization.
            return (per_object * mass).sum(1)

        content_read, image_read = reduce_views(content), reduce_views(image)
        object_valid = evidence.binding.supported[..., None, None]
        status = self.status(torch.where(object_valid, evidence.match_status, 0.0).to(dtype))
        # Fixed camera count preserves a missing-source status, not an estimated
        # confidence converted into an observation availability gate.
        status_read = ((status.float() * role).mean(2) * mass).sum(1)
        robot = self.robot(evidence.robot_delta.to(dtype)).float() * mass.sum(1)
        return content_read, image_read, robot, status_read


class TypedInstructionReferenceRead(nn.Module):
    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.camera_names, self.scale = camera_names, hidden**-0.5
        self.query = nn.Linear(content_dim, hidden, bias=False)
        self.key = nn.Linear(content_dim, hidden, bias=False)
        self.values = TypedChangeValueRead(
            hidden=hidden, content_dim=content_dim, state_dim=state_dim, camera_names=camera_names
        )
        self.output = nn.Linear(4 * hidden, hidden, bias=False)

    def _read(
        self, query: Tensor, content: Tensor, observed: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, cameras, patches, _ = content.shape
        side = math.isqrt(patches)
        if side * side != patches or cameras != len(self.camera_names):
            raise ValueError("instruction comparison requires named square patch charts")
        safe = torch.where(observed[..., None], content, 0.0)
        logits = (
            torch.einsum(
                "bkh,bcph->bkcp", query.float(), self.key(safe.to(self.key.weight.dtype)).float()
            )
            * self.scale
        )
        weights = masked_probability(logits, observed[:, None].expand_as(logits))
        # Values remain separate raw observation charts until named consumers.
        with torch.autocast(device_type=query.device.type, enabled=False):
            read = torch.einsum("bkcp,bcpd->bkcd", weights, safe.float())
            axis = torch.linspace(-1.0, 1.0, side, device=query.device, dtype=torch.float32)
            y, x = torch.meshgrid(axis, axis, indexing="ij")
            grid = torch.stack((x, y), -1).reshape(patches, 2)
            location = torch.einsum("bkcp,pd->bkcd", weights, grid)
        return read, location, weights

    def forward(
        self,
        *,
        reference: InstructionReference,
        current_dino: Tensor,
        current_state: Tensor,
        facts: ObjectFactSet,
        binding: TargetBinding,
    ) -> tuple[Tensor, InstructionChangeEvidence]:
        if (
            current_dino.shape != reference.dino.shape
            or current_state.shape != reference.state.shape
        ):
            raise ValueError("instruction comparison current/reference charts differ")
        object_ok = binding.supported
        query = self.query(
            torch.where(object_ok[..., None], facts.content, 0.0).to(self.query.weight.dtype)
        )
        current_camera = (facts.camera_validity[..., 0] > 0).any(1)
        now_source = current_camera[..., None].expand(current_dino.shape[:-1])
        common = now_source & reference.observed
        # Both sides use the same available pixel chart: a newly missing source
        # cell alone must not masquerade as target movement in identical images.
        now, now_xy, p_now = self._read(query, current_dino, common)
        start, start_xy, p_start = self._read(query, reference.dino, common)
        view_ok = (
            (facts.camera_validity[..., 0] > 0) & common.any(-1)[:, None] & object_ok[..., None]
        )

        def entropy(p: Tensor) -> Tensor:
            safe = torch.where(p > 0, p, torch.ones_like(p))
            return -(p * safe.log()).sum(-1) / math.log(max(p.shape[-1], 2))

        expand = p_now.shape[:-1]
        status = torch.stack(
            (
                entropy(p_now),
                entropy(p_start),
                view_ok.float(),
                now_source.float().mean(-1)[:, None].expand(expand),
                reference.observed.float().mean(-1)[:, None].expand(expand),
                common.float().mean(-1)[:, None].expand(expand),
            ),
            -1,
        )
        evidence = InstructionChangeEvidence(
            content_delta=torch.where(view_ok[..., None], now - start, 0.0),
            image_delta=torch.where(view_ok[..., None], now_xy - start_xy, 0.0),
            robot_delta=current_state.float() - reference.state.float(),
            match_status=torch.where(object_ok[..., None, None], status, 0.0),
            view_observed=view_ok,
            view_weight=masked_probability(facts.log_camera_validity[..., 0], view_ok),
            binding=binding,
            current_state=current_state,
            reference=reference,
            camera_names=self.camera_names,
        )
        evidence.validate(hidden_content=self.query.in_features, strict=True)
        values = self.values(evidence)
        combined = self.output(torch.cat(values, -1).to(self.output.weight.dtype))
        output, _ = smooth_rms_contract(combined, 0.35)
        return output, evidence


class InstructionChangePlanRead(nn.Module):
    """P3 may interpret measured changes using its plan; status is not a change value."""

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.values = TypedChangeValueRead(
            hidden=hidden, content_dim=content_dim, state_dim=state_dim, camera_names=camera_names
        )
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.modulations = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.output = nn.Linear(3 * hidden, hidden, bias=False)

    def forward(self, evidence: InstructionChangeEvidence, query: Tensor) -> Tensor:
        if (
            query.ndim != 4
            or query.shape[0] != evidence.current_state.shape[0]
            or query.shape[-1] != self.query.in_features
        ):
            raise ValueError(
                "instruction plan query must be [B,T,Q,H] in the declared feature width"
            )
        content, image, robot, status = self.values(evidence)
        context = self.query(query) + status.to(query.dtype)[:, None, None]
        changes = (content, image, robot)
        terms = [
            value.to(context.dtype)[:, None, None] * (1.0 + torch.tanh(layer(context)))
            for value, layer in zip(changes, self.modulations)
        ]
        # No status-only value/bias: all three actual changes zero => zero
        # contribution AND zero derivative to plan/status context.
        return self.output(torch.cat(terms, -1))
