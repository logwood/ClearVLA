"""Soft P3 read of observed discrepancy; no fixed K identity or goal oracle."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from clearvla.vision.entity_chart import ImageLogMeasure

from ..executed_world import ExecutedWorldFeedback, ExecutedWorldPlanValues
from .target_binding import TargetBinding
from .types import ObjectFactSet


class ExecutedWorldPlanRead(nn.Module):
    role_basis: Tensor

    def __init__(self, *, hidden: int, content_dim: int, camera_names: tuple[str, ...]) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.camera_names = tuple(camera_names)
        self.query = nn.Linear(content_dim, hidden, bias=False)
        self.key = nn.Linear(content_dim, hidden, bias=False)
        self.null_key = nn.Parameter(torch.zeros(hidden))
        self.semantic = nn.Linear(content_dim, hidden, bias=False)
        self.image = nn.Linear(2, hidden, bias=False)
        self.status = nn.Linear(4, hidden, bias=False)
        self.view_role = nn.Linear(len(camera_names), hidden, bias=False)
        self.context = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        ordered = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(ordered))[[ordered.index(n) for n in camera_names]],
            persistent=False,
        )

    def prepare(
        self, feedback: ExecutedWorldFeedback, facts: ObjectFactSet, binding: TargetBinding
    ) -> ExecutedWorldPlanValues:
        feedback.validate(strict=True)
        if feedback.camera_names != self.camera_names or facts.batch != feedback.semantic.shape[0]:
            raise ValueError("executed discrepancy camera/batch provenance differs")
        if binding.supported.shape != facts.content.shape[:2]:
            raise ValueError("executed discrepancy target has another object chart")
        source = facts.current_image_source
        if source is None or source.spatial is not facts.dense_chart.current_image_support:
            raise ValueError("executed discrepancy requires actual current G3 source")
        rows, columns = feedback.measurement_shape
        # Project actual G3 to the *same pooled grid* used by W measurement.
        # Gradients cannot alter measured evidence or the shared prior model.
        with torch.no_grad():
            image = source.on_image(rows=rows, columns=columns)
            support = image.supported & binding.supported[:, :, None, None, None]
            current_law, _ = ImageLogMeasure(image.log_mass, support).normalized((-2, -1))
            current_law = current_law.flatten(-2)
            current_views = support.flatten(-2).any(-1) & (facts.camera_validity[..., 0] > 0)
            current_law = torch.where(current_views[..., None], current_law, 0.0)
            past_law = torch.where(feedback.view_observed[..., None], feedback.posterior, 0.0)
            overlap = torch.einsum("bkcn,bjcn->bkj", current_law.float(), past_law.float())
            # Density ratio relative to a uniform cell measure, not a fixed
            # threshold or artificial renormalization of uncertain real mass.
            overlap = overlap * (rows * columns)
            legal = (current_views[:, :, None, :] & feedback.view_observed[:, None]).any(-1)
            legal = legal & binding.supported[..., None]
            safe_current = torch.where(binding.supported[..., None], facts.content.detach(), 0.0)
            past_supported = feedback.view_observed.any(-1)
            safe_past = torch.where(past_supported[..., None], feedback.past_content, 0.0)
        dtype = self.query.weight.dtype
        query = self.query(safe_current.to(dtype))
        key = self.key(safe_past.to(dtype))
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bkh,bjh->bkj", query.float(), key.float()) * self.hidden**-0.5
            log_overlap = (
                overlap.masked_fill(~legal, 0.0).clamp_min(torch.finfo(torch.float32).tiny).log()
            )
            logits = (logits + log_overlap).masked_fill(~legal | (overlap == 0), -torch.inf)
            null = (
                torch.einsum("bkh,h->bk", query.float(), self.null_key.float()) * self.hidden**-0.5
            )
            match = torch.softmax(torch.cat((logits, null[..., None]), -1), -1)[..., :-1]
        semantic = self.semantic(
            torch.where(past_supported[..., None], feedback.semantic, 0.0).to(dtype)
        )
        spatial = self.image(
            torch.where(feedback.view_observed[..., None], feedback.image, 0.0).to(dtype)
        )
        status = torch.cat(
            (
                feedback.covariance,
                feedback.null[:, :, None].expand(-1, -1, len(self.camera_names), -1),
            ),
            -1,
        )
        status = self.status(torch.where(feedback.view_observed[..., None], status, 0.0).to(dtype))
        role = 1.0 + torch.tanh(self.view_role(self.role_basis.to(dtype)))
        count = feedback.view_observed.sum(-1, keepdim=True).clamp_min(1)
        value = (semantic[:, :, None] + spatial) * (1.0 + torch.tanh(status)) * role[None, None]
        value = torch.where(feedback.view_observed[..., None], value, 0.0).sum(2) / count
        with torch.autocast(device_type=query.device.type, enabled=False):
            per_current = torch.einsum("bkj,bjh->bkh", match, value.float())
            prepared = (per_current * binding.mass[..., None]).sum(1)
        return ExecutedWorldPlanValues(prepared, feedback, facts, id(self), binding)

    def forward(
        self, prepared: ExecutedWorldPlanValues, context: Tensor, binding: TargetBinding | None
    ) -> Tensor:
        if prepared.reader_identity != id(self) or prepared.binding is not binding:
            raise ValueError("executed discrepancy cache belongs to another reader/target")
        if (
            prepared.value.shape != (context.shape[0], self.hidden)
            or context.shape[-1] != self.hidden
        ):
            raise ValueError("executed discrepancy cache lost B/H axes")
        dtype = self.output.weight.dtype
        value = prepared.value.to(dtype)[:, None, None]
        return self.output(value * torch.tanh(self.context(context.to(dtype))))
