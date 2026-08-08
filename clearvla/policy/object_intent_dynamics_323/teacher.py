"""No-grad object-aligned multi-frame future teacher."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .types import INTERVAL_BOUNDS, FutureObjectDynamics, ObjectFactSet


class ObjectFutureTeacher(nn.Module):
    """Associate every current object with future DINO supports.

    The content projection is fixed and used only as a low-rank association
    key.  Full-width DINO values are retained in the target.  The association
    contains a legal null candidate, supports non-equal pixel positions and
    never enters the deployment forward path.
    """

    def __init__(self, *, content_dim: int, key_dim: int = 64) -> None:
        super().__init__()
        self.content_dim = int(content_dim)
        self.key_dim = int(key_dim)
        self.semantic_key = nn.Linear(content_dim, key_dim, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.semantic_key.weight)
        self.semantic_key.requires_grad_(False)

    @staticmethod
    def _offsets(offsets: Tensor, *, batch: int, supports: int) -> Tensor:
        if offsets.ndim == 1:
            if int(offsets.shape[0]) != supports:
                raise ValueError("future offsets do not match teacher supports")
            return offsets[None].expand(batch, -1)
        if tuple(offsets.shape) != (batch, supports):
            raise ValueError("future offsets must be [F] or [B,F]")
        if not bool((offsets == offsets[:1]).all()):
            raise ValueError("batched future offsets must agree across the batch")
        return offsets

    @torch.no_grad()
    def forward(
        self,
        *,
        facts: ObjectFactSet,
        future_supports: Tensor,
        future_offsets: Tensor,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        # Teacher association is a target-construction plane, not an online
        # BF16 value path.  An outer training autocast would otherwise cast
        # Linear/einsum outputs back to BF16 even though their operands were
        # explicitly converted with ``.float()``.  Keep semantic similarity,
        # the 129-way softmax, moments, and exported targets in FP32.
        if future_supports.device.type in {"cpu", "cuda"}:
            with torch.autocast(
                device_type=future_supports.device.type,
                enabled=False,
            ):
                return self._forward_fp32(
                    facts=facts,
                    future_supports=future_supports,
                    future_offsets=future_offsets,
                )
        return self._forward_fp32(
            facts=facts,
            future_supports=future_supports,
            future_offsets=future_offsets,
        )

    def _forward_fp32(
        self,
        *,
        facts: ObjectFactSet,
        future_supports: Tensor,
        future_offsets: Tensor,
    ) -> tuple[FutureObjectDynamics, dict[str, Tensor]]:
        facts.validate()
        if future_supports.ndim != 6:
            raise ValueError("future supports must be [B,F,C,Y,X,D]")
        batch, supports, cameras, rows, columns, width = future_supports.shape
        if batch != facts.batch or width != self.content_dim:
            raise ValueError("future supports do not match current object content")
        offsets = self._offsets(
            future_offsets.to(device=future_supports.device),
            batch=batch,
            supports=supports,
        )
        objects = facts.objects
        current_key = F.normalize(
            self.semantic_key(facts.content.detach().float()), dim=-1, eps=1e-4
        )
        support_key = F.normalize(
            self.semantic_key(future_supports.detach().float()), dim=-1, eps=1e-4
        )
        semantic = torch.einsum("bkr,bfcyxr->bfkcyx", current_key, support_key)
        axis_y = torch.linspace(-1.0, 1.0, rows, device=future_supports.device)
        axis_x = torch.linspace(-1.0, 1.0, columns, device=future_supports.device)
        coordinate_y, coordinate_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
        coordinate = torch.stack((coordinate_x, coordinate_y), dim=-1)
        coordinate = coordinate.reshape(1, 1, 1, 1, rows, columns, 2)
        max_offset = offsets.float().amax().clamp_min(1.0)
        fraction = offsets.float() / max_offset
        # This is the explicit source->learned-flow displacement exported by
        # G3.  It is only a bounded prior: semantic matching remains global
        # within the camera, so zero flow and non-equal-pixel motion are legal.
        flow_hint = torch.tanh(facts.transport_prior.detach().float())
        prior = facts.coordinates.detach().float()[:, None] + (
            fraction[:, :, None, None] * flow_hint[:, None]
        )
        prior = prior.clamp(-1.0, 1.0)
        delta = coordinate - prior[:, :, :, None, None, None]
        support_width = facts.support.detach().float().clamp(0.03, 1.0)
        geometry = -0.5 * delta.square().sum(dim=-1) / (
            support_width[..., 0][:, None, :, None, None, None].square()
            + 0.08
            + 0.20 * fraction[:, :, None, None, None, None]
        )
        geometry = geometry.clamp(-8.0, 0.0)
        camera_prior = facts.object_to_chart.detach().float().sum(dim=(-2, -1))
        camera_prior = camera_prior / camera_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        camera_log = camera_prior.clamp_min(1e-5).log()[:, None, :, :, None, None]
        # Camera prior already integrates to one across cameras.  Normalize
        # the spatial candidate partition so one null hypothesis does not lose
        # merely because it competes with Y*X cells.  A fixed contrastive
        # temperature then rewards genuinely matching DINO evidence.
        candidate_logit = (
            6.0 * semantic
            + geometry
            + camera_log
            - math.log(float(max(rows * columns, 1)))
        )
        candidate_flat = candidate_logit.flatten(3)
        null_logit = torch.zeros(
            batch,
            supports,
            objects,
            1,
            device=future_supports.device,
            dtype=candidate_flat.dtype,
        )
        posterior = torch.softmax(
            torch.cat((candidate_flat, null_logit), dim=-1), dim=-1
        )
        candidate_posterior = posterior[..., :-1].reshape(
            batch, supports, objects, cameras, rows, columns
        )
        null_probability = posterior[..., -1:]
        support_content = future_supports.detach().float().reshape(
            batch, supports, cameras * rows * columns, width
        )
        candidate_flat_probability = candidate_posterior.flatten(3)
        matched = torch.einsum(
            "bfkn,bfnd->bfkd", candidate_flat_probability, support_content
        )
        successor_per_support = matched + null_probability * facts.content.detach().float()[:, None]
        candidate_coordinate = coordinate[0, 0, 0, 0].unsqueeze(0).expand(
            cameras, -1, -1, -1
        )
        transport_per_support = torch.einsum(
            "bfkcyx,cyxd->bfkd", candidate_posterior, candidate_coordinate
        ) - (1.0 - null_probability) * facts.coordinates.detach().float()[:, None]
        centered = coordinate - (
            facts.coordinates.detach().float()[:, None, :, None, None, None]
            + transport_per_support[:, :, :, None, None, None]
        )
        covariance_xx = torch.einsum(
            "bfkcyx,bfkcyx->bfk",
            candidate_posterior,
            centered[..., 0].square(),
        )
        covariance_xy = torch.einsum(
            "bfkcyx,bfkcyx->bfk",
            candidate_posterior,
            centered[..., 0] * centered[..., 1],
        )
        covariance_yy = torch.einsum(
            "bfkcyx,bfkcyx->bfk",
            candidate_posterior,
            centered[..., 1].square(),
        )
        covariance_per_support = torch.stack(
            (covariance_xx, covariance_xy, covariance_yy), dim=-1
        )
        entropy = -(
            posterior.clamp_min(1e-8) * posterior.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(float(cameras * rows * columns + 1))
        visibility_per_support = 1.0 - null_probability
        # A confident null match is still epistemically uncertain about the
        # future object state.  Plain posterior entropy would incorrectly call
        # it certain, so null mass supplies the fallback uncertainty floor.
        uncertainty_per_support = null_probability + visibility_per_support * entropy

        successor_rows: list[Tensor] = []
        transport_rows: list[Tensor] = []
        covariance_rows: list[Tensor] = []
        visibility_rows: list[Tensor] = []
        persistence_rows: list[Tensor] = []
        uncertainty_rows: list[Tensor] = []
        address_rows: list[Tensor] = []
        support_counts: list[Tensor] = []
        for lower, upper in INTERVAL_BOUNDS:
            selected = ((offsets >= lower) & (offsets <= upper)).float()
            # Configured support sets normally cover every interval.  The
            # fallback selects the nearest support deterministically without
            # creating a zero/random target.
            if not bool((selected.sum(dim=1) > 0).all()):
                midpoint = 0.5 * float(lower + upper)
                nearest = (offsets.float() - midpoint).abs().argmin(dim=1)
                selected = F.one_hot(nearest, num_classes=supports).float()
            weight = selected / selected.sum(dim=1, keepdim=True).clamp_min(1.0)
            successor_rows.append(torch.einsum("bf,bfkd->bkd", weight, successor_per_support))
            transport_rows.append(torch.einsum("bf,bfkd->bkd", weight, transport_per_support))
            covariance_rows.append(torch.einsum("bf,bfkd->bkd", weight, covariance_per_support))
            interval_visibility = torch.einsum(
                "bf,bfkd->bkd", weight, visibility_per_support
            )
            visibility_rows.append(interval_visibility)
            # Mean visibility and track continuity must not collapse into the
            # same target.  The geometric mean drops when any sampled support
            # loses the object and therefore measures persistence across the
            # complete interval rather than average observability.
            persistence_rows.append(
                torch.exp(
                    torch.einsum(
                        "bf,bfkd->bkd",
                        weight,
                        visibility_per_support.clamp_min(1e-6).log(),
                    )
                )
            )
            uncertainty_rows.append(
                torch.einsum("bf,bfkd->bkd", weight, uncertainty_per_support)
            )
            address_rows.append(torch.einsum("bf,bfkcyx->bkcyx", weight, candidate_posterior))
            support_counts.append(selected.sum(dim=1).float().mean())
        successor = torch.stack(successor_rows, dim=1)
        transport = torch.stack(transport_rows, dim=1)
        covariance = torch.stack(covariance_rows, dim=1)
        visibility = torch.stack(visibility_rows, dim=1)
        persistence_probability = torch.stack(persistence_rows, dim=1)
        uncertainty = torch.stack(uncertainty_rows, dim=1)
        future_address = torch.stack(address_rows, dim=1)
        current_validity = facts.validity.detach().float()[:, None]
        # Current object facts are visible and persistent by construction.
        # Export changes around that current state so a neutral/static future
        # is exactly zero and cannot become a constant P2 value shortcut.
        visibility_change = visibility - 1.0
        persistence_change = persistence_probability - 1.0
        validity = visibility * current_validity
        target = FutureObjectDynamics(
            current_reference=facts.content.detach().float(),
            successor_content=successor,
            semantic_delta=successor - facts.content.detach().float()[:, None],
            transport_mean=transport,
            transport_covariance=covariance,
            visibility=visibility_change,
            persistence=persistence_change,
            uncertainty=uncertainty,
            validity=validity,
            future_address=future_address,
            object_coordinates=facts.coordinates.detach().float(),
        )
        target.validate()
        metrics = {
            "object_teacher_visibility": visibility.mean(),
            "object_teacher_visibility_change": visibility_change.mean(),
            "object_teacher_persistence_change": persistence_change.mean(),
            "object_teacher_uncertainty": uncertainty.mean(),
            "object_teacher_semantic_delta_rms": target.semantic_delta.square().mean().sqrt(),
            "object_teacher_transport_rms": transport.square().mean().sqrt(),
            "object_teacher_null_probability": null_probability.mean(),
            "object_teacher_semantic_max": semantic.detach().float().amax(
                dim=(-3, -2, -1)
            ).mean(),
            "object_teacher_semantic_margin": (
                semantic.detach().float().amax(dim=(-3, -2, -1))
                - semantic.detach().float().mean(dim=(-3, -2, -1))
            ).mean(),
            "object_teacher_supports_per_interval": torch.stack(support_counts).mean(),
            "object_teacher_adjacent_cosine": F.cosine_similarity(
                successor[:, 1:].flatten(2), successor[:, :-1].flatten(2), dim=-1
            ).mean(),
        }
        return target, metrics
