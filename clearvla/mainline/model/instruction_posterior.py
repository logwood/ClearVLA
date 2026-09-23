"""Full G3-law correspondence and independently prepared S/P3 posterior reads.

The same current patch queries read both observed charts. G3's actual spatial
law mixes these conditional reads AFTER soft matching, not before it. Every
match includes a learned null alternative; no patch, object, time or task is
selected by argmax. All image work happens during online encoding.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from clearvla.vision.entity_chart import ImageLogMeasure, current_image_grid

from ..instruction_change import InstructionChangeEvidence
from ..instruction_posterior import InstructionChangeValues, InstructionPosterior
from ..instruction_reference import InstructionReference
from .routing import smooth_rms_contract
from .target_binding import TargetBinding, masked_probability
from .types import ObjectFactSet


class PosteriorChangeValueRead(nn.Module):
    """Nonlinear features BEFORE expectations; content, space and joint stay separate."""

    role_basis: Tensor

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        state_dim: int,
        camera_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.content_dim = content_dim
        self.state_dim = state_dim
        self.camera_names = camera_names

        def feature(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, hidden, bias=False),
            )

        self.content = feature(content_dim)
        self.image = feature(2)
        self.joint_content = nn.Linear(content_dim, hidden, bias=False)
        self.joint_image = nn.Linear(2, hidden, bias=False)
        self.joint_output = nn.Linear(hidden, hidden, bias=False)
        self.robot = nn.Linear(state_dim, hidden, bias=False)
        self.status = nn.Linear(8, hidden, bias=False)
        self.view_role = nn.Linear(len(camera_names), hidden, bias=False)
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(name) for name in camera_names]],
            persistent=False,
        )

    def prepare(self, evidence: InstructionChangeEvidence) -> InstructionChangeValues:
        evidence.validate(hidden_content=self.content_dim)
        posterior = evidence.posterior
        if (
            posterior is None
            or evidence.camera_names != self.camera_names
            or evidence.robot_delta.shape[-1] != self.state_dim
        ):
            raise ValueError("posterior value reader requires its own typed source chart")
        dtype = self.robot.weight.dtype
        # Mask BEFORE projections/nonlinearities: unsupported NaN payload must
        # never produce a NaN*0 parameter gradient.
        now = torch.where(posterior.observed[..., None], posterior.current, 0.0).to(dtype)
        start = torch.where(posterior.observed[..., None], posterior.reference, 0.0).to(dtype)
        xy = posterior.coordinates.to(dtype)
        content_now = self.content(now)
        content_start = self.content(start)
        image = self.image(xy)
        position = self.joint_image(xy)[None, None]
        joint_now = self.joint_output(
            torch.nn.functional.silu(self.joint_content(now) + position)
        )
        joint_start = self.joint_output(
            torch.nn.functional.silu(self.joint_content(start) + position)
        )
        with torch.autocast(device_type=now.device.type, enabled=False):
            def contrast(current: Tensor, reference: Tensor) -> Tensor:
                return torch.einsum(
                    "bkcn,bcnh->bkch", posterior.current_probability, current.float()
                ) - torch.einsum(
                    "bkcn,bcnh->bkch", posterior.reference_probability, reference.float()
                )

            content = contrast(content_now, content_start)
            image_delta = torch.einsum(
                "bkcn,nh->bkch",
                posterior.current_probability - posterior.reference_probability,
                image.float(),
            )
            joint = contrast(joint_now, joint_start)
        support = evidence.view_observed[..., None]
        object_support = evidence.binding.supported[..., None, None]
        result = InstructionChangeValues(
            content=torch.where(support, content, 0.0),
            image=torch.where(support, image_delta, 0.0),
            joint=torch.where(support, joint, 0.0),
            robot=self.robot(evidence.robot_delta.to(dtype)).float(),
            status=self.status(
                torch.where(object_support, evidence.match_status, 0.0).to(dtype)
            ).float(),
            evidence=evidence,
            reader_identity=id(self),
        )
        result.validate(hidden=self.hidden)
        return result

    def forward(
        self,
        evidence: InstructionChangeEvidence,
        prepared: InstructionChangeValues | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        values = self.prepare(evidence) if prepared is None else prepared
        values.validate(hidden=self.hidden)
        if values.evidence is not evidence or values.reader_identity != id(self):
            raise ValueError("prepared instruction values belong to another source or reader")
        if evidence.camera_names != self.camera_names:
            raise ValueError("prepared instruction camera chart differs")
        if values.joint is None:
            raise ValueError("posterior value read requires its joint feature law")
        dtype = self.robot.weight.dtype
        role = (1.0 + torch.tanh(self.view_role(self.role_basis.to(dtype)))).float()[None, None]
        weights = evidence.view_weight[..., None]
        mass = evidence.binding.mass[..., None]

        def reduce(value: Tensor) -> Tensor:
            return ((value * role * weights).sum(2) * mass).sum(1)

        status = ((values.status * role).mean(2) * mass).sum(1)
        robot = values.robot * mass.sum(1)
        # Correspondence-null and operated-target-null are separate. Neither is
        # renormalized into a confident real-object observation.
        return (
            reduce(values.content), reduce(values.image), reduce(values.joint),
            robot, status,
        )


class PosteriorInstructionReferenceRead(nn.Module):
    """Current G3 source law → soft current/reference reads; not permanent K IDs."""

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        state_dim: int,
        camera_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.camera_names = camera_names
        self.scale = hidden ** -0.5
        self.query = nn.Linear(content_dim, hidden, bias=False)
        self.key = nn.Linear(content_dim, hidden, bias=False)
        self.query_position = nn.Linear(2, hidden, bias=False)
        self.key_position = nn.Linear(2, hidden, bias=False)
        self.null_key = nn.Parameter(torch.zeros(hidden))
        self.values = PosteriorChangeValueRead(
            hidden=hidden, content_dim=content_dim,
            state_dim=state_dim, camera_names=camera_names,
        )
        self.output = nn.Linear(5 * hidden, hidden, bias=False)

    def _kernel(
        self, query: Tensor, content: Tensor, observed: Tensor, grid: Tensor
    ) -> Tensor:
        safe = torch.where(observed[..., None], content, 0.0).to(self.key.weight.dtype)
        key = self.key(safe) + self.key_position(
            grid.to(self.key.weight.dtype)
        )[None, None]
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = torch.einsum("bcnh,bcmh->bcnm", query.float(), key.float()) * self.scale
            logits = logits.masked_fill(~observed[:, :, None], -torch.inf)
            null = torch.einsum(
                "bcnh,h->bcn", query.float(), self.null_key.float()
            ) * self.scale
            return torch.softmax(torch.cat((logits, null[..., None]), -1), dim=-1)

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
            raise ValueError("posterior instruction current/reference charts differ")
        batch, cameras, patches, channels = current_dino.shape
        side = math.isqrt(patches)
        if side * side != patches or cameras != len(self.camera_names):
            raise ValueError("posterior comparison requires its full named native chart")
        source = facts.current_image_source
        if source is None or source.spatial is not facts.dense_chart.current_image_support:
            raise ValueError("posterior comparison requires the actual G3 image source")
        camera = (facts.camera_validity[..., 0] > 0).any(1)
        now_source = camera[..., None].expand(batch, cameras, patches)
        common = now_source & reference.observed
        # Reproject the ORIGINAL log measure, not an upsampled 8x8 barycenter.
        image = source.on_image(rows=side, columns=side)
        spatial_support = (
            image.supported
            & common[:, None].reshape(batch, 1, cameras, side, side)
            & binding.supported[:, :, None, None, None]
        )
        source_probability, _ = ImageLogMeasure(
            image.log_mass, spatial_support
        ).normalized((-2, -1))
        source_probability = source_probability.flatten(-2)
        view_ok = spatial_support.flatten(-2).any(-1) & (facts.camera_validity[..., 0] > 0)
        source_probability = torch.where(view_ok[..., None], source_probability, 0.0)
        grid = current_image_grid(
            side, side, device=current_dino.device
        ).reshape(patches, 2).float()
        safe_now = torch.where(common[..., None], current_dino, 0.0).to(self.query.weight.dtype)
        query = self.query(safe_now) + self.query_position(
            grid.to(self.query.weight.dtype)
        )[None, None]
        kernel_now = self._kernel(query, current_dino, common, grid)
        kernel_start = self._kernel(query, reference.dino, common, grid)
        with torch.autocast(device_type=current_dino.device.type, enabled=False):
            def mixture(kernel: Tensor) -> tuple[Tensor, Tensor]:
                return (
                    torch.einsum("bkcn,bcnm->bkcm", source_probability, kernel[..., :-1]),
                    torch.einsum("bkcn,bcn->bkc", source_probability, kernel[..., -1]),
                )

            p_now, null_now = mixture(kernel_now)
            p_start, null_start = mixture(kernel_start)
            now = torch.einsum(
                "bkcn,bcnd->bkcd", p_now,
                torch.where(common[..., None], current_dino.float(), 0.0),
            )
            start = torch.einsum(
                "bkcn,bcnd->bkcd", p_start,
                torch.where(common[..., None], reference.dino.float(), 0.0),
            )
            xy = torch.einsum("bkcn,nd->bkcd", p_now - p_start, grid)

            def entropy(probability: Tensor, null: Tensor) -> Tensor:
                law = torch.cat((probability, null[..., None]), -1)
                safe = torch.where(law > 0, law, 1.0)
                return -(law * safe.log()).sum(-1) / math.log(patches + 1)

            shape = p_now.shape[:-1]
            status = torch.stack(
                (
                    entropy(p_now, null_now), entropy(p_start, null_start), view_ok.float(),
                    now_source.float().mean(-1)[:, None].expand(shape),
                    reference.observed.float().mean(-1)[:, None].expand(shape),
                    common.float().mean(-1)[:, None].expand(shape),
                    null_now, null_start,
                ),
                -1,
            )
        posterior = InstructionPosterior(
            current_dino, reference.dino, common, grid, source_probability,
            p_now, p_start, null_now, null_start, source,
        )
        evidence = InstructionChangeEvidence(
            content_delta=torch.where(view_ok[..., None], now - start, 0.0),
            image_delta=torch.where(view_ok[..., None], xy, 0.0),
            robot_delta=current_state.float() - reference.state.float(),
            match_status=torch.where(binding.supported[..., None, None], status, 0.0),
            view_observed=view_ok,
            view_weight=masked_probability(facts.log_camera_validity[..., 0], view_ok),
            binding=binding, current_state=current_state, reference=reference,
            camera_names=self.camera_names, posterior=posterior,
        )
        evidence.validate(hidden_content=channels, strict=True)
        values = self.values(evidence)
        output, _ = smooth_rms_contract(
            self.output(torch.cat(values, -1).to(self.output.weight.dtype)), 0.35
        )
        return output, evidence


class PosteriorInstructionChangePlanRead(nn.Module):
    """P3 reads independently cached projections and dynamic plan context only."""

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        state_dim: int,
        camera_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.values = PosteriorChangeValueRead(
            hidden=hidden, content_dim=content_dim,
            state_dim=state_dim, camera_names=camera_names,
        )
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.modulations = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(4)
        )
        self.output = nn.Linear(4 * hidden, hidden, bias=False)

    def prepare(self, evidence: InstructionChangeEvidence) -> InstructionChangeValues:
        return self.values.prepare(evidence)

    def forward(
        self,
        evidence: InstructionChangeEvidence,
        query: Tensor,
        prepared: InstructionChangeValues | None = None,
    ) -> Tensor:
        if prepared is None:
            raise ValueError("P3 posterior projections must be prepared once during online encoding")
        if (
            query.ndim != 4
            or query.shape[0] != evidence.current_state.shape[0]
            or query.shape[-1] != self.query.in_features
        ):
            raise ValueError("posterior plan query must keep [B,T,Q,H]")
        *changes, status = self.values(evidence, prepared)
        context = self.query(query) + status.to(query.dtype)[:, None, None]
        terms = [
            value.to(context.dtype)[:, None, None] * (1.0 + torch.tanh(layer(context)))
            for value, layer in zip(changes, self.modulations, strict=True)
        ]
        # A status-only difference cannot create a policy-change value.
        return self.output(torch.cat(terms, -1))
