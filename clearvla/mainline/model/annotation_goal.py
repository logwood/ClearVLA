"""A learned annotation-endpoint prior and soft, reference-aligned remaining relation.

Prediction reads only language and the observed instruction start. Current
observations are used afterwards, for comparison. Actual endpoint observations
are accepted only by the training target function. No task rules, success
thresholds, persistent IDs or elapsed-time features are used.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clearvla.vision.entity_chart import current_image_grid

from ..annotation_goal import (
    AnnotatedGoalEvidence,
    AnnotatedGoalValues,
    AnnotationEndpoint,
    EndpointGoalPrediction,
    annotation_goal_metadata,
)
from ..instruction_change import InstructionChangeEvidence
from ..instruction_reference import InstructionReference
from ..supervision import supported_mean


def normalized_observation(content: Tensor, observed: Tensor) -> Tensor:
    """Source masking precedes all normalization/projection arithmetic."""
    with torch.autocast(device_type=content.device.type, enabled=False):
        return F.normalize(torch.where(observed[..., None], content.float(), 0.0), dim=-1, eps=1e-6)


@torch.no_grad()
def reference_observation_law(
    reference: InstructionReference,
    observation: Tensor,
    observed: Tensor,
) -> Tensor:
    """Fixed label/current metric on the COMPLETE native chart, including null.

    The fixed cosine metric is not calibrated object identity. A uniform cell
    measure (-log N) prevents duplicating identical destination cells from
    changing the real/null prior. The reference query support is source-owned.
    No language, future duration, predicted confidence or learnable metric enters.
    """
    if (
        observation.shape != reference.dino.shape
        or observed.shape != observation.shape[:-1]
        or observed.dtype != torch.bool
    ):
        raise ValueError("reference measurement needs its named native source chart")
    with torch.autocast(device_type=observation.device.type, enabled=False):
        query = normalized_observation(reference.dino.detach(), reference.observed)
        key = normalized_observation(observation.detach(), observed)
        n, d = observation.shape[-2:]
        logits = torch.einsum("bcnd,bcmd->bcnm", query, key) * math.sqrt(d) - math.log(n)
        legal = reference.observed[..., None] & observed[:, :, None, :]
        logits = logits.masked_fill(~legal, -torch.inf)
        # A label metric's null must not learn to erase its own supervision.
        logits = torch.cat((logits, torch.zeros_like(logits[..., :1])), dim=-1)
        return logits.softmax(-1)


class EndpointGoalPredictor(nn.Module):
    """All reference patches and views interact with language before decoding.

    The dense expected scene is a finite conditional mean representation and
    can blur ambiguous outcomes. The separately decoded full spatial law can
    retain multiple destinations; neither is a certified goal or success value.
    """

    role_basis: Tensor

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        state_dim: int,
        heads: int,
        camera_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        annotation_goal_metadata(camera_names)
        self.camera_names = camera_names
        self.content_dim = content_dim
        self.input = nn.Linear(content_dim, hidden, bias=False)
        self.position = nn.Sequential(
            nn.Linear(2, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, hidden, bias=False)
        )
        self.view = nn.Linear(len(camera_names), hidden, bias=False)
        self.state = nn.Linear(state_dim, hidden, bias=False)
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.scene_read = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
        )
        self.content = nn.Linear(hidden, content_dim, bias=False)
        self.origin = nn.Linear(hidden, hidden, bias=False)
        self.destination = nn.Linear(hidden, hidden, bias=False)
        self.null = nn.Linear(hidden, 1)
        self.robot = nn.Linear(hidden, state_dim, bias=False)
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(n) for n in camera_names]],
            persistent=False,
        )

    def forward(
        self, *, reference: InstructionReference, language: Tensor
    ) -> EndpointGoalPrediction:
        b, c, n, d = reference.dino.shape
        side = math.isqrt(n)
        if side * side != n or c != len(self.camera_names) or d != self.content_dim:
            raise ValueError("endpoint goal requires a complete square native chart")
        if (
            language.ndim != 3
            or language.shape[0] != b
            or language.shape[-1] != self.input.out_features
        ):
            raise ValueError("endpoint goal language must use the protected S language chart")
        dtype = self.input.weight.dtype
        grid = current_image_grid(side, side, device=reference.dino.device).reshape(n, 2).float()
        raw = normalized_observation(reference.dino, reference.observed)
        spatial = self.position(grid.to(dtype))[None, None]
        role = self.view(self.role_basis.to(dtype))[None, :, None]
        state = self.state(reference.state.to(dtype))[:, None]
        tokens = self.input(raw.to(dtype)) + spatial + role
        flat = tokens.flatten(1, 2)
        # Language and observed robot state are always legal keys. Missing image
        # keys are quarantined, not a source of NaN gradients or learned validity.
        memory = torch.cat(
            (torch.where(reference.observed.flatten(1)[..., None], flat, 0.0), language, state),
            dim=1,
        )
        padding = torch.cat(
            (
                ~reference.observed.flatten(1),
                torch.zeros((b, language.shape[1] + 1), device=flat.device, dtype=torch.bool),
            ),
            dim=1,
        )
        query = torch.cat((flat, state), dim=1)
        update, _ = self.scene_read(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=padding,
            need_weights=False,
        )
        decoded = query + update
        decoded = decoded + self.ffn(decoded)
        scene_hidden = decoded[:, :-1].reshape(b, c, n, -1)
        scene_delta = self.content(scene_hidden)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            scene = F.normalize(raw + scene_delta.float(), dim=-1, eps=1e-6)
        # Future correspondence is learned, including its null alternative;
        # nothing chooses one destination by maximum or confidence threshold.
        origin = self.origin(scene_hidden)
        destination = self.destination(scene_hidden)
        null = self.null(scene_hidden)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            logits = torch.einsum(
                "bcnh,bcmh->bcnm", origin.float(), destination.float()
            ) / math.sqrt(origin.shape[-1]) - math.log(n)
            logits = logits.masked_fill(~reference.observed[..., None], -torch.inf)
            null = torch.where(reference.observed[..., None], null.float(), 0.0)
            log_probability = torch.cat((logits, null), dim=-1).log_softmax(-1)
        result = EndpointGoalPrediction(
            scene.float(),
            log_probability,
            self.robot(decoded[:, -1]).float(),
            grid,
            reference,
            self.camera_names,
        )
        result.validate()
        return result


def compare_goal_to_current(
    prediction: EndpointGoalPrediction, change: InstructionChangeEvidence
) -> AnnotatedGoalEvidence:
    """No label-capable input. Query identity stays rooted in the observed start."""
    posterior = change.posterior
    if posterior is None or change.reference is not prediction.reference:
        raise ValueError("goal comparison requires its own G3/reference evidence")
    # Existing source masks distinguish absent images from a measured change.
    observed = posterior.observed
    probability = reference_observation_law(prediction.reference, posterior.current, observed)
    content = normalized_observation(posterior.current.detach(), observed).detach()
    evidence = AnnotatedGoalEvidence(prediction, probability, content, observed, change)
    evidence.validate()
    return evidence


class AnnotatedGoalValueRead(nn.Module):
    """Nonlinear content, space and joint reads BEFORE any posterior averaging."""

    role_basis: Tensor

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.hidden = hidden
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
        self.view = nn.Linear(len(camera_names), hidden, bias=False)
        names = sorted(camera_names)
        self.register_buffer(
            "role_basis",
            torch.eye(len(names))[[names.index(n) for n in camera_names]],
            persistent=False,
        )

    def prepare(self, evidence: AnnotatedGoalEvidence) -> AnnotatedGoalValues:
        evidence.validate()
        p, change = evidence.prediction, evidence.instruction_change
        posterior = change.posterior
        assert posterior is not None
        if p.camera_names != self.camera_names:
            raise ValueError("goal reader camera roles do not match the source")
        dtype = self.robot.weight.dtype
        now = torch.where(evidence.current_observed[..., None], evidence.current_content, 0.0).to(
            dtype
        )
        end = torch.where(evidence.current_observed[..., None], p.scene, 0.0).to(dtype)
        xy = p.coordinates.to(dtype)
        now_c, end_c = self.content(now), self.content(end)
        spatial = self.image(xy)
        joint_xy = self.joint_image(xy)[None, None]
        now_joint = self.joint_output(F.silu(self.joint_content(now) + joint_xy))
        end_joint = self.joint_output(F.silu(self.joint_content(end) + joint_xy))
        with torch.autocast(device_type=now.device.type, enabled=False):
            # Missing current destinations are unknown, not observed absence.
            # Compare both laws on the same visible destinations plus null.
            support = torch.cat(
                (
                    evidence.current_observed[:, :, None, :].expand_as(p.log_probability[..., :-1]),
                    torch.ones_like(p.log_probability[..., -1:], dtype=torch.bool),
                ),
                dim=-1,
            )
            desired = p.log_probability.masked_fill(~support, -torch.inf).softmax(-1)[..., :-1]
            observed = evidence.current_probability[..., :-1]

            def contrast(goal: Tensor, current: Tensor) -> Tensor:
                return torch.einsum("bcnm,bcmh->bcnh", desired, goal.float()) - torch.einsum(
                    "bcnm,bcmh->bcnh", observed, current.float()
                )

            content = contrast(end_c, now_c)
            image = torch.einsum("bcnm,mh->bcnh", desired - observed, spatial.float())
            joint = contrast(end_joint, now_joint)
            # This is current-object -> start-patch evidence, NOT K-slot identity.
            # It retains both correspondence-null and operated-target-null mass.
            mapping = posterior.reference_probability

            def transport(value: Tensor) -> Tensor:
                with torch.autocast(device_type=value.device.type, enabled=False):
                    raw = torch.einsum("bkcn,bcnh->bkch", mapping.float(), value.float())
                    return torch.where(change.view_observed[..., None], raw, 0.0)

            observed_robot_delta = change.current_state.float() - p.reference.state.float()
            robot_error = p.robot_delta - observed_robot_delta
        result = AnnotatedGoalValues(
            transport(content),
            transport(image),
            transport(joint),
            self.robot(robot_error.to(dtype)).float(),
            evidence,
            id(self),
        )
        result.validate(hidden=self.hidden)
        return result

    def forward(
        self, evidence: AnnotatedGoalEvidence, prepared: AnnotatedGoalValues | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = self.prepare(evidence) if prepared is None else prepared
        values.validate(hidden=self.hidden)
        if values.evidence is not evidence or values.reader_identity != id(self):
            raise ValueError("prepared goal values belong to another source or reader")
        change = evidence.instruction_change
        if change.camera_names != self.camera_names:
            raise ValueError("prepared goal reader camera chart differs")
        dtype = self.robot.weight.dtype
        role = (1.0 + torch.tanh(self.view(self.role_basis.to(dtype)))).float()[None, None]
        weight = change.view_weight[..., None]
        mass = change.binding.mass[..., None]

        def reduce(value: Tensor) -> Tensor:
            return ((value * role * weight).sum(2) * mass).sum(1)

        return (
            reduce(values.content),
            reduce(values.image),
            reduce(values.joint),
            values.robot * mass.sum(1),
        )


class AnnotatedGoalPlanRead(nn.Module):
    """P3 desired-relation route, separate from observed-change/W-error routes."""

    def __init__(
        self, *, hidden: int, content_dim: int, state_dim: int, camera_names: tuple[str, ...]
    ) -> None:
        super().__init__()
        self.values = AnnotatedGoalValueRead(
            hidden=hidden, content_dim=content_dim, state_dim=state_dim, camera_names=camera_names
        )
        self.modulations = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(4))
        self.output = nn.Linear(4 * hidden, hidden, bias=False)

    def prepare(self, evidence: AnnotatedGoalEvidence) -> AnnotatedGoalValues:
        return self.values.prepare(evidence)

    def forward(
        self, evidence: AnnotatedGoalEvidence, query: Tensor, prepared: AnnotatedGoalValues | None
    ) -> Tensor:
        if prepared is None:
            raise ValueError("goal P3 projections must be prepared during online encoding")
        if (
            query.ndim != 4
            or query.shape[0] != evidence.prediction.scene.shape[0]
            or query.shape[-1] != self.values.hidden
        ):
            raise ValueError("goal P3 query must retain [B,T,Q,H]")
        values = self.values(evidence, prepared)
        terms = [
            v.to(query.dtype)[:, None, None] * (1.0 + torch.tanh(layer(query)))
            for v, layer in zip(values, self.modulations, strict=True)
        ]
        return self.output(torch.cat(terms, dim=-1))


def supervise_annotated_goal(
    prediction: EndpointGoalPrediction, label: AnnotationEndpoint
) -> dict[str, Tensor]:
    """Direct measured endpoints: no predicted confidence/target law masks losses."""
    prediction.validate()
    reference = prediction.reference
    if label.dino.shape != prediction.scene.shape or label.state.shape != reference.state.shape:
        raise ValueError("annotated goal target and prediction use different charts")
    target_scene = normalized_observation(label.dino.detach(), label.visual_observed)
    # All real endpoint pixels get scene supervision, including irrelevant
    # objects/background; a learned target/null cannot erase this objective.
    scene_rows = F.smooth_l1_loss(
        torch.where(label.visual_observed[..., None], prediction.scene, 0.0),
        target_scene,
        reduction="none",
    ).mean(-1)
    scene_loss = supported_mean(scene_rows, label.visual_observed)
    target_probability = reference_observation_law(reference, label.dino, label.visual_observed)
    query_support = reference.observed & label.visual_observed.any(-1, keepdim=True)
    # Unseen destination cells are unknown, not negatives. Condition BOTH laws
    # on actual labelled destinations + null; no top-k or confidence rejection.
    destination_support = torch.cat(
        (
            label.visual_observed[:, :, None, :].expand_as(target_probability[..., :-1]),
            torch.ones_like(target_probability[..., -1:], dtype=torch.bool),
        ),
        -1,
    )
    logits = prediction.log_probability.masked_fill(~destination_support, -torch.inf)
    logp = logits.log_softmax(-1)
    positive = target_probability > 0
    safe_q = torch.where(positive, target_probability, 1.0)
    safe_logp = torch.where(positive, logp, 0.0)
    kl_rows = (target_probability * (safe_q.log() - safe_logp)).sum(-1)
    relation_loss = supported_mean(kl_rows, query_support)
    safe_state = torch.where(label.state_observed[:, None], label.state.detach(), 0.0)
    state_target = torch.where(
        label.state_observed[:, None], safe_state - reference.state.detach().float(), 0.0
    )
    state_prediction = torch.where(label.state_observed[:, None], prediction.robot_delta, 0.0)
    robot_rows = F.smooth_l1_loss(state_prediction, state_target, reduction="none").mean(-1)
    robot_loss = supported_mean(robot_rows, label.state_observed)
    return {
        "annotated_goal_scene": scene_loss,
        "annotated_goal_relation": relation_loss,
        "annotated_goal_robot": robot_loss,
        "annotated_goal_total": (scene_loss + relation_loss + robot_loss) / 3.0,
        "annotated_goal_visual_coverage": label.visual_observed.float().mean().detach(),
        "annotated_goal_state_coverage": label.state_observed.float().mean().detach(),
    }
