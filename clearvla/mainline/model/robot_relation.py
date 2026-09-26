"""Robot--object visual relations, resolved per named view before fusion.

These are learned relational features, not calibrated robot--object distances.
No image-plane coordinate is subtracted from a proprioceptive coordinate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..world_robot import RobotWorldObservation
from .types import ObjectWorldBelief


class _RowwiseLinear(nn.Linear):
    """Linear projection with a fixed reduction order for every row.

    Some CUDA/CPU GEMM kernels use a reduction tiling that depends on a row's
    physical position.  That is numerically harmless for training, but it
    violates the strict object/camera permutation audit at this boundary.
    Each row here is tiny, so explicit matrix-vector products keep the
    reduction order fixed without changing the learned function.
    """

    def forward(self, value: Tensor) -> Tensor:
        shape = value.shape
        rows = value.reshape(-1, shape[-1])
        projected = torch.stack(
            [torch.mv(self.weight, row) for row in rows],
            dim=0,
        )
        if self.bias is not None:
            projected = projected + self.bias
        return projected.reshape(*shape[:-1], self.out_features)


@dataclass(frozen=True)
class RobotObjectRelation:
    pooled: Tensor  # [B,K,H], relation features, not averaged image vectors
    per_view: Tensor  # [B,K,C,H]
    belief: ObjectWorldBelief  # provenance; private to this W1/W2 calculation

    def validate_source(self, belief: ObjectWorldBelief) -> None:
        if belief is not self.belief:
            raise ValueError("W relation cannot be reused with a different observation belief")


class RobotObjectRelationEncoder(nn.Module):
    def __init__(self, *, state_dim: int, state_mode: str, content_dim: int,
                 hidden: int, camera_names: tuple[str, ...]) -> None:
        super().__init__()
        if state_dim < 1 or not state_mode:
            raise ValueError("robot-object W requires explicit current state features")
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("robot-object W requires unique named camera charts")
        self.state_dim = state_dim
        self.state_mode = state_mode
        self.camera_names = tuple(camera_names)
        self.state_projection = nn.Linear(state_dim, hidden, bias=False)
        self.object_projection = nn.Linear(content_dim, hidden, bias=False)
        # xy, image velocity xy, positive source-gap flag, named-view one-hot.
        self.view_projection = _RowwiseLinear(5 + len(camera_names), hidden, bias=False)
        self.relation = nn.Sequential(
            _RowwiseLinear(3 * hidden, 2 * hidden, bias=False), nn.SiLU(),
            _RowwiseLinear(2 * hidden, hidden, bias=False),
        )
        roles = {name: i for i, name in enumerate(sorted(camera_names))}
        self.role_basis: Tensor
        self.register_buffer("role_basis", torch.eye(len(roles))[
            [roles[name] for name in camera_names]
        ], persistent=False)

    def forward(self, belief: ObjectWorldBelief) -> RobotObjectRelation:
        belief.validate()
        observation = belief.robot_observation
        if not isinstance(observation, RobotWorldObservation):
            raise ValueError("robot-object W requires explicit current robot observation")
        observation.validate(batch=belief.batch, device=belief.content.device,
                             state_dim=self.state_dim, feature_mode=self.state_mode,
                             check_finite=True)
        if belief.latest_flow_steps is None:
            raise ValueError("robot-object W requires source-timed camera motion")
        if belief.camera_coordinates.shape[2] != len(self.camera_names):
            raise ValueError("robot-object W camera axes differ from named charts")
        valid = torch.nan_to_num(belief.validity.float(), nan=0.0, posinf=0.0, neginf=0.0) > 0
        weight = torch.nan_to_num(
            belief.camera_validity.float() * belief.camera_support.float(),
            nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(0)
        supported = (weight > 0) & valid[:, :, None]
        coordinates = torch.where(supported, belief.camera_coordinates.float(), 0.0)
        motion_observed = (belief.latest_flow_steps > 0)[:, None, None, None] & supported
        # Mask BEFORE rate computation, projection and nonlinear products. A
        # repeated/reset frame supplies a current image but no measured speed.
        displacement = torch.where(motion_observed, belief.camera_transport_prior.float(), 0.0)
        rate = displacement / belief.latest_flow_steps.clamp_min(1)[:, None, None, None]
        roles = self.role_basis.to(device=coordinates.device)[None, None].expand(
            belief.batch, belief.objects, -1, -1
        )
        view_input = torch.cat((coordinates, rate, motion_observed.float(), roles), dim=-1)
        # Advanced object-axis indexing used by causal permutation audits can
        # leave a different stride layout even when every row contains the
        # same values.  Materialize both projected inputs so GEMM reduction
        # order is independent of that layout and the equivariance contract
        # remains bit-exact on CPU/CUDA.
        view_input = torch.where(supported, view_input, 0.0).contiguous()
        content = torch.where(valid, belief.content, 0.0).contiguous()
        dtype = self.state_projection.weight.dtype
        robot = self.state_projection(observation.state.to(dtype=dtype))[:, None, None]
        objects = self.object_projection(content.to(dtype=dtype))[:, :, None]
        views = self.view_projection(view_input.to(dtype=dtype))
        spatial = objects + views
        robot = robot.expand_as(spatial)
        # Interaction happens while camera identity is intact. Equal/opposite
        # image vectors are not pre-averaged into a fictitious zero velocity.
        relation_input = torch.cat((robot, spatial, robot * torch.tanh(spatial)), dim=-1)
        relation = self.relation(relation_input)
        relation = torch.where(supported, relation, 0.0)
        weights = torch.where(supported, weight, 0.0)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        pooled = (relation.float() * weights).sum(dim=2).to(dtype=relation.dtype)
        return RobotObjectRelation(pooled=pooled, per_view=relation, belief=belief)
