"""Resolve geometric effects inside named views before any spatial pooling.

No calibration or Cartesian displacement is inferred here. Context changes the
interpretation of a W-predicted image displacement; it cannot generate an effect
on its own. All support is supplied by current observations and known controls.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..p2_geometry import validate_camera_names


class ViewConditionedTransport(nn.Module):
    def __init__(self, *, hidden: int, camera_names: tuple[str, ...]) -> None:
        super().__init__()
        validate_camera_names(camera_names)
        if hidden < 1:
            raise ValueError("P2 view geometry hidden width must be positive")
        self.hidden = hidden
        self.camera_names = tuple(camera_names)
        # Role columns have canonical named order, NOT input tensor order.
        canonical = {name: i for i, name in enumerate(sorted(camera_names))}
        self.role_basis: Tensor
        self.register_buffer(
            "role_basis",
            torch.eye(len(camera_names))[[canonical[name] for name in camera_names]],
            persistent=False,
        )
        # Current image xy, W's within-support covariance xx/xy/yy, role.
        self.context = nn.Sequential(
            nn.Linear(5 + len(camera_names), hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.query_role = nn.Linear(len(camera_names), hidden, bias=False)
        self.key_interaction = nn.Linear(hidden, hidden, bias=False)
        self.value_interaction = nn.Linear(hidden, hidden, bias=False)

    def validate_names(self, names: tuple[str, ...]) -> None:
        if names != self.camera_names:
            raise ValueError("W/P2 named camera charts differ or are missing")

    def context_features(self, coordinates: Tensor, covariance: Tensor, support: Tensor) -> Tensor:
        """[B,K,C,2], [B,I,K,C,3], Bool[B,I,K,C] -> [B,I,K,C,H]."""
        if covariance.ndim != 5 or covariance.shape[-1] != 3:
            raise ValueError("P2 covariance must retain [B,I,K,C,3]")
        b, _, k, c, _ = covariance.shape
        if c != len(self.camera_names) or coordinates.shape != (b, k, c, 2):
            raise ValueError("P2 current coordinates lost a named view")
        if support.dtype != torch.bool or support.shape != covariance.shape[:-1]:
            raise ValueError("P2 geometry support must retain [B,I,K,C]")
        if coordinates.device != covariance.device or support.device != covariance.device:
            raise ValueError("P2 geometry payload and support must share a device")
        # Quarantine BEFORE learned projections; a final zero multiply cannot
        # remove NaNs from forward products or reverse parameter derivatives.
        xy = torch.where(support[..., None], coordinates[:, None].float(), 0.0)
        cov = torch.where(support[..., None], covariance.float(), 0.0)
        roles = self.role_basis.to(device=xy.device)[None, None, None].expand(*support.shape, -1)
        inputs = torch.where(support[..., None], torch.cat((xy, cov, roles), -1), 0.0)
        first = self.context[0]
        assert isinstance(first, nn.Linear)
        context = self.context(inputs.to(dtype=first.weight.dtype))
        return torch.where(support[..., None], context, 0.0)

    def image_queries(self, action_query: Tensor, projection: nn.Linear) -> Tensor:
        """[B,T,Q,H] -> [B,T,Q,C,2], each query stays in its own image chart."""
        if action_query.ndim != 4 or action_query.shape[-1] != self.hidden:
            raise ValueError("P2 action query must be [B,T,Q,H]")
        if projection.in_features != self.hidden or projection.out_features != 2:
            raise ValueError("P2 coordinate projection differs from image query contract")
        role = self.query_role(self.role_basis.to(self.query_role.weight))
        private_query = action_query[..., None, :] * (1 + torch.tanh(role))
        return torch.tanh(projection(private_query).float())

    def condition(self, base: Tensor, context: Tensor, support: Tensor, *, value: bool) -> Tensor:
        """A conditional linear effect: F(0, context) == 0.

        Context-only parameter gradients then vanish; displacement derivatives
        remain available for learning. There is no context-only additive value,
        LayerNorm of a scaled target,
        allocation quota or cross-view coordinate sum. Real target mass is
        applied by the caller AFTER these object/view-internal transformations.
        """
        if base.shape != context.shape or base.shape[:-1] != support.shape:
            raise ValueError("P2 per-view effect and context lost an identity axis")
        if support.dtype != torch.bool or base.shape[-1] != self.hidden:
            raise ValueError("P2 per-view effect width/support differs from contract")
        clean = torch.where(support[..., None], base, 0.0)
        context = torch.where(support[..., None], context, 0.0)
        interaction = self.value_interaction if value else self.key_interaction
        result = clean + interaction(clean * torch.tanh(context))
        return torch.where(support[..., None], result, 0.0)
