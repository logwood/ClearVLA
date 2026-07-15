from __future__ import annotations

"""Stage-owned nested contraction paths for action refinement."""

import math

import torch
from torch import Tensor, nn


class NestedLowRankContractionBank(nn.Module):
    """Contract a full-rank update along a stage-owned nested low-rank path.

    The unmodified full-rank update is the outer boundary of the family.  For
    an orthonormal basis ``Q`` and ordered transparencies ``m_i(d)`` this bank
    applies

        ``u(d) = u - Q diag(1 - m(d)) Q^T u``.

    At maximum depth every transparency is exactly one, so ``u(d) == u``.
    Reducing depth can only close the same ordered channels continuously.  It
    therefore cannot add a new direction, amplify the update, or compensate
    for a removed channel by increasing a surviving channel.
    """

    def __init__(
        self,
        hidden_size: int,
        condition_size: int,
        stage_count: int,
        rank: int,
        group_count: int,
        *,
        depth_logit_init: float,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        condition_size = int(condition_size)
        stage_count = int(stage_count)
        rank = int(rank)
        group_count = int(group_count)
        if min(hidden_size, condition_size, stage_count, rank, group_count) < 1:
            raise ValueError("nested contraction dimensions must be positive")
        if rank > hidden_size:
            raise ValueError("contraction rank cannot exceed hidden_size")
        if rank % group_count:
            raise ValueError("contraction rank must be divisible by group_count")
        if float(depth_logit_init) <= 0.0:
            raise ValueError("depth_logit_init must be positive")

        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.stage_count = stage_count
        self.rank = rank
        self.group_count = group_count
        self.channels_per_group = rank // group_count

        self.basis_raw = nn.Parameter(torch.empty(stage_count, hidden_size, rank))
        self.condition_norm = nn.LayerNorm(condition_size, elementwise_affine=False)
        # One scalar depth per semantic stage is deliberate: independent rank
        # masks would no longer describe a nested path.
        self.depth_weight = nn.Parameter(torch.zeros(stage_count, condition_size))
        self.depth_bias = nn.Parameter(
            torch.full((stage_count,), float(depth_logit_init))
        )
        self.reset_parameters(float(depth_logit_init))

    def reset_parameters(self, depth_logit_init: float) -> None:
        nn.init.normal_(self.basis_raw, mean=0.0, std=float(self.hidden_size) ** -0.5)
        with torch.no_grad():
            self.basis_raw.copy_(self._orthonormal_columns(self.basis_raw.float()))
        nn.init.zeros_(self.depth_weight)
        nn.init.constant_(self.depth_bias, float(depth_logit_init))

    def factor_parameters(self) -> tuple[nn.Parameter]:
        """Scale-invariant basis parameters, which must not use weight decay."""
        return (self.basis_raw,)

    def control_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Parameters that choose depth without controlling residual amplitude."""
        return self.depth_weight, self.depth_bias

    @staticmethod
    def _orthonormal_columns(value: Tensor) -> Tensor:
        q, r = torch.linalg.qr(value.float(), mode="reduced")
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        sign = torch.where(
            diagonal < 0.0,
            -torch.ones_like(diagonal),
            torch.ones_like(diagonal),
        )
        return q * sign.unsqueeze(-2)

    def prepare_factors(self) -> Tensor:
        """Build all stage bases once for reuse across recurrent steps."""
        return self._orthonormal_columns(self.basis_raw.float())

    @staticmethod
    def _candidate_rows(value: Tensor, stage_candidates: Tensor) -> Tensor:
        flat = stage_candidates.reshape(-1)
        return value.index_select(0, flat).reshape(
            *stage_candidates.shape, *value.shape[1:]
        )

    @staticmethod
    def _selected_candidate(value: Tensor, selected: Tensor) -> Tensor:
        gather_shape = (int(value.shape[0]), 1) + (1,) * (value.ndim - 2)
        gather_index = selected.reshape(gather_shape).expand(
            int(value.shape[0]), 1, *value.shape[2:]
        )
        return value.gather(1, gather_index).squeeze(1)

    def _candidate_bases(
        self,
        stage_candidates: Tensor,
        prepared_factors: Tensor | None,
    ) -> Tensor:
        basis_all = self.prepare_factors() if prepared_factors is None else prepared_factors
        expected = (self.stage_count, self.hidden_size, self.rank)
        if tuple(basis_all.shape) != expected:
            raise ValueError(
                f"prepared contraction bases must have shape {expected}, "
                f"got {tuple(basis_all.shape)}"
            )
        batch, candidate_count = stage_candidates.shape
        return basis_all.index_select(0, stage_candidates.reshape(-1)).reshape(
            batch, candidate_count, self.hidden_size, self.rank
        )

    def _ordered_transparency(self, depth_ratio: Tensor) -> Tensor:
        """Map a scalar depth to a continuous nested prefix of rank channels."""
        group_depth = depth_ratio.clamp(0.0, 1.0) * float(self.group_count)
        group_index = torch.arange(
            self.group_count,
            device=depth_ratio.device,
            dtype=depth_ratio.dtype,
        )
        group_transparency = (
            group_depth[..., None] - group_index
        ).clamp(0.0, 1.0)
        return group_transparency.repeat_interleave(
            self.channels_per_group, dim=-1
        )

    @staticmethod
    def _normalize_stage_probabilities(
        stage_probabilities: Tensor | None,
        *,
        stage_candidates: Tensor,
        device: torch.device,
    ) -> Tensor:
        batch, candidate_count = stage_candidates.shape
        if stage_probabilities is None:
            if candidate_count != 1:
                raise ValueError("multi-stage candidates require stage_probabilities")
            return torch.ones(batch, 1, device=device, dtype=torch.float32)
        probabilities = stage_probabilities.to(device=device, dtype=torch.float32)
        if tuple(probabilities.shape) != tuple(stage_candidates.shape):
            raise ValueError("stage_probabilities must match stage_candidates")
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _depth_override(
        value: Tensor | float,
        *,
        batch: int,
        candidate_count: int,
        device: torch.device,
    ) -> Tensor:
        ratio = torch.as_tensor(value, device=device, dtype=torch.float32)
        if ratio.ndim == 0:
            ratio = ratio.expand(batch, candidate_count)
        elif tuple(ratio.shape) == (batch,):
            ratio = ratio[:, None].expand(batch, candidate_count)
        elif tuple(ratio.shape) != (batch, candidate_count):
            raise ValueError(
                "depth_ratio_override must be scalar, [B], or [B,K]"
            )
        return ratio.clamp(0.0, 1.0)

    def forward(
        self,
        base_update: Tensor,
        condition: Tensor,
        stage_index: Tensor,
        *,
        stage_candidates: Tensor | None = None,
        stage_probabilities: Tensor | None = None,
        contraction_progress: Tensor | float = 0.0,
        prepared_factors: Tensor | None = None,
        depth_ratio_override: Tensor | float | None = None,
        identity_bypass: bool | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if base_update.ndim != 3 or int(base_update.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"base_update must be [B,N,{self.hidden_size}], "
                f"got {tuple(base_update.shape)}"
            )
        batch = int(base_update.shape[0])
        if tuple(condition.shape) != (batch, self.condition_size):
            raise ValueError(
                f"condition must be [B,{self.condition_size}], "
                f"got {tuple(condition.shape)}"
            )
        if tuple(stage_index.shape) != (batch,):
            raise ValueError("stage_index must contain one index per sample")

        stage_index = stage_index.to(device=base_update.device, dtype=torch.long)
        if stage_candidates is None:
            stage_candidates = stage_index[:, None]
        else:
            stage_candidates = stage_candidates.to(
                device=base_update.device, dtype=torch.long
            )
        if stage_candidates.ndim != 2 or int(stage_candidates.shape[0]) != batch:
            raise ValueError("stage_candidates must be [B,K]")
        if stage_candidates.device.type == "cpu":
            if bool((stage_candidates < 0).any()) or bool(
                (stage_candidates >= self.stage_count).any()
            ):
                raise ValueError(
                    "stage_candidates contain an index outside the contraction bank"
                )
        selected_matches = stage_candidates == stage_index[:, None]
        if selected_matches.device.type == "cpu" and not bool(
            (selected_matches.sum(dim=-1) == 1).all()
        ):
            raise ValueError(
                "each selected stage must occur exactly once in stage_candidates"
            )
        selected = selected_matches.float().argmax(dim=-1)
        candidate_count = int(stage_candidates.shape[1])
        stage_probabilities = self._normalize_stage_probabilities(
            stage_probabilities,
            stage_candidates=stage_candidates,
            device=base_update.device,
        )

        normalized_condition = self.condition_norm(condition).float()
        depth_weight = self._candidate_rows(
            self.depth_weight.float(), stage_candidates
        )
        depth_bias = self._candidate_rows(self.depth_bias.float(), stage_candidates)
        depth_logits = torch.einsum(
            "bh,bkh->bk", normalized_condition, depth_weight
        ) / math.sqrt(float(self.condition_size))
        raw_depth_ratio = torch.sigmoid(depth_logits + depth_bias)

        progress = torch.as_tensor(
            contraction_progress,
            device=base_update.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        if progress.ndim != 0:
            raise ValueError("contraction_progress must be scalar")
        selected_raw_depth = self._selected_candidate(raw_depth_ratio, selected)
        if identity_bypass is None:
            # CPU probes may infer the fast path directly. Production passes a
            # Python boolean from the global-step schedule, avoiding a GPU
            # scalar synchronization for every branch and refinement step. An
            # explicit depth override takes precedence over the inferred
            # warm-up boundary so probes exercise the requested operator.
            identity_bypass = (
                depth_ratio_override is None
                and progress.device.type == "cpu"
                and float(progress.detach()) == 0.0
            )
        if identity_bypass:
            # A true topology-level bypass: identity warm-up must not pay for
            # QR/projection work or expose contraction parameters to the main
            # loss through a numerically cancelling graph.
            one_rows = torch.ones(
                batch, device=base_update.device, dtype=torch.float32
            )
            zero_rows = torch.zeros_like(one_rows)
            rank = torch.as_tensor(
                float(self.rank), device=base_update.device, dtype=torch.float32
            )
            metrics = {
                "depth_ratio": one_rows.mean(),
                "depth_ratio_min": one_rows.amin(),
                "depth_ratio_max": one_rows.amax(),
                "raw_depth_ratio": selected_raw_depth.detach().mean(),
                "effective_depth": rank,
                "available_depth": rank,
                "transparency_mean": one_rows.mean(),
                "transparency_min": one_rows.amin(),
                "transparency_max": one_rows.amax(),
                "contraction_progress": progress.detach(),
                "depth_usage_cost": zero_rows.mean(),
                "contraction_ratio": one_rows.mean(),
                "subspace_energy_fraction": zero_rows.mean(),
                "removed_fraction": zero_rows.mean(),
                "removed_rms": zero_rows.mean(),
                "boundary_identity_error": zero_rows.mean(),
                "nonexpansive_violation": zero_rows.mean(),
                "nested_order_violation": zero_rows.mean(),
                "basis_norm_error": zero_rows.mean(),
                "basis_orthogonality_error": zero_rows.mean(),
                "basis_raw_norm": self.basis_raw.detach().float().norm(dim=-2).mean(),
                "depth_ratio_rows": one_rows,
                "effective_depth_rows": one_rows * rank,
                "contraction_ratio_rows": one_rows,
                "subspace_energy_fraction_rows": zero_rows,
                "removed_fraction_rows": zero_rows,
                "nonexpansive_violation_rows": zero_rows,
            }
            return base_update, metrics

        basis = self._candidate_bases(stage_candidates, prepared_factors)
        if depth_ratio_override is None:
            # Exact identity warm-up: progress=0 pins every candidate to the
            # original operation.  The path opens continuously thereafter.
            depth_ratio = 1.0 - progress * (1.0 - raw_depth_ratio)
        else:
            depth_ratio = self._depth_override(
                depth_ratio_override,
                batch=batch,
                candidate_count=candidate_count,
                device=base_update.device,
            )
        transparency = self._ordered_transparency(depth_ratio)

        base_fp32 = base_update.float()
        coordinates = torch.einsum("bnh,bkhr->bknr", base_fp32, basis)
        removed_coordinates = coordinates * (
            1.0 - transparency[:, :, None, :]
        )
        removed = torch.einsum(
            "bknr,bkhr->bknh", removed_coordinates, basis
        )
        candidate_updates = base_fp32[:, None] - removed

        hard_candidate_weight = selected_matches.to(dtype=candidate_updates.dtype)
        if self.training and candidate_count > 1:
            routing_weight = (
                hard_candidate_weight.detach()
                + stage_probabilities.to(dtype=candidate_updates.dtype)
                - stage_probabilities.detach().to(dtype=candidate_updates.dtype)
            )
        else:
            routing_weight = hard_candidate_weight
        update_fp32 = (
            candidate_updates * routing_weight[:, :, None, None]
        ).sum(dim=1)
        update = update_fp32.to(dtype=base_update.dtype)

        selected_depth = self._selected_candidate(depth_ratio, selected)
        selected_transparency = self._selected_candidate(transparency, selected)
        selected_basis = self._selected_candidate(basis, selected)
        selected_coordinates = self._selected_candidate(coordinates, selected)
        selected_removed = self._selected_candidate(removed, selected)
        base_rms_rows = base_fp32.square().mean(dim=(1, 2)).sqrt()
        output_rms_rows = update_fp32.detach().square().mean(dim=(1, 2)).sqrt()
        removed_rms_rows = selected_removed.detach().square().mean(dim=(1, 2)).sqrt()
        contraction_ratio_rows = output_rms_rows / base_rms_rows.detach().clamp_min(1e-8)
        removed_fraction_rows = removed_rms_rows / base_rms_rows.detach().clamp_min(1e-8)
        subspace_energy_fraction_rows = (
            selected_coordinates.detach().square().sum(dim=(1, 2))
            / base_fp32.detach().square().sum(dim=(1, 2)).clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        effective_depth_rows = selected_transparency.sum(dim=-1)
        identity = torch.eye(
            self.rank, device=selected_basis.device, dtype=selected_basis.dtype
        )[None]
        gram = torch.matmul(selected_basis.transpose(-2, -1), selected_basis)
        orthogonality_error_rows = (gram - identity).abs().amax(dim=(-2, -1))
        if self.rank > 1:
            nested_order_violation_rows = torch.relu(
                selected_transparency[:, 1:] - selected_transparency[:, :-1]
            ).amax(dim=-1)
        else:
            nested_order_violation_rows = torch.zeros_like(selected_depth)
        nonexpansive_violation_rows = torch.relu(contraction_ratio_rows - 1.0)
        full_transparency = self._ordered_transparency(
            torch.ones_like(depth_ratio)
        )
        boundary_identity_error = (full_transparency - 1.0).abs().amax()

        # Penalize chosen depth only.  This cannot shrink basis vectors or the
        # full-rank residual scale, and it does not train unselected stages.
        depth_usage_cost = progress * selected_raw_depth.mean()
        metrics = {
            "depth_ratio": selected_depth.detach().mean(),
            "depth_ratio_min": selected_depth.detach().amin(),
            "depth_ratio_max": selected_depth.detach().amax(),
            "raw_depth_ratio": selected_raw_depth.detach().mean(),
            "effective_depth": effective_depth_rows.detach().mean(),
            "available_depth": torch.as_tensor(
                float(self.rank), device=base_update.device, dtype=torch.float32
            ),
            "transparency_mean": selected_transparency.detach().mean(),
            "transparency_min": selected_transparency.detach().amin(),
            "transparency_max": selected_transparency.detach().amax(),
            "contraction_progress": progress.detach(),
            "depth_usage_cost": depth_usage_cost,
            "contraction_ratio": contraction_ratio_rows.detach().mean(),
            "subspace_energy_fraction": subspace_energy_fraction_rows.detach().mean(),
            "removed_fraction": removed_fraction_rows.detach().mean(),
            "removed_rms": removed_rms_rows.detach().mean(),
            "boundary_identity_error": boundary_identity_error.detach(),
            "nonexpansive_violation": nonexpansive_violation_rows.detach().amax(),
            "nested_order_violation": nested_order_violation_rows.detach().amax(),
            "basis_norm_error": (
                selected_basis.norm(dim=-2) - 1.0
            ).abs().detach().amax(),
            "basis_orthogonality_error": orthogonality_error_rows.detach().mean(),
            "basis_raw_norm": self.basis_raw.detach().float().norm(dim=-2).mean(),
            "depth_ratio_rows": selected_depth.detach(),
            "effective_depth_rows": effective_depth_rows.detach(),
            "contraction_ratio_rows": contraction_ratio_rows.detach(),
            "subspace_energy_fraction_rows": subspace_energy_fraction_rows.detach(),
            "removed_fraction_rows": removed_fraction_rows.detach(),
            "nonexpansive_violation_rows": nonexpansive_violation_rows.detach(),
        }
        return update, metrics
