"""Typed adapter from the integrated 3-2-3 top to the extracted V120 bottom.

The action solver in this module is not a reimplementation.  Its execution
blocks, evidence organizer, native noisy-action lift, ordered low-rank
capacity and candidate value reader are the mechanically extracted V120
``EvidenceLatentMMDiTActionDecoder``.  This adapter only translates the new
typed top interfaces into the exact inputs owned by that decoder.

The translation keeps the repaired ownership boundaries:

* P2's protected consequence is written once through V120's protected-detail
  reader while the historical generic trajectory ingress remains neutral;
* all five V120 P3 lanes are optional typed innovations;
* all 512 W transition rows reach the evidence bank without pooling;
* observation banks are never reopened below P1;
* teacher/future tensors cannot be represented by this online signature.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import ObservableHistory
from ..v120_core.profile import build_v120_policy_config
from ..v120_core.role_delta_attnres import PolicyRoleDeltaBank
from ..v120_core.layer_contracts import LayerContractAdapterHeads
from ..v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
)
from .action_contract import ActionQueryEncoder, BottomOutput, V120SeedContext
from .compiler import ObjectPolicyPlanDeltaBank
from .types import ControlledTransitionState, ObjectIntentState


def _build_decoder_config(config: ExperimentConfig):
    """Resolve the extracted V120 decoder against the active typed widths.

    Production defaults reproduce the serialized V120 profile exactly.  The
    explicit replacements make the same implementation usable by the small
    CPU contract tests without introducing a second miniature decoder.
    """

    config.validate()
    dims = config.dimensions
    bottom = config.bottom
    resolved = replace(
        build_v120_policy_config(),
        action_dim=dims.action_dim,
        state_dim=dims.state_dim,
        action_horizon=dims.action_horizon,
        executed_history_length=dims.executed_history_length,
        hidden_size=dims.hidden_size,
        num_heads=dims.num_heads,
        visual_token_dim=dims.visual_token_dim,
        patches_per_camera=dims.patches_per_camera,
        target_future_count=dims.future_supports,
        action_basis_tokens=dims.action_basis_tokens,
        gripper_field_dim=bottom.gripper_field_dim,
        physical_decode_delta_blend=bottom.physical_decode_delta_blend,
        dropout=bottom.dropout,
        latent_cvae_mmdit_depth=bottom.evidence_depth,
        latent_cvae_mmdit_operator_rank=bottom.operator_rank,
        latent_cvae_mmdit_operator_groups=bottom.operator_groups,
        latent_cvae_mmdit_operator_depth_logit_init=bottom.operator_depth_logit_init,
        latent_cvae_mmdit_control_tokens=bottom.controller_tokens,
        latent_cvae_mmdit_controller_depth=bottom.controller_depth,
        latent_cvae_mmdit_controller_heads=bottom.controller_heads,
        latent_cvae_mmdit_max_dwell=bottom.max_dwell,
        latent_cvae_mmdit_execution_warmup_steps=bottom.execution_warmup_steps,
        latent_cvae_mmdit_execution_transition_steps=bottom.execution_transition_steps,
        latent_cvae_mmdit_execution_eval_policy=bottom.execution_eval_policy,
    )
    resolved.validate()
    return resolved


class RestoredV120EvidenceBottom(nn.Module):
    """V120 Evidence-MMDiT with a capability-named typed ingress."""

    def __init__(self, config: ExperimentConfig, *, physical_action_dim: int) -> None:
        super().__init__()
        config.validate()
        dims = config.dimensions
        self.hidden = int(dims.hidden_size)
        self.horizon = int(dims.action_horizon)
        self.basis = int(dims.action_basis_tokens)
        self.physical_action_dim = int(physical_action_dim)
        self.core_config = _build_decoder_config(config)
        if self.physical_action_dim != int(self.core_config.physical_action_dim):
            raise ValueError("typed physical action width does not match V120")

        # This shared V120 query seeds P2/P3, the controlled transition and
        # the two layer contracts.  The restored decoder still owns its own
        # native physical-action lift, exactly as V120 did.
        self.query_encoder = ActionQueryEncoder(self.core_config)
        policy_start = int(self.core_config.depth) - int(
            self.core_config.flow_jepa_policy_blocks
        )
        # Strict V120 exposed P1/P2 contracts and replaced P3 with the typed
        # compiler.  Materialize exactly those two active heads, with their
        # original depth identities, instead of keeping six frozen ancestry
        # heads plus a frozen duplicate P3 head.
        self.layer_contract_heads = nn.ModuleList(
            (
                LayerContractAdapterHeads(
                    self.core_config,
                    layer_index=policy_start,
                ),
                LayerContractAdapterHeads(
                    self.core_config,
                    layer_index=policy_start + 1,
                ),
            )
        )
        for head in self.layer_contract_heads:
            # V120 trained only the small residual adapters in active P1/P2;
            # their weak probe/readout weights were fixed selector geometry.
            head.readout.requires_grad_(False)
        self.decoder = EvidenceLatentMMDiTActionDecoder(self.core_config)
        # These generic intent aliases are structurally absent from the V120
        # object path (which passes only current state and last execution).
        # Freezing unreachable projections changes no forward value and keeps
        # optimizer ownership honest without reintroducing the aliases.
        for source_name in ("task", "state_history", "proposal"):
            self.decoder.evidence_adapter.intent_proj[source_name].requires_grad_(
                False
            )
        # Generic trajectory is an exact-zero source in this path.  Its first
        # LayerNorm scale multiplies zero forever, while the affine biases and
        # following projection remain the trainable V120 null-value geometry.
        trajectory_projection = cast(
            nn.Sequential,
            self.decoder.evidence_adapter.source_proj["trajectory"],
        )
        trajectory_norm = trajectory_projection[0]
        if not isinstance(trajectory_norm, nn.LayerNorm):
            raise TypeError("V120 trajectory projection must start with LayerNorm")
        if trajectory_norm.weight is not None:
            trajectory_norm.weight.requires_grad_(False)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.decoder.blocks

    @property
    def capacity(self) -> nn.ModuleList:
        return self.decoder.operator_contractions

    @property
    def execution(self) -> nn.Module | None:
        return self.decoder.execution_controller

    def action_query(self, noisy_action_field: Tensor, time: Tensor) -> Tensor:
        return self.query_encoder(noisy_action_field, time)

    def action_and_context(
        self,
        noisy_action_field: Tensor,
        time: Tensor,
        history: ObservableHistory,
        *,
        executed_memory: Tensor,
        action_history_keep: Tensor,
    ) -> tuple[Tensor, V120SeedContext]:
        """Build the one shared V120 seed consumed by top and bottom."""

        return self.query_encoder.forward_with_context(
            noisy_action_field,
            time,
            history,
            executed_memory=executed_memory,
            action_history_keep=action_history_keep,
        )

    def clean_action_basis_tokens(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return self.query_encoder.clean_action_basis_tokens(
            batch,
            device=device,
            dtype=dtype,
        )

    def set_training_step(self, global_step: int) -> float:
        return self.decoder.set_execution_training_step(global_step)

    def _state_memory(
        self,
        seed: V120SeedContext,
    ) -> tuple[Tensor, Tensor, Tensor]:
        seed.validate(
            hidden=self.hidden,
            state_history=int(self.core_config.visual_history_length),
            executed=int(self.core_config.action_history_token_count),
        )
        # Match the active V120 object path: current state plus the final
        # causal state-history row, and only the final compressed execution
        # row in the compact intent bank.
        return seed.state, seed.state_history[:, -1:], seed.executed[:, -1:]

    def _neutral_trajectory_memory(self, plan: ObjectPolicyPlanDeltaBank) -> Tensor:
        """Restore V120's neutral generic proposal ingress.

        The protected consequence is retained at full ``[T,Q]`` resolution in
        ``PolicyRoleDeltaBank.protected_detail``.  Sending it again as generic
        trajectory evidence duplicates the same value under another name.
        """

        plan.validate()
        return plan.protected_base.new_zeros(
            int(plan.protected_base.shape[0]),
            self.horizon,
            self.hidden,
        )

    def _transition_event_context(self, transition: ControlledTransitionState) -> Tensor:
        """Apply V120's spatial-anchor pooling to the centered transition."""

        transition.validate(hidden=self.hidden)
        batch, rows, hidden = transition.value.shape
        grid = int(self.core_config.num_cameras) * int(self.core_config.future_grid_size) ** 2
        anchors = int(self.core_config.future_anchors)
        if rows != anchors * grid:
            raise ValueError("event context requires the complete V120 transition chart")
        milestones = transition.value.reshape(batch, anchors, grid, hidden).mean(dim=2)
        boundaries = tuple(int(value) for value in self.core_config.flow_jepa_action_offsets)
        milestones = milestones[:, : len(boundaries)]
        rows_out: list[Tensor] = []
        lower = 0
        for index, upper in enumerate(boundaries):
            if upper <= lower or upper > self.horizon:
                raise ValueError("V120 event milestone boundaries are invalid")
            rows_out.append(
                milestones[:, index : index + 1].expand(-1, upper - lower, -1)
            )
            lower = upper
        if lower != self.horizon:
            raise ValueError("V120 event milestones do not cover the action horizon")
        return torch.cat(rows_out, dim=1)

    def _role_bank(self, plan: ObjectPolicyPlanDeltaBank) -> PolicyRoleDeltaBank:
        plan.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack(
                (
                    plan.factual,
                    plan.precision,
                    plan.effect,
                    plan.temporal,
                    plan.state_change,
                ),
                dim=1,
            ),
            source_names=plan.source_names,
            source_depths=(7, 7, 7, 7, 7),
            protected_detail=plan.protected_base,
        )

    def _layer_contract_canvas(
        self,
        *,
        trajectory: Tensor,
        rollout: Tensor,
        seed: V120SeedContext,
    ) -> tuple[Tensor, dict[str, slice]]:
        """Build only regions read by the position-wise V120 contract head.

        The adapter has no token mixing.  Task/stage/register/proposal rows
        cannot influence trajectory, rollout or state outputs, so omitted
        inactive regions are represented by empty slices rather than by a
        second legacy canvas implementation.
        """

        batch = int(trajectory.shape[0])
        expected_trajectory = (
            batch,
            self.horizon * self.basis,
            self.hidden,
        )
        if tuple(trajectory.shape) != expected_trajectory:
            raise ValueError("V120 layer-contract trajectory has invalid shape")
        if tuple(rollout.shape) != (
            batch,
            int(self.core_config.future_token_count),
            self.hidden,
        ):
            raise ValueError("V120 layer-contract rollout has invalid shape")
        empty = trajectory[:, :0]
        parts = (
            ("state", seed.state),
            ("state_history", seed.state_history),
            ("executed", seed.executed),
            ("proposal", empty),
            ("trajectory", trajectory),
            ("rollout", rollout),
            ("registers", empty),
        )
        slices: dict[str, slice] = {}
        offset = 0
        for name, value in parts:
            slices[name] = slice(offset, offset + int(value.shape[1]))
            offset += int(value.shape[1])
        return torch.cat([value for _, value in parts], dim=1), slices

    def _layer_contracts(
        self,
        *,
        action_query: Tensor,
        p1_fact: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        rollout: Tensor,
        seed: V120SeedContext,
    ) -> list[dict[str, Tensor]]:
        plan.validate()
        expected = tuple(plan.protected_base.shape)
        if tuple(action_query.shape) != expected or tuple(p1_fact.shape) != expected:
            raise ValueError("P1/P2 layer contracts lost [B,T,Q,H]")
        trajectories = (
            action_query + p1_fact,
            action_query + plan.protected_base,
        )
        contracts: list[dict[str, Tensor]] = []
        for head, trajectory in zip(
            self.layer_contract_heads,
            trajectories,
            strict=True,
        ):
            canvas, slices = self._layer_contract_canvas(
                trajectory=trajectory.flatten(1, 2),
                rollout=rollout,
                seed=seed,
            )
            contracts.append(head(canvas, slices))
        return contracts

    @staticmethod
    def _intent_memory(
        intent: ObjectIntentState,
        state_tokens: Tensor,
        executed_tokens: Tensor,
    ) -> dict[str, Tensor]:
        del intent
        return {
            "state": state_tokens,
            "executed": executed_tokens,
        }

    def _set_eval_intervention(self, mode: str) -> None:
        if mode == "learned":
            self.decoder.clear_execution_eval_ablation()
            return
        if self.training:
            raise ValueError("V120 execution interventions are evaluation-only")
        if mode == "no_updates":
            self.decoder.set_execution_eval_ablation(
                policy="neutral",
                capacity_gate=1.0,
            )
            return
        if mode == "full_updates":
            self.decoder.set_execution_eval_ablation(
                policy="hard",
                capacity_gate=1.0,
            )
            return
        raise ValueError(
            "bottom execution_mode must be learned/no_updates/full_updates"
        )

    def compile_evidence_view(
        self,
        *,
        action_query: Tensor,
        p1_fact: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        intent: ObjectIntentState,
        seed: V120SeedContext,
        transition: ControlledTransitionState,
    ):
        """Compile the exact V120 evidence boundary for structural audits."""

        state_tokens, state_history_tokens, executed_tokens = self._state_memory(seed)
        trajectory = self._neutral_trajectory_memory(plan)
        event_context = self._transition_event_context(transition)
        layer_contracts = self._layer_contracts(
            action_query=action_query,
            p1_fact=p1_fact,
            plan=plan,
            rollout=transition.selector,
            seed=seed,
        )
        return self.decoder.evidence_adapter(
            trajectory_tokens=trajectory,
            rollout_tokens=transition.selector,
            transition_memory=[transition.value, event_context],
            event_evidence=layer_contracts[-1]["event_logits"],
            state_memory=[state_tokens, state_history_tokens],
            layer_contracts=layer_contracts,
            intent_memory=self._intent_memory(intent, state_tokens, executed_tokens),
            visual_selector_tokens=None,
            visual_value_tokens=None,
            visual_key_bias=None,
        )

    def forward(
        self,
        *,
        noisy_action_field: Tensor,
        time: Tensor,
        action_query: Tensor,
        p1_fact: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        intent: ObjectIntentState,
        seed: V120SeedContext,
        transition: ControlledTransitionState,
        execution_mode: str = "learned",
        require_execution_supervision: bool = False,
        collect_diagnostics: bool = False,
    ) -> tuple[BottomOutput, dict[str, Tensor]]:
        expected_query = (
            int(noisy_action_field.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(action_query.shape) != expected_query:
            raise ValueError("bottom and P2/P3 must share one action query")
        plan.validate()
        if tuple(p1_fact.shape) != tuple(plan.protected_base.shape):
            raise ValueError("bottom P1 fact does not align with the P2 consequence")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        transition.validate(hidden=self.hidden)
        state_tokens, state_history_tokens, executed_tokens = self._state_memory(seed)
        role_bank = self._role_bank(plan)
        role_bank.validate(hidden_size=self.hidden, horizon=self.horizon)
        trajectory = self._neutral_trajectory_memory(plan)
        event_context = self._transition_event_context(transition)
        layer_contracts = self._layer_contracts(
            action_query=action_query,
            p1_fact=p1_fact,
            plan=plan,
            rollout=transition.selector,
            seed=seed,
        )
        event_evidence = layer_contracts[-1]["event_logits"]
        run_diagnostics = bool(
            collect_diagnostics
            or self.training
            or require_execution_supervision
            or execution_mode != "learned"
        )

        self._set_eval_intervention(execution_mode)
        try:
            raw = self.decoder(
                noisy_physical=noisy_action_field,
                time=time,
                trajectory_tokens=trajectory,
                trajectory_workspace_tokens=trajectory,
                policy_action_tokens=None,
                policy_role_delta_bank=role_bank,
                execution_terminal_probability=None,
                execution_terminal_uncertainty=None,
                rollout_tokens=transition.selector,
                transition_memory=[transition.value, event_context],
                event_evidence=event_evidence,
                state_memory=[state_tokens, state_history_tokens],
                layer_contracts=layer_contracts,
                intent_memory=self._intent_memory(
                    intent,
                    state_tokens,
                    executed_tokens,
                ),
                # P1 already owns the only high-resolution read.
                visual_selector_tokens=None,
                visual_value_tokens=None,
                visual_key_bias=None,
                collect_diagnostics=run_diagnostics,
                evidence_scale=1.0,
                noisy_scale=1.0,
            )
        finally:
            if execution_mode != "learned":
                self.decoder.clear_execution_eval_ablation()

        prefix = raw.get("evidence_mmd_it_prefix_pred_velocity")
        block_updates: tuple[Tensor, ...] = ()
        if isinstance(prefix, Tensor) and prefix.ndim == 5:
            # Defensive compatibility for older candidate charts.
            prefix = prefix.mean(dim=2)
        if isinstance(prefix, Tensor) and prefix.ndim == 4:
            block_updates = tuple(
                prefix[:, index + 1] - prefix[:, index]
                for index in range(int(prefix.shape[1]) - 1)
            )
        tensor_output = {
            name: value for name, value in raw.items() if isinstance(value, Tensor)
        }
        physical_velocity = raw["pred_velocity"]
        if execution_mode == "no_updates":
            if not isinstance(prefix, Tensor) or prefix.ndim != 4:
                raise RuntimeError(
                    "true no-update ablation requires the V120 prefix velocity chart"
                )
            # Prefix row zero is the organized/noisy-action prediction before
            # any host Evidence-MMDiT block executes.  Capacity=0 is not a
            # no-op in V120 (it removes only the owned low-rank subspace), so
            # selecting this row is the only behaviorally exact ablation.
            physical_velocity = prefix[:, 0]
        output = BottomOutput(
            physical_velocity=physical_velocity,
            event_logits=raw["event_logits"],
            motion_logits=raw["motion_logits"],
            action_query=action_query,
            block_updates=block_updates,
            evidence_tokens=transition.value,
            decoder_tensors=tensor_output,
        )
        output.validate(
            action_dim=self.physical_action_dim,
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return output, {}
        metrics = {
            name: value
            for name, value in tensor_output.items()
            if value.ndim == 0
        }
        if "evidence_mmd_it_capacity_ratio" in tensor_output:
            metrics["bottom_capacity_mean"] = tensor_output[
                "evidence_mmd_it_capacity_ratio"
            ]
        if "evidence_mmd_it_dwell_expected" in tensor_output:
            metrics["bottom_expected_dwell"] = tensor_output[
                "evidence_mmd_it_dwell_expected"
            ]
        metrics["bottom_restored_v120_decoder"] = noisy_action_field.new_ones(
            (), dtype=torch.float32
        )
        metrics["bottom_retained_transition_rows"] = noisy_action_field.new_tensor(
            float(transition.value.shape[1]), dtype=torch.float32
        )
        metrics["bottom_rollout_selector_only"] = noisy_action_field.new_ones(
            (), dtype=torch.float32
        )
        metrics["bottom_protected_consequence_value_writes"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_generic_trajectory_neutral"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_event_from_p2_layer_contract"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_terminal_layer_contract_count"] = (
            noisy_action_field.new_tensor(
                float(len(layer_contracts)), dtype=torch.float32
            )
        )
        metrics["bottom_execution_output_block_count"] = noisy_action_field.new_tensor(
            0.0 if execution_mode == "no_updates" else float(len(self.blocks)),
            dtype=torch.float32,
        )
        return output, metrics


__all__ = ["RestoredV120EvidenceBottom", "_build_decoder_config"]
