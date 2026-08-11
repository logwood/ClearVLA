"""Typed adapter from the integrated 3-2-3 top to the extracted V120 bottom.

The action solver in this module is not a reimplementation.  Its execution
blocks, evidence organizer, native noisy-action lift, ordered low-rank
capacity and candidate value reader are the mechanically extracted V120
``EvidenceLatentMMDiTActionDecoder``.  This adapter only translates the new
typed top interfaces into the exact inputs owned by that decoder.

The translation keeps the repaired ownership boundaries:

* P2's protected consequence is written once through V120's protected-detail
  reader while the historical generic trajectory ingress remains neutral;
* P3 precision/temporal/state-change are optional typed innovations;
* all 512 W transition rows reach the evidence bank without pooling;
* observation banks are never reopened below P1;
* teacher/future tensors cannot be represented by this online signature.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import ObservableHistory
from ..v120_core.profile import build_v120_policy_config
from ..v120_core.role_delta_attnres import PolicyRoleDeltaBank
from ..v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
    EvidenceViewAdapter,
)
from .action_contract import ActionQueryEncoder, BottomOutput, canonical_state_history
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


class _IntegratedEvidenceViewAdapter(EvidenceViewAdapter):
    """Keep type identity in K while making optional V streams exact-zero."""

    @staticmethod
    def _replace_range(value: Tensor, replacement: Tensor, bounds: tuple[int, int]) -> Tensor:
        start, stop = bounds
        if tuple(value[:, start:stop].shape) != tuple(replacement.shape):
            raise ValueError("integrated evidence replacement lost source alignment")
        return torch.cat((value[:, :start], replacement, value[:, stop:]), dim=1)

    def forward(self, **kwargs):
        view = super().forward(**kwargs)
        reference = kwargs["trajectory_tokens"]
        value_tokens = view.value_tokens

        # V120 deliberately supplies an all-zero generic trajectory source in
        # the object mainline.  Retain its source-type token as selector/null
        # geometry, but do not let the memory bank's learned type embedding
        # manufacture a value from that zero source.
        trajectory_value = reference.new_zeros(reference.shape)
        value_tokens = self._replace_range(
            value_tokens,
            trajectory_value,
            view.ranges["trajectory"],
        )

        # ``rollout_tokens`` are the 512 source-resolved transition selectors.
        # V120's generic adapter normally projects every rollout row into both
        # K and V.  In the integrated graph that would let the selector bypass
        # the centered ``transition.value`` and become a second free W/current
        # carrier.  Keep all selector geometry, but make its value lane an
        # algebraic null; the following named transition range is the sole
        # controlled transition writer.
        rollout_value = value_tokens[:, slice(*view.ranges["rollout"])].new_zeros(
            value_tokens[:, slice(*view.ranges["rollout"])].shape
        )
        value_tokens = self._replace_range(
            value_tokens,
            rollout_value,
            view.ranges["rollout"],
        )

        transition = self._cat_memory(
            kwargs["transition_memory"],
            name="transition",
            hidden=self.hidden_size,
        )
        if transition is None:
            raise RuntimeError("integrated bottom requires controlled transition values")
        transition_value = self.source_proj["transition"](
            transition.to(device=reference.device, dtype=reference.dtype)
        )
        # The bank's source type embedding belongs to selector geometry.  Its
        # value-side copy was the non-zero default that survived effect-zero.
        transition_value = self.bank.source_norm(transition_value)
        value_tokens = self._replace_range(
            value_tokens,
            transition_value,
            view.ranges["transition"],
        )

        event = kwargs["event_evidence"]
        event_value = self.event_proj(
            event.to(device=reference.device, dtype=reference.dtype)
        )
        event_value = self.bank.source_norm(event_value)
        value_tokens = self._replace_range(
            value_tokens,
            event_value,
            view.ranges["event"],
        )
        return replace(view, value_tokens=value_tokens)


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

        # This query is used only by P2/P3.  The restored decoder owns its own
        # native physical-action lift, exactly as V120 did.
        self.query_encoder = ActionQueryEncoder(
            action_dim=self.physical_action_dim,
            hidden=self.hidden,
            horizon=self.horizon,
            basis=self.basis,
        )
        self.state_projection = nn.Linear(dims.state_dim, self.hidden, bias=False)
        self.executed_projection = nn.Linear(dims.action_dim, self.hidden, bias=False)
        # V120 obtained this source from the controlled rollout contract.  The
        # independent graph restores that same centered-transition boundary;
        # the final event prediction remains owned by the V120 decoder.
        self.event_evidence = nn.Linear(self.hidden, 3, bias=False)
        self.decoder = EvidenceLatentMMDiTActionDecoder(self.core_config)
        integrated_adapter = _IntegratedEvidenceViewAdapter(self.core_config)
        integrated_adapter.load_state_dict(self.decoder.evidence_adapter.state_dict())
        self.decoder.evidence_adapter = integrated_adapter
        # The object-mainline bottom intentionally exposes only current state
        # and the last executed action to the generic intent compiler.  Freeze
        # extracted projections for the disallowed aliases instead of leaving
        # dead trainable tensors in optimizer/checkpoint ownership.
        for source_name in ("task", "state_history", "proposal", "visual"):
            self.decoder.evidence_adapter.intent_proj[source_name].requires_grad_(False)
        # In object-mainline V120 the generic trajectory source is an explicit
        # zero/null alternative.  Make that semantic structural: its selector
        # comes only from the shared source-type embedding, and its value is
        # replaced by exact zero in ``_IntegratedEvidenceViewAdapter``.  A
        # trainable affine projection here is otherwise a dead parameter (or,
        # through bias, an action-independent learned shortcut).
        trajectory_projection = self.decoder.evidence_adapter.source_proj["trajectory"]
        for module in trajectory_projection.modules():
            if isinstance(module, (nn.LayerNorm, nn.Linear)) and module.bias is not None:
                nn.init.zeros_(module.bias)
        trajectory_projection.requires_grad_(False)
        # V120's shared selector/value projections carried affine biases.  A
        # zero controlled delta could therefore reappear as a non-zero value
        # after the evidence adapter.  Keep the mature projection weights but
        # make the two optional value sources algebraically zero-preserving.
        for source_name in ("transition",):
            source_projection = self.decoder.evidence_adapter.source_proj[source_name]
            if not isinstance(source_projection, nn.Sequential):
                raise TypeError("V120 evidence source projection must be sequential")
            source_norm = source_projection[0]
            projection = source_projection[-1]
            if not isinstance(source_norm, nn.LayerNorm):
                raise TypeError("V120 evidence source normalization changed unexpectedly")
            if source_norm.bias is not None:
                nn.init.zeros_(source_norm.bias)
                source_norm.bias.requires_grad_(False)
            if not isinstance(projection, nn.Linear):
                raise TypeError("V120 evidence source projection changed unexpectedly")
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
                projection.bias.requires_grad_(False)
        event_source = self.decoder.evidence_adapter.event_proj
        if not isinstance(event_source, nn.Sequential):
            raise TypeError("V120 event projection must be sequential")
        event_norm = event_source[0]
        event_projection = event_source[-1]
        if not isinstance(event_norm, nn.LayerNorm):
            raise TypeError("V120 event evidence normalization changed unexpectedly")
        if event_norm.bias is not None:
            nn.init.zeros_(event_norm.bias)
            event_norm.bias.requires_grad_(False)
        if not isinstance(event_projection, nn.Linear):
            raise TypeError("V120 event evidence projection changed unexpectedly")
        if event_projection.bias is not None:
            nn.init.zeros_(event_projection.bias)
            event_projection.bias.requires_grad_(False)

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

    def set_training_step(self, global_step: int) -> float:
        return self.decoder.set_execution_training_step(global_step)

    def _state_memory(
        self,
        history: ObservableHistory,
    ) -> tuple[Tensor, Tensor]:
        # The V120 object-mainline boundary intentionally exposed only the
        # current state and final executed action to the generic bottom intent
        # compiler.  Full ordered history is already owned by S, P1, proposal
        # and controlled transition; replaying it here is a direct bypass.
        state = self.state_projection(canonical_state_history(history)[:, -1:])
        executed = self.executed_projection(history.executed_action_history[:, -1:])
        return state, executed

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
                (plan.precision, plan.temporal, plan.state_change),
                dim=1,
            ),
            source_names=plan.source_names,
            source_depths=(7, 7, 7),
            protected_detail=plan.protected_base,
        )

    @staticmethod
    def _layer_contracts(
        plan: ObjectPolicyPlanDeltaBank,
        state_tokens: Tensor,
        executed_tokens: Tensor,
    ) -> list[dict[str, Tensor]]:
        batch, horizon, basis, hidden = plan.protected_base.shape
        protected = plan.protected_base.reshape(batch, horizon * basis, hidden)
        innovations = torch.cat(
            (plan.precision, plan.temporal, plan.state_change),
            dim=2,
        ).reshape(batch, horizon * basis * 3, hidden)
        # V120 exposed the two generic terminal policy records beside the P3
        # typed bank.  Here they remain selector geometry: EvidenceViewAdapter
        # compiles their values from the clean intent memory below.
        return [
            {
                "rollout_tokens": protected,
                "state_history_tokens": state_tokens,
            },
            {
                "rollout_tokens": innovations,
                "state_tokens": executed_tokens,
            },
        ]

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
        plan: ObjectPolicyPlanDeltaBank,
        intent: ObjectIntentState,
        history: ObservableHistory,
        transition: ControlledTransitionState,
    ):
        """Compile the exact V120 evidence boundary for structural audits."""

        state_tokens, executed_tokens = self._state_memory(history)
        trajectory = self._neutral_trajectory_memory(plan)
        return self.decoder.evidence_adapter(
            trajectory_tokens=trajectory,
            rollout_tokens=transition.selector,
            transition_memory=[transition.value],
            event_evidence=self.event_evidence(
                self._transition_event_context(transition)
            ),
            state_memory=[state_tokens, executed_tokens],
            layer_contracts=self._layer_contracts(plan, state_tokens, executed_tokens),
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
        plan: ObjectPolicyPlanDeltaBank,
        intent: ObjectIntentState,
        history: ObservableHistory,
        transition: ControlledTransitionState,
        execution_mode: str = "learned",
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
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        transition.validate(hidden=self.hidden)
        state_tokens, executed_tokens = self._state_memory(history)
        role_bank = self._role_bank(plan)
        role_bank.validate(hidden_size=self.hidden, horizon=self.horizon)
        trajectory = self._neutral_trajectory_memory(plan)
        event_evidence = self.event_evidence(
            self._transition_event_context(transition)
        )
        run_diagnostics = bool(
            collect_diagnostics or self.training or execution_mode != "learned"
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
                transition_memory=[transition.value],
                event_evidence=event_evidence,
                state_memory=[state_tokens, executed_tokens],
                layer_contracts=self._layer_contracts(
                    plan,
                    state_tokens,
                    executed_tokens,
                ),
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
        metrics["bottom_event_from_centered_transition"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_execution_output_block_count"] = noisy_action_field.new_tensor(
            0.0 if execution_mode == "no_updates" else float(len(self.blocks)),
            dtype=torch.float32,
        )
        return output, metrics


__all__ = ["RestoredV120EvidenceBottom", "_build_decoder_config"]
