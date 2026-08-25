"""Typed adapter from the integrated 3-2-3 top to the extracted V120 bottom.

The action solver in this module is not a reimplementation.  Its execution
blocks, evidence organizer, native noisy-action lift, ordered low-rank
capacity and candidate value reader are the mechanically extracted V120
``EvidenceLatentMMDiTActionDecoder``.  This adapter only translates the new
typed top interfaces into the exact inputs owned by that decoder.

The translation keeps the repaired ownership boundaries:

* P2's protected consequence is written once through V120's protected-detail
  reader while the historical generic trajectory ingress remains neutral;
* six P3 lanes remain separate optional innovations until their lane-local
  action-basis readers;
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
from ..v120_core.layer_contracts import LayerContractAdapterHeads
from ..v120_core.primitives import TimeEmbedding
from ..v120_core.profile import build_v120_policy_config
from ..v120_core.role_delta_attnres import (
    AffineVarianceFlooredCenteredNorm,
    PolicyRoleDeltaBank,
)
from ..v120_core.time_domain_mmdit import (
    EvidenceLatentMMDiTActionDecoder,
)
from ..v120_core.trunk_primitives import TemporalDynamicsBoundDiTBlock
from .action_contract import ActionQueryEncoder, BottomOutput, V120SeedContext
from .compiler import ObjectPolicyPlanDeltaBank
from .types import CompletedP1PolicyState, ControlledTransitionState, ObjectIntentState


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
        # V120 P1 was not only the static high-resolution reader.  The reader
        # first wrote protected current detail, then the first policy DiT
        # block updated the live noisy-action trajectory at every ODE step.
        # Keep the expensive detail read cached, but retain that dynamic block
        # and its exact time/content modulation here beside the shared seed.
        self.p1_time = TimeEmbedding(self.hidden)
        self.p1_content_mod = nn.Sequential(
            AffineVarianceFlooredCenteredNorm(
                2 * self.hidden,
                float(self.core_config.flow_jepa_routing_norm_floor),
                affine_maximum=4.0,
            ),
            nn.Linear(2 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        nn.init.normal_(self.p1_content_mod[-1].weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.p1_content_mod[-1].bias)
        self.p1_content_mod_scale = nn.Parameter(torch.tensor(0.10))
        self.p1_policy_block = TemporalDynamicsBoundDiTBlock(
            self.core_config,
            role="policy",
        )
        if (
            self.p1_policy_block.visual_cross_enabled
            or not self.p1_policy_block.policy_explicit_handoff_only
            or not self.p1_policy_block.grounded_policy_explicit_only
        ):
            raise ValueError(
                "the recovered P1 block must use the strict explicit object handoff"
            )
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
        role: Tensor | None = None,
    ) -> tuple[Tensor, V120SeedContext]:
        """Build the one shared V120 seed consumed by top and bottom."""

        return self.query_encoder.forward_with_context(
            noisy_action_field,
            time,
            history,
            executed_memory=executed_memory,
            action_history_keep=action_history_keep,
            role=role,
        )

    def sample_role_table(self, reference: Tensor) -> Tensor:
        return self.query_encoder.sample_role_table(reference)

    def grounding_canvas(
        self,
        *,
        state: Tensor,
        rollout_init: Tensor,
        role: Tensor,
    ) -> tuple[Tensor, dict[str, slice]]:
        return self.query_encoder.grounding_canvas(
            state=state,
            rollout_init=rollout_init,
            role=role,
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

    def complete_p1_fact(
        self,
        *,
        action_query: Tensor,
        protected_detail: Tensor,
        time: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[CompletedP1PolicyState, dict[str, Tensor]]:
        """Apply V120's dynamic P1 write without relabeling it as a fact.

        In the strict object path the policy block's trajectory queries may
        attend only trajectory keys; visual/context/future rows are masked and
        policy writes are trajectory-only.  Evaluating that active subgraph on
        the compact ``[T,Q]`` canvas is therefore mathematically identical to
        retaining the inactive rows, without reopening RGB/DINO or duplicating
        their memory at every ODE step.
        """

        expected = (
            int(action_query.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(action_query.shape) != expected or tuple(protected_detail.shape) != expected:
            raise ValueError("dynamic P1 inputs must align as [B,T,Q,H]")
        if tuple(time.shape) != (expected[0],):
            raise ValueError("dynamic P1 time must be [B]")
        trajectory = action_query + protected_detail
        canvas = trajectory.flatten(1, 2)
        rows = int(canvas.shape[1])
        empty_before = slice(0, 0)
        empty_after = slice(rows, rows)
        slices = {
            "task": empty_before,
            "state": empty_before,
            "state_history": empty_before,
            "executed": empty_before,
            "proposal": empty_before,
            "trajectory": slice(0, rows),
            "stage": empty_after,
            "rollout": empty_after,
            "registers": empty_after,
        }
        trajectory_summary = canvas.mean(dim=1)
        content_delta = self.p1_content_mod(
            torch.cat((trajectory_summary, trajectory_summary), dim=-1)
        ) * self.p1_content_mod_scale.to(
            device=canvas.device,
            dtype=canvas.dtype,
        )
        time_input = time.to(device=canvas.device, dtype=canvas.dtype)
        mod_embed = self.p1_time(time_input) + content_delta
        updated, block_metrics = self.p1_policy_block(
            canvas,
            canvas[:, :0],
            mod_embed,
            slices,
            collect_diagnostics=collect_diagnostics,
        )
        dynamic_delta = (updated - canvas).reshape(expected)
        state = CompletedP1PolicyState(
            factual_base=protected_detail,
            policy_query_residual=dynamic_delta,
        )
        state.validate(
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return state, {}
        policy_updated_trajectory = protected_detail + dynamic_delta
        metrics = {
            "p1_protected_detail_rms": protected_detail.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_dynamic_delta_rms": dynamic_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            # This is the dynamic P1 block's updated policy trajectory, not a
            # completed factual value.  Keep the ownership explicit so log
            # tooling cannot mistake policy bandwidth for protected evidence.
            "p1_policy_updated_trajectory_rms": policy_updated_trajectory.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_factual_base_rms": state.factual_base.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_policy_query_residual_rms": state.policy_query_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_p2_query_rms": state.p2_dock(action_query).combined().detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_policy_content_mod_rms": content_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        # The V120 block already computes its internal numerical contract.
        # Forwarding only the final written RMS hid the exact pre-NaN growth
        # point in Schema35, so expose the complete existing boundary without
        # adding another diagnostic network or changing the forward value.
        for source, target in (
            (
                "modulation_contract_enabled",
                "p1_policy_modulation_contract_enabled",
            ),
            ("gate_self", "p1_policy_self_gate"),
            ("gate_ffn", "p1_policy_ffn_gate"),
            ("residual_self_raw_rms", "p1_policy_self_raw_rms"),
            ("residual_self_proposed_rms", "p1_policy_self_proposed_rms"),
            ("residual_self_bounded_rms", "p1_policy_self_bounded_rms"),
            ("residual_self_written_rms", "p1_policy_self_written_rms"),
            ("residual_self_compression", "p1_policy_self_compression"),
            ("residual_ffn_raw_rms", "p1_policy_ffn_raw_rms"),
            ("residual_ffn_proposed_rms", "p1_policy_ffn_proposed_rms"),
            ("residual_ffn_bounded_rms", "p1_policy_ffn_bounded_rms"),
            ("residual_ffn_written_rms", "p1_policy_ffn_written_rms"),
            ("residual_ffn_compression", "p1_policy_ffn_compression"),
            ("residual_raw_rms", "p1_policy_residual_raw_rms"),
            ("residual_proposed_rms", "p1_policy_residual_proposed_rms"),
            ("residual_bounded_rms", "p1_policy_residual_bounded_rms"),
            ("residual_written_rms", "p1_policy_residual_written_rms"),
            ("residual_compression", "p1_policy_residual_compression"),
            (
                "normalization_denominator_min",
                "p1_policy_normalization_denominator_min",
            ),
            ("normalization_gain_max", "p1_policy_normalization_gain_max"),
            ("modulation_shift_max_abs", "p1_policy_modulation_shift_max_abs"),
            ("modulation_scale_max_abs", "p1_policy_modulation_scale_max_abs"),
            (
                "modulation_shift_raw_max_abs",
                "p1_policy_modulation_shift_raw_max_abs",
            ),
            (
                "modulation_scale_raw_max_abs",
                "p1_policy_modulation_scale_raw_max_abs",
            ),
            ("self_qk_rms", "p1_policy_self_qk_rms"),
            ("ffn_input_rms", "p1_policy_ffn_input_rms"),
        ):
            value = block_metrics.get(source)
            if value is not None:
                metrics[target] = value
        return state, metrics

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
        # Keep the compiler as the single owner of the active lane schema.
        # In particular, do not reconstruct semantic/geometry lanes here or
        # silently merge them before the bottom selector sees them.
        return plan.as_policy_role_bank(source_depth=7)

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
        if mode == "hard":
            self.decoder.set_execution_eval_ablation(
                policy="hard",
                capacity_gate=None,
            )
            return
        if mode == "neutral":
            self.decoder.set_execution_eval_ablation(
                policy="neutral",
                capacity_gate=1.0,
            )
            return
        if mode == "full_capacity":
            self.decoder.set_execution_eval_ablation(
                policy="soft",
                capacity_gate=1.0,
            )
            return
        if mode == "three_basis_reduction":
            rank = max(
                int(self.core_config.latent_cvae_mmdit_operator_rank),
                1,
            )
            self.decoder.set_execution_eval_ablation(
                policy="soft",
                capacity_gate=max(float(rank - 3), 1.0) / float(rank),
            )
            return
        raise ValueError(
            "bottom execution_mode must be learned/no_updates/hard/neutral/"
            "full_capacity/three_basis_reduction"
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
