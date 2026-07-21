# ClearVLA Log Metric Catalog

## Contents

- Evidence hierarchy
- Compact display naming
- Objective and optimization
- Prediction and validation
- Structure and control
- Gradients and interventions
- Coverage and comparison

## Evidence hierarchy

1. `loss_ledger_residual` verifies whether registered weighted contributions reconstruct the scalar sent to backward.
2. `loss_group_*` gives objective budget by action, rollout, execution, latent, and layer groups.
3. `loss_contrib_*` gives weighted component contributions.
4. Raw losses describe task error in their native scale; they do not describe optimization dominance without a weight.
5. Audit gauges such as execution cost, null-space geometry, or sampling probes may be detached and absent from backward.

For pre-ledger logs, the audit utility reconstructs known terms from the serialized trainer config and labels the result `estimated-known-terms`.

## Compact display naming

The V94 console uses medium-length display labels while the JSONL retains the
canonical source keys. The audit utility accepts both the old abbreviated
labels and the current display labels.

- Keep familiar statistical abbreviations such as `rmse`, `f1`, `lr`, and
  `grad`; avoid opaque implementation abbreviations such as `pfn`, `stdr`, and
  `dnratio`.
- Use one to three semantic words: `native_flow`, `rollout_std_ratio`,
  `effective_basis_mass`, and `value_top1_acc`.
- Distinguish raw losses, weighted contributions, and diagnostics:
  `event_loss`, `top_contrib`, and `grip_event_ratio` are different kinds of
  quantities.
- Distinguish training flow errors from sampled validation errors:
  `flow_first8` versus `first8_rmse`.
- `capacity_gate_mass` and `effective_basis_mass` are continuous transparency
  measures, not a physically pruned integer rank or measured compute saving.
  The run header's decoder depth is a separate configured architecture axis.
- `cost_proxy` remains a detached execution-budget proxy, not measured compute
  cost and not an objective contribution.

## Objective and optimization

| Family | Canonical metrics | Read together |
|---|---|---|
| Main action | `physical_flow`, `physical_flow_native`, `arm_fm_per_dim`, `gripper_fm_field` | native/uniform anchors, arm/gripper balance |
| Decode closure | `decoded_action`, `physical_delta_consistency`, `transition_l1` | sampled validation RMSE and event timing |
| Event/motion | `event`, `motion`, event/motion P/R/F1 | predicted and target positive counts |
| Rollout | `rollout_dynamics`, `rollout_contrast`, `rollout_variance`, `rollout_norm`, `rollout_milestone_delta_match` | std ratio, delta norm ratio, tail behavior |
| Layer ownership | `layer_contract`, `loss_contrib_layer_contract`, per-layer contracts | adapter/consequence gradients and schedule |
| Execution value | value loss, target/predicted spread, correlation, pairwise/decision accuracy, common-mode ratio | candidate coverage and selection entropy |

Known compatibility aliases require care: historical `future_latent` and `action_effect` names do not necessarily denote independent objectives. The utility detects numerically duplicate rollout series and checks whether both effective weights are active.

## Prediction and validation

- `full_rmse` is not enough. Always include `first_rmse`, `first8_rmse`, `tail_rmse`, and `tail_first_ratio`.
- Separate `arm_full_rmse` from `gripper_full_rmse`; a lower aggregate can hide semantic-channel regression.
- Read gripper precision/recall/F1 with `gripper_event_ratio` and predicted/target event counts.
- Read event-head metrics separately from decoded gripper events. Agreement is a closure property, not guaranteed by head accuracy.
- Report `event_head_minus_decoded_gripper_f1`; the decoded gripper trajectory
  remains the deployment behavior, while the event head is auxiliary evidence.
- `proposal_utility_mse_gain > 0` indicates improvement over the no-proposal ablation. Always report ablation coverage.
- Training pflow and sampled validation RMSE use different procedures; a large training decrease with flat validation is a real mismatch signal, not a numerical contradiction.

## Structure and control

- `execution_progress` defines the phase. Capacity/depth remaining full during warmup is expected.
- Read `capacity_gate_mass` with `effective_basis_mass`, removed fraction, and
  configured basis count. Do not call this hardware rank reduction.
- Read route and dwell as soft/hard pairs and include their gaps.
- Read terminal prior, soft terminal probability, hard terminal fraction, and
  terminal target/predicted margins together. A prior below one is a policy
  bias, not an execution-cost loss.
- Soft/hard terminal fractions are terminal occupancy after each decision,
  averaged over the execution clock; they are not raw exit-event counts.
- Execution cost is audit-only on the native Evidence path unless source inspection proves otherwise.
- High value common-mode with near-chance decision accuracy indicates weak candidate discrimination even when pairwise accuracy is above chance.
- Selection entropy needs the legal candidate count for an absolute interpretation; max probability supplies a more portable companion gauge.

## Gradients and interventions

- Compare gradients by module and phase; do not compare a scalar parameter norm and a large-module norm as if they had identical dimensionality.
- Zero capacity gradients during execution warmup are expected. Persistent zeros after progress opens are suspicious.
- Layer adapter and consequence gradients establish whether ownership supervision reaches the intended interfaces.
- `z_zero` and `z_shuffle` are intervention deltas, not loss terms. Zero values are interpretable only when the active-path probe ran and coverage is nonzero.

## Coverage and comparison

- Full action validation may cover every batch while sampling gauges and proposal/execution ablations cover only a subset.
- Use `eval_sampling_diagnostic_coverage`, `eval_proposal_ablation_coverage`, and
  `eval_execution_ablation_coverage` in every conclusion about probes or ablations.
- Compare primary soft execution against matched-noise hard, neutral,
  full-capacity, and three-basis-reduction RMSE before crediting the controller.
- Cross-run comparisons require matching data split and action-normalizer fingerprint for native-coordinate anchors.
- Treat partial Stage1 loads (`missing`/`skipped` keys) as structural cold-start evidence.
