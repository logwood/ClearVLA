# ClearVLA Loss and Logging Source Map

## Contents

- Objective construction
- Backward and aggregation
- Validation
- Logging
- Static audit watchlist

## Objective construction

- `clearvla/experiments/observed_state_lab/policy_runtime_v36_3.py`
  - `flow_losses`: base physical flow, proposal, event, motion, decoded-action, delta, and gripper closure objectives.
  - `gripper_transition_metrics`, `event_head_metrics`, `motion_head_metrics`: validation semantics.
- `clearvla/experiments/observed_state_lab/policy_runtime_v39.py`
  - `rollout_dynamics_loss`, `rollout_delta_loss`, `milestone_delta_match_loss`: rollout targets and compatibility overlap.
  - `flow_losses`: execution value supervision, rollout/latent/controller objectives, and audit-only execution cost.
  - `midcut_contract_losses`, `layer_contract_losses`: stage/layer auxiliary objectives.
  - `_attach_v94_loss_ledger`: exact Evidence objective contribution ledger.

## Backward and aggregation

- In `train_v39_policy`, inspect the construction of `total_loss`, the auxiliary scale, assignment back to `losses["loss"]`, ledger attachment, `.backward()`, gradient diagnostics, clipping, and optimizer step in that order.
- `_sync_loss_row` supplies batch console values; `_accumulate_metric_tensors` and `_finalize_metric_tensors` supply epoch means.
- The epoch JSONL is the archival record. Compact console lines are a decision-oriented projection.

## Validation

- `evaluate_v39_policy` uses deploy-style sampling for action metrics.
- Sampling interventions and proposal ablations have independent budgets and coverage.
- Layer-contract teacher-forced evaluation is separately labeled and must not be compared directly with sampled action RMSE.

## Logging

- `_evidence_serial_log_line`: compact V94 batch loss/execution/gradient lines.
- `_evidence_epoch_log_line`: compact epoch, validation, and intervention lines.
- `_filter_inactive_evidence_epoch_metrics`: removes inactive zero placeholders from Evidence epoch records.
- The historical fallback `[v39-layer]` formatter is broad and placeholder-heavy; use it as raw evidence, not as a recommended schema.
- `clearvla/tools/audit_policy_logs.py`: normalized parser, objective budget, trends, rule findings, and run comparison.

## Static audit watchlist

Re-check these against current source rather than assuming they remain true:

1. `rollout_delta_loss` and `milestone_delta_match_loss` can become the same formula when `milestone_step_delta_pred` is present. Enabling both weights duplicates supervision.
2. Compatibility metrics named `future_latent` and `action_effect` are aliases; source determines which tensor their weights actually optimize.
3. The exact ledger is currently attached to the Evidence policy path, while other decoders rely on estimated reconstruction.
4. The layer ledger exposes the total weighted layer contribution but not every internal layer sub-objective as a separately named exact contribution.
5. Event-head accuracy is highly vulnerable to hold-class imbalance; decoded event counts are the deployment-facing closure check.
6. Compact console output is intentionally incomplete; use `v39_policy_epochs.jsonl` for exhaustive epoch fields.
7. Raw metric equality, constant norms, or zero lateral paths may be architectural invariants. Confirm source semantics before calling them collapse.
