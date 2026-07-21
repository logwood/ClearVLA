---
name: audit-clearvla-logs
description: Audit and compare ClearVLA training logs against the current workspace source. Use for attached nohup/text logs, run directories, v39_policy_epochs.jsonl files, loss composition questions, gradient or controller health checks, validation regressions, cross-version experiment comparisons, and decisions about whether a run should continue or the implementation should change.
---

# Audit ClearVLA Logs

Build a source-grounded experiment diagnosis from raw logs. Treat every run as a new artifact; do not inherit conclusions from an older version unless the current manifest and implementation support them.

## Workflow

1. Resolve every supplied log or run directory and identify its actual version header, context manifest, initialization counts, batch range, and completed epoch records.
2. From the repository root, run:

   ```bash
   python -m clearvla.tools.audit_policy_logs LOG_OR_RUN_DIR [MORE_LOGS]
   ```

   Pass a run directory when available so the utility merges `nohup.log` with `v39_policy_epochs.jsonl`. Use `--format json` when downstream calculations need the complete normalized summary.
3. Read [metric-catalog.md](references/metric-catalog.md) for metric semantics and comparison constraints. Read [source-map.md](references/source-map.md) whenever the request asks why a behavior occurs, whether a loss is active, or whether code should change.
4. Inspect the utility findings, then verify every material claim against the relevant raw rows and current source. Rules are triage signals, not proof of causality.
5. Report the smallest decision-complete set of evidence:
   - actual run contract and initialization state;
   - exact loss-group contributions when a ledger exists, otherwise clearly labeled estimates;
   - training trajectory versus epoch validation;
   - horizon, arm, gripper/event, rollout, proposal, conditioning, controller/capacity, and gradient health;
   - diagnostic coverage and missing evidence;
   - causal interpretation, competing explanations, and a continue/stop/change recommendation.

## Evidence Rules

- Prefer the serialized run context and header over shell-script defaults.
- Prefer `loss_contrib_*`, `loss_group_*`, and `loss_ledger_residual` over raw loss magnitudes. Never call a large raw auxiliary loss dominant without multiplying its effective weight.
- Distinguish an audit-only metric from an objective that enters backward.
- Treat an active-path zero gradient as evidence. Ignore zero placeholders from inactive branches.
- Do not use event accuracy alone; pair it with precision, recall, F1, predicted/target event counts, and decoded gripper event ratio.
- Do not infer generalization from training pflow. Compare validation RMSE by horizon and semantic channel.
- Interpret capacity/depth only after execution progress leaves warmup. Interpret z interventions only when diagnostic coverage is present.
- Compare `physical_flow_native_uniform` across runs only when the action-normalizer fingerprint matches.
- Separate implementation facts, log observations, inferences, and experiment proposals in the final answer.

## Comparing Runs

Use one command with multiple inputs. Confirm data split, normalizer fingerprint, decoder, training stage, checkpoint migration, batch size, and effective objective weights before attributing differences to architecture. If those contracts differ, describe the comparison as directional rather than controlled.

## When Evidence Is Missing

State exactly which conclusion cannot be made. Prefer the archived JSONL or run snapshot over adding speculative logging. Recommend a logging change only when the missing observation would alter a concrete experimental decision.
