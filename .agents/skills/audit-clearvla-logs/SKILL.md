---
name: audit-clearvla-logs
description: Diagnose ClearVLA training logs, validation regressions and cross-run differences against each run's actual source and configuration.
---

# Audit ClearVLA Logs

Deliver the evidence needed for the user's specific experiment decision.
A narrow status question needs a narrow check; a requested full health audit
covers the full available curve and completed validation, not just its best point.

## Start from the run

Identify its source, resolved configuration, initialization/clock, normalizers
and completed stages. Use the existing summary tool when it fits:

```bash
python -m clearvla.tools.audit_policy_logs LOG_OR_RUN_DIR [MORE_RUNS] --format json
```

Prefer a run directory with archived metrics to a copied console tail.
For independent mainline runs inspect clearvla/mainline/ and the serialized
manifest; V-number ancestry does not select the current implementation.

## Read only the relevant reference

- [Metric catalog](references/metric-catalog.md): locate the named metric,
  loss group, channel or diagnostic under investigation.
- [Source map](references/source-map.md): locate the producer, objective,
  backward or logging boundary needed to explain an observation.

These include historical variants. Match them to the run; they are not mandatory
full-document reads. Verify material claims against actual source and raw rows.

## Decision-critical distinctions

- Weighted loss contributions and ledger closure determine optimization budget;
  raw magnitude or a detached gauge does not.
- Missing gradients, exact zero, inactive placeholders and tiny finite gradients
  are different. Activation VJP does not establish parameter updates or behavior.
- CALVIN binary commands, continuous gripper fields and auxiliary event heads
  have distinct metrics. Use event counts and precision/recall/F1 when relevant.
- Training, offline validation, saved weights and closed-loop behavior are
  separate completion states. A pending-validation snapshot is not best/latest.
- Cross-run attribution needs compatible data, normalization, exposure,
  initialization, clock, objectives and runtime. Label unmatched comparisons.
- Keep numerical repeat noise, intervention sensitivity and real task
  correctness distinct. K separation or color-response amplitude alone is not
  general instruction grounding.

Use existing evidence to distinguish causes. Add logging or a probe only when
the result can change the next decision. State what is established, what remains
uncertain, and the next justified action; avoid a full checklist for every query.
