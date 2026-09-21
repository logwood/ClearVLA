# ClearVLA research documentation map

Use the smallest document that can answer the question.

Reviewed: 2026-09-19 UTC. This index routes questions; it does not define a
second architecture or authorize an experiment. For a frozen run, its pinned
source/config/checkpoint and serialized context take precedence over today's
workspace documentation.

| Question | Read first |
|---|---|
| What does the current default graph do? | The contract's quick graph and default/opt-in table |
| What does an initialization or runtime mode mean? | [`clearvla/mainline/README.md`](../../clearvla/mainline/README.md#lifecycle-vocabulary) |
| Which source or behavior question is still open? | The issue ledger's status guide and decision table |
| What work is proposed next? | The repair plan, subject to the applicable user request |
| What was last verified for a running task? | The handoff's dated, task-scoped observation |
| Why was an earlier choice made? | The archive or Git history |

## Evidence states

Keep these fields separate whenever reporting a feature or candidate:

- **Implemented:** the named source/config contains it; this says nothing about
  deployment selection or task benefit.
- **Structurally checked:** the named invariants passed at stated coverage.
- **Training:** not started, observed in progress, or completed for an identified
  run/checkpoint. A launch or initialization check is not a completed curve.
- **Behavior:** not evaluated, bounded result, or matched validation under a
  named protocol. A bounded success does not establish general task closure.

The contract owns accepted semantics and default selection; the issue ledger
owns evidence limits and decision status; the handoff owns dated run progress.
Use links across those owners instead of maintaining duplicate live tables.

## Active truth

1. [`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)
   — current graph, ABI, invariants, identity and release gates.
2. [`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md) — open decisions,
   source closures and their separate behavior evidence.
3. [`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md) —
   repository consolidation, refactor order and decision rules.

These three files are maintained in place. Do not create a new versioned copy
for each conversation.

The [active handoff](auxiliary/ACTIVE_MAINLINE_HANDOFF.md) records task scope,
last verification time and operational locations. It is not a permanent grant
or prohibition for another task. Revalidate a process before acting on a PID.

## Supporting references

[`auxiliary/`](auxiliary/README.md) contains the rolling handoff, the current
RDT adapter contract and optional research notes. Read one only when the active
contract or the task requires its detail; none adds a second architecture
authority.

The auxiliary
[`CLEARVLA_WAM_LITERATURE_INTEGRATION.md`](auxiliary/CLEARVLA_WAM_LITERATURE_INTEGRATION.md)
is an exploratory literature-to-source map. It records falsifiable hypotheses
but does not authorize an architecture or experiment change.

## Historical evidence

- [`archive/replay/`](archive/replay/README.md): Schema25 replay provenance,
  source units and donor decisions.
- [`archive/legacy_evidence/`](archive/legacy_evidence/README.md): older
  research notes and reproducibility tools.

The retired `history_design/` tree and the old local-log index are recoverable
from Git commit `b8163cb`. Raw run evidence remains local under the ignored
`new_logs/` directory; only decision statistics, identities and reproducible
commands belong in tracked documentation.

Historical material cannot override active source or a run's serialized
context. The compact indexes under `auxiliary/` are historical even though
they remain reachable for retrieval; they are not part of the current reading
order. Checkpoints, tensor caches, raw JSONL, complete probe dumps and
generated binaries do not belong in this documentation tree.

[`WIP_SEMANTIC_MAP.md`](../development/WIP_SEMANTIC_MAP.md) is a 2026-09-15
classification snapshot. Its former next-step list is superseded by the repair
plan; it is not an additional current work queue.
