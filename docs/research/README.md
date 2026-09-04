# ClearVLA research documentation map

Use the smallest document that can answer the question.

## Active truth

1. [`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md)
   — current graph, ABI, invariants, identity and release gates.
2. [`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md) — unresolved
   questions that can change the next source unit.
3. [`CURRENT_MAINLINE_REPAIR_PLAN.md`](CURRENT_MAINLINE_REPAIR_PLAN.md) —
   repository consolidation, refactor order and decision rules.

These three files are maintained in place. Do not create a new versioned copy
for each conversation.

## Current detailed references

[`auxiliary/`](auxiliary/README.md) contains the rolling handoff, the current
RDT adapter contract and two compact historical indexes. Read one only when
the active contract or the task requires its detail.

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
context. Checkpoints, tensor caches, raw JSONL, complete probe dumps and
generated binaries do not belong in this documentation tree.
