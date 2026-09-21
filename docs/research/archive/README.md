# Archived research

This directory contains high-cost historical context that is useful for
forensic attribution but unnecessary for ordinary implementation.

| Area | Contents | When to open |
|---|---|---|
| [`replay/`](replay/README.md) | Schema25 base card, replay ledger/plan, source-unit register and R1 protocol | ancestry, donor choice or an earlier repair |
| [`legacy_evidence/`](legacy_evidence/README.md) | pre-mainline studies, probe findings and reproducibility scripts | reproducing a historical observation |
| [`operational_handoff_20260919.md`](operational_handoff_20260919.md) | Frozen former handoff, including unpublished task notes | retrieving an old observation or its original scope; never for live instructions/PIDs |

Current architecture and work order remain in
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md).
Archive filenames and version labels are historical identities, not active
configuration.

The larger retired `history_design/` tree and the former local-log index were
removed from the live documentation tree during the 2026-09-04 compaction.
Both remain recoverable from Git commit `b8163cb`; raw run artifacts remain
outside Git.

The handoff snapshot was retained during the 2026-09-19 cleanup because the
working copy contained uncommitted observations that Git history did not yet
preserve. It is a one-time historical record, not another maintained handoff.
Current task scopes and dated observations live only in
[`../auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](../auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
