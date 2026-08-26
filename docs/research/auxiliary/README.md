# Schema25 replay workspace memory

This directory is the self-contained planning memory copied into the isolated
Schema25 replay worktree.

| File | Responsibility |
|---|---|
| ARCHITECTURE_REPLAY_PLAN.md | Sole authority for replay base, unit dispositions and R1 bundle |
| ARCHITECTURE_REPLAY_SOURCE_UNITS.md | Cross-version source boundaries, dependencies, conflicts and rejected mechanics |
| SCHEMA25_R1_IMPLEMENTATION_PROTOCOL.md | Reconciled implementation order, locked mechanisms and release-blocking tests |
| SCHEMA25_R0_BASELINE_FINGERPRINT.md | Untouched Schema25 source, manifest, parameter and test fingerprint |
| SCHEMA25_BASE_VERSION_CARD.md | Closed source/behavior audit of the fixed base |
| ARCHITECTURE_REPLAY_LEDGER.md | Historical donor evidence; not implementation authority |
| ACTIVE_MAINLINE_HANDOFF.md | Frozen Schema39 diagnostic context; not the active replay graph |

The checked-out executable source remains Schema25 until an R1 implementation
slice changes it. Historical documents and the user-supplied external protocol
do not override the active source.
