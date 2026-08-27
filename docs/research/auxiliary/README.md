# Auxiliary research memory

This directory keeps four kinds of non-contract memory without changing the
checked-out ClearVLA architecture:

- the adopted architecture replay procedure;
- frozen source/log audits and diagnostic handoffs;
- design ancestry for changes already implemented in a historical schema;
- prospective research and conditional alternatives not adopted by mainline.

The directory remains flat. Status banners and this index define ownership;
moving files into version folders would add link churn without clarifying the
architecture.

## Document index

| File | Status | Sole responsibility |
|---|---|---|
| `ARCHITECTURE_REPLAY_PLAN.md` | Adopted replay procedure | Selects Schema25 commit `6a6c1bf` as the base, records cross-version unit dispositions and selects the next coherent candidate. This is the only auxiliary file with replay execution authority. |
| `ARCHITECTURE_REPLAY_SOURCE_UNITS.md` | Working source-unit register | Groups Schema26-39 source hunks by producer-to-consumer semantic boundary, records dependencies/conflicts and defines the detailed R1 source closure. It does not independently authorize implementation or training. |
| `SCHEMA25_BASE_VERSION_CARD.md` | Completed base audit | Maps the exact Schema25 source, existing eight-epoch behavior record, forward/backward/runtime/checkpoint boundary, protected semantic invariants and retained `BASE/OPEN` debt before Schema26 review. It does not declare Schema25 architecturally final. |
| `ARCHITECTURE_REPLAY_LEDGER.md` | Frozen historical/source audit | Reconstructs V120 through the Schema39 snapshot, separates evidence classes and inventories donor material. It does not choose the base or prescribe a bundle. |
| `CONVERSATION_INSIGHT_LEDGER.md` | Incremental conversation-history extraction | Records hypotheses, observations, reversals and durable lessons in conversation order. It is historical evidence, not active architecture or replay authority. |
| `ACTIVE_MAINLINE_HANDOFF.md` | Frozen Schema39 diagnostic snapshot | Preserves source identity, log comparisons, unresolved assumptions and the former Schema39-forward audit queue. The filename is retained for link stability; it does not own current replay strategy. |
| `S_W_P2_CLOSED_LOOP_REPAIR_PLAN.md` | Implemented Schema31 ancestry | Preserves the adopted design reasoning and acceptance boundary for the historical S-W-P2 closure. |
| `SCHEMA38_ACTION_CONSUMPTION_CLOSURE_PLAN.md` | Implemented Schema38 ancestry | Preserves the design basis for full-field W/P2 consumption, dynamic-P1 precision, geometry semantic-addressing and diagnostic cleanup. It does not prove training success. |
| `SCHEMA39_DESIGN_CHECKPOINT.md` | Implemented Schema39 ancestry | Preserves Schema39's locked design scope and release process; it is not a performance claim or replay instruction. |
| `P2_FACTORIZED_INTERVAL_OBJECT_ROUTING.md` | Conditional historical proposal | Describes a P2-only factorized routing alternative. Its original Schema32 activation condition is historical and does not automatically activate it during replay. |
| `MIP_BSPINE_RESEARCH_LEDGER.md` | Prospective research | Records unimplemented MIP/B-spine research. It is outside the active replay unless the user explicitly promotes it. |

## Authority boundary

Do not treat any auxiliary document as executable architecture truth. The
checked-out graph remains owned by `../00_CURRENT_ARCHITECTURE_CONTRACT.md`, and
current release blockers remain owned by `../CURRENT_MAINLINE_ISSUES.md`.

`ARCHITECTURE_REPLAY_PLAN.md` governs how a candidate graph is reconstructed;
it does not silently replace the checked-out graph. Move a replay decision into
the active contract only when the user explicitly adopts the corresponding
implementation and it enters active scope.

Historical plans and checkpoints are evidence about why a change was made, not
instructions to restore the whole change. Resolve semantic claims against the
exact source commit and replay state. Serialized run context and supplied logs
remain behavioral evidence; they may rank a closed candidate but cannot define
an implementation unit.

## Working rule

For this large continuous model, the repair unit is a semantic source change
with a closed producer-to-consumer and backward-gradient boundary. It is not
necessarily a line-level diff and it does not require one training experiment
per unit.

Keep accepted units separated by owner, commit, diagnostics and rollback
boundary. Before implementation, record every producer, transformation,
consumer, loss, deployment/runtime call site, optimizer owner, checkpoint
field, tensor axis, zero semantic, scale, repetition and possible bypass. Scan
all donor versions before selecting the bundle; do not make historical versions
the unit of work. Judge the final coherent replay candidate with layered
boundary checks and, only when separately authorized, the limited training
budget.
