# ClearVLA workspace memory

Before changing the V96+ top representation, Flow-DINO/JEPA, role-block
hierarchy, language/history conditioning, long-horizon phase handling, or
top-to-bottom evidence path, read:

`docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`

Start with its `Agent quick contract` and non-negotiable invariants. Read the
long historical audit ledger only when the task concerns ancestry, an old log,
or the reason for a previous repair; do not rebuild the current graph from the
ledger.

That file is the compact source of truth. Update it in place when an
architectural decision changes; do not create a new versioned copy for every
conversation.

Prospective research that has not entered the active mainline scope belongs in
`docs/research/auxiliary/`. This includes literature notes, unapproved MIP/WAM
designs, sampler schedules, experiment matrices, and implementation sketches.
Do not put those notes in `CURRENT_MAINLINE_ISSUES.md` or the current
architecture contract. Promote an auxiliary decision into the mainline
documents only when the user explicitly adopts it and the corresponding schema
implementation is entering the active scope.

Read the two detailed design documents linked from the contract only when the
task needs their implementation details. Do not infer current behavior from an
old experiment name: verify the active script, serialized run context, current
source, and supplied log.

Keep repository memory lightweight. Do not copy checkpoints, tensor caches,
raw attached logs, full probe dumps, contact sheets, or generated binary
artifacts into the architecture-memory documents. Record only decision-making
statistics, source references, contracts, and reproducible probe commands.
