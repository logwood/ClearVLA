# Auxiliary research memory

This directory preserves prospective research without changing the active
ClearVLA mainline contract or problem ledger.

Use it for:

- literature/source-code reviews;
- unapproved MIP/WAM, schedule, solver, or training proposals;
- experiment matrices and implementation sketches;
- unresolved alternatives that have not entered an active schema.

Focused adopted and pending designs:

- `SCHEMA38_ACTION_CONSUMPTION_CLOSURE_PLAN.md`: the source-backed candidate
  plan derived from the completed Schema37 run. It was adopted for Schema38
  and is retained as design ancestry: full-field W/P2 consumption,
  zero-preserving dynamic-P1 precision, geometry semantic-addressing and
  deterministic diagnostic cleanup are now described by the active contract.
  This auxiliary file does not prove that smoke, causal probes or training
  gates passed.
- `P2_FACTORIZED_INTERVAL_OBJECT_ROUTING.md`: a conditional, P2-only
  factorized interval/object soft-routing repair. It is activated only if an
  aligned Schema32 run confirms that healthy W interval residuals are still
  cancelled at the P2 joint-softmax boundary.

Do not treat these files as executable truth. The active architecture remains
owned by `../00_CURRENT_ARCHITECTURE_CONTRACT.md`; current release blockers
remain owned by `../CURRENT_MAINLINE_ISSUES.md`. Move only an explicitly adopted
decision into those files, at the point where its implementation enters the
active schema.

For this large continuous model, the preferred repair unit is a closed module
or a tightly coupled data-flow chain, not necessarily one line-level defect per
experiment. Keep constituent changes separated by owner, commit, diagnostics,
and rollback boundary, but judge a coherent bundle through layered boundary
metrics and the complete training/validation curve.
