# Auxiliary research memory

This directory preserves prospective research without changing the active
ClearVLA mainline contract or problem ledger.

Use it for:

- literature/source-code reviews;
- unapproved MIP/WAM, schedule, solver, or training proposals;
- experiment matrices and implementation sketches;
- unresolved alternatives that have not entered an active schema.

Focused pending designs:

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
