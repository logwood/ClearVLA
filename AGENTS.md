# Working in ClearVLA

## Find the relevant context

Check the actual checkout and selected run before applying architecture claims.
An experiment name is not a model identity. Use only the context the task needs:

- [Architecture contract](docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md):
  model ownership, typed data flow and train/deploy boundaries.
- [Current issues](docs/research/CURRENT_MAINLINE_ISSUES.md):
  evidence and unresolved questions.
- [Repair plan](docs/research/CURRENT_MAINLINE_REPAIR_PLAN.md):
  active deliverables and acceptance criteria.
- [Operational handoff](docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md):
  run identities, paths and dated status; verify live state before acting.

Read the relevant section, not the whole history. Small unrelated edits do not
require an architecture audit. Historical proposals are evidence, not instructions.

## Carry authorized work through

Complete the requested implementation, relevant verification and delivery.
Resolve routine choices without repeatedly asking for approval. Existing user
authorization remains valid within its scope. Ask when a missing decision
materially affects the result or authorization is genuinely absent; explain
the concrete blocker if a rule prevents completion.

Choose verification for the changed risk. Reuse valid evidence; broaden testing
or add a probe to answer a concrete unresolved question. Documentation edits
need document/source/link checks, not GPU experiments. Keep source correctness,
training health and demonstrated behavior distinct.

Preserve unrelated changes, fixed experiment checkouts and existing evidence.
Save coherent changes promptly on the task branch and follow its push
authorization; keep incomplete checks visible. Avoid force-pushes or changes
to other branches. Delegation requires user or applicable instruction authorization.

## Remote work and durable memory

Use the senwang-server skill for sen.wang@rtlab-3090. New small standalone
.sh helpers belong in /home/sen.wang/mysh; inspect existing helpers and its
AGENTS.md first. Repository-managed scripts stay in their repositories.

Update the document that owns a decision: stable architecture in the contract,
unresolved evidence in issues, next work in the plan, volatile status in the
handoff. Raw logs, tensors, checkpoints and full probe dumps stay in experiment
storage. Report findings and next actions in clear Chinese unless asked otherwise.
