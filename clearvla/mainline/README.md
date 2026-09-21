# ClearVLA independent mainline

`clearvla/mainline/` is the only active capability-named implementation.
V-numbered trainers and launchers are historical comparison material.

Scope reviewed: 2026-09-19 UTC, current workspace source and its resolved
configs. This package description does not describe every frozen mainline
revision deployed on the server. Evaluate a checkpoint with its own source,
config and ABI; a newer README does not migrate it.

Read the [architecture contract](../../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md)
for graph semantics, the [issue ledger](../../docs/research/CURRENT_MAINLINE_ISSUES.md)
for unresolved behavior, and the [handoff](../../docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md)
for timestamped process observations. This file keeps only package boundaries
and stable entry points.

## Remote workspace

The only human-facing server root is `/data/senwang/clearvla`. Formal outputs
belong under `experiments/<outlet>/<run-id>/`; `active/<outlet>` is a pointer,
not model identity. New runs use a fixed repository plus commit-specific Git
worktrees under `checkouts/` instead of another full clone per experiment.

The [active handoff](../../docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md)
owns the path contract, dated run observations and task protection boundary.
Do not duplicate those volatile observations here.

## Package boundary

```text
config.py / manifest.py       configuration and serialized identity
interfaces.py                 disjoint online and future-supervision inputs
model/components.py           registered component hierarchy
model/policy.py               static/dynamic composition root
model/observation_*           current RGB/DINO history and G evidence
model/intent.py               S intent owner
model/dynamics.py             W future-world owner
model/v120_p1.py / compiler.py P1, P2 and P3 evidence
model/transition.py           controlled-transition consumer
model/restored_bottom.py      V120 execution bottom
model/action_codec.py         canonical physical action field
training/                     objectives, optimizer and engine
runtime/                      sampling, evaluation, logging and checkpoints
train.py                      direct entry point
```

The package must not import legacy trainers, runtimes or `current_vXXX`
launchers. Numerical primitives under `v120_core/` remain subordinate to
the typed boundaries above.

## Runtime contract

- Online input: three causal RGB/DINO frames, state, executed history and
  language. Future evidence is training-only.
- Training: one online encode, one formal velocity forward and one loss
  composition.
- Deployment: one proposal ODE, one decoded-proposal W rebuild, and one
  refined ODE from the same initial noise. This is a bounded correction, not a
  fixed point.
- Shared core: seven-dimensional action -> `[B,24,18]` physical field
  with arm value, arm adjacent-difference and continuous-gripper branches.
- Outlet semantics stay in adapters: Pen continuous, RDT-8 right-arm/two-view,
  CALVIN relative-command sampler/W adaptation plus isolated binary command.
- Optional B-spine adds a bottom-internal view beside the unchanged raw lift;
  it is not the default codec or an accepted behavior upgrade.

## Lifecycle vocabulary

| Mode | Meaning |
|---|---|
| `new_training` | Fresh model initialization and fresh optimizer/schedule/RNG in a new output directory |
| `exact_resume` | Continuation accepted by the serialized resume contract |
| `validation_replay` | Evaluate saved model state without mutating the checkpoint or restoring training continuation; diagnostics may be written to a separate output directory |
| `component_initialization` | Load admitted model weights/components from a verified checkpoint, then use fresh optimizer/schedule/RNG; this may start a formal training run or an explicitly bounded smoke |

Initialization describes how a run starts. `--smoke` describes a limited run
budget; completed training and validated task behavior are separate results.
`--init-checkpoint` is not an exact resume. Its initialization checks alone are
not training evidence, but a subsequent formal run can provide such evidence.
Any model/data-contract migration must be explicitly selected and admitted.

An output-directory guard is filesystem safety, not architecture identity.
Exact resume requires matching architecture/component ABI, data, optimizer and
continuation contracts.
The mode name alone does not certify sampler/worker next-batch equivalence;
the remaining continuation debt is tracked in the current issue ledger.

## Stable entry points

Remote new runs use `workspace launch` with an explicit commit, config,
interpreter and GPU. Its `--smoke` mode uses a separate run ID from formal
training. The single command/path contract is in the
[operational handoff](../../docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md#commands).
Do not use checkout-local `runs/` examples as a competing remote convention.

The shell launchers remain low-level/historical interfaces. Read-only checkpoint
validation must use the checkpoint's matching source/config and a distinct
output directory. Experiment-specific representation/outlet selection belongs
to that source revision and serialized context, not a stale version label in
this README. No historical checkpoint becomes an initialization merely because
its files are accessible from the new namespace.

## Audit and change policy

Audit a run directory, not a copied console tail:

```bash
python -m clearvla.tools.audit_policy_logs /data/senwang/clearvla/active/pen --format text
```

A complete curve outranks a best checkpoint. Smoke, interface checks and
checkpoint round-trip do not establish learned behavior.

Preserve typed axes, owner boundaries, call counts and the shared action field
unless a separately accepted decision changes them. Record producer-to-consumer
evidence before a semantic repair. Keep outlet conversion at the outlet
boundary, and do not use gains, quotas, artificial gradients, clipping or extra
loss weight to hide an unresolved connection.

Dated process observations belong in the handoff and release evidence in the
issue ledger, not in this package README.
