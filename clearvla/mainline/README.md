# ClearVLA independent mainline

`clearvla/mainline/` is the only active capability-named implementation.
V-numbered trainers and launchers are historical comparison material.

Read the [architecture contract](../../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md)
for graph semantics, the [issue ledger](../../docs/research/CURRENT_MAINLINE_ISSUES.md)
for unresolved behavior, and the [handoff](../../docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md)
for live process state. This file keeps only package boundaries and stable entry
points.

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
| `new_training` | New model/optimizer state and empty output directory |
| `exact_resume` | Continuation accepted by the serialized resume contract |
| `validation_replay` | Read-only checkpoint evaluation; no optimizer/RNG restore or writes |
| `component_initialization` | Bounded construction/smoke; not training evidence |

An output-directory guard is filesystem safety, not architecture identity.
Exact resume requires matching architecture/component ABI, data, optimizer and
continuation contracts.

## Stable entry points

Resolve config and dataset identity before launch; do not infer them from a
run tag. Use a fresh output directory for every formal run.

```bash
# Pen baseline smoke
CUDA_VISIBLE_DEVICES=0 OUT_DIR=runs/pen_smoke bash scripts/smoke_mainline.sh

# Pen baseline training
CUDA_VISIBLE_DEVICES=0 OUT_DIR=runs/pen_formal bash scripts/train_mainline.sh

# RDT-8 smoke/training
CUDA_VISIBLE_DEVICES=1 OUT_DIR=runs/rdt8_smoke bash scripts/smoke_rdt_multitask.sh
CUDA_VISIBLE_DEVICES=1 OUT_DIR=runs/rdt8_formal bash scripts/train_rdt_multitask.sh

# Read-only checkpoint validation
CHECKPOINT=/path/to/checkpoint bash scripts/validate_mainline_checkpoint.sh
```

The opt-in B-spine candidate requires
`MAINLINE_CONFIG=configs/mainline/object_intent_dynamics_323_pen_bspine0.json`;
omitting it selects the baseline. Override data/cache/T5 paths, batch size or
workers only when the resolved environment differs, and serialize the result.

## Audit and change policy

Audit a run directory, not a copied console tail:

```bash
python -m clearvla.tools.audit_policy_logs runs/<run> --format text
```

A complete curve outranks a best checkpoint. Smoke, interface checks and
checkpoint round-trip do not establish learned behavior.

Preserve typed axes, owner boundaries, call counts and the shared action field
unless a separately accepted decision changes them. Record producer-to-consumer
evidence before a semantic repair. Keep outlet conversion at the outlet
boundary, and do not use gains, quotas, artificial gradients, clipping or extra
loss weight to hide an unresolved connection.

Live PIDs and current release claims belong in the handoff and issue ledger,
not in this package README.
