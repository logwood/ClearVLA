# Integration review: 17 September 2026

This is a source-integration candidate, **not a behavior release**. The base is
`codex/calvin-object-binding-20260911` at
`d7ebf00ed4b0041294c5ca44350c6ce79f8e782b`. It already contains `master`, the
active recovery line, the extracted numerical laboratories, the LIBERO bridge
and prior training-acceleration convergence. This review does not relabel that
inherited work as newly implemented. Original branches and running experiments
are not modified.

## Scope and provenance

All 16 original remote heads were inventoried by immutable SHA, ancestry,
merge base and unique patches. The active observation-to-action, checkpoint,
training and launcher paths received focused source review and regression
coverage. This is not a claim to have executed every historical experiment or
read every line of every archived implementation. Exact source heads and
individual dispositions are in `integration_sources.json` alongside this file.

| Source family | Disposition on this candidate |
|---|---|
| CALVIN object binding; `master`; Schema29 mainline; RDT data adaptation | Inherited through the selected base, with narrowly scoped repairs below. |
| Pen geometry/gripper donor | Its independent FFN-config repair is already present as `ed90020`; retain it. Do not reintroduce coupled raw-only/Q5, geometry or six-channel decoder experiments. |
| Pen B-spline routing; remove-B-spine/keep-solver | Retain the already extracted standalone solver and existing explicit B-spine variants. Their Q5/raw-only alternatives are not default replacements. |
| Pen shared core, anchored persistence and frame-scope heads | Preserve distinct experiment provenance; do not wholesale merge their older core, objective and gripper-chart changes. The current base already uses anchored persistence supervision, whereas the shared-core donor uses a deployment-cumulative objective; the frame-scope donor also assumes event/motion weighting absent from this base. |
| RDT multitask-prep | Preserve historical gate evidence, not the machine-specific old-source launch policy or threshold reversal. |
| Hybrid | Keep separate. Its fail-closed deployment intent is already addressed in the base's shared checkpoint path; Hybrid is not silently selected. |
| Schema25 replay, Schema28 estimator/recovery, V94 ownership | Historical or superseded evidence, not competing production entry points. |

No synthetic merge parents mark deferred experiments as merged. One candidate
owns the reviewed active implementation; mutually incompatible hypotheses are
not forced into a single default graph.

## Repairs

**CALVIN null ownership.** In v1, dividing the weighted object sum by the real
probability mass cancels the null score. Putting that mass back into the token
alone is insufficient: the coarse reader's LayerNorm removes scalar magnitude
again. V2 therefore keeps the conditional object context and transports the
existing `tanh(context_gate) * P(real)` as an explicit S-owned sidecar. The
coarse action consumer applies it *after* the entire object read, including its
RMS contraction and feed-forward update. An exact null contributes zero, and
both null and blend owners have action-level VJPs. G and W do not acquire a
language/Teacher input. No parameter, extra loss, ODE pass or RNG draw is added.

`top.calvin_object_binding=calvin_primary_v2` selects
`stateless_object_intent_calvin_binding_v2`. V1 remains explicitly selectable;
its numerical route is not silently rewritten. V1 and v2 have equal parameter
inventories but different component-selection and config identities. The
opt-in candidate config is `configs/mainline/calvin_object_binding_v2.json`.
The optional binding optimizer owner is now reported only when instantiated,
not as a zero-count mandatory Pen/RDT/LIBERO owner.

**Bounded online history.** Retain nine observations and 24 executed actions,
with an independent absolute time counter. Sparse offsets, reset padding and
copy isolation are preserved. An episode no longer retains every RGB frame.

**Launchers.** The root entry delegates to `scripts/train_mainline.sh`, not
V48. The common entry respects `CLEARVLA_PYTHON`, serialized dtype/batch/worker/
output defaults and explicitly supplied overrides. The RDT wrapper no longer
replaces a caller's data/config paths with machine defaults. The historical
CALVIN v1 launcher forwards CLI arguments, including `--help` and `--smoke`.
Its historical GPU/default settings otherwise remain explicit in that script.

**Finite preflight.** Audit online `action_state` and
`codec_gripper_boundary`, not only observation and target tensors. These
address/codec operands can no longer carry NaN through an otherwise successful
batch preflight.

**Deployment ABI.** Validate all active outlets against their registered
projection fields, numeric profile digest and independently owned gripper
boundary. A copied digest cannot authenticate modified coordinates or scales.
Use strict integer dimensions, DINO batch shape and sparse history offsets;
check redundant widths/horizon/dtype against the graph. Export invokes the
same validator used by import. RDT's ordered `high,right_wrist` source aliases
are accepted as the declared global/right-wrist axes, without changing their
array order or inventing additional camera support.

**Deployment source identity.** The checkpoint snapshot now includes the
static closure of the online policy entry as well as the trainer. History,
DINO preprocessing, restoration and online input assembly are protected
source owners. Offline-loader changes retain their existing separate
compatibility treatment.

## Verification and release boundary

The integration checks exercise active mainline contracts, action-level VJPs,
Teacher separation, checkpoint restore, all four deployment outlets, causal
windows, benchmark bridge, acceleration contracts and the new regressions.
Numerical representation/solver tests run separately. GitHub Actions retains
JUnit results and the exact installed environment; those artifacts, rather
than a hand-edited claim here, own run-specific counts.

The local development environment is Python 3.13 / Torch 2.10 CPU, outside the
repository's formal version range. The GitHub checks use Python 3.12 and Torch
2.11 CPU. Neither CPU environment establishes CUDA/BF16 parameter-owner VJP,
GPU memory equivalence, simulator success or a trained policy improvement.
Real dataset/cache/T5 assets and production checkpoints were not available in
this review environment. A full historical test invocation was stopped at its
time limit; it is not reported as a passed full-repository suite.

This source change intentionally changes source identity. Preserving the v1
mathematics is **not** permission to exact-resume or replay an old checkpoint
under a new source digest. Keep historical checkpoints with their recorded
checkout. Do not remove identity gates or overwrite manifests to make a load
succeed. V2 needs a fresh run and matched causal/closed-loop evaluation.

## Remote strict-equivalence blockers

The first Python 3.12.14 / Torch 2.11.0+cpu review found **528 passed,
2 failed, 5 skipped** in the 535-case mainline suite. The two failures are
`test_prepared_block_contexts_preserve_training_update_exactly` (a history
projection gradient) and
`test_experimental_vmap_block_bank_matches_individual_blocks` (maximum observed
absolute output difference 2.384185791015625e-7). Their strict zero-tolerance
assertions remain unchanged and remain included in the integration check.
This is a blocked draft candidate, not an all-green integration or a numerical
repair of those acceleration experiments. The full CI artifacts own later
results and baseline attribution. Context reuse is an explicit CLI option
requiring CUDA Graph training; neither it nor experimental owner-vmap is
promoted to a default by this review. Do not infer CUDA equivalence from CPU
forward agreement or from a small maximum output difference.

## Entry and next gates

A finite candidate smoke, on a machine containing the declared assets:

```bash
MAINLINE_CONFIG=configs/mainline/calvin_object_binding_v2.json \
OUT_DIR=runs/calvin_binding_v2_smoke_new \
CLEARVLA_PYTHON=python bash scripts/smoke_mainline.sh
```

Use a new, empty output directory and explicitly set CUDA visibility on that
machine. This review did not execute that asset-dependent command. For Pen or
RDT, retain the profile's own current config rather than selecting a CALVIN
binding. `scripts/check_integration.py` provides CPU-only regression entry
points without training assets.

Before promotion, require real CUDA/BF16 owner-VJP and memory checks, a fresh
matched CALVIN v1/v2 run, and closed-loop evaluation. Retain the existing
proposal/refinement residual metrics: two five-update passes and one W rebuild
remain a bounded correction, not a solved fixed point. Geometry/typed routing
under-use, far-horizon gripper utility, native RDT 14-D/three-camera support and
mutually incompatible Pen objective variants remain research/ABI decisions,
not bugs declared solved by static inspection.
