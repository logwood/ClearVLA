# ClearVLA mainline extraction

This directory is the only destination for the new active implementation.
The existing V39 CLI, runtime and version-switched trunk are frozen migration
references; they are not architectural parents of this package.

## Purpose

Build one self-contained vertical path in execution order:

```text
experiment spec / manifest
  -> batch and language inputs
  -> current observation encoder and Pre-G evidence
  -> three active typed G role hosts and global-object grounding
  -> stateless intent S
  -> W1-W2 supervised future dynamics
  -> eight-row executed-history clean proposal
  -> static P1 role host and one high-resolution factual read
  -> action-centred controlled transition from W + clean proposal
  -> minimal deployment cache (S / W / P1 / transition / observable history)
  -> P2 consequence read
  -> P3 typed plan compiler
  -> one protected consequence plus typed optional lanes
  -> deterministic read-only Evidence MMDiT / execution controller
```

Training adds a separate, no-grad future-teacher branch.  Future evidence is
not an optional argument to the online forward API.

## Package boundaries

```text
clearvla/mainline/
  config.py          nested active configuration only
  manifest.py        compact serialized architecture identity
  interfaces.py      online/training typed boundaries
  model/             observation, G/S/W/P and bottom composition
  training/          objectives, optimizer, diagnostics and engine
  runtime/           deployment sampling, evaluation and checkpoints
```

The active implementation must not import:

- `clearvla.cli.train_v40_policy`;
- `clearvla.experiments.observed_state_lab.policy_runtime_v39`;
- `TemporalMidcutWorldActionDiT` from the historical trunk;
- a `current_vXXX` launcher or a version-wide validator.

Low-level, version-independent numerical primitives may be reused after their
input/output semantics are audited.  Temporary migration adapters are allowed
only in parity tests and are never part of the public mainline runtime.

## Change policy during extraction

The extraction is not a blind bit-for-bit copy.  Each migrated section is
classified as one of:

1. **Preserve**: mathematics and tensor ownership are already correct.
2. **Repair**: source inspection proves an architectural, numerical, contract
   or runtime defect.  The defect, old evidence, new owner and acceptance test
   must be recorded before changing behavior.
3. **Remove**: inactive ancestry, duplicate aliases, version dispatch or a
   diagnostic-only path that leaks into normal execution.

Unresolved empirical questions are not repaired by adding gates, quotas,
entropy targets or artificial gradients.  They remain experiment questions.

## Non-negotiable migration rules

- Preserve the current G/S/W/P semantic boundaries and every invariant in
  `docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md` unless a recorded source
  defect requires an explicit replacement.
- Keep local hypothesis, global object, camera, space, interval and type axes
  real until their named consumer.
- Build the training teacher once per batch and zero times in deployment.
- Perform the current high-resolution P1 read once per observation.
- Build observation/G/S/W/P1 once, but retain only S/W/P1 and observable
  history across deployment ODE steps.  Raw RGB, DINO, flow charts and
  reconstruction targets belong to the ephemeral training plane.
- Pass one protected consequence and typed P3 innovations to the bottom.
- Give every trainable parameter exactly one optimizer owner.
- Preserve V120's resolved optimization geometry explicitly: proposal `0.625x`,
  active bottom decoder `0.7x`, and no-decay capacity basis `1.4x`; keep the
  G/S/W/P owners at the public base LR.
- Let modules publish their own diagnostic parameter groups; diagnostics may
  not reach into another module's private attributes.
- A run label is metadata and never selects source semantics.

## Release gates and current status

`scripts/train_mainline.sh` and `scripts/smoke_mainline.sh` are the direct
entry points for this package.  They resolve one JSON config and do not inherit
a V-numbered launcher chain.  A formal result still requires all of:

1. independent online and teacher-forced APIs;
2. complete G/S/W/P/transition/bottom forward and backward;
3. canonical loss, optimizer and full active diagnostics;
4. five-step deployment with static evidence built once;
5. exact checkpoint/resume identity and explicit migration behavior;
6. provenance, zero-semantics, numerical and gradient regressions;
7. local BF16 memory below 8 GiB and production batch-eight below 22 GiB;
8. a controlled eight-epoch comparison against the V120 recovery baseline.

The active candidate is manifest schema `19` and contains 171,940,734 total /
171,838,334 trainable parameters.  Its current component identities and graph
invariants are recorded in
`docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`.

Schema 19 retains the independent typed layout and the source-proven numerical
repairs, while restoring or repairing the active mechanisms that the first
extraction weakened:

- three-frame causal DINO/raw history and both adjacent learned flows;
- G1-G3 hosted public context enters the address key, while object-owned
  candidate values remain private rather than receiving a copied public value;
- G reconstruction that cannot reduce its target by copying a public carrier
  or moving its own responsibility target;
- query/global-object-specific P1 reads over the complete local chart;
- additive factual/temporal P3 bases with zero-preserving W interactions;
- P2 selects intervals from observable intent innovations, and noisy ODE
  action can modulate P3 temporal content only through a nonzero consequence;
- ambiguous Teacher-G associations continuously fall back to the current
  object fact and zero geometry before becoming supervision, so neutral W
  rows remain learnable without fitting a diffuse future-patch average;
- 512 dense spatial transition directions pooled to 96 typed action-basis
  rows instead of one global row per horizon;
- a full-width, exact-zero/identity/non-expansive capacity operator;
- V120 per-row horizon mass and raw-unit event semantics;
- V120 role-specific optimizer pressure, without restoring the legacy launcher
  or allowing optimizer ownership to overlap;
- complete semantic logging and parser support for mainline JSONL/console
  output; JSONL keeps active exact zeros losslessly while the console remains
  compact;
- exact weighted `loss_contrib_*` accounting and separate event-balanced versus
  V120-comparable action/decoded rows, so the gripper-event repair does not
  corrupt cross-run train-scale comparisons;
- matched-noise validation ablations for proposal-zero, bottom no-updates and
  bottom full-updates on the bounded diagnostic subset, with signed utility,
  action delta and coverage rather than a bare "path active" flag.

The lower total parameter count relative to the old monolith is removed frozen
ancestry, not a deleted active G/P/bottom path.  The three Evidence MMDiT
blocks, G1-G3/P1 role hosts, history proposal, controlled transition,
capacity/continuation and event/motion heads are present.

Still pending before an empirical recovery claim:

- CUDA BF16 memory validation (the complete local regression, scoped static
  checks and CPU BF16 forward/backward already pass);
- production CUDA batch-eight process memory and throughput;
- fresh server smoke and five-step deployment smoke;
- a fresh full eight-epoch comparison, including all V120 action, horizon,
  gripper/motion, flow, ownership and gradient metrics.

Run those two stages from a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema19_recovery_smoke \
MAINLINE_BATCH_SIZE=1 \
bash scripts/smoke_mainline.sh

CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/schema19_recovery_b8 \
MAINLINE_BATCH_SIZE=8 \
bash scripts/train_mainline.sh
```

After all eight epochs, compare the complete archival streams rather than a
single best RMSE:

```bash
uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema19_recovery_b8 \
  --recovery-baseline v120_long.log \
  --tail 120 \
  --require-recovery \
  --format text
```

Passing the run directory loads both `metrics.jsonl` and
`run_context.json`.  The recovery gate checks the serialized public identity,
all eight epochs, V120-sized metric coverage, final and eight-epoch mean
action/horizon/arm/gripper/event behavior, train-tail scales, G/S/W/P owner
health, owner gradients and matched-noise proposal/execution ablations.  It
returns exit code `3` for either a demonstrated regression or missing proof;
one improved RMSE cannot pass the gate by itself.

## Frozen-mainline early-loss comparison

Run the two full graphs sequentially on one GPU with the same formal data,
sampler seed, batch size and public optimizer/objective settings:

```bash
CUDA_VISIBLE_DEVICES=0 \
COMPARE_BATCHES=20 \
COMPARE_BATCH_SIZE=1 \
bash scripts/compare_v122_mainline_early_loss.sh
```

The report is written to
`runs/v122_mainline_early_loss_compare/comparison.md`.  This comparison does
not require bitwise-equal losses: model initialization and private graph RNG
consumption differ after extraction.  It first requires exact parity of the
7-D/18-D physical action codec, then compares the early mean scale and
first-half/second-half direction of shared action losses.  G/S/W/P-private
objectives are reported only through their top-level representation group and
are not treated as tensor-equivalent terms.
