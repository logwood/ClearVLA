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
- Let modules publish their own diagnostic parameter groups; diagnostics may
  not reach into another module's private attributes.
- A run label is metadata and never selects source semantics.

## Atomic cutover gates

The public launcher remains on the frozen V122 path until the new package has:

1. a resolved active config with no inherited launcher chain;
2. independent online and teacher-forced APIs;
3. a complete G/S/W/P and bottom forward/backward;
4. a canonical loss ledger, optimizer and gradient diagnostics;
5. five-step deployment with verified static cache ownership;
6. checkpoint manifest and explicit compatibility behavior;
7. fixed-input provenance, zero-semantics, loss and gradient regressions;
8. local BF16 memory below 8 GiB and production batch-eight budget below
   22 GiB;
9. three completed source reviews: provenance, numerics/autograd, and
   runtime/performance.

Only after all gates pass will the default scripts switch once and the old
versioned launch/runtime surface move out of the active dependency graph.

## Current candidate status

The independent candidate is manifest schema `17` with top ABI
`object_intent_dynamics_323_conditional_object_writes_v11`.  It contains one
171,355,774-parameter model (171,253,374 trainable parameters); the 98,304
Teacher-G association
parameters are frozen and run only while constructing training targets.

The completed formal V122 run serialized 230,717,082 total / 168,064,059
trainable parameters with its real 4,096-wide T5 condition.  The candidate's
59,361,308 total-parameter reduction is now entirely accounted for by generic
role blocks 4/5/7/8 and legacy heads that are both frozen and skipped in the
old active forward.  The extracted graph no longer removes the trainable
G1-G3/P1 hosts, history proposal or controlled transition; its trainable count
is 3,189,315 above the actual V122 run.  The earlier 227,466,394 /
166,360,123 comparison was a synthetic/default-width inventory, not the
formal-run baseline.  In addition to typed boundary replacements and the
restored 18-D physical action field, schema 16 introduced the actual 4,096-wide
T5-XXL condition boundary used by the formal `.pt` file instead of the
erroneous 768-wide placeholder in schema 15.  Schema 17 adds one independent
512x512 W action-object writer and changes the active top/training semantics:
conditional global-object prototype reconstruction, factorized S object
keys/values, separately contracted W intent/action writes, semantic-band action
mass and budget-preserving gripper event-row pressure.

Completed local gates:

- the active import/source closure contains 50 Python/data-support files plus
  the resolved JSON spec (51 hashed artifacts total) and no legacy
  trainer/runtime/launcher;
- typed online, ephemeral training and future-supervision planes are disjoint;
- G/S/W/P object and interval axes, teacher isolation, exact-zero W/P3
  semantics, five-step static-cache ownership and optimizer ownership have
  executable regressions;
- exact resume validates source/data/language identity, model dtype/finite
  state, optimizer ownership/state, scheduler LR, RNG and owned generators
  before mutating the live run;
- fresh and bottom-migration runs cannot append to an earlier output stream,
  and a fresh run writes its context only after preflight succeeds;
- provenance, numerics/autograd and runtime/performance static reviews are
  complete for the extracted graph;
- learned flow is current-chart aligned, uses true normalized displacement and
  the four-frame raw-pair time unit; literal RGB anchors the geometry loss and
  the radius-2 neighbourhood is sampled in two kernels per refinement;
- frame position stays CPU/audit-only, gradient logs distinguish pre/post clip,
  and the logged warmup LR belongs to the update that produced the loss;
- formal goal/history/proposal condition dropout is exact-null on the policy
  path, leaves the full proposal target supervised, and owns a separately
  serialized training generator;
- an earlier schema-16 conservative synthetic batch-one CUDA BF16 train update
  used 336 RGB and an oversized 24x24 DINO chart, so its 2.674 GiB allocated /
  2.785 GiB reserved result is not the formal cache measurement; the active
  `dinov2_cache_336` ABI is 16x16 (256 patches/camera), with 12 future
  supports and the real 4,096-wide T5 condition, and its memory figure remains
  pending until the server smoke is rerun;
- the formal 18-D legacy-independent physical action field, anchor-band
  action objectives and deterministic information-balanced sampler are owned
  by the new data/training/runtime path; capacity and soft continuation retain
  direct non-zero action gradients without candidate replay;
- 86 independent-mainline, 27 frozen-object-mainline and 11 package-isolation
  regressions pass locally; the production mainline package also passes scoped
  Ruff and Pyright with zero error-level diagnostics.

Still pending before cutover:

- production CUDA batch-eight total process memory below 22 GiB;
- server fresh smoke, five-step deployment smoke and measured throughput;
- a fresh long run with full-epoch validation and frozen causal interventions.

Until those checks pass, `scripts/train_mainline.sh` and
`scripts/smoke_mainline.sh` are candidate-only entry points and the public
V122 launcher remains unchanged.

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
