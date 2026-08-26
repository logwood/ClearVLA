# ClearVLA architecture replay plan

Status: source-first replay method adopted; Schema25-R1 source bundle selected;
R1a/G-01 through R1h/N-01,D-01 plus LC-01 exact-zero cleanup implemented and
statically closed; formal-run boundary awaits separate authorization; no
training run.
Established on 2026-08-26.

This is the only auxiliary document that owns the current replay procedure and
per-change disposition. It does not by itself change the checked-out source or
the active architecture contract. `../00_CURRENT_ARCHITECTURE_CONTRACT.md`
remains the compact source of truth for the checked-out replay graph.

Historical evidence and donor ancestry are owned by
`ARCHITECTURE_REPLAY_LEDGER.md`. The frozen Schema39 diagnosis is owned by
`ACTIVE_MAINLINE_HANDOFF.md`. Neither of those files selects the replay base.

## 1. Fixed replay base

```text
Schema:       Schema25
Commit:       6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6
Commit title: fix: preserve S object ownership in schema 25
Primary log:  schema25_s_owned_typed_b8.log
Coverage:     8 completed epochs, 1,136 logged training windows
```

The closed read-only base audit is recorded in
`SCHEMA25_BASE_VERSION_CARD.md`. That card fixes the source/log boundary,
complete forward/backward/runtime map, protected semantic invariants and the
initial `BASE/OPEN` debt register. It does not endorse every Schema25 operator.

Schema25 is selected because it is the earliest checkout-able
independent-mainline version in the local recovery island with a complete run,
the strongest local physical/gripper behavior, single S-owned typed relevance,
and substantially more remaining `[type,time,object]` freedom than later
graphs.

The Schema25 console log is a context-scale behavioral record, not a weak
summary artifact. Its missing serialized `run_context` prevents a claim of
bit-exact environment identity, but it does not require another training run to
choose the replay base. Source identity is anchored separately to commit
`6a6c1bf` and must be checked statically before implementation begins.
This log paragraph closes the already-made base choice only. Later semantic
units are defined from source and do not require rereading the log.

Schema26 at `caa7e3315e85e5f4119fe3174e86037b47a5903c` is the first donor. It is not
the base and must not be cherry-picked as one indivisible change.

## 2. Execution model

Replay acquisition is cross-version and source-first. Schema26-39 are scanned
as one donor field before an implementation bundle is selected:

```text
lock the exact Schema25 source base
  -> scan every later source diff for producer/consumer changes
  -> group hunks by live semantic boundary rather than commit title
  -> reconstruct each unit's complete forward and backward path
  -> record dependencies, conflicts and rejected exact mechanics
  -> select one coherent multi-unit candidate
  -> apply its units as separately reversible source commits
```

The target is not to recreate any whole later schema. A historical change is
judged against the replay state that exists immediately before it, not against
Schema39 and not against the defect description written by its original
author.

### 2.1 Two-coordinate source review

Every later schema is read through two different diffs:

```text
historical parent -> historical schema
  answers: what changed together, what defect was claimed, and what new debt
           or repair first appeared here?

Schema25 + already accepted replay units -> candidate semantic unit
  answers: does the defect exist on our reconstructed base, does the unit still
           fit its real owners/axes, and what is the least hardened valid form?
```

The historical diff recovers what changed together. The replay-relative
coordinate decides whether the defect exists on Schema25 and what the least
restrictive repair is. Neither coordinate makes a historical version the unit
of work.

All donor versions are scanned before bundle selection. The output is one
cross-version source-unit register, not one card per version. A Schema27 hunk
may therefore join a Schema30 and Schema35 hunk in one grounding unit, while a
different Schema27 hunk is rejected independently.

ARCHITECTURE_REPLAY_SOURCE_UNITS.md owns the detailed source maps. This plan
owns their dispositions and the selected candidate.

No training experiment is required for each replay unit. Exact source mapping,
algebraic checks, sentinel perturbations, call-count checks, forward traces and
reverse VJPs are the default evidence. Existing logs may rank an already
closed candidate, but they do not define an implementation semantic unit.
Training is considered only after the coherent multi-unit source candidate
closes end to end and only as a separately authorized use of the experiment
budget.

## 3. Replay unit and allowed dispositions

A replay unit is the smallest source change that has one coherent semantic
effect. A single historical commit may contain several units and may therefore
receive several different dispositions.

| Disposition | Meaning |
|---|---|
| `BASE` | Part of the fixed Schema25 starting graph; not a claim that it is final truth |
| `KEEP` | Reuse the historical form after its complete boundary is verified |
| `SOFTEN` | Preserve the purpose but remove an unnecessarily restrictive operator |
| `REIMPLEMENT` | Preserve the invariant while rewriting it for the replay graph's actual axes and owners |
| `DEFER` | Evidence is insufficient or a prerequisite owner/path has not yet been reconstructed |
| `REJECT` | The unit removes legal information, duplicates ownership, creates a shortcut or repairs a defect absent at this replay point |

`DEFER` is not a request for an immediate experiment. It means the unit stays
out unless later source/log evidence closes the unresolved assumption.

## 4. Mandatory change record

Before a unit receives `KEEP`, `SOFTEN` or `REIMPLEMENT`, record all of the
following in the chronological register:

```text
donor commit and exact diff hunk
original defect claimed by the donor
whether that defect exists in the current replay state
producer -> transformations -> every consumer
ordinary loss and decoded-action gradient path
tensor axes, dtype, zero/null semantics and expected scale
information introduced, preserved, pooled, zeroed or removed
residual/addition point and repetition frequency
alternate consumer, owner duplication and possible bypass
checkpoint/config/runtime/diagnostic consequences
disposition, reason, prerequisite and rollback commit
static verification performed
unresolved assumptions
```

Shape validity, nonzero gradients and passing unit tests are necessary checks,
not evidence that the unit has correct ownership or functional closure.

## 5. Over-hardening audit

The following operators are review flags, not automatic bans. Each occurrence
must have a source-backed necessity and must be compared with the least
restrictive form that preserves the same invariant.

| Flag | Required question |
|---|---|
| Axis pooling, averaging or reconstruction | Which distinctions disappear, and can the consumer still recover the original type/time/object meaning? |
| Shared posterior or winner competition | Are the alternatives truly exclusive, or merely complementary owners forced to compete? |
| `detach`, `no_grad`, hard masks or branch deletion | Which ordinary gradient or conditioning path is removed, and is another legal path left? |
| Learned null/dustbin or predicted validity | Does it represent absence, or does it gain authority over branch amplitude or physical existence? |
| Fixed divisors, floors, gains or mixture constants | Are they derived from algebra, or compensating for an observed magnitude? |
| `tanh`, clamp, normalization or bounded Jacobian | Is boundedness needed at this exact boundary, and does it preserve zero and relative distinctions? |
| Forced common/residual isolation | Does it prevent leakage, or also forbid a legitimate one-way conditioning relation? |
| Duplicate auxiliary supervision or identity targets | Can a module minimize loss without learning the intended innovation? |
| Repeated owner/value aliases | Does the same evidence reach a selector or consumer through more than one semantic name? |

A unit is over-hardened when it solves a local ownership or numerical problem
by deleting legal conditioning, collapsing a still-meaningful axis, fixing an
unbounded scale competition with a constant, or giving a null/validity route
control over value amplitude. Boundedness alone is not over-hardening; the
location, semantics and lost freedom determine the verdict.

## 6. Donor coordinate register

This table is the traversal order, not a whole-commit verdict. Detailed units
are added below a donor only after its complete boundary has been mapped.

| Order | Schema | Commit | Initial audit focus | State |
|---:|---|---|---|---|
| 0 | Schema25 | `6a6c1bf` | Fixed replay base; full boundary and retained debt in `SCHEMA25_BASE_VERSION_CARD.md` | `BASE CARD COMPLETE` |
| 1 | Schema26 | `caa7e33` | Exact G3 transition source, S boundary, common/residual construction, typed re-entry removal | source-scanned: G-01, S-01 |
| 2 | Schema27 | `2a0d3d1` | Null calibration, removed P3 conditioning, pre-W future supervision, visibility suppression | source-scanned: G-02, S-03, W-02, P3-01; mixed keep/reject |
| 3 | Schema28 | `e7d7f42` | Ownership-dataflow repair and any new attenuation bundled with it | source-scanned: S-01 anti-donor, P3-01 |
| 4 | Schema29 | `5d0bc77` | P2 ownership repair, complementary fusion and fixed division | source-scanned: P2-01; exact fusion rejected |
| 5 | Schema30 | `127fee8` | Grounding reconstruction and remaining fixed averaging | source-scanned: G-02; fixed divisor rejected |
| 6 | Schema31 | `c159651` | S-W-P2 closure, partial OT/dustbin distribution and status/validity coupling | source-scanned: S-02, T-01; Teacher backend deferred |
| 7 | Schema32 | `4ac7e54` | Observable S intent, W future ownership and one-way common-to-residual causality | source-scanned: S-03, W-01 |
| 8 | Schema33 | `a2b5705` | Factorized P2 routing and shared time posterior | source-scanned: P2-01; exact shared posterior rejected |
| 9 | Schema34 | `4363108` | W/P2 information preservation, bilateral isolation and reusable diagnostics | source-scanned: W-01 anti-donor, D-01 |
| 10 | Schema35 | `03235d3` | Causal ownership, physical-null semantics and static/dynamic split | source-scanned: W-02, P1-01; content-only K owner deferred |
| 11 | Schema36 | `9e75d31` | P1/P2 ownership repair and inherited consumer attenuation | source-scanned: boundary warning; no standalone R1 unit |
| 12 | Schema37 | `1b11bf5` | Axis identity, per-lane bottom routing and remaining optional aliases | source-scanned: S-02, W-02, P3-01, B-01 |
| 13 | Schema38 | `6bc6218` | Action consumers, full-field W/P2 path and dynamic precision | source-scanned: P1-01, P2-01, N-01, D-01 |
| 14 | Schema39 | `eac4916` | Spatial/temporal terminal split, no-null terminal and numerical paths | source-scanned: P1-01, P2-01, B-01, N-01, D-01 |
| 15 | Training-path update | `7cd69a7` | Runtime/training identity only; evaluate after graph replay closes | deferred until R1 graph closes |

Known-break or strong-donor labels only prioritize reading. They do not decide
the disposition of any contained change.

### 6.1 Cross-version unit dispositions

The detailed producer, consumer, gradient, axis, null and mechanism records are
in ARCHITECTURE_REPLAY_SOURCE_UNITS.md. The adopted dispositions are:

| Unit | Disposition | Candidate role |
|---|---|---|
| G-01 | KEEP | exact completed G3 source shared by P1 and transition |
| G-02 | REIMPLEMENT | conditional-K G3 plus null-independent exported-content reconstruction; retain the Schema25 base binder inputs |
| S-01 | KEEP purpose / REIMPLEMENT ABI | one typed owner; remove typed CoarseAction re-entry |
| S-02 | REIMPLEMENT | lossless common plus interval-residual views while retaining Schema25 relevance scoring |
| S-03 | KEEP BASE / REJECT donor shortcut | no direct pre-W future-field supervision |
| W-01 | REIMPLEMENT | common may condition residual once; residual cannot rewrite common |
| W-02 | REIMPLEMENT | typed semantic/appearance/geometry field; remove online status and predicted-validity authority |
| P1-01 | REIMPLEMENT | static fact and dynamic policy residual remain distinct; dynamic residual is protected without null |
| P2-01 | REIMPLEMENT | spatial K/camera selection preserves interval, then one physical no-null interval terminal |
| P3-01 | REIMPLEMENT | protected consequence and dynamic precision plus only temporal/state-change optional innovations |
| B-01 | REIMPLEMENT | lane-local optional reads and protected no-null reads at one bottom ingress |
| N-01 | KEEP matching mechanics | FP32 measures, finite all-invalid rows and zero-preserving variance numerics |
| D-01 | KEEP matching diagnostics | only diagnostics whose producer still exists |
| T-01 | DEFER | retain Schema25 Teacher row-softmax for the first candidate |

### 6.2 Selected next candidate

Working name: Schema25-R1 ownership and terminal closure.

R1 is one coherent training candidate, not one historical version and not one
experiment per unit. It is assembled as reversible source commits in this
order:

1. G-01 exact G3 handoff.
2. G-02 conditional-K and reconstruction closure.
3. S-01/S-02 docks and lossless decomposition, with S-03 enforced as a guard.
4. W-01/W-02 causal typed field and online ABI.
5. P1-01 static/dynamic split.
6. P2-01 spatial selection and physical interval terminal.
7. P3-01/B-01 unique lanes and bottom ingress.
8. N-01/D-01 matching numerics and diagnostics.

Slices 1-8 are complete as separate reversible commits. R1c preserves the
Schema25 selector exactly, removes the typed ActionIntentDock/CoarseAction
alias, and gives W lossless common/residual coordinates through its sole typed
dock. R1d processes common once, gives W2 read-only access to completed W1,
retains camera-specific geometry through Teacher/loss/P2, and removes online
status authority. Its closed worksheet is
`R1D_W01_W02_CAUSAL_FIELD_ABI_WORKSHEET.md`; the retained suite is 140/140.
R1e keeps the cached fact and live P1 policy residual disjoint, materializes
their full query only at P2, and passes the raw residual to transition and
bottom without adding a parameter or amplitude contract. Its closed worksheet
is `R1E_P101_STATIC_DYNAMIC_P1_WORKSHEET.md`; the retained suite remains
140/140. R1f retains I through semantic K and geometry K*C selection, lets S
condition only the selected nonzero W key, and uses independent no-null
physical-I terminals before the raw complementary sum. Its closed worksheet
  is `R1F_P201_SPATIAL_PHYSICAL_TERMINAL_WORKSHEET.md`; the retained suite is
  144/144. R1g removes optional factual/static-precision/effect aliases,
  retains only temporal and state-change innovations, and gives each a
  separate shared-parameter Q+null bottom decision. Protected consequence and
  dynamic precision keep separate no-null calls. Its closed worksheet is
  `R1G_P301_B01_UNIQUE_LANE_BOTTOM_INGRESS_WORKSHEET.md`; the retained suite is
  145/145. LC-01 then deletes two source-proven exact-zero terminal-contract
  trajectory aliases and 23,590 frozen readout parameters without changing a
  retained contract/decoder tensor, optimizer owner or the construction RNG;
  its closed worksheet is
  `LC01_EXACT_ZERO_LAYER_TRAJECTORY_CLEANUP_WORKSHEET.md`. R1h then keeps
  observable measures/logs FP32, makes the four active address variance paths
  exact-zero with finite slope, and adds only matching read-only diagnostics.
  Its closed worksheet is
  `R1H_N01_D01_FINITE_NUMERICS_DIAGNOSTICS_WORKSHEET.md`; the final retained
  suite is 155/155 and every LC-01 inventory/RNG sentinel remains exact.

There is no training between these commits. R1 retains the Schema25 Teacher
association backend, S relevance scoring and base K-binder evidence, so the
candidate does not silently import partial OT, equal typed-logit consensus or
content-only K identity. It also does not import fixed candidate-count null
calibration, fixed complementary averaging, bilateral W isolation, six P3
aliases or Schema39's exact S-W score relation.

The run decision comes only after the complete R1 source graph passes
algebraic reconstruction, axis sentinel, exact-zero, call-count, forward-path
and reverse-VJP checks. Those static gates now pass; they do not themselves
authorize dataset/checkpoint access or a training run.

## 7. Base-state audit before the first donor

Before applying Schema26 material, map the complete Schema25 boundaries that a
later donor may touch:

```text
G/G3 -> S -> CoarseAction
Teacher -> W common/residual -> P2/P3
static and dynamic P1 -> transition and bottom
P2/P3 -> retained bottom -> CVAE/workspace -> decoded arm/gripper
loss -> optimizer owner -> checkpoint/runtime consumer
```

For each boundary record axes, zero semantics, scale, repetitions and all
ordinary gradients. Semantic closure is reconstructed from the exact Schema25
source. The existing Schema25 log remains separate behavioral evidence and is
not required to define this boundary. This is not an instruction to run
Schema25 again.

This prerequisite is complete in `SCHEMA25_BASE_VERSION_CARD.md`. Completion
means the first donor review may begin; it does not mean the base graph has no
unresolved structural assumptions.

## 8. Document update rule

- Update this file in place as replay dispositions are made; do not create a
  new plan file for every session or schema.
- Keep `ARCHITECTURE_REPLAY_SOURCE_UNITS.md` organized by live source
  boundary. It owns detailed source maps but cannot independently select a
  bundle or authorize implementation.
- Keep `ARCHITECTURE_REPLAY_LEDGER.md` factual and append historical evidence
  there only when a historical source or behavioral claim is corrected. Do not
  use it as a substitute for reading the exact donor source.
- Keep implementation sketches that have not been adopted in this auxiliary
  directory and mark them as prospective or historical ancestry.
- Update the active architecture contract only when the user explicitly adopts
  a replay implementation and that implementation is entering active scope.
- Never copy raw logs, checkpoints, tensor caches or full probe dumps into this
  plan. Record only source anchors, decision-making statistics and reproducible
  commands.
