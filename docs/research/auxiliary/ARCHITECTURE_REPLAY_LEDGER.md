# ClearVLA architecture replay ledger

Status: historical/source audit; evidence snapshot frozen on 2026-08-26.

This document reconstructs the architectural lineage from V120 through the
Schema39 snapshot at commit `7cd69a7`. It does not select a replay base or an
implementation bundle. The adopted replay method and its decision register are
owned by `ARCHITECTURE_REPLAY_PLAN.md`.

This ledger is not the active architecture contract, not a list of current
issues, and not authorization to modify the mainline. The checked-out graph
remains defined by `../00_CURRENT_ARCHITECTURE_CONTRACT.md`; open release
questions remain in `../CURRENT_MAINLINE_ISSUES.md` until a replay
implementation explicitly enters active scope.

## 1. Evidence boundary

The replay uses four evidence levels and keeps them separate:

1. **Source fact**: a commit, exact source path, tensor algebra or serialized
   interface directly establishes the claim.
2. **Log observation**: a completed training/validation record establishes a
   measured value, but not its cause.
3. **Inference**: source and logs support a causal explanation but do not prove
   a unique cause.
4. **Recovery candidate**: a historical implementation or invariant can be
   reused only after its target producer, consumer and zero semantics are
   verified.

Historical input documents were treated as audit material rather than
instructions:

```text
ClearVLA_V120_to_V25_architecture_replay.md
SHA256 2FF70AF22005B343D47CC75359552D5FCAF4179AB8B622C830659795C09A1DBA

ClearVLA_architecture_replay_ledger.md
SHA256 2187425BB5C421EF594E035D8CE9E707FD0BEDCC7E09D0ABEFB2407491C6D949
```

These two inputs were external handoff attachments. They are identified here
by name and digest but are not retained as repository files. Their claims are
not authoritative unless independently anchored to the code or logs below.

Authoritative code/log anchors:

```text
V120 behavior source: 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
Schema25 source:      6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6
Schema26 source:      caa7e3315e85e5f4119fe3174e86037b47a5903c
Schema39 snapshot:    7cd69a797c7dde2e9eea8a51014c93385478cea2
```

The V120 log serializes action normalizer fingerprint
`32a3a4d7f21f`. The independent-mainline logs do not serialize a directly
comparable fingerprint. Their physical metrics use the same task/data/batch
surface and are useful behavioral anchors, but a statement that Schema25 strictly
beats V120 must remain directional rather than bit-exact.

## 2. Historical synthesis through the Schema39 snapshot

The lineage is not a story in which every version simply became worse. It is a
repeated three-step failure pattern:

```text
identify a real duplicate owner or shortcut
  -> add a locally valid ownership restriction
  -> also remove legal conditioning or information bandwidth
  -> repair the missing consumer in a later version
  -> harden a different boundary while protecting the repair
```

Three historical regions must be distinguished:

- **V120 is the behavior root, not architectural truth.** It trained well but
  already contained complementary/type competition, duplicate P3 carriers and
  a joint lane/basis/null bottom softmax.
- **Corrected Schema24 plus Schema25/26 is the recovery island.** It restored
  most V120 lifecycle/P1 behavior and produced the best local performance, but
  still contained dynamic/static P1 mixing, P2 type competition, status value,
  W typed publicization and the inherited bottom-null shortcut.
- **Schema27 and Schema31 are the two clearest later breakpoints.** Schema27
  simultaneously introduced a large P2 null prior, removed P3 conditioning and
  made W identity-like through pre-W future supervision. Schema31 then changed
  Teacher association distribution; the existing dustbin-to-status coupling
  converted matching ambiguity into physical suppression.

Schema39 is not equivalent to those broken historical graphs. Its source
contains later repairs that remain valid donor candidates: typed/camera-aware
spatial P2, a no-null physical interval terminal, split static/dynamic P1
ownership, per-lane bottom routing and neutral dustbin status semantics. Using
Schema25 as a replay base does not discard those repairs; it changes them from
inherited assumptions into changes that must be reviewed and deliberately
replayed. The Schema39 logs still show an accumulated bandwidth failure:

```text
high Teacher dustbin and low target interval variation
  -> temporally public W field
  -> weak S leverage at the physical interval terminal
  -> broad interval averaging cancels much of the remaining innovation
  -> protected dynamic precision remains easier than static detail/W timing
  -> gripper calibration/timing regresses
```

The lineage supplies the following donor inventory:

```text
V120 lifecycle/bottom behavior
+ Schema25/Schema26 typed ownership and exact transition source
+ V121/V122 innovation/conditioner principles
+ Schema32 one-way W causality
+ Schema35 physical-null/static-dynamic semantics
+ Schema37 per-lane bottom routing
+ Schema39 typed/camera spatial selection and no-null interval terminal
- historical type/status competition
- duplicate P3 value aliases
- learned-null amplitude authority
- forced shared posterior or bilateral isolation
- unverified Teacher distribution changes
```

This inventory is not an implementation prescription and does not determine
which source version must be the base. Per-change dispositions are recorded
only in `ARCHITECTURE_REPLAY_PLAN.md`.

## 3. Behavioral anchors

Completed-log summary:

| Run | Best physical RMSE | Final/last physical RMSE | Component point | Arm at component point | Gripper at component point | Median s/batch | Peak GPU |
|---|---:|---:|---|---:|---:|---:|---:|
| V120 | 0.07931 (e7) | 0.08145 (e8) | e8 final | 0.06325 | 0.14979 | 2.206 | legacy log |
| V122 | 0.08910 (e6) | 0.09110 (e8) | e8 final | 0.0683 | 0.1733 | 2.020 | legacy log |
| Schema25 | 0.07724 (e6) | 0.07887 (e8) | e6 best | 0.06186 | 0.13713 | 1.853 | 11.646 GiB |
| Schema26 | 0.07893 (e6) | 0.08016 (e7) | e6 best | about 0.0620 | about 0.1433 | 1.922 | 11.724 GiB |
| Schema36 | 0.0801 (e6) | 0.0811 (e8) | reported comparison point | 0.0598 | 0.1566 | 1.870 | 11.974 GiB |
| Schema37 | 0.0815 (e6) | 0.0816 (e8) | e8 final | 0.0614 | 0.1549 | 1.925 | 11.913 GiB |
| Schema38 | 0.08175 (e8) | 0.08175 (e8) | e8 final | 0.06182 | 0.15444 | 1.854 | 11.974 GiB |
| Schema39 | 0.0847 (e4) | 0.08551 (e7, incomplete) | e7 incomplete | 0.06128 | 0.16929 | 1.934 | 11.954 GiB |

The component columns intentionally name their comparison point. Earlier
drafts incorrectly labelled all of them as the last epoch even though the
Schema25 and Schema26 component values came from their best epoch.

The Schema39 snapshot gap is not a generic arm-trajectory collapse. Its arm
result is near the anchors; the measured regression is concentrated in the
physical gripper channel and decoded event timing.

Structural comparison at the latest completed points available:

| Metric | Schema25 | Schema26 | Schema36 | Schema37 | Schema38 | Schema39 |
|---|---:|---:|---:|---:|---:|---:|
| Teacher dustbin | 0.0421 | 0.0509 | 0.4923 | 0.4848 | 0.4664 | 0.4577 |
| Teacher semantic delta RMS | 0.3499 | 0.3527 | 0.1716 | 0.2033 | 0.2302 | 0.2198 |
| Teacher interval variation | 0.1343 | 0.1338 | 0.0587 | 0.0447 | 0.0598 | 0.0466 |
| W predicted interval variation | 0.0857 | 0.0839 | 0.0355 | 0.0266 | 0.0362 | 0.0277 |
| W adjacent interval cosine | 0.9087 | 0.9087 | 0.9193 | 0.9679 | 0.9459 | 0.9677 |
| static P1 factual RMS | 0.0402 | 0.0460 | 0.0328 | 0.0200 | 0.0383 | 0.0252 |
| dynamic P1 residual RMS | 0.2521 | 0.2932 | 0.5471 | 0.5038 | 0.3567 | 0.1885 |
| optional P3 precision RMS | 0.3085 | 0.2871 | 0.1895 | 0.0343 | 0.1367 | 0.0202 |

Schema39 separately reports protected dynamic policy precision RMS `0.2312`.
The optional precision lane and protected dynamic precision are different
objects and must never be merged in diagnosis.

## 4. Complete source lineage

### 4.1 V120: strong behavior with inherited structural debt

Source: `0b92d359`.

Keep:

- physical K+null grounding and bounded G3 correction;
- full axis identity until a real consumer;
- S restricted to language, observable state/history and current facts;
- one clean action-conditioned ingress into W;
- W1 near followed by W2 far;
- only supervised future dynamics crossing W->P;
- zero-preserving consequence;
- single-stage lifecycle, dynamic transition, retained bottom/CVAE/workspace;
- no quotas, forced nonzero paths or synthetic gradients.

Inherited debt:

- semantic/appearance/geometry compete with one learned null inside W;
- P2 uses one interval-object posterior followed by semantic/geometry/status
  type softmax;
- protected consequence coexists with optional factual/effect aliases in P3;
- bottom puts lane, basis and branch abstention in one simplex;
- learned null therefore controls both true absence and branch amplitude.

The V120 log also shows that weak goal selectivity and temporally common W were
already present. Later versions amplified these debts; they did not invent all
of them from zero.

### 4.2 V121: first broad semantic cutover

Source: `480f71c`, approximately `+3029/-1148` across 20 files.

Useful ideas:

- semantic and geometry obtain separate P2 selectors;
- P3 removes the explicit factual/effect optional aliases and exposes a
  protected base plus precision/temporal/state-change lanes.

Source-certain defects and qualifications:

- The Teacher constructs `successor_content` from a reliability-weighted stable
  successor but constructs `semantic_delta` from an end-biased successor. The
  two fields no longer satisfy one exact
  `semantic_delta == successor_content - current_reference` optimum.
- The separate P2 selectors are recombined by a learned semantic/geometry
  `type_weight` softmax. This is only a partial split, not complementary closure.
- V121 precision and temporal both consume full protected consequence as a
  condition. The three-lane skeleton is useful; calling all three values
  strictly private innovations is too strong.

Conclusion: never cherry-pick V121 wholesale. Recover only its lane skeleton
and removal of explicit factual/effect aliases, with corrected operands.

### 4.3 V122: identity/innovation principle, failed complete behavior

Source: `ced6f23`.

Strong principle:

```text
owner value x determines whether information exists
conditioner c may organize x
x == 0 => organized value == 0
```

Useful concrete part:

- temporal reads `consequence.effect + consequence.interaction`, not the full
  protected fact;
- state-change is multiplicatively zero-preserving.

Unsafe concrete part:

- precision centres an object-specific P1 field across K and routes it through
  a learned null. It deletes common detail and assumes a real K-specific P1
  axis. The restored V120 P1 produces `[B,24,4,H]`; recreating K with `expand`
  would be a false identity.

Behaviorally V122 is a failed baseline relative to V120. Its Teacher target
RMS remains near V120, so its failure is not the later low-bandwidth Teacher
failure; it is mainly an object/S/consumer regression while the V121 target
conflict remains.

### 4.4 Independent-mainline reset

Source: `91a4237`, about 18,758 inserted lines across 59 files.

The engineering separation into config/manifest/model/training/runtime/tests is
valuable. The first graph was not a mechanical V120 extraction. It rewrote the
outer G/S/Teacher/W/P/bottom lifecycle and immediately produced object and W
pair cosine `1.0`. This establishes that the later schema sequence is a
behavioral recovery sequence, not a clean enhancement sequence.

### 4.5 Schema20-24: recovery of lost V120 behavior

Key commits:

```text
4cfe788  Schema20 restore V120 core
2d0a84c  Schema21 recover V120 behavior
112fd89  Schema22 update V120 contracts
5b80251  Schema23 restore V120 training semantics
ec84c9e  Schema24 restore V120 P1 grounding
32d969f  correct geometry and validation behavior
```

The important recoveries were:

- per-ODE noisy-action transition and learned neutral behavior;
- correct bottom source order;
- query-first N=49 P1 access instead of reducing candidates first;
- real high-resolution detail as the protected factual value;
- no learned-null rejection of protected fact;
- mirrored V120 flow-time and endpoint-head lifecycle;
- one Teacher identity fallback and raw posterior transport moments;
- loss support separated from online selector validity;
- exact V120 24-query, four-glimpse, N=49, 3x3 P1;
- no fake global-K axis on P1;
- corrected cross-camera geometry, active-parameter ownership and validation
  sample coverage.

Debt introduced or retained:

- dynamic P1 was merged into the factual owner;
- dynamic P1 self/FFN branches lacked one owner-total interpretation;
- static/dynamic optimizer and diagnostics were mixed;
- goal/history were pooled into public P1 conditions.

### 4.6 Schema25-26: the recovery island

Key commits:

```text
6a6c1bf  Schema25 preserve S object ownership
caa7e33  Schema26 restore G3 and S boundaries
```

Schema25's durable improvement is single S-owned typed relevance:

```text
G typed facts
  -> S [interval,K,type] relevance/value
  -> consumer-specific docks
```

Each semantic/appearance/geometry relevance has its own fixed-zero null rather
than one cross-type learned-null competition. CoarseAction and W no longer both
raw-read the same typed facts.

Schema26 adds several source-clean repairs:

- exact completed G3 rollout becomes transition source;
- the pseudo interval identity used to manufacture four copies is removed;
- typed common plus exact-zero interval correction replaces near-public scores;
- typed evidence no longer re-enters W through CoarseAction;
- unused `future_address` and a false proposal ablation are removed while RNG
  cadence is preserved.

Schema25 is the selected replay base. Schema26 is the first donor because it
contains several source-clean follow-up repairs; it is not a replacement base
or one indivisible cherry-pick. The recovery island is not a clean final graph.
Remaining debt includes:

- static/dynamic P1 ownership mixing;
- P2 interval-object plus type competition and active status value;
- W type reduction and full cross-object attention;
- predicted visibility controlling online selection;
- duplicate P3 factual/effect lanes;
- joint lane/basis/null bottom routing.

### 4.7 Schema27: first clear post-recovery break

Source: `2a0d3d1`.

Good repairs:

- reconstruction target independent of the online chart;
- typed W sidecars and zero-preserving public conditioning;
- removal of free interval identity authority;
- removal of the duplicate P3 factual lane.

Simultaneous hard cuts:

- real P2 candidates receive `-log(I*K)` before a zero-logit null, making
  neutral candidate-set versus null odds approximately 1:1;
- consequence is removed from precision, temporal and P3 query conditioning;
- a pre-W `DirectIntentFutureSupervisor` predicts the same future fields as W,
  making identity-like W a legal optimum;
- predicted visibility reduces future selector support.

Observed discontinuity from Schema26 to Schema27 includes P2 null increasing
from about `0.077` to `0.410`, W adjacent cosine from about `0.909` to `0.981`,
and P3 precision/state-change collapsing near zero. This is the first clear
post-island break.

### 4.8 Schema28-30: partial repair with new attenuation

Useful recoveries:

- G3 only corrects conditional K and typed evidence cannot resurrect absent
  physical support;
- real camera evidence controls camera reduction;
- S relevance becomes signed and exact-zero;
- learned interval identity is separated from observable innovation;
- per-type P2 routing and positive coordinate evidence return;
- predicted visibility no longer self-masks status;
- P3 temporal is again consequence-conditioned;
- high-resolution P1 innovation becomes precision source;
- P2 outer learned type selector is removed;
- reconstruction assignment is decoupled from learned null and must use
  exported K content.

New/remaining faults:

- Schema28 reintroduces typed evidence through CoarseAction and applies a
  cross-type softmax, duplicating the W ingress;
- fully independent type selectors remove legitimate shared-event context;
- Schema29 fuses complementary values by `/3`, weakening a lone active owner;
- Schema30 changes this to `/sqrt(3)`, which is only valid while exactly three
  comparable owners are active;
- association null is already interpreted as visibility/disappearance in
  Schema30.

### 4.9 Schema31: second independent breakpoint

Source: `c159651`.

Good structure:

- S/W common plus interval-residual factorization;
- single S->W typed ingress;
- separate P2 common and residual reads;
- exact-zero invalid support;
- owner-specific retention and gradient diagnostics.

Major distribution change:

- flat candidate-plus-null softmax becomes background-subtracted partial OT
  with dustbin row/column and fixed dustbin score.

The OT idea is not inherently a quota or hard gate. The failure came from
combining its much larger dustbin mass with the already-existing equation
`dustbin -> visibility/status -> selector validity`. Schema30 to Schema31 moves
Teacher dustbin from about `0.047` to `0.516` and semantic delta from about
`0.537` to `0.173`.

The Schema39 snapshot has removed the status/validity misuse but retains the partial
OT association. Whether its `~0.46` dustbin is calibrated ambiguity or
a matching shortcut remains unidentified.

### 4.10 Schema32: the strongest non-hardening rescue

Source: `4ac7e54`.

Durable principles:

- S is supervised only for observable state/intent; W is the sole future
  consequence owner;
- W common is a causal protected token;
- residual may read common, but residual cannot rewrite common;
- typed-by-base interaction is zero-preserving;
- canonical K content is actually exported;
- camera mixture is permutation invariant.

This is the clearest historical example of strong ownership without deleting
legal conditioning.

### 4.11 Schema33-35: factorization plus renewed hardening

Schema33 (`a2b5705`) correctly factorizes P2 as `P(interval) *
P(object | interval)`, but forces semantic/geometry/status to share one exact
interval posterior and forbids typed/W evidence from changing time.

Schema34 (`4363108`) restores typed S and W evidence to temporal scoring and
adds useful value contracts/type mapping/retention diagnostics, but averages
the three type scores back into one posterior. It also makes W common and
residual bilaterally independent. Ownership only required
`residual !-> common`; forbidding `common -> residual` removes useful context.

Schema35 (`03235d3`) contains many correct semantic repairs:

- content owns physical K+null;
- semantic/appearance only make bounded conditional-K corrections;
- geometry cannot vote physical identity;
- dustbin is association null, not physical disappearance;
- selector validity comes from current chart availability;
- uncertainty/reliability leave the online ABI;
- static fact and dynamic policy residual become distinct objects;
- temporal depends on W effect rather than full protected fact;
- camera-specific geometry and PSD covariance are restored.

It still inherits the shared temporal posterior, status activity, stale
fixed-owner fusion, bilateral W isolation, joint bottom competition and an
under-observed dynamic P1 producer.

### 4.12 Schema36: direct parent boundary, not one of the new replay targets

Source: `9e75d31`, `+1069/-643` across 19 files. Commit intent: repair P1 and
P2 ownership before the broader information-conservation work.

What the source actually established:

- static V120 P1 detail remained the protected factual source;
- dynamic P1 became a separately named policy residual and reached P2 only as
  part of its query;
- P2 had typed semantic/geometry reads, but camera value provenance was reduced
  before the final action-conditioned object read;
- P3 still exposed four optional lanes and bottom still used a joint
  lane-by-basis competition;
- status remained a trained W field despite having no useful P2 value
  consumer;
- the historical information audit made the producer/consumer losses explicit,
  but did not itself repair them.

Completed-log state:

```text
physical RMSE best/final              0.0801 / 0.0811
Teacher dustbin / delta / variation   0.4923 / 0.1716 / 0.0587
W interval variation / adjacent cos   0.0355 / 0.9193
static / dynamic P1 RMS               0.0328 / 0.5471  (16.7x)
P3 precision RMS                      0.1895
```

Schema36 therefore matters as the immediate parent: it shows that acceptable
aggregate RMSE can coexist with high Teacher dustbin, temporally weak W and a
very large dynamic/static P1 ratio. It is not a healthy template to recover
wholesale. The durable part is the static/dynamic semantic split; the exact
consumer algebra and magnitudes are not recovery targets.

### 4.13 Schema37: information identity is preserved, action closure is not

Source: `1b11bf5`, `+3459/-1767` across 33 files. Commit intent: preserve
information ownership across the complete top and bottom boundary.

#### Source changes

S and P1:

- S public state was factorized into common plus interval residual;
- typed nonlinear mean correction was retained instead of silently discarded;
- fixed `/4` K averaging became a conditional K read;
- `FactualIntentDock` kept public, goal, history and typed interval identities
  until the matching static-P1 role query;
- the exact V120 static reader remained 24 queries, N=49 candidates, four
  glimpses and a 3x3 microgrid.

W:

- W1 owned protected common plus the two near intervals; W2 read near and owned
  the two far intervals;
- appearance became a zero-preserving conditioner of semantic successor;
- online visibility/persistence status heads and their loss ownership were
  removed;
- semantic and geometry kept type identity, and geometry kept KxC camera
  structure with a legal PSD covariance;
- only the supervised `FutureObjectDynamics` crossed W->P.

P2, P3 and bottom:

- P2 selected semantic and camera-aware geometry separately and applied one
  shared complementary-value contract;
- typed effect and typed interaction survived consequence construction;
- P3 expanded to six optional lanes:
  `precision`, semantic/geometry effect, semantic/geometry temporal and
  state-change;
- each lane received its own `4 basis + zero-null` bottom read, so unrelated
  lanes no longer competed for one probability mass;
- protected consequence stayed outside optional null routing.

These are real information-conservation repairs in the Schema37 snapshot. They
enter the replay register as historical donor units, rather than as evidence
that the combined Schema37 graph must be inherited.

#### What remained structurally incomplete

- Teacher association stayed on the Schema31 partial-OT distribution. Schema37
  changed the consumer graph without restoring future-target bandwidth.
- Dynamic P1 still reached P2 query construction but did not have a legitimate
  P3 precision value consumer. Static precision therefore carried the optional
  precision lane alone.
- The six P3 lanes preserved identity but also created duplicate-owner
  suspicions: effect was already mandatory inside protected consequence, and
  factual detail was already mandatory inside protected base.
- Preserving S identities to P1 did not make S a strong W-time organizer. That
  requires a downstream action consumer, not another S-side interface name.
- Per-lane nulls fixed cross-lane competition, but they did not decide whether
  each optional value was unique or merely an alias of a protected carrier.

Completed-log state:

```text
physical RMSE e1 / best / final        0.1029 / 0.0815 / 0.0816
arm / gripper final                    0.0614 / 0.1549
Teacher dustbin / delta / variation    0.4848 / 0.2033 / 0.0447
W variation / adjacent cosine          0.0266 / 0.9679
W object-pair cosine                   0.0867
static / dynamic P1 RMS                0.0200 / 0.5038  (25.2x)
optional P3 precision RMS              0.0343
```

The important diagnosis is temporal publicization, not object collapse. W
preserved K identity, but its interval variation was only about 60% of an
already-low Teacher target and adjacent intervals were almost parallel. The
precision collapse is also source-explainable: dynamic P1 had no P3 value
terminal. Schema37 is therefore a strong axis/ownership donor, not a complete
behavioral branch.

### 4.14 Schema38: consumer paths are reconnected, but time still terminates too early

Source: `6bc6218`, `+3852/-927` across 32 files. Commit intent: close the
action-consumption paths left incomplete by Schema37.

#### Source changes

- P2 reconstructed each complete W interval field as `common + residual` and
  removed the separate mandatory-common/optional-residual consumers.
- S no longer cast an independent interval vote. Instead it conditionally
  modified the matching W key; W neutral therefore made the future effect
  exactly zero.
- Semantic and geometry retained separate type-local nulls.
- Geometry KxC evidence made a bounded, within-interval correction to semantic
  K addressing while retaining its own transport value.
- W covariance changed to a PSD parameterization that could approach a zero
  Teacher target instead of having an unreachable positive floor.
- Dynamic P1 obtained a fact-conditioned optional precision consumer without
  being written into protected factual base.
- G3 removed only a softmax-invariant scalar gauge; this was a forward/Jacobian
  equivalence cleanup rather than a capacity change.
- Finite gradient-spike attribution, checkpoint migration validation and
  epoch-tail gradient windows became explicit diagnostics.

#### What it fixed and what it did not

Schema38 is the first of the latest three versions that closes several actual
action consumers rather than only preserving interfaces. The source proves:

```text
W field -> action-conditioned P2 value
S -> conditioner of a W-owned key, not an independent future value
dynamic P1 -> fact-conditioned P3 precision
geometry -> both typed value and semantic spatial address
```

However, spatial selection and interval termination were still fused inside
the same P2 reader. A type-local learned null could reject the entire complete
W field, and K/camera/time were not terminated at independently justified
consumers. Teacher target algebra and association distribution were explicitly
left unchanged. The six-lane optional P3 surface also remained.

Completed-log state:

```text
physical RMSE e1 / final               0.0985 / 0.08175
arm / gripper final                    0.06182 / 0.15444
Teacher dustbin / delta / variation    0.4664 / 0.2302 / 0.0598
W variation / adjacent cosine          0.0362 / 0.9459
W object-pair cosine                  -0.0144
static / dynamic P1 RMS                0.0383 / 0.3567  (9.3x)
optional P3 precision RMS              0.1367
```

This log supports the intended local repairs: static detail, optional precision
and W interval variation all recover relative to Schema37, and the final
behavior returns close to V120. It does not prove healthy long-horizon
organization: W still tracks only about 61% of Teacher interval variation, the
Teacher itself remains low-bandwidth, and gripper remains worse than the
Schema25/26 performance island.

Historical donor assessment at the freeze point:

- complete-field consumption, S-as-conditioner, geometry-to-semantic
  addressing, reachable covariance and spike attribution are donor candidates;
- the fused P2 time terminal and the claim that optional P3 precision is the
  sole owner of dynamic policy precision require separate review;
- a recorded finite spike does not by itself establish that the repaired
  representation was learned.

### 4.15 Schema39: correct terminal ownership, weak terminal relation

Architectural source: `eac4916`, `+3686/-782` across 37 files. The following
commit `7cd69a7` changes only the mainline training path and tests; it is not a
second architectural schema.

#### Source changes

Schema39 splits the fused Schema38 reader into two legal terminal boundaries:

```text
P2 spatial selector:
    action-conditioned K/type/camera selection
    -> preserves interval and semantic/geometry axes

P3 physical interval terminal:
    four real intervals per type
    -> no learned null
    -> typed effect -> shared complementary contract -> consequence
```

The new `SelectedIntervalEvidence` validates that key, common value, residual
value and selected S context all retain `[B,T,Q,I,Z,H]`. S cannot create value,
support or a spatial posterior. The physical terminal selects common once and
adds the posterior-weighted residual once.

Dynamic P1 is also completed into a distinct
`protected_policy_precision = raw_dynamic + fact/action interaction` carrier.
It is read by transition and bottom through no-null protected paths, while the
static factual base remains separate. Optional null routing therefore cannot
delete either protected consequence or protected dynamic precision.

The observation/address path gains a bounded-Jacobian `safe_std`, FP32
log-domain conditional reads and explicit observable availability. Spike
attribution is split between flow and uncertainty channels only when that
specific six-channel head is the actual maximum-gradient owner.

These are source-certain repairs. They should be preserved.

#### Log evidence and remaining failure

The available Schema39 log is incomplete at epoch 7, so it cannot establish a
final eight-epoch result:

```text
physical RMSE e1 / best-so-far / e7    0.0984 / 0.0847 / 0.08551
arm / gripper e7                       0.06128 / 0.16929
Teacher dustbin / delta / variation    0.4577 / 0.2198 / 0.0466
W variation / adjacent cosine          0.0277 / 0.9677
W object-pair cosine                    0.2817
static / dynamic P1 RMS                0.0252 / 0.1885  (7.5x)
optional P3 precision RMS              0.0202
protected dynamic precision RMS        0.2312
```

The low optional precision value is not evidence that all precision vanished:
Schema39 deliberately moved dynamic precision into the protected `0.2312`
carrier. Comparing Schema38 optional precision `0.1367` directly with Schema39
optional precision `0.0202` would mix two different owners.

The physical terminal exposes the unresolved action closure directly:

```text
action score abs                         0.3809
S-conditioned intent score abs           0.0124
S effect on score scale                  about 3.3%
interval innovation retained ratio       0.3790
interval innovation cancelled fraction   0.6210
```

The implemented intent term is an elementwise relation equivalent to
`dot(action * tanh(S), W_key)`. Across hidden width it is naturally much smaller
than `dot(action, W_key)`. Meanwhile the four-way posterior averages an already
public W field, cancelling roughly 62% of its remaining interval residual.

This does not mean the no-null terminal is wrong, that every cancellation is
illegal, or that S is disconnected. It means the legal terminal is weak:

- Schema39 Teacher target variation is only about one third of Schema25/26;
- W again predicts about 60% of that already-small target variation;
- object separation remains present, so the primary collapse is temporal;
- S only weakly perturbs the physical time score;
- the measured action regression is concentrated in gripper timing rather than
  arm trajectory.

Two source-level duplicate-owner suspicions also remain outside Schema39's
locked scope:

- optional semantic/geometry effect lanes re-project an effect already carried
  by protected consequence;
- optional static precision re-projects factual detail already present in the
  protected base.

Historical donor assessment at the freeze point:

- the spatial/temporal terminal split, four physical intervals, no-null
  terminal, protected dynamic precision and bounded-Jacobian address path are
  donor candidates;
- the partial Schema39 result is not a performance anchor;
- gains, quotas, entropy targets and learned nulls are not supported remedies
  for the weak terminal observation;
- any replacement of the attenuating S-W relation depends on preserving
  Teacher bandwidth and unique P3 ownership, with S conditioning W-owned keys
  rather than creating an independent time vote.

### 4.16 Schema37-39 inheritance and donor matrix

| Boundary | Schema37 contribution | Schema38 contribution | Schema39 contribution | Historical donor assessment |
|---|---|---|---|---|
| S identity | Preserves public/goal/history/type to P1 | Conditions W key instead of voting independently | Preserves selected S context to physical terminal | Identity preservation is a donor candidate; the terminal relation needs separate review |
| W field | Typed K/camera and W1-near/W2-far ownership | Complete common+residual field reaches P2 | Same field stays interval-typed until P3 | Typed field ownership is a donor candidate; a second W carrier would duplicate ownership |
| P2 | Separate semantic/geometry and camera-aware reads | Full-field action consumer, type-local null | Spatial K/type/camera only; intervals survive | Spatial/temporal separation is a donor candidate; the Schema38 fused terminal is a distinct alternative |
| P3 | Six optional typed lanes | Dynamic P1 regains optional precision | No-null physical interval terminal and protected dynamic precision | Terminal/protected carriers are donor candidates; optional aliases require unique-owner review |
| Bottom | Per-lane 4+null, protected consequence outside null | Same | Adds protected dynamic precision outside null | Per-lane routing is a donor candidate; joint lane competition is a separate historical design |
| Teacher | Status misuse removed, partial OT retained | Deliberately unchanged | Deliberately unchanged | Unresolved; same-input backend attribution required |
| Numerical audit | Existing finite lifecycle | Precise spike attribution | Bounded address Jacobians and log-domain availability | Diagnostics are mechanically reusable if their observed tensors still exist |

The latest three versions therefore contain non-disposable evidence and donor
material. This statement does not require inheriting their combined graph.
Schema37 supplies axis identity, Schema38 supplies missing action consumers,
and Schema39 supplies a legally placed terminal; the replay plan decides which
parts survive when reconstructed from the Schema25 base.

## 5. Corrections to the supplied replay documents

The documents contain substantial useful source work. The following points
must be corrected before they are used as an implementation basis.

1. `ClearVLA_architecture_replay_ledger.md` is titled V120->Schema25 but in fact
   audits through Schema35. Its scope label is stale.
2. Schema25 is a good replay/reference root. Selecting it as the replay base
   does not authorize a blind whole-version cherry-pick and does not discard
   later repairs: later commits become donor material reviewed one semantic
   change at a time. The adopted base decision is recorded in
   `ARCHITECTURE_REPLAY_PLAN.md`.
3. V121 P2 only separates selectors; its final learned type softmax still makes
   semantic and geometry compete.
4. V121's three P3 lanes are a useful bank skeleton, but precision and temporal
   still read full protected consequence. “Only true innovations” is too broad.
5. V122's temporal operand is reusable; its K-centred precision implementation
   is not. It assumes a K-specific P1 axis and deletes common detail.
6. Schema26's structural repair is source-clean and its run is healthy, but
   absolute V120 performance claims remain subject to the missing comparable
   mainline normalizer fingerprint.
7. Partial OT must not be labelled simply good or bad. Its target distribution
   changed sharply; evidence at the Schema39 snapshot does not identify whether the dustbin is
   calibrated. It needs a same-input counterfactual, not a quota or blind
   removal.
8. Many recommended later repairs are present in the Schema39 snapshot: neutral
   status, static/dynamic split, camera P2, per-lane bottom reads and no-null
   interval terminal. They are ancestry, not automatic Schema39 TODO entries.
9. The Schema39 P3 snapshot still has six optional lanes. Optional semantic/geometry
   effect lanes re-project a typed effect already present in protected
   consequence. Optional precision re-projects `p1_factual_detail` while the
   same factual detail is already in protected base. These remain source-level
   duplicate-owner suspicions; temporal and state-change have clearer private
   operands.

## 6. Historical recovery inventory

### 6.1 Present in the Schema39 snapshot

These items had already been implemented by the Schema39 snapshot. They are
not automatic replay requirements and must not be copied as one bundle. The
replay decision register reviews each item against the reconstructed boundary:

- V120 dynamic lifecycle, retained bottom, CVAE/workspace, transition and
  execution body;
- exact V120 static P1 with 24 queries, N=49 and 3x3 values;
- exact G3 transition source and no fake P1 K axis;
- Schema25/26 single S-owned typed relevance and removal of duplicate typed W
  ingress;
- content-owned physical K plus bounded semantic/appearance conditional K;
- geometry restricted to physical support;
- one Teacher identity fallback and dustbin/status separation;
- loss support separated from selector validity;
- static factual P1 separated from dynamic policy residual;
- semantic and camera-aware geometry spatial P2;
- status removed from the online future value set;
- W common/near/far ownership and zero-preserving consequence;
- lane-local `4 basis + zero-null` optional bottom reads;
- protected consequence and protected dynamic precision outside learned null;
- physical four-interval no-null terminal.

### 6.2 Mechanically recoverable historical components

These have exact source forms, but still require target-boundary review before
editing:

1. **V121 three-lane bank skeleton**: protected base plus
   precision/temporal/state-change, with explicit factual/effect aliases
   removed.
2. **V122 consequence-innovation temporal operand**:
   `effect + interaction`, with exact-zero neutral-W semantics.
3. **Schema25/26 Teacher association backend** as a controlled counterfactual:
   flat candidate-plus-null row softmax, the Schema39 single identity fallback,
   Schema39 per-camera moments and no status/validity leakage.
4. **Schema32 one-way W causality**: common may condition residual; residual
   cannot rewrite common.
5. **Schema34 diagnostics**: type mapping, residual retention and cancellation
   metrics without the exact shared posterior.

### 6.3 Recovery requiring a rewrite rather than a cherry-pick

- **P2 complementary fusion**: retain separate semantic/geometry selection,
  add values, then apply one shared bounded contract. Do not restore a type
  softmax, `/3` or fixed `/sqrt(3)`.
- **P3 optional lanes**: lane count must follow unique owner values. Temporal
  and state-change have clear private operands. Precision is legal only if a
  P1-private residual distinct from protected fact is exposed. Optional effect
  aliases should not repeat protected consequence.
- **S terminal conditioning**: replace hidden-width-attenuated triple product
  only after preserving Teacher/W bandwidth. S may organize W-owned interval
  keys, but cannot create support, value or an independent time vote.
- **Teacher association**: compare the Schema39 partial OT and Schema25 row-softmax on
  identical cached inputs. Preserve candidate axes, identity fallback and
  physical moments; do not decide by matching a desired dustbin number.

### 6.4 Components that must not be recovered

- whole-version Schema25/V121/V122 cherry-picks;
- V120/Schema25 semantic/geometry/status type softmax;
- predicted visibility or dustbin as online selector validity;
- P3 factual/effect aliases alongside protected consequence;
- V120/Schema25 joint lane-by-basis plus one-null routing;
- V121 conflicting stable-successor/end-biased-delta targets;
- V122 K-centred precision on a non-K P1 output;
- Schema27 fixed candidate-set/null 1:1 calibration and hard consequence cuts;
- Schema27/31 pre-W duplicate future supervision;
- Schema28 CoarseAction typed bypass;
- Schema29/30 fixed complementary averaging constants;
- Schema33 exact shared type-time posterior;
- Schema34 bilateral common/residual isolation;
- quotas, entropy targets, forced nonzero paths, hard gates or synthetic
  gradients.

## 7. Unresolved causal chain in the Schema39 snapshot

The source and completed logs establish the following continuous observations:

1. Schema39 Teacher target bandwidth is much smaller than Schema25/26 before W acts.
2. W retains object separation but becomes highly common along the interval
   axis.
3. The Schema39 physical terminal gives S about 3% of the action-score scale.
4. Broad terminal selection cancels roughly 62% of remaining interval
   innovation.
5. Static high-resolution P1 is weaker than the performance anchors while a
   protected dynamic precision carrier remains active.
6. Global gradients are not weak; pressure has shifted away from S/W/static-P1
   and the bottom bridge.
7. The final measured regression is mainly physical gripper timing rather than
   arm trajectory.

What remains unproven:

- whether the Schema39 Teacher dustbin is calibrated ambiguity or a matching
  shortcut;
- how much of gripper regression is caused by Teacher/W/S versus P1 gradient
  ecology;
- whether a non-attenuating S-W relation helps after target bandwidth is
  restored;
- whether the Schema39 optional P3 precision/effect aliases are merely redundant or
  actively harmful;
- whether Schema39 late epoch behavior improves or rebounds, because its log is
  incomplete.

## 8. Execution authority

This ledger does not select the replay base, prescribe a composite bundle or
authorize a source edit. The adopted execution record is
`ARCHITECTURE_REPLAY_PLAN.md`. As of 2026-08-26 that plan selects Schema25 at
commit `6a6c1bf` as the replay base and treats every later change as donor
material. Exact source hunks are first grouped across versions by live semantic
boundary in `ARCHITECTURE_REPLAY_SOURCE_UNITS.md`; the plan then accepts,
softens, reimplements, defers or rejects those units and selects one coherent
candidate. Historical chronology remains a donor coordinate, not the execution
batch.

A base checkout is not a wholesale acceptance of the Schema25 graph, and it is
not a wholesale rejection of Schema26-39. Any later source edit must still
satisfy the workspace subsystem-familiarity rule for the complete boundary it
touches.
