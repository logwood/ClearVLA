# ClearVLA Schema39 diagnostic handoff

Status: frozen Schema39 source-and-log diagnostic snapshot from 2026-08-26.

This file records the checked-out Schema39 state at commit `7cd69a7` and the
diagnosis that existed at handoff time. It no longer owns the replay strategy or
the next execution step. The adopted Schema25-based replay method is defined by
`ARCHITECTURE_REPLAY_PLAN.md`; the historical lineage is defined by
`ARCHITECTURE_REPLAY_LEDGER.md`. The filename is retained to avoid needless
link churn.

This document is not an architecture contract and does not authorize a new
schema. It exists so a new task can independently reproduce the Schema39
diagnosis without inheriting the conversation's conclusions.

## 1. Repository identity and working-tree boundary

```text
workspace: C:\Users\ASUS\Desktop\clearvla_v42_1_cvae_prior_path_fix_with_scripts
branch:    codex/v94-latent-ownership-execution
HEAD:      7cd69a797c7dde2e9eea8a51014c93385478cea2
subject:   fix: update mainline training path
```

The model at this diagnostic snapshot is independent-mainline Schema39:

```text
capability:      object_intent_dynamics_323
manifest schema: 39
manifest hash:   a7701bbbab1c
source hash:     a40ea4781109
run git prefix:  7cd69a797c7d
parameters:      168,687,044
trainable:       152,292,723
```

There are no modified tracked files at handoff time. The worktree contains
untracked audit packages, logs and user documents. They were not added,
deleted, renamed or rewritten during this handoff. In particular, preserve:

```text
.audit/
ClearVLA_schema25_35_architecture_audit_and_plan.md
metrics.jsonl
schema25_s_owned_typed_b8.log
schema26_g3_s_boundary_b8.log
schema38_action_consumption_b8.log
schema39_action_closure_b8.log
v120_long.log
```

The exact V120 source snapshot is already local at:

```text
.audit/v120_exact_source_0b92d359/
```

Relevant committed ancestry is available without reconstructing it from old
conversation text:

```text
0b92d35 checkpoint: integrate V120 object intent dynamics
6a6c1bf fix: preserve S object ownership in schema 25
caa7e33 fix: restore G3 and S boundaries in schema 26
6bc6218 fix: close schema38 action consumption paths
eac4916 fix: close schema39 action and numerical paths
7cd69a7 fix: update mainline training path
```

## 2. Authoritative documents and source map

Read before any edit:

- `docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md`
- `docs/research/CURRENT_MAINLINE_ISSUES.md`
- `.agents/skills/audit-clearvla-logs/SKILL.md`
- `.agents/skills/audit-clearvla-logs/references/metric-catalog.md`
- `.agents/skills/audit-clearvla-logs/references/source-map.md`

Active source boundaries:

```text
observation/G1-G3      clearvla/mainline/model/restored_observation.py
global K grounder      clearvla/mainline/model/grounding.py
S/CoarseAction         clearvla/mainline/model/intent.py
Teacher                clearvla/mainline/model/teacher.py
W                      clearvla/mainline/model/dynamics.py
static P1              clearvla/mainline/model/v120_p1.py, policy.py
dynamic P1/bottom      clearvla/mainline/model/restored_bottom.py
spatial P2/terminal    clearvla/mainline/model/effect_terminal.py
consequence/P3         clearvla/mainline/model/compiler.py
transition             clearvla/mainline/model/transition.py
top orchestration      clearvla/mainline/model/top.py, policy.py
loss/optimizer         clearvla/mainline/training/losses.py, optimizer.py
runtime                clearvla/mainline/runtime/, train.py
```

## 3. Comparison contract

Use three different anchors for three different questions:

- Schema25 and Schema26 are the performance anchors. They currently provide
  the strongest physical/gripper results among the independent-mainline runs
  being compared.
- V120 is the old-main-path health and behavior anchor. It is not the parent of
  Schema39 and its metric vocabulary is not always identical.
- Schema38 is only the direct parent/increment comparison for Schema39. It is
  not a correctness or performance target.

Do not directly compare V120 `first_rmse`/`tail_rmse` with Schema25+ first/tail
metrics. V120's compact log reports those in physical space; the independent
mainline reports normalized first/tail values.

## 4. Run coverage

```text
v120_long.log                       8 completed epochs
schema25_s_owned_typed_b8.log       8 completed epochs
schema26_g3_s_boundary_b8.log       7 completed epochs
schema38_action_consumption_b8.log  8 completed epochs
schema39_action_closure_b8.log      7 completed epochs; epoch 8 reaches batch 1620
```

Schema39 is therefore not a complete eight-epoch result at this freeze point.
Do not call its epoch-7 row the final result.

Reproduce the normalized audit with:

```powershell
uv run python -m clearvla.tools.audit_policy_logs `
  v120_long.log `
  schema25_s_owned_typed_b8.log `
  schema26_g3_s_boundary_b8.log `
  schema38_action_consumption_b8.log `
  schema39_action_closure_b8.log `
  --format json
```

## 5. Performance result: implementation observation, not diagnosis

| Run and point | Physical RMSE | Arm physical | Gripper physical | Normalized RMSE |
|---|---:|---:|---:|---:|
| V120 best, epoch 7 | 0.07931 | 0.06113 | 0.14699 | not emitted in the same schema |
| V120 final, epoch 8 | 0.08145 | 0.06325 | 0.14979 | not emitted in the same schema |
| Schema25 best, epoch 6 | 0.0772445 | 0.061862 | 0.137133 | 0.207398 |
| Schema26 best, epoch 6 | 0.0789336 | 0.0620184 | 0.143303 | 0.207448 |
| Schema38 epoch 8 | 0.0817535 | 0.0618236 | 0.154442 | 0.212467 |
| Schema39 epoch 7 | 0.0855129 | 0.0612758 | 0.169289 | 0.209382 |

Schema39 epoch 7 is 10.7% worse in physical RMSE than Schema25's best point
and 8.3% worse than Schema26's best point. The gap is concentrated in the
gripper: 23.5% worse than Schema25 and 18.1% worse than Schema26, while arm is
slightly better. Its normalized aggregate is already close to the performance
anchors. The active failure is therefore not a generic arm-trajectory failure;
it is primarily physical gripper calibration/timing.

Schema39 decoded-gripper F1 at epoch 7 is 0.3493. For reference, Schema25
epoch 7 is about 0.415 and Schema26 epoch 7 about 0.368. Keep decoded gripper,
the event head and the motion head separate in every future report.

## 6. Source-backed end-to-end diagnosis

### 6.1 Teacher temporal target bandwidth is already much smaller upstream

Latest completed validation rows:

| Metric | Schema25 | Schema26 | Schema39 |
|---|---:|---:|---:|
| Teacher dustbin/null probability | 0.0421 | 0.0509 | 0.4729 |
| Teacher semantic-delta RMS | 0.3499 | 0.3527 | 0.2061 |
| Teacher interval variation | 0.1343 | 0.1338 | 0.0530 |

This is a 9--11x increase in dustbin mass, about a 41% reduction in semantic
delta and about a 60% reduction in interval variation before W predicts
anything.

Current source facts:

- `teacher.py:273-304` constructs the association logit from background-
  subtracted semantic/appearance evidence, geometry, camera prior and
  `-log(rows*columns)`, then compares it with a fixed zero-score dustbin in a
  partial OT assignment.
- `teacher.py:379-384` uses the dustbin as an identity fallback:
  `successor = matched + dustbin * current_reference`.
- Reliability is diagnostic and does not shrink the physical target a second
  time. Visibility/status is not the cause in Schema39.

What is proven: the target presented to W has much less temporal bandwidth
than the performance anchors, and current dustbin algebra is the exact source
boundary at which this can happen.

What is not proven: whether 0.47 is calibrated ambiguity in this data or a
matching shortcut caused by current association calibration. There is no
independent association label. Do not impose a dustbin quota, force nonzero
flow, or use reliability as a loss mask. The next source audit must diff
current Teacher association against commits `6a6c1bf` and `caa7e33` on the
same axes before choosing a repair.

### 6.2 W is temporally public, not globally/object-wise public

Latest completed validation rows:

| Metric | Schema25 | Schema26 | Schema39 |
|---|---:|---:|---:|
| W predicted interval variation | 0.0857 | 0.0839 | 0.0265 |
| W adjacent-interval cosine | 0.9087 | 0.9087 | 0.9604 |
| W object-pair cosine | 0.4866 | 0.5549 | 0.2817 |

Lower object-pair cosine means Schema39 separates objects better. Do not report
this as global W publicization. The severe problem is specifically the interval
axis: it receives a weaker Teacher target and produces fields that are much
more alike across time.

At the Schema39 consumer boundary:

```text
W prediction common effect RMS                 0.145215
W prediction interval residual RMS             0.035673
P2 selected common RMS                         0.050955
P2 selected interval innovation RMS            0.021381
P3 interval innovation retained ratio          0.358160
P3 interval innovation cancelled fraction      0.641840
P3 interval innovation/common ratio             0.136923
```

The continuous causal chain supported by the log is:

```text
high Teacher dustbin
  -> smaller Teacher interval target
  -> smaller W interval innovation
  -> broad physical-interval posterior cancels another 64% of innovation
```

This does not prove that every cancellation is illegal: a real broad posterior
may legitimately average zero-sum interval residuals. It does prove that the
current action consumer sees little distinct temporal evidence.

### 6.3 S is connected but its terminal influence is structurally attenuated

In `effect_terminal.py:495-510`:

```text
action_score = dot(action, W_key)
intent_score = dot(action * tanh(S_context), W_key)
```

All three operands are bounded/normalized in hidden width 512. The elementwise
triple product introduces a dimension-dependent attenuation. Schema39 epoch 7
confirms the scale separation:

```text
P3 interval action score abs                 0.424805
P3 interval intent score abs                 0.013201
S-neutral posterior L1                       0.0039
```

The intent term is about 32x smaller. This is not a disconnected S path, and a
nonzero gradient is not closure. It is a source-backed weak-leverage relation.
Older V120/Schema25/Schema26 intent-score metrics used a different terminal
definition, so their absolute values are directional health evidence only;
they were comparable to their content/action terms rather than approximately
3% of them.

Do not repair this with a learned gain, independent S time vote, entropy target
or quota. Any replacement must retain all of these invariants:

- W=0 gives exact-zero future value;
- S cannot create support, value or an independent future field;
- the four physical intervals remain the only temporal candidates;
- the relation is bounded without hidden-width attenuation.

### 6.4 Static P1 detail is weak while protected dynamic precision survives

Latest completed validation rows:

```text
                           Schema25   Schema26   Schema39
static factual/detail RMS   0.04023    0.04599    0.02275
dynamic P1 residual RMS     0.25207    0.29319    0.18499
```

Schema39 static factual detail is 43--51% below the performance anchors. Its
microgrid value RMS is 0.02084 and spatial variation 0.01507; V120 reports
0.083 and 0.041 respectively, although the older metric implementation makes
that comparison directional rather than bit-exact.

Do not confuse two Schema39 precision metrics:

```text
optional P3 precision lane RMS               0.01814
protected no-null policy-precision RMS        0.22928
```

The protected dynamic precision carrier is alive. In
`compiler.py:1469-1486`, raw dynamic P1 is added to one bias-free action/fact
interaction. In `transition.py:193-199`, one bounded view enters the action
operand. This is a legal owned path, not automatically an illegal bypass.
However, it provides an easier action-conditioned carrier while static P1
receives weaker learning pressure. The log establishes coexistence, not yet
causality.

### 6.5 Backward pressure shifted away from the intended top chain

Tail-median raw gradient L2:

| Owner | Schema25 | Schema26 | Schema39 |
|---|---:|---:|---:|
| intent/S | 0.04888 | 0.05496 | 0.01557 |
| dynamics/W | 0.06731 | 0.06598 | 0.03821 |
| P1 factual | 0.03501 | 0.04230 | 0.02538 |
| P2 effect reader | 0.02379 | 0.02896 | 0.02102 |
| bottom policy bridge | 0.00604 | 0.00803 | 0.00233 |
| global | 0.36445 | 0.38635 | 0.40730 |

Global optimization is not weak. Pressure has specifically shifted away from
S/W/P1/bridge. This supports the forward diagnosis; it does not justify
synthetic gradients or per-owner clipping.

### 6.6 G is not the primary performance gap against Schema25/26

Schema39's G object-pair cosine is approximately the same as Schema25/26, and
its chart overlap is lower/better. Compared with V120, object content remains
more common and G3 correction remains weak, so G is still a health debt. It is
not currently the strongest explanation for the gripper regression. Do not
rebuild G before resolving the upstream Teacher and downstream consumer chain.

### 6.7 Numerical spikes improved but are not closed

```text
Schema38: 93 finite spike reports, maximum global preclip 1763.72
Schema39: 77 finite spike reports so far, maximum global preclip 670.011
```

Schema39 is incomplete, so counts are not exposure-normalized final evidence.
There is no NaN/Inf failure in the supplied log. The remaining spikes are still
concentrated in observation-side target-DINO/flow/raw-flow ownership. Keep
IC-16 open; do not add another local clip without deterministic replay
attribution.

### 6.8 Historical origin before Schema25

The current independent-mainline lineage did not begin from a clean V120
mechanical extraction. There are three earlier layers that must remain
separate in future diagnosis.

First, V120 (`0b92d35`) was a useful behavior anchor but not a perfectly
healthy architecture. Its final tail medians already show weak language/time
organization and moderately public W fields:

```text
G object-content pair cosine          0.496
G3 parent correction L1               0.0129
S goal attention entropy              0.9915
S interval variation                  0.1445
S temporal variation                  0.0910
W1/W2 adjacent-interval cosine        0.957 / 0.915
W1/W2 object-pair cosine              0.441 / 0.451
P2 effect RMS                         0.0975
P3 precision RMS                      0.2540
best/final physical RMSE              0.07931 / 0.08145
```

These are inherited health debts, especially weak goal selectivity, small G3
correction and temporal common mode. They are not equivalent to the later
complete object collapse.

Second, V121 (`480f71c`) was the first pre-mainline semantic break. It changed
the complete G/S/Teacher/W/P boundary at once (about 3,029 insertions and 1,148
deletions across 20 files). The exact source diff introduces two different
interval aggregations:

```text
successor_content <- reliability-weighted stable successor
semantic_delta    <- end-biased successor - current reference
```

while the online W field algebra requires
`successor_content == current_reference + semantic_delta`. The two supervised
fields therefore no longer have one common exact optimum. The same change also
replaced aggregate P1/P3 ownership with a K-object factual dock and removed the
former factual/effect optional lanes in the same cutover. There is no complete
V121 log in the local handoff set, so its behavioral magnitude must not be
invented; the supervision conflict itself is source-certain.

Third, V122 (`ced6f23`) tightened the identity/innovation-only interpretation
instead of first isolating the V121 target conflict. Its completed log is
already a failed behavior baseline relative to V120:

```text
                                      V120       V122
best physical RMSE                    0.07931    0.08910
final physical RMSE                   0.08145    0.09110
G object-content pair cosine          0.496      0.713
S interval variation                  0.1445     0.0315
S temporal variation                  0.0910     0.0060
W2 object-pair cosine                 0.451      0.592
```

V122's Teacher semantic target RMS remains close to V120 (`0.382` versus
`0.3805`), so this particular failure is not the later low-bandwidth Teacher
failure. It is mainly an object/S ownership and consumer-geometry regression,
with the incompatible V121 target semantics still present.

The decisive lineage reset is commit `91a4237`, which added the independent
mainline as 18,758 new lines across 59 files. Its own README explicitly says
the extraction is not a blind bit-for-bit copy. It reimplemented the outer
`G -> S -> Teacher -> W -> P1/P2/P3 -> typed ingress` chain while initially
using a new bottom/transition implementation as well. The first completed
independent-mainline log proves an initialization/structure failure rather
than a late-training drift:

```text
batch 20 G object-content pair cosine       1.000
batch 20 W object-pair cosine               1.000
epoch 8 G object-content pair cosine        1.000
epoch 8 W object-pair cosine                1.000
epoch 8 W adjacent-interval cosine          0.969764
best/final physical RMSE                    0.091067 / 0.091271
```

The subsequent commit history is therefore a recovery sequence, not a clean
enhancement sequence:

```text
4cfe788  restore V120 core in Schema20
2d0a84c  recover V120 behavior in Schema21
5b80251  restore V120 training semantics in Schema23
ec84c9e  restore V120 P1 grounding in Schema24
32d969f  correct V120 geometry/validation behavior
6a6c1bf  preserve S object ownership in Schema25
```

Corrected Schema24 and Schema25 formed a local recovery island. Schema27 is
the first clear break after that island; it is not the original source of the
independent-mainline debt.

### 6.9 Historical onset after Schema25

The first clear post-Schema25 structural break is Schema27, commit `2a0d3d1`.
Schema26 is a small performance retreat but retains the healthy Teacher/W/P1
regime. The Schema26 -> Schema27 change is discontinuous:

```text
                                      Schema26   Schema27
best physical RMSE                     0.07893    0.08211
best gripper physical RMSE             0.14330    0.14850
P2 null mass                           0.07657    0.40994
W adjacent-interval cosine             0.90869    0.98072
W predicted interval variation         0.08389    0.04823
P3 precision RMS                       0.28710    0.00019
P3 state-change RMS                    0.07885    0.00011
```

The historical source diff identifies three simultaneous Schema27 semantic
changes rather than a data/config difference:

- P2 subtracts `log(intervals * objects)` from every real-candidate logit
  before adding one zero-logit null. Equal evidence therefore becomes 1:1
  candidate-set versus null odds instead of allowing the real set to accumulate
  candidate mass.
- W removes its explicit interval identity.
- P3 precision is replaced by the basis-centred residual
  `p1_fact - mean_basis(p1_fact)`, deleting common precision content.

Schema28/29 amplify the same break: P2 null reaches about 0.55/0.54 and W
adjacent cosine about 0.994/0.995. Schema30 partially repairs local symptoms
but does not return to the Schema25/26 action regime.

Schema31, commit `c159651`, is a second independent breakpoint and the direct
ancestor of the current Teacher problem. It replaces flat candidate-plus-null
softmax with background-subtracted partial OT. Across Schema30 -> Schema31:

```text
Teacher dustbin/null                    0.04691    0.51581
Teacher semantic-delta RMS              0.53664    0.17320
```

The later extreme static/dynamic P1 imbalance first becomes explicit in
Schema35 (`0.0167` static versus `0.8542` dynamic, about 51x). Schema37 then
marks another consumer-side contraction of optional P3 future lanes. These are
later compounding failures, not the first onset.

## 7. What Schema39 actually repaired

Relative to Schema38, Schema39 did produce source-backed improvements:

- the P2/P3 physical interval terminal has no learned null;
- protected dynamic policy precision is no-null and cannot be removed by an
  optional lane;
- interval residual survival improved materially relative to the parent;
- finite observation spikes are less frequent and smaller in the current
  partial exposure;
- parameter count is unchanged, so these are wiring/algebra changes rather
  than hidden capacity additions.

These repairs are insufficient for release because gripper performance,
Teacher bandwidth, S leverage and static detail remain unhealthy.

## 8. Historical repair lesson

Similar local paths were modified repeatedly in Schema25--39: S typed
ownership, G3-to-S, null/common-residual algebra, W2/P2 factorization,
Teacher dustbin/status semantics, P1 static/dynamic ownership, mandatory W
consequence and the temporal null terminal. The repeated failure was not that
none of those edits existed. It was that the complete chain below was never
healthy at the same time:

```text
Teacher target
  -> W interval prediction
  -> S-conditioned temporal selection
  -> P2/P3 consequence
  -> bottom bridge
  -> decoded physical gripper
```

The next task must not reopen one local interface in isolation.

## 9. Unresolved assumptions that block a responsible implementation plan

1. Why did Teacher dustbin rise from about 0.04--0.05 to about 0.47? The source
   boundary is known; calibrated ambiguity versus matching shortcut is not.
2. How much of gripper regression is causally attributable to Teacher/W/S,
   versus the static/dynamic P1 gradient ecology? Current correlations do not
   partition that effect.
3. Would a bounded non-attenuating S-W relation improve interval selection, or
   merely amplify weak Teacher distinctions? It must be evaluated only after
   preserving upstream target bandwidth.
4. Is the weak static P1 value caused by reduced action-gradient demand or by
   a producer-side adaptation mismatch? The current forward path is active;
   no hard clamp has been identified.
5. Schema39 epoch 8 and complete late-rebound behavior are not yet available.

## 10. Historical Schema39-forward audit queue

The following was the required next audit under the former Schema39-forward
strategy. It remains useful diagnostic evidence, but it is not a prerequisite
for selecting Schema25 as the replay base and must not override the
chronological donor procedure in `ARCHITECTURE_REPLAY_PLAN.md`.

1. Finish or ingest the complete Schema39 log; rerun the five-run audit.
2. Diff Teacher association end to end between current source, `6a6c1bf` and
   `caa7e33`: input keys, candidate measure, background subtraction, camera
   prior, spatial normalization, dustbin score, OT marginals, identity fallback
   and ordinary loss consumer.
3. Write a producer-to-consumer table for the current Teacher -> W -> P3
   interval path with exact axes, dtype, zero semantics, RMS and Jacobian at
   every boundary.
4. Independently trace the consumer back to decoded gripper and its ordinary
   action/event gradients. Do not stop at consequence or bottom ingress.
5. Trace static and dynamic P1 separately to both transition and bottom. Prove
   whether either is an alternate carrier before calling it a bypass.
6. Only then form one integrated repair proposal. Have a separate reviewer
   audit source completeness, information conservation and backward ownership
   before editing.

## 11. Explicitly forbidden shortcuts

- no dustbin, route or interval quota;
- no forced nonzero flow;
- no S/W/P1 learned gain introduced to make a metric larger;
- no entropy target or hard temporal gate;
- no synthetic/manual gradient;
- no new auxiliary loss before the existing target and consumer are correct;
- no weakening or rebuilding of the retained bottom, CVAE/workspace,
  transition, execution or static high-resolution reader;
- no use of Schema38 as a correctness anchor;
- no direct comparison of physical V120 first/tail with normalized mainline
  first/tail;
- no claim that object separation is worse when the actual defect is temporal
  separation.

## 12. Schema39 snapshot conclusion

Schema39 is not a release candidate. It repaired a local consumer null path and
some numerical singularities but did not recover the performance-anchor action
path. The strongest current source-and-log explanation is an accumulated
bandwidth failure: Teacher temporal evidence is already weak, W makes it more
common, the P3 interval relation gives S negligible leverage and averages much
of the remaining residual, while static P1 receives less pressure and the
physical gripper channel carries the final regression.

That explanation is better supported than any single-module story, but the
Teacher dustbin cause and causal attribution to decoded gripper remain open.
The next task should audit those boundaries before writing code.
