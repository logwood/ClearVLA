# Current ClearVLA Architecture Contract

This file is the compact source of truth for the active graph. It describes
what the current source executes; ancestry, experiment conclusions and
prospective designs do not belong here. Unresolved questions are recorded in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md).

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        38
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
training:               single-stage end-to-end
behavioral reference:   V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
performance references: V25 primary / V26 secondary
release state:          source implemented; fresh smoke and long-run evidence pending
checkpoint policy:      fresh formal run; Schema37 exact/optimizer resume rejected
```

An explicit Schema37-to-Schema38 bottom-only migration may be reported only
when the serialized bottom ABI is exactly unchanged. It is a migration tool,
not the formal comparison path.

Before modifying a subsystem, map its complete producer, axes, transforms,
consumers, loss, backward path, optimizer owner, runtime frequency,
serialization and diagnostics. Passing shapes, nonzero gradients and named
interfaces do not establish functional closure.

## Non-negotiable invariants

- Future supports enter only the detached FP32 Teacher plane. They cannot
  change the online cache or deployment action.
- Observation, G, S, W and static P1 are built once per observation.
- Dynamic P1, P2, P3, controlled transition and bottom run at five Euler nodes
  `0,.2,.4,.6,.8`; one `t=1` forward supplies endpoint heads without updating
  the action.
- Static P1 is the V120 reader: 24 queries, N=49 fine candidates, four factual
  glimpses and a 3x3 RGB/detail/coordinate microgrid.
- The retained bottom still contains Evidence MMDiT, CVAE/workspace, layer
  contracts, execution control and action/event/motion heads.
- Protected factual and consequence bases remain outside optional null routes.
- No owner axis may be reconstructed by pooling and later `expand`.
- Optional routes use ordinary autograd; there are no quotas, hard gates,
  entropy targets, artificial gradients or non-null rewards.
- Schema38 changes top consumers and diagnostics, not the serialized bottom
  body, bottom ingress algebra or deployment lifecycle.

## Executed graph

```text
RGB / cached DINO / raw pair / state and executed-action history / learned flow
  -> restored V120 observation compiler and shared canvas seed
  -> G1 coarse grounding
  -> G2 N=49 candidate rematerialization
  -> G3 bounded raw-logit correction
  -> dense G3 chart + global K+null ObjectFactSet

T5 tokens + ordered observable history + ObjectFactSet
  -> stateless S organizer
  -> public and typed common/interval owners
  -> factual, world and policy intent docks

ObjectFactSet + S + clean CoarseAction
  -> W1 protected common and 4-8/8-16 innovations
  -> W2 reads near and writes only 16-32/32-48 innovations
  -> one supervised four-interval FutureObjectDynamics

dense G3 chart + S factual dock + clean action bases
  -> static V120 P1 once
  -> FactualPrecisionDock [B,24,4,H]

noisy action + time + static P1
  -> dynamic P1 policy-query residual
  -> P2 semantic K / geometry KxC object read within each complete W interval
  -> S conditions the selected W interval key; S owns no independent time vote
  -> one interval-plus-null selector per type over complete W fields
  -> typed zero-preserving consequence
  -> fact-conditioned dynamic precision plus five other P3 lanes
  -> unchanged lane-local 4-basis+null bottom ingress
  -> protected consequence + controlled transition + V120 bottom
  -> physical action field and event/motion heads
```

Training adds one separate target graph:

```text
future DINO supports + current ObjectFactSet
  -> no-grad FP32 ObjectFutureTeacher
  -> detached semantic/transport/covariance targets
  -> W future losses only
```

## G and ObjectFactSet

G1-G3 retain camera, 8x8 chart and candidate axes. G2 rematerializes N=49
candidates. The dense chart remains available to static P1 and Teacher after
the global K+null binder.

Content alone establishes the K+null base competition. Semantic and appearance
can condition K identity without changing object-vs-null mass; geometry
reweights only physical K support and coordinates. Public content plus object
innovation reconstructs each exported object value.

G3 now applies exactly one bounded correction:

```text
raw = 0.5 * tanh(head)
posterior = softmax(log(parent) + raw)
```

The removed parent-weighted scalar subtraction was a softmax gauge and changed
neither value nor Jacobian. Diagnostics therefore report bounded raw RMS/max,
conditional-K span, posterior L1 and assignment change rather than a
gauge-inflated centered residual or unstable residual/margin ratio.

`DenseFactChart.g3_public_scene_audit` is diagnostic only. Grounding
`decode_public_position` remains a reconstruction-only spatial nuisance and is
not exported downstream.

## S: stateless intent, not an independent future selector

S reads the complete T5 token set, ordered state/executed-action history and K
object facts. It exposes four interval rows without treating them as labelled
phases. Its FP32-canonical public and typed common/residual decomposition,
conditional-K typed reads, goal/history/type-preserving factual dock and
adjacent observable-state increment supervision remain unchanged from
Schema37.

At P2, S no longer contributes an independent interval logit. For each type
and interval, P2 first selects an action-conditioned W key and then applies a
zero-preserving S condition:

```text
cS = bounded(public_S + matching_typed_S)
conditioned_W_key = selected_W_key + selected_W_key * tanh(cS)
interval_logit = similarity(action_query, conditioned_W_key)
```

Therefore neutral W makes both the key and effect exact zero regardless of S,
while neutral S reduces to the ordinary action-W read. The public/typed S
condition is diagnosed by its pre-tanh RMS, saturation fraction and matched
neutral-posterior L1, not by a standalone S vote.

## Teacher and W

Teacher matches current K slots to future DINO supports with reduced normalized
keys while retaining full-width DINO successor values. Dustbin remains an
identity fallback and association-uncertainty diagnostic. The exported targets
remain:

```text
successor / semantic_delta  [B,4,K,D]
transport_mean              [B,4,K,C,2]
transport_covariance        [B,4,K,C,3]
```

Teacher OT, dustbin, reliability, successor, transport and flow-prior algebra
are unchanged. There are no visibility/persistence targets or online status
fields because no independent physical label exists.

W1 owns common and the two near intervals. W2 reads completed near rows and
writes only the two far intervals; it cannot rewrite common or near. Appearance
is a zero-preserving conditioner of semantic successor content, while geometry
retains its real camera axis. Only the supervised `FutureObjectDynamics`
crosses W into P2.

W may retain common/residual internally and for supervision, but P2 reconstructs
and consumes the complete interval field:

```text
F[type, interval] = common[type] + interval_residual[type, interval]
```

There is no second common-only consumer. Common and temporal innovation are
accepted or rejected together by the matching type-local interval-plus-null
selector.

The diagnostic-only two-interval W1 decode runs under `no_grad` and is released
before W2. The trainable/exported four-interval field is decoded once by W2;
logging batches therefore do not retain a second decoder autograd graph.

Predicted covariance is an FP32 PSD triple with variances that can approach
zero while preserving the previous initial variance. P2's separate numerical
metric floor is not written back into W output or Teacher targets. Casting the
three PSD entries independently to BF16 remains forbidden.

## Static/dynamic P1

Static P1 performs the complete V120 high-resolution conditional read once. Its
protected output is observation-owned and independent of noisy action/time.

Dynamic P1 retains V120 behavior and owns one named policy-query residual:

```text
P2 query = action_query + factual_base + policy_query_residual
```

The residual remains outside protected fact, consequence, transition factual
source and bottom protected value. Schema38 restores one additional legal
consumer in P3 precision, conditioned by static precision rather than written
back into fact:

```text
s = precision_innovation(static_fact)
d = smooth_rms_contract(policy_query_residual, .35)
g = tanh(variance_floored_centered_norm(s, .25))
precision_source = s + g * d
```

If dynamic residual is zero, precision is the static path. If static precision
is zero, the dynamic interaction is exact zero.

## P2 and typed consequence

P2 has complementary semantic and geometry value owners.

- For every interval, semantic uses an action-conditioned K posterior.
- Geometry retains the action-conditioned KxC posterior, covariance and
  observable camera measure through its legal terminal.
- The camera-marginalized geometry evidence is valid-K centered and bounded,
  then corrects semantic K address only within the same interval. Missing or
  uniform geometry produces exact-zero correction.
- Geometry cannot vote directly for an interval and cannot scale or replace
  semantic value.
- S conditions only the already selected W key; it cannot select an object or
  interval without W.
- Semantic and geometry each own one equal four-interval-plus-null simplex.
  Null rejects only the matching optional type and is not biased by candidate
  count.

The two selected values still share one `.35` contract. Their literal typed
sum forms the physical effect; no fixed averaging or type competition is
introduced. Neutral W gives exact-zero effect and interaction, so protected
consequence recovers static factual base.

## P3 and bottom ingress

P3 exposes the unchanged six named lanes:

```text
precision
effect_semantic
effect_geometry
temporal_semantic
temporal_geometry
state_change
```

Precision now reads the fact-conditioned dynamic P1 interaction described
above. Effect lanes read only matching typed consequence. Temporal lanes require
matching W effect, S temporal context and action query; neutral W makes all four
effect/temporal lanes exact zero. State-change remains independent.

The Schema37 bottom ingress is unchanged. Every lane independently invokes the
shared `4 action bases + null` reader; lanes share parameters but never
probability mass. Six routed values are summed, passed through one shared `.35`
contract and multiplied by `0.25`. Protected consequence travels through its
separate four-basis no-null reader.

## Objectives and backward ownership

Action flow matching remains the primary objective. Top objectives remain:

- one grounding dense reconstruction loss;
- S adjacent observable-state increment loss;
- CoarseAction window-action loss;
- W semantic `0.55`, transport `0.15`, covariance `0.05` internal terms.

No external objective, gain or route quota is added by Schema38. P2,
consequence, P3 and bottom optional lanes learn through final
action/event/motion/execution losses.

The W semantic and transport output heads retain their exact-zero fresh
initialization. At that single neutral point, a zero optional value gives its
P2/S selector no action-only gradient; the existing W future objective first
opens a nonzero field. Once either W value is nonzero, the physical action loss
has an ordinary autograd path through W, the matching S condition, P2, P3 and
dynamic P1, including semantic, transport and covariance boundaries. No
artificial gradient is used to bypass this zero-value selector algebra.

Gradient processing remains:

```text
finite check
-> raw owner diagnostics
-> bottom decoder local clip
-> post-local diagnostics
-> global clip
-> post-global diagnostics
-> optimizer step
```

The compact raw preclip summary is now explicit about aggregation:

```text
gradient_window_preclip_l2_mean
gradient_window_preclip_l2_max
gradient_window_preclip_l2_current
gradient_window_preclip_l2_max_batch_offset / max_global_step
```

An epoch tail shorter than `log_every` is emitted as a separate
`window_boundary=epoch_tail` train row; its mean/max/current and owning global
step are not discarded.

A finite global norm above the audit threshold writes one read-only
`gradient_spike` event before clipping with the maximum-L2 and maximum-absolute
parameter name, role, optimizer group, shape and dtype. Ordinary batches do not
perform a parameter scan. This audit does not modify gradients or optimizer
state.

## Runtime, identity and release

- Manifest schema is 38; capability and topology names are unchanged.
- Schema37 exact and optimizer resume are rejected because the top consumer ABI
  changed. Formal experiments are fresh runs.
- Explicit bottom-only migration from Schema37 is allowed only by the migration
  tool when the serialized bottom ABI is identical. Before any live mutation,
  every bottom tensor must match key, shape and dtype, be non-complex and be
  finite. It is not used for the formal comparison.
- Parameter counts are measured per module at startup and written to run
  context; they are not hard-coded in this document.
- Teacher executes once per training batch and zero times in deployment.
- Five-step sampling does not rebuild observation, G, S, W or static P1.
- The recovery audit treats complete-field reconstruction error (`<=1e-6`) and
  the removed independent S interval vote (`==0`) as algebraic invariants;
  an S-neutral posterior delta may be nonzero because it measures a legal
  condition on the W-owned key.
- Production batch-eight memory must remain below 22 GiB; the release target is
  no more than Schema37 plus 1.5 GiB and median throughput no slower than about
  2.2 s/batch.

Source implementation and executable contract tests do not establish task
quality. Fresh BF16 smoke, epoch-one safety gates and the complete eight-epoch
comparison remain required before Schema38 can be called empirically healthy.

## Current unresolved boundaries

Only empirical or data-identifiability boundaries remain active; details are
in the issue ledger:

- grounding reconstruction still owns a private position-only nuisance head;
- learned flow and Teacher association remain unidentified without association
  labels or matched interventions;
- finite observation-side gradient spikes need attribution in a fresh run even
  though the logger can now identify the responsible parameter;
- gripper/precision quality and full-field W/P2 action value remain unverified
  until fresh smoke, causal diagnostics and the eight-epoch run.

Do not solve these with gain, quota, hard gates, extra blocks or additional
loss weight.

## Authoritative source map

```text
identity/config        clearvla/mainline/manifest.py, config.py
observation/G1-G3      clearvla/mainline/model/restored_observation.py
global K grounder      clearvla/mainline/model/grounding.py
S/CoarseAction         clearvla/mainline/model/intent.py
Teacher                clearvla/mainline/model/teacher.py
W                      clearvla/mainline/model/dynamics.py
static P1              clearvla/mainline/model/v120_p1.py, policy.py
dynamic P1/bottom      clearvla/mainline/model/restored_bottom.py
P2/consequence/P3      clearvla/mainline/model/compiler.py
transition             clearvla/mainline/model/transition.py
top orchestration      clearvla/mainline/model/top.py, policy.py
loss/optimizer         clearvla/mainline/training/losses.py, optimizer.py
gradient audit         clearvla/mainline/training/gradient_audit.py, engine.py
runtime                 clearvla/mainline/runtime/, train.py
```
