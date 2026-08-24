# Current ClearVLA Architecture Contract

This file is the compact source of truth for the active graph. It describes
what the current source executes; ancestry and prospective designs do not
belong here. Unresolved problems are recorded in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md).

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        37
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
training:               single-stage end-to-end
behavioral reference:   V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
checkpoint policy:      fresh run; Schema36 resume and migration rejected
```

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
  entropy targets or artificial gradients.

## Executed graph

```text
RGB / cached DINO / raw pair / state and executed-action history / learned flow
  -> restored V120 observation compiler
  -> shared canvas seed
  -> G1 coarse grounding
  -> G2 N=49 candidate rematerialization
  -> G3 bounded grounding correction
  -> dense G3 chart + global K+null ObjectFactSet

T5 tokens + ordered observable history + ObjectFactSet
  -> stateless S organizer
  -> public common + interval residual
  -> typed K x semantic/appearance/geometry common + interval residual
  -> goal/history/type-preserving factual dock
  -> W and P2 intent docks

ObjectFactSet + S + clean CoarseAction
  -> W1 protected common and 4-8/8-16 innovations
  -> W2 reads near and writes only 16-32/32-48 innovations
  -> supervised FutureObjectDynamics

dense G3 chart + S factual dock + clean action bases
  -> static V120 P1 once
  -> FactualPrecisionDock [B,24,4,H]

noisy action + time + static P1
  -> dynamic P1 policy-query residual
  -> P2 semantic K read + geometry KxC read
  -> typed zero-preserving consequence
  -> six P3 lanes
  -> lane-local 4-basis+null bottom reads
  -> one shared optional-update contract
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
candidates; G3 is a bounded correction. The dense chart remains available to
static P1 and Teacher after the global K+null binder.

`ObjectFactSet` retains public content, K object innovations, semantic,
appearance and geometry fields, typed candidate maps, camera coordinates,
transport priors and chart availability. Public content plus object innovation
reconstructs object content. Semantic/appearance correct K identity;
geometry reweights physical spatial support.

`DenseFactChart.g3_public_scene_audit` is diagnostic only. Grounding
`decode_public_position` remains a reconstruction-only spatial nuisance and is
not exported downstream.

## S: stateless intent with preserved ownership

S reads the complete T5 token set, ordered state/executed-action history and K
object facts. It exposes four interval rows without treating them as labelled
phases.

The public boundary is:

```text
raw = public_intervals - interval_template
bounded = one shared smooth RMS contract over the complete [4,H] sample
public_common = mean_interval(bounded)
public_interval_residual = bounded - public_common
```

The shared sample scale preserves the common/residual relationship. Exact zero
stays zero and small values are not expanded. The final common/residual
decomposition is performed in FP32, including one residual-mean correction,
so BF16 rounding cannot reintroduce a hidden common row at the exported S
boundary.

For typed relevance, the nonlinear differential mean is transferred into the
typed common owner before the interval residual is centred. K values are read
with signed numerator and absolute relevance/validity mass:

```text
read = sum_K(signed_value) / clamp_min(sum_K(abs(relevance)*validity), 1)
```

A single strong K value is not divided by four and weak total mass is not
normalized into a strong value. Because projection and the one-sided contract
can create a new interval mean, the final typed policy residual is canonicalized
again after those transforms and that correction is transferred to its typed
common owner.

`FactualIntentDock` retains:

```text
public_interval_context   [B,4,H]
goal_interval_context     [B,4,H]
history_interval_context  [B,4,H]
typed_interval_context    [B,4,3,H]
```

Static P1 maps semantic, appearance and geometry only to the matching factual
role; coverage receives no typed context. K has already been conditionally read
by S and is not copied into P1.

## Teacher and W

Teacher matches current K slots to future DINO supports with reduced normalized
keys while retaining full-width DINO successor values. Dustbin remains an
identity fallback and association-uncertainty diagnostic. The exported targets
are:

```text
successor / semantic_delta  [B,4,K,D]
transport_mean              [B,4,K,C,2]
transport_covariance        [B,4,K,C,3]
```

The PSD covariance triple remains FP32 through W/Teacher/P2. Casting its three
entries independently to BF16 can make a valid 2x2 covariance acquire a
negative determinant and is forbidden.

Teacher association, successor, transport, covariance, dustbin and flow-prior
algebra are unchanged from Schema36. There are no visibility/persistence
targets or online status fields because no independent physical label exists.

`WConditionSources` preserves object, transport, S, coarse-action and goal
provenance until one declared generic-condition composition. This is a private
condition, not a second exported W value.

W's owned state is:

```text
protected_common     [B,K,3,H]
interval_innovation  [B,4,K,3,H]
type order            semantic / appearance / geometry
```

W1 writes common and the two near intervals. W2 reads completed near rows and
writes only the two far intervals; it cannot rewrite common or near. No far
zero-mean gauge remains, so two far intervals may carry a legitimate same-
direction change.

Appearance is a zero-preserving conditioner of semantic successor content:

```text
interaction = contract(semantic * tanh(variance_floor_norm(appearance)), .35)
semantic_state = semantic + interaction
```

Zero appearance leaves semantic bit-exact; zero semantic prevents appearance
from synthesizing a future value. Geometry remains a separate camera-aware
path. Only the supervised `FutureObjectDynamics` crosses W into P2.

W1 does not materialize a second decoded near-only field during ordinary
training. Its completed common/near owner state is consumed by W2 and decoded
once in the final four-interval field. A near-only field is built transiently
only when diagnostics are explicitly collected and is never stored in the W1
working state.

## Static/dynamic P1

Static P1 performs the complete V120 high-resolution conditional read once. Its
protected output is observation-owned and independent of noisy action/time.

Dynamic P1 retains V120 behavior but owns only a P2 query refinement:

```text
P2QueryDock:
  action_query
  factual_base
  policy_query_residual

combined = action_query + factual_base + policy_query_residual
```

The three sources stay named until P2's actual consumer. The dynamic residual
does not enter protected fact, P3 precision, transition or bottom value as a
second carrier.

## P2 and typed consequence

P2 has two complementary value owners: semantic and geometry.

- Semantic uses an action-conditioned K posterior.
- Geometry keeps transport values, covariance and camera support through an
  action-conditioned KxC joint posterior. Observable camera measure enters the
  posterior once. No valid camera support yields exact-zero geometry.
- Coordinate score chooses KxC and does not vote for a temporal interval.
- Each type has its own interval-plus-null read; types do not compete for one
  probability mass.

The two selected values share one `.35` contract:

```text
raw = semantic + geometry
fused, scale = contract(raw, .35)
semantic_effect = semantic * scale
geometry_effect = geometry * scale
semantic_effect + geometry_effect == fused
```

`ObjectConsequenceState` retains `effect_by_type` and
`interaction_by_type`. Their literal type sums form the physical effect and
interaction:

```text
protected_consequence = factual_base + sum(type effect) + sum(type interaction)
```

Neutral W gives exact-zero typed effect/interaction and recovers factual base.

## P3 and bottom ingress

P3 reuses existing parameters to expose six named lanes:

```text
precision
effect_semantic
effect_geometry
temporal_semantic
temporal_geometry
state_change
```

Precision reads static P1. Each effect lane reads only its typed consequence.
Each temporal lane requires its matching W effect, S temporal context and the
action query. Neutral W makes all four effect/temporal lanes exact zero.
State-change remains independent.

At bottom ingress, every lane independently invokes the shared `4 action
bases + null` reader. Lanes share parameters but never probability mass. Null
can reject only its own optional lane. The six routed values are summed, passed
through one shared `.35` contract and multiplied by the retained fixed `0.25`.
The optional-update bound is therefore `.35 * .25`; protected consequence
travels through its separate four-basis no-null reader.

## Objectives and backward ownership

Action flow matching remains the primary objective. Top objectives are:

- one grounding dense reconstruction loss;
- S adjacent observable-state increment loss;
- CoarseAction window-action loss;
- W semantic `0.55`, transport `0.15`, covariance `0.05` internal terms.

Deleted visibility/persistence weights are not redistributed. Reliability,
dustbin and availability diagnostics do not mask the future loss; detached
current support supplies its legal mask.

P2, consequence, P3 and bottom optional lanes learn through final
action/event/motion/execution losses. Tensor-boundary gradient RMS diagnostics
separately cover S public/typed, static/dynamic P1, W semantic/appearance/
geometry, P2 semantic/geometry and all six P3 lanes. They are observation-only
autograd hooks and return every gradient unchanged.

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

## Runtime, identity and release

- Manifest schema is 37; capability and topology names are unchanged.
- Schema36 exact resume, optimizer resume and bottom-only migration are
  rejected because bottom `source_key` and ingress probability structure
  changed.
- Parameter counts are measured per module at startup and written to run
  context; they are not hard-coded in this document.
- Teacher executes once per training batch and zero times in deployment.
- Teacher association entropy/reliability/effective-support summaries execute
  only on diagnostic batches; successor/transport/covariance target algebra is
  identical with diagnostics disabled.
- Five-step sampling does not rebuild observation, G, S, W or static P1.
- The lane-local bottom selector serializes only four basis keys, but consumes
  the historical 20-row initialization draw before retaining their prefix so
  removing dead rows does not silently reseed any live bottom parameter.
- Production batch-eight memory must remain below 22 GiB; the release target
  is no more than Schema36 plus 1.5 GiB and median throughput no slower than
  about 2.2 s/batch.

## Current unresolved boundaries

Only these remain active; details are in the issue ledger:

- dynamic P1's large query residual has a narrow P2-only consumer and needs
  matched causal attribution (IC-06);
- grounding reconstruction still owns a private position-only nuisance head
  unavailable downstream (IC-07);
- G3, learned-flow and Teacher association action value remain unidentified
  without association labels or matched intervention evidence (IC-11);
- the prior run's gripper/event mismatch remains an experimental risk until a
  fresh Schema37 run is observed (IC-12).

Do not solve these with gain, quota, hard gates, extra blocks or additional
loss weight. A fresh Schema37 smoke and full eight-epoch run must determine
whether the closed structure is sufficient before another architecture change.

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
runtime                 clearvla/mainline/runtime/, training/engine.py
```
