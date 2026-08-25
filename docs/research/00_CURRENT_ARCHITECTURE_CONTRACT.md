# Current ClearVLA Architecture Contract

This file is the compact source of truth for the graph executed by the active
mainline. Historical rationale and prospective designs do not belong here.
Unresolved empirical questions are recorded in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md).

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        39
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
training:               single-stage end-to-end
behavioral reference:   V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
performance references: V25 primary / V26 secondary
release state:          source implemented; fresh smoke and long-run evidence pending
checkpoint policy:      fresh formal run; Schema38 exact/optimizer/bottom migration rejected
```

Before editing a subsystem, map every producer, retained axis, transform,
consumer, loss, backward owner, optimizer group, runtime call, checkpoint field
and diagnostic. A valid shape, a nonzero gradient or a named interface is not
evidence of functional closure.

## Non-negotiable invariants

- Future supports exist only in the detached FP32 Teacher plane. They never
  alter the online cache or deployment action.
- Observation, G, S, W and static P1 are built once per observation.
- Dynamic P1, spatial P2, the physical interval terminal, consequence, P3,
  controlled transition and bottom run at Euler nodes `0,.2,.4,.6,.8`; the
  `t=1` forward supplies endpoint heads without updating action.
- Static P1 remains the V120 reader: 24 queries, N=49 candidates, four factual
  glimpses and one 3x3 RGB/detail/coordinate microgrid read.
- Evidence MMDiT, CVAE/workspace, layer contracts, execution control and
  action/event/motion heads remain present in the retained bottom.
- Protected factual consequence and protected dynamic policy precision never
  enter a learned-null competition.
- K, camera, type or interval axes may disappear only at their named real
  consumer; no axis may be pooled and later reconstructed with `expand`.
- Observable probability measures and their logs remain FP32. Model-width
  BF16 values may not be converted back into support probabilities.
- Optional routes use ordinary autograd. There are no quotas, hard gates,
  entropy targets, artificial gradients or non-null rewards.

## Executed graph

```text
RGB / cached DINO / raw pair / state + executed-action history / learned flow
  -> restored V120 observation compiler and shared canvas seed
  -> G1 coarse grounding
  -> G2 N=49 candidate rematerialization
  -> G3 bounded conditional-K correction
  -> dense chart + global K+null ObjectFactSet + FP32 log measures

T5 tokens + ordered observable history + ObjectFactSet
  -> stateless S organizer
  -> public and typed common/interval owners
  -> factual, world and policy intent docks

ObjectFactSet + S + one clean CoarseAction
  -> W1 common and 4-8/8-16 effect innovations
  -> W2 reads near and writes only 16-32/32-48 innovations
  -> one supervised four-interval FutureObjectDynamics

dense G3 chart + S factual dock + clean action bases
  -> static V120 P1 once
  -> FactualPrecisionDock [B,24,4,H]

noisy action + time + static P1
  -> dynamic P1 policy-query residual
  -> P2 semantic K / geometry KxC spatial selection for every interval
  -> SelectedIntervalEvidence [B,24,4basis,4interval,2type,H]
  -> P3 physical four-interval/no-null terminal, S conditions W-owned keys
  -> typed effect -> one shared semantic+geometry .35 contract
  -> zero-preserving consequence
  -> six optional P3 lanes + protected dynamic policy-precision carrier
  -> one bottom optional .35 x .25 budget plus protected consequence
  -> controlled transition + retained V120 bottom
  -> physical action field and endpoint event/motion heads
```

Training adds one separate target graph:

```text
future DINO supports + current ObjectFactSet
  -> no-grad FP32 ObjectFutureTeacher once
  -> detached semantic / transport / covariance targets
  -> W future objectives only
```

## G and observable log measures

G1-G3 retain camera, 8x8 chart and local-hypothesis axes. G2 rematerializes
N=49 candidates. Content establishes the only K+null base competition;
semantic and appearance provide bounded conditional-K corrections and geometry
reweights only inside physical K support. G3 preserves the parent
object-vs-null mass and adds one bounded conditional-K residual.

The complete owner chain is produced in FP32 log space. `DenseFactChart`
stores owner and typed log priors; `ObjectFactSet` stores
`log_chart_availability` and `log_camera_weight`. Observable reads compute:

```text
observed_log = owner_log + candidate_log_prior + log(observable_validity)
conditional  = softmax(observed_log over supported candidates)
availability = exp(logsumexp(observed_log) - logsumexp(base_log))
value        = availability * conditional_read
```

All-invalid rows return exact-zero probability, availability and value without
an all-`-inf` reduction. Iterative/final/typed reads, camera aggregation and
reverse chart lookup use this boundary; no second divide by a tiny mass is
legal. `camera_chart_availability`, joint `camera_weights` and both log fields
remain FP32 through W and P2.

Address variance uses the zero-preserving bounded-Jacobian map everywhere on
the live online path:

```text
safe_std(v, eps) = sqrt(v + eps^2) - eps
                 = v / (sqrt(v + eps^2) + eps)
```

G2 correction retains its deterministic-resolution scale as
`eps + safe_std`; variance zero therefore does not erase a legal coordinate
correction. The chart-derived `eps` and the maximum local derivative are
logged.

`DenseFactChart.g3_public_scene_audit` remains diagnostic only.
`decode_public_position` remains a reconstruction-only nuisance and is not
exported downstream.

## S and W

S reads the complete T5 token set, ordered state/executed-action history and K
facts. It exposes four interval rows without treating them as labelled stages.
Its public/typed common-residual decomposition, conditional-K typed reads,
goal/history/type-preserving factual dock and adjacent observable-state
increment objective remain unchanged.

S is metadata, not a future value owner. P2 spatial selection uses only action,
W keys, covariance-aware coordinates and producer-owned physical measures.
The same W spatial posterior selects matching S metadata. Only after that
selection may S condition the W-owned interval relation:

```text
a_i = bounded(action_query)
k_i = bounded(selected_W_key_i)
s_i = tanh(bounded(selected_S_context_i))
score_i = tau_action * <a_i,k_i> + tau_intent * <a_i*s_i,k_i>
```

S cannot create K/camera support, a spatial posterior, an effect value or an
independent interval vote. Neutral W gives exact-zero effect regardless of S.

W1 owns common and near intervals. W2 reads near and writes far intervals; it
cannot rewrite common or near. Appearance is a zero-preserving conditioner of
semantic successor content. Geometry retains the real camera axis and FP32 PSD
covariance. Only supervised `FutureObjectDynamics` crosses W into P2; no
generic hidden carrier crosses that boundary.

Teacher dustbin remains identity fallback and association-uncertainty audit.
There are no visibility/persistence targets or online status values. Teacher
reliability cannot mask loss or become a P2 value.

## Spatial P2 and physical interval terminal

P2 performs spatial selection independently for both value types while
retaining all four physical intervals:

```text
semantic candidates: [B,I,K]
geometry candidates: [B,I,K,C]
selected key/value:  [B,24,4basis,I,2type,H]
```

Semantic support comes from FP32 object chart availability. Geometry uses the
producer-owned FP32 camera log measure, explicit support, per-camera coordinate
and covariance. Geometry may provide a valid-K-centred correction to semantic
K address, but cannot vote for time or scale semantic value.

The only interval terminal occurs after spatial selection. For each type and
action query it normalizes over the four physical supported intervals, with no
learned null. Empty physical support returns exact zero. With
`F_i = common + residual_i`, it reads common and residual separately and forms:

```text
selected_value = common_once + sum_i posterior_i * residual_i
```

Semantic and geometry then share one `.35` contract on their physical sum;
the same scale is copied to both typed sidecars. Uniform time selection may
legitimately cancel a zero-sum interval residual. No retention quota or entropy
target is allowed.

## Static/dynamic P1, consequence and P3

Static P1 is observation-owned and independent of noisy action/time. Dynamic
P1 remains the live V120 policy write and has two legal consumers:

```text
P2 query = action_query + factual_base + policy_query_residual

protected_policy_precision = policy_query_residual
  + bias_free(tanh(action_projection + static_fact_gate)
              * policy_query_residual)
```

The protected policy-precision carrier is exact zero for zero dynamic input.
It is never written into factual base or consequence and never enters
transition context/memory. Transition reads one `.35` bounded view in its
action operand only. Bottom reads its four action bases once with the existing
no-null reader, then includes that result inside the single total optional
`.35 x .25` ingress budget. It has no independent amplitude budget.

Typed effect enters the zero-preserving consequence:

```text
interaction_by_type = bias_free(factual_projection(fact) * effect_by_type)
protected_consequence = factual_base + sum(effect_by_type)
                      + sum(interaction_by_type)
```

Neutral W returns protected consequence exactly to factual base. P3 retains six
optional named lanes: precision, semantic/geometry effect,
semantic/geometry temporal and state-change. Each lane owns an independent
shared-parameter `4 basis + zero-null` read. Protected consequence and
protected dynamic precision use no-null basis readers. Lane probabilities are
never pooled into one cross-lane simplex.

## Objectives and backward ownership

Action flow matching is primary. Top objectives remain one dense grounding
reconstruction, S adjacent observable-state increment, CoarseAction window
action, and W semantic `0.55`, transport `0.15`, covariance `0.05` internal
terms. Schema39 adds no objective, gain or route quota.

P2, consequence, P3 and bottom learn from action/event/motion/execution losses.
The exact-zero fresh W output heads first open through their existing future
objective; ordinary action gradients then traverse W, selected S metadata,
P2, the interval terminal, consequence, P3 and bottom. No synthetic gradient
is injected.

Gradient processing remains:

```text
finite check -> raw audit -> decoder local clip -> postlocal audit
-> global clip -> postglobal audit -> optimizer step
```

A finite preclip spike above the audit threshold emits a read-only parameter
report. If the owning parameter is a six-channel observation `delta_head`, the
report additionally separates channel L2 for flow `0:2` and uncertainty
`2:6`. Variance minima/gains and support measures are logged in the ordinary
forward. Per-loss VJPs, if needed, are offline replay only.

## Runtime, checkpoint and release

- Manifest schema is 39; capability and topology names are unchanged.
- Schema38 exact, optimizer and bottom-only migration are rejected because
  observation numerics, P2/P3 terminal and bottom ingress changed.
- Teacher runs once per training batch and zero times during deployment.
- Five-step sampling does not rebuild observation, G, S, W or static P1.
- Parameter counts and active source fingerprints are written at startup; they
  are not hard-coded here.
- Production batch-eight memory must stay below 22 GiB. The target is no more
  than Schema38 plus 1.5 GiB and median throughput no slower than 2.2 s/batch.

Source tests establish algebra and wiring, not task quality. Fresh BF16 smoke,
five-step deployment, epoch-one safety gates and a complete eight-epoch
comparison remain mandatory before Schema39 can be called empirically healthy.

## Current unresolved boundaries

- grounding reconstruction still owns a private position-only nuisance head;
- learned flow and Teacher association remain unidentified without independent
  association evidence;
- observation spike repairs are source-backed but recurrence/owner attribution
  require a fresh run;
- gripper/precision quality, interval differentiation and W action value remain
  empirical questions.

Do not solve these with gain, quota, hard gates, extra blocks or new loss
weight.

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
spatial P2/terminal    clearvla/mainline/model/effect_terminal.py
consequence/P3         clearvla/mainline/model/compiler.py
transition             clearvla/mainline/model/transition.py
top orchestration      clearvla/mainline/model/top.py, policy.py
loss/optimizer         clearvla/mainline/training/losses.py, optimizer.py
gradient audit         clearvla/mainline/training/gradient_audit.py, engine.py
runtime                 clearvla/mainline/runtime/, train.py
```
