# R1c / S-01,S-02 typed ingress and decomposition worksheet

Status: implemented and statically closed on 2026-08-26; no training run.

This worksheet is the mandatory producer-to-consumer and
consumer-to-producer review for R1c. It authorizes only S-01 (one typed W
ingress and no typed CoarseAction re-entry), S-02 (lossless interval common
plus residual coordinates), and the S-03 no-future-owner guard selected in
`ARCHITECTURE_REPLAY_SOURCE_UNITS.md`.

It does not authorize the Schema26 common/differential scoring floor, the
Schema31 independently scored common/residual selector, public/object content
factorization, a W block rewrite, a future-field S supervisor, a new loss, or
checkpoint migration. The current architecture contract was reread before
this review.

## 1. Source identity and donor disposition

| Role | Commit/blob | Disposition |
|---|---|---|
| R1b active intent | `b48fa14:intent.py` / `d829961a5685cc11e86c4c81dec4ec92235e70de` | base scoring and producers |
| R1b active types | `b48fa14:types.py` / `679e238b579265931755b56147ba4062f23ba986` | base ABI to repair |
| R1b active W | `b48fa14:dynamics.py` / `3fba484b44d905ef8e3d6ce415bb5b1e20fe150` | consumer boundary only |
| Schema26 S-01 donor | `caa7e33:intent.py` / `200bd610aafdc87f4bb5af877d82bc2b38eda423` | keep removal of typed CoarseAction input; reject new scoring mechanics |
| Schema26 S-01 types | `caa7e33:types.py` / `df2a45a2f3cda51ef608ec9e7be4ba1e5f6ce5` | keep typed-free ActionIntentDock |
| Schema31 S-02 intent | `c159651:intent.py` / `3e05e357a6ab374c20089e6ed05e2f78dfabbf56` | keep common/residual ownership purpose, not its new selector |
| Schema31 S-02 types | `c159651:types.py` / `e7336783f4811390c2bce336a28f4fca6c83303` | narrow to the existing Schema25 typed value |

The active source has one ownership defect and one missing coordinate:

1. `_typed_relevance` produces one typed value, but
   `typed_action_components` enters `ActionIntentDock`, changes
   `CoarseActionIntentState.tokens`, and reaches W through `action.tokens`.
   The original `[B,I,K,type,R]` value also reaches W directly through
   `WorldIntentDock`. W therefore receives the same S-owned evidence through
   a direct typed path and a K-reduced action alias.
2. The direct WorldIntentDock retains only the full interval-valued tensor.
   W cannot identify a protected S common coordinate separately from genuine
   interval innovations without recomputing it after another owner has
   processed the value.

## 2. Complete active forward dataflow

### 2.1 Producers before typed relevance

`StatelessObjectIntentOrganizer.forward` runs once per observation:

1. full T5 goal tokens are projected and read by four learned goal queries;
2. observable state history, current state and executed-action history form a
   causal history memory;
3. `ObjectFactSet.content` forms public object tokens;
4. the four interval identities read protected goal, history and object
   memories, then `interval_self` produces
   `public_interval_carrier [B,4,H]`;
5. no future support, Teacher value, noisy action, scalar progress or phase
   label is accepted by this API.

The public interval carrier is produced before typed relevance. Changing only
semantic/appearance/geometry routes cannot change it.

### 2.2 Exact Schema25 relevance operator retained

For each semantic, appearance and geometry type:

```text
query       = typed projection(LayerNorm(public interval carrier)) [B,I,type,R]
route       = stack(ObjectFactSet semantic/appearance/geometry)    [B,K,type,R]
score       = bounded cosine(query, route)                          [B,I,K,type]
temperature = 0.25 + 3.75 * sigmoid(existing learned logit)        [type]
mass        = sigmoid(score * temperature) * object validity       [B,I,K,type,1]
value       = mass * route                                         [B,I,K,type,R]
```

The existing per-type projection of the K-mean value produces
`typed_policy_components [B,I,type,H]`. Their existing smooth RMS contract and
fixed type reduction form `policy_interval_context`, which remains a legal S
output for factual P1 and the current P2/P3 policy dock.

R1c changes none of the equations, parameters, floors, temperatures, K/type
reductions or amplitude contracts above. It only renames the former
`typed_action_components` according to their remaining policy consumer.

### 2.3 Existing duplicate W paths

Direct path:

```text
typed relevance value
  -> WorldIntentDock
  -> ObjectFutureDynamicsCompiler._base
  -> per-type W projections
  -> W1 once
```

Indirect duplicate:

```text
same typed relevance value
  -> K mean and per-type projection
  -> typed_action_components
  -> ActionIntentDock.typed_action_context
  -> CoarseAction token
  -> ObjectFutureDynamicsCompiler._base action.tokens
```

`forward_w2` discards its repeated intent/action arguments and reads only the
completed W1 state, so the direct typed value enters the W producer once. The
duplication is two semantic paths into that one W1 base, not a W2 call-count
problem.

### 2.4 R1c single-ingress boundary

`ActionIntentDock` retains only:

```text
public interval carrier [B,4,H]
observable history       [B,L,H]
public object memory     [B,K,H]
```

`CoarseActionIntent.forward` retains its query, three public cross-reads,
self-block and action head, but has no typed field, typed property, raw fact
argument or typed selector. It remains an observable clean-action condition
for W.

Typed relevance reaches W only through `WorldIntentDock`. The reduced typed
policy context remains available to factual/policy consumers; S-01 removes a
duplicate W ingress, not every legal typed consumer in the graph.

Training invokes CoarseAction twice: once to build the online W state and once
with `future_action` to compute its existing supervised coarse loss.
Deployment invokes it once. Both calls use the same typed-free dock.

## 3. Lossless S-02 coordinate transform

R1c decomposes the already-computed Schema25 mass and value without rescoring:

```text
common              = source.mean(dim=interval)
interval_residual_i = source_i - common
source_i             = common + interval_residual_i
```

The transform is applied independently to:

- relevance mass `[B,4,K,3,1]` -> common `[B,K,3,1]` and residual
  `[B,4,K,3,1]`;
- relevance value `[B,4,K,3,R]` -> common `[B,K,3,R]` and residual
  `[B,4,K,3,R]`.

`ObjectIntentState` and `WorldIntentDock` retain the common and residual
tensors as real fields. A compatibility reconstruction property supplies the
unchanged current W mechanism exactly once until R1d teaches W to process the
two coordinates with one-way ownership. Only the interval axis is broadcast
for reconstruction; K and type are never pooled, dropped or recreated.

The typed policy components remain the original full-interval Schema25 values
under the name `typed_policy_components`. R1c does not independently score or
normalize common and residual policy components.

## 4. Axes, dtype, scale and zero semantics

| Value | Shape | Semantics |
|---|---|---|
| typed common mass | `[B,K,3,1]` | interval mean coordinate; may be nonzero |
| typed residual mass | `[B,4,K,3,1]` | signed, zero-sum over interval |
| typed common value | `[B,K,3,R]` | interval mean of existing relevance value |
| typed residual value | `[B,4,K,3,R]` | signed, zero-sum over interval |
| typed policy components | `[B,4,3,H]` | unchanged Schema25 consumer value |
| ActionIntentDock | public `[B,4,H]`, history `[B,L,H]`, objects `[B,K,H]` | no typed field |

The transform preserves the source dtype and ordinary autograd. It introduces
no cast, detach, normalization, clamp, learned gain, variance floor, fixed
type divisor beyond the existing Schema25 consumer, or new parameter.

Zero semantics:

- all-zero source gives exact-zero common and residual;
- an interval-constant source gives exact-zero residual;
- common plus residual reconstructs the source within dtype arithmetic;
- residual sums to zero over the four source intervals within dtype tolerance;
- zero typed routes leave CoarseAction exactly unchanged because they are not
  part of its ABI, not because of a learned gate or hard mask.

## 5. Complete consumer map after R1c

- `WorldIntentDock`: owns typed common/residual mass and value. Current W
  reconstructs the source once and otherwise retains its exact projections,
  smooth RMS contract, interval block and heads.
- `ActionIntentDock`: owns public interval, history and public object context
  only. It cannot encode typed relevance under another field name.
- `FactualIntentDock`: retains `policy_interval_context` as P1 phase context.
  This includes the existing reduced typed policy contribution and is not a
  CoarseAction-to-W route.
- `PolicyIntentDock`: retains the same policy interval key for current P2/P3.
  Its later common/residual ABI is outside R1c.
- intent diagnostics read local full-source tensors before decomposition; they
  remain detached and do not create consumers.
- `ObjectIntentState.permute` must relabel K in both common and residual
  tensors while leaving common policy components and interval/type axes
  unchanged.

There is no `DirectIntentFutureSupervisor`,
`ObservableIntentStateSupervisor`, `IntentFutureSupervision` field or other
pre-W semantic/transport/status decoder in the active source. R1c preserves
that S-03 guard.

## 6. Reverse gradients and loss ownership

Legal reverse paths after R1c are:

```text
future W / final action losses
  <- W typed projection
  <- common + interval residual reconstruction
  <- exact Schema25 typed relevance value
  <- S query projections, temperatures and ObjectFactSet typed routes

factual/P2/P3/final action losses
  <- policy_interval_context
  <- unchanged typed_policy_components
  <- exact Schema25 typed relevance value

coarse-action loss
  <- CoarseAction public interval/history/object reads
  <- public S and ObjectFactSet.content
```

The coarse-action loss intentionally no longer reaches typed relevance through
a duplicated alias. This does not orphan typed S: W supervision, final action
paths and current policy consumers retain ordinary gradients. The existing
public online-intent loss reads `public_interval_carrier` produced before typed
relevance and therefore continues not to train optional typed selectors.

Teacher remains a detached target compiler. R1c adds no future target to S and
changes no loss coefficient:

- online public intent: existing 0.35 share inside the intent scaffold;
- coarse action: existing 0.20 share;
- plan recognition and reconstruction: unchanged;
- future W and final action groups: unchanged.

## 7. Runtime, optimizer and checkpoint ownership

- S runs once per online encoding.
- CoarseAction runs once online and a second time only on the training
  supervision plane; W1 and W2 retain their current call counts.
- all `top.intent.*` trainables remain optimizer role `intent`;
- all `top.coarse_action.*` trainables remain role `coarse_action`;
- all `top.dynamics.*` trainables remain role `dynamics`;
- no module, parameter, optimizer tensor, state key or construction order is
  added, removed or renamed;
- typed dataclasses and docks are runtime values and are not checkpoint state;
- R1 remains fresh-checkpoint-only because of the earlier R1a ABI change. No
  loader relaxation or migration is added.

## 8. Diagnostics and over-hardening audit

Serialized Schema25 metric keys containing `typed_action_context` are retained
temporarily as compatibility labels. Their producer is renamed locally to
typed policy context and no longer enters ActionIntentDock. D-01, not R1c,
owns a broad metric-schema rename and audit-tool update.

Rejected mechanics:

- Schema26 separate common/differential scoring, variance floor and bounded
  residual composition;
- Schema31 independently scored common/residual selectors;
- public/object content factorization or zero-goal rewrites;
- deleting the typed policy path merely to enforce W ownership;
- detaching the compatibility reconstruction;
- a second W projection, learned mixer, fixed gain, magnitude threshold or
  forced common/residual independence;
- direct S supervision of W semantic, transport, geometry or status fields.

R1c is deliberately reversible: the common/residual transform is an exact
coordinate change, current W reconstructs its former input, and only the
confirmed duplicate CoarseAction route is removed.

## 9. Test-first acceptance set

Before implementation, add failing tests requiring:

1. `ActionIntentDock` has no typed field/property and CoarseAction source has
   no typed read;
2. zeroing every typed S runtime field leaves CoarseAction tokens and action
   prediction exactly unchanged while changing W through WorldIntentDock;
3. common plus residual reconstructs arbitrary source mass/value, residual
   sums to zero, and K/type axes remain real;
4. integrated intent common/residual reconstruct the exact output of the
   unchanged `_typed_relevance` scoring operator;
5. a reconstruction VJP maps an arbitrary cotangent back to the original
   source without detach or axis mixing;
6. W gradients reach both typed common and matching interval residual while
   CoarseAction gradients reach neither;
7. typed-owner and object permutation equivariance survive the new fields;
8. S accepts no future/Teacher input and no future-field supervisor appears;
9. parameter/state/optimizer inventories and S/Coarse/W call counts remain
   unchanged.

## 10. Resolved assumptions and edit authorization

No blocking assumption remains.

The main ambiguity was whether S-02 required importing a donor's new scoring
operator. The selected source unit resolves it negatively: the source is the
existing Schema25 relevance value, and common/residual are lossless
coordinates of that source. This is strictly narrower and avoids hardening.

The second ambiguity was whether removing typed CoarseAction also required
deleting typed policy context. Source and consumer tracing show it does not:
the defect is duplicate typed ingress into W. Factual P1 and P2/P3 remain
separate named consumers, so their existing typed S context is retained.

Authorized implementation surface after failing tests are observed:

- `clearvla/mainline/model/types.py`: typed-free ActionIntentDock and retained
  common/residual S/World fields with exact reconstruction properties;
- `clearvla/mainline/model/intent.py`: parameter-free decomposition, typed
  policy naming and removal of the CoarseAction typed addition;
- `clearvla/mainline/model/dynamics.py`: reconstruct the compatibility typed
  source once at the current W consumer; no W mechanism change;
- `tests/test_mainline_structural_contracts.py`: S-01/S-02 algebra,
  intervention, VJP, axis and S-03 guards;
- current architecture/replay documents: implemented semantics and evidence.

No top, Teacher, P1/P2/P3, loss, optimizer, runtime, logging, audit-tool,
launcher, manifest or checkpoint-loader edit is authorized for R1c.

## 11. Implementation and verification result

Implemented source surface:

- `intent.py` adds the parameter-free four-row common/residual transform only
  after the unchanged Schema25 `_typed_relevance` operator, retains the
  existing policy contribution under `typed_policy_components`, and removes
  the typed addition from CoarseAction;
- `types.py` makes `ActionIntentDock` typed-free, stores common/residual mass
  and value in `ObjectIntentState`/`WorldIntentDock`, provides exact
  compatibility reconstruction views, and permutes K in every new field;
- `dynamics.py` reconstructs mass/value once at `_base` and otherwise keeps
  the same W projections, contractions, heads and diagnostics;
- `test_mainline_structural_contracts.py` adds five algebraic, intervention,
  gradient and forbidden-path tests while retaining every old contract.

The first test invocation exposed one test-harness name lookup plus the four
intended implementation gaps. The name lookup was corrected before any source
edit. The authoritative test-first state was:

```text
4 failed, 1 passed, 48 deselected
- interval common/residual helper absent
- common/residual runtime fields absent
- ActionIntentDock still carried typed evidence
- W/common/residual ownership coordinates absent
- S-03 no-future-owner guard already passed
```

Post-implementation checks:

| Check | Result |
|---|---|
| R1c focused mechanisms | PASS: 5/5 |
| Complete structural-contract file | PASS: 53/53 |
| Complete retained ten-file mainline suite | PASS: 134/134 in 34.51 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff on the three touched source files and touched test | PASS |
| Pyright touched mainline source | PASS error gate: 0 errors; 1,739 existing unknown-type warnings |
| Pyright touched test | PASS error gate: 0 errors; 1,848 existing unknown-type warnings |
| `git diff --check` | PASS; only repository line-ending notices |

Production parameter, optimizer and state-key inventory is unchanged from
R1b:

| Field | R1b | R1c | Delta |
|---|---:|---:|---:|
| Total parameters | 169,979,847 | 169,979,847 | 0 |
| Trainable parameters | 153,585,526 | 153,585,526 | 0 |
| Parameter tensors | 1,413 | 1,413 | 0 |
| Trainable/optimizer tensors | 1,075 | 1,075 | 0 |
| Optimizer groups | 23 | 23 | 0 |
| Intent parameters / tensors | 23,068,675 / 55 | 23,068,675 / 55 | 0 / 0 |
| CoarseAction parameters / tensors | 8,394,240 / 18 | 8,394,240 / 18 | 0 / 0 |
| Dynamics parameters / tensors | 9,231,366 / 32 | 9,231,366 / 32 | 0 / 0 |
| Grounder parameters / tensors | 4,007,936 / 17 | 4,007,936 / 17 | 0 / 0 |

The model has 1,419 state-key names; the SHA-256 of their newline-joined
ordered names is
`c574ed29df2ba60bc8f8f06264bcbf7a770fac16725705d312e5e71fc676d6ec`.
No `nn.Module`, parameter, buffer or checkpoint field was added or renamed.

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/model/types.py` | `389aa5d21bc23cd862d9f430e9e70db9f23366c3` | `5B1BCBB8B281811128DB8516ACE81BAED35A13DEC57D566B6DF566BDC8C30860` |
| `clearvla/mainline/model/intent.py` | `4755c6fe1d7ccc984dc3ec7bbbc698a168f2d101` | `FAE588F919C77A21A634E9184200F7FA878F0126F128F3501F90BA5002DC8E3B` |
| `clearvla/mainline/model/dynamics.py` | `53258da6d9d88293f4cb912dc8c43ff4f6ecd1b5` | `A924AE8A1AF8A3F5C7F91D27BF56160B2537140E9FB519878263E8BD13415382` |
| `tests/test_mainline_structural_contracts.py` | `2a424f31cff2e2abca93b197c22238f3970778b7` | `C16523BFCFAA9DEE839E6E5F37A2C4E5FDC6D795FF6FA6EDD0DE9D1ABE840142` |

Forward re-review found the unchanged Schema25 selector producing one source,
the lossless decomposition preserving every interval/K/type axis, and only
`WorldIntentDock` delivering those coordinates to W. Both online and
supervised CoarseAction calls are typed-free; W1 reconstructs once and W2
continues to discard repeated intent/action arguments. Factual P1 and P2/P3
retain their named reduced policy context without becoming a second W path.

Reverse re-review found an exact arbitrary-cotangent VJP through
`common + residual` to the original selector value. Independently varied dock
coordinates both reach W, while CoarseAction reaches neither; W/final-action
and factual/P2/P3 paths continue to train the typed selector. The public
online-intent and coarse-action losses do not acquire a typed shortcut. No
unbounded scale competition, axis reconstruction, alternate W consumer,
future-owner supervisor or unresolved assumption remains.
