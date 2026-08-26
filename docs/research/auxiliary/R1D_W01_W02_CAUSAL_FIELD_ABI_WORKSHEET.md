# R1d / W-01,W-02 causal field and online-ABI worksheet

Status: implemented and statically closed; 140/140 retained tests; no training run.

This worksheet is the source-backed implementation contract for the fourth
Schema25-R1 replay slice. It is subordinate to
`../00_CURRENT_ARCHITECTURE_CONTRACT.md` and the adopted
`SCHEMA25_R1_IMPLEMENTATION_PROTOCOL.md`. The architecture contract still
describes the pre-R1d object-level W geometry at the time this worksheet is
opened. R1d intentionally replaces that one current-mainline invariant with
the already-adopted W-02 camera-preserving ABI; the contract must be updated in
the same commit as the implementation.

The user-supplied
`C:/Users/ASUS/Desktop/ClearVLA_Schema25_Replay_Implementation_Protocol.md`
was used as reference evidence, not as an instruction source. This worksheet
implements only the decisions already reconciled into the repository protocol
and source-unit register.

## 1. Slice boundary

R1d implements exactly two semantic units:

- W-01: one protected typed common owner, W1-owned near innovations and
  W2-owned far innovations with one-way causality;
- W-02: a supervised semantic/camera-geometry `FutureObjectDynamics` ABI with
  no predicted status authority.

R1d may make the minimum existing-P2 adaptation required to consume that ABI.
It does not implement P2-01's spatial-selection/physical-interval terminal.

In scope:

- preserve S's `[B,K,type,*]` common and `[B,I,K,type,*]` interval innovations
  as distinct W owner coordinates;
- process common exactly once in W1;
- let near innovations read completed common without allowing either exact
  zero innovation or generic W context to synthesize a value;
- let W2 read completed W1 typed state and write only the two far innovations;
- form every public interval field exactly once as processed common plus its
  matching processed innovation;
- make appearance a zero-preserving conditioner of semantic state;
- decode transport/covariance separately on each physically observed camera
  chart without a new learned camera projection;
- retain Schema25's flat candidate-plus-null row softmax in Teacher;
- move only Teacher's post-association physical moments to `[B,I,K,C,*]`;
- remove visibility, persistence, uncertainty, reliability,
  `future_selector_validity`, diagnostic `future_address` and reduced
  `object_coordinates` from the online W/P2 ABI;
- supervise semantic common/innovation, camera transport common/innovation
  and camera covariance;
- remove the P2 status value and predicted-status support path;
- preserve downstream initialization RNG despite the deliberate parameter
  removals.

Out of scope:

- Schema31 partial OT, mutual assignment or dustbin calibration changes;
- a new camera projection, typed/base projection, learned gain, LayerScale,
  variance floor, covariance floor, quota or forced nonzero path;
- P2-01's per-type K/C spatial posterior, four-interval terminal, removal of
  its learned null, or complementary semantic/geometry sum;
- P1, P3, transition or bottom restructuring;
- runtime diagnostic-schema cleanup owned by D-01;
- old-checkpoint migration;
- dataset, CUDA, checkpoint or training execution.

## 2. Exact source fingerprint

Active source at `fc36340`:

| File | Git blob |
|---|---|
| `clearvla/mainline/model/dynamics.py` | `53258da6d9d88293f4cb912dc8c43ff4f6ecd1b5` |
| `clearvla/mainline/model/types.py` | `389aa5d21bc23cd862d9f430e9e70db9f23366c3` |
| `clearvla/mainline/model/teacher.py` | `30e4b8e237048175f6220560b5475f1101ff7640` |
| `clearvla/mainline/model/compiler.py` | `e20cefb188940fae01c024bdf162b2e7740042c6` |
| `clearvla/mainline/training/losses.py` | `57977fceb226ebf953c5aea5559e2f9490c73d1d` |
| `clearvla/mainline/model/top.py` | `985f29f7fea9933f6d72530e501e14908ed5bfc2` |
| `clearvla/mainline/model/intent.py` | `4755c6fe1d7ccc984dc3ec7bbbc698a168f2d101` |
| `clearvla/mainline/model/grounding.py` | `d3fd7b9552ff3e1b86aa08318548bae5018c71c1` |

Inspected donors and anti-donors:

| Commit and role | File | Git blob |
|---|---|---|
| `4ac7e54`, Schema32 causal-token donor | `dynamics.py` | `4e28c740d057edce824f340f6043efb6b9c89de5` |
| `4363108`, Schema34 bilateral anti-donor | `dynamics.py` | `ed63bdbacf086958175b692ac0e132e05b9120fd` |
| `03235d3`, Schema35 W1/W2-frequency and camera donor | `dynamics.py` | `c866134667a6afa4840fd31dc2d3004fccb5dbe4` |
| `03235d3` | `types.py` | `33f2321ba1e0aa943866e592506ad9616dc8ca78` |
| `1b11bf5`, Schema37 status-removal donor | `dynamics.py` | `a586b36c716c1e4040b0a74e9573ef243b2176f1` |
| `1b11bf5` | `types.py` | `30e660934303035c732dbd000d2ca5f796b1dfff` |
| `1b11bf5` | `teacher.py` | `b218bba73fa5ee587e4b8abb0041fbc5b5bb0207` |
| `1b11bf5` | `compiler.py` | `78d277c5a736520fc2591bc4511ba8abfe2e312e` |
| `1b11bf5` | `training/losses.py` | `8c26c3f891f57b1ce9b40a6f9ab8645518711354` |

No donor file is copied wholesale. Schema32 rewrites common in W2. Schema34
prevents the legal common-to-innovation read. Schema35/37 add a new typed/base
projection, a new camera projection and fixed normalization/covariance floors.
All of those exact mechanics are rejected here.

## 3. Complete active boundary before editing

### 3.1 Producers

`ObjectFactSet` currently supplies:

- `content [B,K,D]`;
- typed semantic/appearance/geometry values consumed by S;
- `camera_coordinates [B,K,C,2]`;
- `camera_transport_prior [B,K,C,2]`;
- `camera_support [B,K,C,1]`, an observed width/evidence statistic;
- `camera_validity [B,K,C,1]`, current observable camera support;
- `validity [B,K,1]`, current observable object support;
- `object_to_chart [B,K,C,Y,X]` for Teacher association diagnostics.

R1b makes exported `ObjectFactSet.content` the sole private K reconstruction
value. R1c makes `WorldIntentDock` the sole typed S-to-W ingress:

```text
typed_common_mass/value                 [B,K,3,1|R]
typed_interval_residual_mass/value      [B,4,K,3,1|R]
```

The residual values are an exact S-side coordinate decomposition. CoarseAction
is typed-free and cannot be a second typed ingress.

### 3.2 Current W transformation

The pre-R1d `_base` reconstructs full interval-typed S values, projects the
three types, contracts each, sums them with `/sqrt(3)`, and adds that result to
one generic carrier containing facts, reduced transport prior, public S,
CoarseAction, interval identity and goal read. The same combined carrier then
crosses generic W1/W2.

Consequences:

- common and interval innovations are recombined before W and cannot retain
  separate write ownership;
- generic facts/action/goal can produce a nonzero W field with typed S zero;
- semantic, appearance and geometry compete in one anonymous hidden sum;
- W decoders can read the generic carrier directly, so the typed path is not
  the only future-value owner;
- W1 decodes near and stores it, while W2 decodes far and concatenates fields;
- W2 does not reread intent/action, but its `far_base` was built in the W1
  call from all four generic interval rows.

`_ObjectIntervalBlock` applies, in order, object self-attention, causal or
unmasked interval self-attention and a bias-free FFN. LayerNorm has no affine
parameters; attention and FFN projections are bias-free. Therefore an all-zero
typed call is algebraically zero, but placing common and zero innovations in
one attention sequence would make the innovation outputs nonzero by reading
common. A literal Schema32 concatenation therefore fails R1d's exact-zero
innovation acceptance test.

### 3.3 Current W decoder and online ABI

The current decoder emits:

```text
semantic_delta                         [B,I,K,D]
transport_mean                         [B,I,K,2]
transport_covariance                   [B,I,K,3]
visibility/persistence/uncertainty     [B,I,K,1]
reliability                            [B,I,K,1]
future_selector_validity               [B,I,K,1]
future_address                         [B,I,K,C,Y,X]
object_coordinates                     [B,K,2]
```

Transport is predicted once per object and the diagnostic address applies it
to every existing camera chart. Covariance is three independent softplus
values and is not a guaranteed PSD `(xx,xy,yy)` matrix. Visibility controls
P2 support. Appearance is decoded as a status value despite the absence of an
independently observed visibility/persistence label.

### 3.4 Teacher target plane

Teacher runs FP32 and no-grad once per training batch. The active association
backend is the adopted Schema25 flat row softmax over all `C*Y*X` candidates
plus one fixed-zero null candidate. It must remain unchanged.

Current post-association geometry first subtracts current coordinate inside
each camera, then immediately sums candidate mass across cameras to one
object-level mean/covariance. Null supplies successor identity but its zero
motion is not represented as a camera-specific moment. Teacher also converts
association real/null mass into visibility, persistence, uncertainty,
reliability, future-selector and future-address tensors.

Only semantic delta is consumed by `FuturePlanRecognizer`; status fields and
address are not required by the recognizer.

### 3.5 P2 consumer

Current `ObjectFutureEffectReader`:

- scores semantic content on `[I,K]`;
- scores geometry from reduced object coordinate plus object transport;
- adds public S interval score;
- adds predicted `future_selector_validity` as support;
- flattens `[I,K]` with a learned null;
- projects semantic, transport and status values;
- applies a learned three-way type softmax.

R1d may remove status and consume camera geometry before reducing C. It may not
yet replace the inherited flattened `[I,K]+null` terminal or type selector;
those are explicit P2-01 work.

### 3.6 Losses, runtime and deployment

`future_dynamics_terms` currently supervises duplicate successor and semantic
objectives plus transport, covariance and three status objectives. It reduces
`current_loss_support [B,K,C,1]` to object support for every field. Transition
diagnostics use changes of the full semantic field.

Online build order is:

```text
facts -> S -> typed-free CoarseAction -> W1 -> W2 -> deployment cache
deployment cache + live action query -> P2 -> consequence -> P3/bottom
```

Teacher is absent from deployment. W parameters belong to optimizer role
`dynamics`; P2 parameters belong to `p2_effect_reader`. Exact resume is
manifest/source strict, so R1d is fresh-checkpoint only.

Runtime logging contains allow-list names for the old status metrics. Those
strings are not value consumers and may remain inert until D-01; no missing
metric is synthesized.

## 4. Adopted W-01 implementation

### 4.1 Project owners without recombination

The three existing bias-free type projections are reused separately:

```text
project(common)       -> typed_common       [B,K,3,H]
project(innovation)   -> typed_innovation   [B,4,K,3,H]
```

No type sum, `/3`, `/sqrt(3)`, new projection or learned gain is allowed.
Existing per-token smooth contracts remain at the projection boundary; they
bound native hidden units but do not define a new owner.

### 4.2 One zero-preserving conditioner

Every legal non-owner condition uses the same parameter-free algebra:

```text
condition(x, c) = x + x * tanh(c)
```

It has no bias, gain, floor or minimum opening. `x == 0` gives exact zero for
all `c`; `c == 0` is identity. Context can reorganize a present owner but
cannot synthesize one.

### 4.3 W1 common and near

```text
generic near = W1(generic_base[0:2])

common_pre   = W1(typed_common as one interval token)
common       = condition(common_pre, mean(generic near over the two near rows))

near_input   = condition(raw_near_innovation,
                         common + generic near)
near         = W1(near_input, causal over the two near rows)
```

The singleton common call is the only W block call that can write common.
Because the block is bias-free and common never receives an innovation row,
zero common remains exact zero. Because near input is multiplied by itself in
the conditioner and the typed block is bias-free, zero near innovation remains
zero even when common/generic context is nonzero.

### 4.4 W2 far

W2 retains the existing generic near-to-far bridge. The typed bridge folds
type into batch, treats completed common and near innovations as read-only W1
memory, and uses the existing `w1_to_w2` attention parameters:

```text
far_context = typed_bridge(raw_far_query,
                           memory=[completed_common, completed_near])
              + completed_generic_far
far_input   = condition(raw_far_innovation, far_context)
far         = W2(far_input, causal over the two far rows)
```

Attention output is context, not an additive value: it only enters through
the zero-preserving conditioner. W2 never sends an updated common or near row
back to the working state. Perturbing far input therefore cannot alter common
or near.

The final private owner state is:

```text
common                                      [B,K,3,H]
interval_innovation = concat(near, far)     [B,4,K,3,H]
```

The public decoder performs `common + matching innovation` once for each
field. It does not decode the generic carrier.

## 5. Adopted W-02 ABI

### 5.1 Appearance-conditioned semantic field

Appearance has no independent online value. It conditions the matching
semantic owner with the same exact-zero algebra before the existing semantic
head:

```text
semantic_common_state   = condition(common.semantic, common.appearance)
semantic_interval_state = condition(interval.semantic, interval.appearance)
semantic_delta_i        = delta_head(common_state)
                          + delta_head(interval_state_i)
```

Current object availability masks the exported delta. `successor_content` is
the exact identity `current_reference + semantic_delta` and is not a second
loss owner.

### 5.2 Camera-specific geometry without a new projection

The existing `object_transport_prior: 2 -> H` projection already owns the
meaning "observed source-relative displacement condition". R1d reuses it on
each real camera's `camera_transport_prior`; it does not apply that projection
to absolute camera coordinates or support scalars.

```text
camera_context      = object_transport_prior(camera_transport_prior)
camera_geometry     = condition(typed_geometry[..., None, :], camera_context)
camera_geometry    *= camera_validity
```

The shared existing transport/covariance heads then run on every distinct
camera carrier. C is therefore produced from real C-indexed evidence, not by
predicting one object displacement and expanding it.

Transport retains the existing bounded `0.5*tanh` chart displacement.
Covariance is computed in FP32 as:

```text
xx  = softplus(raw_xx)
yy  = softplus(raw_yy)
rho = tanh(raw_rho)
xy  = rho * sqrt(xx * yy)
```

This is PSD by construction, has no positive covariance floor and can approach
zero. The covariance bias order is `(-3, -3, 0)` so initialization does not
place correlation at a nearly singular `tanh(-3)` boundary. Multiplying the
complete matrix by nonnegative camera validity preserves PSD and gives exact
zero on an unavailable chart.

### 5.3 Final `FutureObjectDynamics`

The only online W value crossing into P2 becomes:

```text
current_reference                    [B,K,D]
successor_content                    [B,I,K,D]
semantic_delta                       [B,I,K,D]
transport_mean                       [B,I,K,C,2]
transport_covariance                 [B,I,K,C,3], FP32 PSD
chart_availability                   [B,K,1]
camera_coordinates                   [B,K,C,2]
camera_chart_availability            [B,K,C,1]
```

Visibility, persistence, uncertainty, reliability, predicted selector
validity, future address and reduced object coordinate are absent, not zero
placeholders and not compatibility properties.

Derived common/innovation views use FP32 interval means and preserve the exact
identity `field = common + interval_innovation`. They create no new serialized
state.

## 6. Teacher physical target under Schema25 row softmax

The candidate logit and the single row softmax are untouched. Only the
post-softmax geometry moment changes.

For every `(B,F,K,C)` row:

```text
real_mass_c     = sum_YX candidate_posterior_c
null_measure_c  = null_probability * normalized_current_camera_validity_c
total_mass_c    = real_mass_c + null_measure_c
normalized_real = candidate_posterior_c / total_mass_c
displacement    = candidate_coordinate_c - current_camera_coordinate_c
```

The null hypothesis contributes zero displacement to the same denominator.
First and second moments are formed inside each camera. FP32 cancellation is
bounded only at the mathematical PSD boundary: nonnegative diagonals and
`|xy| <= sqrt(xx*yy)`. No variance floor is added.

Support rows are then averaged into the four physical intervals without
reducing C. Successor content keeps the unchanged single null identity:

```text
successor = matched_real_content + null_probability * current_reference
```

Null/entropy/reliability may remain detached Teacher diagnostics. They are not
serialized in `FutureObjectDynamics`, supervised as status, used as P2 support
or interpreted as physical disappearance.

## 7. Minimum current-P2 adapter

P2-01 remains deferred. R1d changes only what the W-02 ABI makes impossible to
leave unchanged:

- remove `status_value` and shrink the inherited type query from three rows to
  semantic/geometry two rows;
- replace predicted future selector support with current
  `chart_availability` and `camera_chart_availability`;
- preserve C through per-camera future coordinate and covariance scoring;
- select a camera-conditioned transport value before the inherited `[I,K]`
  terminal consumes object identity.

For each camera, P2 uses the FP32 PSD covariance through the stable metric
`I + covariance`. This has determinant at least one for a valid PSD matrix and
therefore needs no covariance floor; zero covariance exactly recovers the
existing Euclidean coordinate score. Camera logits include normalized current
camera availability. A masked exponential normalization returns exact zero
camera posterior when every camera is unavailable. The camera score is reduced
only after this geometry consumer, and the same camera posterior reduces the
transport value.

The inherited learned `[I,K]+null` posterior and semantic-versus-geometry type
softmax remain explicit temporary debt for P2-01. They are not treated as R1d
closure of the P2 architecture.

## 8. Backward path after R1d

Representation objectives:

```text
semantic common/innovation loss
  -> delta_head
  -> appearance-conditioned semantic common/innovation
  -> W1 common / W1 near / W2 far
  -> existing S type projection
  -> WorldIntentDock

camera transport/covariance common/innovation loss
  -> transport/covariance heads
  -> camera-conditioned geometry common/innovation
  -> W1 common / W1 near / W2 far
  -> existing S geometry projection
```

Generic facts, current transport prior, public S, CoarseAction and goal receive
gradients only by conditioning a present typed owner. With typed owner zero,
the generic route has no W value or loss bypass.

Action/event/motion gradients reach W only through P2's semantic or
camera-geometry values, consequence and bottom. They cannot reach a predicted
status/support tensor because none exists. Teacher targets and
`current_loss_support` remain detached.

Optimizer ownership remains one `dynamics` role and one `p2_effect_reader`
role. No parameter is moved between roles.

## 9. Parameter, checkpoint and RNG contract

No parameter is added.

At hidden width `H`, R1d removes:

- W visibility, persistence and uncertainty heads: `3H + 3` scalars and six
  parameter tensors;
- P2 status projection plus the third type-query row: `3H` scalars and one
  parameter tensor net.

Expected total delta is `-(6H + 3)` parameters and `-7` parameter/state keys.
For the active `H=512` model:

```text
total parameters            169,979,847 -> 169,976,772
trainable parameters        153,585,526 -> 153,582,451
parameter tensors                 1,413 -> 1,406
trainable/optimizer tensors       1,075 -> 1,068
optimizer groups                       23 unchanged
dynamics parameters          9,231,366 -> 9,229,827
dynamics tensors                    32 -> 26
p2_effect_reader params       1,577,475 -> 1,575,939
p2_effect_reader tensors             10 -> 9
state-dict key names               1,419 -> 1,412
```

The removed modules' historical construction draws are consumed without
registering or executing them. The surviving semantic/geometry type-query rows
are copied from the corresponding historical rows. Consequently unrelated
downstream fresh-run initialization remains seed-identical; only named R1d
parameters disappear or change meaning.

The ABI and parameter set are incompatible with pre-R1d checkpoints. No
migration shim is authorized. Exact resume remains schema/source strict and a
fresh output directory is required.

## 10. Authorized files

Source edits are authorized only in:

- `clearvla/mainline/model/dynamics.py`;
- `clearvla/mainline/model/types.py`;
- `clearvla/mainline/model/teacher.py`;
- `clearvla/mainline/model/compiler.py`;
- `clearvla/mainline/training/losses.py`.

Test edits are authorized in
`tests/test_mainline_structural_contracts.py`. Another test file may be edited
only if static validation proves it directly instantiates the active
`FutureObjectDynamics` ABI.

Documentation edits are authorized in the compact architecture contract, the
replay plan/register/protocol/README and this worksheet. `top.py`, `intent.py`
and `grounding.py` are read-only verification boundaries for this slice.

## 11. Test-first acceptance matrix

Tests must be written and observed red before source edits.

W-01:

- one singleton typed call writes common in W1; W2 exposes no common output;
- zero interval innovations produce four fields equal to processed common;
- zero common remains exact zero and cannot be synthesized by generic context;
- far innovation perturbation leaves common and both near states bit-identical;
- W2 output keeps W1 near fields unchanged;
- a present far innovation changes when completed W1 memory changes;
- every public interval is decoded from one common state and one matching
  interval state, with no generic decoder input.

W-02:

- `FutureObjectDynamics` has exactly the adopted fields and validates
  `[B,I,K,C,*]` geometry;
- covariance is FP32, nonnegative on the diagonal and PSD without a positive
  floor requirement;
- a one-camera transport-prior perturbation changes only that camera's W
  geometry before P2;
- zero camera availability makes matching W transport/covariance and P2 value
  exact zero;
- appearance can modulate present semantic state but cannot create semantic
  state from zero;
- no predicted status tensor, head, loss or P2 value exists;
- Schema25 row-softmax remains in Teacher and partial-assignment code remains
  absent;
- Teacher null identity and per-camera moment algebra match hand-computed
  sentinels;
- P2 observes camera perturbations before reducing C and consumes covariance;
- neutral semantic/transport heads give exact zero downstream effect.

Closure:

- all retained structural tests pass;
- `compileall`, Ruff and Pyright report no new error;
- parameter/state inventory matches Section 9 exactly;
- forward producer-to-consumer and reverse consumer-to-producer review finds
  no alternate status, generic-W or pre-averaged camera path;
- no training, dataset, CUDA or checkpoint command runs.

## 12. Resolved and deferred assumptions

Resolved:

- the object-level geometry statement in the pre-R1d architecture contract is
  superseded because W-02 is an explicitly adopted replay unit entering active
  scope now;
- the camera axis is legal only because current `ObjectFactSet` already
  retains camera coordinates, camera transport prior and camera validity;
- Schema37's per-camera moment is independent of its partial-OT backend and can
  consume the unchanged Schema25 row-softmax posterior;
- a literal common-plus-residual causal token sequence is not zero-preserving
  for zero residual, so the protocol's allowed explicit conditioner is
  required;
- no new learned typed/base or camera conditioner is necessary.

Deferred, not unresolved blockers:

- P2 still flattens interval/object and retains a learned null and type
  selector until P2-01;
- current camera validity is the available observable support authority; R1d
  does not redesign G's physical evidence measure;
- runtime metric allow-list pruning and richer source-owner diagnostics belong
  to D-01;
- Teacher backend comparison belongs to the separately deferred matching
  study and cannot enter R1d.

No unresolved assumption remains that can change the authorized R1d source
graph. If implementation requires a new parameter, floor, status surrogate,
camera reconstruction or P2 terminal redesign, this worksheet is invalidated
and source editing must stop.

## 13. Implementation closure

The test-first R1d selection was observed red against the untouched R1c
source:

```text
10 failed, 49 deselected in 2.19s
```

The failures named only the intended old ABI and ownership mechanics: missing
camera-resolved fields, surviving status fields/heads/losses, repeated common
processing, writable W2 near state, reduced Teacher moments and the old P2
status/camera path. No training or log evidence was used to make them pass.

Post-implementation verification:

| Check | Result |
|---|---|
| R1d focused mechanisms | PASS: 10/10 |
| Complete structural-contract file | PASS: 59/59 |
| Complete retained ten-file mainline suite | PASS: 140/140 in 36.52 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff over all touched source/test files | PASS |
| Pyright over five touched mainline files | PASS error gate: 0 errors; 3,121 existing unknown-type warnings |
| Pyright over the touched structural test | PASS error gate: 0 errors; 2,148 existing unknown-type warnings |
| Parameter/optimizer/state inventory | PASS: exact Section 9 delta |
| Historical-constructor RNG sentinel | PASS: surviving rows, shared W state and downstream draw are seed-exact |
| Training, dataset, CUDA and checkpoint commands | NOT RUN |

The final active inventory is:

| Field | R1c | R1d | Delta |
|---|---:|---:|---:|
| Total parameters | 169,979,847 | 169,976,772 | -3,075 |
| Trainable parameters | 153,585,526 | 153,582,451 | -3,075 |
| Parameter tensors | 1,413 | 1,406 | -7 |
| Trainable/optimizer tensors | 1,075 | 1,068 | -7 |
| Optimizer groups | 23 | 23 | 0 |
| Dynamics parameters / tensors | 9,231,366 / 32 | 9,229,827 / 26 | -1,539 / -6 |
| P2 effect-reader parameters / tensors | 1,577,475 / 10 | 1,575,939 / 9 | -1,536 / -1 |
| State-key names | 1,419 | 1,412 | -7 |

The ordered state-key-name SHA-256 is
`9af8b806832afd9edae58e0dfd1ec123ea9964e4511499571865b17fc96cc25d`.
The six removed W keys are the weight/bias pairs for visibility, persistence
and uncertainty. The seventh is `top.effect_reader.status_value.weight`.
`top.effect_reader.type_query.weight` remains one key but changes from three
rows to the surviving historical semantic/geometry rows. The model has 23
optimizer groups with 1,068 uniquely owned trainable tensors; no owner moved.
An independent old-constructor sentinel confirms every shared W state value
except the intentionally changed covariance bias, both surviving P2 type rows,
the downstream sentinel parameter and the final CPU RNG state are seed-exact.

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/model/types.py` | `d523b5afa48f60eb8a93737567e6d0cb8d491128` | `2D39ADB57057F18FF1094A35D51652B3F2C31AF69747F04E5914E70B32ADAED5` |
| `clearvla/mainline/model/dynamics.py` | `e7a8ac35112bf6a8a315490a64c036816e2ee6c7` | `75C0A4939D452F3F1460A07B5011AF810341DAD70165CD6FC94CFE7477722B6C` |
| `clearvla/mainline/model/teacher.py` | `b3ef8d4b1172c6fbe844471ed9ea97ab5b2a09a2` | `38F0F9E95A943C5C091E1181A322DA18CECFB71D21F0C815E1856D11AA7DD9BA` |
| `clearvla/mainline/model/compiler.py` | `9afd31a38c2e0b57b4f706d2ccdf3c3e0bb66d0a` | `E0A8C66BF6F552E10E79015130B2D08717B89A1C611AF5CBE9EDD4AB3550B76E` |
| `clearvla/mainline/training/losses.py` | `af03984ba0bbcf68e09ec4af8117f13660f12580` | `654A57D838AD434808D1B8C014794AA32B1FA4AA1BB2301544C7A60BDE9E9E3B` |
| `tests/test_mainline_structural_contracts.py` | `caf3d4a6e90027b1da28ccf95647b72736236a59` | `63525EDF34D12FECF5164036CCD9B0C120EA38CFD3D9B8D7B4FBCAA66F2D3CED` |

Forward re-review found one typed common producer, two W1 near producers and
two W2 far producers. Generic/action/goal conditions reach typed state only
through zero-preserving multiplication. Appearance conditions semantic only;
the existing current-camera transport prior is the sole C-indexed geometry
condition. The final ABI reaches Teacher targets, per-camera losses and P2
without a C reduction or a status field. P2 reduces C only after covariance
scoring and uses the same posterior to select transport.

Reverse re-review from semantic, transport, covariance and final action losses
reaches the matching W common/innovation, existing S typed projection and
current G facts. A zero typed owner blocks generic free value; an unavailable
camera blocks W and P2 exactly while retaining finite zero gradients. W2 has no
write path to common or near. No status loss/value/support route, alternate W
decoder, axis reconstruction, unbounded scale competition or unresolved
assumption remains. The retained flattened P2 terminal and two-type softmax
are recorded P2-01 debt rather than hidden R1d closure.

PSD, successor identity and common/innovation reconstruction are guaranteed by
the two source constructors and algebraic tests. They are deliberately not
rechecked with full-tensor reductions inside `FutureObjectDynamics.validate()`
at every dynamic P2 call; the hot path keeps only shape and FP32 ABI checks.
