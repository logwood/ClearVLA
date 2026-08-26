# R1b / G-02 conditional-K and reconstruction worksheet

Status: implemented and statically closed on 2026-08-26; no training run.

This worksheet is the mandatory producer-to-consumer and
consumer-to-producer review for G-02. It authorizes only the conditional-K
G3 correction and reconstruction-ownership unit selected in
`ARCHITECTURE_REPLAY_SOURCE_UNITS.md`. It does not authorize the Schema27
equal typed-logit consensus, the Schema35 content-only binder, new public
content types, S/W/Teacher rewrites, a new loss, or checkpoint migration.

The user-supplied
`C:/Users/ASUS/Desktop/ClearVLA_Schema25_Replay_Implementation_Protocol.md`
was reviewed as reference material. Its relevant requirement is the static
intervention that learned association null must not directly control physical
reconstruction ownership. It does not prescribe a donor implementation.

## 1. Source identity and donor disposition

| Role | Commit/blob | Disposition |
|---|---|---|
| R1a active grounder | `4b3ca05:grounding.py` / `bf132168c5c116f4f54ca4cbfbfe4c60d0454cd6` | base |
| Schema27 target donor | `2a0d3d1:grounding.py` / `91a24c92b4e9130c4a686cd94796fa439e58c22d` | keep detached current-DINO target and observed mask only |
| Schema30 reconstruction donor | `127fee8:grounding.py` / `6ac7477e36a3862293a4bdd4ce8908fe23ae8bec` | keep conditional-K algebra; reject its larger typed-binder/public-content bundle |
| Schema35 ownership donor | `03235d3:grounding.py` / `ab65c14dd450c280658c12372eb37f709502735c` | keep the proof that a slot residual can be exported rather than deleted; reject content-only K ownership and new ABI |

Schema25 has three source-confirmed defects:

1. `DenseFactChart.dino_content` is reconstructed from the same online local
   `content_slots` being judged, so a collapsed online representation can
   become its own target.
2. G3 adds K residuals beside a fixed zero null logit and re-softmaxes K+null.
   It can therefore alter the parent real-versus-null mass although its
   intended semantic unit is conditional K refinement.
3. `decode_content_residual(slots)` is used only inside reconstruction.
   S, W and Teacher receive `ObjectFactSet.content` without the detail learned
   through that residual.

## 2. Complete forward dataflow

### 2.1 Producers

The active deployment/training producer is
`RestoredV120ObservationCompiler.finalize_grounding`:

1. G1/G2/G3 completes `GroundedFactSet` with axes
   `[B,C,8,8,M,*]`.
2. The current full-DINO chart comes from
   `GroundingObservationBank.address_bank.dense_current_dino_content`.
3. It is detached before entering `LocalFactSet.target_dino_content`.
4. `cell_observed=(~context_mask)[...,None]` records which current chart cells
   were legally visible to the online path.
5. Online candidate values, typed values, owner priors, coordinates, support,
   validity and transport remain the completed G3 fields with the local-M axis
   intact.

`CurrentObservationCompiler` is the only other source constructor. It also
detaches `coarse_dino`, exports a boolean observed mask, and retains the same
local-M candidate ABI. It is not selected by `ClearVLAMainlinePolicy`, but the
shared typed boundary must remain valid for its structural tests.

### 2.2 Dense chart and retained physical binder

`dense_chart_from_local_facts` must make exactly one target change:

```text
DenseFactChart.dino_content = LocalFactSet.target_dino_content.detach()
```

`candidate_content` remains `content_slots`. Semantic, appearance, geometry,
coordinates, support, validity, transport and the Schema25 physical owner
prior are unchanged. In particular, the candidate key remains:

```text
content + (semantic + appearance + geometry) / sqrt(3) + coordinate
```

The iterative Slot Attention binder, its K=4 plus null competition, the final
post-GRU posterior recomputation, local prior placement and physical
validity placement all remain unchanged.

### 2.3 Conditional-K G3

For each flattened candidate row `n`, let the final binder posterior be
`P_parent(K+null | n)` in FP32. G3 is factored as:

```text
real_mass[n]       = sum_K P_parent(K | n)
parent_K[n,K]      = P_parent(K | n) / real_mass[n]
corrected_K[n,K]   = softmax(log(parent_K[n,K]) + bounded_G3[n,K])
P_corrected(K|n)   = real_mass[n] * corrected_K[n,K]
P_corrected(null)  = P_parent(null|n)
```

Subtracting a per-row common residual is algebraically optional because K
softmax already removes it. No new gain, temperature, floor beyond the
existing numerical `clamp_min`, or learned null is introduced.

The ordinary online physical assignment remains:

```text
P_corrected(K|n) * local_prior[n] * physical_validity[n]
```

It continues to own object reads, existence, support, camera reductions,
typed reads, S, W and Teacher. G-02 does not turn a reconstruction rule into a
second online object identity.

### 2.4 Unique exported content and reconstruction

The existing slot residual capacity is retained but its ownership changes:

```text
aggregated_content = physical object read of candidate_content
slot_residual       = existing zero-initialized decode_content_residual(slots)
ObjectFactSet.content = aggregated_content + slot_residual
```

At initialization this is exactly the former exported content. After
training, reconstruction, S and W all see the same enriched K value; Teacher
reads its detached form. There is no second loss-only K value.

Reconstruction receives a separate assignment, not a separate identity:

```text
Q[n,K] = corrected_K[n,K] * local_prior[n] * physical_validity[n]
Q_chart[K,C,Y,X] = sum_M Q[n,K]
```

The existing `decode_position` is retained because deleting unrelated shared
spatial capacity would be hardening. Its input has no K axis. The
implementation must expose that algebra explicitly:

```text
shared_support[C,Y,X] = sum_K Q_chart[K,C,Y,X]
reconstructed = einsum(Q_chart, ObjectFactSet.content)
              + shared_support * decode_position(chart_coordinate)
```

Thus the coordinate decoder is a shared, support-gated spatial term; it
cannot encode K identity. No public mean, new decoder, new content field or
new residual capacity is introduced.

The scalar loss is mean squared error over `cell_observed` only. An entirely
unobserved chart has exact zero loss. The detached DINO target is never an
online value input.

## 3. Axes, dtype, zero and scale semantics

| Value | Shape | Contract |
|---|---|---|
| parent/corrected owner | `[B,N,K+1]` | FP32 conditional probability |
| conditional K | `[B,N,K]` | sums to one for finite binder rows |
| prior / validity | `[B,N,1]` | outside all owner softmaxes |
| reconstruction assignment | `[B,N,K]` | FP32 `conditional_K * prior * validity` |
| structured reconstruction owner | `[B,K,C,Y,X]` | sum over local M only |
| exported content | `[B,K,D]` | sole K-specific reconstruction value |
| shared position | `[B,C,Y,X,D]` | no K axis; gated by summed physical support |
| target / reconstruction | `[B,C,Y,X,D]` | target detached; output restored to chart dtype |
| observed mask | `[B,C,Y,X,1]` | boolean producer fact, detached for loss reduction |

Zero semantics:

- zero validity gives exact zero reconstruction assignment;
- an all-invalid cell cannot be resurrected by conditional K;
- a common K residual changes no conditional owner;
- changing G3 K identity preserves parent real mass and null mass exactly;
- zero slot residual recovers the former exported online content;
- zero observed cells give a finite exact-zero reconstruction loss.

No magnitude target, minimum object mass, entropy quota, forced non-null
reward, normalization budget or fixed reconstruction gain is added.

## 4. Complete consumer and reverse-gradient map

Online consumers of the unique exported content are exhaustive under
`clearvla/mainline/model`:

- S: `StatelessObjectIntentOrganizer._object_tokens` projects
  `facts.content` without detach. Intent/coarse/action losses can therefore
  reach the grounder.
- W: `ObjectFutureDynamicsCompiler._base` projects `facts.content` without
  detach. Online W and its future supervision can reach the grounder.
- W field: `_field.current_reference` is detached only at the explicit
  successor-reference boundary; W hidden deltas still inherit the ordinary
  content path through `_base`.
- Teacher: current content, assignments, chart values and chart addresses are
  deliberately detached. Teacher is a frozen target compiler and supplies no
  reverse path into G.
- `FutureObjectDynamics.neutral` is a typed fallback helper using the same
  exported content; it creates no parameter or alternate value.

The reconstruction reverse paths are:

```text
object reconstruction loss
  <- observed-cell reduction
  <- reconstructed chart
  <- conditional-K reconstruction assignment
  <- bounded G3 residual and final physical binder

object reconstruction loss
  <- ObjectFactSet.content
  <- candidate read + exported slot residual
  <- completed G3 candidates / slots / physical binder

object reconstruction loss
  <- shared decode_position
  <- observable local coordinates
```

No gradient reaches `target_dino_content` or `cell_observed`. There is no
detach between the reconstruction loss and the exported content.

The exact null-independence claim is deliberately narrow and testable:
association null cannot scale or switch off `Q` when conditional K is held
fixed. The physical K+null binder still legally helps produce the one online
`ObjectFactSet.content`; forbidding its ordinary gradient there would require
a detach or a second content definition and would violate the retained
physical-owner contract. Both Schema30 and Schema35 donors retain that
physical read. No separate learned-null token, decoder value or reconstruction
gate is legal.

## 5. Loss, runtime, optimizer and checkpoint ownership

- `ObjectIntentDynamicsTop.build_training_targets` forwards exactly
  `context.facts.reconstruction_error`; deployment never constructs Teacher
  targets but still constructs the online facts once.
- `training.losses.compose_losses` retains the existing nested coefficient:
  `intent_structure * 0.50 * 0.25`. G-02 changes neither the coefficient nor
  any other objective.
- The grounder runs once in `encode_online`. Cached deployment velocity calls
  consume S/W/P state and do not recompute reconstruction or G.
- Every `top.grounder.*` trainable remains in optimizer role `grounder` under
  the inherited AdamW decay and clipping policy.
- `decode_content_residual.weight` and `decode_position.weight` retain their
  existing names, shapes and optimizer ownership. No parameter or optimizer
  tensor is added or removed by G-02.
- Exact resume serializes the complete model and optimizer key sets and is
  already incompatible with R0 because R1a removed
  `transition.interval_identity`. R1 remains fresh-checkpoint-only. G-02 adds
  no migration shim or permissive load.

## 6. Diagnostics and bypass audit

Retained detached diagnostics observe reconstruction MSE, dense objective
count, physical mass conservation, K/null entropy, chart entropy, G3 change,
content similarity, typed-read differences, camera geometry and candidate
scale.

G-02 may add only direct detached audits of the repaired algebra, such as
parent-versus-corrected null error or reconstruction support. Such metrics
cannot enter loss, routing, stopping thresholds or normalization. The later
D-01 unit still owns broad diagnostic redesign.

Forbidden alternate paths:

- no reconstruction from an online self-generated target;
- no K+null G3 re-softmax;
- no private `decoded_slot` differing from `ObjectFactSet.content`;
- no learned null in reconstruction assignment;
- no target DINO use in S, W, Teacher current keys or binder candidate keys;
- no Schema27 typed consensus, Schema35 content-only owner, public-content ABI,
  new gain, floor, quota or hard gate.

## 7. Test-first acceptance set

Before implementation, add failing tests requiring:

1. dense target identity with detached `target_dino_content`, not the online
   candidate mixture, and exact zero loss on an all-unobserved chart;
2. conditional reconstruction assignment equals
   `conditional_K * prior * validity`, sums to `prior * validity`, stays FP32,
   rejects misaligned axes and is exact zero for invalid rows;
3. arbitrary G3 K residuals preserve the final binder's real mass and null
   mass while allowing conditional K identity to change;
4. the existing slot content residual changes the exported
   `ObjectFactSet.content` by the same K value and changes reconstruction only
   through that exported value;
5. reconstruction gradients reach candidate content, the exported residual
   owner and G3 assignment owner, but not the detached target;
6. the Schema25 candidate-token function still responds to content, semantic,
   appearance, geometry and coordinates, proving no deferred binder rewrite
   entered G-02;
7. the complete retained suite, parameter inventory, optimizer ownership,
   call-count and diagnostics-equivalence tests remain valid.

## 8. Resolved assumptions and edit authorization

No blocking assumption remains.

The apparent choice between deleting the private decoder and keeping it was
resolved from the complete boundary: deleting it is unnecessary hardening,
while leaving its output private violates unique content ownership. Folding
the existing zero-initialized residual into `ObjectFactSet.content` preserves
capacity, initial online values, parameter names and downstream access.

The existing coordinate decoder was also resolved from its axes: it has no K
input and, when written outside the K einsum behind summed conditional support,
is a shared spatial value rather than a second object value. It is therefore
retained without a new scale or centering rule.

Authorized implementation surface after the failing tests are observed:

- `clearvla/mainline/model/grounding.py`: detached target, conditional-K
  helper/correction, exported residual, null-independent reconstruction
  assignment, explicit shared position algebra and observed-cell loss;
- `tests/test_mainline_structural_contracts.py`: G-02 mechanism, intervention,
  gradient, axis and anti-donor tests;
- current architecture/replay documents: record the implemented semantics and
  verification result.

No type, intent, W, Teacher, loss-weight, optimizer, runtime, launcher,
manifest or checkpoint-loader edit is authorized for G-02.

## 9. Implementation and verification result

Implemented source surface:

- `dense_chart_from_local_facts` now exposes the producer-detached current
  DINO target rather than rebuilding a target from online local slots;
- G3 factors the final binder posterior into real mass and conditional K,
  changes only the latter, and concatenates the unchanged parent null mass;
- `_conditional_k_reconstruction_assignment` applies the retained local prior
  and observable validity outside the conditional-K softmax in FP32;
- the existing `decode_content_residual` output is now included in
  `ObjectFactSet.content` before every reconstruction/S/W/Teacher consumer;
- reconstruction uses conditional-K assignment, the one exported content and
  an explicitly K-independent shared coordinate term;
- reconstruction MSE is reduced only over the producer's observed-cell mask;
- one detached null-identity diagnostic was added; it has no consumer in loss
  or routing.

Observed test-first result before implementation:

```text
4 failed, 2 passed, 42 deselected
- dense chart still used the online self-mixture as target
- conditional-K reconstruction helper did not exist
- a distinct K residual changed absolute real/null assignment mass
- slot content residual changed reconstruction but not exported content
```

The two tests already passing before implementation were intentional guards:
ordinary reconstruction gradients were present, and every Schema25 binder
input still affected the candidate key.

Post-implementation checks:

| Check | Result |
|---|---|
| G-02 focused mechanisms | PASS: 6/6 |
| Complete structural-contract file | PASS: 48/48 |
| Complete retained ten-file mainline suite | PASS: 129/129 in 34.56 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff on touched source and test | PASS |
| Pyright touched mainline source | PASS error gate: 0 errors; 618 existing unknown-type warnings |
| Pyright touched test | PASS error gate: 0 errors; 1,647 existing unknown-type warnings |
| `git diff --check` | PASS; only repository line-ending notices |

The retained suite increased from 123 to 129 because G-02 adds six mechanism
tests and removes no existing test.

Production parameter and optimizer inventory is unchanged from R1a:

| Field | R1a | R1b | Delta |
|---|---:|---:|---:|
| Total parameters | 169,979,847 | 169,979,847 | 0 |
| Trainable parameters | 153,585,526 | 153,585,526 | 0 |
| Parameter tensors | 1,413 | 1,413 | 0 |
| Trainable/optimizer tensors | 1,075 | 1,075 | 0 |
| Grounder parameters | 4,007,936 | 4,007,936 | 0 |
| Grounder optimizer tensors | 17 | 17 | 0 |

Both existing checkpoint fields remain exact:

- `top.grounder.decode_content_residual.weight`;
- `top.grounder.decode_position.weight`.

No state key, optimizer group, parameter shape, manifest field or
checkpoint-loader rule changed in G-02.

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/model/grounding.py` | `d3fd7b9552ff3e1b86aa08318548bae5018c71c1` | `3C204B7CD45F09004F722E8C24DCB0E4EA8E14195B988CA347110025D3338BE3` |
| `tests/test_mainline_structural_contracts.py` | `18a902a01e6e6569b410202683e9e128edfa612e` | `BFC2CE1DCC4C7482C8F13DCAF1FD0CD467A2AC8D21347D6842B0BE03DDEA61C5` |

Forward re-review found the detached target used only by the one observed-cell
MSE, the original five candidate-key inputs intact, real/null mass preserved,
and exactly one K-specific reconstruction value reaching S, W and detached
Teacher. The coordinate decoder has no K input and is applied only after
summing conditional-K support.

Independent consumer-to-producer review found action/intent/W losses reaching
the unique exported content through ordinary autograd, reconstruction reaching
both that content and conditional assignment, and no gradient reaching the
DINO target. Teacher's content and assignment reads remain explicitly
detached. Source search found one invocation of `decode_content_residual` and
no `decoded_slot` or second K-specific value. Unresolved assumptions remain
empty.
