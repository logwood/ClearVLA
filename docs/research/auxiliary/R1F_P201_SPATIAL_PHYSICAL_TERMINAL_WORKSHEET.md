# R1f / P2-01 spatial-selection and physical-terminal worksheet

Status: `BOUNDARY CLOSED BEFORE SOURCE EDIT`

This worksheet authorizes only R1f/P2-01. It follows the compact current
architecture contract and the adopted Schema25 replay protocol. Donor code is
source evidence, not a wholesale implementation target.

No training, dataset, CUDA or checkpoint command is authorized in this slice.

## 1. Defect and exact replacement

The active P2 reader currently flattens the physical interval and object axes
into one `[I*K]+null` competition. Geometry first reduces camera and then
shares that same flattened posterior. A learned two-type softmax finally makes
semantic and geometry suppress each other. This removes the interval axis at
the spatial consumer and gives a null/type competition authority over two
complementary evidence owners.

R1f replaces that terminal with two named stages:

```text
P2 query [B,T,Q,H]
  + semantic W field [B,I,K,D]
  + geometry W field/covariance [B,I,K,C,2|3]
  + observable G support [B,K,1] / [B,K,C,1]
    -> semantic spatial posterior over K, independently for every I
    -> geometry spatial posterior over K*C, independently for every I
    -> SelectedIntervalEvidence [B,T,Q,I,Z=2,H]

the same W-owned spatial posterior selects existing S metadata
  + reduced public S interval context
    -> zero-preserving W-key conditioning
    -> one no-null four-physical-interval posterior per Z
    -> selected W common once + selected interval innovation
    -> semantic + geometry complementary latent sum
    -> the one existing caller-owned P2 smooth-RMS contract
```

The final `+` between semantic and geometry is an adopted latent-fusion
operator, not a claim that their source coordinates are physically additive.
Its static acceptance is independence: either legal owner survives alone,
neither receives a type gain or type softmax, and their joint pre-contract
result is exactly the sum of the two isolated pre-contract results.

## 2. Source-backed active boundary

Pre-edit source fingerprints:

| File | Git blob |
|---|---|
| `clearvla/mainline/model/compiler.py` | `971bf3d8e443981c924af77f59c48b07d44d9269` |
| `clearvla/mainline/model/types.py` | `f62f7804f44eeb9856aff9fcafe2d9947e7727c7` |
| `clearvla/mainline/model/top.py` | `1c7d390cfd920605f4ebbb5830d53535e3a0d484` |
| `clearvla/mainline/model/dynamics.py` | `e7a8ac35112bf6a8a315490a64c036816e2ee6c7` |
| `clearvla/mainline/model/teacher.py` | `b3ef8d4b1172c6fbe844471ed9ea97ab5b2a09a2` |
| `clearvla/mainline/training/losses.py` | `af03984ba0bbcf68e09ec4af8117f13660f12580` |
| `clearvla/mainline/training/optimizer.py` | `21f5317451e7b3861952b8437304d3e38681e18a` |
| `clearvla/mainline/model/restored_bottom.py` | `d42428cb159ff72d96410accdbec817267f649c0` |
| `clearvla/mainline/model/transition.py` | `47ed2ad113730e032c0e2762cd0862a5fd67367f` |
| `clearvla/mainline/training/checkpoints.py` | `0598a096133ea0775de64e8b3a29b7ac4c8bc7fa` |

### 2.1 Producers and axes

`ObjectFutureDynamicsCompiler` is the only online W producer. It exports:

- semantic successor/delta `[B,I=4,K,D]`;
- transport mean `[B,I,K,C,2]` and FP32 PSD covariance `[B,I,K,C,3]`;
- current observable object support `[B,K,1]`;
- current camera coordinates/support `[B,K,C,2|1]`.

Its `semantic_common`/`semantic_interval_innovation` and
`transport_common`/`transport_interval_innovation` properties are exact
mean-plus-centered-residual views. They add no parameter or new value owner.
Teacher and future-dynamics losses supervise those same W fields and retain C;
there is no P2-specific loss.

`ObjectIntentState` already owns typed S common and interval-residual metadata
at `[B,K,3,R]` and `[B,I,K,3,R]`. `PolicyIntentDock` currently hides them.
R1f exposes those existing tensors read-only; it does not add an S producer,
mass, support, value, selector or loss. P2 semantic maps to S type 0 and P2
geometry maps to S type 2. Appearance remains outside P2.

`CompletedP1PolicyState.p2_dock(action_query).combined()` is the sole producer
of the complete dynamic P2 query `[B,T=24,Q,H]`.

### 2.2 Spatial transformation

Semantic selection retains I and normalizes only K. Geometry selection retains
I and normalizes only flattened K*C after consuming transport, current camera
coordinate and covariance in the same candidate score. Both use only positive
observable G availability as legal support. Support is boolean authority;
availability magnitude is its existing FP32 log measure, not a predicted
validity or a newly learned amplitude gate.

All-invalid rows use a finite masked softmax with an exact-zero output and
finite backward. No candidate-count correction, learned null, uniform fallback
or predicted W validity is allowed.

The spatial posterior selects, with identical weights:

- the projected W key;
- projected common W value;
- projected interval-innovation W value;
- existing typed S metadata.

The selected S tensor is metadata only. Intervention on S cannot change the
spatial posterior, selected W key/value, or support.

### 2.3 Physical interval terminal

For each semantic/geometry type independently:

```text
s = tanh(bounded(selected_S_context))
conditioned_W_key = selected_W_key + selected_W_key * s
interval_score = relation(action_query, conditioned_W_key)
posterior = safe_masked_softmax(interval_score, physical I support)
value = posterior-selected common + posterior-selected innovation
```

This multiplication is the zero-preserving relation: W key zero remains zero
for every S/action input. S provides no independent interval logit. There is
no null candidate at this physical terminal; when all four intervals lack
observable support the output is exactly zero.

No inner RMS contract is added. The terminal returns the raw complementary
sum to `ObjectIntentDynamicsTop.compile_policy`, where the already-existing
single `smooth_rms_contract(raw_effect, 0.35)` remains the only P2 amplitude
boundary.

### 2.4 Consumers, loss and runtime frequency

The raw P2 effect has one consumer: the caller-owned P2 contract. The
contracted effect then enters `ZeroPreservingObjectConsequence`, P3, controlled
transition and the retained bottom through protected consequence. There is no
alternate raw effect, typed sidecar or checkpoint field.

P2 runs once per dynamic policy/ODE call. W and S are cached once per
observation. Deployment never invokes Teacher. Action, event, motion and
execution-value losses can backpropagate through consequence to both spatial
and interval relations; future W losses supervise W directly but do not read
P2.

Optimizer group `p2_effect_reader` owns only `top.effect_reader.*`. Changing
the P2 module inventory changes fresh-run model/state identity and makes exact
resume from the R1e state incompatible. Bottom-only migration remains
unaffected because no bottom field changes in R1f. The slice must record the
new parameter/state inventory and fresh-run RNG consequence rather than
silently claiming compatibility.

## 3. Reverse ownership audit

From the final raw effect back to producers:

1. complementary sum has one unit VJP to each semantic/geometry selected
   value;
2. each selected value is exactly selected common plus selected innovation;
3. each value uses its type-local interval posterior;
4. the interval posterior is action-to-conditioned-W-key only;
5. the conditioner is multiplicative in W, so S cannot create a route when W
   is zero;
6. every selected W/S tensor uses the same type-local spatial posterior;
7. semantic posterior removes K only, geometry posterior removes K*C only;
8. support comes only from current observable G fields;
9. W retains its direct supervised loss independently of the P2 path.

No detach, clone-as-owner, axis reconstruction, `expand`-fabricated camera,
cross-type posterior, learned null, type softmax, per-type gain or second
outer contract is legal.

## 4. Two-level trajectory audit discovered before R1f

The two retained V120 layer-contract trajectory tensors are bookkeeping-only
in the active graph. `LayerContractAdapterHeads` is position-wise, so
trajectory rows cannot affect rollout/state rows. `EvidenceViewAdapter` reads
only `rollout_tokens`, `state_tokens` and `state_history_tokens` from each
layer contract; its separate event input is computed from rollout delta.
Midcut trajectory/action/motion outputs are not consumed by final action,
event, motion or any active loss.

Therefore changing either layer-contract trajectory formula has exactly zero
active behavioral or gradient effect. R1f does not edit this dead path. The
R1e worksheet is corrected to exclude layer contracts from claimed dynamic-P1
gradient consumers. Effective dynamic-P1 consumers remain P2, controlled
transition and the bottom no-null precision read.

## 5. Test-first acceptance matrix

Tests must be observed red before implementation and cover:

- `PolicyIntentDock` exposes the exact existing typed common/residual S views;
- `SelectedIntervalEvidence` is `[B,T,Q,I,2,H]` with boolean `[B,I,2]`
  support and exact `value = common + innovation` identity;
- a one-interval sentinel cannot move before the terminal;
- changing S cannot change spatial W selection or support;
- semantic selection removes only K and geometry removes only K*C;
- covariance changes the geometry posterior/result before C disappears;
- all-invalid support gives finite exact-zero forward and backward;
- zero W gives exact-zero output for arbitrary S/action;
- semantic-only and geometry-only inputs survive independently;
- the joint raw result equals the sum of isolated semantic and geometry raw
  results;
- no learned null, type softmax, fixed type divisor or per-type gain exists;
- terminal posterior is independently normalized over the four physical
  intervals for each type;
- only the existing caller-owned P2 contract remains;
- retained tests, compileall, Ruff, Pyright and `git diff --check` pass;
- no training, dataset, CUDA or checkpoint command runs.

## 6. Closed assumptions and edit boundary

Resolved:

- use current observable chart/camera availability only; do not import donor
  `camera_weights`, predicted visibility or extra log fields absent from R1e;
- retain the active covariance metric `I + Sigma`; a variance-floor redesign
  is not required for the P2 ownership repair and belongs to N-01 only if a
  source-backed numerical defect remains;
- select existing S typed value metadata after the W-owned posterior; do not
  use S masses as support or amplitude;
- public interval S context may join the selected typed context only after
  spatial selection and only inside zero-preserving key conditioning;
- semantic and geometry sum as complementary latent contributors before the
  one inherited outer contract; this is an explicit fusion design, not a
  physical-units claim;
- remove the inherited learned null/type-softmax parameters; do not preserve
  dead parameters merely for state compatibility;
- fresh-run initialization/state identity may change and must be measured;
  exact resume across R1e/R1f is rejected rather than partially loaded.

Authorized production edits are limited to the P2 reader and the read-only
S dock fields/call site required by it. Tests and compact replay/current
architecture documents may be updated. Any need for a new support producer,
W field, S selector, loss, type gain, second contract, P3 change or bottom
change invalidates this worksheet and stops R1f source editing.

## 7. Implementation closure

Test-first red was observed before production editing. Four focused tests
failed on the absent `source_query`/`spatial_select` boundary and absent typed
`PolicyIntentDock` views. These were the intended semantic ABI failures; the
existing test environment and fixtures loaded successfully.

R1f then implemented exactly Sections 1-3. Forward re-review confirms:

- semantic removes K separately inside every I and geometry removes K*C
  separately inside every I;
- covariance and transport are consumed together before C disappears;
- `SelectedIntervalEvidence` retains `[B,T,Q,I,2,H]`, boolean `[B,I,2]`
  support and exact selected common-plus-innovation identity;
- S changes only selected metadata, never spatial W selection or support;
- each type removes I through its own four-physical-interval softmax with no
  null, then semantic and geometry add before the unchanged caller contract.

Reverse re-review confirms finite nonzero ordinary VJPs from the raw effect to
semantic W, camera transport W, typed common S, typed interval-residual S and
the action query on legal support. With no observable support, forward output
is exact zero and backward remains finite. With W values zero, arbitrary S and
action cannot create an effect. No support, value, posterior or independent
interval logit is owned by S, and no alternate raw effect consumer exists.

The final parameter delta is deliberately minimal for the missing meanings:

| Field | R1e | R1f | Delta |
|---|---:|---:|---:|
| Total parameters | 169,976,772 | 170,009,540 | +32,768 |
| Trainable parameters | 153,582,451 | 153,615,219 | +32,768 |
| Parameter tensors | 1,406 | 1,408 | +2 |
| Trainable/optimizer tensors | 1,068 | 1,070 | +2 |
| Optimizer groups | 23 | 23 | 0 |
| State-key names | 1,412 | 1,414 | +2 |
| P2 parameters/tensors | 1,575,939 / 9 | 1,608,707 / 11 | +32,768 / +2 |

The new geometry W-key projection and two type-specific existing-S route
projections replace the removed type-query competition. No null, type gain,
new support field, loss, buffer, contract or bottom parameter was introduced.
Exact resume from R1e is rejected by strict state identity. The R1f ordered
state-key SHA-256 is
`b31e565546456d89eef9add6b1c62df61a64c2282c07ce8fee1a58e9e368afa4`;
the seed-0 post-construction CPU RNG SHA-256 is
`d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.

Verification:

| Check | Result |
|---|---|
| Focused P2 mechanisms and reverse path | PASS: 5/5 after the recorded red state |
| Complete retained ten-file mainline suite | PASS: 144/144 in 43.37 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff over touched source/test files | PASS |
| Pyright over focused production files | PASS: 0 errors; environment import/type warnings only |
| `git diff --check` | PASS; repository line-ending notices only |
| Training, dataset, CUDA and checkpoint commands | NOT RUN |

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/model/compiler.py` | `c5f1ed3204fbdec8078329bdbf71ce8638917669` | `588856FDD1E492CB5142FDAC660BBF2FAA665E6173E23767FEF41B44CC9CBAD1` |
| `clearvla/mainline/model/types.py` | `4387ba6d4fb2aa4245357c3832c227b38727bb99` | `D31C404CCE46CAA700A525608A79B76A3E8877DC74FF6B6E02C1364C4FAD9B30` |
| `clearvla/mainline/model/top.py` | `65d55b55f4def53d3045e53e683f875e35896f39` | `9DFA85181817FF99C1294F85D56F02A58839066D040D668BDB245F85E866662E` |
| `tests/test_mainline_structural_contracts.py` | `7ec6cfad8924c13c1138dcc64f13582cd90137f5` | `E606A82CEB99E23FF4C3205BF6F24C71203E4A94C645EFF6FA231A817F48DD57` |
| `docs/research/auxiliary/R1E_P101_STATIC_DYNAMIC_P1_WORKSHEET.md` | `87bcd188b50e6565f7bbd789b39ad74eb364d645` | `8FA98CB3491AC927A0D9E15E0B5238DB9D774828F986A102F8D1074CF3D79899` |

No unresolved assumption remains inside P2-01. The next authorized activity is
the R1g/P3-01,B-01 source-boundary worksheet, not an experiment.
