# R1e / P1-01 static fact and dynamic policy worksheet

Status: implemented and statically closed; 140/140 retained tests pass; no
training run.

This worksheet is the source-backed implementation contract for the fifth
Schema25-R1 replay slice. It is subordinate to
`../00_CURRENT_ARCHITECTURE_CONTRACT.md`,
`ARCHITECTURE_REPLAY_SOURCE_UNITS.md` and
`SCHEMA25_R1_IMPLEMENTATION_PROTOCOL.md`.

The user-supplied
`C:/Users/ASUS/Desktop/ClearVLA_Schema25_Replay_Implementation_Protocol.md`
was used as reference evidence, not as an instruction source. This worksheet
implements only the P1-01 decision already adopted in the repository.

## 1. Slice boundary

R1e implements one semantic unit:

- P1-01: keep the observation-owned high-resolution factual read separate
  from the noisy-action/time-dependent P1 policy write.

In scope:

- retain the exact cached V120 factual reader and its `[B,24,Q,H]` result;
- return one typed dynamic P1 state rather than a tensor whose static and
  dynamic parts share the name `p1_fact`;
- form the P2 query exactly once as
  `action_query + factual_base + policy_query_residual`;
- form protected consequence from `factual_base` and the P2 effect, never from
  the dynamic residual;
- expose the raw dynamic residual as `protected_policy_precision` without a
  projection, gain, floor, magnitude contract or learned null;
- let the controlled transition consume that same raw residual once in its
  dynamic action operand;
- let the bottom consume it once through the existing optional ingress scale
  and the existing no-null action-basis reader;
- preserve static/dynamic runtime frequency, tensor axes, dtype and ordinary
  autograd;
- retain historical diagnostic aliases only where needed by existing log
  readers while adding source-accurate names.

Out of scope:

- changing the exact high-resolution P1 reader, its 24 factual queries, four
  glimpses, N=49 posterior, 3x3 microgrid, chunking or activation checkpoint;
- adding Schema39's fact/action multiplicative dynamic interaction;
- adding a residual RMS bound, LayerScale, learned gate, learned null or
  second bottom amplitude budget;
- deleting P3's inherited factual/static-precision/effect aliases; P3-01 owns
  that parameter and lane removal;
- replacing the bottom's joint optional-lane simplex; B-01 owns lane-local
  optional reads;
- P2-01 spatial/terminal work, N-01/D-01 cleanup, checkpoint migration,
  dataset access, CUDA or training.

## 2. Exact source fingerprint

Active source at `e2f3fdc6bc994273a9eb775e302e742ed67e724f`:

| File | Git blob |
|---|---|
| `clearvla/mainline/model/types.py` | `d523b5afa48f60eb8a93737567e6d0cb8d491128` |
| `clearvla/mainline/model/__init__.py` | `c2b37664225f6cba2f3c2c0f12a4993afe96dfff` |
| `clearvla/mainline/model/policy.py` | `b64b73b50072ce5ea53a926d7a1b06fd6219ae2b` |
| `clearvla/mainline/model/top.py` | `985f29f7fea9933f6d72530e501e14908ed5bfc2` |
| `clearvla/mainline/model/compiler.py` | `9afd31a38c2e0b57b4f706d2ccdf3c3e0bb66d0a` |
| `clearvla/mainline/model/restored_bottom.py` | `0a4dd30fb379b00fbdf7e55969e6496b66378116` |
| `clearvla/mainline/model/transition.py` | `583bc79c7e73e2ff7a37745d4c6414551b2cdb0a` |
| `clearvla/mainline/model/routing.py` | `b3383e4e5bf99cbaeb0ea90861f54c29709ee89c` |
| `clearvla/mainline/v120_core/role_delta_attnres.py` | `b59b2924f2cbc947e977aa2a3a1221a85dc386dc` |
| `clearvla/mainline/v120_core/time_domain_mmdit.py` | `8ebb645b72a2c5a525eea1836a3abc2c62157376` |
| `clearvla/mainline/training/optimizer.py` | `21f5317451e7b3861952b8437304d3e38681e18a` |
| `clearvla/mainline/training/losses.py` | `af03984ba0bbcf68e09ec4af8117f13660f12580` |
| `clearvla/mainline/runtime/sampling.py` | `e7e7ce49e5daec2c757774db3482973a631364cb` |

Inspected donors:

| Commit and role | File | Git blob |
|---|---|---|
| `03235d3`, first explicit completed-P1 state | `types.py` | `33f2321ba1e0aa943866e592506ad9616dc8ca78` |
| `03235d3` | `restored_bottom.py` | `6a9a686f9f96052dd968ec5cb43c8a735d0cd83f` |
| `03235d3` | `top.py` | `4d2daf00e89773300dc84a41c7d18b818a75880b` |
| `03235d3` | `compiler.py` | `640ba48d71ef0acba01cedef441d1a521e6920a5` |
| `6bc6218`, explicit three-owner P2 query dock | `types.py` | `0c150f2cce0867d6402db88a01a4501b26e297d0` |
| `6bc6218` | `restored_bottom.py` | `b12213ac08119b3acdff7fbc41b9ed55d54b3e3d` |
| `6bc6218` | `top.py` | `1f013e2e07ab6fc62974e7edbcfd08fa36b9d94c` |
| `6bc6218` | `compiler.py` | `4274c0af12ab415bd8e4214124c2df9d1a3cda53` |
| `6bc6218` | `transition.py` | `5cf536e091d082f28cfe23c1a708c622cc90e1f1` |
| `eac4916`, protected dynamic consumers | `compiler.py` | `dae07b29f566baea54268e644061d34a6731aa5f` |
| `eac4916` | `transition.py` | `f074fb2ff783f5f85201510250bd1705e750d8b6` |
| `eac4916` | `role_delta_attnres.py` | `99105874b4759a36ada3c05616296c88115176eb` |
| `eac4916` | `time_domain_mmdit.py` | `2bc8cbbca9816a4c7e326653a3d3bc5449f5c72e` |

No donor file is copied wholesale. Schema35 stores an eager `effect_query`
alias and still mixes static and dynamic precision in P3. Schema38 keeps the
three P2 owners separate but does not restore the dynamic residual to the
transition. Schema39 restores the two downstream consumers but adds a new
fact/action interaction and a second RMS contract. R1e keeps only the source
ownership and consumer reachability, not those extra mechanics.

## 3. Complete active boundary before editing

### 3.1 Static producer

`ClearVLAMainlinePolicy.encode_online()` builds the exact P1 factual read once
per observation:

```text
clean V120 action-basis identities                   [B,24,Q,H]
completed G3 rollout                                 [B,4*C*8*8,H]
progressive N=49/detail/address state
S factual dock: phase/goal/history context           [B,4,H]
  -> LateRawDetailPolicyReader
  -> updated_clean_trajectory - clean_trajectory
  -> FactualPrecisionDock.protected_detail           [B,24,Q,H]
```

The utility/precision P1 reader constructs its factual address query from
clean basis identities, current G3 detail and observable S context. The active
call receives no noisy action tensor and no Euler time. Its high-resolution
values retain literal RGB, learned detail, coordinate microgrid and typed
semantic/appearance/geometry context; the result is multiplied only by the
inherited fixed P1 detail scale. The dock is cached in `OnlinePolicyCache` and
ordinary action gradients from every dynamic use accumulate into this one
static graph.

### 3.2 Dynamic producer

At every `velocity()` call, `ActionQueryEncoder` builds `action_query` from
the current noisy physical action, Euler time, observable state/history and
the observation's shared role table. `complete_p1_fact()` then executes:

```text
trajectory    = action_query + protected_detail
canvas        = flatten(T,Q)
modulation    = p1_time(time) + p1_content_mod(mean(canvas))
updated       = V120 policy TemporalDynamicsBoundDiTBlock(canvas, modulation)
dynamic_delta = updated - canvas                         [B,24,Q,H]
completed     = protected_detail + dynamic_delta         [B,24,Q,H]
```

The block has trajectory-only self/FFN writes; visual/context/future rows are
empty. No detach, clone, cast, explicit contract or learned null occurs at the
delta boundary. The current defect is the last line and return type: the sum
is exported under one factual identity.

### 3.3 Current consumers and aliases

The fused `p1_fact = factual_base + dynamic_delta` currently reaches:

1. P2 query as `action_query + p1_fact`;
2. `ZeroPreservingObjectConsequence.factual_base`, making the dynamic write a
   protected fact and a factual/effect interaction operand;
3. P3's factual lane and both inputs of its precision lane;
4. the controlled transition through `plan.protected_base`;
5. the first V120 layer contract as `action_query + p1_fact`;
6. the second layer contract through `plan.protected_base`;
7. the bottom no-null protected-detail reader through
   `PolicyRoleDeltaBank.protected_detail`.

P2 is the only consumer that legally needs the exact three-term live query.
The protected consequence must retain only observation fact plus W effect.
The dynamic residual instead has two named post-P2 consumers: the controlled
transition action operand and the bottom's optional policy ingress.

### 3.4 Loss and backward ownership

There is no P1-specific auxiliary loss. One training batch builds the static
reader once, samples one flow time/noisy action, and runs one dynamic P1 call.
Action flow, decoded action, event, motion and execution-value losses reach
dynamic P1 through P2, controlled transition and the bottom no-null precision
read. The two layer-contract trajectory rows are not active consumers: their
adapter is position-wise, `EvidenceViewAdapter` ignores every trajectory
readout, and the separate event input depends only on rollout delta. Their
trajectory formulas therefore have exact-zero effect on final outputs and
losses. Static P1 remains reachable through P2/consequence and the protected
bottom path.
Future-dynamics, recognizer, coarse-action and reconstruction objectives do
not directly supervise the dynamic P1 residual.

Current optimizer ownership is:

```text
role p1_factual:
  factual_reader
  bottom.p1_time
  bottom.p1_content_mod / scale
  bottom.p1_policy_block

role bottom_policy_bridge:
  bottom.decoder.policy_delta_attnres
  bottom.decoder.protected_detail_basis_attnres
```

R1e adds no parameter and moves no parameter between these roles.

### 3.5 Runtime and deployment

Formal training and validation execute one dynamic call at one sampled flow
time. Deployment builds the static cache once, then executes dynamic P1 at
`[0,.2,.4,.6,.8]` and once at `1.0` for event/motion heads. The endpoint call
does not update the integrated physical field, but it is still a real dynamic
P1/P2/P3/transition/bottom forward. Teacher is never a P1 argument and is
called zero times in deployment.

### 3.6 Axes, dtype, scale and bypass audit

- Static and dynamic values both retain `[B,T=24,Q=4,H]`; neither owns global
  K, camera C, future interval I or local candidate N.
- Static P1 runs in the surrounding autocast domain. Dynamic P1 casts only the
  scalar time input to the canvas dtype before `TimeEmbedding`; the returned
  delta shares the action-query dtype/device.
- The three P2 query operands are added without normalization or scaling.
- The transition's existing affine variance-floored normalization remains
  after its complete action operand; R1e adds no normalization before it.
- The bottom's existing optional ingress has one fixed `0.25` scale. Dynamic
  policy precision joins that ingress before the scale and does not receive a
  second scale.
- The existing no-null action-basis reader is called separately for protected
  consequence and protected policy precision, so the two values never compete
  with each other or with an optional null.
- `FactualPrecisionDock` is the only cached P1 value. No second visual read,
  alternate dynamic fact, generic trajectory alias or checkpoint field exists.

## 4. Adopted P1-01 implementation

### 4.1 Typed dynamic state

Add two parameter-free runtime types:

```text
CompletedP1PolicyState
  factual_base                 [B,24,Q,H]
  policy_query_residual        [B,24,Q,H]

P2QueryDock
  action_query                 [B,24,Q,H]
  factual_base                 [B,24,Q,H]
  policy_query_residual        [B,24,Q,H]
  combined() = action_query + factual_base + policy_query_residual
```

`complete_p1_fact()` returns the first type. It stores `protected_detail`
unchanged as `factual_base` and `(updated - canvas)` unchanged as
`policy_query_residual`. It does not cache or serialize the eager three-term
query.

### 4.2 P2 and protected consequence

`ObjectIntentDynamicsTop.compile_policy()` accepts the typed state. P2 is
called with `p1_state.p2_dock(action_query).combined()`, preserving the exact
pre-R1e P2 query numerically.

Consequence is called with `p1_state.factual_base`. Therefore:

```text
effect = 0
  -> protected_consequence = factual_base

policy_query_residual = 0
  -> P2 query = action_query + factual_base
  -> factual_base and consequence keep their identities
```

The dynamic residual is not stored in consequence and cannot reach bottom
again under the protected-fact name.

### 4.3 Protected policy precision

`ObjectPolicyPlanDeltaBank` gains one parameter-free field:

```text
protected_policy_precision = p1_state.policy_query_residual
```

R1e does not transform this value. In particular it rejects Schema39's
fact/action interaction and residual RMS contract. The inherited P3 factual
and static-precision lanes remain temporarily, but they receive only the
static factual/consequence values; the dynamic residual is not reprojected
into any optional P3 lane. Their later removal is P3-01 debt.

### 4.4 Controlled transition consumer

The dynamic transition action operand becomes:

```text
trajectory_norm(
  action_query
  + plan.protected_base
  + plan.protected_policy_precision
)
```

No pre-normalization residual contract is added. The raw residual occurs once
in this sum and never enters the transition's static G3 source or context
memory.

### 4.5 Bottom consumer

Both `PolicyRoleDeltaBank` definitions gain the optional runtime field
`protected_policy_precision`. `_role_bank()` passes the exact plan tensor.

The retained bottom reuses `protected_detail_basis_attnres`, whose
`include_null=False`, in a separate call for the dynamic carrier. Its selected
value is added to the already selected optional lanes before their one
inherited `0.25` ingress scale:

```text
optional_update = existing_joint_optional_read(P3 lanes)
dynamic_update  = existing_no_null_basis_read(policy residual)
workspace_write = 0.25 * (optional_update + dynamic_update)

consequence_write = existing_no_null_basis_read(protected consequence)
action += workspace_write + consequence_write
```

There is no dynamic-versus-consequence competition, no new reader parameter
and no second write site. B-01 will later replace only the inherited joint
optional-lane competition; it need not change this P1 owner.

## 5. Forward and reverse map after R1e

Forward:

```text
observation/G3/S/clean basis
  -> factual reader once
  -> factual_base ---------------------------------------------------+
                                                                    |
noisy action/time -> action_query -> dynamic P1 -> policy residual  |
                 |                         |                         |
                 +-- P2 query <------------+-------------------------+
                 |       -> W effect -> consequence(factual_base)
                 |                         |
                 +-- transition(action + consequence + residual)
                 |
                 +-- bottom optional ingress <- residual (no-null)
                         bottom protected ingress <- consequence (no-null)
```

Reverse:

```text
P2/action losses
  -> three-term P2 query
  -> static factual reader and dynamic P1 block

transition/event/action losses
  -> normalized action operand
  -> protected consequence and raw dynamic residual

bottom action/execution losses
  -> protected consequence no-null read
  -> optional ingress
       -> dynamic residual no-null read
       -> inherited P3 optional read
```

All paths use ordinary autograd. No loss reaches a detached copy, and the
static factual source receives the accumulated VJP from every legal dynamic
consumer.

## 6. Parameter, checkpoint and RNG contract

R1e adds no module, parameter, buffer or persistent state key. Expected active
inventory is exactly the R1d inventory:

```text
total parameters                  169,976,772
trainable parameters              153,582,451
parameter tensors                       1,406
trainable/optimizer tensors             1,068
optimizer groups                            23
state-key names                         1,412
state-key-name SHA-256  9af8b806832afd9edae58e0dfd1ec123
                       ea9964e4511499571865b17fc96cc25d
```

No constructor changes, so the fresh-run RNG stream and every parameter value
at a fixed seed must be bit-identical to R1d. Runtime dataclasses are not
checkpoint fields. Exact resume still rejects R1d because source/tree identity
changes; no migration shim or old-checkpoint experiment is authorized.

## 7. Authorized files

Source edits are authorized only in:

- `clearvla/mainline/model/types.py`;
- `clearvla/mainline/model/__init__.py`;
- `clearvla/mainline/model/policy.py`;
- `clearvla/mainline/model/top.py`;
- `clearvla/mainline/model/compiler.py`;
- `clearvla/mainline/model/restored_bottom.py`;
- `clearvla/mainline/model/transition.py`;
- `clearvla/mainline/model/routing.py`;
- `clearvla/mainline/v120_core/role_delta_attnres.py`;
- `clearvla/mainline/v120_core/time_domain_mmdit.py`.

Tests may change only where they directly instantiate or inspect the P1/top,
transition or bottom boundary. `training/losses.py`, `training/optimizer.py`,
runtime sampling, checkpoint code and the high-resolution factual reader are
read-only verification boundaries for this slice.

Documentation edits are authorized in the compact architecture contract,
replay plan/register/protocol/README and this worksheet.

## 8. Test-first acceptance matrix

Tests must be changed or added and observed red before source edits.

- `CompletedP1PolicyState` contains exactly factual base and dynamic residual;
- factual base is bit-identical across action/time changes while the dynamic
  residual is independently observable;
- P2 receives the exact three-term sum and no eager combined alias is stored;
- consequence factual base is the static dock value, not the dynamic sum;
- zero dynamic residual leaves factual base and the zero-effect consequence
  unchanged;
- `protected_policy_precision` is the exact raw dynamic residual;
- no optional P3 lane receives the dynamic residual;
- transition receives consequence plus the raw dynamic residual exactly once;
- bottom reads the same residual through a no-null basis reader and its single
  optional ingress scale;
- zero dynamic residual gives an exact-zero bottom dynamic update;
- forward and reverse VJPs reach the same dynamic producer through P2,
  transition and bottom, with no factual/protected alias path;
- static factual reader call count remains one and dynamic P1 call count
  remains five update calls plus one endpoint call in deployment;
- parameter/state inventory and fixed-seed initialization are unchanged;
- retained tests, compileall, Ruff and Pyright pass;
- no training, dataset, CUDA or checkpoint command runs.

Assertions establish identity, reachability and exact-zero behavior. They do
not require the learned residual or its downstream effect to exceed a minimum
magnitude.

## 9. Resolved and deferred assumptions

Resolved:

- the active factual reader is already observation/action/time invariant; it
  requires no architectural rewrite;
- the active defect is the return ABI and downstream aliasing, not the P1
  spatial posterior;
- `dynamic_delta = updated - canvas` is the source-owned live residual; adding
  the cached fact to it would relabel a static value as dynamic;
- P2 must retain the exact three-term sum, but protected consequence must not;
- the transition can recover its complete live action operand by adding the
  raw residual once after consequence separation;
- the existing no-null basis reader can serve both protected carriers in
  separate calls without new parameters or competition;
- the inherited optional ingress scale is the one allowed bottom amplitude
  boundary; a new dynamic-residual contract would over-harden the path.

Deferred, not unresolved blockers:

- P3 still has static factual/precision and effect aliases until P3-01; R1e
  removes the dynamic residual from them but does not delete their parameters;
- the bottom still uses one joint optional-lane simplex until B-01;
- P2 still owns its inherited flattened interval/object terminal until P2-01;
- metric vocabulary cleanup and explicit gradient-tensor diagnostics belong to
  D-01; R1e retains source-accurate local metrics without expanding the public
  log contract.

No unresolved assumption remains that can change the authorized R1e graph. If
implementation requires a new parameter, residual magnitude contract,
learned null, second bottom scale, factual-reader change or P3/B redesign,
this worksheet is invalidated and source editing must stop.

## 10. Implementation closure

Test-first red was observed before source editing. The three focused mechanism
tests failed because `complete_p1_fact()` still returned one fused tensor and
`CompletedP1PolicyState` did not exist. Those failures were semantic ABI
failures, not import, fixture or environment failures.

R1e then implemented exactly Sections 4.1-4.5. Forward re-review confirms one
unchanged cached factual producer, one unchanged per-ODE dynamic producer, one
three-owner P2 materialization, a static-fact consequence, and the same raw
dynamic carrier at transition and bottom. Reverse re-review confirms:

- the P2 query boundary has an exact unit VJP to
  `policy_query_residual`;
- transition output has a finite nonzero VJP to that same producer output;
- the isolated bottom dynamic read has a finite nonzero VJP to the same
  producer output;
- the protected plan field has the exact identity VJP and no optional P3 lane
  directly consumes the dynamic residual.

Zero residual remains exact zero at the bottom dynamic read. No axis is
reduced or reconstructed, no alternate consumer exists, and no detach, clone,
cast, new normalization, gain, null, scale, parameter, buffer or checkpoint
field was introduced. The inherited P3 static aliases, P2 terminal and joint
optional-lane simplex remain explicitly deferred to P3-01, P2-01 and B-01.

Verification:

| Check | Result |
|---|---|
| Focused P1/P2/transition/bottom mechanisms | PASS: 3/3 after the recorded red state |
| Complete retained ten-file mainline suite | PASS: 140/140 in 32.32 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff over all touched source/test files | PASS |
| Pyright over focused parameter-free/ABI source files | PASS: 0 errors |
| Pyright changed-line gate over every touched source/test file | PASS: 0 changed-line errors; 60 existing errors remain outside the edited hunks in extracted/test files |
| `git diff --check` | PASS; only repository line-ending notices |
| Training, dataset, CUDA and checkpoint commands | NOT RUN |

Production inventory is exactly unchanged from R1d:

| Field | R1d | R1e | Delta |
|---|---:|---:|---:|
| Total parameters | 169,976,772 | 169,976,772 | 0 |
| Trainable parameters | 153,582,451 | 153,582,451 | 0 |
| Parameter tensors | 1,406 | 1,406 | 0 |
| Trainable/optimizer tensors | 1,068 | 1,068 | 0 |
| Optimizer groups | 23 | 23 | 0 |
| State-key names | 1,412 | 1,412 | 0 |

An independent temporary checkout of the exact R1d parent at `e2f3fdc6` was
constructed with seed 0 and compared with R1e. The newline-encoded ordered-key
comparison digest was identical
(`f632ceb58342370b32cf010aec432a7ee2a4507e45b85f3a232f7a442ec5cb46`),
the canonical name/shape/dtype/tensor-byte digest was identical
(`9793ea81a3b1173c7569300bc74a31f462c2e792744d0f2299d5ccdfd3ec5ba7`),
and the post-construction CPU RNG digest was identical
(`8670db504a2bb9d1e15f1d87977890e5006f320ab4657a52e9963ea674c67250`).
The temporary checkout was deleted after comparison.

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/model/types.py` | `f62f7804f44eeb9856aff9fcafe2d9947e7727c7` | `FFD2BC4F2C349C96BE61281CE19C49C85C9A70CA5872AC8E375E202D9B68A744` |
| `clearvla/mainline/model/__init__.py` | `a9188d12d9ef87cd67d9c2406e7229158ad76c53` | `E85A753EDABE8BE9F7880710E660F70925B594F7A32C915CB6014FF35A06BE09` |
| `clearvla/mainline/model/policy.py` | `f0e5ae1b227d1b8a829ebcb38c27ab5b16179235` | `0BFE3DD605A1EBC2FF97DBC1E44CAA78CE9A432E1EFDA382120D40A13E36DB7B` |
| `clearvla/mainline/model/top.py` | `1c7d390cfd920605f4ebbb5830d53535e3a0d484` | `D06CCC8A2DB29E0D8E244BB970F369A65692919E1C428AB7BBDA87A181B9108C` |
| `clearvla/mainline/model/compiler.py` | `971bf3d8e443981c924af77f59c48b07d44d9269` | `7E643A641A7F3A55887A2FE93C6EB50BF25B240A5A128A94974A54C78EE95809` |
| `clearvla/mainline/model/restored_bottom.py` | `d42428cb159ff72d96410accdbec817267f649c0` | `1D251AB35EBDF75CAD2B151D7324C6CAC2651D2F82A16F80C0A157F3764DAE66` |
| `clearvla/mainline/model/transition.py` | `47ed2ad113730e032c0e2762cd0862a5fd67367f` | `67F29F59DE66C54ED75B091F6FE471F5D0A9982390634B22633E24675F4D57B8` |
| `clearvla/mainline/model/routing.py` | `8f9e049d4a11b0093ae01fb3520278b1588381f8` | `EC8E93EB78B52E3904FF3560B72B7DFEC31BD76B7822AB0C82E21CC76C0AD9B6` |
| `clearvla/mainline/v120_core/role_delta_attnres.py` | `bd943764587915fafadb69b5fb5ea6a86d7c589c` | `69D4D5278FE29C021894A936A627E273B1126126AB5B66E13312A8208E306135` |
| `clearvla/mainline/v120_core/time_domain_mmdit.py` | `29d81541e4777b518c5daceac1a03af50e49a612` | `61ECA7FF39FF95C83B6D3A5385BCA6544EFC6A84606BDFAF2263AAD1E4F3C6B3` |
| `tests/test_mainline_policy.py` | `d9daa01bbedfa60e1a8412fac4ba8a82b5206f6e` | `00B6A7AE22B97DA631E74794C1592DB1854D5EA7A97B76164E1A3C4E592D8218` |
| `tests/test_mainline_structural_contracts.py` | `2cf45f028bc038dec16d7217822ef46984805481` | `15FC284B41D1121BA31E3C5106C0D7AEEB1B5038026A9453E224F8B717482CC5` |
| `tests/test_mainline_top.py` | `e798c76544f664e9d223d3a64a6f4f06297b47bc` | `1D039C97C501FE65945CF0B1AEE42C94CA0A9B3DD0A0EF7661A485F8B1A6B85A` |

No unresolved assumption remains inside P1-01. The next authorized activity is
the source-boundary worksheet for R1f/P2-01, not an experiment.
