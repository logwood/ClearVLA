# R1g / P3-01,B-01 unique-lane and bottom-ingress worksheet

Status: implemented and statically closed; 145/145 retained tests pass; no
training run.

This worksheet authorizes only R1g/P3-01 and B-01. It is based on the active
R1f source and the adopted replay contract. Donor implementations are evidence
for the defect and separation mechanics, not code to copy wholesale.

No training, dataset, CUDA or checkpoint command is authorized in this slice.

## 1. Exact owner inventory

The active R1f plan exposes seven semantic names for four actual values:

```text
protected_base                 protected consequence
protected_policy_precision     raw dynamic P1 residual
factual                        projection of already-protected static fact
precision                      another projection of static fact/consequence
effect                         projection of effect already in consequence
temporal                       private temporal innovation
state_change                   private observable state-change innovation
```

The first three optional aliases duplicate protected owners. They then enter a
single `[five lanes * action bases]+null` bottom competition, where unrelated
lanes suppress one another before their meanings are consumed.

R1g retains exactly four values:

1. protected consequence: factual base plus contracted P2 effect plus its
   zero-preserving interaction;
2. protected policy precision: the exact raw dynamic P1 residual;
3. optional temporal innovation;
4. optional observable state-change innovation.

There is no optional factual, static-precision or effect lane. This is an
ownership deletion, not a capacity simplification: protected consequence and
dynamic precision retain their direct no-null reads, while the two values with
genuinely optional private operands retain learned-null routing.

## 2. Pre-edit source boundary

| File | Git blob |
|---|---|
| `clearvla/mainline/model/compiler.py` | `c5f1ed3204fbdec8078329bdbf71ce8638917669` |
| `clearvla/mainline/model/top.py` | `65d55b55f4def53d3045e53e683f875e35896f39` |
| `clearvla/mainline/model/restored_bottom.py` | `d42428cb159ff72d96410accdbec817267f649c0` |
| `clearvla/mainline/v120_core/time_domain_mmdit.py` | `29d81541e4777b518c5daceac1a03af50e49a612` |
| `clearvla/mainline/v120_core/role_delta_attnres.py` | `bd943764587915fafadb69b5fb5ea6a86d7c589c` |
| `clearvla/mainline/model/routing.py` | `8f9e049d4a11b0093ae01fb3520278b1588381f8` |
| `clearvla/mainline/model/transition.py` | `47ed2ad113730e032c0e2762cd0862a5fd67367f` |
| `clearvla/mainline/training/optimizer.py` | `21f5317451e7b3861952b8437304d3e38681e18a` |
| `clearvla/mainline/runtime/checkpoints.py` | `0598a096133ea0775de64e8b3a29b7ac4c8bc7fa` |
| `clearvla/mainline/runtime/logging.py` | `766608f755d6c983a26866da966ac4cca26ed571` |
| `clearvla/mainline/manifest.py` | `929de40cb29dafadd677c22aa2196d7e5c8453ee` |
| `tests/test_mainline_manifest.py` | `6785ebb3177ea147030f6ff9eeea712c934756c6` |

### 2.1 P3 producer and transformation map

`ObjectPolicyPlanCompiler` runs once per dynamic policy call after P2 and
receives the exact post-consequence action query, `ObjectConsequenceState`, the
raw P1 residual and reduced S policy context. It owns no auxiliary loss.

The R1g temporal private source is:

```text
S.temporal_control [B,T,H] expanded over Q
  + bias-free projection(P2 effect + factual/effect interaction)
  -> multiply by tanh(bias-free action relation)
  -> bias-free temporal lane
  -> the one inherited lane smooth-RMS contract
```

It never projects `factual_base` or the complete `protected_consequence` into
the optional value. The action relation may observe the already-defined P3
action query; this is conditioning, not a second factual value owner. With
both temporal S and consequence innovation zero, the lane is exact zero.

The state-change private source is:

```text
S.state_change_evidence [B,H] expanded over T,Q
  * tanh(bias-free action condition + bias-free S-temporal condition)
  -> bias-free state-change lane
  -> the one inherited lane smooth-RMS contract
```

Zero state-change evidence is exact zero regardless of action/temporal
condition. R1g removes the fixed `0.05` multiplier and both inherited
`sqrt(2)` divisors. It introduces no replacement gain, floor, quota or minimum
route.

`ObjectPolicyPlanDeltaBank` contains only `protected_base`,
`protected_policy_precision`, `temporal` and `state_change`. Its optional
source axis is therefore exactly two and its names are exactly
`p3_temporal,p3_state_change`.

### 2.2 Bottom transformation map

`RestoredV120EvidenceBottom._role_bank` carries the two optional values as
`[B,S=2,T,Q,H]` and the two protected values separately as `[B,T,Q,H]`.

The shared optional reader is invoked once per semantic lane:

```text
temporal [B,T,Q,H]     -> one Q-basis + zero-null simplex
state_change [B,T,Q,H] -> one independent Q-basis + zero-null simplex
                           (same parameters, separate normalization)
lane reads add
  + protected dynamic precision through a separate no-null call
  -> one existing optional-ingress fixed scale
```

Changing one lane's values/logits cannot renormalize the other lane. Parameters
remain shared so R1g adds no lane-specific capacity or quota. The optional
lane sum receives no new aggregate RMS contract: each P3 lane already retains
its inherited contract and the bottom keeps its existing candidate contract
and fixed ingress scale.

Protected consequence and protected dynamic precision make separate calls to
the existing shared no-null Q-basis reader. Parameter sharing cannot create a
competition because the softmax invocations are disjoint. Protected
consequence remains outside the optional scale; protected precision joins the
optional sum before that scale exactly once.

The decoder performs one write at each dynamic node:

```text
action = action + scaled_optional_and_precision + protected_consequence_read
```

CVAE, workspace, transition, layer contracts, all three Evidence MMDiT blocks,
ordered contractions, execution controller and final action/event/motion heads
remain reachable and unchanged.

### 2.3 Consumers, losses and backward paths

P3 optional lanes have no direct target. Action flow, decoded action,
event/motion and execution-value losses backpropagate through the lane-local
bottom reads. The transition consumes only protected consequence, raw dynamic
precision and the plan interface; removing optional aliases does not remove a
transition operand. Future W, recognizer, coarse-action, reconstruction and
Teacher losses do not directly supervise P3/bottom routing.

Reverse ownership from the bottom write is:

- protected consequence -> no-null Q reader -> factual/P2 consequence owners;
- protected precision -> no-null Q reader -> exact raw dynamic P1 owner;
- temporal -> its own Q+null posterior -> temporal S and consequence
  innovation, with action as condition;
- state-change -> its own Q+null posterior -> observable S state-change owner,
  with action/temporal S as conditions.

No value reaches bottom twice under another semantic name. No lane axis is
flattened with another lane before its own null decision.

### 2.4 Runtime, optimizer and checkpoint identity

P3 and the bottom reader run at the existing five ODE updates plus the endpoint
head call. Observation/G/S/W, static P1 and transition source remain cached.
Teacher remains absent from deployment.

Optimizer role `p3_compiler` loses only deleted alias parameters. Role
`bottom_policy_bridge` keeps the same parameter owners; its optional
`source_key` tensor shrinks from `5*Q` rows to Q rows. `RoleDeltaAttnRes` may
draw the historical row count and retain only the active prefix so all live
reader weights and every downstream fresh-run tensor keep their controlled
initialization stream. Discarded rows are not registered, serialized, moved or
executed.

Strict exact resume from R1f is intentionally incompatible: six P3 projections
disappear, one retained projection is renamed to its true effect meaning, and
the optional bottom `source_key` changes shape. The manifest therefore names
both the protected-plus-two-optional P3 ABI and lane-local bottom ABI. No
partial-load or migration shim is authorized. The final parameter/state/RNG
delta must be measured.

## 3. Over-hardening decisions

Resolved:

- temporal and state-change are complementary optional innovations, so they
  add after independent null decisions and do not share a probability simplex;
- the optional reader shares parameters across lanes; separate parameters
  would add unsupported type-specific capacity;
- no new aggregate lane-sum contract is added; the existing per-lane P3
  contract, candidate contract and one bottom fixed scale remain sufficient;
- the two protected carriers share parameters but never a normalization call;
  adding a second no-null module is unnecessary capacity;
- removing alias parameters is preferred to freezing or retaining dead state;
- initialization-only discarded draws preserve downstream fresh-run identity
  without preserving dead model capacity;
- zero optional input may remain zero and null mass may take any learned value;
  no minimum optional contribution is required.

No unresolved assumption remains that can change the authorized R1g graph. A
need for another optional value, a second bottom scale, per-lane parameters,
new contract, protected null, target/loss, or CVAE/workspace/controller edit
invalidates this worksheet and stops source editing.

## 4. Test-first acceptance matrix

Tests must be observed red before production editing and cover:

- the plan contains exactly two optional lanes and two protected carriers;
- no optional factual, static-precision or effect attribute/parameter remains;
- zero consequence innovation plus zero temporal S gives exact-zero temporal;
- zero state-change S gives exact-zero state-change for arbitrary conditions;
- changing factual base alone cannot change the P3 optional values when the
  action condition and consequence innovation are held fixed;
- raw P1 residual is bit-identical in `protected_policy_precision` and never
  enters an optional projection;
- P3 source contains no `0.05` or `sqrt(2)` fixed attenuation;
- the optional reader serializes Q source identities, not `lane*Q`;
- temporal and state-change invoke independent Q+null normalizations with
  shared parameters;
- intervening on one lane leaves the other lane's posterior exactly unchanged;
- protected consequence and precision readers have no null and remain
  independent of optional null mass;
- optional lanes and precision meet before the existing fixed scale exactly
  once; consequence remains outside it;
- the decoder write count and all retained consumers remain unchanged;
- legal forward and reverse paths are finite and nonzero without magnitude
  quotas;
- retained tests, compileall, Ruff, Pyright and `git diff --check` pass;
- no training, dataset, CUDA or checkpoint command runs.

Authorized production edits are limited to P3's plan ABI/compiler, the
plan-to-bottom adapter, the active role reader's initialization/normalization
boundary, their direct call sites and the serialized component identities that
must reject the old state ABI. Tests and compact replay/current architecture
documents may be updated. Diagnostics vocabulary cleanup beyond the exact
changed producers belongs to R1h/D-01.

## 5. Implementation closure

The test-first red state established four independent defects: the plan still
published five optional source names, the compiler still required the removed
factual-detail alias, bottom serialized and normalized `5*Q` source rows in one
call, and the manifest still named the five-lane graph. The focused tests then
turned green only after the production boundary matched Sections 1-3.

Forward re-review from producer to consumer confirms:

- temporal is built only from S temporal context and consequence innovation
  under action conditioning; state-change retains its S-private multiplicative
  owner;
- both optional values preserve exact zero and receive their inherited P3
  lane contract once;
- the same bottom reader is invoked in two disjoint Q+null calls, whose raw
  outputs add without a new aggregate contract;
- dynamic precision receives a separate no-null call before the existing fixed
  optional scale, while consequence receives a separate no-null call outside
  that scale;
- CVAE, workspace, transition, Evidence MMDiT, execution controller and final
  heads retain their original consumer paths and call counts.

Reverse re-review confirms finite nonzero VJPs from the P3 outputs to effect,
temporal S, state-change S and action conditions, and from the bottom optional
write to both lane values. Intervening on temporal leaves the state-change null
posterior bit-identical. No duplicate optional carrier, cross-lane simplex,
axis reconstruction, hidden scale, detach or alternate bottom write remains.

Inventory delta:

| Field | R1f | R1g | Delta |
|---|---:|---:|---:|
| Total parameters | 170,009,540 | 168,436,164 | -1,573,376 |
| Trainable parameters | 153,615,219 | 152,041,843 | -1,573,376 |
| Parameter tensors | 1,408 | 1,402 | -6 |
| Trainable/optimizer tensors | 1,070 | 1,064 | -6 |
| Optimizer groups | 23 | 23 | 0 |
| State-key names | 1,414 | 1,408 | -6 |
| P3 parameters/tensors | 3,145,728 / 12 | 1,572,864 / 6 | -1,572,864 / -6 |
| Bottom optional source-key | 20 x 32 | 4 x 32 | -512 parameters |

The six removed H-to-H projections account for `-1,572,864`; the sixteen
discarded bottom key rows account for `-512`. The ordered state-key-name
SHA-256 is
`14effa3654b11923088be6b57f3086a78db82e11daff1cfc91805f20bf7f3540`.
Historical initialization-only draws keep the seed-0 post-construction CPU RNG
SHA-256 exactly R1f-identical at
`d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21`.
The new manifest SHA-256 is
`03ce3702b2253fe04a9109194eb951fe4942ce6f47e41b3ab8b5e9749c9f9051`.
Exact R1f resume is rejected; no migration path was added.

Verification:

| Check | Result |
|---|---|
| Focused P3/bottom/manifest mechanisms | PASS after recorded red states |
| Complete retained ten-file mainline suite | PASS: 145/145 in 47.83 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff over touched source/test files | PASS |
| Pyright changed-line gate over six production files | PASS: 0 changed-line errors; 44 existing errors and 300 warnings outside edited lines |
| `git diff --check` | PASS; repository line-ending notices only |
| Training, dataset, CUDA and checkpoint commands | NOT RUN |

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/manifest.py` | `9d13c572ad331ba16c770f70740da97a2a912be3` | `8582F43B4DB3814D7E15F34BE063912348D10639E75AF4D344B4720F0DF9E9AD` |
| `clearvla/mainline/model/compiler.py` | `366b9c14810005510f6106e0c4070d136cb2bcc8` | `516FBB4740D6061912536BB65EAF26E9E04688C5485439DD501ED8B639C2526E` |
| `clearvla/mainline/model/restored_bottom.py` | `bd0cf3dfe9f6ae1b4df914ec5efa0eec5e3ea58a` | `F7E48B4FC5EEE7BAD13B676F5FD61CFF29A876790BF409C7162E325ADB39523A` |
| `clearvla/mainline/model/top.py` | `e49c205b839a752916d6225c88dfc2d8fc3025f0` | `E7D6C2F46F95821401E5FF1E306A727E9EE29E9A468D5F9BEC0CA5C3BF2A76A0` |
| `clearvla/mainline/v120_core/role_delta_attnres.py` | `0e3d912d5b09b286956f25591665c070f72d82e0` | `21EA69C743630BA38D77F4486F2A3905416B7C4306DDC2E9E6F04EE62F48F561` |
| `clearvla/mainline/v120_core/time_domain_mmdit.py` | `079e2351b5854532f98a1a399554fc6f8aa6403f` | `F5C7145DBC2BAE69F3F86FAE713BCB612B88700CF1416575170273AFE2F155A9` |
| `tests/test_mainline_manifest.py` | `19a463a94fa74bed7ba0f25853c213809a5ee63a` | `C5F16FD849D2B98AC886DA7A215A0FAEE5B16EB43F22F7223B8DFEAB3C9A842E` |
| `tests/test_mainline_policy.py` | `e8e6cfa8a19d0e0a4e9eb6b25e9f6cf2e78e8b09` | `B6AAF8E3990633439A0B9FE2E0DF7DD72C057596061D6FEFFDFFBC814A7EF010` |
| `tests/test_mainline_structural_contracts.py` | `122290eb7347b22d8f6770042a2196ecdf73734f` | `66DFC650D469ECBA46A2D5515F596A0C37DEA83AEA0AF780B8FEB1C83F6C126C` |
| `tests/test_mainline_top.py` | `3833faa00efa926ff8867f49f70499798d4b00b3` | `52BE1B407AF8D8C8CB3A617CA312CA0DE44924DEBE40A37A60DCBE2051289E6F` |

No unresolved assumption remains inside P3-01/B-01. R1h/N-01,D-01 is the
next replay slice; this closure does not authorize an experiment.
