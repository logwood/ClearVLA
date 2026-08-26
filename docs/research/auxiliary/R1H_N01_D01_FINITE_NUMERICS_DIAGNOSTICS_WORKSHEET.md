# R1h / N-01,D-01 finite-numerics and diagnostics worksheet

Status: implemented and statically closed; 155/155 retained mainline tests
pass; LC-01 inventory/RNG sentinels remain exact; no training run.

This worksheet authorizes only the final R1 support slice. It is based on the
active R1g+LC-01 source and the adopted replay contract. Schema38/39 donors are
evidence for a numerical defect or an observation mechanic; they do not
authorize importing their later representation graph.

No training, dataset, CUDA or checkpoint command is authorized in this slice.

## 1. Exact pre-edit source boundary

| File | Git blob |
|---|---|
| `clearvla/mainline/v120_core/flow_dino_evidence.py` | `f9375e9b07e0b15cb18768f1742e0156e3bc86c7` |
| `clearvla/mainline/v120_core/grounded_intent_effect.py` | `e39fcc6b595263fce5256234f2079a2e7a9972b6` |
| `clearvla/mainline/model/restored_observation.py` | `de5ad6258eddca6ea91e32d690e185501c6f599c` |
| `clearvla/mainline/model/types.py` | `4387ba6d4fb2aa4245357c3832c227b38727bb99` |
| `clearvla/mainline/model/grounding.py` | `d3fd7b9552ff3e1b86aa08318548bae5018c71c1` |
| `clearvla/mainline/model/dynamics.py` | `e7a8ac35112bf6a8a315490a64c036816e2ee6c7` |
| `clearvla/mainline/model/teacher.py` | `b3ef8d4b1172c6fbe844471ed9ea97ab5b2a09a2` |
| `clearvla/mainline/model/compiler.py` | `366b9c14810005510f6106e0c4070d136cb2bcc8` |
| `clearvla/mainline/model/policy.py` | `8aa389b1d6689b7cd643310a7ae11b5ee47bf6a2` |
| `clearvla/mainline/model/routing.py` | `8f9e049d4a11b0093ae01fb3520278b1588381f8` |
| `clearvla/mainline/training/engine.py` | `df61a31c133afc0aaaf1d2e5ae7323ed6dbe34b4` |
| `clearvla/mainline/training/optimizer.py` | `21f5317451e7b3861952b8437304d3e38681e18a` |
| `clearvla/mainline/train.py` | `394f2639a7e5b85b900dd2c709921c97f53c84b2` |
| `clearvla/mainline/runtime/logging.py` | `766608f755d6c983a26866da966ac4cca26ed571` |
| `clearvla/mainline/runtime/checkpoints.py` | `0598a096133ea0775de64e8b3a29b7ac4c8bc7fa` |
| `clearvla/mainline/manifest.py` | `2fb2c6118aea1a13e3153514f31dcc98ddd24275` |

## 2. N-01 forward dataflow map

### 2.1 Zero-preserving variance boundary

The active G address path contains four direct variance square roots:

```text
G1 coarse posterior -> aligned/coarse variance
  -> G2 dynamic-candidate geometry standard deviation
  -> coarse geometry key and high-resolution candidate radius
  -> G2 rectifier geometry and coordinate-correction scale
  -> rectified coordinates/support -> G3 facts -> P1 and global-object G
```

The forward values are finite at exact zero, but the derivative of direct
`sqrt(variance)` is unbounded there. Exact-zero variance is legal for identity
motion and concentrated address posteriors, so this is an active backward-path
defect rather than a speculative guard.

The authorized replacement is the analytic map
`sqrt(v + epsilon^2) - epsilon`, written in its cancellation-safe form
`v / (sqrt(v + epsilon^2) + epsilon)`. It is exact zero at `v=0`, has finite
slope `1/(2*epsilon)`, introduces no parameter, buffer, learned gate, minimum
contribution or loss, and uses the address-grid resolution as the existing
physical scale. The G2 correction keeps its historical nonzero epsilon scale
by using `safe_std + epsilon`; no other square root is swept into this change.

Only detached diagnostics for the live variance, transformed standard
deviation and analytic maximum gain are authorized.

### 2.2 FP32 probability/log-measure boundary

The active restored path is:

```text
G2 typed slot evidence [B,C,Y,X,M]
  -> slot log-softmax and probability over M
  -> bounded G3 residual update in log space
  -> GroundedFactSet probability + producer log probability (FP32)
  -> RestoredV120ObservationCompiler -> LocalFactSet
  -> semantic/geometry geometric-mean local prior in log space
  -> DenseFactChart FP32 probability + finite log prior
  -> one existing K+null binder and existing typed reads
  -> ObjectFactSet observable object/camera support + producer logs (FP32)
  -> W1/W2 or detached Teacher-G pass-through
  -> FutureObjectDynamics probability + log measures (FP32)
  -> P2 semantic and camera-supported geometry selection
  -> existing all-invalid exact-zero masked terminal
```

Today G2/G3 probabilities are cast to the rollout dtype before the restored
observation boundary, and downstream code reconstructs logs with fixed
`1e-8`, `1e-6` or `1e-30` floors. That can erase a legal low-probability owner
and changes its relative evidence before P2. The authorized repair retains the
producer's FP32 log-softmax next to its FP32 probability, keeps both through
the active ABI, and performs normalizations from the log form. Finite zero is
used only as the stored value for unsupported entries; the boolean support is
the sole authority for interpreting it.

The existing G identity, local M axis, global K+null competition,
conditional-K reconstruction, typed verification reads, exported content,
camera coordinates, W values, P2 type ownership and losses remain unchanged.
Compact legacy test fixtures may omit producer logs and use an FP32 fallback;
the active restored observation path must supply all three typed logs.

Observable validity/probability fields remain FP32 through `ObjectFactSet` and
`FutureObjectDynamics`. W and Teacher do not infer a new measure: they copy the
current fact producer's probability and log values. P2 consumes those producer
logs directly, and its existing all-invalid support semantics remain finite
exact zero.

## 3. N-01 reverse gradient map

The reverse paths that must remain live are:

- action/event/motion loss -> P2 posterior -> W semantic/transport values ->
  G object content/geometry and owner log probabilities;
- future semantic/transport loss -> W outputs -> typed S/W states -> current G
  facts;
- dense reconstruction loss -> conditional-K assignment and exported object
  content -> local G candidates and binder parameters;
- P1 factual loss path -> progressive address coordinates/support -> the same
  zero-preserving variance transforms;
- Teacher targets remain detached and cannot create a future-to-online
  gradient path.

No epsilon may create support, resurrect an invalid candidate, impose a
minimum owner mass or replace an all-invalid zero. No probability/log field is
detached on the online path. Diagnostics alone detach their inputs.

## 4. D-01 observation boundary

### 4.1 Source-tensor gradient metrics

The authorized hook is a scalar slot registered on an already-existing live
tensor. During backward it copies only a detached FP32 RMS into that slot and
returns the original gradient object unchanged. Axis metrics are allowed only
when the exact producer owns the named axis and the name count equals its
width.

Matching R1 owners are:

- S public interval carrier and typed common/interval-residual values;
- static P1 protected fact and dynamic P1 policy-query residual;
- W2 semantic/appearance/geometry common and completed interval innovation;
- P2 semantic/geometry effect values before their complementary sum;
- P3 protected dynamic precision, temporal and state-change values.

Removed P3 factual/precision/effect aliases, removed layer trajectories,
historical status lanes and old W contribution aliases are not recreated.

### 4.2 Finite raw-gradient spike attribution

The existing lifecycle remains exactly:

```text
backward -> finite raw global norm
  -> optional finite-spike attribution on the already-detected rare path
  -> bottom.decoder local clip -> global clip -> optimizer -> scheduler
```

Attribution scans raw parameter gradients only after the finite global norm
exceeds the configured audit threshold. It records the exact max-L2 and
max-absolute parameter, role, optimizer group, dtype and shape before any
clipping. A six-channel flow/uncertainty split is emitted only when the actual
max-L2 owner is the observation `delta_head` and its first axis is exactly six.
The ordinary below-threshold path performs no parameter scan.

The per-window gradient summary is host-only bookkeeping over the already
materialized scalar. It records weighted mean, maximum and current value plus
the owning batch/step. A partial epoch-tail window is emitted rather than
discarded. None of these values participates in loss, clipping or routing.

### 4.3 Runtime metric vocabulary

Console diagnostic groups are an allow-list of metrics with a live R1
producer. Stale ancestry names are deleted rather than aliased. Exact-zero
mass/identity metrics and `gradient_tensor_*` values remain visible when zero;
absence continues to mean that a producer did not run.

## 5. Runtime, optimizer and checkpoint identity

N-01/D-01 adds no `nn.Parameter`, persistent buffer, optimizer owner, loss or
random draw. The retained parameter tensor count, trainable tensor count,
optimizer partition, ordered state-key digest, retained tensor bytes, decoder
bytes and post-construction CPU RNG digest must therefore remain LC-01 exact.

The runtime/dataclass ABI and serialized component identity do change: an old
exact checkpoint did not assert the producer-owned FP32 log/support contract or
the matching diagnostic runtime. The manifest must name the new ABI, so strict
resume from R1g+LC-01 is rejected. No partial migration shim is authorized.

Deployment uses the same restored observation, S/W/P and bottom call sites.
Gradient hooks are registered only during training diagnostics; spike
attribution and gradient windows are training-runtime observations only.

## 6. Over-hardening decisions

Resolved:

- keep the single existing K+null binder; do not import Schema39's later
  content-only/typed-key binder redesign;
- keep the current exported object content and reconstruction algebra; do not
  add public-content or private-value carriers;
- preserve current camera support semantics; do not add camera evidence mass,
  reliability gates or a new availability amplitude;
- keep P2's two physical types and its exact-zero all-invalid behavior; do not
  add status, a null terminal, per-type gains or minimum mass;
- repair only the four donor-confirmed address variance uses; do not replace
  every square root in the repository;
- retain probability and log forms because they are two numerical views of one
  existing measure, not two learned representations;
- diagnostics may observe only live owners and may not create a loss, target,
  thresholded route, clipping rule or capacity;
- use a single finite-spike threshold only to decide whether to serialize an
  expensive audit; it cannot affect the update;
- remove stale console vocabulary instead of recreating dead producers.

No unresolved assumption remains that can change the authorized R1h graph. A
request for new evidence/reliability semantics, a binder redesign, a new
learned scale, a new loss, a new route, a probability floor that creates
support, channel attribution without the exact owning ABI, or a change to
clipping/update order invalidates this worksheet and stops source editing.

## 7. Test-first acceptance matrix

Tests must be observed red before production editing and cover:

- zero variance maps to exact-zero standard deviation with finite bounded VJP;
- the four active address variance consumers use that map and no broader sqrt
  rewrite occurs;
- masked softmax returns finite exact zero and finite gradients on an
  all-invalid row;
- G2/G3 probability and producer log probability remain FP32 and consistent;
- slot permutation changes probabilities and logs identically, while the mean
  intervention produces the exact uniform probability/log pair;
- the active restored boundary rejects missing typed logs while legacy test
  fixtures retain an explicit fallback;
- a low-probability owner lost by BF16 probability rounding remains distinct
  through the FP32 producer log path;
- object and camera observable measures/logs remain finite FP32 through online
  W, neutral W, Teacher-G, permutation and P2;
- all-invalid P2 rows remain exact zero without NaN/Inf forward or backward;
- gradient hooks fill only on backward and return the gradient unchanged;
- axis hook names must exactly match the producer-owned axis;
- diagnostics enabled/disabled produce identical loss and parameter updates;
- finite spike reports identify the real max-L2/max-absolute owners before
  clipping, and flow channel splits require the exact six-channel owner ABI;
- below-threshold batches skip the expensive owner scan;
- gradient windows own weighted mean/max/current and flush partial epoch tails;
- console rows emit current R1 names and refuse removed ancestry names;
- manifest changes while all parameter/state/optimizer/RNG fingerprints remain
  LC-01 exact;
- retained tests, compileall, Ruff, Pyright and `git diff --check` pass;
- no training, dataset, CUDA or checkpoint command runs.

Authorized production edits are limited to the active numerical producer/log
ABI and its direct consumers; zero-preserving address conversion; detached
diagnostic helpers and matching hook sites; finite-spike/window runtime; metric
allow-list; and serialized component identities. Tests and the compact replay
and current architecture documents may be updated after closure.

## 8. Implementation and bidirectional closure

The test-first state produced eight expected failures and one passing control
before production edits. The implementation then stayed inside Sections 2-5:

- the grid/radius-owned zero-preserving variance transform replaces only the
  four active address uses, with detached variance/std/gain diagnostics;
- G2 produces typed FP32 slot log probabilities, G3 applies the existing
  bounded residual in log space, and Grounded/Local/Dense/Object/Future carry
  the probability/log views of that same measure;
- the one Schema25 K+null binder, conditional-K reconstruction, exported
  content, camera identity, W values, two P2 types and all loss owners remain;
- a legacy no-log fixture uses an explicit FP32 fallback. Its boolean support
  now remains authoritative in both iterative and final binder reads, so an
  exact-zero candidate prior cannot re-enter through a finite stored log zero;
- source-tensor hooks cover the named S/P1/W2/P2/P3 owners, copy detached FP32
  RMS values during backward and return the incoming gradient unchanged;
- finite spike attribution scans parameter gradients only after the already
  materialized finite global norm crosses the audit threshold, and it runs
  before decoder-local/global clipping. Window summaries remain host-only and
  flush an epoch tail.

Forward re-review closes the active path:

```text
G2 typed local-slot log probability
  -> G3 bounded log residual
  -> GroundedFactSet -> LocalFactSet -> DenseFactChart
  -> K+null binder and typed conditional reads
  -> ObjectFactSet object/camera probability + log
  -> online W copy / detached Teacher copy
  -> FutureObjectDynamics -> P2 K or K*C spatial selection
  -> independent no-null I terminals -> complementary latent sum
```

Reverse re-review confirms that action/event/motion loss reaches P2, online W
and the G owner/log path; future losses reach W/S/current G; reconstruction
reaches the conditional-K binder, candidate values and its existing residual;
and the factual P1 path reaches the finite-slope address transforms. Teacher
fields remain detached. Diagnostics add no loss edge, parameter gradient,
support, value or update mutation.

No unresolved scale competition, reconstructed axis, alternate consumer path,
probability floor, status/reliability authority or hidden checkpoint field
remains in the authorized boundary.

## 9. Static identity closure

| Field | LC-01 | R1h | Delta |
|---|---:|---:|---:|
| Total parameters | 168,412,574 | 168,412,574 | 0 |
| Trainable parameters | 152,041,843 | 152,041,843 | 0 |
| Parameter tensors | 1,386 | 1,386 | 0 |
| Trainable/optimizer tensors | 1,064 | 1,064 | 0 |
| Optimizer groups | 23 | 23 | 0 |
| State-key names | 1,392 | 1,392 | 0 |

Seed-0 sentinels:

```text
retained layer-contract keys     46
retained layer-contract digest   801ce2c38e4b552b97500c20bee291cf3c548096ecab5d90356d066c1406a7fc
bottom decoder keys              268
bottom decoder digest            1d85ddad8d3e5c04413f94bb01b9e09532472d16986c02189ac5ff92416be586
post-construction CPU RNG        d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21
ordered state-key-name digest    be7b4b58a8e2ec25c1e3b5c455f303a0954d20a984201173b5de12d2b1f14a20
manifest digest                  964415bd9bbb15d4a8204dcaddedc143ae958f84a0ee211d62fd75aed31c2f93
```

The manifest change names the numerical/runtime ABI only. It does not imply a
state tensor change, and no partial checkpoint migration is provided.

## 10. Verification

| Check | Result |
|---|---|
| Focused N-01/D-01 tests | PASS after the recorded red state |
| Complete retained ten-file mainline suite | PASS: 155/155 |
| Forward producer-to-P2 review | PASS |
| Reverse loss-to-producer review | PASS |
| Retained layer-contract/decoder/RNG identity | PASS: byte-exact digests |
| Optimizer ownership and inventory | PASS: exact LC-01 match |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff over every touched source/test file | PASS |
| Pyright over every touched source/test file | PASS changed-line gate: 0 changed-line errors; 114 existing errors and 516 warnings outside changed lines |
| `git diff --check` | PASS; repository line-ending notices only |
| Training, dataset, CUDA and checkpoint commands | NOT RUN |

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/manifest.py` | `0ea42d518427c149d9268f54c36e57c17e57042f` | `E11FD169C04CCE6C1BE5073F25539E92F1947FFBD23C5FEB6115247039DB0BA4` |
| `clearvla/mainline/model/compiler.py` | `c2829752ea941c7a4821f817938011fd16c841ce` | `BA48F5DC568211AE162274B0B7F0BD54F3B3101D4D943768A1E58969DC664FC0` |
| `clearvla/mainline/model/dynamics.py` | `2cce430f01b7e6d4c98666b9e2487cf87853c1ae` | `04F13EF9FA760FAC19B2E7D6F060248D9B83A2F062F0318FA57DC84882A2F2B9` |
| `clearvla/mainline/model/grounding.py` | `01edd3cf681db740c995042a5548b82e9e506357` | `F5C6E8C1F0363741D08DE6DFEEF88F3FBA6CD85AB327BE9707ED0C057B867E9A` |
| `clearvla/mainline/model/policy.py` | `1a06ca465d74815bfba10b2b55743917599bff40` | `B033F70388D3AFEC033E5F0D8DFDCF1D23C6A789510997C651736B41E7136B87` |
| `clearvla/mainline/model/restored_observation.py` | `81987f94e79c0fc72a029f88aefe55fab6a23150` | `74B12669E9ABA2629DDDB2923A7FEAFF34A87AAC3D83C438409891F156A229B8` |
| `clearvla/mainline/model/routing.py` | `8f81c59849755e3ce612b5ddb152046ab3332727` | `B0F0D69016A5D0C214CD6D5E33EBB2479E96751E0B91B8BF63F56498E1E090A4` |
| `clearvla/mainline/model/teacher.py` | `1273567788e4bc60b659e03a4ecf3781656515b5` | `13C2996E673B8602A395F7E5F332C3D2CC07DE8F409E9E004D349C1D80AB8E1B` |
| `clearvla/mainline/model/types.py` | `67f89c12d0fdb65175e734629e92239f10573a94` | `B7E0AA9FB49842EF6DD17637736418E645C5348C6F0EAE7DABBA1DBD4C635EA7` |
| `clearvla/mainline/runtime/logging.py` | `a51709f731d5c5a59c2de7609604b1fd94bc2d21` | `974750351C0601A24E1938563ACA5E2E241727CB3B788707564B5B251AEF608B` |
| `clearvla/mainline/train.py` | `f5594177bb5a506a483a4a9eb4c8d78e9a9ff4f7` | `9ADB504689373AA23E04983FBCB92D4281CBF65A2D50CF0C47A5C62DB9DA3E81` |
| `clearvla/mainline/training/engine.py` | `0c3f259236fffbdaf1e74781ce314b3085a7a7c4` | `869F24F605BDB47534143DC6B9000881F146C39B1051AA268F87E89878FF344D` |
| `clearvla/mainline/training/gradient_audit.py` | `bdadb3c65cac7e168ae1f9dafec5fd6298699871` | `98994C972976F93C0BA4BB055F448514B39A23E9F54C8396B2D206AA9DA2649C` |
| `clearvla/mainline/v120_core/flow_dino_evidence.py` | `5b75ed9d3cb2d65789a473dfd03dcf9c9b582a54` | `0B88B6003E12D1DB2EAED3B3FCE8FE32D729E7007FE6CB1717F22072A24DF863` |
| `clearvla/mainline/v120_core/grounded_intent_effect.py` | `e6da59bc7aa97e00114baeb4e1584cbe74b5d50c` | `23EEECB0E9AE72202EDB9D3A997A0CAFF299F1C5B287AFD3A331797BAEA92678` |

R1h closes the eighth and final source slice. This static closure does not
authorize an experiment; the next state-changing step is a separately approved
immutable run context and fresh-run boundary.
