# R1a / G-01 exact G3 handoff worksheet

Status: implemented and statically closed on 2026-08-26; no training run.

This worksheet is the mandatory producer-to-consumer and consumer-to-producer
review for G-01. It authorizes only the exact completed-G3 handoff selected in
`ARCHITECTURE_REPLAY_SOURCE_UNITS.md`. It does not authorize any Schema26 S,
W, proposal, manifest or checkpoint-migration change.

## 1. Source identity and donor scope

| Role | Commit/blob |
|---|---|
| Schema25 policy base | `6a6c1bf:policy.py` / `5b8255f94c4401732ad210a07e35c4f6f8200661` |
| Schema25 transition base | `6a6c1bf:transition.py` / `672cec3a877536e96c210bcc1c8384fe184a3dad` |
| Schema25 transition types | `6a6c1bf:types.py` / `679e238b579265931755b56147ba4062f23ba986` |
| Schema26 policy donor | `caa7e33:policy.py` / `5a718a57426843d8fdd99d2d37ad0b8d3c070065` |
| Schema26 transition donor | `caa7e33:transition.py` / `5cf536e091d082f28cfe23c1a708c622cc90e1f1` |
| Schema26 test donor | `caa7e33:test_mainline_policy.py` / `8410aee7944234b4261113280265263c32e75ad0` |

The exact handoff remains present in the Schema39 snapshot. The transition
file changed after Schema26 only for the later Schema39 policy-precision unit;
the G-01 source construction itself survived unchanged.

## 2. Forward dataflow map

### 2.1 Producer and final G3 carrier

1. `FlowDINOEvidenceEncoder` produces `FlowDINOEvidencePack.future_queries`
   with anchor, camera and spatial identity.
2. `ActionQueryEncoder.grounding_canvas` reshapes the rows as
   `[B, I, C*8*8, H]`, adds the inherited rollout anchor/grid/role identities,
   and concatenates them into the grounding canvas. In the active manifest,
   `I=4`, `C=2`, and the flattened rollout has 512 rows.
3. `ObjectIntentDynamicsTop.run_progressive_grounding` applies the three
   active `TemporalDynamicsBoundDiTBlock(role="grounding")` blocks. Each block
   writes the rollout through directed self attention, visual cross attention,
   the grounding dynamics branch and the FFN residual. The blocks retain the
   flattened anchor-major, camera-major, xy-major row order.
4. After each block,
   `RestoredV120ObservationCompiler.advance_progressive_grounding` uses the
   same rollout as the G1/G2/G3 address query. The address state is a sidecar;
   it does not replace the canvas carrier.
5. After block G3, `grounding_canvas[:, grounding_slices["rollout"]]` is the
   completed source. Its exact contract is `[B,4*C*8*8,H]`.

`ProgressiveGroundingAddress.update(stage=3)` also derives
`GroundedFactSet.public_scene_base` from `_clean_query(rollout)`. That helper
reshapes to `[B,4,C,8,8,H]` and averages the four-anchor axis before applying
the inherited RMS contract. `public_scene_base` is therefore a downstream
summary of the completed carrier, not an equivalent representation of it.

### 2.2 Static consumers

- P1 receives the completed rollout directly in
  `LateRawDetailPolicyReader.forward`. It reshapes the exact rows to
  `[B,4,C,8,8,H]`, aligns the four real milestones to the 24 action steps and
  uses both camera summaries and the full xy chart as factual selector
  context. This expensive factual read occurs once in `encode_online`.
- Schema25 transition source construction instead receives
  `context.facts.dense_chart.public_scene_base`, reshapes its single
  `[C,8,8]` chart, expands it across four rows by broadcast addition, and adds
  the new trainable `interval_identity`. This is the G-01 defect.

The R1a transformation is deliberately empty:

```text
g3_rollout = grounding_canvas[:, grounding_slices["rollout"]]
P1(g3_rollout)
ControlledTransitionSource(selector=g3_rollout)
```

Only shape validation is legal. There is no cast, clone, detach, mean,
expand, projection, normalization, gain, floor or new temporal identity at
the handoff.

### 2.3 Dynamic consumers

`OnlinePolicyCache.transition_source` retains the static tensor and its graph.
At each `ClearVLAMainlinePolicy.velocity` call:

1. `ControlledTransitionDynamics.forward` validates the 512-row source.
2. The source is passed unchanged as both `rollout_base` and
   `transition_tokens` to `ControlledResidualLatentDynamics`.
3. That module constructs a per-row low-rank basis and subtracts learned
   neutral coefficients from action coefficients. Its `value` remains
   `[B,512,H]`.
4. `RestoredV120EvidenceBottom` consumes the selector as rollout keys in both
   layer contracts and the Evidence-MMDiT adapter. It consumes `value` as
   transition memory and also pools it by `(anchor, C*8*8)` solely for the
   24-row event milestone context.
5. The retained decoder produces physical velocity, event logits and motion
   logits.

Training calls this dynamic path once for the sampled flow node. Deployment
builds the static source once, then reads it at five Euler nodes and once more
at the clean endpoint. No runtime cache tensor is serialized.

## 3. Axes, dtype, scale and zero semantics

| Property | Contract |
|---|---|
| Row order | `[interval/anchor, camera, y, x]` flattened without permutation |
| Production shape | `[B,4,2,8,8,512] -> [B,512,512]` |
| Small-test shape | `[B,4,2,8,8,32] -> [B,512,32]` |
| Dtype/device | exactly those of the final grounding canvas; no handoff conversion |
| Scale | exactly the final G3 block output; no second amplitude contract |
| Repetition | one static construction; one training read or six deployment reads |
| Zero source | zero remains exact zero at `ControlledTransitionSource.selector` |

G-01 does not assert that a zero source forces the entire learned transition
output to zero. The inherited controlled-transition body contains learned
normalizations and nonlinear projections. The required zero semantic is only
that the handoff itself cannot manufacture four nonzero interval labels from a
zero G3 carrier.

## 4. Reverse gradient map

The legal reverse path is:

```text
action / event / motion / execution losses
  <- retained bottom evidence reads
  <- ControlledTransitionState selector and value
  <- ControlledResidualLatentDynamics
  <- OnlinePolicyCache.transition_source.selector
  <- exact final G3 rollout view
  <- G3/G2/G1 blocks and observation producer
```

`training.losses.action_terms` owns the physical, decoded, event and motion
losses. Execution-value losses also read decoder candidates. There is no
separate controlled-transition loss; transition parameters and G3 receive
gradients through these downstream objectives. The static P1 path supplies a
second legal action-loss route to the same final G3 carrier.

The acceptance VJP selects one output row and requires the gradient with
respect to the captured P1 G3 input to be the identical one-row cotangent.
This proves the direct alias and row mapping, rather than merely proving that
some indirect gradient reaches a summarized fact chart.

## 5. Optimizer and checkpoint ownership

- All `transition.*` trainables belong to optimizer role
  `controlled_transition` with the inherited AdamW decay policy.
- Schema25 `transition.interval_identity` has shape `[1,4,1,H]`; at production
  width it owns 2,048 trainable parameters and one optimizer tensor.
- The field is stored as `transition.interval_identity` in `model.state_dict`
  and its optimizer state is serialized by group ownership.
- R1a removes that field. All remaining transition parameters retain the same
  owner and consumer.
- Exact Schema25 resume is consequently incompatible by both model-key and
  optimizer-group signature. R1 uses a fresh checkpoint as already required
  by the implementation protocol; no migration or permissive load is added.
- The architecture manifest remains a Schema25 label while R1 is an
  intermediate source assembly. Formal R1 manifest identity is a later
  candidate-closure task, not part of G-01.

Deleting the parameter also changes fresh seeded initialization cadence for
modules constructed after it. R1 counterfactuals compare interventions within
one fixed R1 initialization; they do not claim parameterwise initialization
identity with R0.

## 6. Diagnostics and bypass audit

Retained diagnostics:

- `controlled_transition_source_rms`;
- `controlled_transition_source_spatial_variation` over the 128 spatial rows
  inside each real anchor.

R1a adds `controlled_transition_source_anchor_variation` over the four real
anchors. It is observational only and detached. It introduces no threshold,
quota or training term.

Forbidden alternate paths checked in source:

- no `ObjectFactSet` argument remains in transition source construction;
- no `DenseFactChart.public_scene_base` transition read remains;
- no `interval_identity` remains;
- no `mean(...).expand(...)` or equivalent interval reconstruction remains;
- the source is not detached or cloned;
- P1 and transition use one local `g3_rollout` object;
- bottom still receives all 512 selector/value rows before its explicitly
  named event-only spatial pooling.

## 7. Test-first acceptance set

Before source implementation, add tests that fail on Schema25 and require:

1. integrated P1 input and cached transition selector are the same tensor
   object;
2. a single-row cotangent maps identically back to that P1/G3 view;
3. a standalone sentinel row survives `build_source` at the identical index;
4. zero input remains exact zero at the source;
5. invalid row count is rejected;
6. `interval_identity` and an `ObjectFactSet` source API are absent;
7. anchor-variation diagnostics exist without a magnitude floor;
8. the retained five-step static/dynamic call-count test still passes.

## 8. Unresolved assumptions and edit authorization

No unresolved semantic assumption remains for G-01. The fixed values `I=4`,
`C=2`, grid `8x8` and 512 rows are already locked by the active manifest,
`ControlledTransitionSource.validate`, the V120 decoder ABI and current
configuration validation.

Authorized source surface after the failing tests are observed:

- `clearvla/mainline/model/policy.py`: name the final rollout view once and
  pass it to both static consumers;
- `clearvla/mainline/model/transition.py`: accept that tensor directly, remove
  the reconstruction and `interval_identity`, and add the matching detached
  anchor-variation diagnostic;
- `tests/test_mainline_policy.py`: G-01 structural, sentinel and VJP coverage.

No other source file is authorized by this worksheet.

## 9. Implementation and verification result

Implemented source surface:

- `policy.py` binds the final rollout slice once as `g3_rollout` and passes
  that one tensor object to P1 and transition source construction;
- `transition.py` accepts only `g3_rollout`, stores it directly in
  `ControlledTransitionSource`, removes `interval_identity` and records
  detached anchor variation;
- `test_mainline_policy.py` covers integrated identity, one-row exact VJP,
  dynamic transition-to-G3 VJP, sentinel preservation, exact-zero handoff,
  invalid shape and API/parameter absence.

Observed test-first failure before implementation:

```text
2 failed, 26 deselected
- cached transition selector was not the P1 G3 rollout object
- build_source rejected the g3_rollout keyword
```

Post-implementation checks:

| Check | Result |
|---|---|
| G-01 focused tests | PASS: 2/2 |
| Complete retained ten-file mainline suite | PASS: 123/123 in 35.16 s |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff on the two touched source files and touched test | PASS |
| Pyright on the two touched mainline source files | PASS: 0 errors, 0 warnings |
| Full touched test-file Pyright | Baseline debt remains: 14 errors and 106 warnings; no new G-01 error |
| `git diff --check` | PASS; only repository line-ending notices |

The retained suite increased from 122 to 123 because G-01 adds one test rather
than replacing a valid R0 test.

Production parameter/optimizer delta:

| Field | R0 | R1a | Delta |
|---|---:|---:|---:|
| Total parameters | 169,981,895 | 169,979,847 | -2,048 |
| Trainable parameters | 153,587,574 | 153,585,526 | -2,048 |
| Parameter tensors | 1,414 | 1,413 | -1 |
| Trainable/optimizer tensors | 1,076 | 1,075 | -1 |
| Controlled-transition optimizer tensors | 41 | 40 | -1 |
| Controlled-transition parameters | 7,897,097 | 7,895,049 | -2,048 |

The only removed state/optimizer field is
`transition.interval_identity`. No replacement parameter or normalization was
introduced.

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/model/policy.py` | `b64b73b50072ce5ea53a926d7a1b06fd6219ae2b` | `4B29098661E4F264C81FC5253C8C1145DBE26A67B157C9E8C3F7ED0F115B924B` |
| `clearvla/mainline/model/transition.py` | `583bc79c7e73e2ff7a37745d4c6414551b2cdb0a` | `9B183EE87DF54C0CD37F7201693D1C864A4EFA96F992F3C803D1CED22A40CED3` |
| `tests/test_mainline_policy.py` | `ea7e05e4ae16dbfe2aecd84d695496137e1a47c8` | `6872C6D04FCE5ED974CBAC4F77C6BB60AAB6AA880EDDB482F19DEDA41E4597A4` |

Forward re-review found one source object shared by the two static consumers,
all 512 rows retained by the dynamic transition and bottom, and pooling only
at the named event-context consumer. Reverse re-review found both the exact
one-row boundary VJP and a finite nonzero dynamic-transition VJP back to the
captured final G3 view. Unresolved assumptions remain empty.
