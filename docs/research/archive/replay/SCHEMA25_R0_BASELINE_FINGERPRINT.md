# Schema25 R0 baseline fingerprint

Status: untouched model-source baseline reproduced locally on 2026-08-26.
Replay documents were committed separately at `8067dee`; no model, config,
test or runtime source had changed when this fingerprint was closed.

## 1. Git and tracked-source identity

| Field | Value |
|---|---|
| Branch | codex/schema25-r1-replay |
| Commit | 6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6 |
| Commit title | fix: preserve S object ownership in schema 25 |
| Commit tree | 5a35d67a34652e563d384a9da24f27d7bb560b84 |
| clearvla/mainline tree | 1e280dac1180d9eb8e8859c6a8292c4dc87055b6 |
| configs tree | 1b6c31ad2e41f90d7eb2e59dac5b9c21ed45c4be |
| tests tree | 32a728f618514f0eafd35805084484ee27e2a52c |
| Initial worktree status | clean before replay documents were imported |
| Model-source diff after import | none |

Tracked file SHA-256:

| File | SHA-256 |
|---|---|
| pyproject.toml | 8CB82B58AF9B6A300C53DE179C2F3899ABB213948BC519B15F5721FBA2B47FEB |
| uv.lock | 9F034265DC58AF0522ADA7346FD86D814D81BEC2782E67A47EF1D221A1930C52 |
| configs/mainline/object_intent_dynamics_323.json | 1924E5550A9763FA26249E081CFBE8A078B90BA02D70599C0FBBB908CDD7AEC8 |

## 2. Runtime used for the local fingerprint

| Field | Value |
|---|---|
| Python | 3.12.13, conda-forge build |
| Torch | 2.11.0+cpu |
| CUDA runtime in this environment | none |
| torch.cuda.is_available | false |

This environment is sufficient for source compilation, model construction and
CPU structural tests. It is not CUDA, memory, throughput or training evidence.

No dataset, checkpoint, raw log or future target was loaded to produce this
fingerprint.

## 3. Architecture manifest

| Field | Value |
|---|---|
| Capability | object_intent_dynamics_323 |
| Schema | 25 |
| Layout | clearvla_mainline / 1 |
| Topology | 3 / 2 / 3 |
| Intervals | 4-8, 8-16, 16-32, 32-48 |
| Object slots | 4 |
| Manifest digest | e6e0813f1c501021c55304db1da8a564499005c3aebd2364f39f0d5a977d20b9 |

Component identities:

| Component | Identity |
|---|---|
| observation | restored_v120_three_frame_flow_dino_progressive_g123_bank |
| top | v120_progressive_g123_dense_grounder_exact_p1_s_owned_k_typed_relevance_four_interval_w_five_lane_p3 |
| bottom | restored_v120_shared_seed_dynamic_p1_p1_p2_contracts_evidence_mmdit_dense512_execution |
| training | v120_mirrored_physical_flow_exact_teacher_current_support_event_boost_v120_decay_local_global_clip |
| runtime | cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated |

## 4. Parameter inventory

The model was constructed with ExperimentConfig after torch.manual_seed(0).
Counts do not depend on parameter values.

| Module | Total parameters | Trainable parameters | Parameter tensors |
|---|---:|---:|---:|
| action_codec | 0 | 0 | 0 |
| bottom | 55,833,392 | 47,110,398 | 398 |
| factual_reader | 3,612,429 | 2,823,949 | 141 |
| history_proposal | 10,014,727 | 10,010,631 | 54 |
| observation | 13,543,661 | 6,895,950 | 545 |
| top | 78,947,853 | 78,849,549 | 229 |
| transition | 8,029,833 | 7,897,097 | 47 |
| total | 169,981,895 | 153,587,574 | 1,414 |

## 5. Optimizer ownership

The optimizer owns all 153,587,574 trainable parameters exactly once through
1,076 tensors and 23 semantic groups.

| Optimizer group | Tensors | Parameters | LR | Decay |
|---|---:|---:|---:|---:|
| bottom_capacity/nodecay | 3 | 49,152 | 0.000112 | 0 |
| bottom_evidence_adapter/decay | 47 | 3,439,622 | 0.000056 | 0.01 |
| bottom_execution/decay | 43 | 11,565,569 | 0.000056 | 0.01 |
| bottom_heads/decay | 20 | 14,358 | 0.000056 | 0.01 |
| bottom_mmdit/decay | 43 | 14,189,568 | 0.000056 | 0.01 |
| bottom_organizer/decay | 22 | 4,794,560 | 0.000056 | 0.01 |
| bottom_policy_bridge/decay | 7 | 66,336 | 0.000056 | 0.01 |
| bottom_query/decay | 17 | 1,065,472 | 0.000056 | 0.01 |
| coarse_action/decay | 18 | 8,394,240 | 0.00008 | 0.01 |
| consequence/decay | 2 | 524,288 | 0.00008 | 0.01 |
| controlled_transition/decay | 41 | 7,897,097 | 0.00008 | 0.01 |
| dynamics/decay | 32 | 9,231,366 | 0.00008 | 0.01 |
| grounder/decay | 17 | 4,007,936 | 0.00008 | 0.01 |
| grounding/decay | 70 | 25,999,873 | 0.00008 | 0.01 |
| history_proposal/decay | 53 | 10,010,631 | 0.00005 | 0.01 |
| intent/decay | 55 | 23,068,675 | 0.00008 | 0.01 |
| observation/decay | 345 | 6,895,950 | 0.00008 | 0.01 |
| p1_factual/decay | 157 | 12,013,326 | 0.00008 | 0.01 |
| p2_effect_reader/decay | 10 | 1,577,475 | 0.00008 | 0.01 |
| p3_compiler/decay | 12 | 3,145,728 | 0.00008 | 0.01 |
| plan_recognizer/decay | 11 | 2,899,968 | 0.00008 | 0.01 |
| v120_canvas_seed/decay | 39 | 2,470,912 | 0.00008 | 0.01 |
| v120_layer_contracts/decay | 12 | 265,472 | 0.00008 | 0.01 |

## 6. Reproduced checks

| Check | Result | Interpretation |
|---|---|---|
| Python compileall over clearvla and tests | PASS | Source is syntactically compilable |
| Schema25 mainline collection | PASS: exactly 122 tests | Matches the active Schema25 contract |
| Schema25 mainline test suite | PASS: 122/122 in 44.42 s | CPU structural/forward/backward/runtime/checkpoint baseline reproduced |
| Deployment Teacher contract | PASS through mainline tests | Five-step cached deployment constructs no Teacher path |
| Full-repository Ruff | BASELINE FAIL: 781 findings | Pre-existing legacy-tree lint debt; not caused by replay |
| Full-repository Pyright | stopped after several minutes without output | Whole legacy-tree scan is not an informative per-slice gate |
| Mainline plus ten tests Ruff | BASELINE FAIL: 2 import-order findings | Pre-existing in transition.py and v120_p1.py |
| Mainline Pyright with the active Python path | BASELINE FAIL: 343 errors, 1,737 warnings | Historical typing debt; use touched-file delta rather than absolute pass |

The 122-test suite consists of the ten test_mainline files:

- test_mainline_action_field.py
- test_mainline_checkpoint.py
- test_mainline_config.py
- test_mainline_data.py
- test_mainline_interfaces.py
- test_mainline_manifest.py
- test_mainline_policy.py
- test_mainline_runtime.py
- test_mainline_structural_contracts.py
- test_mainline_top.py

## 7. R1 comparison policy

- R1 may not reduce the 122-test baseline except where an old test explicitly
  asserts a mechanic intentionally replaced by an adopted semantic unit; such
  a test must be replaced by a stronger ownership test in the same commit.
- Full-repository Ruff/Pyright debt is recorded rather than repaired inside an
  architecture slice. R1 gates lint and typing on touched mainline/test files
  and prohibits new findings relative to this baseline. If a slice touches
  transition.py or v120_p1.py, its pre-existing import-order finding is either
  removed in that slice or explicitly compared against the two-finding R0
  baseline.
- Parameter and optimizer changes must be explained per semantic slice.
- CUDA smoke, dataset access and training remain outside this R0 fingerprint.
