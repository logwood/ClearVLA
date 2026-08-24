# Historical Schema36 Top Information-Conservation Audit

## Purpose and audit identity

This is the frozen source-backed inventory that preceded Schema37. It describes
the former Schema36 graph only and must not be used as the current contract.
It answered four questions for every boundary:

1. Which axes and values enter?
2. Which transform is applied?
3. Which real consumer receives the result?
4. Is the transform reversible, a conditional read, a hard loss, a dead path,
   a supervision bypass, or still unresolved?

It is intentionally not a repair plan.

```text
source commit:     9e75d31f71bd6a1aa4327d31a280bfc5639a2620
source hash:       5c22837ad22e
manifest schema:   36
audit log:         schema36_p1_p2_closure_b8.log
log coverage:      epoch 1 complete; epoch 2 batch 80 / step 2926
behavior anchor:   V120 long / 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
```

## Classification

| Code | Meaning |
|---|---|
| `P` | Axis/value preserved to the named consumer. |
| `R` | Reversible or exact reparameterization; the original can be reconstructed. |
| `C` | Conditional read/statistical summary. Irreversible, but performed at an actual consumer boundary. It still needs task justification. |
| `H` | Hard irreversible loss before all plausible consumers have used the identity or value. |
| `D` | Dead or no-consumer path. A nonzero shared activation or aggregate gradient does not make it functional. |
| `B` | Training/supervision bypass: a loss can improve through capacity not exported to the online consumer. |
| `U` | Source is mapped, but available evidence cannot yet determine whether the boundary is harmful or useful. |

`C` is not automatically healthy and `H` is not automatically catastrophic.
The classification states what the algebra does; necessity and task value are
separate questions.

## G and grounding

| ID | Boundary | Axes/value entering | Executed transform and consumer | Class | Source/evidence |
|---|---|---|---|---|---|
| G-01 | Observation bank -> G1/G2/G3 | camera, 8x8 chart, local M, DINO/raw/flow/state | Three progressive blocks; G2 rematerializes N=49 candidates; G3 finalizes dense facts. Static P1 and grounder consume the dense result. | `P` | `model/policy.py:275-287`; restored observation implementation. |
| G-02 | Dense chart -> global K+null | `[B,C,8,8,M,*]` | Soft K+null binding. Global K cannot reconstruct the dense chart, but the dense chart remains separately available to static P1 and Teacher. | `C` | `model/grounding.py`; `model/types.py:166-364`. |
| G-03 | Public/object content split | public `[B,D]`, object `[B,K,D]` | `content_innovation = content - public`; downstream W/S reconstruct `public + innovation`. | `R` | `model/types.py:166`; `model/intent.py:229`; `model/dynamics.py:204-209`. |
| G-04 | Typed K support -> S | semantic/appearance/geometry object routes; camera-specific evidence exists in facts | S receives object-level typed routes but no camera-specific intent axis. Camera identity is already reduced for this branch. W/Teacher still receive camera fields separately. | `H` for S branch | `model/intent.py:113-405`; `ObjectFactSet` properties. |
| G-05 | G3 conditional-K correction | parent K logits | Adds bounded common-centered residual; zero residual recovers parent posterior. Common logit gauge is unobservable to softmax. | `R` | `model/grounding.py`; G3 metrics. |
| G-06 | Grounding reconstruction | exported K content plus target DINO | `decode_public_position` creates a coordinate spatial term used by reconstruction only and not exported. The loss can improve through a private head. | `B` | `model/grounding.py:590-609`. Latest public-position RMS is nonzero and growing. |
| G-07 | Geometry candidate assignment field | typed spatial assignment | Stored in `ObjectFactSet`, but no current online consumer was found beyond retained diagnostics/data structure. | `D/U` | `model/types.py`; repository consumer search. |
| G-08 | G3 public-scene audit carrier | dense public chart value | Retained in `DenseFactChart.g3_public_scene_audit`; no S/Teacher/W/P1/P2 consumer. | `D` diagnostic | `model/types.py:112-156`; repository consumer search. |

Latest V36 evidence argues against current global-K homogeneity: content pair
cosine is about `0.50`, innovation pair cosine about `-0.21`, and chart overlap
about `0.15`. This does not identify G3 or learned-flow action value.

## S and its docks

| ID | Boundary | Axes/value entering | Executed transform and consumer | Class | Source/evidence |
|---|---|---|---|---|---|
| S-01 | T5 tokens -> goal set | `[B,L,D]` | Four learned queries conditionally read the full token set. The original tokens do not continue past S. | `C` | `model/intent.py:420-430`. |
| S-02 | Ordered history -> interval reads | 3 state + 8 executed-action rows | Four interval queries conditionally read all 11 ordered tokens. | `C` | `model/intent.py:175-228,444-504`. |
| S-03 | Object content -> interval reads | public plus `[B,K,D]` innovations | Public+innovation is reconstructed before attention; four reads summarize object memory. | `R` then `C` | `model/intent.py:229-318`. |
| S-04 | Typed state -> W/P2 docks | common `[B,K,3,R]`, residual `[B,4,K,3,R]` | K/type/interval axes remain present for W and P2. | `P` | `model/types.py:601-609,629-639`. |
| S-05 | Typed differential score | `[B,4,K,3]` after nonlinear score | Subtracts interval mean after `tanh` and normalization. The removed nonlinear mean is not recoverable from the separately computed common score. | `H` | `model/intent.py:319-330`. |
| S-06 | S -> static P1 factual dock | four goal reads, full history, K/type state | K is averaged in policy components; types are summed; goals are averaged; only last history row is kept; results are expanded to four rows. | `H` | `model/types.py:611-627`. |
| S-07 | Four intervals -> 24 temporal queries | `[B,4,H]` | 24 queries conditionally read interval state. | `C` | `model/intent.py:505-513`. |
| S-08 | History/transport -> state-change | ordered history plus K/camera-reduced transport | Collapses to one `[B,H]` evidence vector consumed by P3. | `C/H` | `model/intent.py:514-558`. |
| S-09 | S -> bottom intent memory | full `ObjectIntentState` | Bottom executes `del intent`; only state/executed tokens remain. S reaches action only through upstream W/P1/P2/P3. | `H` for direct bottom branch | `model/restored_bottom.py:560-570`. |
| S-10 | Executed history -> HistoryActionProposal | 8 ordered action rows | Encodes three older summaries plus four recent tokens; these seven rows condition the bottom seed. A separate 24-step proposal is supervised by an auxiliary head but is not an online proposal value. | `C` plus auxiliary-only branch | `model/proposal.py:133-190`; `model/policy.py:232-240,354-360,411-425`; `model/action_contract.py:307-358`. |

At step 2926, public S RMS was about `1.69` versus typed policy context
`0.174` (`~9.7x`). Goal attention entropy was `~0.996`, interval-goal entropy
`~1.000`, and temporal horizon variation was only `0.039` for temporal RMS
`0.691`. The typed branch has a `0.35` RMS contract while the public exit has
no matching final contract; the current addition therefore has a real scale
competition.

## Teacher and W

| ID | Boundary | Axes/value entering | Executed transform and consumer | Class | Source/evidence |
|---|---|---|---|---|---|
| T-01 | Current/future DINO -> association | full DINO value, low-rank normalized keys, camera/cell candidates plus dustbin | Low-rank keys determine matching; full DINO values remain for successor content. | `C` for matching, `P` for value width | `model/teacher.py:293-380`. |
| T-02 | Per-support posterior -> successor | future support × K × camera × cell | Expectation plus dustbin identity fallback; spatial multimodality is reduced. | `C` | `model/teacher.py:375-388`. |
| T-03 | 12 supports -> four intervals | support time/order | Fixed uniform interval means. Within-interval order, extrema and path shape are lost. | `C` | `model/teacher.py:389-443`. |
| T-04 | Spatial motion posterior -> geometry | K × camera × cells | First/second moments only; multimodal path structure is lost, camera axis remains. | `C` | `model/teacher.py:554+`. |
| T-05 | Future target -> online graph | detached target fields | Teacher is no-grad/FP32 and is built after online context; no future argument exists in compile/deploy. | `P` isolation | `model/top.py:357-424`; `model/policy.py:389-410`; runtime sampling. |
| W-01 | Object/flow/S/action/goal -> generic W base | distinct provenance sources | Projected sources are directly added before W blocks. Typed sidecars remain, but generic-source identity is gone. | `H` | `model/dynamics.py:195-245`. |
| W-02 | Typed common/residual -> W1/W2 | `[B,K,3,H]`, `[B,4,K,3,H]` | W1 owns common/near; W2 conditionally reads near and owns far. K/type/interval axes survive. | `P/C` | `model/dynamics.py:643-812`. |
| W-03 | Near/far residual -> completed residual | four interval residuals | Far-owned zero-sum gauge subtracts near+far sum from far and removes a W2 common-mode direction. | `H` | `model/dynamics.py:523-553,802-812`; `_field()` repeats field gauge. |
| W-04 | Future field -> common/residual properties | complete interval tensor | Mean plus centered residual exactly reconstructs the field. | `R` | `model/types.py:918-989`. |
| W-05 | W appearance owner -> status | appearance owner state | Visibility/persistence heads have exact-zero targets; P2 has no status value consumer. Shared W activity does not give this W appearance owner a functional output path. Appearance remains active in other subsystems. | `D` | `model/dynamics.py:152,332-437`; `model/compiler.py:78`; Teacher zero targets. |
| W-06 | Supervised W output -> P2 | FutureObjectDynamics | The same semantic/transport/covariance object is supervised and consumed; no alternate W hidden crosses the boundary. | `P` | `model/top.py:280-355,426-480`. |
| W-07 | Selector validity -> consumers | `[B,4,K,1]` predicted/target validity | Logged and validated, but P2 authority uses chart availability and future loss uses current support. | `D` diagnostic | `model/types.py:903-906`; `model/dynamics.py:440-455`; `training/losses.py:574-579`. |

Latest W interval variation is about `40%` of Teacher, residual RMS `43%`,
common RMS `58%`, adjacent interval cosine `0.971`, and object-pair cosine
`0.104`. The symptom is interval publicization/underfit, not K collapse.

## Static and dynamic P1

| ID | Boundary | Axes/value entering | Executed transform and consumer | Class | Source/evidence |
|---|---|---|---|---|---|
| P1-01 | G3/P1 bank -> coarse/fine address | C, 8x8, local M, N=49, four glimpses | Soft address queries retain axes until their corresponding read. | `P` to read | `model/v120_p1.py`; `model/policy.py:297-331`. |
| P1-02 | RGB/detail base + precision | microgrid values | Zero-mean precision plus base is reconstructible before projection. Protected base is outside optional typed null. | `R` | `model/v120_p1.py:807+,943+`. |
| P1-03 | N49/camera/cell/M/glimpse -> factual value | complete address banks | Sequential conditional expectations reduce to one `[B,24,4,H]` output. No parallel axis-preserving carrier continues. | `C`, terminal `H` for later consumers | `model/v120_p1.py:1455-3939`. |
| P1-04 | Typed owner -> optional detail | semantic/appearance/geometry/horizon deltas | Learned null may discard optional typed detail; protected RGB/detail remains. Whether current rejection is useful is unresolved. | `C/U` | `model/v120_p1.py`; `model/routing.py:225+`. |
| P1-05 | Action + static fact -> dynamic block | separate action/static tensors | They are added before the block; local provenance is lost, though static fact remains in a parallel protected field. | `H` inside dynamic branch | `model/restored_bottom.py:275-313`. |
| P1-06 | Dynamic P1 output -> policy | `[B,24,4,H]` residual | Only contributes to `effect_query`; no value consumer in P3, transition, layer contracts or bottom. | `D` outside P2 / narrow consumer | `model/top.py:444-470`; repository consumer search. |

At step 2926, dynamic P1 RMS was `0.1886` versus static `0.0180` (`10.5x`).
The final residual contract removed only `~0.19%`, so this ratio is not a clamp
artifact. Aggregate `p1_factual` gradients merge static and dynamic owners and
cannot attribute pressure between them.

## P2, consequence, P3 and bottom

| ID | Boundary | Axes/value entering | Executed transform and consumer | Class | Source/evidence |
|---|---|---|---|---|---|
| P2-01 | Camera geometry -> object value | transport `[B,I,K,C,2]` | Camera-weighted sum produces `[B,I,K,2]` before action-conditioned object read. C remains only in coordinate scoring. | `H` for value provenance | `model/compiler.py:198-226`. |
| P2-02 | K candidates -> selected object | `[B,I,K,type,H]` | Per-type/action-query soft read over K. | `C` | `model/compiler.py:384-529`. |
| P2-03 | Intervals + typed null -> residual | `[B,I,type,H]` | Public-S, matching typed-S and matching W score an interval plus exact-zero null for each type. | `C` | `model/compiler.py:531-597`. |
| P2-04 | Semantic + geometry -> P2 value | two separately selected typed values | Direct addition removes the type difference subspace and permits cancellation. | `H` | `model/compiler.py:143-164,598-616`. |
| P2-05 | P2 effect -> consequence | static fact and effect | Bias-free interaction; `protected = fact + effect + interaction`; effect zero exactly recovers fact. | `R`/zero-preserving | `model/compiler.py:959-993`. |
| P3-01 | Static fact/W/S -> four lanes | precision, effect, temporal, state-change | Source-specific projections; precision and effect duplicate information already present in protected base as optional action modulation. | `C/U` | `model/compiler.py:996-1084`. |
| B-01 | Four P3 lanes × four bases -> bottom delta | 16 optional values | One `16 + null` softmax converts complementary lane/basis sources into mutual competition. | `C/H` | `model/restored_bottom.py:466-481`; `time_domain_mmdit.py:1033-1064`. |
| B-02 | Protected consequence -> bottom | `[B,24,4,H]` | Separate no-null basis reader; cannot be rejected by the P3 null. | `P/C` | `time_domain_mmdit.py:1065-1086`. |
| B-03 | Transition chart -> event context | 512 spatial rows | Spatial rows are pooled by anchor for event only; full transition remains available as Evidence memory. | `C` for event, `P` elsewhere | `model/restored_bottom.py:441-464`. |

At step 2926, P2 residual null mass was `~0.0365`, while residual cancellation
after semantic/geometry fusion was `~0.30`. Semantic common value was about
`12.5x` geometry; semantic residual about `3.9x` geometry. Null is therefore
not the current primary W suppression mechanism.

## Backward ownership and lifecycle

| ID | Boundary | Executed behavior | Class/evidence |
|---|---|---|---|
| L-01 | Future losses -> W | Semantic/transport/covariance/status terms supervise exported W fields with detached current support. Reliability/dustbin/selector validity do not mask loss. | Correct isolation; status owner remains dead because target and consumer are zero/absent. `losses.py:355-571`. |
| L-02 | Action losses -> P2/P3/transition | No separate imitation objective; final action/event/motion/execution supplies gradients. | Aggregate nonzero gradient proves reachability only. |
| L-03 | Optimizer ownership | G, grounder, S, supervisor, W, P1, P2, consequence, P3, transition and bottom have registered owners. | `P` ownership; submodule attribution is still aggregated. `optimizer.py`. |
| L-04 | Gradient processing | finite check -> raw -> decoder-local clip -> post-local -> global clip -> post-global -> step. | `P`; V36 currently finite. |
| L-05 | Runtime frequency | static context once; dynamic graph five Euler steps plus endpoint; Teacher once in training and zero in deployment. | `P`; runtime source and preflight. |

## Unresolved assumptions

These cannot be closed from the current console log and static source alone:

1. Whether Teacher dustbin near `0.445` is calibrated uncertainty or overly
   attractive relative to real matches. There is no true association label.
2. How much useful far common-mode W-02/W-03 removes per batch; pre/post-gauge
   RMS is not logged.
3. What fraction of W's aggregate gradient comes from final action versus its
   future objectives.
4. Whether S identities discarded by `factual_dock()` are needed for this
   single task, even though the loss is algebraically certain.
5. Whether dynamic P1's large query residual causally dominates P2; matched
   static/dynamic JVP or intervention is absent.
6. Which P3 lane/basis the bottom actually adopts. The current run's
   `metrics.jsonl` is unavailable locally and compact console output omits
   source-mass metrics.
7. Independent causal action value of G3 and learned flow.
8. Whether V36 remains numerically healthy after the old epoch-3 Schema35
   failure window and whether its early gripper/event mismatch persists.

No implementation change should begin by assuming an answer to any item above.
