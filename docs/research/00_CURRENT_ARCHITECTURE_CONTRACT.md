# Current ClearVLA Architecture Contract

Updated: 2026-08-13

This is the compact source of truth for the active independent mainline. Run
labels are experiment metadata and never select model semantics. Current open
questions are kept in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md); ancestry and the
evidence that motivated recovery are kept in
[`TOP_ARCHITECTURE_ISSUE_LEDGER.md`](TOP_ARCHITECTURE_ISSUE_LEDGER.md).

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        23
recovery reference:     V120 long, commit 0b92d359a2889a0a1b1eba256007c00ccbc54f3c
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global object slots:    K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0, two adjacent learned flows
formal language:        precomputed 4096-wide T5 .pt required
bottom:                 shared V120 seed + P1/P2 contracts + Evidence MMDiT / execution
long launcher:          scripts/train_mainline.sh (batch 8, workers 4)
smoke launcher:         scripts/smoke_mainline.sh (batch 1, workers 0)
resolved config:        configs/mainline/object_intent_dynamics_323.json
```

The active graph lives in `clearvla/mainline/`. It does not import the V39
trainer/runtime/trunk or a V-numbered launcher. The compact manifest owns graph
identity; typed interfaces and executable checks own shapes, dtypes and zero
semantics. Do not add a version-wide `_validate_vXXX_*` contract.

Schema 23 is the controlled V120 behavior-contract repair. It keeps the
Schema-22 G/S/W/P graph and restores only four proven boundaries: mirrored
flow time, V120 deployment nodes plus clean endpoint heads, exact unsqueezed
Teacher targets, and separate training support versus online selector
validity. It also records the first non-finite named parameter before clipping.
No block, loss weight, gate, capacity or P1 learned null was added.

**Schema 22 is experimentally rejected and must not be released or continued
as a long run.** It completed two epochs, then produced a non-finite parameter
gradient in epoch 3 after global step 7812. More importantly, its first/tail
rollout, gripper calibration, G/W separation and P2 future-effect path remained
far outside the V120 envelope before the crash. The active source is therefore
a repair base, not a behaviorally accepted mainline.

The exact V120 object package is available at commit
`0b92d359a2889a0a1b1eba256007c00ccbc54f3c`; all seven package blobs match the
V120 logged SHA256 values. Current repairs must use source-to-source comparison,
not infer the historical object graph from a descendant filename.

## Active graph in execution order

```text
current-only input
  RGB/DINO at -8,-4,0 + observable state/executed-action history + T5 goal
    -> shared raw/DINO extraction
    -> learned flow -8->-4 and -4->0
    -> flow-aligned causal innovation on the current chart
    -> Pre-G DenseFactChart [camera,8,8,local-M]

DenseFactChart
    -> G1-G3 typed role hosts
    -> one competitive global K+null grounding
    -> semantic / appearance / geometry verification on the same K
    -> ObjectFactSet with reversible K <-> local-chart correspondence

T5 + observable history + ObjectFactSet
    -> V120 cumulative StatelessObjectIntentOrganizer S
         protected goal/history/object tokens
         four interval queries
         24 temporal queries
         observable state-change evidence

ObjectFactSet + S + causal CoarseActionIntent
    -> W1: 4-8 and 8-16
    -> W2: 16-32 and 32-48, causally reading W1
    -> one supervised FutureObjectDynamics field visible below W

ObjectFactSet + S + clean V120 typed action-basis tokens
    -> P1 query-specific read over all 49 progressive candidates
    -> one packed, no-null protected RGB/detail factual read
    -> cached ObjectFactualDock

current noisy action + flow time + cached protected detail
    -> compact exact V120 P1 policy block, once per ODE step
    -> completed P1 fact

completed P1 fact + FutureObjectDynamics + S + current noisy-action query
    -> P2 bounded content/intent/coordinate effect read
    -> zero-preserving protected consequence
    -> P3 five lanes: factual / precision / effect / temporal / state-change

current noisy action + current state/history + compressed executed history
    -> one shared V120 canvas seed
         noisy-action query shared by P2/P3/transition
         current state / causal state-history / executed-history rows

P1 trajectory + protected G3 chart
P2 trajectory + protected G3 chart
    -> exact V120 terminal P1/P2 residual adapters
    -> two layer evidence rows; P2 event evidence

G3 protected [camera,8,8] chart
    -> static 512-row ControlledTransitionSource, built once per observation
current ODE noisy action + learned V120 neutral + plan/history
    -> dynamic 512-row ControlledTransitionState, rebuilt every ODE step

protected consequence + five P3 lanes + transition + shared seed + P1/P2 contracts
    -> V120 EvidenceViewAdapter source ordering
    -> three Evidence MMDiT blocks
    -> V120 ordered low-rank contraction and continuation
    -> supervised execution-value reader
    -> 18-D physical velocity + event/motion heads
```

The history-action proposal remains trained as a causal auxiliary prediction.
Its 24 future proposal tokens do **not** feed G/S/W/P, controlled transition or
the bottom. Its separately encoded 4-recent + 3-summary executed-history memory
does retain the V120 observable-history condition in the shared canvas seed.
`proposal_keep` therefore audits only the unused future-proposal branch; the
action-history keep mask controls the real compressed-history condition.

The generic trajectory/workspace ingress is exact neutral. The protected
consequence is written once through the protected detail boundary; the five P3
lanes are optional typed evidence. Full 512-row transition selector/value
evidence reaches bottom attention. Only the event auxiliary context pools those
rows into four temporal milestones.

Training-only graph:

```text
future DINO supports
    -> frozen FP32/no-grad object-aligned Teacher-G, once per training batch
    -> four ordered FutureObjectDynamics targets
future action/state + teacher targets
    -> V120 whole-segment recognizer target and coarse-action auxiliary
    -> losses only
```

Future evidence is not representable by the online API. Replacing future
supports can change targets/losses only, never a deployment action.

## Non-negotiable invariants

1. V120 `long` is the default behavior reference. An active V120 mechanism may
   be changed only for a recorded source defect or after its changed input
   distribution, gradient geometry and rollback condition are shown first.
2. Camera, space, local-M, global-K, progressive-candidate, interval, horizon,
   action-basis and type axes remain real until a named consumer. A reduced axis
   cannot be recreated with `expand` and renamed as original information.
3. Local M hypotheses are not persistent objects. G owns K=4 competitive
   objects plus null mass. G3 is a bounded correction of inherited ownership;
   typed assignments verify the same physical K support.
4. Learned flow is a continuous source-relative prior, not a nonzero quota.
   Flow-aligned history, feature/RGB warp, cycle, smoothness, uncertainty and
   refinement supervision retain explicit source/target units.
5. S reads full T5, ordered observable state/action history and typed G facts.
   It receives no frame progress, phase label, noisy ODE action or future
   teacher. Its four interval and 24 temporal values use V120 cumulative
   organization; the training recognizer uses one whole-segment latent family.
6. W1 owns the two near intervals and W2 the two far intervals. W2 may read W1
   without replacing either with a mean. The only W value below W is the
   directly supervised `FutureObjectDynamics`; no public W residual crosses.
7. P1 has two distinct frequencies. Every action-basis/global-K query chooses
   among the complete N=49 progressive bank once per observation before value
   contraction, using clean V120 typed action-basis tokens.  That protected
   detail cannot be deleted by a learned null or an additive public-content
   value. A legal learned null is an identity/zero-valued selector hypothesis
   grounded in current observable support/conflict; it may suppress only an
   optional detail innovation. A bounded compatibility residual may read the
   clean goal/action-basis query, but null availability cannot read noisy
   action, future teacher or a direct policy carrier. The V120 P1 policy block then
   completes the live P1 fact once per
   ODE step from noisy action, time and the cached detail.
8. P2 content, intent and coordinate scores are bounded in `[-1,1]`; learned
   temperatures stay in `[0.25,4]`. It cannot reopen RGB/DINO or consume a
   second W carrier.
9. Neutral future effect is algebraically neutral:

   ```text
   effect = 0
   interaction = 0
   protected_consequence = P1 factual base
   ```

10. P3 owns five V120 lanes: factual, precision, effect, temporal and
    state-change. All share the V120 typed noisy-action query. One protected
    consequence remains outside optional role selection.
11. Transition static/dynamic frequency is part of the model: only the G3 chart
    is cached; real-versus-learned-neutral coefficients read the current noisy
    action at every ODE step. Action and observable context come from one shared
    V120 canvas seed. A history proposal or zero proposal is not a legal
    substitute for either side of this counterfactual.
12. Bottom source count, order, value semantics and layer boundaries follow the
    active V120 decoder call. The two terminal generic records are the true P1
    and P2 `LayerContractAdapterHeads`; P3 is represented only by its typed bank.
    Generic neutral sources require an algebraic reason; they cannot be cleared
    merely to reduce memory or latency.
13. Capacity preserves the V120 ordered full-residual contraction:

    ```text
    output = u - Q diag(1-m(c)) Q^T u
    c=1 -> u; reducing c closes an ordered prefix inside span(Q)
    ```

    The orthogonal complement remains. Capacity zero is not a whole-block
    no-op; true no-update evaluation selects the pre-block prefix velocity.
    Execution cost remains audit-only.
14. Online action-owned boundaries use ordinary autograd. No artificial
    gradient, hard gate, entropy quota, route mass target, scalar progress loss,
    forced slot diversity or forced nonzero flow is legal.
15. Formal language training fails when the configured T5 file is absent. Only
    explicit null-goal smoke may omit it.
16. Fresh runs require an empty output directory. Exact resume verifies
    manifest, source/data/language identity, model/optimizer/scheduler and RNG.
    Schema 21/22 cannot exact-resume schema 23; compatible bottom migration must
    be explicit and reported.

## Typed boundaries

```text
DenseFactChart
  public/current DINO and local typed candidates           [B,C,8,8,M,*]
  progressive fine candidates                             [B,C,8,8,M,49,*]

ObjectFactSet
  content / semantic / appearance / geometry               [B,K,*]
  typed assignments                                        [B,K,C,8,8,M]
  camera coordinates / transport / support / validity      [B,K,C,*]

ObjectIntentState
  protected_goal_set                                       [B,4,H]
  history/object/typed object tokens                        [B,L|K,H]
  interval_queries / temporal_queries                      [B,4,H] / [B,24,H]
  state_change_evidence                                    [B,H]

FutureObjectDynamics
  current_reference                                        [B,K,D]
  successor / semantic_delta                               [B,4,K,D]
  transport / covariance / future_selector_validity        [B,4,K,C,*]
  visibility / persistence / uncertainty                   [B,4,K,*]
  future_address                                           [B,4,K,C,8,8]

ObjectTopTrainingTargets
  current_loss_support                                     [B,K,C,1]

ObjectFactualDock
  fact_by_object                                           [B,24,4,K,H]
  object/null posterior                                    [B,24,4,K+1]
  chart posterior                                          [B,24,4,K,C,8,8]
  query coordinates / aggregate fact

ObjectPolicyPlanDeltaBank
  protected_base + factual/precision/effect/temporal/state_change [B,24,4,H]

V120SeedContext
  state / state_history / compressed executed history             [B,1|3|7,H]
Terminal policy layer contracts
  P1/P2 trajectory / protected G3 rollout / state rows            two records

ControlledTransitionSource
  selector                                                 [B,512,H]
ControlledTransitionState
  selector / value                                         [B,512,H]
  real / learned-neutral coefficients                      [B,512,R]
```

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| observation/G | current DINO/raw history, coordinates, two learned flows, observable state | language, proposal, noisy action, teacher |
| S | T5, state/executed-action history, typed G facts | frame progress, phase label, noisy action, teacher |
| W | G facts, cumulative S intent, one causal coarse action intent | target/noisy action, history proposal, teacher, free W residual |
| P1 | full local/progressive chart, S, clean action-basis identity | W, history proposal, teacher, second visual read |
| P2 | factual dock, supervised W field, S interval query, noisy-action query | RGB/DINO reopen, free W hidden |
| P3 | P1 fact, protected consequence, S temporal/state change, noisy-action query | teacher, RGB/DINO, proposal, free W carrier |
| transition source | G3 public chart | W target, proposal, noisy action, teacher |
| transition dynamic | source, shared V120 noisy-action/context seed, plan | target action, teacher, future proposal tokens |
| bottom | protected consequence, five P3 lanes, transition, shared seed, P1/P2 contracts | RGB/DINO, teacher, duplicate W/P base |

## Loss and optimizer ownership

- Physical action flow matching remains the dominant 18-D V120 objective.
- Future dynamics owns successor, semantic delta, transport, covariance,
  visibility, persistence and detached-dispersion uncertainty. Address and
  reliability are target/diagnostic fields, not extra active losses.
- Teacher successor is the fixed uniform interval mean of
  `matched + null_probability * current_reference`; semantic delta is exactly
  successor minus current reference. Transport/covariance are uniform means of
  raw posterior moments. Entropy/reliability never contract these targets.
- Current loss support and online future selector validity are separate owners:
  every future loss and the training recognizer use current G camera validity;
  P2 alone consumes current validity times W-predicted future visibility.
- The V120 future row loss is retained except for one proven mathematical bug:
  its floored direction term now compares the two smoothed directions directly,
  so prediction equal to target has attainable zero loss.
- The whole-segment recognizer, coarse action and history proposal are
  auxiliary owners; their values do not become online shortcuts.
- Flow geometry and V120 execution-value supervision retain their existing
  external weights. Execution cost remains audit-only.
- Every trainable parameter has exactly one optimizer owner. Public G/S/W/P,
  the shared V120 canvas seed, P1/P2 layer adapters and transition use base LR;
  proposal uses `0.625x`, bottom decoder `0.7x`, and no-decay capacity basis
  `1.4x`. Structurally unreachable generic aliases are frozen, not rewritten
  into nonzero values.
- Global clip remains 1.0. Do not add per-owner clipping to conceal one owner
  consuming the gradient budget.

## Runtime, identity and inventory

- Observation/G/S/W, P1's N=49 detail read and the transition source build once
  per observation.
- The shared action/context seed, compact P1 policy block, P2/P3, two terminal
  contracts, dynamic transition and bottom run at every ODE step.
- Deployment performs action updates at mainline times `[0,.2,.4,.6,.8]`, then
  runs one complete dynamic forward at `t=1.0` for event/motion heads without
  updating the integrated physical field.
- Candidate-value tensors required by the execution objective are built on
  every train/eval loss forward, independently of the optional diagnostic
  batch budget. Deployment sampling leaves this loss-only boundary disabled.
- Teacher builds once per training batch and zero times in deployment.
- The N=49 P1 posterior is evaluated in checkpointed query chunks; it is not
  globally materialized for backward.
- Active manifest identity:

  ```text
  schema:       23
  observation:  restored_v120_three_frame_flow_dino_raw_local_chart
  top:          v120_cumulative_intent_four_interval_dynamics_split_support_selector_protected_candidate_p1_five_lane_p3
  bottom:       restored_v120_shared_seed_dynamic_p1_p1_p2_contracts_evidence_mmdit_dense512_execution
  training:     v120_mirrored_physical_flow_exact_teacher_current_support_event_boost_exact_role_lr
  runtime:      cached_observation_gsw_p1_detail_v120_nodes_clean_endpoint_teacher_isolated
  parameters:   182,724,214 total / 164,041,578 trainable
  ```

The lower count than the old monolith reflects removed inactive ancestry and
retired schema-20 replacement modules. Active G/S/W/P, the shared V120 canvas
seed, exact P1/P2 residual adapters, three Evidence MMDiT blocks, controlled
transition, execution/value/capacity and physical codec are present.

Established storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect raw HDF5 merely because cache/checkpoint roots moved.

## Verification and run

Focused local coverage checks full forward/backward, typed axes, G/P1 autograd,
teacher isolation, K permutation, P2 score bounds, neutral consequence,
per-ODE transition frequency, V120 bottom source semantics, optimizer ownership,
exact resume and semantic logging. CPU BF16 is a dtype boundary check, not a
CUDA memory result. Production acceptance still requires:

- fresh BF16 smoke and five-step deployment;
- batch-eight process peak no greater than 22 GiB;
- complete eight-epoch comparison with V120, including action/native,
  first/tail, horizon bands, arm/gripper, event/motion, G/S/W/P and gradients;
- no early divergence or later rebound hidden by a best-RMSE checkpoint.

The Schema-23 source repair is locally executable and covered by regression
tests. Production behavior is not accepted until a fresh smoke and complete
batch-eight run pass the recovery gate. If non-finite backward recurs, the
`gradient_failure` JSONL row owns the first named parameter, role, optimizer
group and finite/NaN/Inf statistics; no optimizer, scheduler or step update is
allowed for that batch. Dead `future_address` remains diagnostic debt and P1
learned null remains explicitly deferred.

Use a new empty output directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/mainline_v120_contract_repair_smoke \
nohup bash scripts/smoke_mainline.sh > mainline_v120_contract_repair_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
OUT_DIR=runs/mainline_v120_contract_repair_b8 \
nohup bash scripts/train_mainline.sh > mainline_v120_contract_repair_b8.log 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/mainline_v120_contract_repair_b8 \
  --recovery-baseline v120_long.log \
  --tail 120 --require-recovery --format text
```

## Authoritative source map

```text
configuration / identity / typed API:
  clearvla/mainline/config.py
  clearvla/mainline/manifest.py
  clearvla/mainline/interfaces.py
observation and top-to-bottom model:
  clearvla/mainline/model/
losses / optimizer / engine:
  clearvla/mainline/training/
sampling / validation / logs / checkpoints:
  clearvla/mainline/runtime/
entry / config / launchers:
  clearvla/mainline/train.py
  configs/mainline/object_intent_dynamics_323.json
  scripts/smoke_mainline.sh
  scripts/train_mainline.sh
```

V98, V107, V113-V120 and V122 are ancestry/comparison experiments. Rebuild an
old graph only from its serialized context, launcher and source commit; never
infer current behavior from its run label or paste an old contract here.
