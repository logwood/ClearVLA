# Current ClearVLA Architecture Contract

Updated: 2026-08-09

This is the compact source of truth for the active top representation. Run
labels identify experiments; they do not select source semantics. Historical
reasoning belongs in Git history and the focused issue ledger.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        4
default run label:      v122
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global object slots:    K=4 plus an explicit null
formal language:        precomputed T5 .pt required
bottom compatibility:   evidence_mmdit_cvae_workspace_v1
launcher:               scripts/current_object_intent_dynamics_323.sh
smoke launcher:         scripts/current_object_intent_dynamics_323_smoke.sh
required capability:    object_intent_dynamics_323
```

`v122` is only a log and output-directory label. The source is selected by the
capability name. Schema 4 is incompatible with schema-3 top weights and must
start fresh. The small serialized `ArchitectureManifest` owns only identity,
topology, intervals, object count, language requirement and bottom
compatibility; executable types own shapes and semantics. Do not add a new
version-wide `_validate_vXXX_*` contract.

## Active graph

```text
current RGB / DINO / raw pair / learned flow
  -> unchanged Pre-G observation bank and local [C,8,8,M] chart
  -> G1-G3 DenseObjectGrounder
       one physical K+null assignment
       semantic / appearance / geometry bounded verification reads
       reversible K -> [C,8,8,M] correspondence
       camera coordinates/transport/support remain [K,C,*]
  -> S StatelessObjectIntentOrganizer
       factorized T5, ordered state/action history, typed K-object memories
       protected query identities separate from causal interval innovations
       24 temporal queries export only their zero-centred read innovation
       zero-centred observable state-change evidence
  -> CoarseActionIntent
       causal online innovation only; its learned query is not W-visible
  -> W1: 4-8 and 8-16 object effects
  -> W2: 16-32 and 32-48 object effects
       separate near/far output heads; W2 reads the ordered W1 sequence
  -> FutureObjectDynamics
       object semantic/status plus per-camera geometry/validity
  -> P1: one current high-resolution read
       K object -> local chart -> existing 3x3 RGB/detail micro-read
       emits ObjectFactualDock on the same K basis as W
  -> P2: separate semantic and geometry selectors over that dock
       geometry remains camera-specific; common-mode-free status only ranks
       physically valid candidates and never changes the null prior
  -> exact-zero-preserving consequence
  -> P3: centred object-detail precision / temporal / state-change innovations
       around one protected consequence
  -> exactly one protected-consequence bottom ingress
  -> unchanged Evidence MMDiT / CVAE / workspace / adaptive execution
```

Training-only graph:

```text
future DINO supports
  -> no-grad, object-specific Teacher-G association
  -> four ordered FutureObjectDynamics targets

future state/action + detached teacher effects
  -> factorized FuturePlanRecognizer
  -> detached action/state/object K/V targets for online S
```

Teacher-G and the recognizer cannot enter deployment values. Replacing future
supports may change targets and losses only; it must never change a deployment
action.

## Non-negotiable invariants

1. Local `M` hypotheses are not persistent objects. The global binder creates
   `K=4` objects with mutually competitive K+null mass. A local hypothesis
   prior is a conditional mixture; its complement is not absence.
2. G3 is `log(parent posterior) + bounded residual`. With a zero residual it
   exactly inherits G2. Semantic, appearance and geometry are verification
   posteriors inside the same physical K support; they cannot invent separate
   object identities or move mass to null. The existing G reconstruction
   budget is 65% object-prototype error, 20% within-object coordinate
   refinement and 15% target-normalized typed-field consistency. The typed
   term asks each read to explain its own field; it never forces posteriors to
   differ.
3. Object, camera, space, local-hypothesis, type and interval axes stay real
   until their named consumer. A reduced axis may not be recreated by
   `expand` and called an object, interval or camera.
4. S reads full T5 tokens, ordered observable history and typed K objects. It
   receives no frame index, scalar progress, phase label, noisy ODE action or
   future teacher. Goal/history/object/action/state K/V remain factorized. Its
   online and recognizer interval blocks are causal, and only innovation is
   matched to innovation.
5. S interval action/state/object K/V are future-oriented innovations. Query
   identities and cumulative diagnostic carriers cannot become W/P3 values.
   The protected
   current object enters W exactly once through `ObjectFactSet.content`.
6. `CoarseActionIntent` is the only action-conditioned W input. It is computed
   causally from online evidence, never target/noisy action, and has no bottom
   bypass. W consumes only its innovation, not its learned query carrier.
7. W1 owns the two near intervals and W2 owns the two far intervals. Their
   output heads are disjoint. W2 may read both ordered W1 tokens but may not
   replace them with a mean.
8. The only W-to-P value is the directly supervised
   `FutureObjectDynamics`. No public world residual or hidden W carrier crosses
   this boundary. W interval and decoder identities may modulate an
   object-owned value multiplicatively; they cannot form an object-free
   additive carrier.
9. Teacher temporal aggregation is object-specific. Stable successor content
   and end-biased semantic change are different targets; static content is not
   erased by a change mask.
10. P1 performs one high-resolution current-fact read. It reuses the global-K
    assignment and exports per-object facts, K+null posterior, chart posterior,
    coordinates and the mature aggregate P1 fact in `ObjectFactualDock`.
11. P2 semantic and geometry paths have different keys, values and posteriors.
    Geometry preserves `[I,K,C]` through selection. Visibility, persistence
    and uncertainty enter only as a bounded relative score after their common
    mode is removed. Physical per-camera validity alone may move mass to null;
    status cannot become a third cheap value or punish every static object.
12. P2 content, intent and coordinate scores are bounded to `[-1,1]`; their
    temperatures stay in `[0.25,4]`; zero-vector normalization has a finite
    Jacobian through a `0.25` norm floor.
13. A neutral future field is algebraically neutral: `effect=0`,
    `interaction=0`, and `protected_consequence=P1_fact` exactly.
14. P3 contains only three real innovations: precision, temporal and
    state-change. It has no duplicated factual/effect lanes. Precision reads
    only posterior-centred, already-materialized K-specific P1 detail and uses
    `effect+interaction` only as a query condition. Temporal reads only S's
    temporal innovation and `effect+interaction`. Learned temporal identity or
    action query cannot synthesize either value; state-change cannot synthesize
    a value when observed deltas and transport are zero.
15. The protected consequence enters the bottom once through protected detail.
    Controlled dynamics sees the pre-top trajectory seed plus current state and
    the final observed history, not a second copy of the P-modified trajectory.
    P2 and P3 also query with that pre-top trajectory seed under ordinary
    autograd; P1 facts and W consequences reach them only through their
    explicit typed operands. A provenance snapshot is not a gradient detach.
    Historical mid-cut and layer-contract towers are not constructed or
    executed for this capability, and the bottom evidence adapter receives an
    explicit typed-null layer row rather than a hidden second P carrier.
16. Online paths use ordinary autograd. No artificial gradient, hard gate,
    entropy target, route quota, progress loss, forced slot diversity or forced
    nonzero flow is allowed.
17. Flags-off historical paths remain reproducible. The active top may not
    silently load an older top manifest; unchanged bottom migration requires an
    explicit report.

## Typed boundaries

```text
DenseFactChart
  public_scene_base / dino_content
  candidate content / semantic / appearance / geometry
  candidate coordinates / support / validity / prior / flow transport

ObjectFactSet
  content / semantic / appearance / geometry             [B,K,*]
  physical candidate_assignment                          [B,K,C,8,8,M]
  typed semantic/appearance/geometry assignments          [B,K,C,8,8,M]
  object_to_chart                                         [B,K,C,8,8]
  camera coordinates / transport / support / validity       [B,K,C,*]
  object support / existence / validity                      [B,K,*]

ObjectIntentState
  protected_goal_set / ordered history
  interval action/state queries + separate innovations    [B,4,H]
  interval object keys/values                             [B,4,K,H]
  typed object tokens                                     [B,K,H]
  temporal queries + separate exported innovations        [B,24,H]
  state_change_evidence                                   [B,H]

FutureObjectDynamics
  current_reference                                       [B,K,D]
  successor_content / semantic_delta                      [B,4,K,D]
  transport / covariance / camera validity                [B,4,K,C,*]
  visibility / persistence / uncertainty                  [B,4,K,*]
  future_address                                          [B,4,K,C,8,8]

ObjectFactualDock
  fact_by_object                                          [B,T,Q,K,H]
  object_posterior + null_posterior                       [B,T,Q,K+1]
  chart_posterior / per-camera coordinates on the same K basis
  aggregate_fact                                          [B,T,Q,H]

ObjectPolicyPlanDeltaBank
  protected_base
  precision / temporal / state_change
```

Each type has one definition in
`clearvla/policy/object_intent_dynamics_323/`; do not duplicate these semantic
objects in `trunk.py` or select them by a V-number.

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| G | current observation bank, coordinates, flow, observable state/registers | language, history, proposal, noisy action, teacher |
| S | full T5, state/executed-action history, typed G objects | frame position, progress, phase, noisy action, teacher |
| recognizer | future action/state and detached teacher effects; training only | deployment value path |
| W | current G objects once, factorized S innovations, one CoarseActionIntent | target/noisy action, generic aliases, teacher, public world residual |
| P1 | G K-to-chart assignment, canonical S query, action-basis query | W effect, teacher, second visual read, mean-goal/last-history aliases |
| P2 | ObjectFactualDock, supervised FutureObjectDynamics, S interval intent, action query | raw/DINO reopen, public W hidden, status value |
| P3 | centred per-object P1 detail, consequence innovation, S temporal innovation/state-change, action query | aggregate P1 fact as optional value, cumulative consequence as optional value, temporal identity, teacher, RGB/DINO, completion/progress |
| bottom | one protected consequence, three optional P3 innovations, observable state/history | second protected consequence through dynamics context |

## Loss ownership

- Action flow matching remains the dominant objective.
- The existing future budget owns stable successor content, ordered semantic
  change, transport/covariance, visibility/persistence and detached teacher
  dispersion. Each component is optimized in its own detached, variance-floor
  unit before the unchanged internal coefficients are applied; native-unit
  errors remain diagnostics. Stable successor and semantic change are no
  longer the same algebraic error.
- The existing interval/structure budget owns object reconstruction,
  factorized online-S matching, recognizer reconstruction, chronological
  transition and coarse action structure. Deleted losses are not reassigned.
- Runtime is the single canonical loss owner. The trunk exports prediction,
  target and diagnostics; it does not build a competing raw future loss.
- Existence is read-conditioned object-vs-null confidence. Validity is physical
  source support. Allocation share, existence and validity are never aliases.
- Confidence/calibration may affect a selector or loss once. It never scales
  the selected P2 effect a second time.

## Runtime and storage contract

- Teacher association is built once per teacher-forced train/representation
  batch and zero times in deployment. The explicit teacher-forced call boundary
  owns permission to carry future evidence; `train()` mode does not.
- Five-step deployment caches visual/G/S/W/P1 once. P2, P3 and the bottom action
  tower remain ODE-step dependent. P1 never opens a second visual read.
- W/P2/P3 audit reductions are conditional on `collect_audit_metrics`; normal
  execution performs the same dynamics/selector/consequence computation but
  does not launch typed-router weights, pairwise cosines, entropy, FP32 RMS or
  interval-mass reductions when those diagnostics are not requested.
- The object capability executes no legacy mid-cut or layer-contract readout in
  training or deployment. Their external loss weight is zero, their modules
  own no optimizer state, and Evidence MMDiT accepts an explicit zero layer row.
- Every trainable parameter belongs to exactly one optimizer group.
- Stage1 initialization is disabled for this fresh top.
- Formal training requires the configured T5 file. Only an explicitly separate
  null-goal smoke may omit language.
- Established defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect the raw-HDF5 default merely because caches and weights moved.

## Verification state

Local CPU tests cover schema/resume identity, physical/typed G continuity,
field-specific typed consistency, camera-axis preservation, mass conservation,
object permutation through S/Teacher/P2, teacher FP32 isolation, ordered
successor-versus-delta targets, query-only innovation zero semantics,
object-owned W conditioning, disjoint W1/W2 heads, common-status invariance,
typed P2 independence and bounded scores, exact-neutral consequence, centred
P3 precision/temporal zero semantics, full BF16 forward/backward, optimizer
ownership, schema-aligned gradient diagnostics, teacher-forced boundaries,
future-target/action isolation,
audit-switch bit-exactness and five-step cache behavior.

This source still requires a fresh server smoke and long run. No action-quality
improvement is claimed until the new v122 log and frozen-checkpoint causal
interventions exist. The decisive empirical questions are kept in
`TOP_ARCHITECTURE_ISSUE_LEDGER.md`.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/current_object_intent_dynamics_323_smoke.sh > v122_smoke.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/current_object_intent_dynamics_323.sh > v122.log 2>&1 &
```

Override `OBJECT_323_BATCH_SIZE`, `DATA_ROOT`, cache roots, T5 path or
`OUT_DIR` only when the server layout differs. The long launcher defaults to
batch 8 and refuses `--resume`.

## Authoritative source map

```text
typed G/S/Teacher/W/P2/P3:
  clearvla/policy/object_intent_dynamics_323/
ancestral observation bank and high-resolution reader:
  clearvla/policy/flow_dino_evidence.py
placement, ObjectFactualDock construction and bottom ingress:
  clearvla/policy/trunk.py
losses, cache, optimizer and logs:
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
configuration and serialized context:
  clearvla/policy/config.py
  clearvla/cli/train_v40_policy.py
launchers:
  scripts/current_object_intent_dynamics_323.sh
  scripts/current_object_intent_dynamics_323_smoke.sh
tests:
  tests/test_object_intent_dynamics_323.py
  tests/test_flow_dino_evidence.py
  tests/test_audit_policy_logs.py
active issue ledger:
  docs/research/TOP_ARCHITECTURE_ISSUE_LEDGER.md
```

## Historical boundary

V98, V107, V113, V114, V117, V118, V119 and V120 are ancestry or comparison
experiments. Reconstruct them from their scripts, serialized run contexts and
Git history. Never paste an old contract back into this file or infer the
active graph from an experiment name.
