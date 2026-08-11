# Current ClearVLA Architecture Contract

Updated: 2026-08-11

This file is the compact source of truth for the active independent mainline.
Run labels identify experiments; they never select source semantics.  The
evidence and reasons behind repairs live in
[`TOP_ARCHITECTURE_ISSUE_LEDGER.md`](TOP_ARCHITECTURE_ISSUE_LEDGER.md).

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        19
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global object slots:    K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0, two adjacent learned flows
formal language:        precomputed 4096-wide T5 .pt required
bottom:                 deterministic 3-block Evidence MMDiT + capacity/execution
long launcher:          scripts/train_mainline.sh
smoke launcher:         scripts/smoke_mainline.sh
resolved config:        configs/mainline/object_intent_dynamics_323.json
```

The active graph lives entirely in `clearvla/mainline/`.  It does not import
the V39 trainer/runtime/trunk or a `current_vXXX` launcher.  The compact
`ArchitectureManifest` serializes graph identity; typed interfaces and
executable tests own tensor shapes and zero semantics.  Do not add a new
version-wide `_validate_vXXX_*` contract.

## Active graph

```text
current-only input
  RGB/DINO at -8,-4,0 + observable state/executed-action history + T5 goal
    -> shared raw/DINO feature extraction
    -> learned flow -8->-4 and -4->0
    -> flow-aligned causal visual innovation on the current chart
    -> Pre-G DenseFactChart [camera,8,8,local-M]

DenseFactChart
    -> G1-G3 typed role hosts
    -> DenseObjectGrounder
         one competitive global K+null ownership
         semantic / appearance / geometry verification on the same K
         reversible K <-> [camera,8,8,local-M] correspondence
    -> ObjectFactSet

T5 + observable history + ObjectFactSet
    -> StatelessObjectIntentOrganizer S
         factorized goal/history/object reads
         four interval innovations and 24 temporal innovations

executed-action history
    -> HistoryActionProposal
         rows -24,-16,-12,-8,-6,-4,-2,-1
         four recent + three summary tokens, two causal blocks

ObjectFactSet + S + one causal CoarseActionIntent
    -> W1 effects for 4-8 and 8-16
    -> W2 effects for 16-32 and 32-48
    -> FutureObjectDynamics (the only W value visible below W)

ObjectFactSet + S + action-basis query
    -> P1 one high-resolution current-fact read
         every [time,query,global-K] reads the full local chart
         one packed 3x3 RGB/detail micro-read
    -> ObjectFactualDock

ObjectFactualDock + FutureObjectDynamics + S + action query
    -> P2 bounded semantic/geometry consequence read
    -> exact-zero W effect and interaction
    -> protected current factual consequence

protected consequence + local P1 detail + S temporal/state change
    -> P3 additive typed plan compiler
         precision base + bias-free consequence interaction
         temporal base + bias-free consequence interaction
         state-change lane

W dense chart + clean proposal - same-network zero proposal
    -> 512 spatial controlled-transition directions
    -> 24 horizons x 4 action bases = 96 read-only transition tokens

one protected consequence + typed P3 lanes + transition + observable history
    -> deterministic organizer
    -> three read-only Evidence MMDiT blocks
    -> full-width nested capacity + monotone soft continuation
    -> 18-D physical action velocity + event head + motion head
```

The selected bottom is deterministic.  Historical `latent_cvae_*` names,
variational CVAE posterior sampling and hierarchical workspace were inactive in
the V120 reference configuration and are not missing active algorithms.  The
three Evidence MMDiT blocks, proposal, controlled dynamics, capacity,
continuation, event/motion heads and physical action codec are active.

Training-only graph:

```text
future DINO supports
    -> frozen/no-grad object-specific Teacher-G association
    -> four ordered FutureObjectTargets
    -> future dynamics losses only
```

Future evidence is absent from the online API.  Replacing future supports may
change targets and losses only; it cannot change a deployment action.

## Non-negotiable invariants

1. Local `M` hypotheses are not persistent objects.  G creates `K=4` global
   objects with competitive K+null mass.  Object, camera, space, local-M,
   interval and type axes remain real until a named consumer; a reduced axis
   cannot be recreated with `expand` and renamed.
2. The three hosted G blocks affect the online graph through a dedicated
   `public_address_key`.  The public scene is address context, not a value
   copied into every local candidate or exported K content.
   G reconstruction is 30% detached-responsibility prototype distortion, 55%
   dense spatial refinement and 15% target-normalized typed consistency.  It
   must not be optimizable by making every K content vector identical.
3. G3 ownership is a bounded correction of inherited ownership rather than an
   independent identity system.  Semantic, appearance and geometry assignments
   verify fields on the same physical K support.
4. Online history consists of three observable visual states and two adjacent
   motions.  Earlier DINO change is flow-aligned through -8->-4->0 before it is
   compared on the current chart.  A same-cell difference across moving frames
   is not a legal history feature.
5. Learned flow is a continuous source-relative geometric prior, not a forced
   nonzero address or an action quota.  Source/target units and alignment
   direction stay explicit.  Learned-feature warp, literal-RGB photometric
   anchoring, cycle, smoothness, uncertainty and refinement supervision remain
   inside the existing geometry budget.
6. S reads full T5, ordered observable state/action history and typed G facts.
   It receives no frame progress, phase label, noisy ODE action or future
   teacher.  Goal/history/object identities stay factorized; query identity is
   not exported as an intent value.
7. W1 owns the two near intervals and W2 the two far intervals.  W2 may read
   ordered W1 outputs but may not replace them with a mean.  The only W-to-P
   value is directly supervised `FutureObjectDynamics`; no public W residual
   or unsupervised hidden carrier crosses the boundary.  Teacher-G must first
   confidence-blend a high-entropy association to current content, zero
   geometry and the unit-mass current address before it becomes a target.
   These neutralized fields and S recognition use physical object validity.
   Reliability remains a calibrated target/diagnostic; it may not be applied
   a second time to erase the neutral W target and create a free carrier.
8. P1 performs exactly one high-resolution current read.  Each action-basis
   query/global-K pair refines over the complete local chart.  Global K is a
   prior, not a single coordinate bottleneck.  P1 uses ordinary autograd so
   action loss can reach G ownership.
9. P2 content, intent and coordinate scores are bounded to `[-1,1]`; learned
   temperatures stay in `[0.25,4]`; zero-vector normalization has a finite
   Jacobian through the configured norm floor.  Its intent score uses only
   observable interval innovations, never the cumulative learned interval
   identity.  Status/calibration cannot form a cheap nonzero value.
10. A neutral future field is algebraically neutral:

    ```text
    effect = 0
    interaction = 0
    protected_consequence = current P1 factual base
    ```

    Neutral W does not delete legal current detail or observable temporal
    evidence.
11. P3 is additive, not an annihilating multiplicative gate.  Precision owns
    query-specific P1 detail plus a bias-free detail-by-consequence
    interaction.  Temporal owns S temporal innovation plus a bias-free
    temporal-by-consequence interaction.  Noisy ODE action can alter the
    temporal lane only inside that nonzero consequence interaction.  W
    interaction is zero when W is neutral; factual/temporal bases remain
    available.
12. Controlled transition first constructs all `4*2*8*8=512` spatial W
    directions.  It uses `coeff(clean proposal)-coeff(zero proposal)` through
    the same deterministic coefficient network, so a
    neutral proposal produces exact zero value.  Pooling retains four action
    bases per horizon (96 tokens) before bottom attention; it may not collapse
    directly to one global token per horizon.
13. Capacity is exact-zero, full-identity and non-expansive:

    ```text
    output = c*u + Q diag(g-c) Q^T u
    c=0 -> 0
    c=1 -> u
    ```

    Rank 32 controls ordered anisotropy; it is not a rank-32 truncation of the
    entire 512-wide block residual.  Execution cost remains audit-only.
14. The protected consequence enters the bottom exactly once.  P3 optional
    lanes and controlled transition are read-only evidence.  The bottom never
    reopens RGB/DINO, reads a future teacher, or receives a second free W base.
15. Online paths use ordinary autograd.  No artificial gradient, detach at an
    action-owned boundary, hard gate, entropy target, route quota, scalar
    progress loss, forced slot diversity or forced nonzero flow is allowed.
16. Formal runs start fresh unless exact resume verifies manifest, source/data/
    language identity, model state, optimizer ownership/state, scheduler and
    RNG before mutating the run.  An old top cannot load silently; compatible
    bottom migration requires an explicit report.

## Typed boundaries

```text
CurrentObservation
  dino_history                                         [B,3,C,P,D]
  raw_rgb                                              [B,3,C,3,R,R]

DenseFactChart
  public_scene_base / current dino_content
  candidate content / semantic / appearance / geometry [B,C,8,8,M,*]
  candidate coordinates / support / validity / prior / flow transport

ObjectFactSet
  content / semantic / appearance / geometry            [B,K,*]
  candidate and typed assignments                        [B,K,C,8,8,M]
  object_to_chart                                        [B,K,C,8,8]
  camera coordinates / transport / support / validity   [B,K,C,*]

ObjectIntentState
  protected goal and ordered history
  interval action/state innovations                      [B,4,H]
  interval object keys/values                            [B,4,K,H]
  temporal innovations                                   [B,24,H]
  observable state-change evidence                       [B,H]

FutureObjectDynamics
  current_reference                                      [B,K,D]
  successor_content / semantic_delta                     [B,4,K,D]
  transport / covariance / validity                      [B,4,K,C,*]
  visibility / persistence / uncertainty                 [B,4,K,*]
  future_address                                         [B,4,K,C,8,8]

ObjectFactualDock
  fact_by_object                                         [B,24,4,K,H]
  object/null posterior                                  [B,24,4,K+1]
  local chart posterior                                  [B,24,4,K,C,8,8,M]
  per-query coordinates and aggregate fact

ObjectPolicyPlanDeltaBank
  protected_base / precision / temporal / state_change

ControlledTransitionState
  selector / value                                       [B,24*4,H]
  real / neutral coefficients                            [B,24*4,R]
```

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| observation/G | current DINO/raw history, coordinates, two learned flows, observable state | language, proposal, noisy action, teacher |
| S | full T5, state/executed-action history, typed G facts | frame progress, phase label, noisy action, teacher |
| W | G facts, factorized S innovations, one causal coarse action intent | target/noisy action, generic aliases, teacher, public W residual |
| P1 | full G local chart, S query, current action-basis query | W effect, teacher, second visual-bank read |
| P2 | factual dock, supervised W field, S interval intent, current action query | raw/DINO reopen, public W hidden, status value |
| P3 | local P1 detail, protected consequence, S temporal/state change, action query | teacher, RGB/DINO, completion/progress, free W carrier |
| transition | dense current/W chart, clean history proposal, observable history | teacher, target action, noisy ODE action |
| bottom | one protected consequence, typed P3/transition evidence, observable history | RGB/DINO, teacher, duplicate W/P base |

## Loss and metric ownership

- Physical action flow matching remains the dominant objective on the 18-D
  action field.  This is the source-resolved V120 branch:
  `legacy_independent` arm absolute/delta coordinates, the first six
  `legacy_handcrafted` gripper coordinates, independent standard-normal
  physical noise and decode delta blend `0.25`; it is not a reconstructed
  manifold/Parseval variant.  Horizon rows use the V120 schedule: bands
  `1.0/1.1/1.2`, first row `+0.05`, then exact unit-mean normalization.  This
  preserves real far-row mass instead of assigning equal total mass to unequal
  band lengths.
- Gripper event labels use raw action units and threshold `0.10`.  Event/hold
  row balancing is separate from horizon mass.  The event head and motion head
  retain explicit auxiliary losses; decoded action events are not aliases for
  either head.
- Future dynamics owns successor, semantic delta, transport, covariance,
  visibility, persistence, uncertainty calibration and address terms within
  the unchanged external future budget.
- Flow geometry averages the two adjacent frame-pair objectives under the same
  outer weights and logs both pairs separately.
- Validation reports normalized and physical RMSE, first/first8/tail,
  `1-4/5-12/13-24`, arm/gripper, temporally tolerant direction-aware decoded
  gripper events, row-wise event-head metrics and physical-field motion-head
  metrics.  Names include their real semantic source.  On the configured four
  diagnostic batches it also reuses the primary cache and initial noise for
  proposal-zero, bottom no-updates and bottom full-updates action ablations;
  each reports coverage, action delta and signed MSE gain versus primary.
- A diagnostic training step currently serializes 568 finite active archival
  metrics, versus 287 parsed batch metrics in V120.  Executable coverage locks
  both that V120 observability floor and per-owner G/S/W/P/transition/bottom/
  gradient prefix floors; compact console filtering never removes the JSONL
  record, including exact-zero active paths.
- The active gripper objective uses event/hold row balancing to address the
  conservative-event failure seen in V120/V122.  JSONL therefore records both
  the real event-balanced action/decoded losses and exact event-unweighted
  `*_v120_comparable` rows.  Cross-run recovery uses the latter; the former
  remains the quantity sent to backward.  Nineteen `loss_contrib_*` rows and a
  separate contribution gap expose every applied objective weight exactly.
- Active logs cover action/flow/G/S/W/P/transition/bottom/owner-gradient
  boundaries.  Exact zero is hidden only for inactive ancestry; legal zero
  contracts, conservation errors and owner gradients remain visible.
- V120's supervised execution-value reader was active and learned a useful
  candidate ranking, but it depended on an expensive multi-candidate action
  chart and did not prevent V122 capacity collapse.  The current one-graph
  controller therefore remains directly action-gradient trained; matched-noise
  no-update/full-update ablations and controller/capacity/basis gradients are
  the required equivalence evidence.  Do not reintroduce execution cost into
  the loss or restore the old candidate chart without causal failure evidence.

## Runtime and storage

- Teacher targets are constructed once per teacher-forced training batch and
  zero times during deployment.
- Five-step deployment builds observation/G/S/W/P1/history proposal/controlled
  transition once.  P2, P3 and the bottom remain ODE-step dependent.
- The observation encoder runs two adjacent online flows; it does not repeat
  them across the five ODE steps.
- Every trainable parameter belongs to exactly one optimizer group.  Diagnostic
  ownership is declared by modules rather than reaching into private fields.
- Zero-initialized action-facing boundaries may suppress their upstream
  gradients on the first optimizer step.  After that boundary has taken one
  update, the second diagnostic step requires every trainable tensor to carry
  a nonzero ordinary-autograd signal; aggregate owner norms may not hide a
  dormant submodule.
- The public G/S/W/P and controlled-transition owners use the base LR.  The
  source-resolved V120 geometry is explicit rather than inherited through old
  launchers: history proposal uses `0.625x`, the active bottom decoder uses
  `0.7x`, and the no-decay capacity basis uses `1.4x` of the base LR.  Every
  resolved role LR is serialized in `run_context.json` and the decision-critical
  values are emitted in train logs.
- Formal training requires the configured T5 file.  Only explicit null-goal
  smoke mode may omit it.
- Established storage defaults:

```text
raw HDF5:    /data/liang.zhang/dataset/grab_pen_single/grab_pen_single
cache root:  /data/senwang/data
T5 weights:  /data/senwang/checkpoint/grasp_pen_embed.pt
```

Do not redirect raw HDF5 merely because cache and checkpoint roots moved.

## Architecture identity and inventory

```text
schema:       19
observation:  causal_three_frame_dino_raw_two_flow_pre_g_v5
top:          object_intent_dynamics_323_keyed_g_local_p1_additive_p3_v14
bottom:       typed_evidence_mmdit_dense_transition4basis_zero_proposal_fullwidth_capacity_v9
training:     single_stage_physical_action_v120_role_lr_horizon_event_v12
runtime:      cached_five_step_ode_lossless_semantic_logging_v11
parameters:   171,940,734 total / 171,838,334 trainable
```

The difference from V120's total parameter count is primarily removed frozen
ancestry, not a smaller active optimizer.  The active three-block bottom,
G1-G3/P1 hosts, proposal and controlled transition are present.

## Verification and run

Local executable coverage (124 focused regressions) includes typed shape/provenance, G/P1 autograd,
teacher isolation, object permutation, neutral W/P3 semantics, capacity
zero/identity/non-expansion, two-flow history alignment, dense transition,
single static cache, optimizer ownership, exact resume and semantic log
parsing.  A complete CPU BF16 forward/backward is finite; this is a dtype
boundary check, not a CUDA memory measurement.  A fresh server smoke and
controlled eight-epoch comparison against V120 are still required before
claiming recovered action quality.

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/smoke_mainline.sh > mainline_smoke.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_mainline.sh > mainline.log 2>&1 &

uv run python -m clearvla.tools.audit_policy_logs \
  runs/schema19_recovery_b8 \
  --recovery-baseline v120_long.log \
  --tail 120 --require-recovery --format text
```

The resolved config defaults to batch 8 and four data workers.  Override
`MAINLINE_BATCH_SIZE`, data/cache/T5 paths or output directory only when the
server layout differs.

## Authoritative source map

```text
configuration / manifest / typed API:
  clearvla/mainline/config.py
  clearvla/mainline/manifest.py
  clearvla/mainline/interfaces.py
current observation and top-to-bottom model:
  clearvla/mainline/model/
losses / optimizer / engine:
  clearvla/mainline/training/
sampling / validation / logs / checkpoints:
  clearvla/mainline/runtime/
entry point and resolved spec:
  clearvla/mainline/train.py
  configs/mainline/object_intent_dynamics_323.json
audit parser:
  clearvla/tools/audit_policy_logs.py
focused issue ledger:
  docs/research/TOP_ARCHITECTURE_ISSUE_LEDGER.md
tests:
  tests/test_mainline_*.py
  tests/test_audit_policy_logs.py
```

## Historical boundary

V98, V107, V113-V120 and V122 are ancestry or comparison experiments.
Reconstruct them from their serialized context, script and Git commit.  Never
infer current behavior from a run label or paste an old contract back here.
