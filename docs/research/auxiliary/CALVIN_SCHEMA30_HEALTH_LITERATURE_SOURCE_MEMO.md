# CALVIN Schema30 health: literature, source and matched-validation memo

Status: auxiliary research ledger; not an architecture contract and not an
authorization to mutate the active training run.

Last evidence refresh: 2026-09-06, Asia/Shanghai.  The full six-signal snapshot
below is frozen at train batch 13850; the action-loss trend was re-read through
batch 13900.

## Scope and evidence vocabulary

This memo starts from the formal remote CALVIN audit.  It does not start from a
preferred W, P2, solver, capacity or grounding repair.  Every conclusion uses
one of the following labels:

- **RUN-IF**: implementation fact for the source snapshot serialized when the
  active remote run started.
- **WS-IF**: implementation fact in the current local workspace.  This may be
  newer than the active run and must not be used retrospectively without a
  source-hash match.
- **LO**: direct remote log or run-context observation.
- **LI**: inference supported by cited literature, with its transfer limit.
- **H**: falsifiable hypothesis, not yet established.

The current architecture contract remains
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md).
This memo records diagnosis and experiment gates only.

## Source provenance and safety boundary

- **LO**: active run:
  `/data/senwang/data/calvin/runs/clearvla_calvin_abc_d_expanded_selective_v1_20260906`;
  planned batches `64106`, batch size `8`, BF16, inference steps `5`.
- **LO**: the run context records Git HEAD
  `09347da1a90399d5cad78c3c9630c54ed0acb878`, config digest
  `7d2a59ca...`, and source digest `eb344e58...`.  The serialized per-file hashes
  prove that the launch tree contained changes not represented by that Git HEAD.
- **LO**: the launch-time `dynamics.py` and `v120_p1.py` hashes are recoverable
  from Git object `cb7f0fd46fd53d4645174841c7cb62023e5ed882`; the launch-time
  `teacher.py` and `grounding.py` hashes are recoverable from
  `177160e62af6e81a5487a10e6dbc7821eb802564`; several other launch-time files
  were dirty snapshots and are identified only by the hashes in
  `run_context.json`.
- **LO**: the remote process directory and current local workspace have since
  changed.  For example, current `dynamics.py` hashes to `f46a1f62...`, while
  the run-start file hashes to `5fd0fb5c...`.
- **Decision**: do not change, reload or stop the active process.  Interpret the
  run using the serialized launch snapshot.  Use current workspace line anchors
  only to locate a future read-only harness or a post-run implementation.

## Formal remote snapshot

At batch 13850 (`13850 / 64106 = 21.6%`):

| Surface | Observation | Current reading |
|---|---:|---|
| integrity | ledger gap / contribution gap | `0 / -1.79e-9`; healthy |
| total / action flow | `0.53123 / 0.45039` | finite, noisy downward trend |
| action horizon | first / tail | `0.39479 / 0.45634`; tail remains harder |
| action bands | 1-4 / 5-12 / 13-24 | `0.41481 / 0.45068 / 0.45816` |
| CALVIN command | binary gripper F1 | `0.91412`; formal gripper head is working |
| W target/prediction | interval variation | `0.10545 / 0.01820` |
| W chronology | W2 adjacent cosine | `0.99678`; temporal rows remain very similar |
| local P2 optional router | entropy / max / carrier ratio | `0.99998 / 0.20115 / 0.00416` |
| compiler P2 effect | pre/post contract RMS | `0.08625 / 0.08601`; not contracted away |
| grounding | owner entropy / chart entropy | `0.06554 / 0.63175` |
| grounding separation | content cosine / chart overlap | `0.58984 / 0.23340` |
| execution capacity | capacity / reported effective basis mass | `0.999988 / 31.9996 of 32` |
| gradient window | preclip mean / max | `1.582 / 2.355`; post-global is about `1` |

**LO**: over train windows 12000-13900, action-flow mean is `0.46273`, linear
slope is `-0.00776` per 1000 batches, and the first/last ten-window means are
`0.46763 / 0.45605`.  This is slow, noisy improvement, not a proven flat line.

**LO**: 45 preclip spikes exceeded the audit threshold `5` through the latest
read.  The maximum was `80.82` at batch 2195, owned by
`observation.encoder.soft_address_compiler.target_dino_key.1.weight`.  Only
three occurred after batch 5000 (`6167`, `10483`, `13394`), and only one after
batch 13000 (`5.106`, Flow delta head).  There is no NaN, Inf, OOM or traceback.

## Six yellow lights

### 1. W keeps too much common mode

**RUN-IF**

- Launch-time W has four target intervals `4-8 / 8-16 / 16-32 / 32-48`.
  Teacher targets average available future visual supports inside each interval
  (`teacher.py` launch snapshot around lines 275-303).
- Future semantic and transport losses explicitly decompose common and interval
  innovation.  Semantic and transport each give equal internal weight to their
  common and innovation terms.  An adjacent-interval transition term is also an
  active contribution through `0.5 * objective.intent_structure`
  (`training/losses.py` launch snapshot around lines 557-669 and 1721-1756).
  Therefore “there is no interval supervision” is false.
- W's only launch-time action ingress is the four-row normalized physical
  `[absolute, adjacent-delta]` condition (`dynamics.py` launch snapshot at
  `cb7f0fd:303-314`).  W1 owns common/near and W2 reads W1 before writing far.
- The deployed action chart has 24 rows while the W future chart reaches 48.
  `PhysicalActionCondition.from_horizon_action` clips each interval to available
  action rows; the 32-48 interval therefore receives only the final available
  row.  This is deterministic ABI behavior, not learned far-horizon evidence
  (`model/types.py`; current equivalent at lines 911-939).
- With the current inclusive interval convention, the four action reductions
  consume `5 / 9 / 9 / 1` rows.  Boundary rows are shared by adjacent windows.
  Teacher also uses inclusive `offset >= lower and offset <= upper`, so supports
  at offsets `8`, `16` and `32` enter two neighboring Teacher means
  (`model/teacher.py` current lines 275-300).  This chart-level smoothing and
  the one-row final action condition must be tested before blaming W capacity.
- Existing validation already has `world_dynamic_neutral` and
  `wrong_action_world` counterfactuals (`train.py` launch snapshot at
  `cb7f0fd:690-823`), with the same refined cache/noise accounting downstream.

**LO**

- Teacher interval variation is about `5.8x` the prediction variation
  (`0.10545 / 0.01820`), while W2 adjacent cosine remains `0.99678`.
- W/P2/P3/bottom owner gradients are non-zero.  This proves connectivity, not
  useful chronology.

**LI**

- [Multi Time Scale World Models](https://arxiv.org/abs/2310.18534) supports
  representing dynamics at more than one temporal scale.  It does not prove
  that these four fixed CALVIN intervals or their loss weights are optimal.
- [DINO-WM](https://arxiv.org/abs/2411.04983) and
  [TD-MPC2](https://arxiv.org/abs/2310.16828) reinforce that a world model must
  be judged by downstream planning/control utility, not representation RMS
  alone.  Both use planning/RL settings unlike this supervised one-pass W.
- [ACT](https://arxiv.org/abs/2304.13705) shows why action chunks can reduce
  compounding behavior-cloning error.  Its action chunk is not a 48-step visual
  world target, so it cannot settle the current action/W horizon mismatch.
- [LingBot-VA / Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998)
  gives video frames and action rows explicit chunk/time identities and uses
  separate video/action conditioning, time embeddings, noise schedules and
  output projections.  This supports an explicit action-row-to-W-support chart;
  it does not support copying its large video generator or action step count.
  The corresponding [official source](https://github.com/Robbyant/lingbot-va)
  exposes these modality-specific paths.
- [Fast-WAM](https://arxiv.org/abs/2603.16666) reports that video/world
  co-training can retain much of its benefit while test-time explicit future
  imagination is skipped.  This directly motivates W-neutral versus W-active
  inference tests.  It does not justify deleting ClearVLA's compact W without
  its own matched result; see the [project page](https://yuantianyuan01.github.io/FastWAM/).

**Ranked hypotheses**

1. **H-W1 target predictability/chart smoothing**: future DINO association
   noise and inclusive boundary overlap make interval innovation less
   predictable than common change.
2. **H-W2 condition horizon**: the 24-row action condition underspecifies the
   32-48 W target, especially the final interval.
3. **H-W3 shared carrier/memory**: one shared physical projection plus W2's
   read from common/near memory favors a temporally smooth solution.
4. **H-W4 optimization**: easier common reconstruction dominates shared W
   parameters despite explicit innovation/transition terms.
5. **H-W5 downstream indifference**: P2/bottom use W common effect but ignore
   exact interval order; increasing W variation would then be cosmetic.

**Required matched tests before a W edit**

1. Reuse existing `world_dynamic_neutral` and `wrong_action_world` first.
2. Add validation-only cache transforms for `common-only`, `innovation-zero`,
   `interval cyclic-shuffle`, and cross-sample W donor.  Preserve current facts,
   support, cameras, initial physical noise and proposal/refined lifecycle.
3. Perturb only the W physical condition: per-interval shuffle, interval-4 tail
   clamp, and interval-3/4 swap.  Verify the first changed W boundary.
4. Rebuild W once with an oracle condition made from the matched ground-truth
   24-row action.  This separates proposal/action-condition quality from the W
   compiler and Teacher target.  It is an audit only and must not leak target
   actions into deployment.
5. Fit episode-split lightweight probes for Teacher common and innovation from
   current facts plus the legal physical action condition.  If even a probe
   cannot predict innovation, increasing W capacity/loss is poorly motivated.
6. Count a path as useful only when wrong order/sample is worse than zero/common
   and the paired target or closed-loop confidence interval excludes zero.

### 2. Local P2 optional routing remains almost uniform

**RUN-IF / WS-IF**

- `flow_jepa_typed_p2_route_*` belongs to the local precision refiner's four
  optional deltas (`semantic`, `appearance`, `geometry`, `horizon`) plus a null
  candidate.  It is not the whole P2 and is not the compiler
  `object_p2_effect_*` selector.
- RGB/detail base and precision form a protected policy carrier outside the
  optional router (`model/v120_p1.py` lines 944-982).  Null can suppress only an
  optional residual; it cannot suppress the main carrier.
- The actual imported router is
  `v120_core/role_delta_attnres.py`, not the similarly named implementation in
  `model/routing.py` (`model/v120_p1.py` lines 17-23).
- Query/key/source identities use small symmetric initialization and the null
  key starts at zero (`v120_core/role_delta_attnres.py` lines 291-303,
  359-390).  Five-way probability near `0.2` is the expected initial state.
- Local `p2_*_zero` modes exist and are valid lane-value operations at their
  first local consumer.  Local `p2_*_shuffle` rolls dimension 0 after
  batch/chunk/basis flattening (`model/v120_p1.py` lines 964-973); it is not a
  strict cross-sample donor and cannot support a strong sample-specificity claim.
- The formal `MatchedP2InterventionAccumulator` is wired to
  `model/compiler.py:ObjectFutureEffectReader`.  Its modes are
  `semantic_far_zero`, `geometry_value_all_zero`,
  `geometry_address_neutral`, and `geometry_value_and_address_zero`
  (`model/compiler.py` current lines 212-311; `train.py` current lines
  1363-1383).  These probe compiler P2 value/address, not the local four-lane
  router that emitted the high-entropy metric.
- Current `train.py` does not wire `v120_p1.py:set_address_eval_intervention`
  into formal validation.  Local four-lane zero/shuffle therefore needs a
  dedicated matched harness; synthetic tests prove the switches execute but do
  not provide action utility.

**LO / interpretation**

- Entropy `0.99998`, max `0.20115`, sample route std `0.00116` confirm almost no
  routing specialization.  Carrier ratio `0.00416` says the optional update is
  currently small.
- Compiler P2 effect RMS around `0.086` and its gradient do not prove that this
  separate local optional router causally helps actions.
- This is a valid yellow light, but not evidence of collapse: the protected
  carrier remains present and the observed distribution follows the initializer.

**LI**

- [Soft MoE](https://arxiv.org/abs/2308.00951) shows that soft mixtures need not
  become hard experts to be useful.
- [Switch Transformer](https://arxiv.org/abs/2101.03961) and
  [V-MoE](https://arxiv.org/abs/2106.05974) study sparse expert capacity and load
  balancing.  All four current lane values are already computed; their quota
  results do not transfer directly.
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) warns
  that auxiliary balancing gradients can interfere with the main objective.
  High entropy alone is not a reason to add entropy or quota loss.
- [Residual Adapters](https://arxiv.org/abs/1705.08045) is structurally
  consistent with a protected shared carrier plus small residual adaptation,
  but does not prove these four residual owners are useful.

**Required tests**

1. Measure exact-repeat numerical floor using identical batch, cache and initial
   noise.
2. Keep the already wired compiler-P2 panel, but report it under a separate
   name; it cannot answer the local-router yellow light.
3. Wire the four local lane-zero modes into an external validation harness at
   the module's actual first consumer.  Reuse only caches formed strictly
   upstream of that consumer and verify the first changed boundary.
4. Implement a validation-harness-only donor derangement before flattening, on
   the true batch axis, excluding identical task/instruction donors.
5. Report action delta and target RMSE by arm/gripper and 1-4/5-12/13-24 bands,
   plus command event F1/timing and closed-loop CALVIN success.
6. Do not change temperature, initialization, entropy loss or quotas unless a
   lane is causally useful, has non-zero owner-specific gradient, and routing
   remains condition-invariant after a sustained training plateau.

### 3. Action loss appears to flatten and the tail is harder

**RUN-IF**

- CALVIN formal flow loss is arm-only when
  `gripper_output_mode=calvin_binary_command`; the continuous gripper field is
  retained as a detached compatibility audit.  The binary command head owns
  formal gripper supervision (`training/losses.py` launch snapshot around
  `cb7f0fd:1084`; current equivalent around lines 1095-1169).
- Selective frame weighting reallocates the existing horizon mass toward motion
  and gripper-transition neighborhoods.  It does not add a hidden action-group
  gain (`training/losses.py` current lines 74-159 and 1133-1143).
- The run samples flow time with `v120_mirrored_beta_1_5_1` and validates with
  five inference steps.  The run context does not serialize the newer local Q5
  schedule, so the active CALVIN baseline must not be called Q5 without
  recovering its launch-time nodes.  Training flow-time sampling, numerical
  integration nodes and the 24-row physical action horizon are three different
  axes.

**LO / interpretation**

- The recent fitted slope remains negative and the last ten windows improve on
  the first ten.  Call this “slow/noisy improvement,” not convergence.
- Tail/band losses are consistently higher, but a weighted training scalar does
  not tell whether the cause is solver discretization, horizon target difficulty,
  data phase imbalance or closed-loop compounding.
- Continuous gripper flow around the historical audit value must not be used to
  diagnose the formal CALVIN head; binary command F1 is about `0.91`.

**LI**

- [Diffusion Policy](https://arxiv.org/abs/2303.04137),
  [pi0](https://arxiv.org/abs/2410.24164), and
  [RDT-1B](https://arxiv.org/abs/2410.07864) support iterative generative action
  decoding and action chunking, but their step counts, architectures and robot
  suites do not select ClearVLA's solver automatically.
- [Flow Matching](https://arxiv.org/abs/2210.02747) gives the continuous-time
  training/ODE framework.  [Rectified Flow](https://arxiv.org/abs/2209.03003)
  motivates straighter transport paths, and
  [DPM-Solver](https://arxiv.org/abs/2206.00927) motivates higher-order/fewer-step
  probability-flow integration.  DPM-Solver is derived for diffusion ODEs and
  is not plug-compatible proof for this robot FM velocity field.
- The [CALVIN paper](https://arxiv.org/abs/2112.03227) evaluates chained
  long-horizon task completion.  Offline per-row loss is necessary but cannot
  replace the official closed-loop sequence metrics.

**Required tests**

1. Finish a checkpointed deploy-style validation before changing the trainer.
2. Produce fixed-flow-time loss surfaces at least at
   `t={.001,.05,.1,.2,.4,.6,.8,.95,.999}`, stratified by task, phase, event,
   source, arm/gripper and horizon band.  Do not infer a time-sampling problem
   from the aggregate loss.
3. With fixed checkpoint, initial noise and observation/cache lifecycle, compare
   the launch baseline against candidate schedules/solvers twice: equal number
   of velocity evaluations and equal number of accepted intervals.  Report
   endpoint action delta, target RMSE, wall-clock latency and CALVIN success.
4. Compare action horizons/execution prefixes only as deployment experiments;
   a better numerical solver can improve sampling but cannot explain a training
   flow-loss plateau by itself.

### 4. Grounding assignments are becoming hard

**RUN-IF / WS-IF**

- G fuses DINO content, semantic, appearance, geometry and coordinate evidence
  before K-plus-null competition (`model/grounding.py` current lines 198-255).
  Typed reweighting cannot resurrect zero physical support (lines 444-477), and
  the only G objective is dense DINO reconstruction (lines 598-649).
- G does not read language.  T5 enters the trainable intent organizer through
  `goal_input`, then combines with history and object facts into public
  intervals (`model/intent.py` current lines 274-323).  A language effect must
  therefore be traced through S/P, not attributed to G directly.

**LO / interpretation**

- Owner entropy near `0.066` is sharp, but chart entropy `0.632`, content-pair
  cosine `0.590` and chart overlap `0.233` still show non-identical object and
  spatial carriers.  “Grounding collapsed” is not established.
- The stronger risk is shortcutting on color/material/texture or learning an
  in-distribution partition that fails compositional commands.

**LI**

- [Slot Attention](https://arxiv.org/abs/2006.15055) and
  [DINOSAUR](https://arxiv.org/abs/2209.14860) show that competitive slots plus
  reconstruction/self-supervised features can form object-centric carriers;
  low entropy alone is not collapse.
- [Grounding DINO](https://arxiv.org/abs/2303.05499) and
  [MDETR](https://arxiv.org/abs/2104.12763) use direct language-object alignment.
  They are useful alternatives, not evidence for a language-free G.
- [CLIPort](https://arxiv.org/abs/2109.12098) is closer to the current separation
  of semantic “what” and spatial “where,” though its architecture and benchmark
  remain different.
- [CLEVR/CoGenT](https://arxiv.org/abs/1612.06890),
  [ClevrTex](https://arxiv.org/abs/2111.10265), and
  [VIMA](https://arxiv.org/abs/2210.03094) motivate compositional color/shape,
  texture/light and systematic-generalization splits.  They do not provide a
  CALVIN success threshold.

**Required tests**

1. Stratify the fixed validation panel by `red/blue/pink x verb x object role`.
2. Same-scene color-word swaps: require G tensors/metrics to remain bit-exact,
   then trace change through S typed relevance, compiler P2 posterior/effect,
   action and closed-loop outcome.
3. Goal-token null and task-deranged shuffle with identical episode and initial
   noise.  A changed action alone does not prove visual selection because
   language has more than one downstream S/bottom route.
4. A genuine generalization claim needs a new training split that holds out
   verb-color or role-color combinations while keeping each individual word in
   training, plus lighting/texture and paraphrase tests.

### 5. Execution capacity is saturated

**RUN-IF / WS-IF**

- The reported `31.9996 / 32` is not 32 transformer layers.  The actual evidence
  depth is 3; 32 is `operator_rank`.  The source reports
  `capacity_ratio * rank` as effective basis mass/depth and explicitly says it
  is a continuous low-rank gate, not a guaranteed hardware-rank reduction
  (`v120_core/time_domain_mmdit.py` current lines 3151-3169).
- `operator_depth_logit_init=2.268683541`, so the launch capacity starts at
  `sigmoid(logit)=29/32=0.90625`, not at one.  Weight is zero-initialized and
  bias holds this configured logit (`v120_core/controller.py` lines 466-485).
- There is no positive compute/capacity cost in the formal loss.  Execution loss
  trains value prediction; compute-cost telemetry is detached.  A dense
  low-rank projection is still executed when the continuous gate is below one,
  so gate reduction alone does not guarantee latency reduction.
- Existing eval modes include learned, full-capacity and `29/32`
  (`three_basis_reduction`) matched paths; current workspace also exposes a
  reversible eval-only capacity override (`time_domain_mmdit.py` lines
  1209-1233).

**LO / interpretation**

- The model has learned to increase capacity from `0.90625` to about `0.99999`.
  This is allowed by the objective and may be the performance-optimal answer.
- Saturation is presently an efficiency yellow light, not an action-health red
  light.  It also cannot explain wall-clock waste until an actual kernel profile
  shows a reducible compute path.

**LI**

- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) and
  [PonderNet](https://arxiv.org/abs/2107.05407) obtain accuracy/compute tradeoffs
  using a ponder cost or halting prior.  Recurrent halting is not this rank gate.
- [SkipNet](https://arxiv.org/abs/1711.09485) and
  [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) obtain real FLOP control
  from hard skipping or an explicit compute budget.  Their speedups cannot be
  assigned to the current dense contraction.
- [DyDiT](https://arxiv.org/abs/2410.03456) supports timestep-dependent width or
  token allocation in image DiTs, but needs a specialized compute path and
  training; its speed numbers do not transfer to robot action decoding.

**Required tests**

1. Fixed checkpoint/cache/noise: learned -> full capacity -> 29/32.  Only if
   29/32 is non-inferior should the panel expand to 31/32, 24/32 and 16/32.
2. Report target/action deltas by task, flow time, horizon band, arm/gripper and
   closed-loop success.  `full ~= learned` only confirms saturation.
3. Profile CUDA wall-clock and kernels.  A non-inferior continuous gate does not
   imply acceleration; a hard prefix/rank implementation must demonstrate it.
4. Performance has priority: if lower capacity hurts, keep capacity at one and
   remove “automatic efficiency” from the expected contract instead of forcing
   sparsity.

### 6. Historical BF16/visual-Flow gradient spikes

**RUN-IF / WS-IF**

- The logged spike is the raw global norm before decoder-local and global
  clipping.  Global clipping then limits the step norm to about one.  Frequent
  clipping could still compress other owners on the same batch, so it is not
  automatically harmless.
- The gradient audit records parameter/group owner and splits the six-channel
  Flow delta head into flow and uncertainty channels.  Several correlation,
  normalization, grid-sampling and warp computations already have FP32/floored
  numerical boundaries; the owner name alone does not prove BF16 overflow.

**LO / interpretation**

- Spikes are concentrated early and their frequency has fallen sharply.  The
  45 events are mostly Flow delta head (27) and soft-address target DINO key
  (13), but one later event belongs to `top.intent.goal_input.weight`; they must
  not all be assigned one BF16-Flow cause.
- Current finite gradients, normal recent windows and improving loss support
  continued monitoring, not stopping the run or globally lowering LR.

**LI**

- [BFLOAT16](https://arxiv.org/abs/1905.12322) explains why BF16 shares FP32's
  exponent range; this reduces overflow risk but does not guarantee accurate
  gradient direction.
- [On Layer Normalization in the Transformer](https://arxiv.org/abs/2002.04745)
  and [Pascanu et al.](https://arxiv.org/abs/1211.5063) support warmup/clipping as
  protection against large early gradients.  Their mechanisms are not a direct
  diagnosis of this Flow-DINO graph.
- [RAFT](https://arxiv.org/abs/2003.12039) also trains optical flow with gradient
  clipping, while [SEA-RAFT](https://arxiv.org/abs/2405.14793) changes flow
  parameterization/refinement.  Neither proves that the current spikes are safe
  or share RAFT's cause.

**Required tests**

1. Reconstruct each spike's sampler indices and join episode/task/source row,
   motion percentile, flow magnitude, occlusion, correlation denominator/gain,
   warp/cycle loss, masks and loss contributions.
2. Fixed batch/checkpoint/RNG, backward-only and no optimizer step: compare BF16
   with FP32 using global norm ratio, per-owner norm, six delta-head channels and
   gradient cosine.
3. If FP32 is also large and the episode repeats the pattern, prioritize data or
   rare-motion diagnosis.  If BF16 alone repeatedly magnifies and rotates the
   gradient, move only the implicated operation/owner to FP32.
4. Intervene only if late spike frequency rises, non-finite values appear, or
   persistent clipping coincides with validation degradation.

## Decision-critical literature matrix

This table keeps the closest evidence separate from analogy-only citations.

| Work | Direct result relevant here | Legitimate use | Transfer limit |
|---|---|---|---|
| [CALVIN](https://arxiv.org/abs/2112.03227) | official evaluation is closed-loop, language-conditioned and long-horizon | use chained success/average sequence length after offline screening | does not select W intervals, solver or P2 routing |
| [Fast-WAM](https://arxiv.org/abs/2603.16666) | preprint Tables 1-2 report RoboTwin `91.8` and LIBERO `97.6` for test-time-fast WAM, versus `83.8/93.5` without video co-training; it also reports about `190 ms` and over `4x` speed versus imagine-then-execute | test whether W is valuable mainly in training or also at inference | different WAM, data and runtime; no license to remove ClearVLA W |
| [LingBot-VA](https://arxiv.org/abs/2601.21998) | explicit video-frame/action-row identities and separate video/action time/noise/output paths | make the 24-action-to-48-W temporal chart explicit and auditable | its 25/50-step generative configuration and scale do not transfer |
| [DINO-WM](https://arxiv.org/abs/2411.04983) / [TD-MPC2](https://arxiv.org/abs/2310.16828) | learned world state is judged through planning/control | require downstream intervention, not W activation-only evidence | planning/RL world models differ from supervised ClearVLA W |
| [Soft MoE](https://arxiv.org/abs/2308.00951) / [Aux-loss-free balancing](https://arxiv.org/abs/2408.15664) | soft routing can work; auxiliary balancing can interfere with task gradients | do not equate high P2 entropy with collapse | current optional residual lanes are not sparse experts |
| [ACT](https://arxiv.org/abs/1603.08983) / [PonderNet](https://arxiv.org/abs/2107.05407) / [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | adaptive compute is trained with cost/prior/budget and executable skipping | explain why an unpenalized dense capacity gate need not save compute | recurrent halting/token budgets are not a continuous rank multiplier |
| [BFLOAT16](https://arxiv.org/abs/1905.12322) / [RAFT](https://arxiv.org/abs/2003.12039) | BF16 has FP32 exponent range; flow training commonly clips gradients | motivate paired numerical replay and retain clipping | neither identifies the current spike owner or proves safety |

## Minimal checkpoint experiment matrix

The run directory currently exposes no checkpoint artifact, only
`metrics.jsonl` and `run_context.json`.  Execute this matrix at the first safe
checkpoint or completed epoch; do not attach it to the live training process.

| Priority | Panel | What it distinguishes | Existing support / smallest gap |
|---|---|---|---|
| P0 | exact repeat | numerical floor | same batch/cache/noise; harness only |
| P0 | W full vs world-neutral vs wrong-action-world | whether any W and sample-specific action-conditioned W help | already in matched core attribution |
| P0 | compiler P2 value/address panel | causal value of compiler semantic/geometry P2 | existing matched accumulator; not the local-router yellow light |
| P0 | capacity learned/full/29-of-32 | behavior sensitivity to capacity | existing eval modes |
| P0 | deploy validation on 34 tasks x 60 rows | offline health and color/task strata | run context already fixes 2040-row panel |
| P1 | W common-only / innovation-zero / interval shuffle / donor | chronology versus magnitude/sample identity | validation-only cache transform |
| P1 | W action-condition interval shuffle/far clamp | 24-to-48 condition bottleneck | validation-only W rebuild |
| P1 | W oracle ground-truth action condition | proposal/chart limitation versus W/Teacher limitation | audit-only W rebuild; never deploy target leakage |
| P1 | local P2 four lane-zero + true cross-sample typed-lane donor | local optional-router value and sample specificity | wire local intervention; move donor before flatten in harness |
| P1 | language null / task-deranged donor | language path and sample specificity | intervene before `encode_online` |
| P1 | fixed-time loss surface | training-time imbalance versus solver issue | aggregate existing per-sample losses |
| P1 | equal-NFE/equal-interval solver panel | numerical integration quality/latency | external evaluation only; active baseline remains fixed |
| P1 | spike BF16/FP32 replay | numerical versus data-driven spike | backward-only, no optimizer step |
| P2 | compositional color/role split | shortcut/OOD grounding | requires new split and retraining |
| P2 | official CALVIN 5-task chains | closed-loop long-horizon utility | benchmark rollout, not offline RMSE |

### Statistical unit and signs

- Resample at the episode or official CALVIN sequence level, stratified by task;
  do not treat 24 action rows or multiple batches from one episode as independent.
- Use paired bootstrap confidence intervals across identical examples/seeds.
- Existing accumulators define
  `mse_gain_vs_primary = primary_mse - counterfactual_mse`.  A useful removed or
  shuffled path should therefore produce a significantly **negative** gain.
- Action-delta RMS must exceed the exact-repeat floor.  Activation RMS or a
  non-zero gradient alone is not causal utility.
- Offline non-inferiority is a screening test.  Final architecture decisions
  require closed-loop CALVIN success/retry/sequence length and measured latency.

## Change gates and likely repair order

1. **Outer interface/target first**: if W innovation is useful and predictable
   but the far action condition is insufficient, repair the 24-to-48
   conditioning/target interface before enlarging W or increasing loss.
2. **Representation adapter second**: if a P2 lane or language/color distinction
   is useful but misrouted, correct donor semantics, normalization, or
   source-specific adapters before changing the final DiT block.
3. **Core network last**: alter W2, P2 router temperature, execution capacity or
   final action blocks only after paired intervention isolates that owner.
4. **Solver remains a deployment choice**: accept a different solver/schedule
   when it improves closed-loop performance at equal compute.  Do not use it as
   an explanation for the training loss.
5. **No forced minimality**: fewer steps, lower rank or lower entropy are not
   objectives by themselves.  Keep the larger computation when it wins on
   performance.

## CALVIN versus Pen/RDT portability

- CALVIN uses arm-only continuous flow plus a binary command head.  Pen/RDT and
  other embodiments may use continuous gripper, dexterous hand or bimanual
  actions.  Always report arm and end-effector/hand surfaces separately.
- Do not enlarge the shared physical field merely to make embodiments look
  uniform.  Keep source-native charts and embodiment adapters outside the core,
  then test whether the shared W/P2 evidence remains useful.
- A CALVIN color/command or binary-gripper repair must not silently become the
  Pen/RDT contract.  Conversely, a solver or capacity improvement is portable
  only after source-native decoding and closed-loop tests on each embodiment.

## Bottom line

The literature supports the diagnostic strategy, not an immediate architecture
change.  W chronology is the highest-priority causal question because its target
contains substantially more interval variation than its prediction.  P2 high
entropy, hard grounding, full capacity and historical spikes all have plausible
healthy explanations in the actual implementation.  The active run remains
healthy enough to continue.  The next decision point is a frozen-checkpoint,
paired validation panel—not a live-source repair.
