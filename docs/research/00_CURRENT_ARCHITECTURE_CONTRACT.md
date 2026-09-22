# Current ClearVLA architecture contract

Updated: 2026-09-22

This is the compact source of truth for the active mainline graph. Read it
before changing the V96+ top representation, Flow-DINO/JEPA, role hierarchy,
language/history conditioning, long-horizon phase handling or the
top-to-bottom evidence path.

Operational state belongs in
[auxiliary/ACTIVE_MAINLINE_HANDOFF.md](auxiliary/ACTIVE_MAINLINE_HANDOFF.md),
open behavior questions in
[CURRENT_MAINLINE_ISSUES.md](CURRENT_MAINLINE_ISSUES.md), and future repository
or refactor work in
[CURRENT_MAINLINE_REPAIR_PLAN.md](CURRENT_MAINLINE_REPAIR_PLAN.md).
Expanded historical narratives remain recoverable from Git and the archive;
they do not define the current graph.

## Structural rebuild branch status

`codex/structural-rebuild-20260921` is based on immutable commit
`0f07160d692ec8c8302880420a3d93d74512c39b`. The implemented units are **M1a physical-step history** and **M1b real-tail
current-state coverage with source-owned future-label support**.
M1c additionally implements native-state/feature-state separation; M1d adds
source-timed visual history with one support contract through flow, G memory,
S/W motion and Teacher. The latest explicit candidate is
`configs/mainline/structural_rebuild_m4c_calvin.json`. Legacy and M1a/M1b/M1c/M1d/M2/M3/M4b
configs remain reproducible controls, not automatic aliases of this candidate.
M2 preserves full G1 support through the actual G2 sampler and G3/P1
value consumers. M3 adds one soft local observation law for typed properties
and their positions. M4a connects completed G3 context to the actual K binder.
M4b maps full candidate mass into the actual current image for entity reverse
lookup, geometric summaries and reconstruction. This is not persistent tracking.
Global/persistent G3 entity identity, S-task,
W-supervision and the deeper P redesign remain open. No learned behavior
is established by these source changes. The user-authorized rebuild may revise
historical implementation details; accepted source semantics belong here,
future work in the repair plan, and test/publish status in the handoff.

## M1c native state and model feature boundary

`configs/mainline/structural_rebuild_m1c_calvin.json` selects
`calvin_tcp_rotation6d_v1` together with real-time history and observed-tail
labels. It is a new-training candidate, not a checkpoint migration or a claim
of improved task success. The M1a/M1b and legacy configurations remain controls.

CALVIN source state is still seven native coordinates: world TCP xyz in metres,
roll/pitch/yaw in radians, and gripper opening in metres. Statistics are fitted
on the same real native state rows as before. The model input is ten values:
affine-normalized xyz, the first two columns of `Rz(yaw) Ry(pitch) Rx(roll)`
in column order, and affine-normalized opening. Rotation columns are never
normalized using the Euler statistics. The shared encoder owns current,
history, future-target and online state features; unavailable future features
are zeroed after encoding as well as masked in every target consumer.

Native action, action-state, gripper boundary, sampling labels and environment
commands remain seven-dimensional. No action is converted to this observation
feature chart, and no inverse-Euler decoder is introduced. History changes in
rotation features are chord differences per physical control step, not angular
velocity or a Cartesian twist. Seconds per step are unknown unless an actual
source/controller declares them; no nominal frequency is fabricated.

Configuration, restored visual/bottom bridges and deployment ABI carry the
feature mode and feature width. Checkpoint state-normalizer width is derived
from the native source profile (7), not the model feature width (10). The exact
feature metadata is validated before inference. An old graph cannot silently
acquire this feature chart. There is no old optimizer/weight migration allow-list.

## M1d visual source-time and observable motion

The explicit `observation.source_time_mode=source_history_steps_v1` consumes
actual integer offsets from `ObservableHistory.timing.state_offsets`. For the
current synchronized data/online adapters, visual and state histories share
those source rows. No duration is inferred from RGB similarity, flow
confidence, saved-video FPS, a future target, or an ODE iteration.

`VisualSourceTime` retains the actual frame coordinates and positive pair gaps.
At reset [0,0,0] there is one image and no motion pair; [-1,-1,0] contains two
images and one one-step pair; [-8,-4,0] contains three images and two four-step
pairs. Duplicate payload and context-dropout masks are canonicalized to the
latest copy before any learned projection. This does not throw away the real
current frame when all past slots repeat it.

| Boundary | Source-time rule |
|---|---|
| Semantic/raw flow objectives and outputs | A zero-gap pair has no observed motion support; it owns no motion loss, value or gradient. Positive-gap pairs keep actual source displacement. |
| Dense temporal organizer | Read each physical frame once using a key mask and actual-time Q/K features. Current visual values remain available at reset. |
| Generic G visual read and summary | The same frame/pair support masks keys, values and the summary, before normalization/products. |
| Perceptual future queries | Motion read uses actual pair time; missing pairs cannot re-enter through future-history attention or its learned projection. Factual context and learned future priors remain separately available. |
| Local chart -> global facts -> world belief | `latest_flow_steps: Long[B]` survives materialization and object permutation; no duplication of camera axes. |
| S/W motion features | Use image-chart displacement per observed control step; raw camera transport remains displacement for geometric association. |
| Teacher extrapolation | Multiply observed displacement by future offset / actual observed gap. No pair means no observed motion prior, not a fabricated stationary target; search uncertainty still grows with the future horizon. |
| Config/ABI | New explicit source-time selection and exact metadata; old fixed-history selection remains unchanged. |

Time projections have the actual consumer's hidden width (organizer and main
carrier differ); the clock has one source owner. The unused fixed history-type
parameter is frozen in source-time mode. Current-validity authority is not
transferred to a predicted visibility score, and missing flow does not remove
current image evidence. Cached clocks/support are read-only throughout both
ODE passes and the candidate-world rebuild. There is no new recurrent task
state, extra W call, or cross-decision cache. Seconds per control step remain
unknown unless declared by the source/controller.

This unit does not resolve multi-camera coordinate pooling, G entity
association, W action/target mismatch, or P3 execution feedback. M1d's passing
boundary tests cannot be promoted to learned task success.

## M2 full-posterior candidate support

`observation.candidate_support_mode=full_posterior_lattice_v1` preserves the
complete 64-point G1 camera-chart support in the actual dynamic G2 sampler.
G1 already retained its full posterior; this unit changes its consumer, not
just the returned diagnostics. Legacy `moment_local_v1` continues to use the
49-point local lattice and remains a separately selected control.

Every support point is individually rectified by the existing G2 correction
head using the source-relative edge map. Its own coarse key and current
DINO/RGB/detail evidence are read at that point. Real historical source-query
coordinates are retained rather than inventing shifted historical pixels.
The FP32 G1 log distribution is a normalized parent measure in G2;
local content can update it without a hard top-k, entropy quota, or task label.
A center/variance remains a summary, not the unique location of evidence.

G3 exports current content from the expectation of observed DINO values over
actual candidate positions, not a sample at a potentially empty barycenter.
Its content-position measure uses the same geometry posterior. DINO channel
tiling and checkpointed recomputation retain all channels and their gradients;
they do not add an encoder, physical observation or W/ODE pass.

P1 consumes the new candidate count explicitly. Its 3x3 microgrid is still a
local patch: each microcell is sampled around each possible location BEFORE
candidate marginalization. Resizing the old 49-to-9 global weighting basis
would instead create a camera-wide thumbnail and is not used. The center cell
reuses the exact cached RGB/detail read. Out-of-image cells and unsupported
candidate payload are excluded before products; local posterior conditioning
uses logit-space softmax for finite reverse derivatives under extreme logits.
`fine_offset_zero` removes the local patch radius but does not choose a
fictitious central index of the global candidate set.

Config, restored core, observation validation and deployment metadata agree
on this selection, candidate count and current-content rule. Checkpoint ABI
rejects missing/malformed metadata and legacy aliases. No new old-checkpoint
migration is defined. This changes source semantics even though there are no
additional trainable parameters in the candidate support helpers.

This unit is NOT persistent entity tracking or a proof that local hypotheses
represent independent objects. Typed attributes and global K binding retain
their existing algorithms pending M3/M4. P1 interface adaptation is not M7's
shared-target precision review. End-to-end action gradients to G1 establish
connectivity, not learned task success. Revisit density/memory and localization
choices when later consumers expose a better representation.

## M3 coupled local observation identity

`observation.local_ownership_mode=coupled_observation_v1` requires M2's full
candidate support. G2 combines the three learned typed compatibility scores
and the normalized G1 parent measure into one FP32 soft location posterior.
Semantic, appearance and geometry values remain separately encoded but are
read using that same location law. G2 integrates its candidate law to one
local-M hypothesis measure; G3's three existing correction heads jointly
refine that measure instead of defining three unrelated M identities.

The combination is an energy-based learned compatibility model, not a claim
that its sources are calibrated independent sensor likelihoods. It adds no
hard winner, target color, entropy quota, or visibility prediction. Conflicting
local evidence may remain ambiguous: sharing a law is not hard selection and
does not turn a multimodal mean into a uniquely localized physical entity.
M2's candidate support and P1 local quadrature remain intact.

Fine and M posteriors remain FP32 under autocast. The typed fields retained by
the legacy interfaces refer to the same law; they are not separate trainable
owners. The dense global binder consequently receives the same local prior
for every attribute, while retaining sole control of real-versus-null mass.
All-invalid fine rows have zero observable mass and zero payload gradients;
an all-invalid M row retains a neutral prior so absence is routed to null
rather than destroying mixture mass. Invalid candidate values are quarantined
before projections, aggregation or the local correction.

Existing three G2 query projections and three G3 correction heads remain
active evidence producers. In connectivity tests, nondegenerate observation
features are necessary: the state-chart fixture's channel-constant DINO rows
correctly normalize to zero and cannot measure semantic-query connectivity.
Tests must not modify trained parameters or force a gain to obtain gradients.
A nonzero test gradient proves connectivity only, not useful learned influence.

Config, restored core, progressive state and deployment metadata carry this
new selection explicitly. Missing or wrong ABI fields are rejected; there is
no old-checkpoint migration. The independent-typed control remains available
for reproducibility and is checked for unchanged parameters/order and sampling.
M4 must still address entity objectness, multimodal-support reduction into
K facts, the third G block's entity-value role, and causal entity association.
The shared local law is NOT persistent tracking or S's operated-target binding.

## M4a completed current G3 context in the entity binder

`top.entity_context_mode=completed_g3_v1` requires complete current-camera
candidate support. It samples the completed third G block's public chart at
those support points using the same geometry location measure as the other
local facts. Sampling precedes expectation; a possibly empty barycenter is
never substituted for the actual support. Current learned context remains a
separate value from observed DINO and semantic/appearance/geometric properties.

`LocalFactSet.context_slots` -> `DenseFactChart.candidate_context` -> the actual
`DenseObjectGrounder` candidate projection carries this field into K competition
and iterative binding. It is not a loss-private feature, a diagnostic, or a P1
bypass. Producer-owned candidate validity quarantines missing payload before
learned projections. The independently observed current DINO chart remains the
reconstruction target; learned context does not declare observation support or
become its own reconstruction truth. Teacher may see updated current assignments
but does not gain access to online future information or learned target authority.

The explicit new context projection is created only for this mode; the
candidate-only control has no placeholder parameters and preserves constructor
RNG/order. Missing context in the new mode, or undeclared context in the old
mode, is an error rather than an ignored sidecar or silent fallback. Config and
deployment ABI retain the semantic selection; new training is required. No old
checkpoint migration is introduced. Sampling uses the existing channel-tiled
checkpointed expectation, including when only the learned chart needs gradients.

The current G chart and entity context are computed once per observation and
read unchanged through candidate-world rebuilds and numerical solver nodes.
This unit adds neither entity persistence nor an ODE-time state update. Its
entity-only backward test measures the third G block without relying on P1 or
action-decoder shortcut paths. Connectivity and source tests do not prove that
K slots now correspond to correct physical entities. Objectness, multimodal K
support, cross-camera association and causal entity persistence remain M4 work.

## M4b current-image entity support

`top.entity_chart_mode=current_image_support_v1` requires completed G3 context,
full candidate support and the coupled local observation law. The M4a query-grid
mode remains a separately serialized control; no old checkpoint is migrated.

`CurrentImageSupport` carries actual `[B,C,Yq,Xq,M,N,2]` current-camera points,
their FP32 conditional probability AND producer log probability, and Boolean
source support. It is materialized from G2's existing complete law, not from a
mean/variance Gaussian. Valid local hypotheses must have matching source support.
`DenseFactChart` retains this reference without resampling or re-encoding.

The binder's assignment tensors remain indexed by query/local hypotheses.
Their pushforward is instead indexed by CURRENT image pixels. A bilinear
align-corners write transports every candidate's mass, separately for K and
camera. No top-k, predicted visibility mask or fabricated observation is used.
`ImageLogMeasure` is the authoritative spatial measure. The public
`object_to_chart` is its normalized probability view; camera centers come from
conditional reads of the same log measure, not a second coordinate estimator.
K permutation carries the log measure as well as all existing entity fields.

Segmented log-sum-exp is necessary: normalizing a tiny positive image mass in
ordinary probability space can produce NaN reverse derivatives even when the
forward output is finite. Source logs are preserved BEFORE exponentiation;
segment maxima are detached numerical shifts. Empty support yields finite zero
values/gradients. Underflow is not silently reclassified as object absence.

Reconstruction uses the conditional K mixture at each destination pixel, not
source-query density. Its shared positional feature uses the destination pixel's
coordinate. The fixed observed target/mask still owns the entire objective,
including uncovered observed pixels; predicted coverage cannot erase a label.
Observed DINO content remains the reconstruction/Teacher target. Teacher's typed
current keys still integrate observed local content with query-indexed assignment;
its camera prior and geometry now consume the current-image measure/centers.

This unit adds no physical-time state, extra G/W call or new task condition. The
support is created once per observation and is read-only throughout ODE and W
refinement. It does not resolve global physical objectness, persistent entity
association, cross-camera physical calibration, or later S/W/P responsibilities.
CUDA scatter determinism, production memory/latency and learned task success
remain hardware/behavior verification, not consequences of CPU tests.

## M4c1 observed temporal evidence for current entity grouping

`top.entity_history_mode=flow_pulled_history_v1` requires source-timed visual
history and M4b current-image candidate support. History/current observed content,
cell masks, chart axes, device and latest source gap must agree; a stale or
foreign history packet is rejected, not adapted silently. Current-only M4b remains a
separately selected control, not an implicit fallback.

The observed-history packet contains only the already causal DINO frames,
the existing two directed flow estimates, actual frame offsets and the exact
context-dropout/source masks. It reuses the current-observation content grid's
full-width normalization; it does not call a future-target API. Backward flow
is indexed on the later SOURCE image and converts from chart-cell displacement
to normalized xy once. Earlier coordinates compose the two adjacent maps at
the appropriate intermediate location; multiplying the latest flow by age is
not equivalent and is not used. Current-observation aliases at reset are counted
once. Leaving the image cannot manufacture an observation by reentering later.

Masked values are removed before projection and interpolation. Fractional
coverage is retained, not divided out. Confidence/occlusion are inferred path
features, NEVER observational validity, entity existence or a loss mask. The
history relation contains per-camera feature and image displacement changes
per actual control step, and separate coverage/status/age features. These are
visual correspondences, not world velocity, contact truth or task progress.
There is no cross-camera raw-coordinate average in this new path.

Learned temporal relations are formed on the CURRENT image chart before the
same full-candidate quadrature used by G3. They enter the real K competition
key AND iterative slot update, so this is not an unused diagnostic sidecar.
All downstream facts and their S/coarse/W/P consumers use the resulting single
K allocation. The independent observed current DINO target, cell mask and
real/null ownership law are unchanged. Dense history stays on the ephemeral
observation plane, not in the compact candidate-W/ODE cache.

Train and deployment recompute the identical finite causal window. No hidden
cross-observation state, task counter, extra G/W solve, deployment-only recurrent
cache or ODE-time memory update is introduced. This deliberately bounded first
association unit supplies temporal evidence for grouping, but does NOT claim
persistent K IDs, long-occlusion re-identification, calibrated multiview physical
identity, complete segmentation, or learned behavioral success. M4c2 still owns
those remaining global/persistent-identity design questions; M5-M11 are not
completed by this upstream interface change.

The explicit config and deployment ABI reject missing/unknown/drifting history
semantics. New parameters are instantiated only in the selected mode and are
owned by the real grounder. Current-only controls have no placeholder parameters
or changed constructor RNG/order. No old-checkpoint or optimizer migration.

## Agent quick contract

~~~text
capability:             object_intent_dynamics_323
default manifest:       schema 30
optional B-spine:       schema 31 only when explicitly selected
registered layout:      clearvla_mainline layout 2
component selection:    mainline-modular-v1
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
future intervals:       4-8 / 8-16 / 16-32 / 32-48
global objects:         K=4 plus explicit null mass
visual history:         DINO/raw at -8 / -4 / 0; two learned adjacent flows
training:               one online encode, one formal velocity pass, one loss composition
deployment:             proposal ODE, one W rebuild, refined ODE
shared action core:     24 x 18 value/adjacent-difference/gripper field
W action condition:     four interval means by default; opt-in causal 24-row prefix
default config:         configs/mainline/object_intent_dynamics_323.json
Pen launcher:           scripts/train_mainline.sh
RDT-8 launcher:         scripts/train_rdt_multitask.sh
checkpoint validation: scripts/validate_mainline_checkpoint.sh (read-only)
working branch label:   codex/schema29-mainline (historical name, not semantics)
~~~

The default is the Schema28-core recovery behavior under the schema-30 ABI,
with a profile-owned continuous-gripper boundary and outlet-scoped binary
adaptations. The shared action codec, G/S/W/P graph, training call count and
two-pass deployment lifecycle are common across outlets. CALVIN and the
repaired ManiSkill v2 base change only their outlet sampler/chart and terminal
binary-command behavior; the ManiSkill choice is explicit and does not change
Pen/RDT/LIBERO defaults. The opt-in B-spine is an experiment selection, not a
new default.

## Default and opt-in selections

These are source/config selections, not claims about trained-model success.

| Boundary | Default | Explicit alternative |
|---|---|---|
| Future-label support | `strict_complete_v1` / existing explicit boundary controls | `observed_tail_v1`; all labelled real current states, independent observed action/state/visual support, no synthetic future truth |
| S/history encoding | `paired_rows_v1` historical control | `timestamped_streams_v1`; separate causal state/control streams, source times and padding provenance |
| P2 spatial intent | `post_pool_only`; typed S selects interval after spatial pooling | `shared_target_prior_v1`; one shared S-owned target-K prior conditions semantic K and geometry K*C before pooling |
| Visual NPY read transport | `mmap` | `pread`; cache layout, logical indices and dataset identity are unchanged |
| W camera condition | `motion_prior_only` | `coordinate_role_v1`; separate initialization migration |
| W action condition | `interval_mean_v1`; four physical interval means and current-anchored deltas | `sequence_prefix_v1`; exact known 24-row action sequence with causal prefix reads |
| ODE schedule | Uniform E5 proposal/refined | Registered Q5 plus its versioned flow-step context |

An opt-in's source tests do not promote it to the default. The checkpoint's
serialized selection remains authoritative for evaluation and resume.

## Implemented M1a history contract

An observation `o[t]` precedes execution of `a[t]`. Times below are integer
**physical control steps**, not seconds, wall-clock inference latency, task
progress, action-horizon indices, or flow/ODE time. State samples retain
`[-8,-4,0]`; already executed command samples retain
`[-24,-16,-12,-8,-6,-4,-2,-1]`. They are never paired by array row number.

`clearvla.data.history_clock.sparse_history_clock` is the common source-index
rule for the training dataset and `CausalHistory`. It exports actual state
source offsets, requested command offsets, `state_observed`, and
`action_executed`. A repeated reset state is padding, not an additional
observation; an absent command is not a recorded zero command. Absolute episode
position is used only to construct this provenance and is not a model feature.

The explicit new mode requires `ObservableHistory.timing: HistoryTiming`.
Missing metadata is an error, not a guessed legacy clock. Value/time admission
is strict at the input boundary; shape/device checks inside the ODE do not
perform tensor-value reductions. The loader and online adapter both use this
same record. The native `action_state`/gripper boundary remains a separate
outlet-owned observation and is not relabeled as an executed command.

`TimedHistoryEncoder` maps state and action events separately, includes their
actual relative times and event types, sorts them chronologically, and uses
causal attention. At equal time, state comes before command. The current state
appears once and owns the last state row. Its state-change feature is a
normalized-coordinate difference divided by the actual positive control-step
gap, admitted only when both observations exist. This is **not** a claim of
metric velocity or a corrected rotational chart; rotation/unit auditing
remains in M1b. The mixed stream currently has 3 state + 8 command slots, not
8 pseudo-paired rows. Masked padding has zero value and cannot be an attention
key. S's interval history read, observed-state-change read and coarse history
read consume the corresponding masks. No-observed-interval state change is
exact zero, including at reset.

Conditioning quarantines padded numeric values before all consumers. The
auxiliary history proposal also uses actual command offsets, gap-aware command
changes, masked summary reads and exact-zero absent memory. Its future query
may still express a learned prior; that is not presented as executed evidence.
Action-history condition dropout changes the execution mask together with the
values and disables the executed-memory path; it never removes real state
observation provenance. Existing compact downstream seed/state consumers keep
their prior topology for now; a clock-aware source does **not** mean M9's
complete bottom/history information-retention review is closed.

`CausalHistory` retains the latest 9 observations and 24 commands, plus one
fixed reset observation/action boundary; the separate absolute step counter
survives eviction. Snapshots copy values, and attempts to read future or evicted
non-reset entries fail. The physical clock and masks are read-only throughout
proposal/refined ODE evaluation and are rebuilt on a new physical observation.

New component selections are `timestamped_observable_history_v1` and
`timestamped_object_intent_v1`; the deployment observation ABI additionally
requires `physical_step_sparse_history_v1`. Old default config serialization
omits the legacy selection to retain its prior identity. New mode is an
explicit new-training candidate, **not** an exact-resume or silent partial-load
migration from old weights. Ordinary exact restoration checks the new config,
component selection and source closure. New history modules belong to that
closure. Preserved legacy initializer/state-dict arithmetic does not waive the
old source-identity check across changed source revisions.

## Implemented M1b real-label support contract

`observed_tail_v1` is an explicit **data/supervision** change, not a new online
condition. It requires `timestamped_streams_v1`, stride one, and a verified
CALVIN/LIBERO terminal observation. A converted episode's original labelled
start is retained. If `L` is its last real observation, every labelled current
center through `L-1` is admitted; `L` itself has no recorded next command.
The old complete-window upper bound is not misread as the last real label.
This operates on the existing admitted source inventory; previously excluded
annotations are not claimed to have been recovered. Original data files,
splits, decoded/DINO caches, language bank and normalizer fitting are unchanged.

`clearvla.data.future_clock.future_source_rows` owns the pre-action clock:
commands `a[t+h]` are known for `t+h<L`; successor states `o[t+h+1]` are known
through `L`; sparse visual targets at their declared offsets are known through
`L`. Safe gather indices may repeat a real terminal observation for transport,
but the repeats have false label masks. Stored absorbing suffixes never become
new observed actions, states or images. Missing values are quarantined before
normalization-sensitive target operations, projections and losses.

The loader creates one `FutureLabelSupport` shared by `ActionSupervision` and
`FutureSupervision`. Its boolean action/state/visual masks are immutable by
convention and strictly admitted at the source boundary; they are not learned
confidence or current object validity. No such record exists in
`OnlinePolicyInput`, the candidate W world or the ODE cache. A missing new-mode
mask or a supplied mask in a legacy-mode batch is rejected. Preflight verifies
that policy and world labels agree on their shared real command rows.

The mask is consumed by the actual Teacher association, fixed-interval targets,
future dynamics/transition losses, recognizer attention/reconstruction, online
S target match, coarse and history-proposal auxiliary losses, flow-matching
bridge, action/decoded/gripper/motion losses, execution-value targets and
validation error counts. A W interval is supervised only if ALL supports of
that fixed interval are observed; a shorter partial interval is not relabeled
as the full interval. Unavailable interval tokens cannot become recognizer
attention keys. A sample may therefore own a valid next action while owning no
fixed-interval W or S future target.

Unknown rows in the flow bridge remain the sampled source noise. They are not
trained toward zero, a frozen gripper, a synthetic stop, or an invented expert
future. Real-label error reductions count only observed entries while retaining
the existing declared horizon weighting. The historical full-24-row native
tangent-projection diagnostic is reported only on fully labelled samples,
with its contributing sample count, because that operator mixes the horizon.
Model-only output-delta diagnostics do not need future labels and keep their
original full-prediction domain.

Train, validation and test datasets all use the selected real-label contract.
Validation records actual observed action rows and per-error element counts;
an empty band is not evidence of a perfect prediction. Its score is not a
strict-window/E8-equivalent score. Window-sampling motion/event signals use
real source commands only; no new fixed boundary quotas were introduced.
Current-state coverage counts are distinct from the union of future target
rows. Source normalizers still exclude synthetic absorbing rows.

The mask record adds no policy parameter; it still changes config/data/source
identity and is not an exact resume from M1a. New helpers belong to the source
closure; checkpoint tests exercise ordinary exact save/reload and reject
silent old-config reuse. No migration allow-list is expanded. The supplied
candidate is CALVIN. The continuous outlet loss path is covered with a LIBERO
profile fixture, but the legacy LIBERO conversion/config-generation CLI is not
newly advertised as an observed-tail workflow.

This closes only M1b's **real-tail/label-support unit**. Physical rotation charts,
coordinate/normalizer units, nominal step-versus-seconds metadata, downstream
compact seed missing-evidence semantics and reset-distribution alignment remain
open before declaring M1 complete. It does not resolve the W candidate-action
versus demonstration-future mismatch, Q5 endpoint-head training, entity/task
memory, or G/S-task/P architecture. Those keep their scheduled reviews.

## Authority and document ownership

When records disagree, use:

1. active source plus the artifact's serialized run context;
2. this contract;
3. the current issue ledger and repair plan;
4. the bounded current adapter/design document named by this contract;
5. archived documents and Git history.

Branch names, run-directory names, old launchers and conversation labels are
descriptive only. They cannot override source, config, manifest, dataset or
checkpoint identity.

Maintain this file as architecture, not chronology. Replace decisions in
place. Keep only accepted semantics, invariants, typed boundaries and stable
source locations. Metrics and unresolved causal questions belong in the issue
ledger; PID, host, path and next-command state belong in the handoff; completed
proof belongs in Git or the archive. Never copy checkpoints, caches, raw logs
or full probe dumps into repository memory.

## Active graph

### Observation and grounding

~~~text
RGB/DINO at -8,-4,0
  -> V120 raw/DINO compiler + two learned adjacent flows
  -> current-only G1/G2/G3 progressive grounding
     G2 rematerializes N=49 fine candidates once
  -> camera x 8x8 x local-M hypotheses
  -> dense global K=4 plus null grounder
  -> ObjectFactSet and reversible K-to-chart correspondence
~~~

G cannot read language, executed-history proposal, noisy action or future
Teacher evidence. Local-M rows are hypotheses, not persistent objects. The
dense grounder owns real/null mass; G3 may refine only the distribution over
real K objects. Reconstruction uses detached current DINO on observed cells
and exports one K-specific content value shared by reconstruction, S, W and
the detached Teacher.

### Intent, action condition and world

~~~text
T5 + observed state/history + ObjectFactSet
  -> S public interval carrier + typed [interval,K,type] relevance
  -> typed-free CoarseAction proposal [B,4,7]
  -> outlet-owned physical action condition
     default: PhysicalActionCondition [B,4,14]
              absolute interval means + current-anchored adjacent deltas
     opt-in:  PhysicalActionSequenceCondition [B,24,7]
              source rows + canonical value/delta + row-time/chart identity
  -> W(ObjectWorldBelief, selected physical action condition)
     W1 owns intervals 0/1; W2 reads W1 and owns intervals 2/3
  -> action-tagged CandidateWorld / FutureObjectDynamics
~~~

S is the sole intent owner. W cannot read goal tokens, S values, coarse hidden
tokens, Teacher or noisy ODE action. Semantic successors retain
[B,4,K,D]; transport, covariance and camera support retain
[B,4,K,C,*] until P2. W predicts no validity, visibility or status authority.

The language/object repair is part of this existing S path, not a CALVIN
sidecar or a second object container. S keeps two explicit K-preserving views:
the content-only interval evidence remains the owner of the established typed
relevance decomposition, while a bounded symmetric
semantic/appearance/geometry-enriched public K memory supplies the coarse
object read. Both views are masked by producer-owned validity. Goal/history
innovations condition the interval object query, and that same conditioned
query is carried into the coarse object's K read and history read. W still
receives intent only through the outlet-owned physical action condition. The
default P2 path reads typed S only after spatial pooling and uses it for
interval selection.

The opt-in `shared_target_prior_v1` additionally exports one S-owned FP32
`[B,I,K]` address before P2 spatial pooling. A bias-free exact-zero `[1,3]`
owner combines the typed semantic, appearance and geometry evidence without
consuming constructor RNG. Supported K are reference- and mean-centered in
FP32, so uniform evidence is exactly neutral under autocast. The same K prior
is added to semantic K and geometry K*C logits, broadcasting only over C; it
cannot create support, rewrite W values, flatten camera identity or introduce
independent target priors for the two readers. Their final K posteriors may
still differ because their W source and coordinate likelihoods remain
independent. Supported NaN/Inf is rejected and unsupported K is quarantined.
No raw language, color label or rank-1 object pointer enters W or the bottom.

The opt-in `sequence_prefix_v1` preserves the known 24-step action proposal
instead of pooling it into four means. Coarse action keeps its established four
queries, interpolates them to 24 rows and adds one exact-zero row-offset owner.
The outlet adapter is the sole factory for both online/coarse and deployment
rebuild conditions. It records normalized source action, canonical value and
delta, current boundary, control-row times, FP32 normalizer metadata/fingerprint
and outlet/chart identity. W reuses its physical row projection, adds a learned
row-time projection and one recurrent action encoder, then reads causal prefix
endpoints 8, 16, 24 and 24. The fourth world interval is therefore conditioned
on the same known 24-step prefix; rows 25--48 are not fabricated. This mode
does not introduce language, S hidden state, Teacher evidence or noisy ODE
action into W.

### Precision, policy, transition and execution

~~~text
completed G chart
  -> one cached P1 high-resolution read
     24 factual queries, N=49, real 3x3 RGB/detail microgrid
  -> FactualPrecisionDock

noisy action + time + factual detail
  -> dynamic P1 residual
  -> P2 semantic-K and geometry-K*C selection per interval
  -> no-null physical interval terminal per type
  -> semantic + geometry consequence

P2 consequence + S/action context
  -> optional temporal P3
observed state change + S/action context
  -> optional state-change P3

completed G3 rollout
  -> cached ControlledTransitionSource
noisy action + consequence + dynamic P1
  -> ControlledTransitionState
all protected/optional carriers
  -> V120 execution bottom
  -> physical velocity, motion and optional outlet command
~~~

Protected consequence and raw dynamic-P1 precision are no-null carriers. Only
the temporal and state-change P3 lanes own zero-null choices. Evidence MMDiT,
CVAE/workspace, continuous capacity and execution-value machinery remain part
of the bottom.

## Training, deployment and runtime

Future DINO, action and state are available only to the no-grad Teacher and
named training targets. Teacher builds once per training batch and never in
deployment.

The default training path is:

~~~text
encode online observation once
  -> build detached Teacher/targets
  -> sample one FlowMatchingState
  -> one formal velocity forward
  -> one loss composition
~~~

There is no train-time endpoint estimate or train-time W rebuild.

Visual cache transport is independent of model semantics. The default remains
`mmap`; opt-in `pread` reads exact ordered NPY rows with bounded process-local
handles while preserving cache admission, row repetition/order, camera order,
normalizers, sampling and dataset identity.

Deployment and validation perform exactly two complete five-update ODE passes
from identical initial physical noise:

~~~text
W(coarse)
  -> proposal ODE at t = 0,.2,.4,.6,.8
  -> decode 24-row proposal
  -> rebuild W once from the selected physical action condition
  -> refined ODE at t = 0,.2,.4,.6,.8
  -> final action
~~~

This is one bounded correction, not a fixed point. The final action may differ
from the action that conditioned the rebuilt W, so final interval/delta
mismatch in the default mode, or sequence/delta mismatch in the sequence mode,
remains a residual metric. Recomputing W after the final action without another
policy consumer would not close anything. Training still performs one online
encode and one formal velocity/loss path; future action supervises coarse action
only and never causes a train-time endpoint estimate or W rebuild.

The default remains the exact uniform E5 grid above. The registered Q5
schedule (`t_i=(i/5)^1.25`) is opt-in. A non-uniform schedule now carries the
versioned bottom-context contract
`FlowStepContext=(t,Δt,k/N,endpoint)` through the sampler and the matching
training bridge, so the learned bottom field can account for the interval it
is asked to integrate. The schedule schema/fingerprint records that context
contract; an old sampler-only schedule artifact is rejected rather than being
treated as an exact-resume alias. The bottom still predicts an instantaneous
physical velocity and the integrator applies `Δt` exactly once. The 24-row
action-time chart, B-spline basis and W rebuild remain separate contracts.

Observation/G/S, static P1 and ControlledTransitionSource build once per
observation. Dynamic P1, P2/P3, transition and the bottom run at every ODE
node. Only the bottom receives the numerical `FlowStepContext`; it is not a
semantic or outlet/task condition. The t=1 read supplies retained endpoint
motion/command state but does not update action. Execution candidates remain
mandatory for train/eval loss forwards and disabled during ordinary deployment
passes.

An explicit deployment-only fastpath may reuse the exact block-major dwell
prefixes and the current decision's already-computed neutral dwell-one result.
It retains the complete candidate chart, values, probabilities, diagnostics,
terminal baseline and action ABI; it is disabled by default, forbidden in
training, and retains no reuse state across a decision or ODE node.

The deployment ABI owns DINO preprocessing, dtype and reference batch shape.
When the serialized profile requires a reference batch, causal padding may
reproduce that encoder shape before slicing back to the real three history
rows.

## Action and outlet contract

The shared canonical action is seven-dimensional and its physical flow field
is [B,24,18]: six arm values, six arm adjacent differences and six continuous
gripper coordinates. The two arm branches and cumulative decode remain common
to every outlet.

| Outlet | Native boundary | Mainline claim |
|---|---|---|
| Pen | Seven-dimensional continuous action; current action-state owns gripper anchoring | Core-behavior outlet |
| RDT-8 | Right-arm seven-dimensional projection, high plus right-wrist RGB; previous executed command owns gripper anchoring | Adapter and cross-task outlet, not native three-camera/depth/14-D bimanual support |
| CALVIN | Seven-dimensional relative TCP command; current action-state plus explicit binary gripper command | Outlet-specific command semantics, not a replacement shared codec |
| LIBERO | Seven-dimensional normalized OSC_POSE relative command plus continuous gripper; previous clipped command owns the gripper boundary | Isolated external comparison; its ingress/egress adapter does not alter the shared codec or network core |
| ManiSkill v1/v2 continuous | Seven-dimensional normalized `pd_ee_delta_pose` relative command; previous clipped command owns the gripper boundary | Retained compatibility baseline |
| ManiSkill v2 binary | Same seven-dimensional relative arm command; private native gripper command owns dimension 7 and emits strict `{-1,+1}` | New StackCube base outlet; does not modify the shared 18-D codec |

For every continuous gripper profile, one causal boundary owns row-zero delta,
the full-horizon anchor, cumulative decode, loss and evaluation. Event masks
select rows only; decoded events are evaluation metrics, never runtime gates.

CALVIN applies exactly three outlet rules:

1. motion sampling scores normalized command minus normalized raw zero;
2. before W, the default mode converts four centered relative-command means to
   cumulative canonical displacement plus per-interval command delta; the
   opt-in sequence mode instead preserves all 24 centered commands, their
   cumulative canonical values and rowwise deltas;
3. the binary command head maps argmax to {-1,+1}, while the six
   compatibility-only future-gripper coordinates are zero before every dynamic
   ODE consumer.

Its dataset boundary is independently fail-closed. Split membership is by
official `ep_start_end_ids.npy` source trajectory, never by language annotation
row or sampled window. Every real action through the annotation's terminal
state minus one must occur in a 24-row policy target. The extra support needed
by the 48-frame Teacher follows CALVIN's absorbing convention: repeat terminal
RGB/state, zero relative arm commands and hold the final binary gripper command.
Synthetic absorbing rows are excluded from normalizer fitting. These are
outlet data-interface rules and do not alter the shared policy graph. An
annotation is admitted only when its source trajectory also provides the full
24-frame causal history; short-prefix annotations are rejected before HDF5
materialization rather than failing after a partial output is written.

Pen and RDT bypass the CALVIN W adapter numerically. Task identity may control
sampling, validation and logging but is not a hidden S/W/P condition.

LIBERO has its own fail-closed ingress/egress contract.  Conversion derives the
policy instruction from the official filename-based `Task.language` surface and
records a BDDL `:language` value, when present, only as simulator provenance.
The official BDDL task inventory is mandatory conversion input; HDF5 language
metadata alone cannot assert the evaluator instruction identity.
The HDF5 action/state arrays are required to be native seven-dimensional
`OSC_POSE`/EEF charts with a reset-zero then previous-action `action_state`.
The upstream LIBERO `create_dataset.py` records `obs[t]` after executing
`action[t]`; that post-action row must never be presented as the observation
that generated the same action.  The causal boundary converter therefore
renders row zero from raw `states[0]`, shifts released post-action rows right by
one, and treats the released final post-action row as the genuine terminal
observation.  Prefix and terminal-suffix roots retain a numeric-only copy of
the legacy real state rows solely to keep the E8 state-normalizer fingerprint
identical; that reference is not model evidence.
The controlled LIBERO continuation pair is explicit: `causal_prefix_v1`
admits centers `0..T-49`, while
`causal_prefix_terminal_suffix_v2` admits `0..T-1` after appending exactly 48
absorbing rows (zero arm, held final gripper, repeated terminal state/RGB).
Both reserve `1 prefix + 7 strict` or `1 prefix + 1 tail + 6 strict` in B8,
keep strict validation as the checkpoint-selection metric, and allow the
strict-to-causal transition only through the tested model-only migration
`libero_window_boundary_supervision_v1`.
An optional retarget training overlay may augment only the terminal-suffix
train split.  Each admitted source demonstration owns one symmetric nonzero
XY translation pair generated by simulator replay: the object initial free
joint and the expert EEF trajectory change together, while the original
receptacle remains the release target.  Manifest lineage, HDF5 content hashes,
success, tracking/saturation/placement bounds and paired response direction
all fail closed before loading.  The immutable base train split alone fits the
action/state normalizers; overlay rows are appended only after that fit, and
base validation/test membership and strict checkpoint-selection metrics remain
unchanged.  This changes dataset/cache identity but not the model, objective,
codec or outlet ABI.  A terminal-suffix checkpoint may initialize such a run
only through the tested model-only migration
`libero_retarget_training_overlay_v1`, which requires byte-identical action
and state normalizer SHA-256 values.  Its source allow-list composes the
ordinary model-only initialization/replay allow-list with the data-boundary
files needed by the overlay; selecting the data migration must not revoke an
already-admitted validation-only source change, and it still rejects drift in
every source file outside that union.
Conversion writes into a staging root, audits the complete inventory, and
atomically publishes only an audited root.  XML model assets are relocated
from the released `chiliocosm` absolute prefix to the installed LIBERO asset
root with an existence check; the relocation is serialized as provenance.
Terminal replay starts at raw `states[0]` and executes the complete recorded
action sequence, because the OSC controller target is hidden outside the
flattened MuJoCo state.  The raw writer's final reward/done entries are
official terminal markers (the upstream writer sets them on the final row),
so replay reward/done/success is retained as an audit comparison and never
replaces or silently filters the raw label; mismatches are reported per
episode.  Conversion still fails closed on malformed arrays, missing assets,
state-restore failure, or a replay process exception.  Evaluation applies the
fixed init
state, performs the official five zero-action physics warmup without reading
success/done, then asks the policy for a 24-row chunk and executes its first
row after normalized `[-1,1]` clipping.  The clipped row is the next
`action_state`; success is evaluator-only.  These are LIBERO outlet rules and
do not add a second network forward path or change the shared action codec.
The evaluator resume identity also includes the action/state normalizer
digests, so a relocated bridge is allowed but a silently re-charted bridge is
not.  Fixed init states retain their official first-N identity, and every
resumed rollout must carry a complete executed-action audit.  `libero_100` is
an archive union rather than an evaluator suite, so 90 and 10 are evaluated
separately.  Only a checkpoint-backed, all-task 20-episode/600-action run with
the five-step warmup, seed 10000 and 128-pixel camera input is labeled as using
the official protocol; transport smokes never acquire that label.  Result
identity includes `image_side` plus content digests of each used first-N fixed
state matrix and BDDL file.  Bridge endpoint is relocatable provenance, not
policy or benchmark identity.  Partial per-task checkpoints remain explicitly
incomplete until the final bridge-identity check succeeds.

Physical-chart units and nominal limits are metadata. They do not clip,
normalize, decode, condition the model or silently define a full scale.
Observed normalizer extrema and diagnostic probe constants are not physical
unit declarations.

RDT data and adapter detail lives in
[auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md](auxiliary/RDT_FT_DATA_MULTIVIEW_BIMANUAL_ADAPTATION.md).
Simulation and benchmark operational detail lives in
[../development/SIMULATION.md](../development/SIMULATION.md).

## Optional simulator residual-RL pilot

The opt-in `clearvla.rl` pilot freezes a verified same-task ClearVLA/DINO base
and learns an external bounded unit-residual SAC actor with independent twin
Q critics. It runs after original outlet finalization and native decoding,
before environment clipping/execution; the actual executed command owns the
next history. Reward/success stay outside deployed policy inputs. The mainline
18-D codec, G/S/W/P, one-W-rebuild two-pass sampler, Teacher and supervised
trainer are unchanged; execution-value heads are not RL critics.

New base preparation selects `maniskill_pd_ee_delta_pose_7d_v2` with the
`maniskill_binary_command` / `maniskill_7d_binary_v2` outlet. The legacy v1
profile remains readable for existing runs, but is not admitted as a binary
base by the repaired RL runner. Passing `--gripper-output-mode continuous`
remains an explicit compatibility baseline; it does not silently convert an
existing checkpoint. Both are explicit StackCube/panda_wristcam boundaries,
not Pen/CALVIN/LIBERO aliases. Relative-command W conversion, previous-command
gripper anchoring, the shared 18-D codec and the neural graph remain unchanged.

The ManiSkill binary outlet follows the Calvin isolation contract without
copying Calvin's rejected direct-arm chart: a private two-class head owns the
native gripper command, all six compatibility future-gripper field coordinates
are zeroed before every dynamic consumer, and finalization plus W rebuild write
the strict native command (`+1` open, `-1` close). Binary preparation admits
only exact `{-1,+1}` values in both `action` and `action_state`; it therefore
requires a fresh data/config/checkpoint identity rather than an exact resume of
a continuous v2 run.

V2 records a causal rotation vector of the fixed chart `Rx(pi)^-1 R_tcp`,
choosing the nearest equivalent log branch from the last observed rotation
and clearing that state on reset. This is an invertible outlet-side chart
with the same 7D state width, not a globally nonsingular SO(3) parameterization.
Training and deployment use exactly the same producer implementation.
Every original source action owns a center from 0 through source_length-1;
at least 48 real post-success hold steps provide future Teacher support.
Only pre-reset history may pad: repeat the reset RGB/state and use the safe
reset action_state for unexecuted command slots, never the first future action.
All future supports remain actual recorded steps. Episode attributes and
admission checks enforce first-executed-row coverage, not merely presence
somewhere in a 24-row target. New data/caches/normalizers/checkpoints have new
identities; no live v1 directory is recharted or exact-resumed into v2.
Environment and reset-seed provenance remain serialized at the outlet.
During fixed source-plus-hold collection only, a successful task terminal may
be continued with live physics; the native terminal is retained in evaluator
metadata. Failure/time-limit termination and ordinary online-RL termination
are never suppressed by this data-collection rule.

Adapter checkpoints have separate base/source/environment/replay identities.
This is an implemented experimental interface, not a validated RL improvement
or a replacement mainline. Preparation, limitations and gates live in
[`../../clearvla/rl/README.md`](../../clearvla/rl/README.md).

The single-task information sampler carries fractional event quotas across
batches (B4/default alternates 0/1 dedicated event rows); integral B8 quotas
retain their historical sample sequence. Logging separates configured fractions
from realized dedicated quota. ManiSkill open/close names follow positive/negative
native gripper changes respectively; this naming correction does not alter total
event F1, RMSE, or losses. These changes require a fresh source identity.

## Optional B-spine contract

The default execution-bottom selection has B-spine disabled. Opt-in selections
use dedicated Pen configs and distinct schema-31 component identities. Each
reads the same deployed noisy physical field after the unchanged codec and
adds a parallel fixed coarse/detail numerical view beside the raw lift:

~~~text
u_raw   = existing noisy lift of x_t
u_spine = fixed-analysis/coarse-detail lift of x_t
action += (u_raw + u_spine) * action_state_factor
~~~

The retained `fixed_bspline_coarse_detail_v1` comparison applies the shared
cubic K=12 temporal split to all 18 physical fields. The preferred isolated
candidate `fixed_bspline_arm_only_v1` applies it only to the arm-absolute and
arm-delta groups (`11000` in the serialized action-group order); all six
gripper fields reach the bottom only through the unchanged raw lift. The two
implementations have distinct manifest bottom identities and component
selections even though both remain in the schema-31 experiment family.

The follow-up candidate `fixed_bspline_arm_coarse_context_v1` keeps the same
arm-only fixed chart and `11000` mask but owns only the learned coarse
projection `P(X_arm)W_c`; it has no learned detail lift or gripper input. The
raw action lift remains the complete row-local path, so this candidate tests
only the non-local arm context supplied by the spline. It is a fresh Schema31
identity and is not a migration or a new default.

The matched routing control `fixed_bspline_arm_private_reader_v1` uses that
same coarse arm chart and mask, but keeps the chart out of the shared MMDiT
action seed:

~~~text
raw physical field -> existing raw lift -> shared MMDiT -> terminal heads
coarse arm chart  -> ArmPrivateBSpineReader -> arm-only physical correction
~~~

`ArmPrivateBSpineReader` is a non-affine LayerNorm followed by a bias-free
projection to the first `2*arm_dim` physical channels. The coarse lifts remain
zero-initialized (and reader construction restores the host RNG), so model
construction and the initial output are baseline-preserving while the arm loss
can open the new route. The correction is applied by
`TerminalActionController` to every candidate and final physical read; it is
never read by the shared MMDiT blocks, gripper state/command head, motion head,
or gripper physical channels. This is an explicit arm-private comparison, not
a hidden second codec or an arm/gripper stop-gradient claim.

No B-spine experiment may change the codec, target, source noise, flow
objective, loss weights, ODE nodes, W rebuild count or output ABI. The raw
lift remains active. Component choice, action-group mask, basis digest and spec
fingerprint are serialized in run context. Acceptance requires matched
behavior evidence; schema 31 does not promote any implementation to the
default. For the private-reader control, report the spline-owner and
private-reader gradient probes separately so a shared optimizer-owner total
cannot hide an arm/gripper routing difference.

The canonical basis mathematics and standalone representation boundary live in
[../../clearvla/action_representations/bspline/README.md](../../clearvla/action_representations/bspline/README.md).
Current checkpoint observations, when needed, live in the temporary
[auxiliary/SCHEMA31_BSPINE_EARLY_AUDIT.md](auxiliary/SCHEMA31_BSPINE_EARLY_AUDIT.md).

## Non-negotiable invariants

1. Camera, spatial, local-M, global-K, N=49, interval, horizon, basis and type
   axes stay real until a named consumer. Reduced evidence cannot be recreated
   with expand and called original evidence.
2. Online evidence keeps ordinary autograd unless this contract names a
   no-grad Teacher or audit scope. Activation gradients do not substitute for
   parameter-owner VJP.
3. Learned flow is a continuous source-relative prior, never a forced-nonzero
   target or route quota.
4. S owns intent. It cannot manufacture W support/value or enter W through a
   hidden second path.
5. W is the only future-world producer. ControlledTransition consumes policy
   transition evidence and cannot create another world.
6. Physical validity and camera support are producer-owned. Confidence or
   allocation share cannot silently replace them. Every S/coarse/W K read
   receives that support; invalid rows are quarantined before normalization
   and object-axis reduction, and an all-invalid row has a finite zero-output
   fallback.
7. P1 retains N=49 and its real 3x3 detail read until factual selection is
   complete.
8. Semantic K and geometry K*C selection are independent and complementary.
   They do not compete in one type softmax, and physical interval selection has
   no learned null.
9. Neutral P2 is algebraically neutral: zero effect and interaction leave the
   protected factual consequence unchanged.
10. The V120 seed, terminal contracts, CVAE/workspace, Evidence MMDiT,
    capacity and execution paths remain present.
11. The shared 18-D value/adjacent-difference action field and cumulative
    decode cannot be replaced for one outlet. Outlet-native conversions stay
    at the adapter/finalizer boundary.
12. Future observation/action/state can affect detached Teacher or targets
    only; changing it cannot change deployed action.
13. Every trainable parameter has exactly one optimizer owner. Decoder-local
    clipping precedes global clipping; finite post-clip values cannot hide a
    missing raw owner gradient.
14. Formal runs fail closed on missing language mapping, identity mismatch,
    stale CandidateWorld identity and non-finite values.
15. Checkpoints, tensor caches, raw logs and full probe dumps never enter
    architecture-memory documents.
16. Numerical step conditioning and semantic object binding are independent
    contracts. `FlowStepContext` may enter the bottom flow, while raw language,
    color/object pointers and role logits may not; object binding must be
    resolved upstream and cross the seam only as a compiled physical condition.
17. A sequence-conditioned W may read only the known 24-row physical prefix.
    Its four interval reads end at 8, 16, 24 and 24; it cannot synthesize rows
    25--48, consume future supervision or replace the one-rebuild Q5 lifecycle.

## Typed boundary summary

| Boundary | Required semantics |
|---|---|
| OnlinePolicyInput | Causal RGB/DINO, state, executed history and language only |
| FutureSupervision | Disjoint training-only future action/state/DINO evidence |
| ObjectFactSet | K=4 physical objects plus explicit null and observable object/camera mass |
| ActionIntentDock | Public S interval/history/K memory plus producer-owned K validity mask; no typed fact re-entry |
| PolicyIntentDock | Reduced typed S context for P2/P3; opt-in shared FP32 target-K address |
| PhysicalActionCondition | Four physical interval means plus current-anchored deltas |
| PhysicalActionSequenceCondition | Exact known 24-row source action, canonical value/delta, current boundary, row times and FP32 outlet/chart/normalizer identity |
| ObjectWorldBelief | Compact current G belief; no S/Teacher/noisy action |
| CandidateWorld | One exact action condition atomically paired with FutureObjectDynamics |
| FutureObjectDynamics | Semantic successor/delta and camera-resolved transport/covariance |
| FactualPrecisionDock | Completed protected P1 detail; no new reader or compression |
| CompletedP1PolicyState | Static factual base separate from dynamic action/time residual |
| SelectedIntervalEvidence | Interval-retaining semantic/geometry values with no-null physical terminal |
| ControlledTransitionSource | Exact completed G3 rollout, built once per observation |

## Registered component hierarchy

One experiment selects exactly one lazy implementation for every slot:

~~~text
ConditioningStage
  -> ObservationStage
  -> SharedRoleQueryBridge
  -> GroundingStage
  -> IntentStage
  -> WorldStage
  -> P1Stage
  -> PolicyCompilerStage
  -> ControlledTransitionStage
  -> ExecutionBottomStage(TerminalActionController)
  -> OutletAdapter
~~~

Static observation work and per-ODE dynamic work are separate. Boundary
containers pass existing references without detach, clone, hidden projection
or reconstructed axes. WorldStage accepts only ObjectWorldBelief plus the
selected PhysicalActionCondition or PhysicalActionSequenceCondition.
Language-to-object binding is owned above this seam: S/P2 may use language with
the language-free ObjectFactSet and must compile the result into the existing
physical action/consequence carriers. The dynamic
bottom accepts only the prepared physical field, flow time, bottom-local
FlowStepContext, shared action query, compiled physical plan, V120 seed and
transition state; goal, RGB/DINO, ObjectFactSet, raw object pointers, color
labels and task identity cannot cross it. A physicalized consequence may cross
because it is the execution condition, not a second semantic binding route.

TerminalActionController owns every physical-velocity candidate read, not just
the final head. OutletAdapter owns native dimensionality, normalizer-aware
chart conversion, W projection, dynamic sanitization, finalization and metrics;
it may coordinate these formulas but cannot leak outlet identity into the
shared graph.

Component selection and compatibility ABI are serialized before comparison.
Legacy layout/key/order translation is checkpoint compatibility data; it is not
a second registered owner or a runtime façade. Component boundaries never
create a no-grad plane.

## Loss, optimizer and diagnostics

The logged loss ledger is authoritative. Interpret weighted contributions and
group totals before raw auxiliary magnitudes. The main groups are action,
representation and execution.

Pen/RDT continuous-gripper trajectory terms supervise the deployed continuous
codec. CALVIN action-flow and decoded-action objectives are arm-only and its
binary command uses ordinary horizon-weighted cross entropy by default. An
explicit CALVIN-only `motion_event_v1` experiment may reallocate that same
horizon supervision mass across rows using zero-centered relative-TCP motion
and a bounded gripper-transition neighbourhood. It drops no rows, normalizes
the combined row weights back to the original horizon mass, is serialized in
the resolved objective config, and leaves Pen/RDT on exact unit frame weights.
Diagnostics and matched interventions are audit-only unless source shows a
positive objective weight.

Every train window requires an exact ledger, finite raw owner gradients and
one optimizer owner per parameter. Any compatibility ordering needed for an
old checkpoint is explicit and tested; constructor history is not an
architectural semantic.

## Identity and checkpoint contract

- Manifest, resolved config, component selection, source provenance, dataset
  inventory, normalizers, language artifact, optimizer ownership and random
  continuation state are serialized in run context/checkpoint metadata.
- The current dataset inventory is composite and must not be called a raw-file
  checksum; identity separation remains an open repair-plan item.
- Exact resume fails closed on incompatible architecture, component, data,
  optimizer or continuation state. A migration requires an explicit tested
  contract.
- `p2_shared_target_prior_v1` adds exactly the zero `[1,3]` address owner.
  `p2_shared_target_prior_pread_v1` additionally admits only the five reviewed
  physical-reader paths. Both are fresh optimizer/schedule/RNG initialization,
  require identical dataset/language/normalizer identity and are not exact
  resume across modes.
- `p2_shared_target_prior_sequence_prefix_pread_v1` is the combined opt-in
  migration. It preserves every old model tensor exactly and adds exactly seven
  tensors: the FP32 exact-zero `[1,3]` target address, one exact-zero
  `[1,24,H]` coarse row offset, one sequence-time projection and one GRU's
  input/hidden weights and biases. Only the reviewed 18-path source allow-list
  is admitted. It is a fresh optimizer/schedule/RNG initialization, requires
  identical dataset/language/normalizer identity and is not exact resume.
- `p2_post_pool_pread_control_v1` is the matched physical-reader control. It
  keeps `post_pool_only`, changes only `mmap` to `pread`, preserves the exact
  model state-key set and uses the same fresh initialization boundary as the
  shared-target treatment.
- Validation replay is read-only: it does not load optimizer/scheduler/random
  continuation state and writes no checkpoint.
- Formal output directories are new and empty, and checkpoint writes are
  atomic. Filesystem overwrite protection is separate from architecture
  compatibility.
- A smoke checkpoint is gate evidence, not a formal initialization source.

## Release and audit rules

Interface closure requires focused tests, state/owner coverage, real
CUDA/BF16 parameter-owner VJP, checkpoint round trip and bounded memory.
Learned-behavior closure additionally requires a complete comparable curve and
the matched intervention named by the current issue. Outlet claims require
their native metric and, where applicable, a closed-loop benchmark.

Hard stops are non-finite values, identity/lineage failure, an open loss
ledger, vanished formal parameter-owner VJP, checkpoint ABI violation or
process memory above 22 GiB. Finite gradient-threshold crossings, early event
F1, small geometry amplitude and capacity warm-up are telemetry unless tied
reproducibly to a hard failure or behavior regression.

## Authoritative source map

~~~text
identity/config:
  clearvla/mainline/manifest.py
  clearvla/mainline/config.py
  clearvla/mainline/interfaces.py
  clearvla/mainline/model/component_contracts.py

composition:
  clearvla/mainline/model/policy.py
  clearvla/mainline/model/components.py

observation/G:
  clearvla/mainline/model/restored_observation.py
  clearvla/mainline/model/observation_contract.py
  clearvla/mainline/model/grounding.py

S/W/P:
  clearvla/mainline/model/intent.py
  clearvla/mainline/model/dynamics.py
  clearvla/mainline/model/v120_p1.py
  clearvla/mainline/model/compiler.py

action/transition/bottom:
  clearvla/mainline/model/action_codec.py
  clearvla/mainline/model/transition.py
  clearvla/mainline/model/restored_bottom.py

training/runtime:
  clearvla/mainline/training/
  clearvla/mainline/runtime/
  clearvla/mainline/train.py
~~~

Historical replay provenance is indexed by
[archive/replay/README.md](archive/replay/README.md). Open it only for ancestry,
an old log or the reason behind a past repair; never reconstruct the current
graph from the ledger.
