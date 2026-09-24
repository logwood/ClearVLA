## M8m/M8n — bounded train/validation measurement (2026-09-24 UTC)

`fit-heldout` is an explicit qualifier operation, NOT a new neural mode. Its
real CALVIN lane audits the complete train/val episode inventory, including all
training episodes used for the existing train-only normalizers. Distinct
annotation filenames do not establish independence: shared raw trajectory IDs,
contradictory trajectory bounds, overlapping raw intervals (including a shared
boundary frame), aliased paths, unknown provenance or unverified normalizer scope
block this diagnostic before model construction. Original splits and examples
are not rewritten, dropped or repaired. This checks recorded provenance, not
adversarial metadata authenticity, content deduplication or semantic independence.

Default probe rows spread over distinct source groups and take a central eligible
window, without consulting labels, contact, model outputs or success. Explicit
train/val dataset indices may be supplied and are recorded; collated CPU audit
identities must agree with the chosen split rows. The legacy single-batch
operation retains its old selection. Actual action-label, endpoint-label and
physical-feedback supports are reported, not used as success scores or gates.

Only the fixed training batch receives optimizer updates. Training and validation
batches each have fixed-seed before/after measurements at their actual completed
update clocks; evaluation preserves the training RNG streams and module modes.
No test batch is selected for model evaluation, no early stopping or checkpoint
selection is performed, and no decrease is required for numeric completion.
The inherited data loader may still inspect/materialize the complete source
inventory. A pair of synthetic batches explicitly carries NO real-separation
claim. One validation batch is not a population estimate or physical-skill test.

Configured warmup/minimum ratio, objective budgets, neural parameters and physical
history semantics remain unchanged. Missing real paths stay blocked. The bounded
probe does not install dependencies, upgrade XVLA, run formal training or emit
weights; full production CUDA/BF16, recorded-data learning and closed-loop results
remain separate requirements. Source preservation and exact test results are
recorded in the delivered report rather than inferred from the operation name.

## M8l — bounded fixed-batch learning measurement (2026-09-24 UTC)

The runtime qualifier has an explicit `fit-batch` operation requiring both a
source (`real` or `synthetic`) and a positive finite update count. It admits the
same actual batch and model as the existing path, freshly encodes each engine
call, and records every completed optimizer update plus fixed-seed before/after
loss and endpoint-goal metrics. Evaluation restores CPU/active-CUDA random
streams and module modes. It uses the real completed-update phase, not an
emulated inference clock; before/after measurements may therefore differ in
both weights and controller phase. Loss decrease is measured, not required or
called held-out/physical success.

The bounded diagnostic schedule keeps configured warmup and minimum LR ratio
with the explicitly requested update horizon; it records that this is not the
formal dataset schedule and never secretly shortens warmup. Missing real data,
runtime/device mismatch, nonfinite computation, interruption or partial updates
remain blocked/failed as applicable. No checkpoint is emitted, dependencies
installed, existing server job changed, neural mode introduced, model parameter
added, loss reweighted or action/W/ODE formula changed. The helper is numerical
learning qualification, not a completed real-data experiment.

## M8k — recovered spatial-causality audit (2026-09-24 UTC)

The restored M8j source already has native reference-patch coordinates in the
endpoint predictor's query/key/value tokens, before nonlinear attention and
FFN reads. Do not infer missing spatial conditioning by changing CURRENT G3
posteriors while keeping its instruction-start input fixed. That invariance is
intentional. Same-content/same-centroid START layouts can change predicted
endpoint laws and global robot relations; removing positions in a diagnostic
copy removes the global distinction. Preserve this source/clock separation.
This verifies a representational path, not learned physical correspondence or
success. No extra spatial head or new neural mode is justified by that failed
hypothesis. Controlled source probes and exact inventories stay outside this
architecture ledger; no old partial test count is a new acceptance result.

## M8j — reference-conditioned annotated endpoint relation candidate (2026-09-23)

`top.annotation_goal_mode=annotated_endpoint_relation_v1` is explicit in
`structural_rebuild_annotated_goal_calvin.json`; all earlier modes stay unchanged.
This is a supervised demonstration-end prior, not an oracle, calibrated physical
relation, success classifier or elapsed-time progress value. The default new
objective weight 0.01 is a declared untuned budget, not a measured optimum.

The converter's annotation ID is now preserved by the HDF5 loader. A separate
training-only AnnotationEndpoint owns source annotation/context/start/end/center,
real endpoint state and full native DINO chart. The endpoint is read independently
of the action/W horizon in the existing grouped token load. Unknown provenance
or a censored prefix supplies no goal label but retains its behavioral example;
contradictory source indices fail admission. Terminal storage padding is never
the target. The source chronology must agree with the online instruction anchor.
Neither these indices, duration nor the actual endpoint enter OnlinePolicyInput.

The new S predictor reads ONLY the causal instruction-start DINO, robot feature
and protected language tokens. It predicts a normalized endpoint scene, a full
reference-patch-to-endpoint-patch-plus-null distribution and robot feature delta.
Current observations cannot change this prior under the same weights/reference/
language. Predictions are not detached from their direct losses or policy use.
A conditional mean scene can blur ambiguous futures; the finite soft relation is
not a persistent physical-object ID or complete semantic task specification.

Current observations and terminal labels use the same fixed full-native cosine
observation law, with uniform real-cell measure and a null alternative. The
label/current metric has no learned projection that could erase its supervision.
It is explicitly uncalibrated appearance correspondence, not metric 3D geometry.
S and P3 independently transform content, coordinates and joint features BEFORE
posterior expectation. Existing soft current-G3-to-start support transports the
remaining relation to current K without matching slot indices. Target/null mass
is applied after reading, not divided away. Robot relation remains a separate
feature chart. P3 consumes these values on its task/temporal lane, not by renaming
observed-change or W-retrodiction discrepancy values. Its projections are cached
once per online encoding; the goal adds no G/W/ODE image calls, but does add
learned parameters and quadratic native-patch attention work.

Partially missing current destinations condition BOTH goal and observed laws on
the same available cells plus null; unseen cells do not become goal failure.
The fixed physical goal prior remains unchanged by this comparison support.

Scene Huber, relation KL and robot Huber are independently source-masked and
averaged into the new budget. Missing destination cells condition both laws and
are not trained as negatives. Endpoint labels are detached; learned target/null
confidence cannot erase goal supervision. No end-of-window heuristic, success
threshold, task string branch or deployment-only memory is introduced. Typed
cache/deployment/checkpoint ownership distinguishes the candidate from old models.

This is a saved implementation candidate. Exact completed tests, failed attempts,
resource costs and runtime identity belong to its reports; a source save is not
supported-runtime, real-data learning, GPU or closed-loop acceptance.

## Executed-control world discrepancy candidate (2026-09-23)

Explicit `top.world_feedback_mode=executed_four_step_world_v1` is selected in
`structural_rebuild_executed_world_calvin.json`. Older configs stay unchanged.
This requires the declared CALVIN state/command/time chart, aligned sequence W,
current-image G3, shared soft target selection and typed P3.

At observation o[t] before a[t], the input owns state/boundary at t-4, its causal
visual source (reset-clipped t-12,t-8,t-4), and EVERY acknowledged/recorded command
a[t-4:t]. The older sparse history cannot supply t-3. Dataset, grouped token
loader, typed batch, bounded history, bridge and policy carry the same record.
Selected history retains 13 observations; default stays 9. Before four actual
steps the transition is unknown. Instruction changes keep physical history;
only environment reset clears it. History action dropout invalidates it too.

One detached shared G/W1 replay uses the older input and confirmed controls
under CURRENT weights, exporting only the existing four-step endpoint. This
is RETRODICTION, not a forecast saved before execution. No current image,
language, task age or future target enters this W query. An observation-only
frozen association primitive measures the current frame against the old G
chart. Its metric/checkpoint parameter names and original soft/null equations
are shared with W labels; future-label forward/recognizer are not called online.
The measurement is the existing POOLED NORMALIZED W grid, not native-token
resolution, calibrated 3D motion, physical contact or persistent object ID.

Current and old K are softly matched through common-current-image support and
learned Q/K plus learned null; slot indices do not establish identity. Soft
support overlap accounts for the grid cell measure. No argmax, confidence
threshold, task keyword or instruction-age rule selects a match. Discrepancy,
source W/G and factual match inputs are detached. P3 reader parameters and the
existing operated-target selection receive policy gradients. Zero error or
absent/null correspondence yields zero feedback regardless of status; the whole
policy is not forced to zero. Covariance modulates error reading but is not a
calibrated error bound. Prepared values are cached once and checked for exact
source/reader/target ownership before the real P3 action consumer.

Cost is explicit: one extra prior G and W1 per encoding, no extra W2. Normal
two-pass sampling has three W1 and two W2 calls, not two worlds total. The
three-frame DINO adapter also runs one extra old-source encoder batch, reusing
the two overlapping current-cache frames exactly. No per-ODE image reopening.
S's desired terminal relationship, calibrated object association, real-data
learning, GPU resources and physics remain open. A source_end index is not
silently promoted to success or a progress label. No formal training, legacy
checkpoint migration, workflow/master change or XVLA upgrade is implied.

The execution container reset before the first source packet was fully saved.
Recovery was performed on the independently hash-verified prior source tree;
rebuilt commits/tests are new identities. Lost/incomplete results are not part
of the new acceptance totals. See delivered recovery note and exact-source logs.

## Soft G3 posterior instruction evidence candidate (2026-09-23)

`top.instruction_change_mode=g3_posterior_reference_v1` requires the current-image
G3 chart, causal instruction reference, shared target law and typed P3. The
cumulative config is `structural_rebuild_g3_posterior_calvin.json`; M6n remains
an unchanged experimental control, not an automatic checkpoint alias.

The actual G3 log read and original local location law are retained and pushed
onto the full native DINO chart. Same current-patch queries softly read both
current and reference images, with a learned null competing with every real
key. The current G3 law mixes the conditional matches AFTER softmax. No K-slot,
argmax, threshold, task string or instruction age selects a correspondence.
This is not a calibrated or persistent physical-ID guarantee.

S and P3 independently project content, coordinates and their joint features
BEFORE expectation, instead of forcing every observation through its centroid
and entropy. These finite learned features can still collapse or fail to
distinguish some distributions. Both correspondence-null and operated-target-null
retain their real mass. Image summaries are null-aware moments, not calibrated
physical displacement. Source availability and match entropy remain status,
not a progress value, success label or target-conditioned visual fact.

P3 projections are built once in online encoding, owned by the evidence and
reader, and passed through the actual intent/policy dock. ODE calls consume
these projections without reopening images or adding a W build. Identical
comparable observations yield exactly zero measured content/image/joint change;
status alone cannot generate a P3 change value. Robot feature change is separate.
Future Teacher labels cannot enter this observed comparison. This candidate
still lacks a desired-relation target, calibrated object association and an
object-matched prediction-versus-observation record for actual executed controls.
It must not be described as a learned feedback policy or a task-progress model.

Source preservation precedes milestone acceptance. Exact test inventories and
failed/timeout records are in the checkpoint report, not inferred from pytest
dots. Production runtime/GPU/real-data/physics qualification remains separate.

## Continuation: complete scheduler curve identity (2026-09-23)

WarmupCosineSchedule now serializes its v2 schema, unchanged formula/clock,
warmup/total/minimum-ratio definition, base LRs and completed step. Exact restore
and standalone schedule loads admit the full curve before mutation. Save/load
also verify the actual optimizer owner; runtime parameter-group references reject
reordering without depending on replaceable group dictionaries. The old two-field
state is not silently migrated. Valid LR arithmetic is unchanged, including the
existing total<=warmup convention. No model parameters, task state or solver work
are added. CPU checks are scoped to recorded tests, not GPU/data/skill acceptance.

## 2026-09-23 takeover candidate — training/update and resume admission

Continues the published M8g tree c15627029b45af116eadc25ee6a6328da76e695e,
without a new neural mode, checkpoint migration, solver/W call or workflow change.
The training scalar must be finite before diagnostic VJPs, backward and optimizer
work; a nonfinite additive loss can have finite gradients. Rejection does not
claim to roll back already executed forward/RNG activity or every mutable buffer.
Checkpoint epoch/global/schedule clocks must be nonnegative Python integers,
not coerced bools, floats or strings. Validate before file creation or live
model/state loading, including model-only validation/initialization entry points.
Exact resume additionally compares fixed optimizer options by type and finite
value, separately from parameter-ID mapping and the time-dependent LR check.
It cannot certify an arbitrary custom optimizer's internal behavior or infer the
entire future scheduler curve from the old step/base-LR-only schedule payload.
Local tests and preserved source are not supported-runtime, GPU, real-data or
closed-loop acceptance. Existing finite numerical paths are retained.

## M8g — expose the controller phase in production qualification (2026-09-23)

Fresh-weight inference defaults to the zero-update execution clock. It cannot
qualify the post-warmup controller merely by constructing H=512. The bounded
probe now admits an explicit nonnegative `--inference-step` only for inference,
calls the real `set_training_step` before online encoding, and records both the
requested clock and returned execution phase. This emulates the capacity
schedule on fresh weights; it is NOT a resumed or trained model. Data inspection
and one-update training reject a nonzero emulated clock. No parameters, modes,
solver steps, production numerical operations or user environments are changed.

## M8f — qualification process ownership (2026-09-23)

The runtime probe now binds Linux workers to the owning parent process before
importing Torch; parent death cannot leave a full model running as an orphan.
Parent SIGINT/SIGTERM and timeouts terminate the owned process group, retain the
last completed stage, and never convert partial output into success. Abrupt
SIGKILL can leave a last-stage report marked running; it is not acceptance.
The parent process metadata is separate from the child-owned atomic report.
Relative configuration paths are resolved before changing worker cwd. Sampled
outlet tensors are saved separately and explicitly are not accepted/physical
commands. All measurements identify fresh deterministic weights, not a trained
policy. No external runtime packages, workflows, or user training jobs change.

## M8e — P1 native-chart activation policy (2026-09-23)

P1 query tiling now accounts for native candidate cardinality relative to its
public query canvas. All queries, candidates and local micro cells are still
read. Wide native posterior charts enable the existing raw checkpoint flag
even at narrow test widths, and request pure-quadrature recomputation whenever
gradients are enabled, including eval-mode VJPs. Inference under no_grad has no
new recomputation. Small-chart legacy computational policy is unchanged.
G-aligned P1 no longer constructs a candidate-distance graph that it immediately
overwrites with zero: future-conditioned likelihoods remain disallowed at P1.

The 4-GiB shared container killed grouped native-chart backward tests; these
are resource failures, not passes. Narrow-width native-256 backward has separate
standalone measurements. Routine branch/gradient tests use 100 native patches;
full native-256 forward, ABI, stress backward and H=512 inference are separate
checks, not substitutes for production-scale training/GPU qualification.

## M8d — native candidate chart is not the public query canvas (2026-09-23)

Full-posterior G1 uses every patch of the configured native DINO chart. G2,
G3 and P1 retain that same N; the 8x8 public canvas is a distinct query axis.
The real H=512/256-patch/336-RGB production forward exposed a hard-coded N=64
validator and ABI. Runtime validation now compares values to the independent G1
coordinate/key/logit source, and deployment metadata obtains N from graph
`patches_per_camera`. Truncation does not redefine the source size. Legacy
moment-local support remains 49. Existing 64-patch full-posterior metadata is
unchanged; stale 256-patch ABI claiming 64 is rejected, not silently migrated.
No candidates, dimensions, heads, or learned weights are removed or added.
The fixed production FP32 run reached two-pass sampling; actual reports state
synthetic inputs, batch size and unqualified runtime. It is not skill acceptance.

## M8c — bounded runtime and data qualification (2026-09-23)

`scripts/qualify_mainline_runtime.py` inspects the existing interpreter and
configured data, or executes one actual-model inference / optimizer update in
a bounded child process. It never installs dependencies, replaces missing real
data with synthetic tensors, falls back from CUDA to CPU, changes dimensions,
or runs formal training. An explicitly requested dtype override and declared
runtime mismatch are recorded. Synthetic DINO/T5 inputs qualify numerical
execution only, not external encoders, real data, learned skill, or physics.
The atomic report distinguishes input admission, model construction, forward
attempt, completed computation, timeout, and process failure. Output directories
must be new/empty and outside the checkout. The current server XVLA environment
is not to be upgraded merely to meet the repository's declared runtime.

# Current ClearVLA architecture contract

## M8b: CALVIN sequence/task/command lifecycle

Both repository CALVIN evaluators now wrap the real env with
`CalvinExecutionEnvironment` and enable command acknowledgement. Only a successful
`env.reset` clears the physical timeline. Official per-subtask `model.reset`
discards unexecuted rows and starts a new instruction anchor, retaining the last
accepted command and history even for identical instruction text. A language
change also invalidates a reused action chunk. This applies with/without video.

A prepared row is pending, not an execution record. Only a returned, validated
step commits it. `ExecutedCommand` may explicitly report a different accepted
canonical command; otherwise the adapter confirms the submitted command at the
successful call boundary and rejects input-buffer mutation. This does NOT
observe low-level actuator commands, displacement or contact. Wrong submissions
fail before physics; uncertain/failed steps or remote planning failures require
an environment reset rather than a duplicate history append. Observations and
commands alternate exactly once in receipt-enabled chunked execution.

The legacy unwrapped duck-typed model mode remains explicitly marked
`legacy-assumed`, not receipt-certified. Repository evaluators require the new
bridge instruction capability; external clients must use the same wrapper or
supply equivalent lifecycle events. No reward, oracle or scene state is sent
as policy input. Official loop reference: mees/calvin, evaluate_policy.py,
`evaluate_sequence` vs `rollout`, inspected 2026-09-23.

## M8a: instruction events across the benchmark bridge

Bridge protocol v3 distinguishes `begin_instruction` from physical `reset`.
The explicit task event calls the policy anchor hook without clearing causal
history or reseeding action RNG, including identical consecutive text. Wire
flags are scalar bool/uint8 events, not arbitrary truthy data. Unsupported task
events and invalid observations/instructions fail before history mutation.
Existing clients without a task event preserve their old request semantics.
This is an execution-interface fix, not a task-success or persistent-ID claim.

## M7l: numeric layout agreement is not future-control dependence

Same physical action/time prefixes on legacy 24/48-row layouts are admitted
exactly, but FP32 per-token RMS may round differently on a full tensor versus
its slice. Cross-extent output comparisons use rtol=0 and an explicit
16*FP32-epsilon*reference-peak budget per typed output, without a fixed absolute
floor. The budget is a regression policy, not a global error theorem. Tests
separately require bit-exact near outputs when only later controls change in
the SAME layout. Diagnostic canonical-row RMS made the two prior failures
bit-exact but is not installed in production; no slow numerical workaround is
added to satisfy an unjustified across-layout zero tolerance. Zero/signed/NaN/
materially wrong tiny signals remain rejected by dedicated tests. The PyTorch
2.10 numerical-accuracy contract explicitly distinguishes full/batched and
sliced computation; checkpoint/diagnostic and same-layout invariants stay exact.


## M7f–M7g: sampling admission and adjacent command provenance

Sampling verifies the caller's dimensions, observation, top/bottom graph and
input profile/named-camera chart against the instantiated production policy.
Data paths, output locations, training budgets and explicit solver schedules
remain legitimate operational overrides. Explicit initial fields must be finite
floating-point tensors with the outlet's physical shape. Graph/seed/schedule
rejection happens before encoding, random draws or changing the model's mode;
integer fields cannot truncate velocities by casting them back to integer.
These are entry-boundary checks, not per-ODE reductions. Numerical standalone
test fields retain their explicitly smaller ABI. Valid sampling is unchanged.

Expensive preflight validates the one-step response record's exact [-1,-1,0]
clock (or an absent zero clock). If sparse action history also samples t-1,
both records must agree on execution support and the normalized accepted
command. No array-position guess or forced t-1 entry in other sparse charts is
introduced. Unobserved NaNs remain masked and unmodified. This audits observed
command provenance; it does not infer measured robot displacement or contact.

## M7h: reusable graph tensor versions, without copying observations

The existing input/model/step/mode identity guard additionally snapshots tensor
references, PyTorch mutation versions and metadata of the online input records
and named parameters. Ordinary in-place writes (including views), parameter
replacement, requires-grad changes, load_state_dict and external optimizer
updates invalidate a reusable graph before Teacher, flow RNG or velocity work.
Fresh encoding remains legal; independent future labels are not part of this
snapshot. Gradient accumulation/zero_grad does not change parameter values.
No tensor storage is copied or hashed, and no per-ODE check is introduced.

This is not complete storage immutability. Inference tensors have no mutation
counter and are explicitly listed as unversioned while retaining reference and
metadata checks. `.data`, raw-memory/custom writes without a PyTorch version
bump, and buffers modified externally remain caller-owned invalidation cases.
The execution phase buffer is intentionally excluded because the engine fills
its canonical value even when the phase is unchanged. Normal sampling does not
use this training-graph snapshot. The runtime stamp is not serialized in model
checkpoints and does not add trainable parameters or numerical forward work.

## M7i–M7j: exact cache fixtures and finite bottom numerical scales

The default legacy top does not materialize current_world_belief; its deployment
cache wraps the same current fact tensors. The cumulative robot-conditioned top
must instead preserve the materialized world's exact owner. Tests now check
both cases separately, and DeploymentTopCache admits exactly its actual three
fields rather than inventing an extra copy of the online source field. No
numerical tolerance is relaxed. Missing external video backends remain a CI
environment failure, not a neural pass or an automatically skipped test.

Bottom residual scales, normalization floor, FFN expansion and operator-depth
initial logit must be finite and positive. They have no infinity-as-disabled
convention. A min-of-values positivity check can admit NaN and positive infinity;
those values are rejected during configuration validation, before model creation
or random initialization. Existing finite positive choices remain configurable;
this does not freeze them to legacy constants or change valid forward math.

## M7a–M7c: cumulative graph correctness, without new neural modes

These source-wide corrections continue the published M6p tree
`7abffddad974616518b87a8465dd4bdb46e015e0`. The cumulative model selection is
still `configs/mainline/structural_rebuild_m6n_calvin.json`. Saving a source
unit before milestone CI is not a claim of supported-runtime or robot acceptance.

**M7a — one completed-update execution clock.** `MainlineTrainingEngine.global_step`
counts completed optimizer updates. Successful `train_step` publishes the phase
for that completed count only after backward, optimizer and schedule advance.
Both `encode_eval` and `eval_step` resolve the same phase before building/reading
an evaluation graph. Deployment already restores it from the checkpoint count.
This closes the old one-step difference between in-memory evaluation and reload
at the 200/1000 controller schedule. Current-update diagnostics remain detached
snapshots, not aliases of the next phase. Failed forwards do not advance the
step; updates made outside this engine still require explicit resynchronization.
No controller schedule, parameter, inference W or ODE count is changed.

**M7b — language support before the first trainable projection.** T5 padding is
quarantined by the source mask BEFORE `goal_input`, not only before attention.
An unsupported NaN can otherwise leave a finite forward value but poison the
Linear weight VJP through `NaN * 0`. The mask must align with token axes. This
changes no arithmetic for supported tokens, adds no fallback label or learned
null, and does not hide nonfinite supported evidence. Preflight uses the same
language support law. All-padding inputs may still use existing learned priors;
zero padded evidence does not imply that the whole policy must output zero.

**M7c — finite audit follows causal support and shared label ownership.** The
expensive preflight checks current action-state and codec-gripper boundaries as
observed values, and checks state/action history only on their separately owned
observed/executed supports. It validates the actual history clock before using
those masks. Complete-window and observed-tail policy/world action labels must
share the same normalized recorded prefix; absence of a support record means
complete support, not permission for conflicting labels. The future-time chart
is checked before this equality, so malformed clocks fail at their own boundary.
These reductions remain preflight-only, not additional per-ODE host syncs.

The old candidate-prefix test harness now explicitly admits the existing
`arm_private_correction=None` keyword and rejects non-None private-arm data it
does not implement. Its numerical and prefix-sharing assertions are retained.
Final test inventories and process exits belong to the handoff/archive, not
inferred from partial pytest dots or a successful source push.

## M7d–M7e: cumulative verification ownership

M7d updates source-owned tests to the already implemented camera/time/cache ABI:
single-view top fixtures declare their actual camera names; tensor equivariance
keeps its prior tolerances while non-tensor provenance is compared exactly;
current-world/causal reference fields cannot be mistaken for future labels.
No production path or numerical assertion is relaxed by these fixture updates.

M7e keeps historical baseline and current test outcomes independent. A failed
or killed baseline child cannot skip current-source testing. The read-only shell
gate attempts both inventories, stores their individual exits, and returns a
failure if either failed; static/source preparation errors still stop before
unprepared testing. Cumulative discovery includes mainline, CALVIN, benchmark
language/config, deployment-prefix, simulation recording, residual RL and the
review scripts. It runs real test files in fresh processes, not replacements.
Optional video import availability is recorded in runtime metadata. An absent
codec/backend remains an environment failure, not a skipped successful video
check. Workflows and their environment installation are not changed here.

## M6l–M6p batch: execution, endpoint, command and supervision ownership

This batch continues the M6k source tree; it does not declare the entire M6 or
system rebuild accepted. The isolated branch is the only intended remote target.
Five connected units have been implemented within this continuation, with local
source commits at closed boundaries and one final report. Complete checkpoints
are retained without adding or changing GitHub workflow files.

### M6l: `bottom.controller_value_mode = separate_magnitude_v1`

The native EvidenceExecutionController and NativeExecutionValueReader preserve
selector/value ownership downstream of M6j. Source normalization remains in K;
source V and operation-private-state V use bias-free unnormalized projections.
The candidate reader now constructs evidence K from the actual selector stream,
not by reusing the value stream as a key. Its memory/action/temporal attentions
normalize Q/K but not V, with bias-free attention/output projections. Explicit
matching value streams are mandatory in this mode. The configuration requires
M6j magnitude-preserving evidence and records independent deployment metadata.

This is a learned computation/capacity policy, NOT the robot actuator. Existing
capacity sigmoid and 200-step warm-up, candidate centering, GRU identity state,
FFNs, decoder operator budgets and terminal decisions remain unchanged. Their
hidden nonlinearities have legitimate representation roles: zero evidence does
not force a zero controller state, zero capacity or zero policy output. The
constraint is on the identified reads, not global scale equivariance.

### M6m: `bottom.endpoint_supervision_mode = clean_command_v1`

Binary-gripper modes may train existing command CE and motion BCE on a separate
clean-arm endpoint forward, reusing the SAME causal OnlinePolicyCache. Its
numerical context is exactly t=1; where schedule contexts are enabled, dt=0,
normalized index=1 and endpoint=1. A common helper defines both training and
existing proposal/refinement endpoint calls. Known arm rows use detached clean
action labels; unlabelled rows keep source noise; binary-gripper field lanes
keep source noise and are neutralized by the same existing outlet boundary as
in deployment. No clean gripper label is fed to the command head as an input.
Unavailable head rows are quarantined BEFORE CE/BCE; masking a NaN loss afterward
would not protect its gradient.

The interior velocity forward retains flow-matching, clean-trajectory and
execution-value objectives. Only the existing command/motion budgets change
owner to the endpoint; no duplicate head loss or new objective weight is added.
The endpoint forward does not request candidate execution supervision and never
rebuilds G/S/W. Training has one additional velocity evaluation; deployment keeps
its existing solver and two endpoint reads. Teacher-forced clean arm inputs
match the numerical endpoint, not necessarily the sampled inference distribution.
This is not a guarantee of correct gripper behavior or an implicit task-stop gate.

### M6n: controller-boundary receipts and immutable recording

`ExecutedCommand(submitted, applied)` holds independent read-only finite native
[7] arrays. The rollout protects the raw policy proposal from mutating clipping
and step adapters. Following a successful step, it checks the receipt against
the exact submitted request and next action_state; its applied value becomes the
single command used by history, the HDF5 action label and execution telemetry.
Old adapters without receipts retain their identity/allclose admission rule and
cannot silently change a command. Repository Alicia and ManiSkill adapters
produce receipts. ManiSkill freezes the accepted command before handing a copy
to a backend that may reuse inputs. These receipts identify the command boundary,
not measured joint movement or undocumented internal actuator transformations.

EpisodeRecorder also copies successor observations, preventing simulator buffer
reuse from altering saved labels. A receipt/label disagreement is rejected before
recording. The residual-RL collector protects its clip-check input and rejects a
receipt that changes its identified residual action mapping BEFORE replay write;
silently associating a different actuator command with the original residual
would change the learning problem. Evaluator telemetry never enters policy input.
External CALVIN evaluation scripts that are not in this repository remain unaudited.

Candidate configurations accumulate changes in structural_rebuild_m6l_calvin.json,
structural_rebuild_m6m_calvin.json and structural_rebuild_m6n_calvin.json. M6n uses
the M6m neural graph plus the source-wide simulation boundary; it adds no model
feature, reward input or learned stage machine. Legacy disabled-mode serialization
and model initialization/output are compared independently against M6k.

## M6o: source-supported loss closure

Action-row support is applied before trajectory decoding, CE and BCE, including
the interior-head control mode. Candidate existence and label support are distinct
producer masks, jointly applied before candidate cost/Huber computations and
audit reductions. Empty support yields finite zero contributions; valid NaNs
remain errors. Gripper audit denominators and terminal identity checks count
only actual supported candidates/rows. No online tensor or deployed decoder is
masked by future label support. This is a training-loss/audit correction, not a
new model mode, noise distribution, source-truth mask, or inference fallback.
It does not make NaNs in upstream supported activations acceptable.

## M6p: reusable training graph ownership

`EncodedTrainingBatch.capture` binds one causal input object, model instance,
engine optimizer step and train/eval mode to paired online/Teacher source planes.
Cached loss composition rejects a different online observation, model, engine
step or mixed source planes BEFORE Teacher, flow-noise sampling or velocity.
Both planes must retain the same intent and candidate-world objects. Future
labels/audit metadata may change while retaining the same online input; they
do not own or overwrite its cache. Guards are CPU object/step checks plus the
existing tensor contracts, once per composition, not per ODE node.

This record is ephemeral, not a serialized checkpoint or immutable tensor
container. In-place input tensor writes and parameter mutation outside the
engine are not detected. Callers must treat source tensors as immutable and
rebuild after any optimizer/external weight change. Equal-valued reconstructed
inputs need re-encoding. No per-step image hashing, cloning, learned parameters,
device reductions or simulation state are added by the ownership guard.



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
S/W motion and Teacher. The latest explicit source candidate is
`configs/mainline/structural_rebuild_m6n_calvin.json` (validation scope in handoff).
M6o/M6p correct shared loss/cache semantics; they do not add another model mode.
M5a adds one soft operated-
object distribution across actual S/coarse/P1/P2 readers; its detailed contract
is below. M5b adds a causal instruction reference; M6a/b/c separate action-matched W
supervision, known-control evidence and current robot/view relations. M6d
resolves the same physical 24-step future grid through all active consumers. Learned
progress, physical object identity and deeper downstream redesign remain open. Legacy and M1a/M1b/M1c/M1d/M2/M3/M4b
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

## M6k typed CT feature conditioning

`bottom.transition_condition_mode=typed_plan_v1` requires magnitude-preserving
bottom values, typed-horizon P3 and the control-aligned future grid. Default
`summed_legacy_v1` retains the old constructor, parameter order, RNG and arithmetic;
it is omitted from legacy config serialization. No weight/optimizer migration.

ControlledTransition remains a decoder feature operator over completed G3,
NOT the action-conditioned physical W or a measured execution response. Its
input now preserves eight independent value owners: noisy action query, P1
precision, semantic/geometry effects, their separate interactions, and P3
plan/observed-change contributions. Each has a bias-free projection. All
source/24-control-row/4-basis identities survive into the latent read; there
is no pre-projection sum and no global direct-mean shortcut. Source IDs and
row/basis addresses enter keys only. P1 current fact and the seed's complete
state/history/execution rows condition the query, not an additive CT value.
These history rows are existing encoded summaries, NOT newly claimed adjacent
physical state/action pairs.

Both attention stages preserve unnormalized values. A bias-free coefficient
head plus tanh reads the latent values, then contracts them with the existing
form of G3-aligned low-rank basis. Zero dynamic feature values give exact zero
coefficients and write, even with nonzero source/type/context. Context/selector
parameters have no value gradient in that zero case; source-value gradients can
still learn. Small signals are not unit-normalized. The reference is a structural
zero-feature origin: no learned neutral queries, neutral bias, unconsumed visual
base head or extra neutral forward exists in the new mode. ZERO NATIVE ACTION
DOES NOT MEAN ZERO QUERY (role/time content remains), so this is not a physical
no-op guarantee. Legacy centering remains only in the explicitly old graph.

The production compiler passes the SAME consequence record as its protected
P3 carrier. New CT refuses a missing or independently reconstructed carrier;
modular and restored bottom consumers check the value-mode tag. P3's existing
protected/direct lanes remain unchanged: the new use in CT is plan-dependent
conditioning, not a second residual alias added at the bottom. No future inputs,
new W builds, DINO calls, solver steps, mutating ODE state or execution oracle.
The existing delta-neutral probe still executes the selected network, then
neutralizes only its output; diagnostics identify the new origin honestly.

Remaining limits: nonlinear selectors/tanh/basis can change gains; no whole-
policy scale law, calibrated physics, tracking or performance claim. CT latent
compression remains learned; the effective sensitivity and computational cost
need trained/GPU evaluation. Controller memory/value normalization, terminal
heads/Q5 endpoints, native commands and milestone debt remain separate open work.

## M6j magnitude-preserving native evidence values

`bottom.evidence_value_mode=magnitude_preserving_v1` is a new explicit graph/ABI.
Default `normalized_legacy_v1` preserves previous registered weights, RNG draws,
arithmetic and serialized experiment config (the default field is omitted).
No checkpoint/optimizer migration is implied.

The actual evidence adapter uses bias-free source projections for rollout,
transition, event and state. The generic trajectory is still a neutral zero
placeholder in the mainline; it now uses an Identity with no dead parameters,
not a trainable null value. Selector tokens keep source normalization, type
embeddings and the existing role prior. Value tokens reuse those exact source
ranges WITHOUT the bank LayerNorm or type/depth embeddings. Layer values still
come only from clean-intent attention, never the mixed layer-selector chart.
This separation does not remove current factual/intent content.

Every native host MMDiT block uses separate bias-free K/V projections (shared K
bias cancels in softmax; a V bias invents content). Only K inputs are normalized.
V and its output projection are bias-free and unnormalized. The evidence write
uses the existing smooth RMS upper bound at 1 rather than rescaling small
signals toward a unit direction. This is an upper bound, not an amplitude floor
or source allocation. Keys/global condition fixed, a zero value stream produces
an exact zero evidence cross-write and zero derivative to its Q/K parameters;
small values retain their scale. Other self/FFN/factual/intent contributions are
not required to vanish, and selectors may still redistribute other values.

An explicit value stream is mandatory in the new host path: selector fallback
is rejected. EvidenceView carries its value identity and the decoder checks it
before the host path. Config parsing, restored core config, parameter inventory,
checkpoint ABI and same-mode reload include the change. All normal two-pass
sampling, candidate probes and execution paths use the same host implementation.
No additional W, DINO or ODE calls, time states or token banks are introduced.

**Limits:** this unit covers adapter -> evidence values -> native host cross-write,
not invariance of the entire nonlinear policy or calibrated physical magnitudes.
ControlledTransition reads completed G3 plus noisy-plan features, not a W physical
rollout; its learned neutral response is not a calibrated no-op counterfactual.
Its summed/normalized proposal ingress, independent direct P3 route, execution
controller/value normalization, final output normalization, command endpoint
conditions and outlet/history semantics still require deeper review. No robot
performance, exact physical error or complete milestone acceptance is claimed.

## M6i source-supervised object operation expectations

`top.operation_intent_mode=object_outcome_v1` requires shared K-plus-null
binding, the control-aligned 24-step physical grid and typed-horizon P3. Default
`posterior_distillation_v1` leaves the complete old constructor RNG stream,
registered parameter set and arithmetic unchanged and is omitted from serialized
legacy graph config. New modes are explicit graph/ABI identities, not migration.

S owns `OperationExpectation`: FP32 semantic/DINO deltas [B,4,K,D], named-view
image displacement [B,4,K,C,2], and robot state-feature deltas [B,4,S]. A current
G object-content/state and causal S interval query predicts all supported K
outcomes; there is no candidate action or future-label input. W separately
predicts effects of a PARTICULAR action. S expectations estimate demonstrated
instruction-conditioned outcomes, not measured change, terminal goals or an oracle.

For this mode the trainable whole-scene FuturePlanRecognizer is REPLACED by
`OperationExpectationSupervisor`, which has no parameters. Supervision is direct
Huber error against the existing detached Teacher per-K semantic/per-view image
statistics and per-interval successor robot feature means minus current state.
Errors are reduced AFTER preserving object/view identities: opposite objects
cannot cancel before loss. Current source support and each label source's
availability determine loss entries. Target/null probabilities and future match
confidence do not weight or erase losses. Robot labels survive absent visual
supports; observed visual effects do not need future action labels for this
instruction-conditioned expectation (unlike action-conditioned W supervision).
All required samples of a visual/state interval must be available; missing rows
are quarantined before arithmetic. All-missing labels yield attached zero loss.
The previous .35 online + .20 posterior scaffold budget becomes .55 direct
operation expectation; the .25 reconstruction and .20 coarse budgets are unchanged.
The direct objective is the mean of independently supported semantic, image and
robot errors. Source counts and per-type terms are exported; no success loss is
added and there is no moving learned latent target encoder to collapse jointly.

Distinct bias-free value projections interpret content, per-camera image
movement and robot features before view/object fusion. Named-view conditioning
precedes view pooling, and the same K-plus-null mass is applied AFTER within-K
reads without real-mass renormalization. S consumes these expectation values in
its public carrier, reaching coarse/P1/P2 and temporal contexts. P3 owns a separate
typed read into its PLAN lane, not the observed-change lane. Physical-center
interpolation supplies plan features for action rows, not simulated endpoint
states. Zero expectation yields zero additional read and no plan-context gradient.

`OperationExpectation` retains current state/content, camera names, grid and the
shared target binding. Cache admission checks identical source owners; the K
permutation and policy dock preserve binding/evidence together. Online encoding
builds expectations once. All ODE/refinement calls read them without another S/G/W
pass or mutable state. Teacher receives explicit named cameras in this mode;
its numeric target construction is unchanged. New expectation losses train S,
not the detached Teacher, source labels or the online candidate W.

**Limits:** current G entities and Teacher correspondences remain learned and
fallible; their per-object deltas are not physical IDs. Demonstrator continuations
are not normative terminal goals or success labels. Expected future interval
statistics are NOT subtracted from instruction-start-to-current change or from
one-step robot feedback: these have different time and feature semantics. This
source unit does not establish calibrated progress, contact, risk, end-state
goals, physical tracking, CUDA speed or successful robot behavior. S outcome
calibration, W/observation correspondence and complete bottom/transition review
remain open. Existing regression failures remain failures until resolved.

## M6h typed instruction-reference change

`top.instruction_change_mode=typed_reference_v1` requires the existing causal
instruction-start reference, shared K-plus-null target and typed-horizon P3.
Default `mixed_reference_v1` keeps the historical parameters, draws and outputs;
it is omitted from the legacy serialized graph. This is an explicit new graph,
not a checkpoint migration. Configuration, source closure and ABI own the mode.

The matcher uses the SAME current object-content query in the two named raw
DINO image charts. It has no additional task/direction query and receives no
elapsed-task input, future label or generated action. This removes a second
instruction-dependent identity choice after S target binding. The query itself
is learned and does not establish physical identity or correct correspondence.

`InstructionChangeEvidence` retains distinct FP32 [B,K,C,*] content differences,
apparent image-centroid differences and correspondence status, plus [B,S] robot
state-feature differences. Each supported comparison uses the intersection of
raw source-observed pixel support on both sides. A missing source pixel alone
cannot manufacture change for identical content. Status separately records
entropy, comparable view support and current/reference/common availability;
it is not a learned validity gate, contact probability, task risk or movement.
Current camera availability comes from the source-derived fact chart, not a
learned confidence threshold. Both image charts retain their named view axes.

S and P3 own separate typed value readers. Each interprets content, image and
robot differences using distinct bias-free projections; named-view modulation
precedes view reduction. The SAME target mass is applied AFTER each within-K
view read and is never renormalized to force a real object. Robot state change
has its own source and cannot populate the target visual measurement fields.
S may use correspondence status as task context; identical images do NOT imply
that all of S, P3 or the final action must be zero.

P3 directly reads the three actual change values and uses its current plan and
correspondence status only to condition their interpretation. Zero actual
content/image/robot differences imply exact zero from THIS contribution and
zero plan/status derivative. No status-only bias creates a claimed movement.
This contribution joins the existing observed-change lane before its RMS bound;
protected current facts/precision, semantic/geometry candidate effects and the
M6g detached one-step proprioceptive innovation retain their separate owners.

The measurement is computed once per online encode and stored through the real
S policy dock. Cache admission checks the identical current-state, instruction-
reference and target-binding sources. K permutation updates evidence and binding
together; camera/feature charts are validated. ODE calls only interpret this
read-only evidence; no recurrent task memory, extra DINO/W pass, extra solver
step or future-target dependence is added. Consumers quarantine unavailable
values before learned operations. Existing optimizer groups include all new
parameters; no external loss fabricates progress labels.

**Limits:** these are learned soft visual comparisons and proprioceptive feature
differences, not calibrated world displacement, goal completion, contact or
persistent object identity. Ego-camera motion and ambiguous objects remain real
limitations. The comparison is not a W endpoint prediction error; W continues
to supply interval statistics. Desired-task-relation supervision, physical
association and complete transition/bottom review remain open. Extra value
projections run per P3 call; production GPU cost has not been measured.

## M6g matched one-step proprioceptive response

`top.robot_feedback_mode=one_step_proprioceptive_v1` selects an independent
robot-response observer inside the P3 compiler owner. It requires timestamped
CALVIN rotation-column state features, typed-horizon P3, and a positive explicit
`objectives.robot_response` (candidate 0.1). Legacy modes allocate no observer
parameters and omit default mode/objective fields from serialized graphs.

The dataset and the actual online adapter supply the exact adjacent pair
`o[t-1], a[t-1] -> o[t]`. The source of `a[t-1]` is the recorded controller
command, NOT a generated row that was never executed. Dataset row boundaries
are real source rows; an instruction start is not a physical reset. CausalHistory
already retains nine observations: the new scalar-state predecessor uses that
storage, without another RGB/DINO encoding or additional history capacity.
Reset has no pair. Missing predecessor in a non-reset online snapshot fails
closed; it is never substituted by the nearest sparse state (-4/-8).

`ExecutedRobotStep` keeps actual relative offsets (-1,-1,0) and an observed
mask. Source admission checks exact clocks and equality to the last executed
command. Action-history condition dropout masks this separate path too, so it
cannot bypass dropout. Unknown payload is quarantined BEFORE learned operations.

The response MLP predicts the one-step CHANGE in the configured state-feature
chart using only detached prior state and recorded command. A masked observed
response MSE supervises it directly from the now-available current observation.
This loss is in the representation group; it needs no future Teacher label.
The predictor cannot access current state, G, S, instruction, candidate actions,
future observations or a task-success flag as its predictor INPUT. Current state
is used only as the now-observed response target and difference after prediction.

The feedback is `(current - previous) - predicted_delta`, detached before the
policy read. Thus action loss cannot train the predictor to manufacture task
features disguised as an error. P3's separately parameterized error read IS
trained by the action path; it modulates horizon context and joins the existing
observed-change innovation lane before its RMS bound. It does not replace the
protected facts/precision, alter W, or add an optional-source routing gate.
Zero innovation gives exact zero contribution and zero context derivative.

The observer runs once per online encode, never once per ODE node/refinement.
Its response loss stays in OnlineTrainingState; cached feedback is detached and
paired by source identity with the same current history and executed step.
The two ODE passes reuse it read-only. No mutable cross-decision state is added.
Expectations are recomputed under the current model from past inputs, not falsely
claimed to be a previously issued forecast with another parameter version.

**Limits:** this is model innovation in proprioceptive feature space, not a
calibrated controller tracking error, physical twist, force, contact truth,
world-coordinate object motion or task progress. Feature rotation differences
are chord differences, not angular velocity. One control step has no assumed
seconds-per-step. Sparse W interval statistics remain separate and never become
one-step endpoints. Lack of response can reflect dynamics or model error; the
observer does not decide that a grasp/push succeeded. Object-matched feedback,
expected task relations, S progress and deeper transition/bottom review remain
open. No demonstration command is asserted to be measured actuator torque.

Config, normalizer identity, source closure, cache consumers and deployment ABI
include this mode. No silent weight/optimizer migration. This module does not
add a W materialization, DINO pass or ODE solve, but adds one small response MLP
per encode and a per-P3 dynamic error read; GPU cost is not measured.

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
to normalized xy once. The RGB producer's confidence and occlusion are instead
FORWARD-flow quantities on the EARLIER source image. Read these after each
inverse step at the updated earlier-frame coordinate; reading all fields on
the later chart is wrong. The status chart is explicit in the history record
and deployment ABI. They remain predicted features, not observed support.
Earlier coordinates compose the two adjacent maps at
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

The historical default and the accumulated structural candidate are different
explicit graphs. Do not read legacy scalar constants below as overriding a
serialized candidate selection. Neither graph is promoted by passing unit tests.

| Boundary | Historical default/control | Cumulative candidate through M7c |
|---|---|---|
| Configuration | `object_intent_dynamics_323.json` | `structural_rebuild_m6n_calvin.json` |
| Topology | G1/G2/G3, S, W1/W2, P1/P2/P3 | Same responsibilities with versioned typed interfaces |
| Physical future grid | `legacy_48_v1`: 4–8, 8–16, 16–32, 32–48 | `control_aligned_24_v1`: action [0,4), [4,8), [8,16), [16,24); successor supports (lo,hi] |
| W action ownership | Means or clipped 24-row prefixes under selected control | Candidate prefixes 4/8/16/24; independent matched observed-action W supervision |
| Target identity | Legacy independent reads or explicit shared prior | One soft K+null operation binding across S/coarse/P1/P2 |
| Spatial evidence | Historical local 7×7 candidates | Declared full current-image support; P1 real 3×3 reads at candidate positions |
| History | Historical paired-row control | Source-timestamped state/control streams and source masks |
| Training | Historical one-velocity graph | One causal G/S encoding; separate supervised W; interior velocity plus clean endpoint head forward on the SAME online cache |
| Deployment | Proposal, one W rebuild, refined pass | Same lifecycle; physical clocks and current observation are immutable through ODE |
| Native command | Outlet-owned chart | Confirmed applied command owns recording/history, not an unexecuted proposal |
| Execution phase | Historical restored control | Engine/eval/deployment use completed-update count; no task clock |
| Shared action field | 24 × 18 value/difference/gripper chart | Unchanged; native CALVIN 7-D command and model 10-D state remain distinct |
| Working branch | Historical branches are controls | `codex/structural-rebuild-20260921` only |

The default remains schema-30 recovery with explicit outlet adaptations and
optional schema-31 B-spine controls. Registered layout and component selection
remain `clearvla_mainline` layout 2 and `mainline-modular-v1`. Camera roles,
source history times, future control time and numerical solver time are not
interchangeable. Configuration, manifest and deployment ABI are authoritative;
never reinterpret a legacy checkpoint as the accumulated candidate by relabeling
its tensors. Pen/RDT/LIBERO defaults are not implicitly changed.

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

For reader-local configurations, the following reader arrangements remain
the control graph. M5a supersedes their independent target selections only
when explicitly selected. S is the sole intent owner. W cannot read goal tokens, S values, coarse hidden
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

1. Camera, spatial, local-M, global-K, declared candidate-support N, interval,
   horizon, basis and type axes stay real until a named consumer. N=49 belongs
   to the legacy local chart; full-support candidates keep their declared chart.
   Reduced evidence cannot be recreated with expand and called original evidence.
2. Online evidence keeps ordinary autograd unless this contract names a
   no-grad Teacher or audit scope. Activation gradients do not substitute for
   parameter-owner VJP.
3. Learned flow is a continuous source-relative prior, never a forced-nonzero
   target or route quota.
4. S owns intent. It cannot manufacture W support/value or enter W through a
   hidden second path.
5. W owns candidate-action physical-world prediction. S may predict explicitly
   typed demonstration outcomes, not another candidate-action rollout. CT
   conditions decoder features and cannot claim physical world/error semantics.
6. Physical validity and camera support are producer-owned. Confidence or
   allocation share cannot silently replace them. Every S/coarse/W K read
   receives that support; invalid rows are quarantined before normalization
   and object-axis reduction, and an all-invalid row has a finite zero-output
   fallback.
7. P1 retains the selected support chart and its real 3x3 local detail reads
   until factual selection. Full-support mode reads local regions at each
   candidate position before posterior averaging, not at an average location.
8. Semantic and geometry values have separate owners. Legacy controls may use
   independent K/K*C posteriors; shared-operation mode MUST keep the common K
   marginal and null mass, allowing geometry to choose a view within each K.
   Types do not compete in one softmax; physical interval selection has no null.
9. Neutral P2 is algebraically neutral: zero effect and interaction leave the
   protected factual consequence unchanged.
10. The V120 seed, terminal contracts, CVAE/workspace, Evidence MMDiT,
    capacity and execution paths remain present.
11. The shared 18-D value/adjacent-difference action field and cumulative
    decode cannot be replaced for one outlet. Outlet-native conversions stay
    at the adapter/finalizer boundary.
12. Future observation/action/state belongs to training-only targets, Teacher,
    matched observed-action W or clean head supervision; none can populate
    the online candidate cache. At fixed weights/online input/noise, changing
    future labels cannot change deployed action. Training updates may of course
    change later weights; this is not a ban on supervised learning.
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
17. Online sequence-conditioned W reads only the actual candidate control
    prefix under its declared FutureTimeGrid and CandidateControlDomain. In
    aligned mode endpoints are 4/8/16/24. Legacy 8/16/24/24 clipping is a control,
    not an aligned forecast. Unknown controls cannot become synthetic zeros.
    Observed-action training branches use separately typed matched labels and
    never replace online W or the one-rebuild Q5 lifecycle.

## Typed boundary summary

| Boundary | Required semantics |
|---|---|
| OnlinePolicyInput | Causal RGB/DINO, state, executed history and language only |
| FutureSupervision | Disjoint training-only future action/state/DINO evidence |
| ObjectFactSet | K=4 physical objects plus explicit null and observable object/camera mass |
| ActionIntentDock | Public S interval/history/K memory plus producer-owned K validity mask; no typed fact re-entry |
| PolicyIntentDock | Typed task/context with shared operation binding in the cumulative mode; observed change and expected outcome remain separate |
| PhysicalActionCondition | Four physical interval means plus current-anchored deltas |
| PhysicalActionSequenceCondition | Exact known 24-row source action, canonical value/delta, current boundary, row times and FP32 outlet/chart/normalizer identity |
| ObjectWorldBelief | Current G facts with versioned observed robot/view conditions; no S/Teacher/noisy action |
| CandidateWorld | One exact action condition atomically paired with FutureObjectDynamics |
| FutureObjectDynamics | Semantic successor/delta and camera-resolved transport/covariance |
| FactualPrecisionDock | Completed protected P1 detail; no new reader or compression |
| CompletedP1PolicyState | Static factual base separate from dynamic action/time residual |
| SelectedIntervalEvidence | Interval-retaining semantic/geometry values with no-null physical terminal |
| ControlledTransitionSource | Completed current G3 feature chart, built once per observation; NOT a physical rollout |

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

## M5a one operated-object law per observation

`top.target_binding_mode=shared_operation_v1` is an explicit new graph, selected
by `configs/mainline/structural_rebuild_m5a_calvin.json`. It replaces the
additive `shared_target_prior_v1`; selecting both is an error. The default
`reader_local_v1` omits this field from old configuration serialization and
retains the original reader modules. Old checkpoints are not silently migrated.

S computes one differentiable FP32 normalized K-plus-null distribution from
language, causal timestamped history and observed entity values. Physical
source support and learned target confidence are separate. The binding refers
to the current G entity slots, not persistent physical IDs or known colors.
The same binding object flows through S, coarse, P1 and P2 and is reused for
proposal/refined sampling and all ODE evaluations. There is no task clock or
recurrent update inside numerical time. Every new observation rebinds.

Task-independent current target evidence retains four distinct attribute
values and one token per declared camera. Camera tokens keep image coordinates,
view identity and observed robot state separate before learned projection; no
pixel/world subtraction or calibrated 3D claim is made. Each reader attends
INSIDE each object over multiple attribute/view tokens, then multiplies by the
shared object mass. It does not run its own K softmax. Multiple keys retain real
Q/K derivatives; a singleton attention with constant softmax is not used.
Target mass is applied after the complete internal read without normalization
by total real mass. The bias-free value path cannot invent target evidence at
full null. Smooth RMS contraction is not a target attention quota. Null means
no target evidence, not an environment stop command.

P1 adds its target-specific current evidence after the clean-basis residual
subtraction. It therefore cannot cancel merely because the same target was
written to both sides of that subtraction. This lane does not require nonzero
W motion or future labels. Existing high-resolution scene detail remains a
separate context lane; M7 still owns the deeper target-local spatial read.
The actual P1 reader belongs to the `operated_target_reader` logical namespace
and `p1_factual` optimizer group, with ordinary checkpoint parameter ownership.
No uncaptured parameters or optimizer-registration exception is admitted.
The inherited W output heads initialize at zero, so coarse may initially have
zero policy-only VJP through W. Its separately defined supervised action loss
is responsible for initial updates; this is distinct from parameter ownership.
M6 must revisit action-matched supervision without assuming immediate W VJP.

P2's semantic object marginal is exactly the shared target mass. Geometric
selection factors into that mass times a camera distribution conditioned INSIDE
each object. Different action queries can choose views/intervals, not redefine
which K is operated. A source without camera support contributes no fabricated
geometry. Typed operation gates vary across future intervals and property
kinds but no longer give each kind an independent operated-object identity.
Source-invalid W/intent payload is quarantined before projections and interval
common/residual decomposition. Evaluation-only identity neutralization preserves
real-versus-null mass. Exact full-null evidence and zero predicted effects
remain zero even though unrelated proprioception/scene context may remain.

S's recent observed motion is projected per view before camera conditioning
and target marginalization. View identity alone cannot manufacture nonzero
motion when displacement is zero. This is recent target-related motion, NOT
cumulative task progress: an instruction-start reference, endpoint objectives,
future-recognizer redesign and actual executed-prefix feedback remain open.
Raw multi-view future transport in P2 still needs M7's deeper chart review.

Deployment ABI carries the exact operated-target contract, source dependency
identity includes `target_binding.py`, and incompatible metadata/resume config
is rejected. Focused tests run real G/S/W/P/bottom, optimizer, ordinary
checkpoint and deployment loading. They establish connectivity, null/mask
semantics and numerical identity, not useful learned grounding or robot success.

## M4d current-entity motion law

`top.entity_motion_mode=current_entity_support_v1` reads estimated current-minus-
previous displacement at the same current-image entity measure used for camera
centers and reverse lookup. The latest inverse map is indexed by current cells;
its negative is integrated under each K/camera conditional law. Conditioning is
performed in FP32 log space before exponentiation. No query-anchor flow, mean
position lookup or cross-camera vector average enters this G3 export. Source
clock gaps remain the sole conversion to per-control-step rate; a missing pair
is zero with zero duration, not an observed static scene. Confidence and
occlusion cannot change current validity or reweight the spatial entity law.
An estimated inverse location outside the older image remains an estimate, not
an admitted historical correspondence. The finite causal evidence path separately
checks correspondence coverage.

The corrected M4c status reader samples forward confidence/occlusion on earlier
source cells after each inverse step. Its explicit metadata rejects first-M4c
chart assumptions on exact deployment loading. Old query-anchor motion is an
explicit control, not a fallback. The motion selection adds no parameters,
encoder calls, W builds, ODE evaluations or mutable physical state. S/W receive
the ordinary exported entity motion through existing consumers; their current
cross-camera summaries are NOT certified as common-world vectors and remain
part of their deeper review. This unit does not establish physical object IDs.


## M5b causal instruction-start observation and current relation

Explicit candidate: `structural_rebuild_m5b_calvin.json`, with
`top.instruction_reference_mode=instruction_start_observation_v1` and M5a's
shared operation binding. Omitted mode remains `none`; undeclared references
are rejected. This unit is admitted only for the source-timed CALVIN profile,
whose instruction origin is actually available. It does not guess origins for
other datasets or retrofit old checkpoint identities.

`InstructionReference` is an online causal input, not future supervision. It
owns frozen observed DINO tokens per camera, the same ten-dimensional robot
state feature chart, a Boolean source support, and a nonnegative physical age.
Age is admission/provenance only: no numerical age, episode fraction, timer or
oracle progress enters the neural reader. The real dataset uses
`source_start-context_start`, never the sampled window center or the causal
prefix start. The reference may legitimately be older than the rolling history.
The cached loader groups its read with existing requests but explicitly splits
reference rows from all 12 future-support rows. Missing declared origins and
centers earlier than the instruction are errors, not fabricated start labels.

The online policy owns one reference per instruction. Episode reset clears it;
`begin_instruction(text)` explicitly starts a new instance, including repeated
identical text, without resetting the action RNG. Text changes also start a new
instance at the current real observation. Callers must signal a repeated same-
text instruction or reset; elapsed steps do not infer an instruction boundary.
The reference uses the existing current DINO encode, not another encoder run.
Returned diagnostic/input tensors are owned copies, not aliases through which
external callers can mutate the stored anchor. Time rollback without reset is
rejected. Weight checkpoint restore is supported; an in-flight episode anchor
is not a secretly serialized checkpoint state. A new runtime must reset/start.

S uses a current entity/task query to read both current and start patch charts
with the same key/value maps. No old K index is compared to the current K index:
no persistent IDs are invented. Coordinates are values within each view, not
cross-world coordinates or location-only matching keys. Visual difference is
zero for identical images under the same query. Full match distributions are
retained until value reads; entropy, source availability and patch coverage are
separate status features. An unavailable reference is not encoded as confident
zero motion. Per-view features receive a role-aware map before view reduction.
The shared operated-object mass is applied after these reads without dividing
by real-versus-null mass. Full null contributes no reference context, which is
not a robot stop command. Robot feature-state change is a separate same-chart
value; it is never subtracted from an image coordinate.

The compact reference context enters the actual S interval source before its
self block. Real coarse, factual and policy consumers subsequently use those
intervals. Comparison happens once in `encode_online`, not on W rebuilds or ODE
nodes. Dense start/current correspondence remains outside the compact numerical
cache. New parameters belong to the existing intent optimizer/ownership group;
ordinary training and checkpoint paths update/save/restore them.

Cached DINO is FP16 storage; raw G's learned projections cannot assume that this
is their parameter dtype. The observation compiler now casts source values once
to its neural parameter dtype without mutating the caller. FP32 source/model is
a no-op; autocast still controls operations. New reference reads make the same
storage/compute distinction. Future Teacher targets remain observed targets.
The source identity records the change; ABI records the exact instruction
reference chart. Failed numerical or type checks are never bypassed by a new
fallback mode.

This is a causal comparison representation, not a calibrated 3D displacement,
a learned completion certificate, a phase controller or persistent tracking.
Richer desired-relation supervision, FuturePlanRecognizer ownership, matched-
action W, deeper view-aware P geometry and actual execution error feedback are
still separate work. Source-only tests use declared external transport fixtures
and real G/S/W/P/bottom, optimizer and checkpoint implementations.

## Diagnostic noninterference in intent and coarse readers

The S/coarse attention value path does not depend on logging. `_CrossRead`
and the S state-change read always execute with `need_weights=False`.
Optional weights are detached FP32 pre-dropout head means computed from the
same normalized Q/K and packed bias-free projection, with autocast disabled.
This observer supports only the actual zero-dropout reader topology and
rejects incompatible variants instead of reporting invented attention.
All-invalid keys and no-observed-rate sentinels produce zero diagnostic mass.
The observer does not write cache/state, own losses, consume random numbers,
or change parameter registration. Attention/value gradient parity is checked
with the same autograd mode on both sides; no cross-kernel bit-equivalence or
learned-behavior guarantee is implied.

## M6a action-matched world supervision (explicit candidate)

`top.world_supervision_mode=matched_observed_sequence_v1` keeps the online
24-row candidate world unchanged, and reuses current G facts in one additional
training-only W1/W2 pass. The outlet adapter canonicalizes actual labelled
controls through the 48-step world horizon. The same 24-row chart factory is
composed across blocks: relative arm displacement carries its prior cumulative
value, absolute-action and gripper deltas retain the previous real boundary.
Time remains measured on the online 24-control-step scale, never renormalized
to make a 48-step observation look like a 24-step candidate.

`ObservedActionSequenceCondition` and `SupervisedWorld` are distinct from
`WorldActionCondition` and `CandidateWorld`. Online caches, the ordinary world
factory, refinement and the outlet adapter reject the training-only condition.
True future controls are detached labels and never replace online coarse
controls, candidate worlds, noisy flow actions or observation caches. No
second G/S encoding, extra deployment W pass or new learned parameter is added.

For supervision, prefixes end at 8/16/32/48, matching their actual future
intervals; online prefixes remain 8/16/24/24 under the existing policy ABI.
Unavailable labels are not zero commands: quarantine precedes chart arithmetic,
masked rows do not advance W recurrence, and a target interval is admitted only
when every preceding control is known AND its visual/state support exists.
The source support must be a contiguous prefix; gaps are rejected rather than
imputed. S/recognizer/coarse target masks retain their separate ownership.

Only the configured future-dynamics loss consumes the supervised prediction.
The action loss still consumes the candidate-conditioned online graph. A
matched-mode loss missing its supervised world is an error; a legacy loss
receiving one is also an error. Diagnostic names distinguish the two W paths.
The learning contract is serialized in config and deployment metadata; the
default omission preserves legacy identity. Old weight/optimizer migration
is not added. This mode requires fresh training or its own exact resume.

This fixes action/target pairing, not every limitation of W. The online
25-48-step control continuation is still unspecified under the legacy 24-row
proposal: long-range matched supervision alone does not resolve that train/
deployment conditioning mismatch or justify far-horizon planning claims. M6
still owns control-horizon alignment, robot-object physical relationships and
Teacher association quality. M4c2, remaining S objectives and P redesign remain
open. Full-stage CI and learned behavior are not implied by source preservation.


## M6b candidate control domain (explicit candidate)

`top.world_control_mode=known_prefix_v1` is selected by
`configs/mainline/structural_rebuild_m6b_calvin.json`. It requires sequence
physical controls and matched observed-action W supervision. The old mode is
omitted in serialized legacy configs and remains a reproducible control.

W constructs an immutable `CandidateControlDomain` from the actual candidate
length; it accompanies W1's working state and W2's final prediction. It is NOT
pixel validity, target probability, predicted uncertainty, or a learned gate.
CandidateWorld checks agreement with its exact action tag; policy caches and P2
reject a missing or unconfigured domain. No host read of a CUDA value is added
at numerical nodes. Object permutations retain the same physical time domain.

The existing interval endpoints are 8/16/32/48. A known 24-step proposal fully
covers the first two, not the last two; no zero-filled continuation is invented.
P2 quarantines unknown semantic/transport/covariance values BEFORE nonlinear
keys, geometry scores, and common/residual decomposition. Common effects use
only admitted intervals. Spatial and terminal posteriors exclude unknown
intervals; shared target mass and null semantics are preserved. An empty
domain yields no effect, not a confident prediction of zero motion, and leaves
the factual P1 path intact. Current observation masks/coordinates are unchanged.

Raw extrapolated fields remain explicitly non-admitted diagnostics. They cannot
enter P3 or the bottom through P2; changing them, even to NaN, leaves the actual
same-noise sampled action unchanged after a real training update. Training-only
SupervisedWorld has its own per-sample observed-control support and no candidate
domain; real 32/48-step supervision still trains W2. No parameters, W passes,
ODE passes, ground-truth deployment inputs or recurrent state are added.

This is an honest boundary, not full-horizon planning: the old 16..32 interval
cannot supply a separately supervised 16..24 prediction. A later time-grid or
explicit candidate-continuation redesign must address that granularity, with
matching Teacher/S/data/ABI changes. Do not relabel the 32-step prediction as
24 steps. Robot-object relation conditioning and remaining M6/M7/M8 work are
still open. Full milestone/GPU/behavior acceptance is separate from source save.


## M6c observed robot/object/view relation in W

`top.world_robot_condition_mode=observed_state_views_v1` is explicit in
`structural_rebuild_m6c_calvin.json`. It requires the current entity chart,
source-timed motion, named camera geometry and known candidate-control domain.
The legacy mode `implicit_g_only_v1` remains omitted in serialized configs.
This is a new graph identity, not an automatic checkpoint migration.

G exports a compact ObjectWorldBelief. Its optional RobotWorldObservation holds
only the SAME authoritative current history.state tensor and its feature-chart
identity: normalized xyz, rotation-column features and opening in the CALVIN
candidate. It is not a candidate action, language value, future state, contact
oracle or instruction-start state. Candidate W, observed-control supervision W,
and outer refinement use the same belief. Training and policy caches check the
configured chart and current-source identity; W1/W2 reject stale relation reuse.
The deployed compact belief still excludes dense visual/history/target banks.

RobotObjectRelationEncoder combines robot state, each object's current content,
each named camera's image coordinates, its observed image velocity and an
explicit positive-flow-gap flag. Missing views are quarantined before learned
projections/products. At reset a current image remains valid but motion is
unobserved, not positive evidence of zero velocity. Coordinates are never
subtracted from proprioceptive xyz. These are learned relational FEATURES, not
calibrated 3D object poses, metric distances or inferred contact truth.

A nonlinear per-view relation (including robot/object-view interaction) is
computed BEFORE any camera reduction. Conditional source-supported weights pool
features, not raw 2D motion vectors. The generic W base consumes those features
instead of the camera-averaged raw displacement/rate. Per-view relation also
conditions the existing typed geometric carriers, including covariance. The
existing source validity and independent observation loss targets do not change.
No goal relevance enters W; shared target selection stays in S/P.

Relation encoding runs once for each W materialization and is reused by W1,
W2, mean and covariance branches. It is private working state, not a mutable
ODE memory. Numerical solver nodes consume the final field; the number of W,
ODE and G/S evaluations is unchanged from M6a/b. New parameters are owned by
the existing dynamics optimizer/checkpoint group, with a new explicit config
and deployment ABI. Unknown horizons remain excluded under M6b, while real
48-step observed controls retain their separate supervised W2 learning.

This repairs physical conditioning, not physical correctness by construction.
A learned encoder may still rely on shortcuts. Lack of calibrated 3D/contact
state, physical cross-window entity association, 16..24 control-grid resolution,
S desired-operation supervision and P3 executed-prefix feedback remain open.
Do not claim successful manipulation or whole-M6 acceptance from these checks.


## M6d physical control-aligned future clock

Explicit `top.future_time_grid_mode=control_aligned_24_v1` is selected by
`structural_rebuild_m6d_calvin.json`, with six visual supports and the M6a/b/c
contracts. `legacy_48_v1` remains the default omitted from old serialization;
old experiments are not silently relabelled. New training/checkpoint identity
includes the exact grid, target convention and manifest intervals.

`FutureTimeGrid` is the authoritative physical-step resolver. The four control
intervals are [0,4), [4,8), [8,16), [16,24). Given o[t] before a[t], target
successor observations are (0,4], (4,8], (8,16], (16,24]. Dense action/state
summaries use rows 0:4, 4:8, 8:16, 16:24, with state row j representing
observed o[t+j+1]. Sparse visual supports are exactly 4,8,12,16,20,24. A visual
boundary belongs to one interval, never both. Teacher targets are means over
these declared sparse observed supports: not dense-time integrals, and not
endpoint states for the last two intervals. Missing supports invalidate the
whole affected interval; no nearest-frame substitute or shortened target is
permitted in this mode. Policy row-zero labels survive incomplete world labels.

W1 owns intervals 0/1; W2 owns 2/3. Both candidate and observed-supervision W use
causal control prefixes ending 4/8/16/24, with the existing 24-step action-time
unit. The real dataset produces 24 future action/state rows and six visual
supports; preflight explicitly checks this clock instead of assuming 48 rows.
All four candidate intervals are fully controlled and admitted in P2. The
source-owned control-domain quarantine remains in place and is not replaced
by a predicted confidence or an all-true hard-coded mask.

S and the training-only recognizer receive deterministic physical endpoint
codes in addition to their learned interval features. Coarse row queries
interpolate knots at physical centers 2/6/12/20 rather than treating unequal
intervals as uniformly spaced. Learned 24-row detail remains. W interval and
P2 terminal keys use the same endpoint code; P3 consumes S's matching grid.
This is relative FUTURE CONTROL time, not elapsed task phase or numerical ODE
time. The extracted visual AND action decoder profiles carry matching window
endpoints, interval boundaries and future-support offsets. Unused historical
V120 graph code does not define the current Teacher target convention.

Typed S/docks, W fields, supervised controls and preselected P2 evidence retain
clock identity. Cache, loss, Teacher/recognizer, P2 spatial/terminal, P3 and ABI
boundaries reject same-shaped evidence from a different clock. No new trainable
parameters, no extra W/G/S evaluations, no new ODE steps or persistent state
are introduced. Legacy-mode parameters/buffers and same-noise output remain a
separate exact control, not a compatibility promise for the new graph.

This chooses a complete current-action support over an untrained artificial
48-step continuation. The candidate does not claim controlled forecasts after
24 steps; longer task intent/reference remains S's responsibility. W2 is now
usable for 8..24 control consequences, not a relabelled 32/48 prediction. W2's
latent dependency is still not a compositional physical simulator. Precise
contact/support relations, persistent physical entity identity, S's desired
operation supervision, P2 camera-aware downstream value design and P3 matched
executed-prefix feedback remain open. Source checks are not robot success.


## M6e named-view W consequence read

`top.p2_geometry_mode=view_conditioned_transport_v1` in the explicit M6e config
requires the shared target, observed W views and aligned 24-step clock. Legacy
`pooled_transport_v1` remains omitted from old serialization. W now exports its
actual ordered camera names; P2 and caches reject a missing/relabelled chart.
No camera role is inferred from an axis number or an object's colour.

P2 forms context from each camera's current normalized image xy, predicted
within-support correspondence covariance and canonical named role. The existing
transport/key projections run on [B,I,K,C,2] BEFORE spatial pooling. A learned
context-dependent linear interaction interprets each displacement locally:
`E_c = L(d_c) + J(L(d_c) * tanh(context_c))`. All terms vanish for zero displacement;
position, role and covariance alone cannot create motion effects. Displacement
VJP remains available at zero. A learned named-role modulation also makes each
image's spatial query distinct; it is not a calibrated robot projection.

Shared K mass (including null) is applied AFTER view-internal conditional reads.
P2 can select an informative camera inside a target, not pick a different target
or renormalize away null. Unknown-control and source-invalid payload is masked
before context, projection, nonlinear keys or common/residual decomposition.
Feature-space fusion can still learn cancellations; the structure removes the
mandatory cancellation of raw vectors expressed in unrelated camera charts.

SelectedIntervalEvidence carries its geometry mode: legacy xy has width 2,
new geometry features have width H. The temporal terminal rejects incompatible
semantics, uses the feature value directly and does not project it a second
time. P3/consequence consume hidden effects, NOT physical coordinates. New
metrics label feature norms as features, not physical displacement. The ABI
records camera roles, input units, target statistic, query and value semantics.
The new parameters belong to the existing P2 optimizer group/source closure.

W transport remains the uniform mean at declared sparse successor supports;
covariance remains the mean within-support correspondence covariance. Neither
is a single-step endpoint, total temporal mixture variance, calibrated confidence
or a task-risk score. No observed-target, loss, W call count, ODE step count or
persistent state is changed. P3 executed-prefix comparison is still open and
cannot subtract these interval features from a newly observed physical state.


## M6f typed P3 horizon coordination (partial P3 review)

`top.p3_coordination_mode=typed_horizon_v1` in the M6f candidate replaces the
pointwise P3 internals, not the P2 producer or the protected bottom ingress.
It requires the explicit aligned 24-step grid, shared target binding and named
view geometry mode. The old omitted `pointwise_legacy_v1` selection is unchanged.
New graphs/checkpoints are explicitly different; no partial weight migration.

`P3HorizonContext` carries the original action query, current P1 facts, dynamic
P1 precision, semantic effect plus its interaction, geometry FEATURE effect plus
its interaction, S temporal plan and observed-change summary. Producer names,
axes and time identity are retained before separate source projections. Typed
P3 never receives an already summed action-plus-protected-consequence as its
original action query. Both modular and top composition paths implement this.
Semantic and geometry are projected before fusion rather than being compulsorily
summed in the original hidden basis. This removes a structural cancellation,
not a guarantee that learned projected values never cancel.

One attention read coordinates all T*Q proposed plan features without averaging
away either the action row or basis axis. Bidirectional attention is over the
currently imagined plan, not future observed data. The W control-prefix causal
contract is unchanged. Physical row midpoints `(j+0.5)/24` supply deterministic
Q/K-only time codes; this is not task elapsed time, an endpoint prediction or ODE
time. Q/K are normalized, while raw projected value amplitudes are retained. A
local term retains row-specific information alongside the contextual attention
read. No dropout, mutable cross-node cache or persistent task state is added.

The same two optional delta outputs enter the existing role bank. The original
protected consequence and dynamic policy precision return by tensor reference;
there is no additional exact copy or pseudo-owner in that bank. Their content
may condition the learned innovation, just as an ordinary residual block reads
its input. There are no inactive legacy P3 projections in the new graph. All
new weights belong to the existing `p3_compiler` parameter group.

The observed-change lane is a bias-free value projection modulated by the
coordinated plan; zero S change yields exactly zero value and zero gradient to
plan context, while its derivative to an observed-change input can remain
nonzero. Its meaning remains S's observable-change feature. It is NOT a measured
contact signal, task success, or a comparison with a previous predicted world.
True executed-prefix feedback still requires matched future endpoints, actual
executed commands, observed-time alignment and entity association. Interval-mean
W outputs must not be reused as one-step errors.

Configuration, deployment metadata/source closure and exact reload carry this
selection. Attention always uses `need_weights=False`, including diagnostics.
A source save is not supported-runtime/whole-M6/P3/robot acceptance. The new
P3 has 14*H^2 weights vs 6*H^2 in the legacy pointwise compiler; at H=512 the net
increase is 2,097,152 weights. One contextual P3 attention is evaluated per
EXISTING dynamic policy call; no W/ODE count changes, but GPU latency/memory is
not yet measured. Review this cost, long-horizon conditioning, S progress and
bottom role routing at their actual consumers rather than declaring them closed.


## M7k: numeric source admission precedes command equality

The adjacent-command preflight agreement must run after the supported-value
finite audit. Nonfinite history must retain its named numeric diagnostic, not
be mistaken for disagreement merely because NaN is unequal to itself.
Temporal/observability validation remains earlier, the response record retains
its own strict finite-payload check, and no values are repaired or reclassified
as unobserved. This corrects a regression found by the cumulative M7j inventory.
