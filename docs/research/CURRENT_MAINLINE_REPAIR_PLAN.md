# ClearVLA Schema30 / Schema28-core recovery execution and closure plan

Status: the atomic Schema30 modular owner rewrite and the separately developed
B-spine-0 source are integrated in one local tree. The modular path still
matches its captured baseline at zero tolerance, and the combined structural,
runtime, checkpoint, interface and B-spine suites pass. Real CUDA/BF16 and
read-only production-checkpoint replay remain open; do not start the fresh Pen
training run until those remote gates close.
Updated: 2026-09-04

This plan contains only the remaining execution sequence. Architecture truth is
in [`00_CURRENT_ARCHITECTURE_CONTRACT.md`](00_CURRENT_ARCHITECTURE_CONTRACT.md);
open questions are in
[`CURRENT_MAINLINE_ISSUES.md`](CURRENT_MAINLINE_ISSUES.md); live PIDs, run tags
and last observed steps are in
[`auxiliary/ACTIVE_MAINLINE_HANDOFF.md`](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
The completed Schema25-to-Schema30 implementation narrative is recoverable from
Git commit `f60bd80` and the replay archive. The historical remote curves are
comparison evidence only.

## Objective

The historical plan used one core source commit and two experimental outlets.
The current recovery pass uses one fresh Pen formal run to close the disputed
core behavior. The existing RDT-8 Schema29/cachefix run remains a separate
adapter/interface outlet; do not merge its behavior with the recovery result.

| Outlet | What it decides | What it does not decide |
|---|---|---|
| Pen single-task | far horizon, continuous gripper, W/P2/consequence/CT closure, final refinement mismatch and gradient/optimizer health | cross-task or richer camera/action adapter behavior |
| RDT-8 multitask | dataset/language/camera/action ABI, balanced sampling, per-task behavior and shared-core competition | native three-camera, depth or full 14-D bimanual modeling |

Any future recovery-comparison pair may differ only in
dataset/adapter/task profile and declared normalizers. It must serialize the
same capability, manifest, source digest, optimizer ownership and loss ledger.
Component experiments described below are new source units and must instead
serialize their complete component selection and compatibility ABI. The current
Schema29/cachefix RDT run is explicitly not a recovery pair.

## Recovery scope (2026-09-03)

Keep only the source changes whose evidence is unambiguous: the proven
`d8a77a1` AMP/BF16 cache/VJP isolation, data/split/normalizer/identity and
checkpoint/logging infrastructure, compiler and adapter cardinality fixes, and
the Schema28-core restoration of the training call graph, S/W/camera semantics,
and the profile-owned non-core gripper codec boundary. Pen uses current
action-state and RDT uses previous command consistently through all continuous
gripper codec consumers; arm/network conditioning remains current-state based.
The historical detached train-time self-conditioning and other Schema30
semantic deltas are excluded. The manifest schema stays 30 only to keep the
capability namespace stable; the recovery/boundary ABI suffix rejects stale
checkpoints. Formal commit
`0973f1920cb8467e3b5f048aaaa180e34e27c83f` started fresh; no anchor checkpoint
was resumed or migrated.

## Research decision (2026-09-03): one numerical-view hypothesis

The source review and the external spline literature support one narrow
hypothesis, not a broad architecture rewrite:

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Active source | the bottom receives a 24-row, 18-channel noisy physical field and currently lifts each row before the MMDiT blocks | that the current per-row lift is disconnected or that a spline should replace it |
| [B-spline Policy](https://arxiv.org/abs/2607.09648) / [Spline Policy](https://arxiv.org/abs/2606.07386) | fixed control points can expose local temporal support, continuous resampling and a coarse trajectory shape | that a hidden bottom spine should become the action ABI, a learned knot system, or an extra inference loop |
| Historical ClearVLA spline commits | there is basis/evaluation code and a record of output-writer failure modes | that the old output head or query-direction writer is safe to restore in the active graph |

The resulting candidate is **B-spine-0**: a fixed, bottom-internal numerical
view of the deployed noisy field. It is not a new action head, a smoothing
objective, a top-level carrier, or Minimal Iterative Policy. The original
full-resolution path remains the primary path; the candidate is allowed to
survive only if it adds an independently useful temporal view without erasing
detail or changing the action ABI.

This remains a provisional **Schema31** source unit. The recovery graph is not
a Schema31 candidate and must not be used as a migration checkpoint. Create the
Schema31 manifest/source identity only after recovery E8 closes, the atomic
Schema30 modular rewrite passes behavior equivalence, Gate A freezes one basis
and Gates B-C pass. No Schema31 training run is currently authorized.

The high-information questions are deliberately separated:

| Question | Single selecting observation | Falsifier |
|---|---|---|
| Is a temporal chart present in the field? | cross-outlet coarse/detail energy and locality table | no stable split beyond the DCT/identity controls |
| Does a learned view reach the action path? | matched `spine_zero` action delta and MSE change | zero delta, or removal improves MSE |
| Does it help the intended failure? | complete-curve far/gripper result with near/arm guard | far/gripper regression or a near/arm trade |
| Does it preserve sharp information? | event/post-event/hold residuals plus detail share | apparent gain comes from suppressing event/detail rows |

These four questions are answered by one offline traversal and one paired
training run; there is no degree/control-count training sweep.

## Current phase: finish the Pen recovery curve

The formal Pen run started fresh from commit
`0973f1920cb8467e3b5f048aaaa180e34e27c83f`, source digest
`827268b10af578c61be4ec27581c6fbbd208626a20c080ca0ed820497efa6d9e`,
with 63 train / 5 validation episodes and normalizer `32a3a4d7f21f`. Epoch 4
is complete and epoch 5 is active. No historical checkpoint is a recovery
input.

At matched epoch 4, physical validation is:

| Run | full | arm | gripper | 1-4 | 5-12 | 13-24 |
|---|---:|---:|---:|---:|---:|---:|
| Recovery | 0.07694 | 0.05853 | 0.14453 | 0.03174 | 0.05727 | 0.09653 |
| Schema28 | 0.07951 | 0.06114 | 0.14775 | 0.03308 | 0.05959 | 0.09956 |
| Schema29 | 0.08046 | 0.06261 | 0.14764 | 0.03386 | 0.06045 | 0.10065 |

The main rejected-self-conditioning regression is also largely gone. Recovery
proposal/refinement, final interval mismatch and final delta mismatch are
`0.01921 / 0.02297 / 0.01117`, versus Schema29
`0.17451 / 0.19102 / 0.07348`. They remain 20-28% above matched Schema28 E4,
so the complete curve is still required.

Health gates are closed: no non-finite, OOM, traceback, identity or ledger
failure; all active formal owner VJPs remain present; median/p90 runtime is
`1.918 / 2.048 s/batch`; process peak estimate is `11.811 GiB`. Finite
threshold crossings remain secondary telemetry and do not stop the run.

Execution from here is intentionally narrow:

1. let Pen recovery run to E8 unless a hard-stop condition occurs;
2. do not modify structure, add a numerical control or start B-spine from the
   favorable E4 point;
3. at E8, overwrite the same local recovery log trio, then audit the complete
   Recovery/Schema28/Schema29 curves once;
4. run the formal V120 recovery gate on the complete run directory and replace
   the observations in `CURRENT_MAINLINE_ISSUES.md` in place;
5. only if that decision shows no new hard/core failure may Gate A start.

The current RDT-8 log is Schema29/cachefix and is judged separately per task.
It may validate the adapter/data surface and supply RDT data to the offline
chart probe, but it cannot be called a recovery-core experiment.

The canonical audit command is:

```bash
python -m clearvla.tools.audit_policy_logs runs/<run-tag> --format text
```

For a release comparison, pass the complete run directory together with the
locally available V120/Schema28 anchors. Verify split, action normalizer,
decoder, batch size and completed epochs before calling a delta architectural.

## Health gate at every checkpoint

Continue only when all of the following remain true:

- manifest/config/source and dataset identities match the run context;
- `loss_ledger_gap` and contribution gap remain numerical zero;
- formal parameter-owner gradients remain present on active paths;
- G mass conservation, P2 no-null terminal, CandidateWorld tag identity and
  capacity non-expansiveness remain valid;
- CUDA values remain finite and process memory stays at or below 22 GiB;
- checkpoints are atomic and remain exact-resume compatible only with the same
  source identity.

Hard-stop only for non-finite values, lineage failure, an open ledger, vanished
formal VJP, checkpoint ABI failure or memory overflow. Finite threshold
crossings do not change gate status by themselves, regardless of count or
maximum; investigate them only when they are reproducibly coupled to sustained
post-event damage, validation regression or another hard-stop failure. Early
event F1, small geometry RMS and full capacity are also review signals rather
than independent stop rules.

## Pen decision surface

Compare every completed validation epoch with the complete Schema28 anchor on:

- physical and normalized full, first, first-8, tail and three horizon bands;
- arm and gripper RMSE;
- decoded gripper precision/recall/F1, predicted/target event counts and
  post-event horizons;
- W prediction versus Teacher semantic/transport variation for every interval;
- proposal-to-refined action change and final action-to-W condition mismatch;
- G/S/W/P1/P2/P3/CT/bottom owner gradients and exact loss contributions;
- optimizer/gradient recovery, using finite threshold crossings only as
  secondary correlation evidence when a later health or behavior change exists.

Schema30 passes the core behavior gate only if improvement is not bought by a
near/far or arm/gripper trade and the final result, not merely the best epoch,
remains competitive.

## RDT-8 decision surface

For each of the eight tasks report:

- train/validation sample count and sampler mass;
- camera and action profile plus normalizer/language identity;
- full, arm-or-joint-group, gripper and horizon metrics;
- decoded event counts and validity coverage;
- relevant core owner gradients and any task-local non-finite event; finite
  threshold crossings remain optional context rather than a per-task score.

First classify a failure before touching the core:

| Pen | RDT-8 | Primary interpretation |
|---|---|---|
| bad | bad across tasks | shared core or training engine |
| healthy/improved | broadly bad | adapter, profile, sampler or task competition |
| healthy/improved | most healthy, a few bad | task-local data/action/camera semantics |
| bad | mixed/healthy | Pen behavior contract or task-specific core demand |

Stop only the affected outlet when identity and evidence localize the failure
outside the shared core.

## How the next source unit is chosen

After a decision checkpoint:

1. state the behavior failure without proposing a mechanism;
2. trace its exact producer -> transform -> consumer path in current source;
3. trace loss -> tensor VJP -> parameter owner -> optimizer in reverse;
4. use an existing matched intervention, or add the smallest validation-only
   one whose result would select between competing explanations;
5. change one coherent semantic unit, with explicit axes, zero semantics,
   dtype, scale and call frequency;
6. re-run forward/reverse, checkpoint, CPU/static, real CUDA VJP and both smoke
   gates before the next formal pair.

Do not reopen a broad architecture audit after step 1 unless the selected path
exposes a new cross-boundary contradiction. Repeated source review without a
new behavior decision is also not progress; change only a coherent semantic
unit whose expected observation is stated before implementation.

Do not bundle an amplitude gain, quota, hard event gate, entropy target, extra
clip or loss-weight change with a connection repair. Such changes make the
result harder to attribute and risk numerical hardening.

## Atomic architecture modularization source unit

This is one behavior-preserving source unit whose purpose is to make the
existing graph genuinely replaceable. The difficult part is the complete code
wiring, so contracts, registered ownership, factory selection, static/dynamic
composition, terminal/outlet handling, training objectives, runtime and
checkpoint mapping land together. There is no accepted façade-only,
half-rewired or old-owner/new-owner intermediate graph.

**Implementation status:** the local rewrite was explicitly authorized and is
now present as layout schema 2. Historical/formal runs remain identified by
their serialized layout-1 source and are not exact-resume inputs to this tree.
The B-spine integration and local zero-tolerance/lifecycle validation are now
closed. The remaining acceptance work is the real CUDA/BF16 and read-only
production-checkpoint gate; no fresh Pen training is authorized before it.

**Atomic acceptance rule:** implementation steps may be ordered internally, but
they share one working branch and one final source identity. Do not train,
checkpoint, compare experiments from, or merge any partial state. The unit is
accepted only after the complete modular graph passes the behavior-equivalence
gate against the pre-rewrite source.

The behavioral schema remains Schema30 because the mathematical graph, loss and
runtime behavior do not change. The modular registered layout receives a new
Schema30 ABI suffix, manifest digest and source identity. Existing checkpoints
are translated only through the explicit key map for parity/audit or a
separately authorized initialization; they are not exact-resume compatible
across source identities. B-spine remains the provisional Schema31 behavior
unit.

### Feasibility result: topology-preserving relocation

The 2026-09-04 source probe confirms that the atomic rewrite is implementable,
provided “behavior unchanged” is treated as a graph-isomorphism requirement,
not merely equal final actions. With seed 0, all 1,391 current state keys had a
collision-free proposed destination; mapped tensors had identical shape, dtype,
value and storage identity; attaching the already-constructed children changed
neither the RNG state nor the parameter inventory. The continuous baseline has
168,417,179 parameters (1,385 parameter tensors), 1,063 trainable tensors and
23 optimizer groups. The CALVIN selection constructs its additional command
head only when selected (1,389 tensors / 1,067 trainable), so lazy outlet
selection is also feasible.

The construction and execution proof must use two distinct order ledgers:

1. **Construction ledger.** Instantiate parameterized children in the current
   order, including initialization and buffer creation, then register each child
   once under its final path. Never instantiate a candidate just to inspect its
   shape. This preserves initialization RNG and values.
2. **Behavior-parameter ledger.** The logical hierarchy necessarily changes the
   raw `model.parameters()` traversal (the proposed nesting changed all 1,385
   positions), even though optimizer group membership can remain identical. A
   serialized legacy-name-to-new-name order must therefore drive optimizer
   construction, global gradient norm/clipping and any order-sensitive
   diagnostic reduction. Comparing sets or group names alone is not sufficient.

The current straight-line call planes are the relocation anchors and must stay
in this exact order:

```text
static / observation:
  condition_keep(goal) -> condition_keep(history)
  -> history_proposal
  -> observation.prepare
  -> bridge.sample_role_table
  -> bridge.grounding_canvas
  -> observation.build_grounding_bank
  -> begin progressive grounding
  -> G1 block/update -> G2 block/update -> G3 block/update
  -> observation.finalize_grounding
  -> grounder -> intent -> coarse action -> PhysicalActionCondition -> W
  -> bridge.clean_action_basis -> factual_reader -> transition.build_source

dynamic / each ODE node:
  outlet preparation (CALVIN zeroing, if selected)
  -> bridge.action_and_context
  -> P1 dynamic completion
  -> P2 effect/consequence -> P3 plan
  -> transition.forward -> execution_bottom.forward

training transaction:
  static encode -> Teacher/targets once -> one FlowMatchingState
  -> one formal velocity -> one LossComposer call -> backward
  -> decoder-local clip -> global clip -> optimizer step

deployment transaction:
  five updates + endpoint -> deterministic action projection
  -> one W rebuild -> same five updates + endpoint
```

The state map also has a non-tensor sidecar. `set_training_step` updates the
decoder's persistent execution-progress buffer and its Python fast-path value;
evaluation interventions live in the observation encoder, factual reader,
consequence, transition and decoder; several codec/query position buffers are
non-persistent. The new owners must either retain these methods at the same
owner or forward them through the composition root, and the harness must set
and compare them before/after load. A state-dict-only comparison can therefore
report a false pass while changing the warm-up, intervention or eval route.

The allowed and forbidden changes are deliberately narrow:

| Allowed in this source unit | Forbidden because it changes behavior |
|---|---|
| Move a function body and its child modules to a new owner | Change expression order, floating-point association or reduction order |
| Replace a call with a typed reference-only contract | Add `detach`, `clone`, implicit cast, reshape/reduction or a replacement projection at a boundary |
| Rename class/file/module paths and apply the explicit state-key map | Change initialization method, constructor order, dropout/RNG draw or persistent-buffer default |
| Inject the selected terminal controller and outlet adapter | Keep a direct terminal-head bypass, add an extra endpoint/ODE pass or alter masking |
| Split training/runtime/loss code into owners while preserving one transaction | Move Teacher/future evidence into deployment or add a second W/target build |
| Serialize component selection and ABI metadata | Instantiate inactive alternatives or register duplicate trainable owners |

One hidden decoder dependency remains a concrete implementation gate: there are
seven direct candidate `velocity_head` reads in the active decoder, in addition
to `_read_output_heads`:

```text
_probe_native_candidates                  (1)
_run_differentiable_native_candidates     (1)
_run_dynamic_execution                    (3: prefix, idle, post-step)
EvidenceLatentMMDiTActionDecoder.forward  (2: prefix, post-block)
```

All seven must call the injected `TerminalActionController` while retaining
their current detach, shape and masking semantics. The explicit parameter-order
ledger must likewise be consumed by
`training.optimizer.build_optimizer`,
`training.engine.MainlineTrainingEngine._gradient_lifecycle`,
`training.optimizer.gradient_diagnostics`, and the named-owner failure reports;
otherwise a visually correct hierarchy can still perturb the global clipping
scalar. Until these sites are covered by the dual-source harness, the rewrite
is feasible but not yet safe to land.

### Definition of done

The atomic rewrite is complete only when all of the following are true:

1. every replaceable component is a real registered owner under its final path,
   not a view over `top` or `bottom`;
2. `ClearVLAMainlinePolicy` is a thin composition root and contains no outlet,
   CALVIN, G-step, P1-internal, terminal-head or loss formula;
3. a `ComponentSelection` chooses exactly one implementation for every slot
   before module construction;
4. baseline selection executes the same tensor operations in the same order,
   with the same RNG draws, axes, dtypes, masks, normalization points and call
   counts as the current source;
5. old and new state tensors have a complete one-to-one key map, and the mapped
   model has identical forward, reverse, optimizer-step and deployment results;
6. no legacy monolith remains registered and no module is registered beneath
   two paths;
7. Pen, RDT and CALVIN behavior is expressed by outlet/terminal implementations
   rather than conditionals scattered through policy, sampling, loss and
   evaluation code;
8. resolved component names and compatibility ABI are serialized in
   `run_context.json` and checkpoint metadata;
9. the complete local, checkpoint, real-CUDA VJP and outlet smoke matrix passes
   before any modular experiment starts.

### Final registered hierarchy

The final baseline model owns this hierarchy:

```text
ClearVLAMainlinePolicy
  conditioning
    history_proposal
  observation
    compiler
  bridge
    query_encoder
  grounding
    blocks
    content_mod
    content_mod_scale
    grounder
  intent
    organizer
    coarse_action
  world
    dynamics
  p1
    factual_reader
    dynamic_time
    dynamic_content_mod
    dynamic_content_mod_scale
    dynamic_policy_block
  policy_compiler
    effect_reader
    consequence
    plan_compiler
  transition
  execution_bottom
    layer_contract_heads
    decoder
      terminal_controller
        action_norm
        velocity_head
        optional_command_head
        motion_head
  training_targets
    teacher
    recognizer
  outlet_adapter
    codec
```

`transition` may keep its current root name because it already has a clean
cached-source/per-node split. `observation` may keep its public root name, but
its compiler must be accessed only through the observation/grounding contracts.
All other former `top.*`, `factual_reader.*` and `bottom.*` children move to
their final owners in this same rewrite.

The factory must preserve the legacy parameter-construction sequence even
though final registration follows the logical hierarchy. Construct raw children
in the current RNG order, then attach them once to the final owners without
creating compatibility copies:

```text
observation compiler
-> physical codec
-> grounding blocks/content/grounder
-> intent/coarse action
-> world dynamics
-> teacher/recognizer
-> effect/consequence/plan compiler
-> history proposal
-> factual P1 reader
-> controlled transition
-> shared query encoder
-> dynamic P1
-> layer-contract heads
-> V120 decoder and terminal heads
```

No registry may instantiate an unused candidate to discover its shape or
metadata. Compatibility metadata is static and checked before construction.

### Current function to final owner ledger

| Current source function | Final owner and method | Exact wiring requirement |
|---|---|---|
| `ClearVLAMainlinePolicy.__init__` | `build_policy(config, selection)` | Construct only selected components in the legacy RNG order, register them once under the final hierarchy, then inject them into the thin policy root. |
| goal/history block inside `encode_online` and `HistoryActionProposal.forward` | `conditioning.prepare` | Proposal still reads complete observable history; goal and history masks are sampled exactly once and enter only the same consumers. |
| `RestoredV120ObservationCompiler.prepare` | `observation.prepare` | Preserve early mask ownership and the raw/DINO/flow computation. |
| `RestoredV120EvidenceBottom.sample_role_table` | `bridge.sample_role_context` | Move `ActionQueryEncoder` to `bridge.query_encoder` and create one role table per observation. |
| `RestoredV120EvidenceBottom.grounding_canvas` | `bridge.build_grounding_seed` | Consume the cached role table; never resample it. |
| observation `build_grounding_bank`, `begin_progressive_grounding`, `advance_progressive_grounding` and `finalize_grounding` plus `ObjectIntentDynamicsTop.run_progressive_grounding` | `grounding.build_current` | Preserve literal G1/block/update, G2/block/update, G3/block/update order and return evidence, exact G3 rollout and facts. |
| `ObjectIntentDynamicsTop.build_online_context` call to `grounder` | `grounding.materialize_facts` | `ObjectFactSet` becomes the only G-to-S/W public result. |
| `build_online_context` calls to `intent` and `coarse_action` | `intent.organize` and `intent.propose_action` | Construct `PhysicalActionCondition` from the exact coarse proposal tensor and current action state. |
| `ObjectIntentDynamicsTop.build_candidate_world` | `world.materialize` | Accept only `ObjectWorldBelief` and `PhysicalActionCondition` and return their atomic `CandidateWorld`. |
| `ObjectIntentDynamicsTop.refine_deployment_world` | composition-root cache replacement around `world.materialize` | Use the same W implementation for initial and refined worlds; rebuild once only. |
| `RestoredV120EvidenceBottom.clean_action_basis_tokens` | `bridge.clean_action_basis` | Use the moved query encoder's exact horizon/basis identities. |
| `LateRawDetailPolicyReader.forward` call in `encode_online` | `p1.build_static` | Move the factual reader to P1 and own late-detail/progressive-address assembly and `FactualPrecisionDock` construction. |
| `RestoredV120EvidenceBottom.action_and_context` | `bridge.build_dynamic_query` | Atomically return the shared action query and sole `V120SeedContext`. |
| `RestoredV120EvidenceBottom.complete_p1_fact` | `p1.update_dynamic` | Move time/content modulation and policy block to P1; keep factual base and dynamic residual separate. |
| `ObjectIntentDynamicsTop.compile_policy` | `policy_compiler.compile` | Preserve P2 contract, consequence and P3 order; export plan separately from trace-only effect/consequence. |
| `ControlledTransitionDynamics.build_source` and `forward` | `transition.prepare_source` and `transition.step` | Retain two APIs and the current normalization position. |
| `RestoredV120EvidenceBottom.forward` and `compile_evidence_view` | `execution_bottom.step` and `compile_evidence_view` | Bottom accepts only field, time, query, plan, seed and transition. Remove full `ObjectIntentState` because the active `_intent_memory` discards it. |
| `EvidenceLatentMMDiTActionDecoder._read_output_heads` and every direct `velocity_head` call | `terminal_controller.read_heads` and `predict_candidate_velocity` | Route endpoint, prefix, idle, probe and differentiable candidates through the injected controller without changing existing masking semantics. |
| `PhysicalActionFieldCodec` plus CALVIN branches in policy, sampling, loss and evaluation | `outlet_adapter` | Own encoding, noise, pre-core sanitization, continuous decode, command finalization, execution weights and validation accumulation. |
| `runtime.sampling._integrate_cache` | `integrate_physical_field` plus `outlet_adapter.finalize` | Keep five update nodes and one endpoint-producing full forward per pass. |
| `runtime.sampling.refine_cached_world` | composition root using `OutletActionOutput.world_condition_action` | Project only the declared canonical proposal, reuse identical initial noise and rebuild W once. |
| `MainlineTrainingEngine._forward_encoded` | thin training transaction | Call target stage, outlet flow state, policy step, objective stages and one composer exactly once. |
| `training.losses.compose_losses` | `OutletObjective`, `RepresentationObjective`, `ExecutionObjective` and `LossComposer` | Move formulas without changing weights, term names or the three final groups. |

### Final cross-component contracts

Add the shared types to
`clearvla/mainline/model/component_contracts.py`. These dataclasses validate
references but do not detach, clone, cast, reduce or project tensors.

```python
@dataclass(frozen=True)
class SharedRoleContext:
    role_table: Tensor                  # [8,H]


@dataclass(frozen=True)
class GroundingSeed:
    canvas: Tensor
    slices: Mapping[str, slice]


@dataclass(frozen=True)
class GroundedObservationBundle:
    evidence: ObservationEvidence       # retains N=49/detail through static P1
    facts: ObjectFactSet                # formal G -> S/W boundary
    g3_rollout: Tensor                  # [B,4*C*8*8,H]
    role_context: SharedRoleContext


@dataclass(frozen=True)
class DynamicQueryBundle:
    model_field: Tensor                 # [B,24,18]
    action_query: Tensor                # [B,24,4,H]
    seed: V120SeedContext


@dataclass(frozen=True)
class PolicyCompileResult:
    execution_plan: ObjectPolicyPlanDeltaBank
    trace: PolicyCompileTrace           # effect/consequence; not a bottom input


@dataclass(frozen=True)
class TerminalHeadOutput:
    physical_velocity: Tensor
    motion_logits: Tensor
    command_logits: Tensor | None
    diagnostics: Mapping[str, Tensor]


@dataclass(frozen=True)
class OutletActionOutput:
    deployed_action: Tensor
    world_condition_action: Tensor
    continuous_action: Tensor | None
    command_logits: Tensor | None
    command: Tensor | None
```

The decisive interfaces are:

```python
class P1Stage(Protocol):
    def build_static(
        self,
        *,
        evidence: ObservationEvidence,
        g3_rollout: Tensor,
        clean_basis: Tensor,
        intent: FactualIntentDock,
        collect_diagnostics: bool,
    ) -> tuple[FactualPrecisionDock, dict[str, Tensor]]: ...

    def update_dynamic(
        self,
        *,
        action_query: Tensor,
        factual: FactualPrecisionDock,
        time: Tensor,
        collect_diagnostics: bool,
    ) -> tuple[CompletedP1PolicyState, dict[str, Tensor]]: ...


class PolicyCompilerStage(Protocol):
    def compile(
        self,
        *,
        candidate_world: CandidateWorld,
        intent: PolicyIntentDock,
        p1_state: CompletedP1PolicyState,
        action_query: Tensor,
        collect_diagnostics: bool,
    ) -> PolicyCompileResult: ...


class ExecutionBottomStage(Protocol):
    def step(
        self,
        *,
        model_field: Tensor,
        time: Tensor,
        action_query: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        seed: V120SeedContext,
        transition: ControlledTransitionState,
        execution_mode: str,
        require_execution_supervision: bool,
        collect_diagnostics: bool,
    ) -> tuple[BottomDecoderOutput, dict[str, Tensor]]: ...
```

The execution-bottom input deliberately excludes goal, RGB/DINO,
`ObjectFactSet`, `CandidateWorld`, Teacher, outlet/task id and full
`ObjectIntentState`.

Terminal behavior is injected inside the decoder:

```python
class TerminalActionController(Protocol):
    def predict_candidate_velocity(
        self,
        normalized_terminal_state: Tensor,
    ) -> Tensor: ...

    def read_heads(
        self,
        normalized_terminal_state: Tensor,
        *,
        collect_diagnostics: bool,
        collect_gripper_diagnostics: bool,
    ) -> TerminalHeadOutput: ...
```

The baseline implementations are `ContinuousPhysicalController` for Pen/RDT
and `CalvinBinaryCommandController` for CALVIN. The latter preserves the
current difference between raw candidate prediction and final emitted
continuous-gripper velocity masking.

Outlet behavior is:

```python
class OutletActionAdapter(Protocol):
    def encode_target(...) -> Tensor: ...
    def sample_noise(...) -> Tensor: ...
    def prepare_model_field(self, field: Tensor) -> Tensor: ...
    def decode_field(...) -> Tensor: ...
    def finalize(
        self,
        continuous_action: Tensor,
        terminal: TerminalHeadOutput,
    ) -> OutletActionOutput: ...
    def action_terms(...) -> dict[str, Tensor]: ...
    def execution_component_weights(...) -> Tensor: ...
    def make_validation_accumulator(...): ...
```

Baseline adapters are `Pen7DContinuousOutlet`,
`RdtRightArm7DOutlet` and `Calvin7DBinaryOutlet`. All expose the canonical
`[B,24,7]` core action and `[B,24,18]` physical field. CALVIN clears the six
compatibility-only future-gripper coordinates in `prepare_model_field` before
the bridge, P1, P2/P3, transition or bottom can read them.

### Final composition code path

The rewritten static path is exactly:

```text
conditioning.prepare
-> observation.prepare
-> bridge.sample_role_context
-> bridge.build_grounding_seed
-> grounding.build_current
-> intent.organize
-> intent.propose_action
-> world.materialize
-> bridge.clean_action_basis
-> p1.build_static
-> transition.prepare_source
-> cache/training-state assembly
```

`GroundedObservationBundle.g3_rollout` is the exact final canvas rollout slice.
Static P1 and `ControlledTransitionSource` receive that same object.
`ObservationEvidence` remains alive through static P1 and is then retained only
by the training state.

The rewritten per-node path is exactly:

```text
outlet_adapter.prepare_model_field
-> bridge.build_dynamic_query
-> p1.update_dynamic
-> policy_compiler.compile
-> transition.step
-> execution_bottom.step
```

P2 remains the only caller of `P2QueryDock.combined()`. Transition remains a
third slot rather than being absorbed into W or bottom.

Sampling is:

```text
integrate five updates
-> one endpoint-producing full policy step
-> outlet finalization
-> world.materialize once from world_condition_action
-> integrate the same five updates from identical initial noise
-> one endpoint-producing full policy step
-> final outlet finalization
```

Training is:

```text
one static online encode
-> one training-target build
-> one outlet flow-state sample
-> one formal dynamic policy step
-> outlet/representation/execution objectives
-> one LossComposer call
```

### One-shot implementation procedure

The following numbering describes dependency order inside the one atomic
working branch. None of these points is a releasable or experimentally usable
intermediate version.

1. Capture the pre-rewrite source identity, dirty diff, resolved configuration,
   module construction order, state inventory, optimizer groups, post-init RNG
   hash and deterministic parity inputs.
2. Add final contracts, `ComponentSelection`, static compatibility metadata and
   lazy factories. Do not yet expose a second executable policy path.
3. Create the final registered component classes and move the existing child
   modules into them. Preserve child construction order and initialization code;
   do not create copies, shims with parameters or unused alternatives.
4. Rewrite `ClearVLAMainlinePolicy.__init__` to receive the complete resolved
   assembly and delete the old monolithic `top`/`bottom` registration paths.
5. Rewrite `encode_online` and `velocity` directly to the final static and
   per-node compositions above. Do not retain a legacy runtime switch.
6. Route every decoder terminal read through the injected terminal controller,
   then remove direct terminal-head reads from the decoder.
7. Move Pen/RDT/CALVIN branching into the selected outlet adapter and delete the
   equivalent conditionals from policy, sampling, losses and evaluation.
8. Split training target/objective functions and reduce
   `MainlineTrainingEngine._forward_encoded` to the fixed transaction without
   changing any formula or call count.
9. Install the explicit legacy-to-modular state map, update optimizer ownership,
   module inventory, manifest ABI suffix, source identity and serialized
   component selection.
10. Remove obsolete monolith classes or reduce source-only helpers to
    unregistered construction utilities. A repository search must find no live
    call through the old `top.compile_policy`,
    `bottom.complete_p1_fact`, `bottom.action_and_context` or scattered outlet
    mode checks.
11. Run the complete behavior-equivalence and lifecycle gate. Only its fully
    passing source becomes the new baseline.

One composition owner integrates this unit because the critical work is shared
wiring. Other workers may prepare isolated tests or component implementations,
but they do not land independent partial graphs or edit the shared composition
in parallel. The central hotspots are `model/policy.py`,
`model/restored_bottom.py`, `model/top.py`,
`v120_core/time_domain_mmdit.py`, `runtime/sampling.py`,
`runtime/evaluation.py`, `training/losses.py`,
`training/optimizer.py` and `train.py`.

### State, optimizer and checkpoint rewrite

The state migration is part of the same source unit:

| Current prefix | Final prefix |
|---|---|
| `history_proposal.*` | `conditioning.history_proposal.*` |
| `bottom.query_encoder.*` | `bridge.query_encoder.*` |
| `top.grounding_blocks.*` | `grounding.blocks.*` |
| `top.grounding_content_mod.*` | `grounding.content_mod.*` |
| `top.grounding_content_mod_scale` | `grounding.content_mod_scale` |
| `top.grounder.*` | `grounding.grounder.*` |
| `top.intent.*` | `intent.organizer.*` |
| `top.coarse_action.*` | `intent.coarse_action.*` |
| `top.dynamics.*` | `world.dynamics.*` |
| `top.teacher.*` | `training_targets.teacher.*` |
| `top.recognizer.*` | `training_targets.recognizer.*` |
| `factual_reader.*` | `p1.factual_reader.*` |
| `bottom.p1_time.*` | `p1.dynamic_time.*` |
| `bottom.p1_content_mod.*` | `p1.dynamic_content_mod.*` |
| `bottom.p1_content_mod_scale` | `p1.dynamic_content_mod_scale` |
| `bottom.p1_policy_block.*` | `p1.dynamic_policy_block.*` |
| `top.effect_reader.*` | `policy_compiler.effect_reader.*` |
| `top.consequence.*` | `policy_compiler.consequence.*` |
| `top.plan_compiler.*` | `policy_compiler.plan_compiler.*` |
| `bottom.layer_contract_heads.*` | `execution_bottom.layer_contract_heads.*` |
| `bottom.decoder.action_norm.*` | `execution_bottom.decoder.terminal_controller.action_norm.*` |
| `bottom.decoder.velocity_head.*` | `execution_bottom.decoder.terminal_controller.velocity_head.*` |
| `bottom.decoder.gripper_command_head.*` | `execution_bottom.decoder.terminal_controller.optional_command_head.*` |
| `bottom.decoder.motion_head.*` | `execution_bottom.decoder.terminal_controller.motion_head.*` |
| remaining `bottom.decoder.*` | `execution_bottom.decoder.*` |

Implement the mapping as data, not suffix guessing. It must prove:

- every old state key maps to exactly one new key;
- no new key has two old sources;
- tensor shape, dtype and value are unchanged;
- mapped load has no missing or unexpected keys;
- parameter and persistent-buffer counts remain unchanged;
- every trainable parameter has exactly one semantic optimizer owner;
- each mapped parameter receives the same learning-rate scale, decay policy and
  clipping order as before.

The final hierarchy is allowed to expose a different human-facing state-key
order, but the optimizer and gradient lifecycle are not allowed to inherit that
incidental reorder. Store the mapped legacy parameter sequence as explicit
metadata and use it to build the per-group lists and the decoder-local/global
clipping lists. If a consumer intentionally uses a different order, its
reduction must be proven numerically equivalent on the baseline before it is
accepted.

The old ordered state-key hash is retained as migration evidence, not as the new
layout's expected hash. The final layout records its own count and digest. The
post-construction RNG hash should remain unchanged because the rewrite adds no
parameterized alternative and preserves construction order.

### Behavior-equivalence gate

The sole success criterion is unchanged behavior under the baseline component
selection. Compare old source and fully rewritten source with the same weights,
inputs and RNG state.

The frozen pre-rewrite inventory is:

```text
total parameters:        168,417,179
trainable parameters:    152,046,448
parameter tensors:       1,385
trainable/optimizer:      1,063
optimizer groups:        23
state-key names:          1,391
old state-key SHA-256:    70a8a5be21de40c460de6cff899942d5331837700db289350a0b1920c133b053
post-init RNG SHA-256:    d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21
```

Required comparisons are:

1. **Fresh initialization:** with the same seed, every old/new parameter pair
   from the key map has identical value and the post-construction RNG state is
   identical.
2. **Static forward:** `encode_online` returns equal cache, training-state and
   metric values. Role-table sampling, goal/history masks, G1/G2/G3 updates,
   N=49 detail, facts, S, coarse action, CandidateWorld, factual P1 and
   transition source are compared at every named boundary.
3. **Tensor identity:** one role table is reused by G and the bridge; one coarse
   action tensor is the `PhysicalActionCondition.interval_action`; one condition
   is paired with its CandidateWorld; one G3 tensor reaches P1 and transition;
   one `V120SeedContext` reaches transition and bottom.
4. **Dynamic forward:** compare `time={0,.2,.4,.6,.8,1}` for action query,
   dynamic P1, P2 effect/consequence, P3 plan, controlled transition, decoder
   tensors, terminal heads, physical velocity and metrics.
5. **Training:** compare the complete `LossLedger` term, contribution, group and
   total dictionaries. Run backward and compare every raw parameter gradient
   through the key map before clipping.
6. **Optimizer:** compare decoder-local clip, global clip, group assignment,
   learning rate, decay and one optimizer-step parameter delta through the key
   map.
7. **Continuous outlets:** Pen and RDT target field, noise, decoded action,
   motion output, validation accumulators and execution-value targets are
   unchanged and expose no binary command tensor.
8. **CALVIN outlet:** the six legacy gripper input coordinates are zero before
   every dynamic consumer; logits, `{-1,+1}` command, arm-only execution value,
   continuous audit action and validation metrics match.
9. **Terminal execution:** compare endpoint, prefix, idle, hard/soft route,
   differentiable candidate and terminal-identity predictions. No direct
   velocity-head caller may remain outside the controller.
10. **Deployment lifecycle:** two complete five-update passes use identical
    initial physical noise, rebuild W exactly once and perform two endpoint
    forwards: twelve bottom calls total. Final W-condition mismatch metrics
    match.
11. **Teacher isolation:** Teacher/future support executes once per training
    batch and zero times in deployment and is unreachable from online
    interfaces.
12. **Axes and gradients:** camera/spatial/K/N=49/interval/horizon/basis/type
    axes and online autograd remain intact; no contract constructor detaches or
    recreates evidence.
13. **Static/checkpoint:** changed-file lint, compile, structural suite,
    legacy-key conversion, new checkpoint round trip and old-source exact-resume
    rejection pass.
14. **Real runtime:** real Pen B8 CUDA parameter-owner VJP, Pen smoke, RDT-8
    smoke, CALVIN smoke and read-only checkpoint validation pass within the
    22 GiB release boundary with no material throughput regression.

CPU deterministic paths require exact equality. CUDA/BF16 comparisons use the
established deterministic setup and must retain the same kernel batch shapes;
any nonzero delta must be explained and rejected unless it is proven to be
platform nondeterminism rather than changed graph behavior.

### Minimal dual-source equivalence harness

The feasibility probe is a construction check only. Before accepting the
rewrite, implement one harness that runs the frozen source and the complete
modular source in separate processes (and, where possible, separate worktrees)
so Python module state and RNG cannot leak between them. The harness has five
small, explicit parts:

1. `LegacyKeyMap`: a checked data table from every old state/buffer key to one
   new key. It rejects collisions, omissions, unexpected destinations, shape or
   dtype changes, and duplicate parameter identities before loading weights.
2. `BoundaryTrace`: wrappers at `encode_online`, `velocity`, the training
   transaction and `_integrate_cache` record a named event, call index, tensor
   shape/dtype, object-identity token and a content digest. The trace also saves
   CPU/CUDA/global/explicit-generator states before and after each event.
3. `OrderLedger`: the old ordered parameter names are mapped to new names and
   used to construct optimizer/clipping lists. It compares raw gradients before
   clipping, decoder-local and global norms, group hyperparameters and the
   one-step parameter delta; it must not rely on `model.parameters()` order.
4. `LifecycleCounter`: hooks count history proposal, G1/G2/G3, factual P1,
   dynamic P1, transition, every decoder block, terminal-controller read,
   Teacher and W materialization. It asserts one static pass, one Teacher (train
   only), two five-update deployment passes, one W rebuild and twelve bottom
   forwards.
5. `compare_tree`: CPU uses `torch.equal` for every named boundary and gradient;
   CUDA/BF16 first measures baseline self-repeat noise with identical kernels,
   then requires old/new deltas to stay within that measured noise. The report
   is a compact pass/fail table, never a tensor dump.

The first executable parity run should use the existing reduced test config and
fixed synthetic batch, then repeat the same harness for the resolved Pen, RDT-8
and CALVIN selections. A failed event, RNG state, identity token, parameter
order, owner gradient or lifecycle count is a source-unit failure and stops the
acceptance gate; do not relax tolerances or add a compatibility branch to make
the report pass.

#### Low-cost execution ladder (measured 2026-09-04)

The dual-source gate is intended to run during the rewrite, not only after it.
Its ordinary edit loop therefore uses a topology-complete reduced graph and an
immutable baseline capture. The frozen source is captured once; each later edit
starts only one candidate process and compares its result with that capture. The
two 168M-parameter models are never kept resident together.

Do not import `tests/test_mainline_policy.py::_config` verbatim as the golden
configuration. Its `bottom.controller_depth=1` deliberately removes the second
execution-controller recurrent/FFN layer and therefore omits six parameter and
state keys. Implement a dedicated `build_reduced_equivalence_config()` in the
mainline harness. It may retain the test widths (`hidden_size=32`,
`visual_token_dim=16`, `goal_token_dim=16` and the small observation widths),
but it must retain `action_basis_tokens=4`, `bottom.controller_depth=2`, the
three evidence blocks and the production goal, history and bottom dropout
values. On the current source this configuration has:

```text
parameters:             947,239
parameter tensors:      1,385
state keys:             1,391
state tensor bytes:     3,797,152
parameter-name order:   exactly equal to ExperimentConfig()
state-key order:        exactly equal to ExperimentConfig()
```

On the current Windows CPU environment, construction takes about 0.06 seconds;
one full training transaction takes 4.21 seconds and the complete two-pass,
five-update deployment lifecycle takes 6.04 seconds. The two paths together
take 10.25 seconds before artifact serialization. A candidate-only daily gate
therefore has a practical target below 30 seconds, while retaining every
registered owner and the complete dynamic call topology.

The same topology-complete transaction was also run twice in independent
Python processes with fixed global and explicit-generator seeds. A recursive
exact-byte digest covered the post-step model state, optimizer and scheduler
state, training result, complete deployment result and final RNG state. Both
processes produced
`8330c56d9fe3f8d6e1441fb88bd808304b2c03cf51bcf5bf39ec49643e71f717`.
This establishes that zero-tolerance CPU comparison is achievable on the
current environment; a same-source self-repeat remains the first calibration
performed by the future harness.

Use these three execution tiers:

1. **Freeze once:** after the recovery entry gate, retain an immutable baseline
   source ref, the branch-rich synthetic fixture, its SHA-256, the harness source
   digest and one reduced capture. Never regenerate this baseline from the
   candidate tree. Baseline and candidate captures always run in independent
   processes with `PYTHONHASHSEED=0`, one CPU thread and deterministic PyTorch
   algorithms.
2. **Every edit:** construct only the topology-complete reduced candidate and
   run structure, one training step and the complete five-step deployment
   sample. CPU tensors, RNG states, gradients and one-step updates must be
   exactly equal; the permitted tolerance is zero. Stop at the first divergent
   named event.
3. **Acceptance only:** repeat with the resolved full Pen, RDT-8 and CALVIN
   selections, production shapes, a real read-only checkpoint, checkpoint
   round trip and CUDA/BF16 smoke. Measure a same-source CUDA self-repeat before
   interpreting any nonzero GPU delta. This tier runs once for the complete
   atomic source unit, not after each file move.

The thin adapter should live at
`clearvla/tools/mainline_equivalence.py`; keep the historical V76-V88 build and
fixture logic in `policy_golden.py` unchanged. Reuse its deterministic seeding,
`ArtifactBuilder` and exact comparison code for the reduced capture. The
adapter needs the following small, function-level capture surface:

| Surface | Current function boundary | Exact record |
|---|---|---|
| construction | `ClearVLAMainlinePolicy.__init__`, `build_optimizer` | mapped state metadata/digest, registration and legacy parameter order, alias graph, `requires_grad`, optimizer group/order/hyperparameters, pre/post-init RNG |
| static online | `ClearVLAMainlinePolicy.encode_online` plus `HistoryActionProposal.forward`, `RestoredV120ObservationCompiler.prepare/build_grounding_bank/begin_progressive_grounding/advance_progressive_grounding/finalize_grounding`, `ObjectIntentDynamicsTop.run_progressive_grounding/build_online_context`, `LateRawDetailPolicyReader.forward`, `ControlledTransitionDynamics.build_source` | goal/history masks, history proposal, G1/G2/G3 event order, role table, facts, S/W, static P1, transition source, cache identity/alias relations and RNG |
| dynamic online | `ClearVLAMainlinePolicy.velocity`, `RestoredV120EvidenceBottom.action_and_context/complete_p1_fact/forward`, `ObjectIntentDynamicsTop.compile_policy`, `ControlledTransitionDynamics.forward`, `EvidenceLatentMMDiTActionDecoder.forward` and every injected terminal-controller read | dynamic P1, compiled P2/P3, transition, each decoder block, all seven terminal sites, decoded terminal values and ordered call trace |
| training | `MainlineTrainingEngine._forward`, `_gradient_lifecycle`, `optimizer.step`, `WarmupCosineSchedule.step` | every loss term/group/contribution, raw/post-local/post-global gradient by legacy logical name, decoder/global norms, one-step parameter delta, optimizer state and scheduler state |
| deployment | `sample_action` | values and event order at `t={0,.2,.4,.6,.8,1}`, one static build, one W rebuild, zero Teacher calls and `2 * (5 + 1) = 12` P1/transition/bottom forwards |
| sidecars | `set_training_step`, `set_execution_training_step`, `set_execution_eval_ablation`, P1/transition/compiler intervention setters, checkpoint load | Python fast-path progress, persistent progress, intervention modes, non-persistent query/position buffers and their post-load reconstruction |

Cross-process object addresses cannot be compared directly. `BoundaryTrace`
must assign local monotonically increasing identity tokens and compare the
resulting equality/alias pattern: for example, “the cache handed to `velocity`
is the exact object returned by `encode_online`.” Raw `id()` or `data_ptr()`
values themselves are process-local and must not enter the golden result.

The full-profile tier must not call the current
`ArtifactBuilder.add("state/initial", state_dict)` path: that helper clones every
tensor and the comparator loads the complete tensor file, which is cheap for
the 3.80 MB reduced graph but wasteful for the 168M-parameter graph. Add a
streaming `add_digest_tree()` path in the new adapter that records, in legacy
logical order, each tensor's key, shape, dtype, stride, `requires_grad` and
SHA-256 of contiguous exact bytes. Store full values only for the small named
boundaries. If a digest differs, rerun the frozen source and candidate with a
`--focus <logical-name>` option and dump only that first tensor and its immediate
producer inputs. This keeps successful full captures compact without making a
failure impossible to localize.

The minimum daily acceptance row is consequently one compact report:

```text
key map | init/RNG | event order | boundaries | losses | raw grads |
local/global clip | optimizer step | 5-step lifecycle | sidecars
  PASS       PASS          PASS          PASS      PASS       PASS
       PASS                 PASS               PASS            PASS
```

Any failed column rejects the atomic rewrite. Passing the reduced tier gives a
fast edit signal; only the full-profile/checkpoint/CUDA tier closes behavioral
equivalence.

### Component selection after the atomic rewrite

Use a separate experiment object, not boolean combinations in
`ExperimentConfig`:

```python
@dataclass(frozen=True)
class ComponentSelection:
    conditioning: str
    observation: str
    role_query_bridge: str
    grounding: str
    intent: str
    world: str
    p1: str
    policy_compiler: str
    transition: str
    execution_bottom: str
    terminal_controller: str
    outlet_adapter: str
    objectives: str
```

Compatibility is checked before construction and includes at least hidden
width, horizon=24, basis=4, objects=4, cameras=2, G grid=8x8, P1 N=49,
canonical action dimension=7, physical field dimension=18, terminal command
mode and component ABI revision.

A launcher may expand several `ComponentSelection` values into independent
runs. Each process constructs exactly one implementation per slot; alternatives
are never held in a trainable `ModuleDict`. A future worker changes one
implementation and registry entry, while the composition root and all other
component contracts remain fixed. Any contract change is a new atomic ABI unit,
not a local experiment.

### Interaction with B-spine

B-spine Gate A remains read-only and may start after the common E8 entry
condition. B-spine Gate B starts only after this atomic modular rewrite passes
the full behavior-equivalence gate. It enters as a new
`execution_bottom` implementation and may not change the bridge, P1, P2/P3,
transition, terminal/outlet finalization or top-to-bottom contracts. The
modular rewrite remains Schema30 with a new ABI suffix; B-spine remains the
provisional Schema31 behavioral source unit.

## Conditional next phase: bottom-first B-spine

If Schema30 has no new hard/core failure and its complete behavior is at least
stable, the next candidate is a **bottom-internal numerical spine derived from
a B-spline chart**. `B-spine` names this ClearVLA numerical spine; it does not
mean that the action output itself becomes the B-spline Policy paper's spline
parameter vector. This is a conditional implementation plan, not an active
architecture contract, and it must not be used to conceal an unresolved
Schema30 S/W/P/outer-closure failure.

Gate B also requires the complete atomic modular rewrite and all of its
behavior-equivalence/lifecycle gates. Gate A remains read-only and has only the
E8 recovery entry condition.

### Why bottom first

The active path is source-confirmed as:

```text
normalized native action [B,24,7]
  -> PhysicalActionFieldCodec -> flow target/noise x_t [B,24,18]
  -> RestoredV120EvidenceBottom
  -> EvidenceLatentMMDiTActionDecoder.noisy_lift(x_t)
  -> horizon/semantic/policy additions
  -> three Evidence MMDiT blocks
  -> unchanged physical-velocity head
```

`ActionQueryEncoder` is shared by P2, P3 and ControlledTransition, so changing
it would not be a bottom-only unit. The clean insertion point is beside
`NativeTimePhysicalActionTokenLift` in
`clearvla/mainline/v120_core/time_domain_mmdit.py`, before the existing MMDiT
blocks. The original full-resolution `x_t -> noisy_lift` path remains present.

The active field producer is
`clearvla/mainline/model/action_codec.py::PhysicalActionFieldCodec`; its
coordinates are `arm_absolute[0:6]`, `arm_delta[6:12]`,
`gripper_value[12]`, `gripper_delta[13]`, and four gripper auxiliary channels
`[14:18]`. Do not substitute the inactive `mainline/model/bottom.py` prototype,
the legacy `v120_core/codec.py::DCTFlowCodec`, or either historical
output/action-space spline writer. Those have different field/ABI semantics.

The fusion point is fixed before implementation:

```text
u_raw   = noisy_lift(x_t)
u_spine = BSpine0(x_t)
action += (u_raw + u_spine) * action_state_factor
```

`action_state_factor` is the existing `noisy_scale` expansion used by the raw
branch. Applying it to both views preserves the time-gate and ablation
semantics; adding an unscaled spine would be a silent new amplitude controller.
At zero spine weights, the forward graph is bit-equivalent to Schema30.

This placement addresses a specific representation hypothesis. The current
lift sums arm/gripper component projections and component embeddings, then
applies `LayerNorm -> MLP`. In an eight-seed random-initialization structural
probe, zero input and ordinary random input had similar output RMS
(`0.179 / 0.182`), and doubling the input changed output RMS by only `1.005x`
on average. This is not checkpoint or behavior evidence: token direction can
still carry magnitude, and training may change the response. It justifies a
read-only checkpoint scale/Jacobian probe and a parallel raw-number path; it
does not justify deleting or rewriting the existing lift.

The external evidence has a narrower scope than this proposal:

- B-spline Policy (`arXiv:2607.09648`) predicts knots/control points as a
  drop-in replacement for discrete action chunks and exploits local support;
- Spline Policy (`arXiv:2606.07386`) shows that a spline output can expose
  resampling and local-correction structure while retaining a backbone;
- BEAST (`arXiv:2506.06072`) uses fixed-length B-spline action tokens;
- FAST (`arXiv:2501.09747`) shows that a DCT frequency representation is a
  strong alternative for temporally correlated action sequences.

These works support testing an explicit numerical trajectory view. They do not
show that a spline must replace ClearVLA's flow field, decoder or action head,
or that B-spline is superior to DCT inside this model. ClearVLA commits
`966dd02` and `5a81137` were respectively an output-only B-spline arm writer
and a query-direction spline writer. They are ancestry evidence and possible
basis-code references, not implementations to revive.

The papers' adaptive knot insertion and segment-alignment procedures belong to
an output/execution interface; importing either would add a new action ABI or
inference state. Likewise, the Spline Policy vector-field correction relies on
regularity and projection assumptions that are not established for a noisy flow
field. Those ideas remain later experiments, not hidden behavior in
B-spine-0.

### Non-negotiable B-spine-0 boundary

The first unit must satisfy all of the following:

1. It reads only the deployed noisy physical flow state `x_t [B,24,18]` and
   existing static configuration. It cannot read targets, future Teacher,
   labels, decoded events, task ids or a new top carrier.
2. The original 24-row raw path remains unchanged and independently usable.
   The spline chart is a parallel view, never a compression bottleneck.
3. It does not change `PhysicalActionFieldCodec`, flow targets/noise, the
   physical velocity head, action loss, ODE update count, W rebuild count,
   P2/P3/transition ownership or the final action ABI.
4. It uses fixed clamped basis/analysis matrices. It learns neither knots nor
   smoothing strength and adds no trajectory-smoothing loss.
5. It introduces no clipping, quota, entropy target, amplitude target, event
   gate, hand-set gain or extra objective weight. A disabled branch is exactly
   Schema30, and the enabled branch begins at exact zero contribution.
6. All derived gripper coordinates remain distinct. Because independent noise
   makes intermediate `x_t` off the clean action manifold, B-spine may label
   channel roles but must not forcibly reconstruct clean-field relations at
   intermediate flow times.
7. The present physical-chart unit/range fields remain metadata only. Pen's
   verified nominal references and RDT's still-unknown source-native units
   cannot silently become model inputs in this unit.

### Gate A — one dataset/replay probe, no training sweep

**Entry condition:** the Pen recovery E8 curve and formal recovery audit are
complete with no new hard/core failure. The RDT adapter/data contract must be
available for the cross-outlet dataset traversal, but the existing
Schema29/cachefix RDT behavior run is not recovery evidence. Until then this
gate is a written plan only; do not run a partial probe against a moving
baseline.

Implement one CPU-capable, dataset-only mainline probe. It must reuse the
formal episode selection, train-only z-score normalizer, action profile and
`PhysicalActionFieldCodec`, then deterministically construct both:

```text
clean normalized action + action_state -> target physical field x_1
fixed standard-normal source + formal mirrored-Beta time -> actual x_t
```

Run every candidate in one traversal of the same windows:

- open-uniform clamped B-spline, degree 2 and 3;
- control count `K in {4,6,8,10,12}`;
- truncated orthonormal DCT at the same K values;
- full identity/round-trip references only, not additional train candidates.

Open-uniform knots are intentional for this first gate: manually placing knots
at the current action bands would bake today's diagnosis into the chart. An
interval-aware knot layout, adaptive knots, and degree-1 alternatives are
separate future questions, not hidden variants in this run.

For each fixed synthesis matrix `B [24,K]`, use its Moore-Penrose pseudoinverse
`A [K,24]` (no silent ridge term) and calculate in FP32:

```text
c        = A @ x
x_coarse = B @ c
x_detail = x - x_coarse
```

The solve is performed once when constructing the basis, never inside a model
forward. Deterministic matrix checks already bound the candidate family:
quadratic K=4..12 has condition number about `2.74..2.84` and analysis norm
`1.07..1.83`; cubic has about `4.67..5.28` and `2.14..2.99`. Cubic K=24 is
excluded as a numerical chart because its condition number/analysis norm rise
to about `66.5 / 63.8`. DCT remains the unit-condition global-support control.
Report locality and boundary leverage for both `B` and `A`; a locally supported
synthesis matrix can still acquire a globally coupled or amplifying analysis
operator. If a solve needs regularization, stop and record that as a separate
chart decision instead of silently importing the historical ridge value.

The probe output is a compact decision table, not a raw-window dump. Report
separately for clean `x_1`, actual sampled `x_t`, flow-time bins, Pen and each
RDT task where available:

- exact decomposition and partition-of-unity error;
- basis rank, condition number, analysis operator norm and temporal locality;
- coarse/detail RMS, energy share and correlation for every physical-field
  role: arm absolute, arm delta, gripper value, gripper delta and four gripper
  auxiliary coordinates;
- full and `1-4 / 5-12 / 13-24` reconstruction/decode error;
- first/second temporal-difference and endpoint error;
- event, post-event and hold gripper residuals with their row counts;
- sensitivity to input rescaling and the corresponding fixed-operator norm.

All per-role values are summaries over the same deterministic windows; the probe
must not emit raw windows or add a train-time data pass. Its only persistent
artifact is the compact decision table and the basis specification/hash.

Do not select the lowest reconstruction error mechanically, because larger K
will usually win that scalar. First reject non-finite, rank-deficient,
non-partitioning or strongly amplifying candidates. Then choose one
cross-outlet Pareto point using conditioning/locality, far-band structure,
event preservation and the incremental benefit of another control point. If
no candidate creates a coherent coarse/detail split on both outlets, stop this
proposal before touching the training graph. Freeze one basis kind, degree and
K in the plan before implementation; there is no training hyperparameter
sweep.

In the same pre-training gate, run a server-side read-only probe on a real
Schema30 checkpoint at `0`, `0.5x`, `1x` and `2x` physical input, split by
channel role and flow time. Record output norm/direction plus input JVP and
loss-side VJP for `NativeTimePhysicalActionTokenLift`. Do not download the
checkpoint. If the trained lift is scale-sensitive, the B-spine rationale is
long-horizon organization rather than an amplitude repair; do not rewrite the
rationale after seeing the result.

### Gate B — one bottom-only semantic unit

After Gate A freezes the basis, implement the smallest useful graph:

```text
x_t [B,24,18]
  +-> existing noisy_lift ------------------------------> u_raw [B,24,H]
  |
  +-> fixed FP32 A/B -> c, x_coarse, exact x_detail
                        |                 |
                        v                 v
              zero-init bias-free   zero-init bias-free
              role-split lift       role-split lift
                        |                 |
                        +-------> u_spine [B,24,H]

existing action seed + (u_raw + u_spine) * action_state_factor
                                      -> existing MMDiT blocks/head
```

For each role `r`, the intended algebra is
`c_r = A @ x_r`, `u_coarse = sum_r B @ L^c_r(c_r)`,
`u_detail = sum_r L^d_r(x_r - (B @ A @ x_r))`, and
`u_spine = u_coarse + u_detail`. `L^c_r` and `L^d_r` are independent
bias-free maps to the existing hidden width; no cross-role map or hidden
normalizer is inserted before the existing MMDiT blocks. The fixed matrix
products run in an autocast-disabled FP32 scope and are cast only at the
existing hidden-token boundary, while gradients remain attached to `x_t` and
the new maps.

The role split is the current model chart, not claimed physical units:
`arm_absolute[0:6]`, `arm_delta[6:12]`, `gripper_value[12]`,
`gripper_delta[13]`, and `gripper_auxiliary[14:18]`. The coarse branch lifts
control coefficients and synthesizes hidden time rows with B; the detail
branch lifts exact full-resolution residuals. Both learned lifts are
bias-free, zero-preserving and zero-initialized. They contain no internal
LayerNorm or learned scalar gate, so input magnitude is not deliberately
discarded and there is no new amplitude controller to collapse or saturate.
With two role-split linear maps this adds only about `2 * 18 * H` weights
(`18,432` at H=512), fixed small matrix multiplies and no Transformer block.

The new basis spec (`kind`, degree, `K`, horizon, matrix hash and field-role
map) must be serialized in the Schema31 config/manifest. It is not an
environment-only switch. The Schema30 config and manifest remain unchanged.

The implementation must expose an evaluation-only `spine_zero` intervention
without changing training behavior. Log only the decision-complete surface:
coarse/detail field RMS, coarse/detail token RMS, total spine update RMS,
spine/raw token ratio, exact decomposition error, and separate coarse/detail
parameter-owner gradients. Detailed per-role rows belong in the bounded probe,
not every train step. Add one explicit `bottom_spine` optimizer owner for
`bottom.decoder.spine.*`, with the existing bottom-decoder LR/decay policy
(`0.7x`) and no movement of any existing parameter owner.

### Gate C — forward/reverse and lifecycle review

Before any formal run, complete and record all of these gates:

1. basis partition, endpoint, rank and exact `x=coarse+detail` tests in FP32;
2. CPU and autocast dtype/shape tests for Pen and RDT profiles;
3. bit-exact disabled-path equivalence and exact-zero initialization output;
4. raw-only, coarse-only and detail-only JVP/VJP checks, including nonzero
   parameter-owner VJP after a nonzero learned state;
5. full loss -> action -> MMDiT -> raw/spine reverse trace with no detach or
   future/label ingress;
6. source-boundary check: the active `PhysicalActionFieldCodec` and
   `NativeTimePhysicalActionTokenLift` are used exactly once; no legacy
   `DCTFlowCodec`, inactive bottom prototype or historical output writer is
   reachable from the new path;
7. optimizer inventory: every new weight appears once in the `bottom_spine`
   group and no existing owner changes group or learning-rate scale;
8. fresh checkpoint round-trip, old-schema exact-resume rejection, RNG and
   pass0/pass1 AMP-cache checks;
9. unchanged two bottom calls per train batch and twelve per deployed action;
10. changed-file static/unit suite, real Pen B8 CUDA VJP, then one Pen B8
    one-batch smoke and read-only checkpoint validation; the user-scoped first
    experiment does not launch an RDT or CALVIN job;
11. runtime/memory comparison against Schema30, including the repeated
    twelve-call deployment cost rather than quoting one isolated forward.

Any failure is repaired at its owning boundary and the complete forward/reverse
review is repeated. Do not compensate with gain, clipping or an auxiliary loss.

### Gate D — one formal pair and its decision

Only after Gates A-C pass, start one fresh Pen run from the new manifest/source
identity. Schema30 remains the baseline; do not train separate degree/K
variants. The usual hard stops remain the only early stops. This first result
is Pen-scoped and does not claim cross-outlet acceptance; RDT and CALVIN remain
unlaunched until a later explicit decision.
Use existing validation budgets for a matched `spine_zero` action intervention
and report coverage, action delta and MSE gain by near/mid/far and arm/gripper.

Accept the semantic unit only when all of the following agree:

- health, ledger, owner VJP, checkpoint, memory and runtime gates stay closed;
- final and complete-curve Pen far/gripper behavior improves or remains stable
  without a near/arm trade;
- `spine_zero` proves the learned view reaches deployed action and does not
  improve matched MSE by removing it;
- detail/event information is retained rather than obtaining a cosmetically
  smooth trajectory by suppressing transitions.

If the branch stays unused, is harmful, or helps only by erasing detail, reject
the unit. Do not rescue it with a hand-tuned fusion scale, stronger loss or
forced coarse/detail allocation.

### Deferred stages: physical units, top use and MIP

B-spine-0 does **not** solve cross-dataset units; it operates in the current
normalized 18-D flow chart. A later B-spine-1 may add a unit-bearing channel
registry only after Pen/RDT/CALVIN source units, controller semantics,
availability and mechanical ranges are verified. That must be a separate ABI
decision and must distinguish physical meaning from train-split z-score
statistics without adding clipping.

The Schema31 pair, if it is reached, keeps the already-validated Pen and RDT-8
adapter contracts unchanged. Native RDT three-camera/depth/14-D bimanual
consumption is a later adapter ABI, not an extra arm of the B-spine experiment;
otherwise a chart result could be confused with a data-interface result.

A top-level spline carrier is also deferred until the bottom unit demonstrates
independent action utility. The first question is whether the bottom can use a
coherent numerical view, not whether S/W can emit another plan object.

The older Minimal Iterative Policy proposal remains archived. There is no
separate established method under that exact name that supplies a missing
ClearVLA consumer; nearby iterative-refinement work addresses inference reuse
or additional denoising. ClearVLA already performs five flow updates per pass,
two complete passes and one W rebuild. Reconsider MIP only if the trained
B-spine exposes an explicit reusable numerical state and a bounded correction
can replace work or consume demonstrably new information. It must not be added
as an extra loop merely because B-spine exists.

## Completion

The recovery-core decision is complete only after Pen reaches E8, all hard
gates remain closed, and the complete curve passes the comparable-anchor and
formal recovery audit. RDT-8 remains a separately judged adapter outlet. A
future Schema31 candidate still requires one fresh Pen+RDT-8 pair under the
same new source identity. At each closure, update the architecture contract,
overwrite superseded ordinary observations, archive only decision-relevant
history, and retain only the evidence needed to select the next semantic unit.
