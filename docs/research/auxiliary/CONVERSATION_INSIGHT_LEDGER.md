# ClearVLA conversation insight ledger

Status: incremental historical conversation extraction; not an active architecture
contract and not implementation authority.

This ledger records decisions, hypotheses, observations, reversals and durable
engineering lessons from the long-running ClearVLA conversation. It is organized
by conversation order rather than by the checked-out source version. Historical
messages are evidence about what was believed or attempted at that time; they are
not instructions to reapply old code.

Active architecture truth remains in `../00_CURRENT_ARCHITECTURE_CONTRACT.md`.
Current blockers remain in `../CURRENT_MAINLINE_ISSUES.md`. The source-level
V120-to-Schema39 reconstruction remains in `ARCHITECTURE_REPLAY_LEDGER.md`.

## Extraction source and cursor

```text
Codex thread: 019f7501-e0e1-70f0-9f7b-9635d99b6f9a
Thread title: 主执行者 (originally 审查 V94 日志与源码)
Raw rollout:
  C:\Users\ASUS\.codex\sessions\2026\07\18\
  rollout-2026-07-18T19-34-50-019f7501-e0e1-70f0-9f7b-9635d99b6f9a.jsonl

Processed visible-message cursor: 79
Last processed substantive turn: 2026-07-18, old V92/V94 RMSE semantics
Next extraction point: first substantive message after visible item 80
```

The cursor counts non-empty visible `user` and `assistant` message items in the
raw rollout. Injected plugin lists and environment contexts are indexed but are
not treated as project insights. Historical source line numbers below are the
line numbers cited in the conversation at that time; the current checkout may
have moved.

## Segment 001 — V93/V94 execution semantics and evidence discipline

### Segment boundary

```text
Date:                 2026-07-18
Visible messages:     2-79
Starts with:          clean-room static review of the current V94 source/log
Ends with:            interpretation of the old V92/V94 physical RMSE
Primary historical artifacts:
  scripts/current_v94_latent_ownership_execution.sh
  scripts/current_v93_native_dynamic_execution.sh
  clearvla/policy/time_domain_mmdit.py
  clearvla/policy/refinement.py
  clearvla/policy/trunk.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
  attached old V93/V94/V92 nohup logs
```

### CI-0001 — Review the realized graph, not the version story

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The user requested a clean-room V94 audit based only on the current workspace
and the newest available V94 evidence, explicitly rejecting inherited conclusions
and code changes before diagnosis.

【当时的 insight / hypothesis】

An experiment identity is not established by its filename or banner. It is the
composition of the active wrapper chain, final CLI values, serialized policy and
trainer configs, active consumer code and the actual log. Missing evidence must
remain a stated limitation rather than being filled with an older version's
result.

【具体做法】

- Separated code facts from experimental judgments.
- Traced V94 through its wrappers into the active Evidence decoder.
- Compared each claimed switch with the field actually read by the active
  consumer.
- Refused to treat V91/V92 logs as V94 results when no V94 log was initially
  present.

【观察到的结果】

- The workspace initially had a V94 script but no V94 training log.
- V93 printed `[v92]` because it inherited the V92 wrapper, while its serialized
  output directory and `dynamic_block_route=1` identified the actual run.
- Later old logs also lacked a serialized git SHA, so an args-only comparison
  could not uniquely attribute behavior.

【当时的解释】

Several apparently model-level failures were first experiment-identity and
configuration-closure failures. A version label could not prove which path ran.

【后来怎么看（当前读取进度）】

Retained. This became a recurring project rule: verify active script, run context,
source and log together. The full later-history evaluation is pending subsequent
segments.

【状态】

保留。

【证据锚点】

- Conversation messages 2-5, 18-24, 38-43 and 73-76.

【关联】

Experiment identity, source fingerprints, checkpoint provenance and all later
cross-version audits.

### CI-0002 — A declared switch is not a functional contract

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

V94 declared layer-gradient control, transition detach, z probes, learned dwell
and value supervision, but it was unclear whether the active Evidence path and
trainer consumed those values.

【当时的 insight / hypothesis】

A switch is functional only when its complete path closes:

```text
declaration -> wrapper precedence -> parsed config -> serialized owner
-> active consumer -> loss/runtime branch -> diagnostic evidence
```

Interface names, comments and non-null tensors do not establish closure.

【具体做法】

- Distinguished `latent_cvae_layer_grad_scale` from the active
  `layer_contract_grad_scale`.
- Checked whether the active Evidence trunk, rather than only legacy CVAE,
  consumed `latent_cvae_transition_detach`.
- Checked whether the z intervention existed in the active organizer.
- Traced `dwell_mode` into the trainer object read by `flow_losses()`.
- Added a preflight to inspect final arguments rather than wrapper intent.

【观察到的结果】

- V94's historical script declared `0.15`, while the active path still received
  a different layer-contract value.
- The original z probe and transition detach existed only in legacy CVAE.
- V93 serialized `dwell_mode=learned` in policy config, but the trainer lacked the
  field and defaulted to `fixed`; `evval` and value-reader gradient were exactly
  zero.
- Once the trainer boundary was repaired, early V94 showed non-zero value target
  spread, predicted spread and value-reader gradient.
- A later runtime check found an old wrapper could override the newly selected
  transition-detach default, demonstrating why final precedence matters.

【当时的解释】

The model did not fail to learn those functions; some functions were never on
the active path. Logging a requested value was insufficient evidence.

【后来怎么看（当前读取进度）】

Retained as a general closure rule. Whether the particular V94 repairs survived
later designs is outside this segment.

【状态】

保留；具体 V94 接线已被后续架构吸收或替代。

【证据锚点】

- Conversation messages 5, 8-10, 18-24, 68-72 and 73-76.
- Historical V93 evidence: `evval=0`, value-reader gradient `0`.
- Historical repaired V94 evidence: non-zero value loss/spread/gradient by
  batches 20-420.

【关联】

Optimizer ownership, diagnostics truthfulness, serialized manifests and later
producer-to-consumer audits.

### CI-0003 — `transition_detach` changed from attribution tool to rejected default boundary

【时间 / 版本】

Legacy CVAE introduction on 2026-07-13; active V94 discussion and reversal on
2026-07-18.

【当时在解决什么问题】

The active Evidence decoder consumed `controlled_delta`, `rollout_effect_pred`
and `event_context`. The question was whether final action loss should update the
upstream dynamics that produced those transition features.

【当时的 insight / hypothesis】

`transition_detach` changes gradient ownership without changing the forward
value:

```text
detach=1: decoder reads transition evidence, but action loss stops at the boundary
detach=0: action loss also trains the upstream transition/dynamics producers
```

It is neither feature removal nor a decoder freeze.

【具体做法】

- Identified the first historical configuration/legacy use in commit `7f1976c`
  on 2026-07-13.
- Connected the switch to the active Evidence path on 2026-07-18.
- Initially proposed `detach=1` for an ownership experiment plus a `1 vs 0` A/B.
- After inspecting the other gradient discontinuities, changed the default to
  `0` and retained detach only as an explicit compatibility/causality boundary.

【观察到的结果】

- Before the active-path repair, the V94 comment claimed detach while the active
  graph remained attached.
- Making detach real plausibly reduced retained backward activations and GPU
  memory, while preserving forward values.
- The user then rejected a blanket ownership cut: legal upstream information
  should flow forward and receive ordinary action gradients backward.

【当时的解释】

The initial ownership experiment optimized attribution clarity, but it also
prevented the final task loss from correcting weak upstream evidence. Combined
with hard route selection and no-grad candidate probes, it made an already
fragmented gradient graph worse.

【后来怎么看（当前读取进度）】

The same segment superseded the blanket-detach recommendation. The durable rule
is selective isolation: retain stop-gradient only for target leakage, teacher
isolation or an explicitly scoped causal experiment; do not use one global
detach to manufacture ownership.

【状态】

部分保留：显式梯度所有权概念保留；默认整块 detach 已否定。

【证据锚点】

- Conversation messages 11-17, 27-28 and 54-66.

【关联】

Teacher isolation, lawful end-to-end gradients, causal information boundaries
and later complaints about disconnected top-to-action paths.

### CI-0004 — Host depth, operator aperture and compute cost are different quantities

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The logs showed `depth=3`, `effective_depth=8` and an execution cost near `0.011`.
These were being conflated with each other and with actual GPU work.

【当时的 insight / hypothesis】

Three axes must remain separate:

```text
host depth:        number of full MMDiT blocks/iterations
operator aperture: retained low-rank update directions inside an operation
execution cost:    either a declared proxy or measured wall-clock/FLOPs/memory
```

【具体做法】

- Reconstructed the logger field definitions for `evexec` and `evcap`.
- Derived the effective-depth formula from `rank=32`, `groups=4` and `ceil`.
- Traced the cost proxy to `progress * mean(capacity * dwell)`.
- Listed costs omitted by the proxy.

【观察到的结果】

- `effective_depth=8` came from one retained group of width eight; it was not an
  eighth MMDiT layer or one quarter of real compute.
- `depth=3` still meant three host operations.
- The cost proxy excluded fixed host attention/FFN, candidate probes, kernel
  timings, data movement, memory and actual FLOPs.
- Its loss weight was zero, so it was audit-only.

【当时的解释】

The historical logger named a quantized operator-capacity statistic as depth and
a capacity-dwell proxy as cost. Those names encouraged stronger conclusions than
the implementation supported.

【后来怎么看（当前读取进度）】

Retained. A representation-control metric must not be reported as a compute
measurement. Real compute claims require wall-clock, peak memory and/or FLOP
instrumentation that covers all operations.

【状态】

保留。

【证据锚点】

- Conversation messages 29-40.

【关联】

Later runtime/memory audits and the distinction between semantic capacity and
hardware efficiency.

### CI-0005 — `groups=4` hid small, potentially healthy capacity changes

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

Capacity appeared to jump between full rank 32 and effective rank 8. The user
argued that a modest reduction such as `32 -> 29` could be a valid result.

【当时的 insight / hypothesis】

Capacity control should be fine enough to represent small changes, and a model
should not be forced to display dramatic sparsity to prove that the controller
works. Control resolution and actual compute reduction are separate problems.

【具体做法】

- Identified `groups=4` as a convenience/stability heuristic without an
  experiment-backed reason that four was required.
- Proposed continuous ordered transparency during training or one-rank groups.
- Retained hard dispatch only as an evaluation/deployment option.
- Refused to add a cost loss merely to make the rank visibly decrease.

【观察到的结果】

- The hard forward could represent only `0/8/16/24/32`.
- A learned capacity of `0.92` still logged 32 because
  `ceil(0.92 * 4) = 4`.
- Historical V93 reached capacity near `0.01` and quantized effective depth near
  eight; historical V94 stayed near full capacity and depth 32.
- Changing group count alone would not avoid full-rank projection/reconstruction.

【当时的解释】

The quantizer both hid small control changes and made a one-group floor look like
a meaningful learned depth. Dramatic collapse could be a scale shortcut rather
than intelligent compute allocation.

【后来怎么看（当前读取进度）】

The general principle is retained. The segment implemented `groups=32` plus
continuous transparency, but later-history value is not yet assessed.

【状态】

原则保留；具体 V94 实现待后续历史归类。

【证据锚点】

- Conversation messages 34-37 and 44-53.

【关联】

Fine-grained rank control, scale shortcuts, one-sided contracts and later
information-bandwidth discussions.

### CI-0006 — V93's execution controller was a post-block update controller

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The V93 mainline was advertised as learned dynamic execution, but the user
suspected that its control point was structurally misplaced.

【当时的 insight / hypothesis】

If a controller is intended to control execution, its decision must precede the
expensive operation. Applying capacity only to the produced residual controls
the update representation, not whether attention/FFN ran.

【具体做法】

- Expanded the actual wrapper and call chain.
- Traced controller, hard candidate selection, host execution, contraction and
  no-grad probes in their true order.
- Distinguished route, capacity, dwell and exit as separate decisions.

【观察到的结果】

- The dynamic path still ran a fixed three-iteration loop.
- It had current/next and dwell choices but no true skip/stop/no-op.
- The full host block ran before contraction.
- Candidate probes could rerun complete host operations but were omitted from
  the cost proxy.
- Historical route statistics showed little or no next-block usage.

【当时的解释】

The realized mechanism was better described as:

```text
fixed host operations + dynamic block/repeat choice + post-block direction aperture
```

It was not yet compute-aware scheduling.

【后来怎么看（当前读取进度）】

Partially retained. The early repair moved the gate to the residual-writer
boundary, which improved semantic/gradient ordering but still did not skip QKV
or FFN FLOPs. The user explicitly deprioritized compute savings relative to a
correct action-gradient path.

【状态】

部分保留；计算优化当时延期，控制点语义保留。

【证据锚点】

- Conversation messages 34-40, 44-53 and 64-66.

【关联】

Runtime lifecycle, static/dynamic recomputation and later performance audits.

### CI-0007 — Hard routing separated task learning from route learning

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

The value reader had a dedicated candidate loss, but the chosen integer route
and dwell did not receive ordinary final-action gradients.

【当时的 insight / hypothesis】

An `argmin -> integer index -> selected operation` boundary is piecewise constant.
Non-zero controller/value gradients elsewhere do not prove that task loss can
train the discrete selection. Training and deployment may legitimately use
different routing algebra:

```text
train:  soft or straight-through candidate mixture
deploy: hard committed route
```

【具体做法】

- Distinguished capacity-mask gradients from route/dwell-selection gradients.
- Identified the dedicated value target as an indirect path rather than final
  action-loss closure.
- Implemented an attached candidate action chart and softmax mixture for
  training while retaining hard deployment dispatch.

【观察到的结果】

- Unselected candidates were outside the task-loss graph.
- V93's value reader received no dedicated supervision due to the trainer bug.
- Repaired V94 value-reader gradients proved its auxiliary loss was active, not
  that the hard selection boundary was differentiable.

【当时的解释】

The controller could learn candidate scores without learning how small score
changes affect the final action. This made route semantics dependent on an
auxiliary estimator and default/tie behavior.

【后来怎么看（当前读取进度）】

The gradient-closure principle is retained. The exact candidate-mixture
implementation needs later-history review before being treated as a recovery
unit, because it can add memory and duplicate operation paths.

【状态】

原则保留；具体实现被后续方案吸收或替代，尚待回读。

【证据锚点】

- Conversation messages 52-66.

【关联】

Soft training/hard inference, action-path closure, optional route shortcuts and
later per-lane/null routing debates.

### CI-0008 — Cost was deliberately audit-only; capacity collapse was not a cost optimum

【时间 / 版本】

V93-V94, 2026-07-18.

【当时在解决什么问题】

Historical V93 capacity rapidly approached a small value. It was unclear whether
the execution-cost objective caused this and whether a compute loss should be
used to control the mechanism.

【当时的 insight / hypothesis】

At this stage, action quality and gradient closure should determine useful
capacity. Compute cost should be observed but should not force compression. A
rank staying at 32 is not automatically a failure, and a drop to eight is not
automatically a success.

【具体做法】

- Verified `execution_cost_weight=0.0` and detached cost logging.
- Kept the proxy out of both the primary loss and candidate value target.
- Added separate capacity, soft/hard depth, route, dwell and real-runtime
  diagnostics as the desired accounting surface.

【观察到的结果】

- V93 capacity collapsed despite no active cost loss.
- V94 capacity stayed full while learned dwell quickly reached its maximum.
- Historical low capacity therefore behaved more like a learned residual-scale
  or aperture shortcut than a proven compute-quality tradeoff.

【当时的解释】

Without a true pre-operation compute boundary, adding a cost penalty would have
optimized a misleading proxy and risked sacrificing action quality. It would not
have made the fixed host work disappear.

【后来怎么看（当前读取进度）】

Retained for this experimental phase. A future compute objective would require
a measured and causally complete cost surface, not merely a non-zero weight.

【状态】

保留。

【证据锚点】

- Conversation messages 21-37 and 44-66.

【关联】

Execution-cost semantics, audit-only metrics, scale shortcuts and hardware
profiling.

### CI-0009 — Old V93 and old V94 failed in opposite execution modes

【时间 / 版本】

Old V93/V94 logs reviewed on 2026-07-18.

【当时在解决什么问题】

The old logs needed to be classified without confusing them with the newly
modified V94 source.

【当时的 insight / hypothesis】

Controller health is multi-dimensional. Non-zero gradients or a decreasing task
loss cannot establish useful route, dwell and capacity behavior. Each decision
surface must be observed separately.

【具体做法】

Compared progress, capacity, effective depth, dwell, route fractions, value
spread/common-mode, owner gradients and the available validation record.

【观察到的结果】

- Old V93/V92-style run: value reader remained dead, dwell stayed at one,
  capacity collapsed toward `0.01`, effective depth quantized near eight and
  next-block route was effectively unused.
- Old V94: value reader became active, but capacity stayed near one, effective
  depth stayed 32, dwell moved toward two and the value output contained a large
  common mode.
- Both used `transition_detach=1`, `rank=32`, `groups=4` and predated the later
  continuous/attached changes.
- The V93 log stopped before epoch-end validation, so it could not establish
  final effectiveness.

【当时的解释】

The earlier run used capacity as a cheap shrinkage control without a trained
value reader. The later run trained the value reader but mostly learned maximum
dwell/full capacity rather than discriminative routing.

【后来怎么看（当前读取进度）】

Historical evidence only. These logs define failure modes and diagnostic needs,
not a donor architecture.

【状态】

被后续方案吸收为失败模式与日志口径。

【证据锚点】

- Conversation messages 18-24, 41-49 and 73-76.

【关联】

Later controller diagnostics, common-mode collapse, route identifiability and
the rule that complete validation matters.

### CI-0010 — Low aggregate RMSE can coexist with poor action behavior

【时间 / 版本】

Old V92 validation reviewed on 2026-07-18.

【当时在解决什么问题】

The old run reported a seemingly low physical `full_rmse` near `0.098`, raising
the question of whether the policy was already strong.

【当时的 insight / hypothesis】

Aggregate raw-action RMSE is a dimension- and time-weighted average. It can be
made visually small by many arm dimensions, easy early timesteps and conservative
near-zero predictions while hiding gripper events and long-horizon failure.

【具体做法】

- Traced validation to decoded raw physical actions rather than normalized-space
  loss.
- Reconstructed full RMSE from six arm dimensions and one gripper dimension.
- Compared first/first4/first8/tail RMSE and event metrics.
- Checked teacher-forcing and target-action-leak indicators.

【观察到的结果】

```text
full_rmse          0.09818
arm_full_rmse      0.07841
gripper_full_rmse  0.17490
first_rmse         0.0327
first4_rmse        0.0411
first8_rmse        0.0533
tail_rmse          0.1142
tail/first ratio   3.49
gripper F1         about 0.381
event-head F1      about 0.197
pred/target events 2430 / 1357
deploy eligible    0
```

The six-to-one dimension weighting reproduces the reported full RMSE. Evaluation
did not use target action or teacher forcing, and proposal MSE gain was tiny.

【当时的解释】

The policy fit average arm motion and easy early steps, while gripper decisions,
event counts and the tail remained weak. Strong contraction could additionally
encourage smooth conservative actions.

【后来怎么看（当前读取进度）】

Retained. Aggregate RMSE must always be paired with arm/gripper, first/tail,
horizon bands, event/motion and deployment metrics. Training flow loss and raw
validation RMSE are different quantities and must not be directly compared.

【状态】

保留。

【证据锚点】

- Conversation messages 73-79.

【关联】

All later validation comparisons, normalized-versus-physical metric semantics,
conservative gripper behavior and early-platform diagnoses.

## Segment 001 synthesis

The earliest conversation already established the project's most durable audit
logic:

```text
version label is not run identity
declared switch is not a closed path
forward information and backward ownership must be audited separately
representation capacity is not hardware cost
non-zero gradient is not functional selection closure
aggregate RMSE is not policy health
```

It also contains an early warning that later became central: a locally clean
ownership mechanism can damage the global learning path when it removes legal
conditioning or gradients. The within-segment reversal from blanket transition
detach to selective causal isolation should therefore be preserved as an actual
insight, not edited out as a historical mistake.
