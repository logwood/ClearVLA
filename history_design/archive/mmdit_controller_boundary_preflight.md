# MMDiT Controller Boundary: Preflight

## Purpose

This document freezes the preparation boundary before changing the native-time
action MMDiT. It is not an implementation plan for adding more controller
features. The immediate goal is to remove ambiguity about update ownership:

- the MMDiT operator produces the action update;
- the controller selects computation mechanisms;
- no semantic controller owns the action residual amplitude.

## Current Tree Status

- Current branch: `codex/v78-lowrank-adaptive-refine`.
- Current `HEAD`: `82700f3`, tagged `pre-controller-spectral-refine`.
- The worktree is heavily dirty and contains the current packaged-policy and
  native-time MMDiT changes as uncommitted edits.
- Therefore `HEAD` is a rollback point, but it is not the exact source of the
  currently discussed V91 path. Before implementation, the current dirty tree
  must be captured as a named local checkpoint or a patch bundle.
- No existing user changes may be reverted as part of this preparation.

## Active Path Observed

The active packaged path is:

```text
V39PolicySystem
  -> policy.trunk
  -> EvidenceLatentMMDiTActionDecoder
  -> TimeDomainMMDiTBlock
  -> ActionOnlyPhysicalVelocityHead / event head
```

The current behavior is:

```text
action = horizon + trajectory + semantic + noisy_lift(x_t)
self_direction = action self-attention
evidence_direction = action-query/evidence-key-value attention
composed_direction = sqrt(0.5) * (self_direction + evidence_direction)
attention_update = shared_gate * composed_direction
action += attention_update
```

This is not the original MMDiT topology. It is a custom split-reader block:
action self-attention plus evidence cross-attention followed by manual residual
composition. There is no current class named `source_router` in the packaged
block, but the source/update split still has the same ownership risk.

The current code also exposes `evidence_scale` and `noisy_scale` as compatibility
inputs. Their default is `1.0`, but they must not become learned semantic
amplitude controls in the new path.

## Hard Boundary

### Controller may control

- attention visibility or source-key bias;
- which operator/rank channels are active;
- block execution and learned dwell;
- operation/block routing;
- memory retrieval and query formation;
- whether a mechanism is available at a particular horizon position.

### Controller may not control

- `controller * noisy_update`;
- `controller * evidence_update`;
- `controller * composed_update`;
- a learned noisy/evidence residual budget;
- the output velocity or event amplitude directly.

The block may retain its own numerically stable time/condition modulation, but
the unified controller must not be wired into the block residual amplitude.
Capacity, route, and dwell must change computation, not merely shrink the
answer.

## Materials Required Before Implementation

1. **Exact source baseline**
   - A local commit or patch bundle of the exact current V91 source.
   - The effective launch script and its resolved arguments.
   - The current model/config missing-key report.

2. **Training evidence**
   - One complete V91 short run, preferably through at least epoch 1.
   - The corresponding validation section, not only the first training row.
   - The source/evidence contribution metrics and their definitions.

3. **Behavior fixtures**
   - Existing policy golden fixture and current self-check output.
   - Deterministic synthetic input for action/evidence stream ablations.
   - A checkpoint or stage-I loading case for the real integration smoke.

4. **Runtime environment**
   - Training-machine Python/Torch environment for CUDA forward/backward.
   - A working linter or formatter invocation. The current Windows `.venv`
     has Python but no Torch or Ruff, so local static verification is limited
     to parsing/compile checks until the training environment is available.

## Required Acceptance Checks

### Static ownership checks

- No controller output is multiplied into an action residual.
- No source-specific residuals are summed outside the MMDiT operator.
- `evidence_scale` and `noisy_scale` are either removed from the active path or
  retained only as fixed diagnostics/ablation controls.
- Controller parameters remain in the optimizer; removing amplitude ownership
  must not remove their route/capacity/dwell gradients.
- Evidence values are not overwritten by action updates.

### Tensor and gradient checks

- Action stream receives `x_t` and time/position information.
- Evidence stream retains typed memory and top-level intent.
- Both streams have finite forward and backward values.
- An evidence ablation changes action predictions without changing action state.
- Controller gradients reach route/capacity/dwell heads, while no controller
  gradient is required to scale the action residual directly.
- Full operator gradients remain nonzero when controller outputs are neutral.

### Training and replay checks

- Existing codec, noise bridge, DCT/manifold, action head, and event semantics
  remain unchanged in the first architecture experiment.
- The first experiment compares the current custom block against a clean
  MMDiT block; it must not simultaneously change DCT, gripper supervision,
  controller retrieval, or dwell schedules.
- Use current golden infrastructure for construction/interface regressions,
  but do not demand exact numerical equality after an intentional block
  replacement. Exact equality remains required for the untouched surrounding
  path.
- Run a short real training smoke before any long experiment.

## Open Decisions To Resolve Before Coding

1. **MMDiT interaction direction**: use fully joint bidirectional attention as
   the original architecture, or preserve a read-only evidence value boundary
   while sharing the joint attention core.
2. **Native block gate**: retain the original AdaLN/LayerScale-style internal
   gate for numerical initialization, or replace it with fixed normalization in
   the first comparison. The unified controller must remain outside this gate.
3. **Controller integration timing**: first establish a clean no-router MMDiT
   baseline, then reattach capacity/dwell/route one mechanism at a time.
4. **Compatibility strategy**: initialize the new block from scratch or provide
   an explicit migration map. Silent partial loading is not acceptable.

## Recommended Order

1. Snapshot the dirty V91 source and record the resolved configuration.
2. Add a pure block-level MMDiT forward/backward test without changing the
   production decoder.
3. Implement the clean action/evidence block behind a new decoder variant.
4. Keep the existing action head, loss, codec, noise, and evaluation unchanged.
5. Run construction, boundary, and synthetic gradient checks.
6. Run one real training smoke and compare all available diagnostics.
7. Only then reconnect capacity, route, and dwell as mechanism selectors.
