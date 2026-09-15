# ClearVLA WAM research map

Updated: 2026-09-05

This is an exploratory research note. It is not an architecture contract,
implementation authorization, benchmark result, or training plan. Current
semantics remain defined by
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md);
open decisions remain in
[`../CURRENT_MAINLINE_ISSUES.md`](../CURRENT_MAINLINE_ISSUES.md).

## Working model

ClearVLA is usefully analyzed as a lightweight world-action model:

```text
current visual/history evidence
  -> S intent and physical proposal
  -> action-conditioned W future
  -> P1/P2/P3 evidence consumers
  -> V120 action generator
  -> deployed physical action
```

This is a functional description, not a claim that ClearVLA is a video
generator. The testable requirement is bidirectional usefulness:

```text
PhysicalActionCondition --zero/shuffle/scale--> W / FutureObjectDynamics
W / FutureObjectDynamics --zero/shuffle-------> final action
```

If only the first arrow is measurable, W may be predictive but optional. If
only the second is measurable, the action block consumes an unproved future
carrier. Neither is world-action closure.

## Current source facts

| Question | Current owner | Boundary |
|---|---|---|
| Current evidence | `policy.encode_online` | One static cache per observation; future evidence is training-only |
| Intent/action proposal | `model/intent.py` | S owns interval intent and the canonical physical condition |
| Future object state | `model/dynamics.py` | W1/W2 own action-conditioned object dynamics |
| Precision/consequence | `model/v120_p1.py`, `model/compiler.py` | P1 retains high-resolution detail; P2/P3 consume named carriers |
| Action generation | `runtime/sampling.py`, V120 bottom | Two complete five-node passes and one W rebuild |
| Outlet semantics | `data/action_chart.py`, `physical_chart.py`, `model/action_codec.py` | Native charts stay outside the shared core |

There are four distinct clocks:

| Clock | Meaning | Rule |
|---|---|---|
| `t` | noise-to-clean flow coordinate | Keep continuous and explicit |
| `tau` | row inside the 24-step chunk | Keep separate from `t` |
| `t_phys` | physical elapsed time | Add only at an outlet boundary when rates differ |
| task phase | contact/transition progress | Treat as a target or diagnostic until causal |

Solver stage `k` and step size `dt` are additional coordinates if an adaptive
solver is introduced. They must not replace `t`.

## Evidence grading

- **R1:** direct robot or established manipulation evidence.
- **R2:** mechanism evidence close to a ClearVLA boundary but with a different
  model, data contract, or runtime.
- **D:** diagnostic method; useful for locating a causal break, not for claiming
  better policy behavior.
- **N:** recent or lightly replicated evidence; use only to motivate a probe.

Internal RMS, attention, reconstruction quality and paper averages never
authorize a source change by themselves. A current-checkpoint intervention or
fresh matched run must connect the proposed mechanism to physical action.

## Literature-to-probe map

| Direction | What it motivates | First falsifiable probe |
|---|---|---|
| Object-centric future models (SlotFormer, DINO-WM, V-JEPA 2) | Compact future/object supervision without pixel generation | W field/identity/transport intervention, then action utility |
| Privileged future and contact targets (FLARE, ForeTime-VLA, PHR-VLA) | Teacher-only phase, transition-time or wrist-contact targets | Stop-gradient target probe; deployment reachability must remain zero |
| World-action models (LaWAM, ImageWAM, AcrossWAM) | Action-to-future and future-to-action reachability | Matched W/action zero, shuffle and scale interventions |
| Iterative action generation (Diffusion Policy, MIP, π₀) | Compare computation depth with behavior, not call count | Direct head vs current flow under the same evidence and W lifecycle |
| Flow solvers (Trajectory-Consistent Flow Matching, Dense-Jump) | Audit integration error and time coverage | Fixed-checkpoint step-doubling and schedule panel |
| Long-horizon/chunk methods (ACT, OpenVLA-OFT, ChunkFlow) | Seam and receding-horizon diagnostics | Consecutive-chunk replay with executed history |
| Lightweight/runtime methods (SmolVLA, Reflex, C³ache) | Static/dynamic cache partition and latency accounting | Measure encode, W rebuild, dynamic call, solver and robot wait separately |
| Unified outlets (Octo, RDT-1B, π₀) | Explicit outlet adapters, not hidden task conditions | Hold tensors fixed while renaming task metadata; output must remain identical |

These references motivate tests; they do not authorize copying their losses,
solver schedules, extra planners, latent actions or training objectives.

## Arm--gripper--dexterity compatibility evidence

The literature does not usually state the exact claim “a scalar gripper is
less compatible than a dexterous hand.” It does, however, converge on the
mechanism behind that hypothesis: arm motion and gripper/contact commands are
often heterogeneous action types, while a dexterous hand is high-dimensional
but can be given a typed joint or latent chart.

| Work | Evidence close to this question | ClearVLA reading |
|---|---|---|
| [Continuous-Discrete RL for Hybrid Control](https://proceedings.mlr.press/v100/neunert20a.html) | Explicitly argues that robotics problems with continuous variables and discrete outputs should be solved in their native hybrid form instead of forcing everything into one action type | A continuous arm plus binary/contact gripper should have separate owners; a scalar gripper is not automatically a normal arm coordinate |
| [Hierarchical Diffusion Policy](https://arxiv.org/abs/2403.03890) | Factorizes task-level pose and binary gripper from low-level joint-trajectory generation; the paper's action description contains a separate gripper action | Direct precedent for arm trajectory generation plus a separate gripper/mode path rather than one undifferentiated diffusion field |
| [GVLA / Gripper-aware VLA](https://arxiv.org/abs/2608.24603) | On 103K demonstrations over five gripper types, uses platform/type/instance tokens and platform/gripper adapter routing. A layer probe finds gripper-type sensitivity rises sharply in the final action-expert layer, so the adapters are inserted there; the reported average rises from 58.38% for its Pi-0.5 baseline to 66.00% | This is the closest architectural match to the proposed ClearVLA insertion point: condition and adapt the last action-generating block while preserving shared earlier evidence. It still requires new tokens, adapters, losses, data and training |
| [Pick-and-place Across Grippers Without Retraining](https://arxiv.org/abs/2502.15613) | Changing only the gripper morphology changes observation/action distributions through TCP offsets and jaw widths; the method adds gripper-aware mapping and constrained denoising | Even a parallel gripper is not a harmless one-dimensional scalar: geometry and execution constraints belong at an outlet adapter |
| [Mirage](https://arxiv.org/abs/2402.19249) | Finds Cartesian state policies can transfer across several arms after dynamics alignment, while a three-jaw target gripper is a notable failure case; cross-painting and state alignment separately address visual and control gaps | Direct evidence that arm compatibility does not imply gripper compatibility. ClearVLA should measure end-effector appearance, kinematics and command semantics as separate gaps |
| [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) | Standardizes many datasets to a nominal seven-dimensional end-effector-plus-gripper interface, while documenting large underlying observation/action differences across robots | Equal tensor width is only a transport schema. Per-dataset frames, units, absolute/relative meaning and gripper semantics still require explicit charts and finalizers |
| [UMI](https://arxiv.org/abs/2402.10329) and [LEGATO](https://arxiv.org/abs/2411.03682) | Reduce the embodiment gap with a deliberately shared physical/tool interface, relative tool-frame trajectories, latency treatment and IK retargeting | Their success comes from enforcing a common interaction interface, not from assuming arbitrary native grippers are interchangeable. This supports a tool/tip role above native outlets |
| [Latent Action Diffusion for Cross-Embodiment Manipulation](https://arxiv.org/abs/2506.14608) | Identifies heterogeneous end-effector action spaces as a cross-embodiment barrier; aligns anthropomorphic hands, a human hand and a parallel gripper in a latent space, then uses embodiment-specific decoders | Shared policy semantics and native actuator decoding should be separated; this supports a role/latent boundary rather than enlarging the shared gripper field |
| [XL-VLA / Cross-Hand Latent Representation](https://arxiv.org/abs/2603.10158) | Reports that a shared latent action space across dexterous hands outperforms raw-joint baselines in its setting | High-dimensional hand control can be made structurally compatible through a typed latent chart; it is not evidence that the current 7D scalar-gripper ABI supports dexterity |
| [Cloak](https://arxiv.org/abs/2606.22836) | A VLA trained on one parallel-jaw gripper transfers zero-shot to another gripper, another arm and a five-fingered hand by separating two gaps: FK/IK tip-pose retargeting handles kinematics, while an augmented geometric wrist-view mask suppresses source-end-effector appearance | This is the closest direct evidence that a hand need not be less transferable than a gripper. The transferable interface is tip geometry plus an embodiment-aware visual boundary, not the source gripper scalar or raw joint vector |
| [CrossDex](https://arxiv.org/abs/2410.02479) | Uses human-hand eigengrasps as a universal action space and retargets them to each robot hand; one policy is evaluated on four embodiments and zero-shot on two unseen embodiments | A high-DoF hand can expose a low-dimensional, typed continuous interface. This is a candidate `HandRole` chart, not a reason to reuse `gripper_field_dim=6` |
| [GenDexGrasp](https://arxiv.org/abs/2210.00722) and [AnyDexGrasp](https://arxiv.org/abs/2502.16420) | Factor embodiment-independent contact/grasp intent from embodiment-specific pose or decision mapping | Contact geometry is a better shared carrier than jaw width or raw joints when the outlet family changes. These works concern grasp generation, so they do not by themselves solve closed-loop chunk control |
| [D(R,O) Grasp](https://arxiv.org/abs/2410.01702) | Represents the desired robot--object interaction rather than using only raw joints; validates cross-hand grasp generation in simulation and on a real hand | Supports an interaction/contact role beside arm motion. Its optimization-based finalization is evidence for an outlet boundary, not a drop-in action head for the current checkpoint |
| [Cross-Embodiment World Models for Dexterous Manipulation](https://arxiv.org/abs/2511.01177) | Defines a shared state/action space using particles and particle displacements across human and robot hands, then plans through a learned world model | Especially close to the WAM view: shared future dynamics can live above native actuator coordinates. Applying it to ClearVLA would require a new target, decoder and checkpoint |
| [FoAR](https://arxiv.org/abs/2411.15753) | Uses a future-contact predictor to gate high-frequency force information and uses different temporal buffers in contact and non-contact phases; simple fusion caused interference | Compatibility is temporal and causal as well as dimensional. Contact evidence should not be copied into every row/channel or fused uniformly throughout the task |
| [FA-RDP](https://arxiv.org/abs/2607.28596) | Uses the same 1.6-second physical horizon at 10 Hz before contact and 30 Hz during reactive contact, with a distilled fast sampler for the latter | Direct evidence against one fixed arm/gripper/contact clock. It motivates an explicit physical-time grid and phase/contact-conditioned execution, but its new training/distillation cannot be used as a current-checkpoint sampler flag |

The resulting distinction is important:

```text
parallel gripper: low dimension, but mixed continuous/discrete/contact semantics
dexterous hand:    high dimension, but potentially typed continuous joint/synergy semantics
```

This is mechanism evidence, not a direct head-to-head ClearVLA result. For the
current model it strengthens three rules: keep binary/contact commands outside
the continuous ODE when appropriate; do not smooth or B-spline the gripper
event path together with arm motion; and do not claim dexterous-hand support by
changing `action_dim` or `gripper_field_dim` alone. Any future hand extension
needs a role-sized chart, an embodiment-specific decoder/finalizer and a fresh
checkpoint.

The stronger combined conclusion is therefore:

```text
compatibility is not inversely ordered by actuator DoF

bad shared interface:
  arm coordinates + source-specific jaw scalar/raw hand joints

better shared interface:
  arm motion role
  + tip/contact/grasp intent role
  + continuous aperture/force state where available
  + discrete closure/contact event where required
  -> embodiment-specific gripper/hand finalizer
```

No cited paper performs the exact controlled comparison “parallel gripper is
less compatible than a dexterous hand under the same ClearVLA backbone.” The
claim remains a ClearVLA hypothesis. It is now a well-motivated and directly
testable one rather than an unsupported intuition.

### ClearVLA compatibility panel

The first panel should not change the current action ABI. Decode the same
checkpoint output normally, but report the existing arm and gripper paths
separately:

| Probe | Measurement | Decision it can support |
|---|---|---|
| Gripper-anchor swap | Pen current-state anchor versus RDT previous-command anchor on matched replay inputs | Whether outlet anchor semantics dominate gripper error |
| Event-boundary replay | Open/close precision, recall, F1, transition-time error and chatter near contact | Whether the continuous gripper flow loses event semantics |
| Role-specific solver panel | Arm endpoint/trajectory error and gripper event/state error for the same schedule | Whether additional refinement helps arm and gripper equally |
| Geometry perturbation | TCP offset and jaw-width perturbations through an external replay/finalizer harness | Whether a gripper-aware kinematic adapter is required |
| Wrist embodiment intervention | Mask or replace only end-effector pixels while holding the scene and action labels fixed | Whether the model relies on source-effector appearance |

Only if this panel implicates the outlet representation should a fresh-schema
experiment compare three matched variants:

1. Current arm flow plus continuous scalar gripper flow.
2. Arm flow plus continuous aperture/state and a separate closure/contact
   event head.
3. Arm flow plus a shared tip/contact latent and an embodiment-specific
   gripper or dexterous-hand finalizer.

Variant 3 is the literature-backed compatibility direction. It is also the
largest change: it requires new supervision, a new native profile, new
checkpoint weights and closed-loop evaluation. None of these variants should
be introduced by widening the present 18-channel physical field in place.

For the proposed final-DiT integration specifically, GVLA changes the prior
from “plausible placement” to “placement with direct published evidence.” A
minimal fresh-schema candidate would keep the shared bottom action stack and
arm field unchanged, then add at its last action block:

```text
platform/type/instance condition
  -> small residual gripper adapter or routed adapter bank
  -> typed GripperRole outputs
       continuous state/width (only where native)
       open/close/hold logits
       optional contact/endpoint sidecar
  -> embodiment-specific finalizer
```

The first experiment should use one small adapter per known outlet rather than
a full mixture of experts. A router is justified only after multiple gripper
families and enough balanced data exist. Gripper identity must also be
counterfactually tested: with visual/history evidence fixed, changing only the
declared gripper must change gripper-feasible behavior without corrupting the
arm trajectory. The GVLA project page currently links its “Code” button to the
unrelated GCA-Bench repository, so this note records paper-level design
evidence, not a verified reusable implementation.

### Current-source insertion seam

The active source already has a narrower seam than a generic new DiT block:

- `EvidenceLatentMMDiTActionDecoder.blocks` produces the shared action-token
  state.
- `TerminalActionController` is the sole owner of candidate and final action
  reads.
- `ActionOnlyPhysicalVelocityHead.forward_with_gripper_state` already preserves
  an arm read from the base tokens and constructs a private gripper state via
  a zero-initialized residual gate. The gripper value/delta heads read that
  private state, but the compatibility-only `grip_extra` head still reads the
  base state; the current six-channel gripper field is therefore not wholly
  private.
- CALVIN's optional binary command head reads that private gripper state, while
  active V120 disables the old generic event head and classifies deployed
  events after the physical codec.

Therefore the safest final-action experiment is not to let a gripper adapter
rewrite the arm tokens. Extend the existing private gripper seam:

```text
final shared action tokens
  |-> unchanged arm physical-velocity heads
  `-> conditioned private gripper adapter
        (platform/type/instance or explicit geometry)
      -> continuous gripper field head where native
      -> binary/event/contact head where native
      -> outlet finalizer
```

This is late conditioning in the sense supported by GVLA. It can protect the
arm readout from a direct write at one fixed terminal state, but it does not
prove that the complete deployed arm trajectory remains unchanged: modified
continuous gripper fields can feed later solver nodes, proposal-to-W rebuilding
and finalization. `grip_extra` also needs an explicit role decision rather than
being silently moved into the private branch. Candidate and final reads must
use the same conditioned terminal owner; patching only the final returned
tensor would make the execution-controller candidates score a different
policy. The fixed-state arm readout and the full-lifecycle arm trajectory must
therefore be measured separately. The current schema forbids outlet identity
inside the shared dynamic bottom, so this seam is a new-schema experiment, not
a compatible modification to a current checkpoint.

## Highest-value read-only diagnostics

Run these in order, with the checkpoint, observation, initial noise and
normalizer fixed:

1. **Outer triangle:** proposal action, rebuilt W, refined action, W-condition
   residual and task/action error.
2. **Solver oracle:** five-step Euler versus step-doubling, 10/20-step Euler,
   Heun and an explicit RK reference; report runtime and memory.
3. **Future-consumer matrix:** interval, object, camera and type interventions
   at W, P2 and the terminal action consumer; report effective rank and energy
   beside target error.
4. **Chunk seam replay:** first/second differences, contact timing, history
   permutation, duplicate-current and future-leak guards.
5. **Latency split:** static encode, W build, one dynamic call, complete two-pass
   solver and robot wait.

These probes must remain read-only. They do not add a loss, W pass, phase token,
adaptive branch or production solver schedule.

## Experiment lanes

### P0 — no training or architecture change

Use the five diagnostics above. Produce compact decision tables only.

### P1 — after the complete Schema30 behavior decision

Consider one isolated hypothesis at a time:

- solver schedule/order only if P0 shows integration error;
- Teacher-only phase/contact target only if future bandwidth is the bottleneck;
- direct action head or MIP only if action computation is the bottleneck;
- history/chunk treatment only if seam replay shows exposure error;
- B-spine only as a representation-only study with its raw path retained.

### P2 — new schema

Only after a P1 result selects one owner may a new schema test add a terminal
controller choice, MIP stages, trajectory-consistent training, native outlet
dimensions, a new visual target, or a cache policy. Never combine B-spine, MIP,
solver, Teacher and native-RDT changes in one run.

## Decision rule

```text
complete current behavior gate
  -> read-only solver / seam / W-consumer diagnostics
  -> identify the failing owner
  -> run one matched source unit
  -> require physical behavior, lifecycle, memory and ABI evidence
  -> only then update the architecture contract
```

Useful primary references include
[Diffusion Policy](https://arxiv.org/abs/2303.04137),
[Octo](https://arxiv.org/abs/2405.12213),
[RDT-1B](https://arxiv.org/abs/2410.07864),
[FLARE](https://arxiv.org/abs/2505.15659),
[V-JEPA 2](https://arxiv.org/abs/2506.09985),
[Trajectory-Consistent Flow Matching](https://arxiv.org/abs/2605.08511),
[LaWAM](https://arxiv.org/abs/2606.15768), and
[SmolVLA](https://arxiv.org/abs/2506.01844),
[Continuous-Discrete RL for Hybrid Control](https://proceedings.mlr.press/v100/neunert20a.html),
[Hierarchical Diffusion Policy](https://arxiv.org/abs/2403.03890),
[GVLA](https://arxiv.org/abs/2608.24603),
[Pick-and-place Across Grippers Without Retraining](https://arxiv.org/abs/2502.15613),
[Latent Action Diffusion](https://arxiv.org/abs/2506.14608), and
[XL-VLA](https://arxiv.org/abs/2603.10158).

The complete historical literature sweep is recoverable from Git history; this
file keeps only the source boundary and the probes that can change a decision.
