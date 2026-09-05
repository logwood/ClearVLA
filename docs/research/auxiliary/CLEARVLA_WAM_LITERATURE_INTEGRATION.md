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
| Iterative action generation (Diffusion Policy, MIP, \u03c0\u2080) | Compare computation depth with behavior, not call count | Direct head vs current flow under the same evidence and W lifecycle |
| Flow solvers (Trajectory-Consistent Flow Matching, Dense-Jump) | Audit integration error and time coverage | Fixed-checkpoint step-doubling and schedule panel |
| Long-horizon/chunk methods (ACT, OpenVLA-OFT, ChunkFlow) | Seam and receding-horizon diagnostics | Consecutive-chunk replay with executed history |
| Lightweight/runtime methods (SmolVLA, Reflex, C\u00b3ache) | Static/dynamic cache partition and latency accounting | Measure encode, W rebuild, dynamic call, solver and robot wait separately |
| Unified outlets (Octo, RDT-1B, \u03c0\u2080) | Explicit outlet adapters, not hidden task conditions | Hold tensors fixed while renaming task metadata; output must remain identical |

These references motivate tests; they do not authorize copying their losses,
solver schedules, extra planners, latent actions or training objectives.

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
[SmolVLA](https://arxiv.org/abs/2506.01844).

The complete historical literature sweep is recoverable from Git history; this
file keeps only the source boundary and the probes that can change a decision.
