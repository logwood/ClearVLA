# ClearVLA residual-SAC pilot

Status: implemented research pilot, not a measured RL improvement. It is an
opt-in package. The existing mainline trainer does not import or optimize it.

## Decision and pretraining

Use a **same-task, same-robot ClearVLA behavior-cloning (flow-matching) base**
for ManiSkill `StackCube-v1`, `panda_wristcam`, `pd_ee_delta_pose`. Freeze that
base and its DINO encoder; learn only a small Gaussian residual actor and two
independent Q networks. Fixed alpha SAC is the first controlled baseline.

New bases must use `maniskill_pd_ee_delta_pose_7d_v2` and the repaired data
window contract below. Their default gripper owner is the explicit
`maniskill_binary_command` outlet: a private two-class head emits strict native
`{-1,+1}` commands while the six compatibility future-gripper field channels
are neutralized before every dynamic consumer. The old v1 experiment and an
explicit continuous-v2 run remain retained baselines; neither is silently
relabelled or admitted as a binary checkpoint. Rebuild data,
normalizers/caches and train a fresh base in a new namespace.

The 2026-09-09 read-only inventory of
`/data/senwang/data/clearvla_sim` found official motion-planning trajectories,
official PPO expert weights, and a two-episode controller-conversion probe
(107/94 steps). It did not find a converted ClearVLA expert root or same-task
ClearVLA checkpoint there. This is a scoped inventory, not a claim that no
copy exists anywhere. **Pretrain the StackCube BC base first.** Existing
Pen/CALVIN/LIBERO weights and official PPO weights are not interchangeable
initializations. The PPO expert may later collect demonstrations with separately
verified provenance; it is not loaded by this adapter.

Prepare at least 10 verified demonstrations for a pilot (prefer roughly
50–100 if replay success and coverage permit), episode/reset-seed-disjoint
80/10/10 train/validation/test splits, independent T5/DINO caches and
normalizers. Eight epochs/B8 is an initial budget, not a convergence guarantee.
Select/check checkpoints using held-out closed-loop success as well as BC
validation error. Measure at least 20 unseen reset seeds before RL; the code
requires at least one observed success. This is an exploration feasibility gate,
not a confidence interval or proof that residual SAC will improve the base.
If the baseline never succeeds, improve demonstrations/BC before sparse RL.

## Exact insertion and probability semantics

```text
native causal RGB/state/executed history + fixed task instruction
  -> frozen ClearVLA (unchanged two-pass E5, one W rebuild)
  -> original finalizer and z-score DECODE -> native [24,7] chunk
  -> baseline = clip(chunk[0], native_low, native_high)
  -> unit residual u ~ tanh Gaussian
  -> execute clip(baseline + 0.10 * (high-low) * u)
  -> native environment step -> actual executed action goes into next history
```

For native [-1,1], the maximum correction is 0.2/channel, not 0.1 z-score
standard deviations. Mean residual is exactly zero at initialization; stochastic
training deliberately explores. `--mode base` bypasses the residual entirely.
Once that binary BC base is frozen, no learned change happens inside the
mainline field, codec, G/S/W/P, flow loss, ODE nodes, gripper finalizer or
optimizer during residual SAC. ManiSkill receives its own profile and outlet
identity; it reuses the numerical relative-command W conversion, not LIBERO's
named native controller contract.

SAC's action is **u**, so both Q(z,u) and tanh-corrected log pi(u|z) are in
unit-residual space. Clipping belongs to the transition map, not a fabricated
executed-action density. There is no need for the frozen flow's log probability.
The augmented observation z includes the exact base chunk sampled for this
decision. The next base decision is sampled once, stored, and reused by the
collector; replay never silently redraws it.

The external feature sidecar consumes the already encoded DINO history, not a
second encoder call: fixed seeded 768->32 projection and 2x2 per-camera pooling,
all three visual times, normalized proprioception/action history and the
clipped 24-row base chunk. The 1,027-D vector is a **lossy pilot feature**, not
a reconstructed K/object/precision carrier. The residual has no reward, success,
future state, Teacher or privileged simulator object poses in its input.
Language is admitted through an exact, single-task T5 bank; v1 is not multitask.

## Learning, replay and continuation

- Target: r + gamma * (1-terminated) * [min(Q1_target,Q2_target)-alpha*log pi(u')].
- `truncated` (environment or collector time limit) bootstraps from the actual
  final observation. It never uses the next reset state. True terminals use an
  explicitly unused zero next-feature sentinel and never bootstrap.
- Actor: mean(alpha*log pi(u)-min(Q1,Q2)); twin independent critics use MSE;
  target parameters use Polyak updates. No policy gradient enters the base.
- FP32 learner, checked finite raw gradients, local clipping, one optimizer
  per parameter; dedicated actor/replay RNGs do not alter the base RNG.
- The existing conservative 22-GiB CUDA process gate runs after base decisions
  and updates; current non-PyTorch/shared-GPU use is included, not silently
  subtracted to make the learner appear smaller. CUDA OOM can still precede
  this post-forward check; measure the complete base/simulator footprint first.
- Bounded replay stores current/next features, sampled u, actual command,
  reward, terminated and truncated. 10k transitions are about 80 MiB before
  framework overhead; this is not the full-model or simulator memory budget.
- Adapter snapshots contain actor/Q/targets/optimizers/replay/RNG and exact
  base/normalizer/language/feature/source/environment identity. They contain
  no base weights. Files are immutable and output directories must be empty.
- Resume is an explicit **episode-boundary reset** continuation, not restoration
  of mid-episode physics. Snapshot is written after the requested training budget;
  interruption before it retains episode reports but loses unsaved learner state.
- BC expert reset seeds and baseline evaluation reset seeds are excluded from
  online training. Matched adapter evaluation uses the reserved baseline seeds.

## Preparation and commands

Use `--settle-steps 48 --state-chart fixed_down_causal_rotvec_v2` to record real
zero-EE/hold-gripper simulator transitions after successful source replay.
Admission requires every source action (including release) to be a first-row
target with 48-frame real future support. Centers start at source reset (zero)
and end at source_length-1; no source approach phase is dropped. Unavailable
pre-reset histories repeat initial RGB/state and the safe reset action_state,
exactly as online CausalHistory; they never read the unexecuted first action.
No synthetic future frames are invented. Normalizers still fit training data
only and include the real holds; their new values belong to the new identity.

The v2 7D state uses a fixed down-facing rotation reference plus causal nearest
equivalent log-branch selection, reset per episode. This removes the observed
Panda pi-pose seam without pretending three coordinates are globally nonsingular
on SO(3). The action chart remains native pd_ee_delta_pose, not transformed
relative to the state reference. Both data recording and deployment call the
same state-chart producer.

The old B4 dedicated-event count rounded 0.5 to zero. Fractional mass is now
carried across batches (0/1 alternating), preserving 12.5% over pairs. B8's
integral count/sample sequence is unchanged; scheduled/effective fractions are
reported. ManiSkill's positive gripper delta is named open and negative close,
without changing aggregate event F1 or RMSE.

Fixed source-plus-hold collection may step past a successful task termination
(as the vendor demonstration replay does). `sim/metrics_json` retains the
native terminal and the continuation flag; source provenance names this rule.
This is collection-only: failure or time-limit termination is not suppressed,
and the ordinary RL environment retains its native success terminal. Early
success is not mislabeled as an incomplete failed expert.

1. Convert official demonstrations with the existing ManiSkill controller
   conversion workflow. Use a new source/output namespace; the existing two-row
   probe cannot satisfy the base preparation gate.
2. Set explicit `STACKCUBE_ROOT`, `STACKCUBE_TRAJECTORY`, `STACKCUBE_METADATA`,
   `STACKCUBE_COUNT`, `CUDA_VISIBLE_DEVICES`, simulator asset/model cache paths;
   run `bash scripts/prepare_stackcube_base.sh`. This replays experts, adds real
   holds, audits/splits them, and builds independent artifacts. It does not
   launch training. The default episode time limit is 400; use the same limit
   for BC base evaluation and RL. Missing cached encoder models fail closed.
3. Run bounded real-data CUDA training/checkpoint-validation gates in fresh
   output directories; only then run a fresh full BC job:

```bash
python -m clearvla.mainline.train --config "$STACKCUBE_ROOT/prepared/config.json" \
  --output-dir "$STACKCUBE_ROOT/bc" --device cuda
```

4. Evaluate base, train adapter, evaluate on identical reserved seeds:

```bash
python -m clearvla.rl.train --base-checkpoint "$BASE_PT" --mode base \
  --output "$RL_ROOT/base_eval" --episodes 20 --max-episode-steps 400
python -m clearvla.rl.train --base-checkpoint "$BASE_PT" --mode train \
  --baseline-report "$RL_ROOT/base_eval/evaluation.json" \
  --output "$RL_ROOT/train1" --steps 10000 --max-episode-steps 400
python -m clearvla.rl.train --base-checkpoint "$BASE_PT" --mode adapter \
  --adapter-checkpoint "$RL_ROOT/train1/adapter.pt" \
  --baseline-report "$RL_ROOT/base_eval/evaluation.json" \
  --output "$RL_ROOT/adapter_eval" --episodes 20 --max-episode-steps 400
```

Use the dedicated simulator Python on Linux; no simulator runs from importing
the package. `--t5-condition` and `--dinov2-model` allow explicit local artifact
locations, still verified by the base loader and adapter identity.

## Limits and next gates

This implements online SAC only. Offline AWR/AWAC/IQL warm-start is deliberately
not mixed into v1: an expert action can lie outside the bounded residual support;
clamping it and claiming exact expert imitation is invalid. The existing
simulator HDF5 lacks final next-observation storage suitable for arbitrary
offline RL; it is not directly loaded as SAC replay.

A 0.2 gripper correction cannot reverse an already saturated +/-1 base command.
If failures are missed open/close events or large action errors, improve the
base or design a separately identified gripper/residual-support experiment.
No support expansion, alpha adaptation, model-based W rollout, whole-flow
PPO, decoder unfreezing or safety guarantee is hidden in this pilot.

Before claiming readiness for a formal RL run: perform real-checkpoint zero
residual parity, actual same-robot CUDA rollouts, per-step latency/full process
memory measurement (including DINO reference-batch padding), and at least
three matched RL seeds against the frozen base. Report task success, return,
native saturation, residual size, gradients and wall-clock/sample budget.
Current unit/fake-environment/CUDA learner tests are engineering evidence only.

Diagnostic audit/import/readiness summaries go to stdout rather than additional
retained JSON dumps. Required config, split, dataset/ABI identity, evaluation
gate reports and checkpoint state are not disposable audit files.
