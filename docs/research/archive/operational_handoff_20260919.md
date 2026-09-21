# Historical operational handoff snapshot

Archived: 2026-09-19 UTC. **Historical evidence only; not a live handoff or
execution queue.** Original statements such as “current”, “running”, “do not
start” and listed PIDs retain only their original dated/task-specific meaning.
They cannot authorize or prohibit work in a later task. Do not infer process
liveness or current model selection from this snapshot.

The working copy contained uncommitted observations, so its full text is
preserved here rather than being lost during compaction. This file is not
maintained in parallel with the active handoff. Current task scope and dated
observations are owned by
[ACTIVE_MAINLINE_HANDOFF.md](../auxiliary/ACTIVE_MAINLINE_HANDOFF.md).

Original file: `docs/research/auxiliary/ACTIVE_MAINLINE_HANDOFF.md`.
Original bytes SHA-256:
`de69bd4f3c6cfc383bf26cab64ca11a4d8d82fedcaecc8063b99827c0d72f342`.

## Preserved former text

# ClearVLA active operational handoff

2026-09-18 — structure-first review: the **last evidence-completion round is
closed**. Do not start another model probe, export round, training job or closed-
loop panel. User explicitly authorized direct coordination with 新主执行者
(task `01a09943-12b5-7342-8b9a-d398de7e3a6a`); no need to ask again for that
coordination. It owns source implementation; do not edit its active script,
process or shared model changes independently.

Completed source artifacts at `new_logs/current/20260918-target-y-shift-q5-v2/`:
`causal_ladder.json` and `causal_ladder_tensors.npz`. Reuse these files only.
Independent review and reproducible offline script are at
`artifacts/shared_s_candidate_20260918/causal_ladder_audit.json` and
`audit_causal_ladder.py`. Snapshot/checkpoint/tensor hashes match; BF16 arithmetic
reproduces all 9216 proposal geometry values exactly without model inference.
Entity attribution remains unresolved: joint assignment yields red top K3 /
wrist K2 in both snapshots; the target-K oracle was skipped. Object-to-chart is
a read posterior, not pixel ownership; combined target K/mass are not accepted
entity labels. Per-camera W differences are 0.0628% / 0.0778% of W RMS, whereas
proposal cross-C retention is 0.999750 / 0.988400. These two initial states do
not support strong cross-camera cancellation or establish a unique failure
cause. Initial W applies only to proposal calls 0-5; missing refined W and
terminal weights are final limits, not a reason to request more data.

Joint design review selects one unit, B1: opt-in W camera conditioning from
explicit stable camera identity and existing current camera coordinates, with
the original motion-prior path. Legacy stays default; new condition branch
starts at zero. No G/K assignment, S placement, P2 pooling, coarse-action, codec
or loss change in this unit. Read `STRUCTURAL_REVIEW.md` for typed interfaces,
migration/ABI, producer ownership, zero/invalid behavior, permutation and
train/deploy lifecycle acceptance conditions. 新主执行者 implements/tests within
its user-authorized scope; this task reviews the completed source/test artifacts.
Do not declare implementation or acceptance checks complete before seeing them,
and do not promote the mode or start matched training automatically.

B1 first source implementation is now available in config/policy/top/dynamics.
Independent review passed the 4 added tests and confirmed nonzero-head legacy /
zero-init equivalence in FP32 and CPU BF16, including unchanged construction RNG.
The owner added explicit `--init-model-contract-migration
world_camera_coordinate_role_v1`. Independent tiny-checkpoint review confirms
normal B1 initialization and nine config/state rejection cases without model/RNG
mutation. The inherited-whitelist finding is now fixed: B1 uses exactly the six
reviewed files, and an independently rerun migration regression passes, including
rejection of P2/transition drift without parameter mutation. Ordinary init/resume
was not widened. **B1 source and restricted component-initialization review is
closed with no open source blocker.** This is not default promotion or behavioral
validation. Final fingerprints/limits are in `latest_followup_state.json` under
`b1_final_review`; earlier accepted drift rows in `b1_migration_review_checks.json`
are explicitly marked pre-fix evidence. The same-
checkout legacy control still rejects B1 source drift, so a later matched run
must specify a legal frozen baseline checkout or separate bounded control path.
Await user direction for a matched training design; no renewed evidence probes,
automatic source expansion, training or closed-loop evaluation.

Actual baseline remains `1e51e288c700636ef7225b480c401db1603738ad`, E1/11012,
Q5/Q5, with the shared-S query fix already present and model files matching
frozen intake `22d2380`. Established checkpoint SHA is
`ac989d3c4e38feff106503aac0a0f74bfd8dad63ea20d551c58a7a5ef0ffdc17`.
The owner used a verified ordinary-disk temporary copy of the frozen tmpfs
checkpoint for the final probe. Its separate training-run best.pt had been
overwritten and was rejected; never substitute an unverified similarly named
file. Prior pair audit confirms identical initial robot/history and an 80.004 mm
target shift with at most 14.113 mm pre-contact TCP response; high-alignment
states/histories are different. Do not confuse the earlier proposal-local probe
with this final full Q5 lifecycle or turn channel RMS into causal utility.

Old two-5000-step arms remain unstarted. The old preflight failed at data loading
(exit 1, 0 batch/epoch) because selected converted episodes lacked converter-v2
absorbing terminal padding; PID 121292/122015 were already gone at 15:59 UTC.
Evidence remains in `preflight_failure/`; do not retry, relax the contract or
poll the ended run. Do not modify existing experiments or active pointers.
Heartbeat `clearvla-s` keeps a 30-minute schedule and is quiet without meaningful
new implementation/review evidence. Final evidence gathering must not restart.

2026-09-18 — CALVIN provenance clarification: the task `新主执行者` evaluated
checkpoint `1e51e28`, epoch 1 / step 11012, in the push-color6 Q5 video panel.
Its entire `clearvla/mainline/model/` tree equals candidate intake `22d2380`.
It already has the shared S language-conditioned queries; the missing-query
finding for historical `635213d` must not be attributed to it. The new pair is
fresh full-split training with E5, not continuation of its six-task raw-overlay
Q5 checkpoint. Use that existing checkpoint as an identified behavioral
reference. Exact provenance: `artifacts/shared_s_candidate_20260918/calvin_main_executor_provenance.json`.

2026-09-18 — shared-S candidate implementation is now isolated on branch
`codex/shared-s-pilot-20260918`, final commit `6a646dc491dd3e86674c2aaf575eb2e1a0db2c86`.
Local candidate/evidence: `artifacts/shared_s_candidate_20260918/` (see
`IMPLEMENTATION.md` and `implementation.patch`). Remote code is the clean
worktree `/data/senwang/clearvla/checkouts/6a646dc491dd3e86674c2aaf575eb2e1a0db2c86`;
151 copied files and the 113-file active source digest were verified. The
intake snapshot `22d2380` is separate from this turn's bounded-history,
single-edge S control, probe and checkpoint-retention changes. GitHub was not
pushed. Local checks: 232 unique tests passed, plus 22 key tests in the isolated
candidate. Synthetic language/K probes are structural evidence, not behavior
success; the remote CPU probe timed out and is not counted as passing.

At the original launch, GPU 5 was idle (24110 MiB free). A two-train-batch/one-validation-
batch B8/BF16 preflight was started under the supported environment
`/data/senwang/envs/clearvla-schema30-boundary/bin/python` (Python 3.12.14,
Torch 2.11 package metadata). Initial launcher/trainer PID: `121292/122015`.
Output: `/data/senwang/clearvla/experiments/calvin/20260918-shared-s-preflight-goal-history`.
Before context publication its stdout is the parent directory's hidden
`.20260918-shared-s-preflight-goal-history-starting-*.log`. The earlier HDF5-load
wait is historical; the final failed state above is authoritative. Initial GPU
availability was not a reservation. Any future experiment needs fresh resource
and data-contract checks after its matched structural design is specified.

The bounded two-arm driver is prepared at
`/data/senwang/clearvla/archive/shared-s-pilot-20260918/run_pilot_pair.py`.
It requires successful preflight, exact audited data/language/normalizer/source
identities, a clean checkout, and an available GPU; each arm has a 5000-step
limit and retains 2k/5k checkpoints. It has NOT been started. The earlier user
authorization was followed by the structure-first course correction above;
automatic launch is no longer the current plan. Keep the prepared driver as
historical research tooling. A future training comparison must follow the
reviewed structural hypothesis and match the actual baseline identity.
Original V1/V2 were left running; historical observed V2 E1/8950 (no checkpoint),
V1 E5/109908 must not be represented as a fresh status check.

2026-09-18 17:36 CST — verified-integration intake is now owned by the user-designated
task `增强执行者` (`01a0b3cd-d5f6-7051-b6c6-2b5b3a5af0f6`), with architecture/code
review as the priority. The detailed review and reproducible evidence are in
`artifacts/verified_integration_audit_20260918/REVIEW.md`; raw logs and source
snapshots remain outside architecture memory.

The isolated remote checkout is
`/home/sen.wang/workspaces/clearvla_verified_635213d1/repo`, detached and clean
at `635213d1b489c0ba14ebcf52c319cb333753b2b8`. Trainer PID `76469` and launcher
`72610` use the metadata directory
`/home/sen.wang/workspaces/clearvla_verified_635213d1/logs/calvin_v2_subset_formal_20260917_232011_expz2f7x`;
the formal output is
`/home/sen.wang/workspaces/clearvla_verified_635213d1/runs/calvin_v2_formal_20260917_232011_h_mmd353`.
The live run reached E1/7800, with no completed formal validation or checkpoint.
The downloaded analysis snapshot ends at E1/7700. `subset` refers only to the
two-batch preflight; formal training uses all 12593 train episodes. This remains
the historical direct-arm CALVIN binding V2 experiment, not the accepted
shared-codec/S-language-query mainline. The older V1 run remains PID `3723394`,
at E5/108808 (four completed epochs), from source `34ee78e`.

Historical intake action: prepare a source-reviewed candidate around the existing
shared S repair and accepted action codec. That preparation is complete; the
current next action is the structure-first review above, not another copy of
the existing S repair. Restore decision-critical diagnostics in offline evidence.
Do not infer V2 improvement from the lower total loss: its two V1 binding
objectives are absent. Current training was not stopped, restarted or patched.
V2 shares physical GPU 1 with this user's two bridge processes; its Python
3.10/Torch 2.6 environment differs from the declared 3.12/2.11 contract.
The root filesystem was 97% used, while `/data` was 59% used; place new outputs
under `/data/senwang` and recheck all PIDs/GPU/source identities before acting.

2026-09-11 LIBERO expert-prefix handoff diagnosis completed on the isolated
tree `/data/senwang/clearvla/checkouts/libero-r3-exact-20260911`; the temporary
bridge (PID **1209510**, GPU 5) was stopped after the runs and no training job
was touched.  Local evidence is under `artifacts/libero_timing_20260911/` and
the compact analysis is `expert_handoff_analysis.md`.  The 32-step/12-prefix
pair only tested approach (the source demo does not close until row 37): cold
and handoff both had zero contacts, but final EEF-to-bowl distance improved
from `0.16590` to `0.11749 m`.  The informative 80-step/36-prefix pair gave
the expert branch a `0.13563 -> 0.06669 m` handoff distance, gripper--bowl
contact at step 40 and bowl--plate contact at step 49; cold contacted the bowl
at step 44 but never the plate.  Both task-success flags remained false because
the expert branch held the bowl above/near the plate and never opened through
step 80.  Source inspection shows that `On(bowl, plate)` requires current
contact plus bowl/plate center XY distance `<0.03 m`; the expert contact at
step 49 was still `0.1124 m` apart, while its later aligned rows had no
contact and ended `0.0791 m` above the plate.  The source demo first opens at
row 86, so this 80-step probe also truncates the lowering/release phase.  This
supports reset/approach and post-grasp transport/release as closed-loop issues;
it does not support gripper loss dragging the arm (the same-arm timing A/B was
bit-identical).  The converted demo's fixed-init state is not byte-identical
to official init 0, so treat this as causal diagnostic evidence rather than an
official score or exact replay.

2026-09-11 LIBERO retarget-overlay r3 is complete and retained locally as a
diagnostic artifact; no LIBERO bridge or probe process remains on
`senwang-server`, and no active remote training job was touched.  The run is a
one-epoch, 120-train-batch/four-validation-batch **component/data-contract
initialization** from the terminal-suffix E2 best checkpoint:
`optimizer_load=disabled`, `schedule_load=disabled`, `rng_load=disabled`.
The train-only overlay contains two source demos and four admitted `+/-3 cm`
Y-translation episodes; validation/test membership and action/state normalizer
digests are unchanged.  It is not fresh-from-scratch and not an exact resume.

The retained local outputs are:

- `artifacts/libero_retarget_20260910/training_r3/` (run context, metrics,
  console snapshot and `audit_policy_logs_v2.json`)
- `artifacts/libero_retarget_20260910/closed_loop_audit_r3_v1.json`
- `artifacts/libero_retarget_20260910/closed_loop_r3_v4/`
- `artifacts/libero_retarget_20260910/closed_loop_r3_long_init0_v1/`
- `artifacts/libero_retarget_20260910/overlay_manifest_demo01_pm3cm_v1.json`

The deployment-only timing attribution is also retained under
`artifacts/libero_timing_20260911/`: `timing_probe_v1.json` plus the baseline
and delayed-close MP4s in `timing_probe_v1_videos/`.  Its isolated remote
source worktree was
`/data/senwang/clearvla/checkouts/libero-timing-probe-20260911` (detached at
the r3 source commit); the diagnostic entry points are
`clearvla/benchmarks/libero_timing_probe.py` and
`scripts/probe_libero_timing_attribution.py`.

The re-audited validation is source-native full/arm/gripper RMSE
`0.213688/0.131133/0.465255`, decoded gripper event F1 `0.1153` with `425/26`
predicted/target events and ratio `16.35`; weighted gripper-trajectory loss
contribution is `0.00805` (about `2.4%` of total).  The closed-loop summary is
`9` rollouts, `0/9` successes, with strong close around steps `5--6`, strong
open around step `21` in the applicable traces, and final EEF-to-bowl distance
about `0.171--0.202 m`.  Target-shift signs are correct in `5/6` pairs, but the
probe has no gripper qpos/contact-force channel.  Keep this as evidence for a
timing/feedback alignment diagnosis, not as proof of physical grasp timing or
as an official LIBERO score.

The timing pair used task 0 / fixed init 0, five zero-action warmup steps and
32 policy steps from one post-warmup simulator snapshot.  Baseline bridge
inference was replayed with a second branch that suppressed the first 12
positive close commands while copying arm[0:6] bit-for-bit.  Both branches
failed and made no gripper--bowl or bowl--plate contact.  The arm action
max/RMS difference was `0/0`, gripper command delta RMS was `0.44066`, bowl
displacement was `5.02e-5 m`, and post-step EEF trajectory delta RMS was
`9.10e-6 m`; final EEF-to-bowl distance was about `0.185 m` in both.  This
rejects early-close drag as the leading explanation for this state, while
leaving arm reach and observation feedback as the next attribution target.

Reproduction/checks:

```text
python -m clearvla.tools.audit_policy_logs artifacts/libero_retarget_20260910/training_r3 --format json
.venv\Scripts\python.exe -m pytest -q tests/test_libero_boundaries.py tests/test_libero_retarget.py tests/test_libero_short_probe.py tests/test_libero_terminal.py tests/test_benchmark_config.py tests/test_libero_timing_probe.py
```

Immediate next action, if the user asks to continue experimentation: diagnose
arm reach and observation feedback with one equally bounded intervention (for
example, an expert-prefix handoff or a model-only observation replay) while
keeping the shared codec and formal evaluator unchanged.  Recheck remote GPU
availability and use a new output directory; do not alter protected jobs or
promote the overlay.  The timing result is diagnostic evidence, not a formal
LIBERO score.

2026-09-10 StackCube first-action-event A/B is running on `senwang-server`,
PID **1237090**, physical GPU **1**, in the fresh run
`/data/senwang/clearvla/experiments/maniskill/20260910-stackcube-v2-first-action-event-b4`.
It uses the same v2 data, B4/BF16, workers4, 8 epochs and checkpoint-free
initialization as the completed E8 base, with the only behavioral experiment
switch `sampling_gripper_event_scope=first_action`. The historical control was
`window_any`, which marked a window if any of its 24 target rows had a gripper
event; the A/B reserves the same 12.5% event quota for an event on the first
row actually executed by receding-horizon deployment. Smoke passed with
`event_windows=118`, `event_scope=first_action`, and the formal run was then
started from the same prepared v2 dataset/config. Do not start SAC or alter the
completed E8 run while this A/B is in progress.

Rechecked at 2026-09-10 14:44 CST: the same PID/cwd is alive; E2 training
reached step 3196 and E2 validation is still running (only E1 is complete).
The A/B active-source digest is
`52a084b608d9c13e4d7665947090bb6c3aee2379aaa23c249806b59c5125e00a`.
Its live source matches that manifest. Compared with the old v2 run, resolved
config differs only in output path and event scope; splits, normalizers,
optimizer roles/groups, model manifest and E5/E5 deployment schedule match.
The existing v2 checkout now contains the five-file sampler experiment overlay;
do not mistake its current source digest for the old checkpoint source.

2026-09-11 StackCube approach-sampling A/B completed in separate worktree
`codex/stackcube-approach`; its old PID **3977526** has exited. Final r1 output
is `/data/senwang/clearvla/experiments/maniskill/20260911-stackcube-v2-approach-sampling-b4-r1`.
It keeps the same v2 data, B4/BF16, workers4 and 8 epochs, and reserves
`sampling_approach_fraction=0.125` / `sampling_approach_steps=24` for the first
24 centers of each episode. Smoke and 30 CPU regression tests passed. Formal
run_context confirms 1416 approach windows, 799 approach rows/epoch, 799 event
rows/epoch and motion fraction 0.25. This is a reset/approach coverage test;
do not combine it with first-action-event results until equal completed epochs
and matched closed-loop checks exist.
The completed E8 curve is source-native full/first/arm/gripper RMSE
`0.148591/0.087711/0.045571/0.376954`, decoded event F1 `0.465035`, with
event ratio `2.40476`. A matched seed-1000001 zero-residual 400-step rollout
then had `success=false`, no grasp/stack, min TCP-cube distance `0.08185 m`,
cube height unchanged at `0.020 m`, and gripper saturation `92.5%`. Approach
sampling improves offline error but did not yet improve this closed loop.
CPU replay of the actual sampler shows E1 first-row-event exposures **91 ->
860** (1.42% -> 13.45%) and covered event centers **75/118 -> 118/118**;
E2 exposures are 94 -> 855. This establishes the sampling manipulation, not
the cause of the closed-loop failure. At matched E1, source-native gripper
RMSE improves 0.61049 -> 0.53587, arm RMSE worsens 0.06203 -> 0.07207, and
decoded-event F1 is essentially unchanged (0.06636 -> 0.06637). Do not promote
the candidate or start SAC from these early mixed results. Compare completed
equal-budget curves and matched reset/noise-seed closed-loop rollouts.
An E1 frozen first-row check (54 validation windows, 7 episodes, all 14
switches plus neighboring/hold rows, noise seeds 101/202) detected all 14
switch directions but produced 34/32 false events on hold rows; exact-row
precision was 0.292/0.304. This enriched diagnostic panel is not population
validation F1 or a closed-loop test. It loaded no optimizer, wrote no
checkpoint/diagnostic file, and peaked at a 2.853-GiB process estimate.

The user-requested E2 closed-loop video is complete: 400 steps at reset seed
1000001/noise seed 0, zero residual, no grasp/stack/success. Checkpoint SHA-256:
`d04e50d55623ab1de8099543457813186db46daf2fa759e5fd080051bd62bdf9`.
Local media/text artifacts are under
`runs/stackcube_v2_eval_20260910/closed_loop_e2_seed1000001/`; recorder and
read-only controller replay helpers are in its parent. The 401-frame MP4 is
20.05 s at simulator 20 Hz, not real-time policy speed (median 3.133 s/call).
Minimum TCP/red-cube distance was 0.08329 m; cube height never exceeded 0.02 m;
gripper command sign switched 27 times. Exact replay of the first 200 commands
matched every measured state (max abs 0) and observed 104 IK None returns,
first at step 91. ManiSkill falls back to current joint targets on IK failure,
explaining stationary-arm periods despite nonzero policy commands. This does
not yet identify why targets became unsolvable; it is E2/one-seed evidence,
not final-model quality or a full-episode IK count. Full replay was deferred
because its selected GPU became occupied. The training process was untouched.

2026-09-10 E6 diagnosis completed on an isolated GPU0 process. The frozen
first-action candidate checkpoint (step9588) SHA-256 was
`8f949432b4afe9060bdf82e4139cd3faaf2468b2827611375298c048325cca9d`.
Train/val phase panels each used 70 windows from 7 episodes: mid-approach
translation direction matched the expert well, but pre-close gripper signs
were wrong on 7/7 train and 5/7 val rows for one fixed inference seed. A
seed5/center27 red-only ±4-cm position test kept robot/history bit-exact; X
command correction was weak despite changed RGB/DINO, while Y did respond.
Crucially, a matched-noise seed5 80-step pair found no cold-start grasp, but
27 expert prefix actions followed by policy actions produced a real grasp
at step44, sustained through step80. Both branches had zero IK failures;
handoff state error was exactly0. This prioritizes reset/approach and closed-
loop correction, not a claim that the model cannot grasp or that IK is the
sole cause. One scene/short budget is not a task success-rate evaluation.
Details and reproducible helpers: `runs/stackcube_v2_eval_20260910/expert_diagnosis_e6.md`.
Remote logs are the sibling `20260910-expert-state-localization-e6` and
`20260910-expert-handoff-pair` directories under the ManiSkill experiments
root. No trainer, checkpoint or diagnostic JSON was changed/written.

2026-09-10 StackCube v2 control completed **8/8 epochs, step 12784** around
10:08 CST on `senwang-server`; former PID **4059975** has exited. It ran on
physical GPU **0** (`GPU-97edf7da-61df-e114-dfca-b1ef9ad136eb`), initially
exclusive. Fresh **B4/BF16, workers4, 8 epochs**, no old/smoke checkpoint load.
Its source checkout was
`/home/sen.wang/workspace/robotics/clear/checkouts/stackcube-v2-20260910`,
commit `4be13c2ea59ba73974d703a6a4a8b577f4d05dc5`, saved source digest
`2647eff4d45170760a01c6f7d171da1cfe4e08465f54935c0bb1d89e674e8dcd`.
Run: `/data/senwang/clearvla/experiments/maniskill/20260910-stackcube-v2-bc-b4`;
its `console.log` links to the matching `-launch/console.log`. No code symlink
or existing active pointer was changed. The old v1 PID 3330577 remained running.

V2 data root: `/data/senwang/data/clearvla_sim/stackcube_bc_20260910_v2`.
All 73 real expert replays passed (11,375 frames including 48 real tail steps),
with 7,871 unchanged source actions and identical 59/7/7 episode membership.
All source actions and all 73 releases have first-row centers; no large state
chart jumps remain. Actual window counts are **6389/743/739**. RGB and DINO
caches completed 73/73 with no skipped episodes. Required config is
`prepared/config.json`; the same verified task T5 bank is in `language/t5_xxl.pt`.
Formal CLI explicitly overrides the preparation preset's B8 to B4.

Gates: 122 focused server tests passed, reset/last-source visual parity was
bit-exact, real B4/BF16 2-train/1-val completed with 7.4407-GiB process estimate,
and checkpoint read-only replay passed with source_delta_files=0 and
optimizer/scheduler/RNG loads/checkpoint writes disabled. Final control
validation: normalized action RMSE 0.40320, source-native action/first/arm/
gripper RMSE 0.15290/0.09005/0.03962/0.39272, decoded-event F1 0.44406,
predicted/target events 808/336. Event quota is 799 dedicated windows per epoch
(12.5%) with fractional carry enabled. E8 best SHA-256 is
`7635e55bb73d24d2aa17a505f224dabea4e95cd4f6674146c8e58680bb6adbb9`.
One E8 400-step zero-residual rollout at reset seed 1000001 failed without
grasping; gripper saturation was 93%, policy latency median about 3.006 s.
The earlier E7 rollout used a different seed (1000000) and also failed: these
are two separate checkpoint/seed observations, not a matched success-rate
estimate or a completed 20-seed baseline. Residual SAC has not started.

Per user preference no extra diagnostic JSON was retained; required configs,
split/cache identities and normal training/checkpoint state remain. Only the
code/test archive is local under `runs/stackcube_v2_launch_20260910/`.
Recheck PID/cwd/log destination before any later action. Semantics remain in
the current architecture contract and `clearvla/rl/README.md`.

2026-09-15 Calvin comparison closed the next interface repair in the local
working tree.  The failed StackCube v2 run above was continuous-gripper
training even though its source labels were exact `{-1,+1}`; it remains a
continuous baseline and must not be interpreted as evidence for the new path.
Repaired v2 preparation now defaults to the explicit
`maniskill_binary_command`/`maniskill_7d_binary_v2` outlet: a private command
head owns native dimension 7, six compatibility field coordinates are zeroed
before every dynamic consumer, and finalization/W rebuild write `+1` open or
`-1` close.  Binary admission rejects any non-binary action or action-state row.
The local contract/tests pass, but no remote binary training job has been
started or uploaded in this turn; create a fresh namespace and checkpoint, then
run closed-loop video plus TCP/gripper/contact/IK telemetry before making a
behavior claim.

2026-09-09 StackCube BC base is running on `senwang-server`, PID **3330577**,
physical GPU **4** (`GPU-20cfead0-2eb0-a46f-bb45-dff9a8c9d333`). It shares the
card with the existing Pen job, which was left untouched. The explicitly
recorded allocation choice is **B4/BF16, workers4, 8 epochs, fresh initialization**,
with a 9-GiB PyTorch allocator cap. B8 passed smoke, but a new co-resident job
appeared before launch; B4 was independently smoke-tested and checkpoint-replayed
before the formal start. Shared-card timings are not exclusive-GPU benchmarks.

Source: `/home/sen.wang/workspace/robotics/clear/checkouts/stackcube-rl-20260909`,
commit `b6eaf822fe013caf1f63eead3aa9997e034ea0f4`; active-source digest
`0dad9ae6300c632bfa45d63a5282ad45bb7b8cb83739682fe1b4477cd9526678`.
Run: `/data/senwang/clearvla/experiments/maniskill/20260909-stackcube-bc-b4`.
Its `console.log` links to the matching `-launch/console.log`; `run_context.json`,
`metrics.jsonl`, and future `checkpoints/` are inside the run. No `code` symlink
or existing active outlet pointer was changed.

Inputs: `/data/senwang/data/clearvla_sim/stackcube_bc_20260909_v1` contains 73
replay-successful wristcam experts / 10,207 real frames, including 32 real
post-success hold steps per episode. Episode/reset-seed splits are 59/7/7 and
window counts are 4029/463/459. RGB and BF16 DINO caches completed 73/73 with
no skipped episodes; independent T5 bank is `language/t5_xxl.pt` and the base
config is `prepared/config.json`. The source config's B8 was explicitly
overridden to B4 by the formal CLI and serialized as B4 in run context.

Gates: 101 focused server tests passed; first/tail visual parity was exact;
B8 and capped B4 each completed 2-train/1-val CUDA smoke and read-only checkpoint
replay with source_delta_files=0 and optimizer/scheduler/RNG loads/checkpoint
writes disabled. B4 peak reservation was 6.8965 GiB, conservative shared-card
estimate 19.1837 GiB. No smoke checkpoint initialized the formal run.
Latest verified formal window: E1/step80, loss 1.693943, ledger gap 0,
contribution gap 1.19e-8, raw global gradient 3.660846. Startup evidence only;
no formal epoch validation or learned closed-loop success is claimed yet.
The residual SAC stage has **not** started: first evaluate a formal same-task
BC checkpoint on reserved reset seeds with the zero-residual adapter.
Small context/metric/audit snapshots are local in `runs/stackcube_launch_20260909/`;
checkpoints, RGB, DINO and source HDF5 remain remote. Recheck PID/cwd before action.

2026-09-08 LIBERO single-task bring-up is running on `senwang-server`, PID
4125397, physical GPU 5
(`GPU-7efc36ab-89d2-0c9f-1236-9fe2f2f0b94c`).  It is a fresh Schema-30
`object_intent_dynamics_323` run with the `libero_7d_continuous_v1` outlet,
8 epochs, B8, workers4 and BF16; no smoke checkpoint initializes it.  Source
is the isolated checkout
`/home/sen.wang/workspace/robotics/clear/checkouts/libero-bringup-20260908`.
The run is
`/data/senwang/data/libero/runs/clearvla_libero_spatial_task0_full_v1_20260908_schema30_b8_e8_seed0`
and its external console is
`/data/senwang/data/libero/logs/clearvla_libero_spatial_task0_full_v1_20260908_schema30_b8_e8_seed0.log`.

The data contract is 50 episodes / 5,068 steps with episode-manifest
train/val/test `40/5/5` and window counts `1250/123/95`.  Decoded and DINO
caches both completed 50/50 with no skipped episodes.  The isolated one-row
T5 bank is
`/data/senwang/data/libero/language/libero_spatial_task0_full_v1_20260908_t5_xxl_bank.pt`;
the resolved config is
`/data/senwang/data/libero/configs/libero_spatial_task0_full_v1_20260908.json`.
The profile is `libero_relative_7d_v1`, arm flow uses
`relative_command_adapter`, gripper output is continuous and the raw-unit
event threshold is 0.10.

Bring-up evidence: 32 focused local tests passed; a bounded typed-data smoke
passed; a real B1 2-train/1-val CUDA smoke completed with finite owner
gradients, exact loss-ledger closure and checkpoint-v4 round trip.  The full
run completed E1/step157 and wrote both best/latest checkpoints.  E1
source-native validation RMSE was 0.646328 (arm 0.455666, gripper 1.295525;
first/first8/tail 0.509064/0.579108/0.677441).  Decoded gripper events were
still severely over-predicted at this early point (F1 0.05777, 2756 predicted
versus 83 target); this is a learning observation, not a transport or ABI
failure, and the run continued into E2.  These are offline validation metrics,
not an official LIBERO closed-loop score.  Small configs/reports/log snapshots
are mirrored locally under `runs/libero_spatial_task0_bringup_20260908/`;
checkpoints and the 7 GiB caches remain remote.  No CALVIN/Pen process or
active outlet pointer was changed.

2026-09-08 verified launch: the new Schema28-recovery shared-core Pen is
running at commit `cd43eee988caa39a18bdab19c6e5a48b937dfc42`, PID 3430333,
physical GPU 6 (`GPU-d0924ae6-c618-812b-e4c1-035c0484da14`).
The last checked complete window was E1/batch20/step20: loss 1.675854,
ledger gap 0, contribution gap 5.96e-9. The console subsequently reached
batch37. This is startup evidence only, not validation performance.

Source: `/home/sen.wang/workspace/robotics/clear/checkouts/cd43eee988caa39a18bdab19c6e5a48b937dfc42`.
Run: `/data/senwang/clearvla/experiments/pen/20260908-pen-shared-core`.
`console.log`, `entry.json`, `run_context.json` and `metrics.jsonl` are inside
the run; future checkpoints belong in `checkpoints/`. No `code` symlink was
created, and `active/pen` was not changed.

Fresh 8 epochs, B8, workers4, BF16, unlimited train/val batches, recovery E5.
`task_anchor_v1` and `event_motion_v1` both report active. No checkpoint was
used for initialization. Only this Pen was launched; RDT/multi was not.
Remote checks: 42 focused tests passed, B8 2-train/1-val CUDA smoke completed,
then matching-config checkpoint validation passed with source_delta_files=0
and optimizer/schedule/RNG loads plus checkpoint writes disabled.
Checks: `/data/senwang/clearvla/experiments/pen/20260908-pen-shared-core-checks`.
GPU 6 also had an existing ~1.8 GiB simulation process, left untouched; it was
not claimed to be exclusively idle. Read current process/metric state before
later operations. Older snapshots below are historical, not live status.


Snapshot: 2026-09-06, canonical namespace and live import verified on `senwang-server`.

2026-09-07 launch update: current Pen is the P2 independent-read trial,
PID 235743, commit `c7eccded96a4e784a083f5820b49e5782076d22b`, E1/step40 at
startup verification. Only one new formal run was started (fresh 8 epochs/B8,
Q5/Q5; no resume or smoke initialization). Code is in
`/home/sen.wang/workspace/robotics/clear/checkouts/c7eccded96a4e784a083f5820b49e5782076d22b`;
output is `/data/senwang/clearvla/experiments/pen/20260907-p2-independent`,
with `console.log`, `metrics.jsonl` and `run_context.json` inside it.
`active/pen` points there. The old `5dbe5ee` Pen finished E8 and is retained.
Per the user's storage instruction, source/main files belong in `clear/`,
not `/data`; the older code-root diagram and `workspace launch` directions
below are superseded for source placement and must not be used to create
source checkouts in `/data` or recreate the removed `code` symlink.

This file is the single path contract and volatile process map. It does not
define model architecture. Recheck each PID, `/proc/PID/cwd`, output target and
serialized `run_context.json` before acting. Architecture lives in
[../00_CURRENT_ARCHITECTURE_CONTRACT.md](../00_CURRENT_ARCHITECTURE_CONTRACT.md);
open decisions live in
[../CURRENT_MAINLINE_ISSUES.md](../CURRENT_MAINLINE_ISSUES.md).

## Canonical remote namespace

The only human-facing ClearVLA root is:

```text
/data/senwang/clearvla/
├── repo/          fixed repository used to manage source
├── checkouts/     commit-specific Git worktrees
├── experiments/   canonical outlet/run views and future outputs
├── active/        one current formal-run pointer per outlet
└── archive/       retained inactive evidence
```

- `repo/` and `checkouts/<commit>/` own code. Do not create another full clone
  for every experiment or version label.
- `experiments/<outlet>/<short-run-id>/` is the canonical run location shown
  to people and tools.
- `active/<outlet>` is only a convenience pointer to the selected formal run.
  It is never a model-version, source or checkpoint identity.
- `archive/` retains inactive evidence. Archiving does not merge model
  versions or authorize checkpoint deletion.

Legacy `/home/sen.wang/workspace/robotics/clear/...` paths and existing
`/data/senwang/...` run roots remain physical storage details, not competing
human entry points.

## Experiment view contract

Each experiment exposes these stable data names (plus operational `entry.json`):

```text
experiments/<outlet>/<short-run-id>/
├── console.log
├── metrics.jsonl
├── run_context.json
├── checkpoints/
└── code
```

For the three already-running jobs these entries are links to the
existing physical files and checkout; no live source, output, log or checkpoint
is moved or deleted. Resolve `console.log` from the target of
`/proc/PID/fd/1`; do not guess it from a run name. A view does not rewrite paths
serialized inside `run_context.json`. Treat the views as read-only: a symbolic
link is not a permission barrier, and writing through it changes the original.

Future formal launches write directly to their canonical `experiments/` run
directory and use a commit-specific worktree under `checkouts/`. Model and
resume identity still comes from manifest, resolved config, source digest,
data identity and `run_context.json`, never from the directory name.

## Commands

The root `workspace` entry points to the standard-library tool in the fixed
repository. It neither imports the model nor assumes a training Python path.

```bash
/data/senwang/clearvla/workspace list
/data/senwang/clearvla/workspace show --outlet pen --name 20260906-ws-p2-q5
tail -n 80 /data/senwang/clearvla/active/pen/console.log
```

For a **separately authorized future run**, first fetch the selected source
branch into `repo`, then pass its verified commit explicitly. Example argument
names below are placeholders, not a queued experiment:

```bash
git -C /data/senwang/clearvla/repo fetch origin "$SOURCE_BRANCH"
/data/senwang/clearvla/workspace checkout --ref "$VERIFIED_COMMIT"
/data/senwang/clearvla/workspace launch --outlet pen --name "$RUN_ID" \
  --ref "$VERIFIED_COMMIT" --config "$CONFIG_IN_COMMIT" \
  --python "$TRAINING_PYTHON" --gpu "$FREE_GPU_UUID"
```

`launch` runs in the foreground. For an explicitly requested detached run,
retain launcher stderr as well; do not discard it into `/dev/null`, because
an invalid ref/config/interpreter fails before the experiment is reserved.
The tool captures trainer output itself. During
startup stdout briefly occupies a uniquely named staging file beside the run;
after the trainer writes `run_context.json`, the same open file is atomically
published as `console.log` inside the run. Failed startup also retains its log
and exit code there. This preserves the trainer's empty-output-directory check.
An existing run ID or dirty checkout is rejected; no resume or migration is
implicit. `activate --outlet pen --name "$RUN_ID"` explicitly selects a verified
run and refuses to displace a running or unverified active entry. Launch does
not silently switch the user's active pointer.

`import-run --outlet ... --name ... --output ... --console ... --code ... --pid ...`
adds an existing run without moving it. It verifies context output, PID/cwd and
the live stdout path. `entry.json` records only operational paths/process identity;
it does not replace the model's `run_context.json` or save the shell environment.

## Protected active jobs

| Status | Job | Parent PID | Canonical experiment view | Physical checkout/output |
|---|---|---:|---|---|
| current Pen | Pen P2 independent read | 235743 | `experiments/pen/20260907-p2-independent` | `/home/sen.wang/workspace/robotics/clear/checkouts/c7eccded96a4e784a083f5820b49e5782076d22b`; canonical experiment directory |
| current CALVIN | CALVIN ABC-D selective | 613786 | `experiments/calvin/20260906-abc-d-selective` | `/home/sen.wang/workspace/robotics/clear/clearvla_sim_mainline`; `/data/senwang/data/calvin/runs/clearvla_calvin_abc_d_expanded_selective_v1_20260906` |
| running historical | Pen hybrid v1 | 4030788 | `experiments/pen/20260905-hybrid` | `/home/sen.wang/workspace/robotics/clear/hybrid-v1-pen-20260905`; `runs/hybrid_v1_pen_b8_formal_final_20260905` |

The outlet pointers are:

```text
active/pen    -> ../experiments/pen/20260907-p2-independent
active/calvin -> ../experiments/calvin/20260906-abc-d-selective
```

The hybrid job remains protected while it runs, but it is historical and must
not become `active/pen`. Preserve every DataLoader worker belonging to the
three parent processes. Do not edit their checkouts or clean their outputs and
caches while active.

## Existing retained runs

Five completed run trees, about 27.03 GB total, remain physically under
`/data/senwang/archive/clearvla-runs/2026-09-06/`. Their old checkout-local
`runs` paths are compatibility symlinks:

```text
schema31-bspine-pen-20260904
schema31-arm-only-pen-20260905
schema28-core-recovery-rdt8-formal-20260904
schema28-core-recovery-pen-20260903
schema29-rdt-multitask
```

Expose retained evidence through the canonical `archive/` namespace without
copying it again. Two inactive environments previously removed from
`rdt-multitask-prep/.venv` and `clearvla_v112_fix_belongings/.venv` can be
recreated from their `pyproject.toml` and `uv.lock`.

## Transition and cleanup boundary

- Import an existing active run as a link-only view; never relocate it during
  execution.
- Create future runs directly in `experiments/` and select a tested fixed
  checkout from `checkouts/`.
- Keep one human-facing namespace. Documentation must link here instead of
  publishing another current-path table.
- Treat dirty checkouts, unknown ownership, datasets, caches used by an active
  process, formal checkpoints and serialized contexts as retained state.
- Before any archive or cleanup, verify process identity, checkout status,
  latest file time and destination capacity. Preserve old absolute paths when
  compatibility requires them.
- Keep raw logs and binary artifacts out of Git; document only identity,
  decisions, source references and reproducible audit information.
