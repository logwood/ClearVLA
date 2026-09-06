# Mainline end-to-end source audit and repair

Updated: 2026-09-06. Scope: the active modular Schema28-recovery/raw-only
candidate in `.worktrees/remove-bspine-keep-solver-20260906`, branch
`codex/remove-bspine-keep-solver-20260906`, base `a4a170e6` plus uncommitted
changes. This is not a release or a new learned-performance result.

No training was stopped/restarted, no remote source was changed, and no new
checkpoint or shuffle sweep was requested. The independently dirty root and
another worker's raw-only consolidation are preserved.

## Delivered fixes and candidate classification

| Finding | Source change | Reverse check |
|---|---|---|
| S appended current state to history already ending at zero | `model/intent.py::_paired_history` replaces the last row, checks nonempty/aligned inputs | Last delta is current minus preceding observation; stale duplicate row has no gradient; stationary history stays zero |
| S diagnostics selected a different attention value kernel | Every active S MHA uses `need_weights=False`; detached FP32 Q/K side probabilities supply diagnostics | FP32/BF16 output, loss, parameter VJP and RNG equality with diagnostics off/on; mask and probability tests |
| Deployment checked self-consistency but not current component semantics | `runtime/deployment.py` validates architecture manifest on ABI creation and load | Same-shape old observation/top/bottom/training/runtime identities all rejected; current identity accepted |
| Non-finite loss could pass finite-gradient checks and update weights | `training/engine.py` rejects non-finite total before backward | Explicit NaN/Inf scalar with finite derivative cannot call optimizer, advance schedule/step, or populate gradients |
| W temporal selector lacked explicit typed interval coordinates | Retained/reviewed `typed_interval_qk_v1`, Q/K only; common/near/far metadata aligned | Actual MHA input hooks, unchanged V, zero preservation, causal mask, FP32/BF16 VJP; no added parameters |
| Optional local precision used independent absolute reads from the same pool | Candidate: policy-relative queries and same-modality read differences in `model/v120_p1.py` | Protected carrier exact preservation, zero offsets/empty pixels/equal reads neutral, modality ownership, every typed owner VJP, unchanged state inventory and three local attention calls |

The first four are source-demonstrable implementation/lifecycle defects.
W is an explicit selector-identity repair, not proof of why every W value was
similar. The local P2 change is a structural candidate: prior low action
sensitivity motivates it, but does not establish this mechanism as the cause
or prove a better trained RMSE. It does not alter the separate
`ObjectFutureEffectReader` semantic/geometry W-consumer in `compiler.py`.

## Forward and reverse coverage

All paths below were followed from their active profile/entry points, not from
historical experiment names. Retained inactive monolith/prototype branches
are not claimed as part of the active execution graph.

| Boundary | Forward facts reviewed | Reverse/contract checks |
|---|---|---|
| Data and causal state | `data/dataset.py`, normalizer, interfaces; distinct state/action charts, three observed rows, eight executed-action rows, separate future supports | Current row ownership; profile-owned gripper anchor; no future tensor in online input |
| Conditioning/shared seed | `components.py`, `action_contract.py`, `proposal.py`; history proposal before dropout, downstream exact-null masked contribution, one shared role table | History/goal gradients preserved; zero history cannot enter through proposal residue; no task/episode progress as hidden conditioning |
| Flow-DINO and G | Active raw/progressive profile in `v120_core/profile.py`, `flow_dino_evidence.py`, `restored_observation.py`, `grounding.py`; two directed adjacent solves, normalized coordinates, G2 rematerialization, G3 binding | N=49 and 3x3 factual read retained; one content reconstruction value; conditional allocation is not physical validity; current-only target detach does not detach online values |
| S/coarse action | `intent.py`, `components.py`; public versus typed ownership, four intervals, observed-change lane, 24-row coarse target reduction | Current row and diagnostic repairs; recognizer target detached; coarse S/W condition remains attached |
| W | `dynamics.py`, `types.py`; only current belief plus action condition; common/near/far states and camera support retained | Correct interval-Q/K metadata; no S/goal/Teacher values in W; generic values and validity ownership unchanged |
| P1/local precision | `v120_p1.py`, P1 stage; static 24-query factual read, basis expansion after selection, dynamic residual separate | Policy-relative candidate tested on the actual mainline class; zero pixels and typed neutrality; no extra reader/calls/weights |
| W-effect P2 and P3 | `compiler.py`, compiler stage; separate semantic K and geometry K*C selection, no-null interval terminal, typed consequence, two optional P3 lanes | Exact neutral consequence, zero physical support, spatial/terminal ownership, natural action/typed gradients; no gain/quota/entropy repair |
| Controlled transition | `transition.py`; exact completed G3 source, all 512 selector/value rows, full action basis tokens | Attached action-minus-neutral transition; no synthetic second W or object-mean reconstruction |
| Execution bottom | `restored_bottom.py`, active `components.py` owner and V120 decoder/controller/codec paths; CVAE/workspace, evidence MMDiT, candidate mixing, capacity and terminal heads | Candidate actions attached to task loss; value-regression targets detached; AMP no-grad weight-cache isolation; unique terminal controller; optional deployment prefix reuse remains eval/no-grad only |
| Teacher and objective | `teacher.py`, target stage and `training/losses.py`; no-grad FP32 Teacher, recognizer, common/innovation/adjacent W losses, arm/gripper/motion and execution ledger | No future value in deployment; event-balanced diagnostics not mistaken for active physical loss; hold rows remain supervised; no new losses or weight changes |
| Optimizer and runtime | `training/engine.py`, `optimizer.py`, checkpoint and sampling modules | One parameter owner; decoder-local then global clipping; finite loss and raw gradients before update; warmup progress maintained; two ODE passes, same initial noise, one W rebuild, endpoint read not an update |
| Source/deployment identity | Config, manifest, component selection and core deployment reconstruction | Old same-shape semantic weights fail before model construction; archived replay must use matching source |

## Limits intentionally not relabeled as fixed

- The state/action history pack still uses positional padding and a normalized
  row coordinate. Removing the duplicate current state does not create an
  exact timestamp-matched sparse sensor/action sequence.
- Four W intervals span 48 frames, while the policy proposal contains 24 action
  rows. The final condition therefore uses the last available row. Inclusive
  interval target supports also overlap at boundaries. These are existing
  learning semantics, not a missing gradient or a defect cured by Q/K position.
- One proposal/W/refined pass is a bounded correction, not a fixed point.
  Final action/W mismatch is not eliminated by this audit.
- Geometry amplitude, far-horizon performance and gripper RMSE/F1 are still
  behavior questions. No forced amplitude, extra clipping, entropy/quota term,
  extra training forward or additional W rebuild was introduced.
- The root contains newer CALVIN shared-codec/W-adapter/data work absent from
  this older Pen integration branch. Merge by ownership; do not replace the
  root wholesale or claim these tests certify the current remote CALVIN run.
- Local CPU BF16 tests do not certify production CUDA kernels, peak memory,
  speed or learned behavior. The old checkpoint is not silently reused under
  the changed semantics.

## Validation and reproduction

The focused tests cover direct formulas and parameter VJPs, not checkpoint
shuffle performance. Whole-path diagnostic equality is tested in both train
and eval mode, FP32 and CPU BF16, after execution warmup, with W output heads
opened so neutrality cannot hide the W/P2 paths. Every principal online owner
has a nonzero finite gradient. Fresh checkpoint save/load and optimizer
ownership are part of the broad mainline suite.

Commands from the integration worktree, with `OMP_NUM_THREADS=2` and
`MKL_NUM_THREADS=2`:

```powershell
$auditTests = @(rg --files tests -g 'test_mainline*.py' -g 'test_modular*.py' -g 'test_deployment_candidate_prefix_reuse.py' -g 'test_pen_w_interval_q5_preset.py')
python -m pytest -q $auditTests tests/test_physical_action_codec.py clearvla/action_solvers/flow_solver/tests -rs
```

Final combined result after the second review: **324 passed, 1 skipped in
121.73 seconds**. The one
skip is `test_mainline_amp_cache.py::test_internal_parameter_probes_preserve_cuda_bf16_vjps`:
the local environment has no CUDA device. CPU BF16 and all four train/eval
diagnostic-invariance cases passed. Changed-source/test Ruff and
`git diff --check` passed. No remote CUDA or checkpoint-performance claim is
inferred from these local results.

## Candidate identity

Manifest SHA-256:
`183045c232d64d4169f551fae5a7665835d3069174895b3b02188478e8953675`.
S and P1 component selections identify the new semantics; W retains its
`typed_interval_qk_v1` selection. Parameter/state-key inventory is unchanged.
The Pen candidate retains Q5/Q5 (`t_i=(i/5)^1.25`), five Euler updates per pass,
ten updates plus two endpoint reads total. Training-time sampling is unchanged.
No run of this combined candidate has been launched.

## Second forward/reverse review and deployment preparation

The second review found no additional core numerical/wiring defect. It did
find and repair one launcher inconsistency: checkpoint validation unconditionally
overrode configured data, RGB/DINO cache and T5 paths with Pen defaults.
`scripts/validate_mainline_checkpoint.sh` now follows the training launcher:
config paths are authoritative unless the corresponding environment variable
explicitly requests relocation. Real Bash argument tests cover quoted paths,
both launchers and explicit relocation without executing Python/training.

Additional release-path regressions exercise the actual activation-checkpoint
recomputation branch (including P1's microgrid contraction), with unchanged
forward loss and RNG and matching parameter VJPs (`atol=1e-6`, `rtol=1e-5`)
in FP32/CPU BF16. Complete Q5 proposal/W/
refined sampling is bit-identical with diagnostic and deployment fastpath
switches; every case uses 12 velocity calls and one W rebuild. The memory-policy
test uses the same functions at reduced width/batch, not a production GPU claim.

Publication inventory was checked locally: 93 active source dependencies, all
tracked by Git, and no retired B-spline source in the active closure. The new
Pen config and new regression/report files must be explicitly included in the
candidate commit; no raw logs, caches, checkpoints or root-only WIP belong in it.

Current executable source digest:
`680a9e97538acd2230e221c0b0fa52e66ced4e7c1e4845d86405e1e5d967771b`.
Formal Pen resolved-config digest (excluding relocatable paths):
`394e6b69d2a8597467a7c3b548d2c4bdc75fdda66be10b57b4b47073bdb5ea3f`.
Launcher text is reviewed/tested separately from the Python source digest.

Prepared execution order, not performed during this review:

1. Commit only this integration worktree's reviewed candidate. Publish its
   `codex/remove-bspine-keep-solver-20260906` branch without force pushing or
   updating the busy root/mainline checkout.
2. Use Git to create/synchronize a separate Pen checkout on `senwang-server`.
   Resolve the actual destination and verify commit/source/manifest/config
   digests before launching. Do not copy over running CALVIN/LIBERO sources.
3. Recheck free GPU and the configured Python/CUDA BF16 environment. Run the
   CUDA AMP regression and a Pen B8 smoke using the new candidate config and
   the actual one-forward training engine. Do not use the retained historical
   `probe_schema29_real_batch` cache0/cache1 self-conditioning comparison as a
   representation of the recovery training call graph.
4. Smoke settings: B8, workers 0, `--smoke --max-train-batches 2
   --max-val-batches 1`. Validate its `checkpoints/latest.pt` read-only with
   those same settings and a distinct output directory. The smoke/validation
   resolved-config digests were checked equal locally; no optimizer, scheduler
   or RNG state is loaded for validation.
5. Only after CUDA/B8/read-only checks pass, start one fresh formal Pen run:
   eight epochs, B8, workers 4, Q5/Q5, unlimited train/validation batches, no
   `--smoke`, resume or migration. The old checkpoint is not initialization.

Every phase gets a new output directory and a clearly named sibling `.log`
under `runs/`, not `/tmp`. A sibling log avoids making the still-uninitialized
run directory nonempty before the trainer's fresh-output check. Verify that
both console and `metrics.jsonl` exist after the trainer starts. No remote
operation, candidate commit, push or training launch occurred in this review.
