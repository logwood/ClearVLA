# Training acceleration lab

Updated: 2026-09-07

This ledger records controlled training-throughput experiments on branch
`codex/training-acceleration`.  It is deliberately separate from the current
architecture contract: no experiment here changes the accepted model
semantics.  Small JSON reports and profiler tables remain in the authorized
remote lab directory; checkpoints, tensor caches, Inductor caches and raw logs
must not be copied into repository memory.

## Acceptance contract

- Primary target: at least `2.0x` training throughput on the same GPU, batch,
  data, seed, dtype, warm-up and audit settings.
- Values below `2.0x` are retained as intermediate evidence, not described as
  completion.
- The total loss, every weighted contribution, raw and clipped gradients,
  parameter update, optimizer state and continuation RNG state must remain
  equivalent.  Diagnostic batches must retain the complete metric surface.
- Positive and negative experiments are kept as individual commits with their
  reproduction command and result artifact.
- TF32 is excluded because it changes the numerical contract.

## Environment

```text
remote host: senwang-server
remote lab:  /data/senwang/clearvla-vlm-sidecar/bases/clearvla-schema29-d8a77a1/.codex_training_acceleration
python:      /data/senwang/envs/clearvla-schema30-boundary/bin/python
PyTorch:     2.11.0+cu130
dtype:       bfloat16
production profiling batch: 8
available experiment GPUs at the time of measurement: physical 3-6
```

The profiler entry point is `python -m
clearvla.tools.profile_mainline_training`.  A representative controlled run is:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. \
  /data/senwang/envs/clearvla-schema30-boundary/bin/python \
  -m clearvla.tools.profile_mainline_training \
  --device cuda --batch-size 8 --num-workers 0 \
  --steps 6 --warmup 2 --repeat-batch \
  --disable-gradient-spike-audit \
  --output runs/profile_baseline_v2_b8_g5.json
```

## Retained results

The percentages below use the matched baseline named in the same row or note;
cross-GPU values are not treated as controlled speedups.

| Experiment | Commit | Batch / GPU | Throughput | Controlled result | Remote artifact |
|---|---:|---:|---:|---:|---|
| Original paired baseline | `c6ebe4c` era | B8 / GPU5 | 4.67223 samples/s | reference | `runs/profile_baseline_v2_b8_g5.json` |
| Candidate-prefix reuse v2 | `c6ebe4c` | B8 / GPU5 | 4.95966 samples/s | `1.0615x` | `runs/profile_candidate_prefix_v2_b8_g5.json` |
| Same-GPU compile baseline | `2e6a234` era | B8 / GPU6 | 4.30986 samples/s | reference | `runs/profile_compile_match_baseline_b8_g6.json` |
| Default Inductor on MMDiT blocks | `4a706e0` / `2e6a234` | B8 / GPU6 | 4.82328 samples/s | `1.1191x` | `runs/profile_compile_default_b8_g6.json` |
| Prefix reuse plus default block compile | `c6ebe4c` / `2e6a234` | B8 / GPU6 | 5.16131 samples/s | `1.1976x` | `runs/profile_compile_prefix_b8_g6.json` |
| Prefix reuse v1 with per-candidate GPU-to-CPU checks | `8d82ee2` | B8 / GPU5 | 1.92252 samples/s | `0.4115x` versus GPU5 baseline | `runs/profile_candidate_prefix_b8_g5.json` |
| `reduce-overhead` block compile/CUDAGraph path | `4a706e0` | B1 / GPU4 | 0.488623 samples/s | negative result | `runs/profile_compile_mmdit_g4.json` |
| Retain audit-only post-global norm | `f1b40b9` | B1 / GPU3 | 0.815009 samples/s | reference | `runs/profile_postglobal_baseline_g3.json` |
| Skip audit-only post-global norm on ordinary batches | `a9fda45` | B1 / GPU3 | 0.854483 samples/s | `1.0484x` | `runs/profile_skip_postglobal_g3.json` |
| Retain all training execution diagnostics, long matched run | pending execution-only commit | B8 / GPU5 | 4.01939 samples/s | reference | `runs/profile_execution_diag_baseline_long_b8_g5.json` |
| Keep only formal execution tensors on ordinary batches, long matched run | pending execution-only commit | B8 / GPU5 | 4.28224 samples/s | `1.0654x` | `runs/profile_execution_only_long_b8_g5.json` |
| Fresh same-GPU baseline | `1c58d89` era | B8 / GPU4 | 4.26010 samples/s | reference | `runs/profile_pre_context_baseline_g4.json` |
| Fresh same-GPU prefix reuse + block compile | `1c58d89` era | B8 / GPU4 | 4.80396 samples/s | `1.1277x` | `runs/profile_pre_context_prefix_compile_g4.json` |

The current best controlled combination is therefore only about `1.20x`, not
the required `2.0x`.  Its peak allocated memory was 10.224 GiB versus 11.269
GiB for the same-GPU baseline.

The execution-only split also has a deliberately retained short-run reversal:
`runs/profile_execution_diag_baseline_b8_g5.json` measured 4.59502 samples/s,
while `runs/profile_execution_only_b8_g5.json` measured 4.42582 samples/s
(`0.9632x`).  Reversing run order and increasing the measured window from five
to ten steps produced the `1.0654x` result above.  The short pair is treated as
timing noise, not discarded evidence.

## Operator evidence

The one-step operator profile at `runs/operator_profile_g4.txt` observed about
9,625 `copy_`, 5,700 `mul`, 3,304 `div`, 2,318 `mm`, and 2,243 `sum` calls.
Its summed GPU operator time was about 229 ms while profiler-mode wall time was
about 4.7 s.  The actionable interpretation is many small kernels plus
Python/dispatch overhead, not one dominant matrix multiply.

## Rejected full-forward compile

Commit `1303dc8` added an opt-in compilation of the complete formal forward
with graph-break fallback.  The 2026-09-06 GPU4 run generated more than 100
graph fragments.  The first material graph break was the Python truth test of
`torch.isfinite(...).all()` in `grounded_intent_effect.py:261`.  The first
training step then failed before the optimizer update with a non-finite global
gradient; `observation.encoder.raw_mask_token` had finite fraction `0.0` and
the global norm was `NaN`.

No throughput JSON was written, because a failed mathematical-equivalence gate
cannot publish a speed result.  The intended artifact name was
`runs/profile_compile_forward_b1_g4.json`; the code remains at `1303dc8` as a
reproducible negative experiment.

## Static MMDiT context reuse (third experiment)

Commit `0ab1864` adds an opt-in common-subexpression path.  For each decoder
call it computes the action-independent global modulation, evidence K/V
projections and causal self mask once per MMDiT block, then reuses those tensors
for candidate/repeat operations.  The tensors remain attached to the autograd
graph; no detach, checkpoint, optimizer, RNG or sampling contract is changed.
The default path remains unchanged unless the profiler flag
`--reuse-prepared-block-contexts` is supplied.

The strict CPU training-step gate passed at zero tolerance for loss, parameter
updates and gradients (`tests/test_training_acceleration_prefix_reuse.py`).  A
hard-routing subset-indexing guard was also added so the opt-in path cannot
misalign precomputed rows after repeated active-row filtering.  Same-GPU CUDA
throughput is intentionally still pending; no speedup is claimed until matched
baseline and optimized runs are completed on one idle GPU.

The profiler also now has an opt-in `--compile-visual-submodules` probe.  It
compiles only the recurrent Flow-DINO and raw-flow tensor modules, leaving the
outer dataclass/mask/progressive-routing graph in Python.  This is a separate
measurement path and has no production-default effect.

For the same reason, `--compile-execution-submodules` probes the recurrent
controller and candidate value reader independently.  It is kept separate
from the earlier whole-forward compile failure so a graph break in one small
submodule does not contaminate the accepted baseline.

`--compile-mainline-blocks` is a third compile probe for the repeated
grounding, P1, world, transition and layer-contract blocks.  It is deliberately
independent from all other flags so a numerical or graph-break failure remains
an attributable experiment.

`--compile-candidate-prefix` finally probes the attached candidate-prefix
chart as one compiled boundary.  This targets Python dispatch between the
already validated prefix operations; it is opt-in because the chart contains
attached gradients and must pass its own update gate.

The `--batched-candidate-prefix` probe uses `torch.func.vmap` over independent
owner chains at the exact identity-contraction boundary.  It stacks live
parameters (not detached ensemble state), so gradients still reach every
registered block.  The local CPU gate is bitwise equal at the identity
boundary when controller source reuse is off.  A separate contraction-bank
vmap and a complete training update at execution progress `0.5` also pass the
`2e-5` relative / `2e-7` absolute floating-point gate.  Full candidate
operation diagnostics still fall back to the reference prefix implementation.
No throughput benefit is claimed before the same-GPU measurement; the first
implementation deliberately remains a retained checkpoint because it stacks
the live parameter bank inside every decision and may therefore be slower.
The follow-up moves that stack and the deterministic-probe module inventory to
one construction per formal decoder forward.  A separate opt-in terminal-row
probe also omits the velocity-head evaluation for the identity candidate whose
result is immediately replaced by the already-computed prefix velocity.  Its
CPU loss, gradients and update pass the `2e-5` relative / `2e-7` absolute gate;
it remains a distinct flag because changing the GEMM batch shape may alter
BF16 rounding and its speed contribution is expected to be small.

The same commit adds an opt-in execution-controller source-lane reuse probe
(`--reuse-prepared-controller-context`).  It shares the projected
global/time/evidence lanes across the three recurrent decisions.  The forward
loss remains bitwise equal in the CPU gate; because the attached autograd graph
accumulates the shared lane in a different order, the update gate allows only
the measured `2e-5` relative / `2e-7` absolute floating-point tolerance.  It is
not enabled by default and will be accepted for throughput only if the remote
BF16 comparison stays inside that bound.

## Full ordinary-step CUDA Graph feasibility

The local RTX 4060 test-sized model now captures the complete ordinary-batch
online encode, Teacher targets, formal velocity forward, loss composition and
backward.  The existing finite check, two clipping stages, AdamW update and
scheduler remain outside the graph.  The capture work removed only host/device
construction and synchronization blockers: device constants are created or
registered on device, fixed candidate topology uses static slices/padding, and
the known neutral owner bypasses dynamic row grouping.  It does not remove a
network block, loss, gradient path, random draw or optimizer owner.

A 3-warm-up / 10-measured-step local run reported:

```text
eager median:        0.6893402 s/step
graph replay median: 0.1804889 s/step
median speedup:      3.8193x
```

The checked runner, including static-input copy, result-metric snapshot,
clipping, AdamW and scheduling, reported on a separate 3/10 run:

```text
eager median:        0.6505594 s/step
graph runner median: 0.1841423 s/step
median speedup:      3.5329x
one-time capture:    1.1925041 s
```

The same checked runner started at step `700` (execution progress about `0.5`,
so the real contraction/controller path is active) reported:

```text
eager median:        0.7050974 s/step
graph runner median: 0.1860824 s/step
median speedup:      3.7892x
one-time capture:    1.2919135 s
```

These are feasibility results, not the production B8 result: they use the
test-sized local model and exclude data-loader transfer.  The checked runner
does enqueue a copy into every static model-input slot, even when the source
batch object is repeated.  Reproduce the two measurements with:

```powershell
$env:PYTHONPATH='.'
D:\CondaData\envs\torch312\python.exe scripts/probe_local_training_cuda_graph.py
D:\CondaData\envs\torch312\python.exe scripts/probe_local_training_cuda_graph_runner.py
D:\CondaData\envs\torch312\python.exe scripts/probe_local_training_cuda_graph_runner.py --start-step 700
```

The strict multi-step gate separately compared eager execution with graph
replay at identity steps `0/1` and non-identity execution-progress steps
`700/701`.  Total loss, every group/contribution/term, raw and clipped leaf
gradients, parameters, AdamW state, schedule/global step, and CPU/CUDA/flow/
condition RNG continuations passed.  The CPU algebraic rewrite tests retain
the existing `rtol=2e-5`, `atol=2e-7` full-update gate.  On CUDA, two separate
eager engines with identical initial state, input and RNG already differed by
up to `2.92435e-7` in a parameter after one update while their loss and all RNG
continuations were exact.  The graph gate therefore uses the measured
CUDA-control envelope `rtol=2e-5`, `atol=1e-6`; an intermittent graph comparison
at the old CPU-only threshold reached `7.84603e-7` after two steps and is
retained as the reason for that calibration, not hidden by an arbitrary
tolerance change.  The observed loss/gradient-norm pairs were:

```text
step 0:   2.4620974064 / 0.5691227317
step 1:   2.5168545246 / 0.6266048551
step 700: 2.4621999264 / 0.5692583323
step 701: 2.5173258781 / 0.6289275289
```

The production-runner gate additionally passed a complete eager diagnostic
step, replay after that diagnostic, a different batch copied into the same
static slots, restoration of the first batch, and an isolated identity-to-
active topology recapture.  Every case retained the same loss surface,
clipped update, optimizer state and RNG continuation within the CUDA-control
envelope; capture count changed from one to two only at the topology boundary.

Reproduce with:

```powershell
$env:PYTHONPATH='.'
D:\CondaData\envs\torch312\python.exe scripts/probe_local_training_cuda_graph_equivalence.py
```

The remaining acceptance work is to expose a checked static-input copier,
retain complete eager diagnostic batches without invalidating captured gradient
buffers, recapture at the identity/non-identity topology boundary, and run the
same-GPU production B8 BF16 comparison.  No `2x` completion claim is made
until that controlled remote result exists.

## Remote 16-file sync and production B8 gate (2026-09-06)

The user-authorized sync was performed into the isolated remote lab
`/data/senwang/clearvla-vlm-sidecar/bases/clearvla-schema29-d8a77a1/.codex_training_acceleration`.
The incoming archive contains exactly the 16 approved paths and has SHA-256
`50f72cce4f14a04a4ec68f444af1b7acc5ea48c57d5f6d8d60153b8a5edf768b` on both
the local and remote hosts.  The 11 pre-existing remote source files were
saved, without overwrite, as
`runs/source_snapshots/pre_sync_7fee2ac_20260906_103428.tar.gz` (SHA-256
`7bee716ecd1e65db2e343663b66996510e6a323407c1b48431198e3e66ee2b58`).

The strict remote CUDA equivalence probe passed on GPU 4 with the boundary
`rtol=2e-5, atol=1e-6`:

```text
equivalent_step 0/1
equivalent_step 700/701
runner_equivalent eager_diagnostic
runner_equivalent post_diagnostic_replay
runner_equivalent copied_input
runner_equivalent restored_input
runner_equivalent active_recapture
cuda_graph_equivalence_ok
```

The probe emitted one PyTorch `AccumulateGrad` stream-mismatch warning during
backward.  It did not fail the gate, but it is retained as a possible
synchronization cost and is not silently treated as a correctness proof.

Matched production B8/BF16 profiles used the same code, config, repeated batch,
15 steps with 5 warmup steps, and GPU 4.  The eager and graph artifacts are:

| Run | Throughput | Step time | Relative to matched eager | Artifact |
|---|---:|---:|---:|---|
| eager | 4.29264 samples/s | 1.86365 s | 1.000x | `runs/profile_cuda_graph_eager_b8_g4_7fee2ac_20260906_103428.json` |
| CUDA Graph | 7.43241 samples/s | 1.07637 s | 1.731x | `runs/profile_cuda_graph_graph_b8_g4_7fee2ac_20260906_103428.json` |
| reverse-order graph | 7.26746 samples/s | 1.10080 s | — | `runs/profile_cuda_graph_reverse_graph_b8_g4_7fee2ac_20260906_103428.json` |
| reverse-order eager | 4.26092 samples/s | 1.87753 s | — | `runs/profile_cuda_graph_reverse_eager_b8_g4_7fee2ac_20260906_103428.json` |

The two orderings average about `1.718x`, so the result is robust but below the
requested `2x` gate (`8.58529 samples/s`, or `0.93183 s/step` for this eager
baseline).  The graph capture setup is one-time (`7.42--8.69 s`) and is not
included in steady-state throughput.

Three already equivalence-gated reuse paths add only a small positive result:
`7.54055 samples/s` (`1.756x`), artifact
`runs/profile_cuda_graph_context_reuse_b8_g4_7fee2ac_20260906_103428.json`.
The batched candidate-prefix variant is slightly negative at `7.50229
samples/s`; artifact
`runs/profile_cuda_graph_batched_context_b8_g4_7fee2ac_20260906_103428.json`.
It is therefore not recommended for the production combination.

Default (non-CUDA-Graph) MMDiT block compilation layered onto the reuse graph
reaches `7.60491 samples/s` (`1.774x`), with `26.69 s` compile setup and
`100.86 s` graph capture; artifact
`runs/profile_cuda_graph_compile_mmdit_context_b8_g4_7fee2ac_20260906_103428.json`.
This is a performance observation only until a matching compiled-path
equivalence gate is added; it is not yet an accepted mathematical-preservation
claim.

A short operator profile (`runs/operator_cuda_graph_context_b8_g4_7fee2ac_20260906_103428.txt`)
measured about `1.031 s` CUDA time per profiled step.  The largest class was
72 SGEMM launches (`168.9 ms`), followed by thousands of elementwise and
reduction kernels.  This explains why dispatch capture alone plateaus around
`1.75x`: the remaining gap is mostly real GPU compute, not Python scheduling.

The three reuse flags were then re-run through the complete remote equivalence
probe (not only the earlier CPU rewrite gate).  With
`CLEARVLA_EQUIV_CONTEXT_REUSE=1`, identity and active steps, diagnostic eager
fallback, copied/restored static inputs, optimizer state, RNG continuation and
topology recapture all passed; the probe again ended in
`cuda_graph_equivalence_ok`.  This makes the `7.54055 samples/s` combination
the accepted short-run path under the current numerical contract; the long
varying-batch envelope below remains a separate acceptance item.

## Inductor submodule-combination results (not accepted)

The profiler guard was extended only for non-nested, default-mode submodule
compiles so their cost and failure modes could be measured without changing the
production default.  The speed candidates were attractive:

| Combination | Throughput | Capture / compile setup | Peak allocated | Status |
|---|---:|---:|---:|---|
| visual + mainline + MMDiT compile + graph | 9.93592 samples/s (`2.315x` vs 4.29264) | 55.29 s / 15.23 s | 8.93 GiB | rejected by equivalence |
| visual + mainline compile + graph | 9.81743 samples/s (`2.287x`) | 121.90 s / 6.69 s | 8.95 GiB | rejected by equivalence |

The first all-submodule attempt (including execution-controller compile) did not
produce a profile: during capture, Inductor constant folding attempted an
`aten.full` CUDA operation and raised `CUDA error: operation not permitted when
stream is capturing`.  This is retained as a negative result rather than
silently dropping the experiment.

The graph-side compiled-vs-uncompiled strict gates explain why the fast numbers
cannot currently be deployed:

* all compiled blocks: `loss.total` differed by `9.9087e-4` (relative
  `4.024e-4`), above the `2e-5/1e-6` gate;
* visual + mainline: `loss.total` differed by `9.1863e-4` (relative
  `3.731e-4`);
* mainline only: `loss.groups.execution` differed by `4.4730e-5` (relative
  `1.480e-3`);
* visual only: `loss.total` differed by `9.5916e-4` (relative
  `3.896e-4`).

The likely cause is Inductor reassociation/fusion changing BF16/FP32 reduction
order, not a host-dispatch issue.  These runs are useful speed ceilings, but no
compiled submodule path is accepted as mathematically equivalent until a future
deterministic/fused kernel implementation reproduces the current reduction
contract.

## Long varying-batch stress (2026-09-06)

The short gate deliberately reuses a batch, so an additional long probe was
added.  It creates two independent but elementwise-identical copies of a new
synthetic batch at every update, uses B4 for 256 updates (`1024` sample slots),
and crosses the step-200 identity-to-active recapture boundary.  The first
version accidentally passed the same Python batch object through eager and
graph; that aliasing was fixed before interpreting the result.

With the corrected independent inputs, the strict per-tensor CUDA gate first
failed at step 5 for the recommended context-reuse graph path: a tiny raw-flow
gradient element differed by `1.359e-6` absolute.  The pure graph path showed a
similar step-5 difference of `1.673e-6`.

The non-aborting collect run completed all 256 updates.  The context-reuse
graph path reported strict-envelope mismatches on `253/256` updates and
recaptured exactly twice (identity plus active).  This count is a diagnostic,
not an acceptance failure by itself, because the reference control below has
the same long-run issue.

This is not yet evidence that CUDA Graph changes the algorithm.  A paired
control using two independent eager engines and the same changing batches also
failed the old `rtol=2e-5, atol=1e-6` rule at step 3, with a `4.222e-6` absolute
gradient difference.  A relaxed diagnostic envelope (`rtol=1e-3,
atol=1e-5`) reached step 20 before seeing a `2.356e-5` absolute difference.
The long-run gate therefore needs calibration against an eager-vs-eager
control (and preferably deterministic-kernel settings) before it can be used
as an acceptance criterion.  The short repeated-batch result must not be
described as 1000-sample long-horizon proof.

For the same 256-update/B4 sequence, two independent eager engines (no graph)
completed all `1024` slots and reported strict-envelope mismatches on `247/256`
updates.  Thus the graph count is close to the natural eager-vs-eager CUDA
variation at this granularity; the current per-element gate is a useful
tripwire, but not a calibrated long-run equivalence metric.  A real-data B8
long run without `--repeat-batch` is still a separate pending test.

### Cause isolation and calibrated envelope (2026-09-06)

The strict tripwire was not enough to explain the long-run result, so the probe
now collects every result, gradient, parameter, optimizer-state and RNG tensor
even after the first failed comparison.  It also verifies that the independently
generated eager/graph batches are elementwise equal and do not share storage.
The two full runs below use PyTorch `2.11.0+cu130` on an RTX 3090 (GPU 2),
`B=4`, `256` updates, and the same `1024` synthetic sample slots:

| Comparison | First gradient/result step outside `rtol=2e-5, atol=1e-6` | Steps outside (gradient / result) | Worst gradient `max_abs` | Final parameter `max_abs` | RNG |
|---|---:|---:|---:|---:|---|
| eager vs independent eager | 3 / 10 | 252 / 244 | `2.86523e-3` | `1.58325e-4` | exact |
| context-reuse CUDA Graph vs eager | 3 / 10 | 253 / 236 | `2.86074e-3` | `1.60743e-4` | exact |

The corresponding worst gradient relative-L2 envelopes were `0.02410` and
`0.02455`; loss absolute maxima were `2.26498e-5` and `1.43051e-5`.  Both
paths recaptured the identity/active topology exactly as intended (Graph:
`capture_count=2`), and both kept global step and scheduler index equal.  The
near-identical envelopes mean the Graph is not the source of the long-run
drift; it is within the natural CUDA eager-vs-eager variation on this stack.
The complete logs are retained in the authorized lab as
`runs/long_drift_stats_eager_b4_g2_e8b4ef2_20260906.log` and
`runs/long_drift_stats_graph_context_b4_g2_e8b4ef2_20260906.log`.

The operator cause is now identified rather than inferred.  Enabling
`torch.use_deterministic_algorithms(True)` fails during the first training
backward with:

```text
grid_sampler_2d_backward_cuda does not have a deterministic implementation
```

With `warn_only=True`, the same run reports the active non-deterministic
backward paths as `grid_sampler_2d_backward_cuda`,
`adaptive_avg_pool2d_backward_cuda`, and (when enabled) memory-efficient
attention backward.  The active observation compiler contains many CUDA
`grid_sample`/pooling sites in the Flow-DINO/raw-flow path.  As a causal
control, freezing only `model.observation` and forcing math attention reduces
the 16-step eager-vs-eager envelope to at most `5.68e-14` gradient difference,
zero loss/parameter/RNG drift at the configured gate.  Freezing observation
while leaving the default attention kernel gives only about `3e-8` gradient
differences.  Therefore the dominant source is the trainable observation
compiler's CUDA sampling/pooling backward, not Graph replay, random-number
misalignment, or AdamW.

Two narrower 8-step controls identify the submodule: freezing only
`encoder.flow` still reaches `4.12e-5` gradient difference by step 7, whereas
freezing only `encoder.raw_flow` keeps the maximum gradient difference at
`1.91e-7` and never crosses the strict gate.  The first control leaves the
raw-pyramid/refinement path active; the second removes its trainable backward.
The practical suspect is therefore `_RawPyramidFlow` (its dense refiners and
cycle/warp `grid_sample` plus resize/pooling operations), not the coarse
latent Sea-RAFT branch.

This does not invalidate the speed result or imply a semantic change: it means
that exact bitwise long-horizon equality is impossible for the current CUDA
operator contract unless the observation backward is replaced by a deterministic
implementation (or its training path is deliberately excluded).  The safe
acceptance rule is consequently: retain the short strict graph gate, report the
eager-vs-eager envelope beside every long run, and do not call the `1.76x`
path bitwise-equivalent over 1000 samples until a real-data B8 envelope is
measured.

The extended collector is reproducible with:

```bash
PYTHONPATH=.:/data/senwang/envs/clearvla-schema30-test/lib/python3.12/site-packages \
  CLEARVLA_EQUIV_LONG_STATS=1 CLEARVLA_EQUIV_LONG_STEPS=256 \
  /data/senwang/envs/clearvla-schema30-boundary/bin/python \
  scripts/probe_local_training_cuda_graph_equivalence.py
```

## Retained execution-diagnostics split

Separate tensors required by the formal execution-value objective from
detached decoder/controller diagnostics.  An ordinary training batch still
needs the candidate value field, candidate velocity chart, validity mask and
baseline velocity, but it does not need attention entropy, norm dictionaries,
capacity/dwell audit reductions, hard-policy audit selection, quantiles or
correlations.  A logging batch must continue to compute all of them.

The first implementation of this split passed an exact CPU update gate: total
loss, all loss groups and contributions, raw leaf gradients captured during
backward, clipped gradients, parameters after the optimizer step, complete
optimizer state and the global/flow/conditioning RNG continuations were equal
at zero tolerance.  The bounded diagnostic call retained its complete metric
surface.  Reproduce with:

```bash
PYTHONPATH=.:/data/senwang/envs/clearvla-schema30-test/lib/python3.12/site-packages \
  /data/senwang/envs/clearvla-schema30-boundary/bin/python -m pytest -q \
  tests/test_training_acceleration_execution_diagnostics.py \
  tests/test_training_acceleration_prefix_reuse.py
```

## Implementation-level portable acceleration contract (2026-09-06)

The goal of this layer is compatibility across implementation branches, not a
deployment/online behavior change.  The CUDA-Graph runner now consumes a small
adapter contract rather than reaching directly into the active decoder:

```text
versioned model -> TrainingAccelerationAdapter -> shared backend
                                      ├── eager fallback
                                      └── CUDA Graph capture/replay
```

The adapter owns only version-specific capture preparation and the current
topology identity.  The shared backend owns static batch copying, RNG
registration, capture lifecycle, gradient buffers and optimizer boundaries.
The mainline adapter preserves the already-gated neutral-owner and candidate
prefix rewrites, so this refactor does not change the model objective or
checkpoint ownership.

Capture keys now include:

* adapter identity;
* a model structure signature (module classes, parameter/buffer shapes and
  dtypes, without parameter values);
* the adapter topology signature; and
* the existing typed batch signature.

A branch that replaces a block stack or width calls
`CudaGraphTrainingStepRunner.invalidate()` at a step boundary; the old graph is
released and the new branch is captured under its own signature.  Ordinary
optimizer updates do not invalidate the graph.  A branch without a bespoke
adapter receives a safe generic static adapter and can still use the backend if
its training surface is static.

This makes the acceleration mechanism reusable across versions, but it does
not promise a fixed absolute samples/s value: changed FLOPs, memory traffic or
dynamic topology still change the model's intrinsic cost.  Every branch must
run the same short eager/Graph equivalence gate and a matched throughput
profile before promotion.

The read-only source preflight is:

```bash
python scripts/probe_training_acceleration_compatibility.py --pretty \
  . ../schema28-core-recovery-pen-20260903 ../v86-slot-controller \
  ../pen-bspline-routing-20260906 ../rdt-data-adaptation
```

On the 2026-09-06 worktrees it classified the active acceleration branch as
`ready-portable-mainline`; the Schema28/Pen and RDT mainline branches as
`mainline-needs-backend-backport`; and the older V86 branch as
`legacy-surface-needs-bridge`.  This is intentionally a compatibility
preflight, not a speed or behavior claim.  It tells us which thin adapter is
needed before a branch is sent to the GPU gate.

### Adapter refactor verification

Commit `7a8a52f` passed the new contract tests, Python compilation and Ruff
checks.  On the local RTX 4060 Laptop (PyTorch `2.11.0+cu126`), the existing
short CUDA-Graph gate still ended with `cuda_graph_equivalence_ok`.  The local
synthetic B2/FP32 feasibility profile measured eager median `0.7252 s/step`
versus Graph replay median `0.1810 s/step` (`4.01x`); this is a regression
check for the refactor, not a replacement for the production B8/BF16 number
(`7.54 samples/s` on the authorized RTX 3090 lab).  The remote production
profile was not overwritten or rerun by this commit.

### Capture-precondition diagnostic (2026-09-06)

An older remote Graph snapshot initially aborted before recording any kernels:
the raw-flow activation-checkpoint wrapper called CUDA RNG state capture, and
the active execution candidate probe called `torch.random.fork_rng`.  Both
operations are legal in eager mode but `current_seed/get_rng_state` is
forbidden while a CUDA stream is being captured.  This is a capture precondition
failure, not evidence that raw-flow arithmetic is invalid or that the model
cannot be benchmarked.

For a diagnostic-only run, the checkpoint wrapper was changed to
`preserve_rng_state=False` and the candidate probe kept its existing module
training-state save/restore while omitting the `fork_rng` state query.  The raw
image pyramid contains only deterministic convolution/normalization/activation
operators; a local forward/backward check with and without checkpoint RNG
preservation returned `raw_checkpoint_rng_equivalence_ok`.

On the remote RTX 3090 (GPU 3), this capture-safe diagnostic reached one graph
capture and measured:

| Path | Throughput | Capture setup | Status |
|---|---:|---:|---|
| matched eager fastpath | `4.54067 samples/s` | — | timing baseline |
| Graph + capture-safe RNG diagnostic | `6.90903 samples/s` | `6.27 s` | `1.522x` diagnostic speedup |

The first identity steps of the strict gate matched.  The active-step portion
then exceeded the existing `rtol=2e-5, atol=1e-6` envelope at
`observation.compiler.encoder.flow.correlation_temperature_log` with maximum
absolute gradient difference `9.55e-6`.  The result is therefore retained as a
successful capture/performance diagnostic, not promoted as a formal
mathematical-equivalence pass.  Full output is in the remote lab under
`compat_versions/schema29_accel_eda7578/results/`:

* `graph_rng_probe_safe_test.json`
* `eager_matched_test.json`
* `equivalence_rng_probe_safe_v2.log`

The production implementation should express this as a version adapter
contract (or use a dedicated RNG-safe path), rather than relying on a global
test monkeypatch.  A branch may still be benchmarked when this precondition
fails; only the corresponding formal equivalence claim is withheld.

## Pen and RDT versioned equivalence acceptance (2026-09-07)

The acceptance was run independently against the two requested source
revisions; Pen results were never used as a proxy for RDT:

| Variant | Revision / layout | CPU golden coverage | CUDA short-gate result |
|---|---|---:|---|
| Pen B-spline | `a4a170e` / modular | `15,014` tensors, `10,443,735` elements, zero differences at `atol=rtol=0` | `cuda_graph_equivalence_ok` (4/4 repeats) |
| RDT | `f8d0c2f` / legacy | `15,023` tensors, `10,443,744` elements, zero differences at `atol=rtol=0` | `cuda_graph_equivalence_ok` (3/7; 4 natural-tail stops) |

The CUDA gate uses the production `CudaGraphTrainingStepRunner`, with
`rtol=2e-5, atol=1e-6`.  It covers both the identity (`step=0`) and active
(`step=700`) topology, two captures, copied and restored static inputs, the
complete loss ledger/metrics, raw and clipped gradients, AdamW state,
scheduler/global step, CPU/CUDA/flow/condition RNG, and an exact
`save_checkpoint`/`load_checkpoint_exact` continuation.  The checkpoint
continuation reported `rng_exact=true` and a fresh capture after restore for
both variants (Pen `12,213,030` bytes; RDT `12,188,710` bytes).

The long diagnostic then used `256` independent one-update cases at B4 (`1,024`
sample slots), split evenly between identity and active topology, and compared
each Graph run with an independent eager-vs-eager control:

| Variant / path | Result max abs | Clipped-gradient max abs | Parameter sample max abs | Parameter sample outliers | RNG |
|---|---:|---:|---:|---:|---|
| Pen Graph | `5.96e-8` | `1.40e-7` | `4.02e-6` | `2 / 17` | exact |
| Pen eager control | `5.96e-8` | `1.39e-7` | `2.68e-6` | `1 / 17` | exact |
| RDT Graph | `5.96e-8` | `1.17e-7` | `1.53e-6` | `0 / 17` | exact |
| RDT eager control | `2.98e-8` | `1.01e-7` | `4.40e-7` | `0 / 17` | exact |

Parameter/buffer/optimizer surfaces are sampled at 17 evenly spaced state
checkpoints; result and gradient surfaces are checked on all 256 cases.  The
small Pen parameter outliers are in the same raw-flow CUDA tail region seen in
the eager control (relative-L2 maxima `3.42e-8` Graph versus `2.30e-8` eager).
They are not evidence of a changed optimizer rule or checkpoint state.  RDT's
Graph envelope likewise stays below the configured parameter tolerance ratio
(`0.593` maximum).

Across seven independent RDT gate invocations, three passed and four stopped at
different single-element CUDA tails (representative maxima `1.01e-6`, `1.39e-6`
and `2.79e-6`).  One failure occurred in the basic raw-gradient gate and the
others in the runner's intentional eager-diagnostic/input-copy sequence.  None
was a loss-ledger, optimizer-scalar, scheduler or RNG mismatch; the
eager-vs-eager controls exhibit the same CUDA `grid_sample`/pooling-backward
non-determinism.  Therefore the acceptance decision is:

* Pen's short production Graph path and checkpoint/RNG contract are accepted;
* RDT's path is functionally accepted for benchmarking, but remains
  **conditional** for a strict per-element certification until its
  non-deterministic observation backward is replaced or the calibrated envelope
  is made an explicit release criterion;
* neither variant is advertised as bitwise-equivalent over 1,024 CUDA samples;
  a zero-tolerance long-run requirement needs a deterministic replacement for
  the current observation backward operators.

Small reproducible summaries are in `compat-tests/results/`, and the shared
version adapter launcher is `compat-tests/run_variant_cuda_graph_equivalence.py`.
