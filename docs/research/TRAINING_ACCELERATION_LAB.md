# Training acceleration lab

Updated: 2026-09-06

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

The same commit adds an opt-in execution-controller source-lane reuse probe
(`--reuse-prepared-controller-context`).  It shares the projected
global/time/evidence lanes across the three recurrent decisions.  The forward
loss remains bitwise equal in the CPU gate; because the attached autograd graph
accumulates the shared lane in a different order, the update gate allows only
the measured `2e-5` relative / `2e-7` absolute floating-point tolerance.  It is
not enabled by default and will be accepted for throughput only if the remote
BF16 comparison stays inside that bound.

## Next experiment

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
