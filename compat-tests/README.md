# Versioned training-acceleration acceptance

This directory contains the shared harness and small result summaries for
independent Pen and RDT checks.  The source snapshots under `local/` and the
archive files are intentionally ignored; they are reproducibility inputs and
must be restored from the authorized lab or recreated from the revisions below.

## Sources

| Variant | Source revision | Layout | Local snapshot |
|---|---|---|---|
| Pen B-spline | `a4a170e` | modular | `local/pen_bspline_a4a170e` |
| RDT | `f8d0c2f` | legacy | `local/rdt_f8d0c2f` |

Pen and RDT are always run as separate processes.  A result from one variant
does not stand in for the other.

## Reproduction

CPU golden captures (zero tolerance) use the shared fixture and
`clearvla/tools/mainline_equivalence.py`; compare each baseline/candidate pair
with `compare`.  The CUDA acceptance gate uses the production
`CudaGraphTrainingStepRunner` through the version-local factory:

```powershell
$py = 'D:\CondaData\envs\torch312\python.exe'
& $py compat-tests\run_variant_cuda_graph_equivalence.py `
  --root (Resolve-Path compat-tests\local\pen_bspline_a4a170e) `
  --config-factory _bspine_config
& $py compat-tests\run_variant_cuda_graph_equivalence.py `
  --root (Resolve-Path compat-tests\local\rdt_f8d0c2f)
```

The factory argument is part of the variant identity.  In particular, a Pen
run that uses the default ``_config`` factory exercises the Pen source layout
but does **not** enable its B-spline path and cannot be reported as a Pen
B-spline acceptance result.

The default gate covers identity and active topology, input-copy and restored
input replay, raw/clipped gradients, the complete loss ledger, parameters,
AdamW state, scheduler/global step, CPU/CUDA/owned-generator RNG, and an exact
checkpoint save/load continuation.  The long diagnostic uses 256 independent
one-update cases at B4 (1,024 sample slots), with 128 cases on each topology.
It is compared with an eager-vs-eager control before any long-horizon claim is
made; CUDA `grid_sample`/pooling backward is not bitwise deterministic on the
current stack.

Result JSON files in `results/` are intentionally small and reviewable.  Do
not check in checkpoints, tensor caches, full source snapshots, or raw logs.

On the 2026-09-07 RTX 4060 run, Pen passed four independent short-gate
invocations.  RDT passed three of seven; the four failures were isolated
single-element CUDA backward tails that also occur in eager-vs-eager controls.
Treat RDT as conditionally accepted for speed experiments, not as a bitwise
CUDA certification, until a deterministic observation backward is available.
