# Verified integration review — 2026-09-17

This is a **local integration candidate**, not a release or a claim that all
remote experiment branches have been merged. The pinned base is
`95851520cc49817c4dc401d48386303eefdec5c6`, whose model source is the CALVIN
object-binding tip `d7ebf00ed4b0041294c5ca44350c6ce79f8e782b`. The separately
observed `7f1512cf4f0872e1f4273dbf266cf47d7bc10298` adds only an upstream CI
preparation workflow; that workflow is not adopted or executed here.

## Integrated changes

The online history now retains nine observations and twenty-four executed
actions, while its absolute time index, reset-prefix padding and sparse window
values remain unchanged. It no longer retains an entire episode of RGB copies.
Deployment rejects lossy integer coercions for the DINO reference batch size
and receding-horizon execution row count. The root `run_current_policy.sh`
requires `--legacy-v48` before invoking its archived launcher; the default path
cannot silently start V48.

CALVIN binding is an optional parameter owner in optimizer and gradient audit
reports. It must not appear as an obligatory zero-filled Pen/RDT/LIBERO owner.
The correction changes reporting ownership, not parameter construction or
optimizer membership.

The standalone Hybrid V1 representation factory and its tests were selectively
ported from `cd9489afbf080f592c6badfc23a06d9b8b8d64d8`. This does not enable the
mainline Hybrid training/runtime lane. The independent flow solver already
exists in the base; no solver is silently substituted into active sampling.
No whole-branch merge or archival branch deletion is performed.

## Opt-in CALVIN binding V2

Use `configs/mainline/calvin_object_binding_residual_v2.json` only as an
experimental fresh-run configuration. It selects
`top.calvin_object_binding=calvin_primary_v2` and component
`stateless_object_intent_calvin_binding_v2`. Existing presets and V1 selection
remain unchanged. Server-specific data paths still require explicit setup.

V1 normalizes the selected object context by total real-object mass and applies
its learned scalar before the coarse reader's LayerNorm. Consequently, null
mass does not reliably attenuate the selected context, and zeroing the scalar
does not restore the original K-object read. V2 preserves the public K read
and gates the **completed reader residual**, not its normalized input:

```text
m = sum(pointer over real objects)
a = tanh(context_gate) * m
object_delta = public_read + a * (selected_read - public_read)
```

This provides an exact original-read fallback for zero gate or all-null input,
retains K-permutation equivariance and permits formal task-loss gradients to
reach the null scorer. The parameter set is unchanged, but the computation and
component/configuration identity are different. Do not bypass identity checks
to resume a V1 checkpoint as V2.

The auxiliary binding supervised-loss helper is not wired into the formal loss
ledger. V2 does not invent slot labels, privileged simulator observations or a
new loss budget. The existing formal task/action path supplies its gradient.
The bridge's historical parameter initialization is 0.25, not zero. A bounded
reader innovation is not a certified bound on final robot action or task risk.

## Preserved boundaries

G remains current-visual-only; S owns intent; W receives object belief and
physical action conditions. No direct goal, future Teacher feature or noisy ODE
state is added to W. V2 adds one selected read at the existing coarse-action
seam, not an extra world model invocation.

Formal training remains single-stage with its existing loss composition.
Deployment remains two complete five-update ODE passes with identical initial
noise and one consumed W rebuild. Each pass additionally reads its heads at
the clean endpoint: twelve velocity calls are ten ODE updates plus two head
calls, not twelve integration steps. The final action/world mismatch remains
a measured residual, not a fixed-point convergence claim.

Pen/RDT/CALVIN/LIBERO retain their outlet-owned charts, gripper modes,
normalizers and checkpoint boundaries. A shared seven-dimensional shape is
not a shared physical action meaning.

## Evidence and release limits

Completed targeted local suites contain 309 passing and two skipped unique
tests, including all 47 tests in `test_mainline_policy.py`. Both skips require
CUDA. These are selected suites, not a complete test-repository run. Broader
combined runs exceeded the local execution time limit and were not counted as
passes; four selected checkpoint identity/round-trip tests completed.

The local environment is Python 3.13.5 / Torch 2.10.0+cpu. It is **outside** the
repository's declared Python 3.12 / Torch 2.11 range. The expanded CI workflow
uses the declared range, but no remote CI execution of this candidate is
claimed. No production data, production checkpoint, CUDA/BF16 runtime, training
curve or real closed-loop benchmark is validated here.

Small-shape CPU FP32 comparisons preserve exact parameter, action and physical
field hashes for Pen and CALVIN V1 under matched seeds and synthetic inputs.
This is not evidence of production-checkpoint compatibility or task success.

Before promotion, run supported-environment static/contract checks, real
CUDA/BF16 forward/backward and memory gates, checkpoint-owned causal replay,
and matched closed-loop ablations. Keep Pen frame-scope/anchored losses,
B-spline selection/deletion, Hybrid mainline and geometry/interval changes as
explicit experiments until their behavioral and identity gates close. Do not
merge these mutually conflicting defaults together.

Detailed branch locks, merge-conflict lists, reproduction probes and JUnit
results are supplied in the external audit package, not copied into the active
architecture memory as raw log dumps.
