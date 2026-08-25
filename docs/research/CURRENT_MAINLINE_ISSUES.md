# ClearVLA Current Mainline Issues

This is a pure unresolved-problem ledger for the active Schema38 source. It
contains no repair ancestry, implementation plan or list of changes. A source
contract is not an empirical result: keep an item until its remaining
uncertainty is actually observed or causally attributed.

## IC-07 — grounding reconstruction owns a private spatial nuisance

`decode_public_position` contributes to current-DINO reconstruction but is not
exported in `ObjectFactSet`. Grounder loss can therefore improve through
capacity unavailable to S, Teacher, W or P2.

Keep with-position, position-zero and exported-K-only reconstruction as three
separate diagnostics. Aggregate reconstruction is not evidence that online K
facts became more useful, and this nuisance must not be promoted into a second
public value path without an explicit downstream owner.

## IC-11 — learned flow and Teacher association remain unidentified

The latest completed parent run had Teacher dustbin around `0.46-0.47`,
reliability around `0.25-0.26`, and the best real match below the dustbin
background. This can be calibrated ambiguity or an overly attractive fallback;
there is no independent association label that distinguishes them.

Learned flow remained finite and did not collapse to zero in that run, but its
action value is still unidentified. Do not force nonzero flow, impose dustbin
or route quotas, reinterpret reliability as a loss mask, or change Teacher
target algebra without matched association/action evidence.

## IC-12 — Schema38 gripper and precision behavior is unobserved

Schema37 ended close to V120 in aggregate physical RMSE but remained worse than
V25/V26, with the clearest gap in gripper/precision behavior. Schema38 restores
fact-conditioned dynamic-P1 precision and changes the W/P2 consumer algebra,
but no fresh run yet shows whether this closes that gap without sacrificing
arm, first/tail or horizon behavior.

Decoded gripper events, event head and motion head remain separate objects.
Another event-positive boost, a precision gain or an external loss change is
not justified before the complete fresh validation curve is available.

## IC-16 — finite observation-side gradient spikes need fresh attribution

Schema37 contained finite preclip spikes above `5`, including a raw global norm
near `44.67`, concentrated in the broad observation owner group. The former log
mixed a window mean with final-batch owner diagnostics and could not identify
the responsible parameter.

Schema38 now records window mean/max/current and emits a parameter-level
`gradient_spike` event only after a finite threshold crossing. This closes the
observability defect, not the numerical cause. A fresh run must determine
whether spikes recur and, if so, whether one stable parameter/operation owns
them. Do not add clipping, gains or observation-network changes speculatively.

## EXP-01 — Schema38 empirical and causal release gates are pending

The source implements complete-field W/P2 consumption, S-conditioned W keys,
geometry-conditioned semantic address, fact-conditioned dynamic precision,
reachable covariance and corrected G3/gradient diagnostics. These boundaries
still require fresh runtime evidence:

- BF16 train forward/backward and five-step deployment;
- Teacher isolation and cached static-call frequencies;
- neutral-W, S-neutral, geometry-neutral and dynamic-precision causal checks;
- epoch-one physical/normalized, first/tail, horizon, arm/gripper/event/motion
  safety comparison;
- batch-eight memory and throughput;
- complete eight-epoch comparison against V120, V25 and V26, including late
  rebound and frozen action interventions.

Do not claim performance recovery, healthier W differentiation or useful
geometry/precision action bandwidth from source tests or nonzero gradients
alone.

## Explicitly not active diagnoses

The active source has structural closure for the following former issues; do
not reopen them without new source or experimental evidence:

- S no longer owns an interval vote independent of W at P2.
- P2 consumes reconstructed complete W fields; there is no common-only value
  consumer parallel to an optional residual consumer.
- Dynamic P1 has a named, fact-conditioned P3 precision consumer while staying
  outside protected fact and consequence.
- Geometry conditions semantic K address without becoming an interval vote or
  scaling semantic value.
- W covariance can approach a zero target while remaining PSD.
- G3 no longer logs a softmax-gauge-inflated correction.
- The bottom lane-local `4 basis + null`, CVAE/workspace, Evidence MMDiT,
  controlled transition and execution graph were not changed by Schema38.

## Closure rule

An issue is deleted only when all relevant items are source-backed and the
remaining empirical claim has been observed:

1. producer and retained axes;
2. every transform, residual/addition point and alternate path;
3. every online and training consumer;
4. loss and ordinary backward owner;
5. zero semantics, dtype and scale/Jacobian boundary;
6. deployment frequency and checkpoint identity;
7. a log or causal observation that tests the claimed behavior.
