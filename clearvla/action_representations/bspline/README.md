# ClearVLA B-spline action representation

Status: accepted only as a standalone schema-v1 CPU/static mathematical and
compatibility prototype.
It is not B-spine-0, is not wired into the active mainline, and closes none of
Gate A, Gate B, the CUDA gate or the production gate.
This directory is the canonical long-term workspace for B-spline action
representation work. It is intentionally independent of a particular ClearVLA
policy version.

The implementation is entirely contained here:

- `spec.py` owns the immutable time grid, modes, metadata and fingerprint;
- `basis.py` owns deterministic float64 construction, not-a-knot detail
  interpolation, canonical QR completion and basis identity;
- `representation.py` owns the parameter-free encode/decode/evaluate API;
- `compat.py` is a duck-typed outer-boundary compatibility façade around a
  native/physical codec;
- `tests/` owns representation and current-boundary compatibility checks.

No existing mainline, policy, codec or training source is modified by this
package. The package itself imports neither `clearvla.mainline` nor a historical
experiment module.

## Ownership boundary

This workspace owns only:

- fixed B-spline basis, knot and time-grid construction;
- native arm-trajectory encoding, decoding and continuous resampling;
- optional, explicitly declared affine-origin coordinates and boundary
  metadata;
- first/second analytical derivatives of the decoded curve;
- a lossless coarse/detail chart and an explicitly lossy compact chart;
- deterministic basis identity, numerical diagnostics and representation-floor
  audits;
- unit tests and a small, stable public API for integration owners.

This workspace does not own:

- `PhysicalActionFieldCodec`, the 18-channel absolute/delta/gripper field, its
  active replacement or the B-spine-0 Gate-B cut;
- flow-matching source distributions, time sampling, loss composition or ODE
  updates;
- policy heads, V120 bottom blocks, G/S/W/P, Teacher or execution-value logic;
- normalizers, dataset selection, checkpoint migration or robot controllers;
- inter-chunk alignment, asynchronous execution or task-specific safety logic.

Those systems may consume this representation through adapters, but they must
not be imported into this package. Conversely, this package must not import
from `clearvla.mainline`, `clearvla.policy` or an experiment-version module.

## Standalone native-action scope versus planned B-spine-0

The active mainline currently accepts a normalized native action chunk
`[B,24,7]`: six arm channels and one gripper channel. Its current codec expands
that into `[B,24,18]`: six arm absolute values, six arm deltas and six gripper
field values.

This standalone package acts on a native arm trajectory `[B,T,D_arm]` before
the redundant absolute/delta field. Gripper and any future
discontinuous/event-like channels remain exact pass-through values owned by the
caller. That scope is useful for basis mathematics, serialization and offline
representation audits, but it is not the planned model insertion point.

`PhysicalActionFieldBSplineAdapter` only demonstrates outer-boundary shape and
semantic compatibility. It decodes the native arm trajectory and delegates the
physical-field behavior to an already supplied codec. It is deliberately
marked `bspine0_gate_b_compatible=False` and
`repeated_bottom_call_safe=False`; it is neither the B-spine-0 implementation
nor authorization to replace the active codec.

The planned B-spine-0 is a different, integration-owned bottom unit. It reads
only the existing deployed noisy physical flow state `x_t [B,24,18]`, after the
unchanged `PhysicalActionFieldCodec`, and forms a parallel numerical view beside
`NativeTimePhysicalActionTokenLift` / `noisy_lift`:

```text
x_t [B,24,18]
  +-> existing noisy_lift ------------------------------> u_raw
  |
  +-> fixed FP32 B-spine analysis/synthesis ------------> u_spine

existing action seed + (u_raw + u_spine) * action_state_factor
```

The original raw path remains independently usable. B-spine-0 may not change
`PhysicalActionFieldCodec`, flow targets or source noise, the raw lift, the
physical velocity head, the number of ODE updates or W passes, or the existing
action/output ABI. This package does not implement that branch and must not be
used as a native-action compression bottleneck in its place.

The public payload API performs defensive finite and identity checks. On an
accelerator those checks may synchronize with the host, so `encode`, `decode`,
`evaluate` and the compatibility façade must not be placed inside an ODE loop
or the deployed path's 12 bottom calls per action. A future Gate-B
implementation must be a separate integration-owned tensor primitive that
reuses construction-time precomputed FP32 analysis/synthesis buffers and
performs direct tensor algebra without rebuilding a basis or validating Python
payloads on every bottom call.

## Design decision

The extracted `v120_bspline_reintegration` package is useful reference material,
but its fixed cubic `T=24, K=12` manifold is not the default design here. A
rank-12 projection necessarily discards twelve temporal degrees of freedom
before any model is trained. Its geometry-closure tests are valid, but they do
not establish that first-step, contact-adjacent or other high-frequency action
detail is safe to remove.

The default representation will be a **fixed, hierarchical, lossless-capable
B-spline chart**. Compact B-spline-only output is an opt-in mode that remains
disabled until a real-data and task-performance gate passes.

### Lossless hierarchical chart

For one arm channel sampled at `T` action times, let `B_c in R[T,K]` be a fixed
coarse B-spline collocation matrix. A deterministic float64 QR factorization
gives

```text
B_c = Q_c R_c
```

and a deterministic orthogonal completion gives

```text
Q = [Q_c, Q_d] in R[T,T].
```

The model-safe coordinates are

```text
z_c = Q_c^T x
z_d = Q_d^T x
x   = Q_c z_c + Q_d z_d.
```

This has three useful properties:

1. Keeping both `z_c` and `z_d` is an exact round trip at the original action
   timestamps, so the representation itself has no action error floor.
2. `[z_c,z_d]` is orthonormal in sampled action space. Euclidean MSE, isotropic
   noise and gradient scale are therefore not silently distorted by a
   non-orthogonal control-point chart.
3. `c = R_c^-1 z_c` exposes ordinary local B-spline control points whenever an
   integrator or controller needs them.

For continuous evaluation, the detail basis is represented through a fixed,
full-rank interpolating B-spline chart. It must reproduce `Q_d` exactly on the
original time grid. Thus the complete curve remains a fixed piecewise-polynomial
spline, while the original discrete actions remain recoverable. Basis creation
runs once in float64 on CPU; runtime tensor operations use an explicit FP32
numerical scope and preserve autograd.

The completion is canonicalized (including column signs) and identified by a
hash so that two processes cannot silently use different detail coordinates.

A local float64 structural preflight for uniform `T=24`, cubic coarse `K=12`
finds a coarse collocation condition number of `4.6703`, full coarse/detail
round-trip maximum error of `1.78e-15`, orthogonality error of `7.77e-16`, and
full-rank interpolation condition number of `3.8919`. Its dense worst-case
sample-to-curve infinity amplification is about `1.98`. These are structural
numbers for one grid, not dataset, controller or task-performance results.

Exact mode proves only that this coordinate chart can preserve the sampled
action tensor. Because it retains all `T` temporal degrees of freedom, it does
not by itself provide compression, smoothing, improved chunk continuity or any
other B-spline behavioral value. Those benefits require a separately accepted
lossy representation or execution design.

### Modes

The public specification exposes exactly two modes:

- `hierarchical_exact`: coarse plus all complementary detail coordinates;
  default and required compatibility mode.
- `compact`: coarse B-spline coordinates only; lossy, experimental and accepted
  by the compatibility façade only when the caller explicitly passes
  `allow_experimental_lossy_projection=True`.

There is no partial-detail or budgeted mode. Canonical detail coordinates have
no principled coarse-to-fine ordering, so retaining an arbitrary prefix would
give configuration order an unsupported meaning. The reserved `detail_budget`
field exists only so older prototype configurations fail with a targeted
error. Dropping all detail is an explicit representation decision, never an
implicit side effect of decoding or configuration defaults.

### Fixed numerical preflight

Every public runtime construction is fail-closed under
`clearvla-bspline-runtime-safe-v1`. It audits both the canonical float64 bundle
and the exact float32 buffers/operators that the representation will use, with
513 dense samples. Runtime float32 buffers must all be finite; sample times
must remain strictly increasing; knot ordering, distinct-knot topology and both
spline domains must survive conversion; and the sample/dense operators must be
finite. It then enforces these source-owned limits:

| Metric | Maximum |
|---|---:|
| coarse collocation condition number | 1000 |
| full interpolation condition number | 1000 |
| coarse analysis operator 2-norm | 1000 |
| interpolation analysis operator 2-norm | 1000 |
| dense value L-infinity gain | 64 |
| dense velocity L-infinity gain, normalized by average sample step | 128 |
| dense acceleration L-infinity gain, normalized by average sample step squared | 512 |
| float32 sample-grid evaluate/decode closure error | 0.001 |
| float32 dense value L-infinity gain | 64 |
| float32 dense velocity L-infinity gain, normalized by average sample step | 128 |
| float32 dense acceleration L-infinity gain, normalized by average sample step squared | 512 |

`basis_preflight(spec)` is the diagnostics-only entry point. It reports failed
metrics but never returns unchecked runtime matrices. `build_basis_bundle(spec)`
and `BSplineActionRepresentation(spec)` reject the same spec if any limit is
non-finite or exceeded; callers cannot raise or disable the limits.

For example, the clustered cubic grid
`times = linspace(0, 1, 24) ** 3`, `K=12` is expressible as a specification but
is diagnostics-only: its coarse condition number is about `4064` and its dense
value gain is about `2247`, so runtime construction fails. An offline
diagnostics-only reproduction with
`x = sin(linspace(0, 8, 24))` has sampled maximum magnitude `0.989358`, while
the unchecked continuous curve reaches `40.890664` and its velocity reaches
`4881.258916`. Those values explain the rejection; the public runtime API does
not expose that unchecked representation. Non-uniform time grids are supported
only when both the canonical and actual float32 runtime preflight pass.

The runtime check also rejects time charts that are harmless in float64 but
cannot be represented by the deployed float32 path. In particular,
`[1e8, 1e8+1]` collapses its 24 timestamps and knot domains, `[0,1e-40]`
creates non-finite float32 derivative operators, and `[0,1e39]` overflows
registered buffers. These fail during `basis_preflight` eligibility reporting
and runtime construction, before an object can reach `evaluate`.

### Time, degree and knots

- The original action timestamps are explicit inputs to the spec; the module
  does not assume that all datasets have the same control rate, but arbitrary
  grids are not accepted merely because they are strictly increasing in
  float64. They must preserve their grid, knot topology and finite operators in
  the actual float32 runtime representation.
- Public basis domain checks use endpoint-ULP-aware tolerances in the time
  coordinate's own units. There is no absolute `1.0` scale floor: one
  representable value inside either endpoint is accepted, while one full ULP
  outside is rejected rather than silently clamped.
- Degrees 1, 2 and 3 are supported by the API. Degree 3 is a smooth candidate,
  not a universal default. Degree 2 is a required comparison because it has
  tighter local support and lower interpolation amplification. The full-rank
  degree-1 interpolant is the non-overshooting reference. The hierarchical
  curve is a sum of coarse and detail spline spaces whose knots need not nest,
  so its own dense operator gain is still reported and gated for every degree.
- The first compact candidates use fixed open-uniform knots. A single
  train-data-derived, globally frozen non-uniform knot schedule may be audited
  later.
- Per-sample adaptive or learned knots are excluded from the first
  implementation. They change the support and payload shape and make a stable
  replacement boundary substantially harder.
- A caller-declared arm-coordinate origin may be supplied as an affine
  translation. An optional state-anchored continuous curve may eventually
  include a true physical state as the fixed sample at time zero, but that is
  not implemented and must not be inferred from tensor shape.

Schema v1 implements the first behavior only: `origin` is subtracted before
analysis and added after synthesis as a constant affine translation. It does
not prepend a hidden sample or constrain the first action row to the current
state. In the CALVIN and ManiSkill codec boundary, `action_state` is the
previous executed command, not an absolute TCP pose. The adapter therefore
never infers `origin` from `action_state`; a future anchored-curve mode would
require a new explicit spec field, a correct geometry source and a new
representation identity.

### Channel policy

- Smooth arm position channels may use the B-spline chart.
- Gripper commands, event flags, modes and other discontinuous channels are not
  spline-fitted by this module.
- Quaternion or rotation channels cannot be treated as independent Euclidean
  scalars without an explicit geometry adapter. Such an adapter is outside this
  package.
- Normalization, units, joint limits and clipping remain caller-owned. The
  representation records metadata but never changes values secretly.

## What is retained from the V120 reference package

- open-uniform basis construction and partition-of-unity checks;
- fixed, parameter-free matrices with explicit dtype handling;
- separation of arm geometry from gripper semantics;
- idempotence, interpolation/velocity closure and finite-backward test ideas;
- representation-floor slicing by first step, early horizon and tail;
- condition-number, leverage and operator-norm diagnostics.

## What is not migrated into the canonical package

- the `patched_clearvla` repository copy and version-specific patch files;
- the mainline flow adapter, source-noise policy, solver projection and loss
  edits;
- a hard-coded `T=24`, `D_arm=6`, cubic degree or `K=12` default;
- the lossy manifold as the default representation;
- arbitrary first-step atoms before a data audit establishes their need;
- generated matrix CSV files as runtime sources of truth;
- the term `contact` for rows detected only from a gripper transition. A true
  contact metric requires contact evidence; otherwise the metric is named
  `gripper_event_neighborhood`.

The original extracted directory and zip may remain as read-only provenance
until the new package has feature and test parity. Deleting or archiving them is
a separate cleanup decision.

## Performance-preservation gates

No representation can claim unchanged task performance from basis tests alone.
All externally meaningful gates remain open. The following sequence separates
representation fidelity from integration quality.

### Local structural verification

- exact-mode `decode(encode(x))` error at original timestamps is at FP32
  numerical floor;
- coarse/detail reconstruction, orthogonality, rank, partition, endpoints and
  basis hash are deterministic;
- backward, mixed-precision boundary, batch/device transfer and serialization
  tests pass;
- dense resampling reports overshoot plus velocity/acceleration extrema; it
  never hides them with clipping.

These checks can reject an unsafe representation, but passing them does not
close Gate A, Gate B, the CUDA gate or the production gate.

### Gate A: real-data representation audit — open

Use the real train-only normalizer and identical windows to compare:

- identity/raw and the lossless chart;
- B-spline degrees 1/2/3 with `K` candidates selected for the horizon;
- a same-rank orthonormal DCT control;
- full, first, first-4, execution bands and tail errors;
- per-arm-channel maximum error and first/second temporal differences;
- gripper-event-neighborhood and ordinary rows, named accurately;
- fixed-operator conditioning, analysis amplification and dense interpolation
  overshoot.

The lossless chart must remain at numerical floor. A compact candidate is
eligible only when every critical-slice representation error is comfortably
below the matched baseline model error; the initial gate is at most 10% of that
error, including early/action-critical slices. If no compact candidate passes,
the exact detail coordinates remain enabled.

### Gate B: integration non-inferiority — open

The integration owner runs the matched training/deployment A/B. Acceptance
requires no meaningful regression in near/mid/far arm error, gripper behavior,
task success, runtime health or chunk-boundary behavior. Continuous high-rate
resampling is a separate opt-in gate because an interpolating cubic can be exact
at sample times and still overshoot between them.

The CUDA gate remains open until the CUDA-specific autocast/backward test runs
on real CUDA hardware rather than being skipped. The production gate remains
open until Gate A, Gate B, runtime profiling, serialized identity checks and the
owner's deployment acceptance all pass.

## CALVIN diagnostic boundary

The available rollout audit does not show that an independent per-chunk compact
projection fixes the observed execution problem:

- The `open_drawer` snapshot contains 290 steps. Latency mean/p95/max is
  `3.2425 / 3.6876 / 6.6105 s`.
- Cross-chunk first-action L2-RMS is about `0.1252` with p95 `0.2029`; previous
  chunk row 1 versus current row 0 mismatch is about `0.1528` with p95
  `0.2784`.
- `65.123` is a path-length ratio computed over the six-dimensional command
  sequence; it is not a physical TCP path ratio.
- `790` counts command-coordinate sign flips with no deadband; it is a jitter
  clue, not a contact, instability or task-failure proof.
- A cubic `K=12` compact projection produced first-row element RMSE about
  `0.002587`, path ratio about `65.56` and `788` flips, so those indicators were
  essentially unchanged.

Cross-chunk state, overlap/refit, SE(3) geometry, command timing and the
execution adapter are entirely outside this component. They must be diagnosed
and owned at their actual runtime boundary.

## Public API

The stable schema-v1 interface is:

```python
spec = BSplineSpec.uniform(
    horizon=24,
    arm_dim=6,
    num_control_points=12,
    degree=3,
    mode="hierarchical_exact",
)
codec = BSplineActionRepresentation(spec)

preflight = basis_preflight(spec)
encoded = codec.encode(
    arm_trajectory,
    times=times,
    origin=explicit_arm_coordinate_origin,
)
arm_at_grid = codec.decode(encoded)
arm_at_query = codec.evaluate(encoded, query_times)
velocity = codec.derivative(encoded, query_times, order=1)
acceleration = codec.derivative(encoded, query_times, order=2)
control_points = codec.coarse_control_points(encoded)
report = codec.diagnostics(arm_trajectory, encoded)
```

The encoded payload and spec carry a schema version and basis digest. No global
registry, environment variable or mainline feature flag belongs in this
package; integration owners may register this implementation behind their own
action-representation boundary.

`PhysicalActionFieldBSplineAdapter` accepts an existing codec by structural
interface rather than importing its class. Its `encode_representation` /
`decode_representation` methods expose the native arm chart, while its
codec-shaped façade exists only for outer-boundary compatibility experiments.
It checks the complete forwarded codec protocol at construction, passes all
channels after the arm prefix through exactly, delegates their physical-field
semantics to the wrapped codec, defaults to exact mode, and requires explicit
opt-in for compact projection. This does not validate the façade for Gate B or
for repeated bottom calls.

For the current codec ABI, construction also requires
`uses_relative_command_direct` and `arm_motion_magnitude`. The `encode`,
`decode`, `gripper_decode_branches` and `delta_consistency` signatures must
explicitly accept keyword-only
`codec_gripper_boundary: Tensor | None = None`. The façade and its
`to_physical` / `from_physical` helpers forward a supplied boundary unchanged;
they never reinterpret it and never use it as a B-spline `origin`.

Runtime coordinates have shape `[B,K,D_arm]` for the coarse block and
`[B,L,D_arm]` for detail. In exact mode `K+L=T`; in compact mode `L=0`.
Basis construction is CPU float64, while normal runtime algebra is
autocast-disabled FP32. Explicit float64 inputs retain float64 for numerical
audits.

## Evidence behind the choice

- [B-spline Policy](https://arxiv.org/abs/2607.09648) supports continuous
  control-point actions, adaptive offline fitting and segment alignment, while
  its public preprocessing fits all but the final action channel. Its adaptive
  knots and execution alignment remain optional integration-stage ideas here.
- [ABPolicy](https://arxiv.org/abs/2602.23901) demonstrates flow matching in
  continuous cubic B-spline control-point space. Its asynchronous execution and
  continuity-constrained refitting are outside this workspace.
- [Spline Policy](https://arxiv.org/abs/2606.07386) explicitly treats spline
  order as task/controller dependent and primarily uses piecewise quadratic
  splines, which is why cubic is not hard-coded as the only candidate.
- [BEAST](https://arxiv.org/abs/2506.06072) supports B-spline action
  tokenization, but discrete tokenization is not required for this component.

These works support B-splines as a replaceable action representation. None of
them proves that a lossy fixed `K=12` projection preserves ClearVLA behavior;
that question remains subject to the gates above.

## Implementation and acceptance sequence

1. **Implemented locally:** deterministic spec/basis/payload,
   `hierarchical_exact` encode/decode/evaluation/derivatives, all-or-none detail
   policy and fail-closed numerical preflight.
2. **Implemented locally:** representation tests for exact closure, explicit
   compact projection, gradients, BF16 boundaries, arbitrary safe `T/D/time`
   grids, explicit origin, serialization, identity, diagnostics and the
   outer-boundary codec façade.
3. **Gate A open:** run the dataset-only audit without importing a policy
   model; freeze a compact candidate only if it passes.
4. **Gate B open:** integration owner designs the tensor-only cut and performs
   matched training/deployment A/B using the serialized spec fingerprint and
   basis digest.
5. **CUDA and production gates open:** run the CUDA test on actual hardware,
   profile the accepted integration and complete deployment acceptance.

Local structural reference for uniform `T=24`, cubic `K=12`:

- coarse condition number: `4.6703225845`;
- not-a-knot interpolation condition number: `3.8918663598`;
- float64 orthogonality maximum error: `7.77e-16`;
- dense sample-to-curve infinity norm: `1.9779786204`.

Reproducible local checks:

```text
.venv/Scripts/python.exe -m pytest -q clearvla/action_representations/bspline/tests
.venv/Scripts/python.exe -m ruff check clearvla/action_representations/bspline
.venv/Scripts/pyright.exe --pythonpath .venv/Scripts/python.exe --level error clearvla/action_representations/bspline
```

The verified acceptance scope is only a standalone CPU/static prototype. This
directory makes no active-mainline, B-spine-0, CUDA or production claim; its
local test pass cannot close the open gates listed above.
