# ClearVLA B-spline action representation

Status: standalone package contract. This directory owns fixed B-spline basis
mathematics, native-arm representation and a boundary compatibility façade. It
does not own current experiment status, mainline architecture decisions or
task-performance claims.

The default mainline imports only BSplineSpec and build_basis_bundle for the
opt-in bottom component. It does not route through
BSplineActionRepresentation or PhysicalActionFieldBSplineAdapter. Mainline
semantics live in
[../../../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md](../../../docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md);
open experiment decisions live in
[../../../docs/research/CURRENT_MAINLINE_ISSUES.md](../../../docs/research/CURRENT_MAINLINE_ISSUES.md).

## Package map

| File | Owner |
|---|---|
| spec.py | Immutable time grid, mode, metadata and fingerprint |
| basis.py | Deterministic basis construction, canonical QR completion, runtime preflight and basis identity |
| representation.py | Parameter-free encode, decode, evaluate, derivative and diagnostic API |
| compat.py | Duck-typed native/physical codec compatibility façade |
| tests/ | Representation, numerical-safety and façade contracts |

Dependency direction is one-way. This package imports neither
clearvla.mainline nor a historical experiment module. Integration owners may
consume its public API; this package does not reach back into their policy,
normalizer, dataset or runtime.

## Ownership boundary

This package owns:

- fixed B-spline knots, basis and sample-time construction;
- native arm-trajectory encoding, decoding and continuous evaluation;
- a lossless coarse/detail coordinate chart and an explicitly lossy compact
  chart;
- first and second analytical curve derivatives;
- optional caller-declared affine origin and channel/time metadata;
- deterministic representation/basis identity;
- fixed numerical eligibility checks and representation-floor diagnostics.

It does not own:

- the mainline 18-channel PhysicalActionFieldCodec;
- flow source distributions, targets, time sampling, losses or ODE updates;
- policy heads, G/S/W/P, Teacher, transition or execution-value logic;
- gripper command/event semantics;
- normalizers, data selection, checkpoint migration or robot control;
- cross-chunk alignment, asynchronous execution or task-specific safety.

## Coordinate chart

For one arm channel sampled at T action times, let the fixed coarse
collocation matrix be B_c in R[T,K]. Canonical float64 QR gives:

~~~text
B_c = Q_c R_c
~~~

A deterministic orthogonal completion supplies Q_d:

~~~text
Q = [Q_c, Q_d] in R[T,T]

z_c = Q_c^T x
z_d = Q_d^T x
x   = Q_c z_c + Q_d z_d
~~~

Keeping both coordinate blocks is an exact sampled-time chart. The coordinates
are orthonormal in sampled action space, so Euclidean MSE, isotropic noise and
gradient scale are not silently changed by a non-orthogonal control-point
parameterization. Ordinary coarse B-spline control points remain available as
c = inverse(R_c) z_c.

Continuous detail evaluation uses a fixed full-rank interpolating B-spline
chart that reproduces Q_d at the original grid. Basis construction is CPU
float64; registered runtime algebra is FP32 and keeps autograd.

Exact sampled reconstruction does not imply compression, smoothing,
cross-chunk continuity or task benefit. Those require separate integration
evidence.

## Modes

The schema-v1 spec exposes two modes:

- hierarchical_exact retains coarse plus the complete detail complement and is
  the default compatibility mode;
- compact retains only coarse coordinates and is explicitly lossy.

There is no partial-detail mode. The canonical complement has no principled
coarse-to-fine order, so a prefix would give arbitrary construction order a
semantic meaning. The reserved detail_budget field exists only to reject old
prototype configurations with a targeted error.

The compatibility façade accepts compact mode only when the caller explicitly
passes allow_experimental_lossy_projection=True. Dropping detail is never an
implicit decode or configuration side effect.

## Time, channel and origin rules

- Sample timestamps are part of representation identity and must be finite and
  strictly increasing in both the declared chart and the deployed FP32 path.
- Degrees 1, 2 and 3 are supported. Cubic is not a universal default.
- Control-point count is at least degree+1 and strictly below the horizon.
- Fixed open-uniform coarse knots are the first supported family. Learned or
  per-sample adaptive knots are outside schema v1.
- Origin is an explicit affine translation: it is subtracted before analysis
  and added after synthesis. It does not prepend a hidden sample or anchor the
  first row to robot state.
- Smooth Euclidean arm channels may use the chart. Gripper commands, event
  flags and other discontinuous channels remain caller-owned pass-through
  values.
- Quaternion or rotation coordinates need an explicit geometry adapter and
  cannot be treated as unrelated Euclidean scalars by this package.
- Units, joint limits, normalization and clipping are metadata/caller
  responsibilities and are never silently changed here.

Endpoint domain checks use tolerances based on the time coordinate's own ULP.
One representable value inside an endpoint is accepted; a full ULP outside is
rejected rather than clamped.

## Numerical preflight

Every public runtime construction is fail-closed under
clearvla-bspline-runtime-safe-v1. It checks canonical float64 construction and
the exact FP32 buffers/operators used at runtime over 513 dense samples.

Required boolean checks include finite buffers/operators, strictly increasing
FP32 sample times, nondecreasing knots, preserved distinct-knot topology and
valid coarse/interpolation domains.

| Metric | Maximum |
|---|---:|
| Coarse collocation condition number | 1000 |
| Full interpolation condition number | 1000 |
| Coarse analysis operator 2-norm | 1000 |
| Interpolation analysis operator 2-norm | 1000 |
| Dense value infinity gain | 64 |
| Dense velocity infinity gain per average sample step | 128 |
| Dense acceleration infinity gain per average sample step squared | 512 |
| FP32 sample-grid evaluate/decode closure error | 0.001 |
| FP32 dense value infinity gain | 64 |
| FP32 dense velocity infinity gain per average sample step | 128 |
| FP32 dense acceleration infinity gain per average sample step squared | 512 |

basis_preflight reports eligibility and failed metrics without returning an
unchecked runtime object. build_basis_bundle and
BSplineActionRepresentation reject the same unsafe spec; callers cannot raise
or disable these source-owned limits.

For orientation only, the uniform T=24, cubic K=12 chart has coarse condition
number about 4.6703, interpolation condition number about 3.8919, float64
sample round-trip error about 1.78e-15 and dense value gain about 1.98. These
are structural measurements, not dataset or task results.

Strongly clustered or FP32-unrepresentable time grids can be expressible in a
spec yet remain diagnostics-only. Runtime construction rejects them before
evaluate can consume unchecked matrices.

## Public API

~~~python
from clearvla.action_representations.bspline import (
    BSplineActionRepresentation,
    BSplineSpec,
    basis_preflight,
)

spec = BSplineSpec.uniform(
    horizon=24,
    arm_dim=6,
    num_control_points=12,
    degree=3,
    mode="hierarchical_exact",
)

report = basis_preflight(spec)
representation = BSplineActionRepresentation(spec)

payload = representation.encode(
    arm_trajectory,
    times=times,
    origin=explicit_arm_coordinate_origin,
)
arm_at_grid = representation.decode(payload)
arm_at_query = representation.evaluate(payload, query_times)
velocity = representation.derivative(payload, query_times, order=1)
acceleration = representation.derivative(payload, query_times, order=2)
control_points = representation.coarse_control_points(payload)
diagnostics = representation.diagnostics(arm_trajectory, payload)
~~~

The spec and encoded payload carry a schema version and deterministic identity.
Runtime coordinate shapes are [B,K,D_arm] for coarse and [B,L,D_arm] for
detail. In exact mode K+L=T; in compact mode L=0. Explicit float64 inputs
retain float64 for numerical audits, while reduced-precision runtime inputs use
FP32 basis algebra.

## Compatibility façade

PhysicalActionFieldBSplineAdapter accepts an existing physical codec by
structural protocol. Its encode_representation and decode_representation
methods expose the native-arm chart; its codec-shaped methods delegate
physical-field semantics and pass non-arm channels through exactly.

The façade:

- validates the complete forwarded codec protocol at construction;
- forwards an explicit codec_gripper_boundary without reinterpretation;
- never infers B-spline origin from action_state;
- defaults to exact mode and requires explicit compact opt-in;
- declares bspine0_gate_b_compatible=False and
  repeated_bottom_call_safe=False.

Public payload validation may synchronize an accelerator with the host.
Therefore the representation and façade do not belong inside an ODE loop. A
mainline bottom integration must register fixed operators once and use direct
tensor algebra at each call.

## Mainline integration boundary

The opt-in mainline B-spine component is owned outside this package. It reads
the existing noisy physical field after the canonical codec and forms a
parallel numerical view beside the raw lift:

~~~text
x_t [B,24,18]
  +-> unchanged raw lift --------> u_raw
  +-> fixed B-spline view -------> u_spine

action seed + (u_raw + u_spine) * action_state_factor
~~~

That integration cannot turn this standalone façade into the model codec,
compress away the raw path, change source/target support, alter loss or add an
ODE/W pass. Its component identity, checkpoint behavior and task evidence are
owned by the mainline contract, run context and issue ledger.

## Representation comparison boundary

The appropriate first comparison is a three-way, single-factorial control:

~~~text
same native 24-row arm trajectory
  -> raw rows       (identity control)
  -> fixed B-spline (local-support chart)
  -> fixed DCT      (global-support chart, owned by a separate experiment)
~~~

Gate-A reports should keep the solver at the declared E5/E5 baseline and
compare sampled reconstruction, endpoint error, first/second differences, and
local contact or switch windows. The raw path is not a claim that the
uncompressed chart is optimal; it is the attribution control that prevents a
solver, decoder, or data change from being mistaken for a representation
effect. DCT is deliberately not implemented or imported here, and gripper
event/binary channels remain outlet-owned in all three controls. Only after a
single representation and a single solver each pass their own evidence gates
should a representation-by-solver interaction be tested.

## Validation

Package tests cover basis partition/endpoints, rank, deterministic identity,
exact and compact behavior, serialization, arbitrary eligible time grids,
unsafe FP32 grids, analytical derivatives, gradients, reduced precision and
complete façade forwarding.

~~~text
.venv/Scripts/python.exe -m pytest -q clearvla/action_representations/bspline/tests
.venv/Scripts/python.exe -m ruff check clearvla/action_representations/bspline
.venv/Scripts/pyright.exe --pythonpath .venv/Scripts/python.exe --level error clearvla/action_representations/bspline
~~~

Passing package tests establishes representation and boundary correctness only.
A compact replacement or integrated bottom component needs its own real-data,
CUDA, checkpoint, runtime and matched task-performance gates.

## Design provenance

B-spline Policy, ABPolicy, Spline Policy and BEAST motivate local-support
action coordinates and continuous evaluation. None establishes that a fixed
lossy K=12 projection preserves ClearVLA behavior. Historical ClearVLA
manifold-flow proposals, version-by-version failure analysis, future
multiresolution sketches and one-off rollout diagnostics remain recoverable
from Git; they are not part of this package contract.
