# Composite temporal action representation

Status: standalone prototype. It is not imported by the mainline sampler, does
not change the `[B,24,18]` physical field, and does not define a schema.

## Purpose

Different actuator semantics need different temporal views. This package
combines lossless-capable continuous charts with typed endpoint sidecars while
keeping one shared flow clock and one caller-owned native-action finalizer:

```text
continuous state D_state -> role charts -> complete velocity
endpoint sidecars   -> typed logits/modes
continuous + sidecars -> caller codec/outlet -> D_action
```

The solver sees only the complete floating-point state and same-shape velocity.
It does not choose a chart or numerical method per role.

## Current role policy

| Role | Initial view | Hard boundary |
|---|---|---|
| smooth arm motion | fixed B-spline beside raw rows | rotations need an explicit geometry chart |
| continuous gripper | identity/raw | do not smooth away switch timing |
| binary command/contact | typed endpoint sidecar | never an ODE coordinate |
| dexterous joints/synergies | role-sized chart plus exact detail | compact synergy cannot claim native recovery |
| force/impact residual | raw/high-rate lane | cubic smoothness is not assumed across impact |

Roles share action time and flow time, so coupled dynamics remain learnable. A
role declares an ordered partition of `D_state`, chart identity and decode
group. `retain_raw=True` keeps exact sampled rows beside a structured
view; a lossy chart is rejected without that fallback.

Endpoint specs declare axes, dtype, vocabulary, temporal alignment and producer.
They are never concatenated into the continuous state. Decode groups name
exactly one final owner for every `D_action` slice; the package itself
does not own codec math, normalizers, robot limits or safety control.

## Mainline relationship

The opt-in B-spine component uses this package only as an outer representation
boundary. It keeps the current raw lift and adds a fixed B-spline view:

```text
x_t [B,24,18]
  -> raw lift
  -> fixed B-spline view
action seed + (raw + spine) * action_state_factor
```

This package cannot remove the raw path, alter source/target support, add a
loss, or add an ODE/W pass. Mainline identity and task gates live in the
architecture contract and run context.

## Minimal construction

```python
from clearvla.action_representations.composite import (
    CompositeActionRepresentation, CompositeActionSpec,
    ContinuousRoleSpec, DecodeGroupSpec, EndpointSpec, OwnerRef,
)
```

The full object is intentionally verbose because every axis and owner is part
of the serialized identity. Keep sample times identical across all charts and
reject stale endpoint payloads after a continuous-state or dtype change.

## Validation

```text
.venv/Scripts/python.exe -m pytest -q clearvla/action_representations/composite/tests
```

Package tests establish chart/spec/owner correctness only. Promotion requires a
matched real-data comparison, CUDA/checkpoint checks and a separate solver
comparison. Do not combine those hypotheses in one experiment.
