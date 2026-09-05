# Standalone flow-solver lane

Status: provisional and disconnected from the mainline sampler. The package
integrates an existing instantaneous physical velocity field for fixed-checkpoint
probes. It does not change `RuntimeConfig`, model construction, codec,
W rebuild, outlet finalization or endpoint semantics.

The accepted mainline path remains five Euler updates per pass:

```text
proposal pass -> decode proposal -> rebuild W once
refined pass  -> final action -> separate t=1 endpoint head
```

The refined pass starts from the exact same initial physical field and a fresh
cache. Solver history never crosses the W boundary.

## Package map

| File | Responsibility |
|---|---|
| `spec.py` | schedule, solver, two-pass identity and fingerprints |
| `protocols.py` | velocity, endpoint, cache-rebuild and time-factory protocols |
| `integrate.py` | Euler, Heun and explicit two-pass runner |
| `diagnostics.py` | step-doubling and final-state probes |
| `panel.py` | deterministic candidate replay |
| `gates.py` | promotion checklist |
| `compat.py` | façade around an existing `model.velocity` |

The dependency direction is one way: Python stdlib and PyTorch only. There are
no imports from mainline, datasets, normalizers, checkpoints or simulation.

## Schedule semantics

`ScheduleSpec.boundaries` includes both endpoints. For
`[t0,...,tN]`, physical field calls use left endpoints
`[t0,...,tN-1]`; `tN=1` is only a boundary unless Heun needs the
end field. Endpoint-head calls are counted separately.

| Candidate | Meaning |
|---|---|
| `E5/E5` | current Euler-5 proposal and refined reference |
| `E5/DJ5(.5)` | same-NFE front-dense/terminal-jump refined probe |
| `E5/E6-uniform` | one extra uniformly placed refined update |
| `E5/E6-late` | one extra late refined update |
| `E5/H5` | Heun order comparison |
| `E5/RK4-oracle` | higher-order numerical oracle only |

Proposal-only `E6*/E5` controls isolate W sensitivity. `E10/E5`
and `E5/E10` separate proposal from refined compute. Candidate metadata
does not run a model or mutate configuration.

## Numerical rules

Euler:

```text
v_i = v(x_i,t_i)
x_(i+1) = x_i + (t_(i+1)-t_i) * v_i
```

Heun:

```text
v_i       = v(x_i,t_i)
x_predict = x_i + h*v_i
v_next    = v(x_predict,t_(i+1))
x_(i+1)   = x_i + h*(v_i + v_next)/2
```

Heun's end-field call is a physical NFE. RK4 is an oracle, not a default or a
claim of convergence; dense references must declare their grid and convergence
check.

## Public API

```python
from clearvla.action_solvers.flow_solver import (
    ScheduleSpec, SolverSpec, candidate_by_name, integrate, run_two_pass,
)

spec = SolverSpec(schedule=ScheduleSpec.dense_jump(steps=5, t_jump=0.5))
trace = integrate(initial_state, velocity_field, spec, fixed_cache)
plan = candidate_by_name("E5/DJ5(.5)").plan
result = run_two_pass(
    initial_state, velocity_field, plan,
    proposal_cache, rebuild_cache, endpoint_head=endpoint_head,
)
```

## Promotion gates

A solver candidate stays diagnostic until it has:

1. fixed-checkpoint matched replay with identical observation, noise and W
   lifecycle;
2. finite values, correct physical/endpoint NFE and deterministic identity;
3. proposal/refined/W-condition residuals reported separately;
4. dense-reference or step-doubling error, wall time and peak memory;
5. full action, arm, gripper, horizon and contact behavior;
6. no checkpoint, ABI, cache or endpoint regression.

B-spine, representation, Teacher, MIP and solver changes remain factorially
separate. Literature evidence can motivate a probe but cannot promote a
production schedule.

## Validation

```text
.venv/Scripts/python.exe -m pytest -q clearvla/action_solvers/flow_solver/tests
```
