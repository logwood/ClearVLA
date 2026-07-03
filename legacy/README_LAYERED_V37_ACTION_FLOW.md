# V40.1 Stable + Layered V37 Residual Action Flow

This package keeps the stable V40.1 policy path as the base action head and adds a **zero-start residual** V37-style action/event decoder with **layer-pair progressive injection**.

## Core contract

```text
legacy_velocity = direct_velocity + rollout_residual_velocity
residual_velocity = LayeredV37ResidualActionFlow(...)
pred_physical_velocity = legacy_velocity + alpha * residual_velocity
```

The residual velocity/event/motion output heads are zero-initialized, so the stable checkpoint is behavior-preserving at load time.

## What is new

`--final-action-decoder layered_residual_action_flow`

Each residual block receives a different token-level memory:

```text
block0: context + L0 + L1 + (L1 - L0) + routed layer mix
block1: visual  + L1 + L3 + (L3 - L1) + routed layer mix
block2: rollout + L3 + L5 + (L5 - L3) + routed layer mix
block3: rollout/event + L5 + L7 + (L7 - L5) + routed layer mix
```

The default schedule is:

```bash
--action-flow-residual-layer-pair-schedule 0:1,1:3,3:5,5:7
```

The layer memories are token-level, not pooled-only. Each layer memory includes rollout/action/consequence/event-related tensors where available:

```text
rollout_tokens
trajectory_pooled
rollout_effect_pred
rollout_delta_pred
policy_effect_tokens
policy_effect_time_tokens
unified_intervention_latent_pred
neutral_latent_pred
milestone_step_delta_pred
```

By default layer memories are detached:

```bash
--action-flow-residual-layer-detach 1
```

so the new residual branch learns to read the stable V40.1 hierarchy without dragging the trunk around on the first experiment.

## Scripts

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v40_1_stable_plus_layered_v37_action_flow_smoke_b8_10.sh \
  > runs_v40_1_layered_v37_actionflow_smoke_b8_10.log 2>&1 &
```

Formal run:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v40_1_stable_plus_layered_v37_action_flow_policy.sh \
  > runs_v40_1_layered_v37_actionflow_policy_b8.log 2>&1 &
```

## What to watch

Logs now include:

```text
aflow = residual norm after alpha
araw  = raw residual norm before alpha
rmax  = mean max stage-router weight
rent  = mean stage-router entropy
agrad = residual branch grad norm
```

Healthy early smoke should look like:

```text
stdr ~ 0.8-0.9
dnratio ~ 0.3-0.4
d_shuffle not collapsed
aflow starts near 0 and grows slowly
agrad non-zero
rmax not NaN
```
