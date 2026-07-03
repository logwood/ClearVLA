# V40.1 stable + V37-style residual action-flow denoiser

This package starts from the stable V40.1 baseline and does **not** replace the final action head.  It keeps:

- direct physical velocity head
- rollout residual head
- controlled residual latent dynamics
- recurrent milestone consequence
- layer contract/eval/data path

It adds an optional safe residual decoder:

```text
legacy_velocity = direct_velocity + rollout_residual_velocity
residual_velocity = V37StyleResidualActionFlowDenoiser(...)
pred_velocity = legacy_velocity + residual_velocity
```

The residual module borrows the useful V37 ideas: high/action/event tokens, self-attention among them, cross-attention to dense latent memory, and a transition-aware physical velocity head.  Its velocity/event output layers are zero-initialized, so loading the old stable checkpoint is behavior-preserving at initialization while the residual head still receives gradients.

Run smoke:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v40_1_stable_plus_residual_v37_action_flow_smoke_b8_10.sh \
  > runs_v40_1_residual_v37_actionflow_smoke_b8_10.log 2>&1 &
```

Run full:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v40_1_stable_plus_residual_v37_action_flow_policy.sh \
  > runs_v40_1_residual_v37_actionflow_policy_b8.log 2>&1 &
```

Key early checks:

- `stdr` should stay around the stable V40.1 range, roughly 0.8 to 0.9.
- `dnratio` should stay around 0.3 to 0.4.
- `d_shuffle` should remain clearly non-collapsed.
- `aflow` starts at 0 because the residual head is zero-initialized; it should grow slowly, not explode.
- `agrad` should be non-zero after backward, proving the residual path is being trained.
