# V41 latent-main action decoder

This package keeps the V40.1 data/training/eval contract intact, but replaces the final action policy path.

## What stays unchanged

- HDF5 dataset contract, split, cache, zscore normalizer, physical action codec, target velocity, loss/eval interfaces.
- V40.1 layer-contract / recurrent-consequence trunk.
- Existing `legacy`, `residual_action_flow`, and `layered_residual_action_flow` modes are still available for comparison.

## What changes

New mode:

```bash
--final-action-decoder latent_main_action
```

In this mode, final action is emitted by one clean main decoder:

```text
V40 layer latents + controlled_delta + rollout_effect + event_context
        -> HierarchicalLatentMainActionDecoder
        -> pred_physical_velocity
```

It no longer does:

```text
pred = legacy_velocity + alpha * residual_velocity
```

For `latent_main_action`, the old direct/rollout action heads are not used as the final path.

## Latent injection

Every available V40 layer memory is injected into every decoder block as an all-layer summary token bank. Each block also receives full token-level memories from its scheduled layer pair and their token-level delta.

Default schedule:

```text
0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7
```

Default shortcut policy:

```text
latent_action_visual_memory = 0
latent_action_context_memory = 0
latent_action_transition_memory = 1
latent_action_layer_memory = 1
```

That means the new action decoder does not directly read dense visual/context memory by default; it must use V40 latent/consequence outputs.

## Gripper/event path

The event tokens are inside the main action decoder, not a standalone residual branch. They influence the gripper transition latent before the single physical velocity head emits the final 14-D physical velocity.

## Optional adaptivity

`--latent-action-stage-router 1` enables a local stage router. It is off by default because the first V41 test is meant to stay clean and deterministic.

## Scripts

Smoke:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v41_latent_main_action_smoke_b8_20.sh \
  > runs_v41_latent_main_action_smoke_b8_20.log 2>&1 &
tail -f runs_v41_latent_main_action_smoke_b8_20.log
```

Formal:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v41_latent_main_action_policy.sh \
  > runs_v41_latent_main_action_policy_b8.log 2>&1 &
tail -f runs_v41_latent_main_action_policy_b8.log
```

No training script exports `HF_HOME` or `TRANSFORMERS_CACHE`.

## Local checks already run

- `python -m compileall -q clearvla`
- `python -m clearvla.cli.train_v40_policy --help` includes `latent_main_action` and latent-action arguments.
- Toy forward smoke for `flow_training_forward` and iterative `sample` with `latent_main_action`.
- Shell syntax checks for both V41 scripts.

No real dataset training was run in this sandbox.
