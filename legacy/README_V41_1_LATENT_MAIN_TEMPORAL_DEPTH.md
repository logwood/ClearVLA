# V41.1 Latent Main Action Decoder with Horizon-Dependent Depth

This package keeps the V40.1 data contract, normalizer, physical action codec, losses, evaluation code, and latent/consequence trunk.  It still uses the clean V41 `latent_main_action` final path rather than `legacy_velocity + residual_velocity`.

## Main change

V41.1 adds horizon-dependent update depth inside the single `HierarchicalLatentMainActionDecoder`:

- near horizon tokens, by default steps 0-3, update only through the first 2 decoder blocks;
- mid horizon tokens, by default steps 4-7, update through the first 4 decoder blocks;
- far horizon tokens, by default steps 8-23, update through all 8 decoder blocks.

High tokens and event tokens remain updated in every block.  The mask only freezes horizon action-token updates after their assigned depth.  This is not a separate head and not a residual bypass.

The default per-block active action-token counts with H=24, depth=8, near=4, mid=8 are:

```text
block0: 24 active action tokens
block1: 24 active action tokens
block2: 20 active action tokens
block3: 20 active action tokens
block4: 16 active action tokens
block5: 16 active action tokens
block6: 16 active action tokens
block7: 16 active action tokens
```

Every decoder block still injects all layer summaries, and also injects scheduled token-level layer pair memories and pair deltas.  The default schedule is:

```text
0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:7
```

## New CLI flags

```bash
--latent-action-temporal-depth 1
--latent-action-near-steps 4
--latent-action-mid-steps 8
--latent-action-near-depth 2
--latent-action-mid-depth 4
```

Set `--latent-action-temporal-depth 0` to recover the original V41 behavior where all horizon tokens update through every decoder block.

## New diagnostics

Training logs now include latent-main diagnostics in addition to legacy residual placeholders:

```text
lmem=<number of injected layer memories>
lupd=<mean action-token update mask over blocks/horizon>
lgate=<mean event gripper gate>
lgrad=<gradient norm of the latent main decoder>
```

For the default temporal-depth mask, `lupd` should be about `0.792` because the active action-token counts are `[24,24,20,20,16,16,16,16]` over 8 blocks.

## Scripts

Smoke run:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v41_1_latent_main_temporal_depth_smoke_b8_20.sh \
  > runs_v41_1_latent_main_temporal_depth_smoke_b8_20.log 2>&1 &
```

Formal run:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v41_1_latent_main_temporal_depth_policy.sh \
  > runs_v41_1_latent_main_temporal_depth_policy_b8.log 2>&1 &
```

No `HF_HOME` or `TRANSFORMERS_CACHE` is set by these scripts.
