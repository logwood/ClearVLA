# V42 Latent-CVAE Action Head

This package adds `final_action_decoder=latent_cvae_action`.

Design goal:

- Keep the V40.1 data pipeline, normalizer, physical action codec, losses, eval split, DINO cache, and latent/consequence trunk unchanged.
- Do **not** keep the old V40.1 direct/rollout action head as a final-action base.
- Do **not** add a residual/side branch.
- Replace the oversized V41/V41.1 81M Transformer action decoder with a compact CVAE-style action head.

Information path:

```text
V40 latent/consequence trunk
  -> L0...L7 layer summaries + controlled_delta/rollout/event context
  -> LatentCVAEActionDecoder
  -> pred_physical_velocity
```

Training path:

```text
condition c = aggregate(all V40 layer latents, transition memory, trajectory summary)
posterior q(z | c, target physical action)
prior     p(z | c)
decoder   pred_velocity = f(c, z, noisy_physical, trajectory_tokens)
loss      existing flow/action losses + latent_cvae_kl_weight * KL(q || p)
```

Inference path:

```text
condition c = aggregate(all V40 layer latents, transition memory, trajectory summary)
z = mean(p(z | c))        # deterministic by default
pred_velocity = f(c, z, noisy_physical, trajectory_tokens)
```

Key defaults:

```bash
--final-action-decoder latent_cvae_action
--latent-cvae-z-dim 64
--latent-cvae-decoder-depth 3
--latent-cvae-ffn-expansion 2.0
--latent-cvae-layer-memory 1
--latent-cvae-transition-memory 1
--latent-cvae-context-memory 0
--latent-cvae-visual-memory 0
--latent-cvae-event-gripper-gate 1
--latent-cvae-inference-sample 0
--latent-cvae-output-init-std 1e-3
--latent-cvae-kl-weight 1e-4
--latent-cvae-action-decoder-lr-scale 1.0
```

Expected parameter scale with the official 512-hidden config is about 25M for the CVAE head, rather than about 81M for the V41 latent-main Transformer decoder.

Diagnostics added to the batch log:

- `cz`: latent z norm
- `ckl`: KL(q || p)
- `cstd`: prior std
- `cgate`: CVAE gripper gate mean
- `cgrad`: CVAE decoder gradient norm

Run smoke:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v42_latent_cvae_action_smoke_b8_20.sh \
  > runs_v42_latent_cvae_action_smoke_b8_20.log 2>&1 &

tail -f runs_v42_latent_cvae_action_smoke_b8_20.log
```

Run full policy:

```bash
CUDA_VISIBLE_DEVICES=1 nohup ./run_v42_latent_cvae_action_policy.sh \
  > runs_v42_latent_cvae_action_policy_b8.log 2>&1 &

tail -f runs_v42_latent_cvae_action_policy_b8.log
```

No training script in this package exports `HF_HOME` or `TRANSFORMERS_CACHE`.
