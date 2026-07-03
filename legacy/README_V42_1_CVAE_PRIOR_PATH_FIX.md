# V42.1 CVAE prior-path fix

This patch fixes the main failure mode seen in early V42 logs: the training loss was optimized through
`q(z | condition, target_action)` while inference used `p(z | condition)`, so the posterior latent could
hide target-action information that the deploy path would never see.

## Main changes

- `latent_cvae_action` now always returns the prior/deploy path as `pred_velocity`.
- During training, the posterior path is decoded separately as `post_pred_velocity`.
- The normal policy losses operate on the prior path.
- The posterior path is only a weak auxiliary reconstruction loss.
- Added diagnostics:
  - `latent_cvae_prior_z_norm`
  - `latent_cvae_post_z_norm`
  - `latent_cvae_mu_gap`
  - `latent_cvae_post_flow`
  - `latent_cvae_post_decoded_action`
  - `latent_cvae_posterior_recon`
- Default KL is raised to `5e-4`.
- Added trainer knob: `--latent-cvae-posterior-recon-weight`, default `0.25`.

## Expected log change

Batch logs now print:

```text
cz=<prior z norm> cpz=<posterior z norm> cmug=<posterior/prior mu gap> ckl=<KL> cpflow=<posterior flow> cstd=<prior std>
```

The key check is that the main validation metrics now reflect the prior path, not the posterior path.
If `cpz/cmug/ckl` grow but validation does not improve, the posterior is still too strong or the prior
condition is not action-sufficient enough.
