#!/usr/bin/env bash
# =============================================================================
# V72: shelf discipline -- cut the action->progress->workspace echo.
#
# Chain: v69 arm-manifold base -> v71 (logit gate OFF; H3 + autocast +
# instruments unconditional) -> v72 (this) adds exactly ONE structural
# variable: --latent-cvae-progress-action-isolation 1.
#
# Rationale (measured in v69a/b, 8 epochs):
#   - The per-step progress update fed on the raw action summary; progress is
#     a workspace evidence source, so the action's own content returned as
#     "world evidence" one refine step later -- invisible to attention-share
#     gauges (it rides inside progress VALUES, not as a separate source).
#   - wpupd (progress update norm) grew monotonically 5.96 (E1) -> 10.0/13.8
#     (E8, a/b) while val was saturated from E5: capacity flowing into a
#     channel with no deploy payoff.
#   - Doctrine: the shelf is for world evidence; content the action wrote
#     must not come back as evidence.
#
# Instruments shipped with the same code state (unconditional, both arms):
#   - wpact  : progress_action_dependence probe -- fraction of the progress
#              update attributable to the action input (deterministic double
#              forward, detached, no RNG consumed; reads 0 under isolation).
#              Read the v72-OFF arm (=v71) for the conviction number.
#   - mdnaT / mdwaT : x_t and workspace attention shares stratified by flow
#              time (3 buckets; sum/count keys so epoch means are exact).
#              The S3 verdict gauge: legitimate flow-matching need for x_t
#              lives at HIGH t; share that persists at LOW t (x_t ~ oracle at
#              train, ~ own output at deploy) is the shortcut signature.
#   - console: wprog now shows the progress attention SHARE, wpupd the update
#              norm (old wprog); dead cm* micro keys removed.
#
# Reading the A/B (v71 vs v72, paired seed):
#   - If val improves or holds with wpupd collapsing: echo was parasitic ->
#     keep isolation on going forward.
#   - If val degrades materially: the "echo" was load-bearing recurrent state;
#     revisit with a supervised-progress replacement before re-cutting.
#   - Expect mdwaT/wprog to rise if progress becomes a genuine evidence
#     carrier once it can no longer relay action content.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARM_NOISE_TEMPORAL_RHO="${ARM_NOISE_TEMPORAL_RHO:-0.0}"
export OUT_DIR="${OUT_DIR:-runs/v72_progress_isolation_rho${ARM_NOISE_TEMPORAL_RHO}_b8}"

exec bash "${SCRIPT_DIR}/current_v71_h3_isolation.sh" \
  --latent-cvae-progress-action-isolation 1 \
  "$@"
