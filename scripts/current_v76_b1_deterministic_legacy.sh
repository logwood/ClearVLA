#!/usr/bin/env bash
# =============================================================================
# CR1 / B1: deterministic legacy mapping (do_before_v76 §15, recovery point
# cvae-r1-deterministic-legacy).
#
# Exactly ONE variable vs the V75 baseline (B0):
#   --latent-cvae-variational 0
#     - posterior encoder branch never runs (no q, no sampling, no KL,
#       no auxiliary full decode, no posterior recon loss);
#     - the deploy mapping is BIT-IDENTICAL to B0: z = mu_p(cond) with the
#       same tanh mu_bound / logvar clamps (their removal is B2/B3 scope).
#
# This arm answers the program's gate question:
#   "Does variational TRAINING itself buy reproducible value?"
# Read (paired seed vs B0=V75):
#   - B1 ~= B0 on E3 val  -> CVAE-ness convicted as dead weight; Part B may
#     proceed on the deterministic premise; also pockets ~2x decode compute.
#   - B1 <  B0            -> posterior aux path earns real money; STOP Part B
#     design and attribute (posterior supervision? bottleneck shaping?)
#     before building ICC on a false premise.
# Expected side effects: ckl/cpz/cmug/post_* gauges read 0 (branch off);
# train wall-time per batch drops noticeably; deploy metrics at E0 identical.
#
# PROTOCOL: run B0 (current_v75_hierarchical_workspace.sh), B1 (this) and
# v76a (current_v76a_owned_intent_mmdit.sh) with the SAME seed and data order.
# Attribution chain: (B0-B1) = variational scaffold value; (B0-v76a) = full
# replacement value; (B1-v76a) = ICC/ownership value net of variational.
# Judgments at E3 epoch summaries, never at batch snapshots (v73b lesson).
#
# GAUGE CAVEATS on legacy arms (B0/B1):
#   - wevt reads 0 by construction (legacy bank has no source mapped to the
#     "event" role); read transition-family share via otrn/wtrans instead.
#   - rstem (legacy stem effect) and, in diagnostic runs, zzero/zshuf are the
#     CR0 attribution gauges -- they exist ONLY on legacy arms.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-runs/v76_b1_deterministic_legacy_b8}"

exec bash "${SCRIPT_DIR}/current_v75_hierarchical_workspace.sh" \
  --latent-cvae-variational 0 \
  "$@"
