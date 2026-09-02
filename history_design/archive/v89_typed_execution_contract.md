# V89 Typed Execution Contract

## Problem

V88 used six semantic stage indices for three separate responsibilities:

- memory-stage identity;
- nested contraction basis identity;
- learned dwell candidate identity.

It also let the unified controller multiply the completed host update by a
second `update_keep`.  The result was not one coherent gate.  It was two
amplitude owners plus a stage label that could change capacity without changing
the underlying full-rank function.

## Ownership

V89 keeps one central multi-token control state and typed actuators:

| Quantity | Owner | May do | Must not do |
| --- | --- | --- | --- |
| evidence retrieval | unified controller/workspace | choose addresses and roles | rewrite evidence values |
| dwell/function choice | value reader | choose current/next real MMDiT block | treat memory stages as functions |
| residual amplitude | block host LayerScale | set physical write strength | receive a second controller keep |
| compute capacity | controller branch-budget reader | contract an ordered block-owned subspace | amplify or invent update directions |
| stage memory | workspace | preserve role/content evidence | index a function basis |
| DCT aperture | spectral reader | continuously expose frequency capacity | change bridge dimensionality |

The gates therefore are not scattered.  Policy state and cross-actuator
coordination stay central; only the final actuator for a quantity lives beside
the function that owns that quantity.

## Topology

- The executable repertoire has `hierarchical_mmdit_depth` candidates.
- The memory repertoire keeps `hierarchical_mmdit_operator_stages` stage slots.
- Each full MMDiT block owns one nested contraction basis per branch.
- The controller operator head emits branch-depth logits only.
- `operator_update_logits` remains a constant compatibility/audit tensor at the
  neutral keep boundary and has no parameters in this mode.
- Candidate value probes compare real block outputs under the same committed
  action, workspace evidence, controller state, and DCT aperture proposal.

## Compatibility

`legacy_stage_keep` preserves the V88 topology and parameter shapes.
`typed_block_budget` enables V89.  The two modes must never silently infer one
another from checkpoint contents.

## Diagnostics

`hmfcand=cosine/diversity/update_rms/update_spread/valid_count` measures actual
candidate behavior in physical flow space.  It is the primary function
specialization instrument; controller token diversity alone is insufficient.

Required invariants:

- `hmkeep=1/1/1/1/1` in V89 because no external update keep exists;
- control operation count equals full-rank block count;
- every block/branch contraction bank has stage count one;
- maximum depth is the exact host operation;
- memory stage count and execution operation count remain separately logged.
- execution logs print `hmstage=memory-only`; stage shelf activity must be read
  from workspace retrieval metrics, while `hmblock` owns dwell/function usage.
