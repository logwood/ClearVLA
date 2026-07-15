# Policy Package Layout

This branch moves implementation ownership without changing policy behavior.
The experiment modules remain compatibility facades until Golden and real-data
replay both pass.

## Current Ownership

- `clearvla.policy.primitives`: shared position, time, and FFN primitives.
- `clearvla.policy.config`: V362/V38/V39 configuration lineage.
- `clearvla.policy.contracts`: layer-contract constants and gradient views.
- `clearvla.policy.codec`: native action to physical-flow coordinates.
- `clearvla.policy.proposal`: rejectable executed-history proposal.
- `clearvla.policy.intent`: typed condition organization and intent contracts.
- `clearvla.policy.evidence`: evidence memory, retrieval, and stage promotion.
- `clearvla.policy.refinement`: stage-owned nested contraction operators.
- `clearvla.policy.decoder`: serial-owned MMDiT action refinement and readout.
- `clearvla.policy.trunk_primitives`: shared V38 world/action trunk components.
- `clearvla.policy.trunk`: current DiT trunk and layer-contract heads.
- `clearvla.policy.system`: top-level current policy composition.
- `clearvla.policy.gauges`: pure diagnostic reductions.
- `clearvla.policy.legacy`: residual, latent-main, and CVAE decoder generations.

The dependency direction is strictly inward: files under `clearvla.policy`
cannot import `clearvla.experiments`. Legacy paths re-export the packaged class
objects, so old imports and old full-object pickle references remain resolvable.
Module attributes, constructor statement order, state-dict paths, forward
signatures, metric keys, and numerical expressions are frozen during this
phase. V84 adds the contraction sidecar, stage/dwell controller, exhaustion
probes, and oracle exit head inside these existing ownership boundaries; it
does not move evidence values back into the decoder or experiment facade.

`policy_v39.py` is now a compatibility facade only: it contains no class or
function definitions. Runtime, evaluation, and CLI code intentionally continue
to import that facade until dynamic Golden and real-data replay pass. The next
refactor boundary is runtime/training; splitting the dense legacy CVAE file more
finely is optional cleanup and not required to promote the current policy.

## Promotion Gates

1. packaged and legacy import paths resolve to identical class objects;
2. no reverse import from `clearvla.policy` into experiment modules;
3. baseline and candidate Golden captures match through checkpoint;
4. one-epoch legacy/candidate replay matches data order, checkpoint categories,
   optimizer ownership, metric keys, and training behavior;
5. only then may CLI imports point directly at `clearvla.policy`.
