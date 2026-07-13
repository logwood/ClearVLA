# Policy Package Layout

This branch moves implementation ownership without changing policy behavior.
The experiment modules remain compatibility facades until Golden and real-data
replay both pass.

## Current Ownership

- `clearvla.policy.primitives`: shared position, time, and FFN primitives.
- `clearvla.policy.contracts`: layer-contract constants and gradient views.
- `clearvla.policy.codec`: native action to physical-flow coordinates.
- `clearvla.policy.intent`: typed condition organization and intent contracts.
- `clearvla.policy.evidence`: evidence memory, retrieval, and stage promotion.
- `clearvla.policy.decoder`: serial-owned MMDiT action refinement and readout.
- `clearvla.policy.gauges`: pure diagnostic reductions.

The dependency direction is strictly inward: files under `clearvla.policy`
cannot import `clearvla.experiments`. Legacy paths re-export the packaged class
objects, so old imports and old full-object pickle references remain resolvable.
Module attributes, constructor statement order, state-dict paths, forward
signatures, metric keys, and numerical expressions are frozen during this
phase.

## Remaining Boundary

`TemporalMidcutWorldActionDiT`, its contract heads, `V39PolicyConfig`, and
`V39PolicySystem` remain in the V39 facade for now. The trunk still has dense
dependencies on V38 world-model components and legacy decoder families; moving
it before those dependencies are isolated would replace one monolith with a
cycle. The next batch first quarantines that legacy cluster, then extracts the
trunk and system composition.

## Promotion Gates

1. packaged and legacy import paths resolve to identical class objects;
2. no reverse import from `clearvla.policy` into experiment modules;
3. baseline and candidate Golden captures match through checkpoint;
4. one-epoch legacy/candidate replay matches data order, checkpoint categories,
   optimizer ownership, metric keys, and training behavior;
5. only then may CLI imports point directly at `clearvla.policy`.
