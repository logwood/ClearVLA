# Policy Golden Validation

This harness freezes behavior before the policy monolith is split. It does not
load a trained checkpoint and it does not build a parallel test-only model.
Every capture calls the production `V39PolicySystem`, production loss
composition, optimizer grouping, clipping, sampling, and checkpoint filters.

## Stages

1. **Fixture**: one deterministic CPU batch with the production 24-step action
   horizon, eight trunk layers, arm motion, two gripper transitions, visual
   history, future visual targets, and fixed sampling noise.
2. **Construction**: ordered modules, state entries, parameter aliases,
   optimizer groups, initialization tensors, and parameter accounting.
3. **Boundaries**: physical codec, condition organizer, intent contracts,
   prepared evidence, workspace step zero, and complete action decoder.
4. **Train steps**: two full future/counterfactual forwards with independent
   deterministic RNG phases, deploy loss, layer auxiliary loss, backward,
   diagnostics, local/global clipping, optimizer, and scheduler. The second
   update exercises upstream paths hidden behind zero-initialized output heads.
5. **Sampling**: fixed-noise sampling before and after the optimizer step,
   including event/motion and manifold drift diagnostics.
6. **Checkpoint protocol**: deterministic stage-one migration categories,
   unexpected-key rejection, and strict synthetic resume of model, optimizer,
   and scheduler state.

Each stage owns an independent RNG seed. Baseline and candidate are imported in
separate processes. A model alias, a changed registration order, or an
accidental extra random draw therefore remains visible.

Every captured floating-point tensor must be finite. The builder rejects NaN
or infinity before writing, the reader rejects them again before comparison,
and exact comparisons use `torch.equal`. Optimizer construction also rejects
unassigned trainable parameters, assigned frozen parameters, and duplicate
parameter ownership.

Use `--through construction`, `--through boundaries`, or `--through train` for
fast intermediate gates. `--through sample` adds fixed-noise pre/post sampling;
the default `--through checkpoint` runs the complete protocol.

## Variants

- `v76`: `serial_owned_v2`, projected branch scale.
- `v77`: `serial_owned_rms_v3`, identifiable post-normalization RMS scale.
- `v84`: `post_gate_contraction_sidecar_v11_oracle_router`, V77 host blocks
  with stage-owned nested contraction and a detached-target exit controller.

The variants are complete presets, not individual flags. Compare a refactor
against the matching historical preset. A direct V76-to-V77 comparison is an
intentional semantic diff and should not pass.

For the V84 package migration, compare the monolithic V84 source and packaged
candidate with `--variant v84`. The class extraction intentionally changes the
`structure/modules` owner strings; review that manifest and, if it is the only
structural difference, ignore that prefix while keeping all tensors, state
keys, optimizer ownership, gradients, and checkpoint categories exact.

## Worktree Comparison

Run from the repository root on a machine with the training environment:

```bash
bash scripts/run_policy_golden_worktrees.sh \
  <fixed-v77-baseline-ref> \
  HEAD \
  v77 \
  runs/golden/v77
```

For the reconstructed V76 boundary:

```bash
bash scripts/run_policy_golden_worktrees.sh \
  v76-owned-intent-mmdit-checkpoint \
  <candidate-ref> \
  v76 \
  runs/golden/v76
```

The candidate must be committed or tagged because the worktree runner compares
Git objects. Historical V76/V77 objects predate tracking of the Python package
`clearvla/data`: the runner supplies that one support tree from the current
checkout only when a ref does not contain it. Both effective trees are hashed
file by file into each capture, so this compatibility overlay cannot silently
change one side of the comparison. New checkpoints must track `clearvla/data`;
the repository-level dataset directory alone is ignored as `/data/`.

The original `v77-gate-identifiable-checkpoint` is a v1 historical reference,
not a valid v2 health baseline. The stricter capture correctly rejects its
orphaned trainable `planner.content_mod_scale`; its archived v1 artifact remains
useful for auditing the old behavior. Establish the next worktree baseline only
after the fixed current tree passes the v2 self-check and a real one-epoch smoke.

Before committing a new semantic baseline, validate the current working tree
in two independent processes:

```bash
bash scripts/run_policy_golden_current.sh \
  v77 \
  runs/golden/v77-current-health
```

This is a determinism and health check, not historical equivalence. Once it
passes and the intended semantic changes have survived a real training smoke,
commit/tag that source and use the worktree runner for subsequent refactoring.

During local development, capture an uncommitted candidate source directly in
a second process:

```bash
export PYTHONHASHSEED=0

python clearvla/tools/policy_golden.py fixture \
  --output runs/golden/dev/fixture.pt

python clearvla/tools/policy_golden.py capture \
  --source-root /path/to/baseline-worktree \
  --fixture runs/golden/dev/fixture.pt \
  --variant v77 \
  --output runs/golden/dev/baseline

python clearvla/tools/policy_golden.py capture \
  --source-root . \
  --fixture runs/golden/dev/fixture.pt \
  --variant v77 \
  --output runs/golden/dev/candidate

python clearvla/tools/policy_golden.py compare \
  --baseline runs/golden/dev/baseline \
  --candidate runs/golden/dev/candidate \
  --report runs/golden/dev/comparison.json
```

The default comparison is exact. Nonzero tolerances or ignored artifact prefixes
must be justified by an intended semantic change and recorded with the result;
they are not the default escape hatch for a structural refactor. An intentional
class extraction may use `--ignore-prefix structure/modules` after reviewing
that manifest diff. This does not ignore state keys, optimizer ownership,
checkpoint categories, gradients, or behavior.

## Refactor Acceptance

Use the harness as a sequence of gates:

1. construction and boundary capture while extracting one module;
2. complete two-step training equality after wiring the module back;
3. sampling and checkpoint equality before deleting the old implementation;
4. one real training smoke for import/device/data integration;
5. legacy-versus-real training only after synthetic equality is exact.

Missing/skipped key order is part of the checkpoint artifact. A refactor that
silently hides those differences has not passed.

The same applies to source dependencies: `metadata/support_trees` records the
logical path, aggregate digest, and every file digest used outside a historical
Git object. A support-tree mismatch is a failed comparison, not ignorable
environment noise.
