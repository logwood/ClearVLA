# Structural rebuild Pen preflight

`configs/mainline/structural_rebuild_pen_current.json` is the structural
rebuild's explicit copy of the current Pen `object_intent_dynamics_323`
configuration. It keeps the current eight-epoch budget and data identity, and
spells out the structural branch defaults that the compact baseline config
normally omits. In particular, `top.world_action_condition_mode` is explicitly
`interval_mean_v1` and `top.world_camera_condition_mode` is explicitly
`motion_prior_only`.

The current A/B coordinate experiment JSON files are not interchangeable with
this branch. Their `observation.online_context_mask_mode` and
`observation.coordinate_contract_mode` fields belong to the later canonical
RGB processor ABI and are rejected by this branch's strict config loader. The
branch-native `top.world_camera_condition_mode=coordinate_role_v1` is a W
camera-role condition, not an alias for that processor ABI. The structural Pen
run therefore uses the branch-native baseline semantics until a separate
coordinate ABI port is implemented and tested.

Local gates for this commit are:

- `compileall`, `git diff --check`, and `uv lock --check`;
- the lightweight 67-test suite;
- the structural changed-source audit (Ruff clean; Pyright compared with the
  pinned base; platform-specific Windows diagnostics recorded separately);
- config load/validation and bounded qualification with the explicit Pen config.

The remote gate must still use the exact pushed commit, an immutable detached
worktree, a fresh `--smoke` run, and a new formal run only after smoke passes.
The remote interpreter and Torch build must be checked separately because the
project declares Python 3.12 and Torch 2.11.
