# Structural rebuild Pen and CALVIN adapters

`configs/mainline/structural_rebuild_pen_current.json` is the Pen outlet
adapter for the cumulative structural graph. It keeps the Pen native chart
(`identity_7d_pen`, 7-D state/action, continuous gripper and strict complete
windows) while selecting the graph pieces that do not depend on CALVIN's
relative command or annotation producers:

- source-timestamped state/control history;
- full-posterior coupled observation support and completed G3 current-image
  entities;
- a shared K+null operation binding;
- coordinate-role camera conditioning and the 24-row sequence-prefix W path;
- control-aligned 4/8/16/24 world support, matched observed-action
  supervision, view-conditioned P2 and typed-horizon P3;
- magnitude-preserving evidence, separate controller values and typed
  transition conditioning;
- the registered Q5 proposal/refined deployment schedule.

The adapter leaves CALVIN-only source producers disabled: rotation6D state,
instruction-start reference, binary command head, executed-world feedback,
proprioceptive response and annotated-endpoint supervision. Pen remains on its
native affine state chart and continuous action finalizer. This is an explicit
structural Pen selection; it is not a relabelled CALVIN checkpoint or a change
to the historical default config.

CALVIN remains a separate explicit outlet in
`configs/mainline/structural_rebuild_m6n_calvin.json`. Its adapter keeps the
structural source chart and enables CALVIN's relative-command conversion,
source-timestamped history, 24-row sequence-prefix W, rotation6D state,
shared object binding, instruction-start reference, observed robot views,
clean endpoint supervision and binary gripper finalization. The Pen adapter
never mutates or aliases this CALVIN selection.

Both configs are data-path identities, not evidence of trained behavior. Load
and validate them before real-data qualification; missing configured remote
paths remain a separate gate.
