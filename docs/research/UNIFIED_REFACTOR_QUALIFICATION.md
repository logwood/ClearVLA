# Unified refactor continuation — source and test qualification

Branch: `codex/causal-unified-refactor-20261008`.
Continuation input tree: `c88b6ca9ba1103b97a8937c4c2ce7fd98e51baad`
(remote source `6fe0a5d8`; inherited implementation `be48d7fd`,
fixture correction `0644e694`).

## Production corrections

1. `model/canonical_grounding.py`: use producer validity before multiplying
   owner/typed priors into a rasterized measure. The integrated ordinary backward
   test checks finite output/gradients, zero unavailable-prior gradient and
   nonzero valid-prior gradient.
2. `model/observation_association.py` and `model/source_measurement.py`:
   measured successor images own their spatial support. The current mask limits
   the source only. Explicit per-successor cell support is validated and
   quarantined before finite checks; unavailable destinations remain unknown.
   A known descriptor move into a currently masked cell is the positive control.
3. `training/identity.py`: exclude all-unobserved groups from group means using
   the producer's support counts. Correspondence and source prediction have
   different valid sets and keep separate denominators. No new identity label,
   multiplier, learned-confidence mask or negative-pair proposal is introduced.

Before these fixes, the focused regression produced seven failures, including
zero measured displacement for a known +2 chart-coordinate displacement and a
halving of identity loss after adding an all-unknown group. These are component
counterexamples, not learned-checkpoint behavior or physical-motion accuracy.

## Complementary test repair

The first complementary run on `6fe0a5d8` retained nine failures. Three minimal
candidate-reuse fixtures omitted the production context-selector method; a
multitask fixture omitted the action target's `row_valid`; one optional T5
asset was absent; four older decoder expectations described an obsolete graph.

The tests now call the real context selector, supply explicit true action-row
support, and distinguish the external asset test from a synthetic file-format
test. Decoder checks retain common-parameter equality, fixed-graph evaluation
and seeded-training repeatability, while distinguishing different execution
graphs. A single reader family has zero across-family diversity; an actual
two-family test still requires positive diversity. Configuration rejection
calls its public `validate()` method. Neutral legacy update logits match the
existing compatibility constant 20, not the unrelated depth initialization.
No legacy production output was changed to satisfy these tests, no tolerance
was widened, and original failing artifacts remain in Actions.

## Audit routes

`run_unified_refactor_checks.py` defaults to all tracked test files; the previous
primary subset remains explicitly selectable. Each selected file is executed
in an isolated process. Syntax and explicit detach locations are inventoried,
not advertised as a formal proof of dynamic autograd.

`check_unified_model_flow.py` performs ordinary optimizer updates and complete
proposal/W-rebuild/refined sampling, recording each trainable parameter's
missing/zero/nonzero gradient and actual update. A nonzero aggregate norm does
not certify every parameter. Inputs to this gate are artificial, with no
pretrained DINO/T5 loading or real-data behavior claim.

`run_unified_real_data_check.py` must separately admit actual CUDA, source
dataset, language/visual assets and source checkpoint before executing the
training CLI. An admission failure is not a passed real-data training test.

## Read-only configuration contract follow-up

The static gate correctly failed on the new optional-source fixture and on a
renamed legacy check that exposed mutable configuration protocols against
frozen dataclasses. The fixture now explicitly requires its image source;
the decoder, codec, controller, evidence, intent and trunk configuration
protocols expose read-only properties. Both the retained implementation and
mainline consume these fields without mutation. No casts, ignores or reduced
type-checking rules are used to hide the incompatibility.

AST checks retain every executable calculation outside protocol declarations
(module docstrings/import ordering excepted). Configuration validation, model
weights, initialization order and optimizer updates are unchanged by this
follow-up. Only the named unified source migration admits these additional
type-only mainline paths; the original A/B migration is not broadened.

## Nondegenerate gradient qualification and training receipts

The expanded legacy suite exposed four zero-gradient assertions in fixtures
with only one older history key. A one-element softmax is identically one;
query-normalization gradients are mathematically zero, and tiny older-kernel
roundoff is not evidence of a live selector. Complete-model all-parameter
checks now declare two distinct older keys, retain the strict no-missing/
no-zero assertions, and separately preserve singleton and multi-key controls.
No proposal calculation, source rule or loss coefficient is changed for this
test correction.

The real-data command does not turn a process exit code alone into training
acceptance. Successful completion also requires matching source and BS8
identity, finite train/validation epoch records, the declared optimizer-update
count and nonempty archived artifacts. This receipt check does not validate
checkpoint tensor contents or behavioral improvement. Missing real assets
remain blocked before any training execution.

## Remaining scientific gates

Instance-level physical identity, natural target choice, long-gap observed
motion, task maintenance, arm/gripper compatibility, native controller state,
cross-dataset learned behavior and resource qualification remain unresolved
until their actual source-specific tests pass. Keep the detailed research
ledger and original A/B promotion blocks; these numerical/interface repairs
do not override them.

## Typed S ordinary-gradient boundary

The inherited typed value decomposition still used `preserve_common_grad=True`.
Although its forward recombination is identity, its backward is `I+P` (P is
interval averaging), doubling the common direction without adding interval
information. This was not repaired by valid-source masking.

`top.typed_interval_gradient_mode=ordinary_v1` now uses the true decomposition
Jacobian. The historical `legacy_common_surrogate_v1` remains the omitted default
for exact legacy replay; old A/B configs and weights are not relabelled.
Only unified check configs select the new mode. The mode propagates through
policy -> top -> S, enters config/ABI identity, and is admitted for initialization
only by the explicit unified source migration. Forward values, parameter set,
constructor RNG and initial sampling outputs remain unchanged; training updates
will differ and require fresh qualification. No gain or binding distribution is
changed. Direct VJP/JVP/gradcheck and actual online-S reconstruction tests exercise
the difference, rather than certifying the lane from an aggregate gradient norm.

## Execution-phase gradient qualification

Warmup deliberately disables nine execution-controller/operator parameter
paths in the small full-topology fixture. The new `--completed-step` fixture
option selects the execution gates while retaining a fresh local optimizer
warmup; it does not pretend that random parameters were trained for that many
updates. A/B tests compare clock 0 and clock 1200 using ordinary production
losses, two optimizer updates and the full proposal/W/refined sampler. The
nine missing warmup paths must become present, nonzero and actually updated
after gate opening. Every other trainable parameter must also have a gradient
tensor then; exact branch-specific zeros remain visible in the ledger and
are not converted into false activity claims.
