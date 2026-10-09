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

## Conditional object values at the S/P2 boundary (2026-10-08)

The prior unified source still exported `binding * value` from S and then read
that value with binding again in P2. Its common task supplement could respond
to concentration without distinguishing an equal-mass K swap. In addition,
P2 added the public interval residual after target selection without retaining
real/null mass, so an all-null target could have a nonzero target-value lane.
The legacy counterexample is retained in an executable test.

`top.typed_object_value_mode=conditional_object_v1` is a separately identified
candidate. S produces `G[k,type] * sigmoid(operation[i,type]) *
(1+tanh(query[i,type]))`, quarantining producer-invalid values before products.
This is a per-object source-conditioned value, not a query-only value copied
to K. No K probability is stored inside it. S's policy-context read and P2's
spatial read each consume their single existing binding once. P2's public task
residual retains the actual admitted real mass. All-null yields exactly zero
on this target route, not a requirement for the whole robot command to vanish.
The geometry reader can still select a view inside K; it cannot change K mass.

No parameter, gain, optimizer, source ownership, matcher, loss coefficient,
solver, controller or policy input is added or changed. Forward values and
training gradients DO change; this is not an output-equivalence repair.
Historical configs omit the new field and keep legacy semantics. Value meaning
is explicit in ObjectIntentState, PolicyIntentDock, K permutation, config and
ABI. Diagnostics separately name conditional values while preserving the
meaning of mass-weighted selected-value statistics.

Only the new `causal_unified_values_v1` initialization permits this contract
from the same strictly admitted mature source. Previous A/B and unified source
migration allowlists are not extended. The new A/B check presets and trainer
parser are wired through the actual config, factory and runtime. The real-data
checker requires `--migration causal_unified_values_v1` for these presets.

Checks include ordinary double-precision first/second derivatives, source-zero,
invalid NaN quarantine, K permutation, batch independence, equal-entropy target
swap, producer-to-dock identity, actual P2 with 0/0.5/0.999/1 null fractions,
legacy counterexample, constructor/parameter parity, ABI rejection, migration,
and two ordinary optimizer updates plus the complete two-pass sampler in A/B.
Inputs remain artificial and do not certify learned objects or task success.

This candidate does not solve unqualified long-gap correspondence, physical
instance separation or arm/gripper behavior. Nor does it remove the intentional
posterior expectation across future intervals: centered values may legitimately
cancel there. Those independent issues and all previous promotion blocks remain.

## Actual batch and autocast qualification

The full-topology fixture now declares its actual synthetic batch size and
serialized compute dtype. Sampler noise, endpoint shape and optional identity
source labels use that same batch; optimizer batch metadata no longer reports
a different size from the artificial batch. Defaults remain B1/FP32. BF16 uses
the production autocast contexts, not a result merely converted to BF16.

Separate A/B regressions exercise BS8 CPU BF16, ordinary loss backward, two
AdamW updates and complete proposal/W/refined sampling. These remain artificial
inputs and random weights, not real pretrained BS8 training. The local 4-GiB
container could complete B1 BF16 but its initial BS8 attempt was terminated
under memory pressure; that is a resource-blocked attempt, not a passing check.
Retain the remote runtime/RSS and test receipts before qualifying BS8.


## Independent current/reference observation support (2026-10-08)

The source-consistent S reader still restricted current G probability by
`current_observed & reference_observed` at identical pixel coordinates. A
current cell at x=-1 was therefore discarded even when its exact descriptor
was observed at reference x=+1. The pre-fix actual S component returned zero
source mass and zero displacement for this constructed -2 chart-coordinate
move. This is distinct from the already fixed successor Teacher mask issue.

Current G/current image support and reference destination support now own
their separate masks. Their same-coordinate intersection remains diagnostic
only. `InstructionPosterior` retains both masks, checks each probability
against its own producer, and preserves source = real + unknown mass. P3's
prepared nonlinear content/joint projections quarantine each image using its
own mask. The endpoint goal/current comparison also uses current support,
not the old intersection. Current masking follows canonical G's existing
nearest source-mask projection onto the declared endpoint chart. Missing
reference images keep current source mass with unknown=1, and emit no visual
change. This does not invent a correspondence for ambiguous descriptors.

Legacy learned matching keeps its old shared-mask calculation and omitted
optional fields. The source-consistent instruction metadata advances to v2;
the causal ABI states `independent-current-reference-v1`. Changed-source
initialization is admitted only through `causal_unified_reference_v1`, which
extends the values migration with the posterior source type and endpoint
comparison files. Previous A/B/source/values allowlists are not broadened.
The new mode still requires the original mature source identity, explicit
conditional values and the previously required complete repair selections.
It adds no parameters, labels, cameras, loss weights or policy selectors.

Focused tests cover disjoint but matched domains, missing current/reference,
independent NaN quarantine, strict probability-domain rejection, K permutation,
same-source no change, the original legacy counterexample, S/P3 ordinary
backward, actual source-log-measure VJP versus finite differences, the endpoint
consumer, metadata and migration rejection. Descriptor matches and -2 values
are synthetic chart-coordinate checks, not physical meters or new checkpoint
motion accuracy. Full-topology and real-data qualification remain separate.

Real-data command on an asset-owning CUDA host (use an actual source checkpoint):
`python scripts/run_unified_real_data_check.py --config configs/mainline/unified_values_b_calvin_check.json --migration causal_unified_reference_v1 --checkpoint <mature-source.pt> --output-dir <new-run-directory> --report <new-report.json> --device cuda:0 --execute`.
Missing private data/weights/CUDA is still blocked, not replaced with artificial
features or relabelled as training success. Historical behavior scores remain
unchanged until the corresponding newly trained policy is actually evaluated.


## Combined candidate and partial-reference integration

The default full-topology check remains its declared legacy typed-value fixture;
that alone cannot qualify the selected conditional-value candidate with missing
instruction-reference cells. An explicit `--reference-support alternating`
fixture now leaves current/future support unchanged and marks half the start
cells unavailable with NaN payload. It does not manufacture a displacement label
or replace any online module. The full-width A/B regression selects conditional
object values, source-consistent measurement, H512, 336px artificial RGB, gate
clock 1200, two ordinary updates and full two-pass sampling. JUnit records the
actual source configuration, support counts, batch/dtype and runtime. These are
CPU BS1/FP32 checks; the separate small BS8/BF16 and all real-data/behavior gates
retain their own scope and outcomes.


## Measured robot result consumption (2026-10-08 continuation)

The source previously retained `RobotResponseFeedback.observed_delta` and
`predicted_delta` but its explicit reader consumed only innovation. S's
`before_proposal_v1` read measured WORLD results, not that one-step robot packet.
Two correctly predicted results (stall/motion) could therefore have the same
zero explicit robot feedback. Existing RGB/proprioceptive history was still
available; this does not claim those sensors were absent.

The opt-in `robot_world_before_proposal_v2` passes the existing robot packet
from policy encoding into the S interval producer. The internal
`ObservedRobotOutcomeRead` consumes actual current state, confirmed command,
actual feature delta and an observation witness, modulated by the existing
interval context. Response prediction and innovation are not arguments to its
value calculation. No new sensor, physical unit conversion, contact flag,
controller setpoint, phase threshold, selector, loss or G/W replay is created.
The single existing world outcome read is unchanged. Mechanical facts can be
available with a null visual target; they do not assert object identity.

Producer identity, exact one-step offsets, observed finite inputs and actual
state-difference equality are checked. Missing-source NaNs are quarantined
before any projection. The new output starts at zero and only three new
parameter tensors are added; inherited random initialization and initial full
sampler outputs are preserved. Ordinary task loss reaches these parameters
once the zero-start output begins learning; observation tensors and response
predictor cannot be rewritten by that task gradient.

The graph is serialized in config and causal ABI, requires ordinary conditional
S values and one-step measured robot feedback, and uses only the new
`causal_unified_outcome_v1` migration. Its exact parameter inventory and neutral
initialization are required; earlier source allowlists are not broadened. A/B
check configs are `unified_outcomes_{a,b}_calvin_check.json`. The corresponding
real-data admission command must select `--migration causal_unified_outcome_v1`.
These are bounded training CHECKS, not full training budgets or promoted models.

Tests distinguish unchanged/changed result at identical zero innovation,
forecast-only noninterference, missing-source NaNs, observed zero vs unknown,
ordinary first/second context derivatives, batch/interval permutations,
constructor RNG, source identity through conditioning, ABI/migration rejection,
two ordinary updates and full proposal/W/refined sampling. The full-policy
lifecycle test follows the CONDITIONED producer packet (the conditioning layer
can replace source wrappers); it does not compare the raw preconditioning
wrapper by identity. Original failed fixture output remains an audit artifact.

No behavior or real-data result is inferred from these artificial checks.
Physical instance identity, target choice, contact-compatible arm/gripper,
long-gap correspondence, controller state and sustained task maintenance retain
their outstanding scientific gates. An actual asset-owning CUDA training run
and new closed loops are still required.

## Typed registration transfer and complete optional outcomes

The constructor's temporary-owner extraction returned `object` and did not
validate concrete stage types before removing them from the old registry.
It now requires an explicit expected type, rejects mismatches before mutation,
and returns the same registered object. Parameter identity/order, constructor
RNG and state_dict values are preserved in full A/B constructor comparisons.
No new numerical expression, model tensor or controller policy is introduced.

World feedback validation now explicitly narrows all four optional outcome
fields before tensor operations. The all-absent legacy record remains legal;
every partial record is rejected. This retains the old all-or-none rule rather
than ignoring Optional diagnostics. Focused checks retain wrong-type sources,
verify exact module/parameter/buffer moves, and exercise complete, missing and
inconsistent world outcome records. The 23 inherited type errors in these two
changed producer/constructor files are eliminated; this is not a claim that
every untyped member warning across the repository is resolved.


## P3 execution context uses the same missing-source mask (2026-10-08)

The new S robot-outcome reader already quarantined absent context, but the older
P3 robot/world feedback consumers only zeroed their values. An absent row with
NaN/Inf context still entered a Linear/tanh and produced zero-times-NaN output
and nonfinite shared-parameter gradients. Fourteen new boundary assertions
failed on de390553 (ten robot producer/read checks and four world read checks);
these are artificial missing-source counterexamples, not measured failures of
a trained policy.

Both original P3 feedback readers now quarantine absent context BEFORE the
projection, using the existing observed transition/window Boolean only. World
prepared value and status use the same mask. Observed zero innovation is not
a mask: a supported uncertain comparison retains its status value. No target
threshold, new source, loss, selector, parameter, or gain is introduced. Finite
valid-source forward values and parameter/context VJPs retain exact parity.
Ordinary first/second derivatives, FP32/BF16 missing payloads, mixed-batch
isolation, and actual online-generated world packets are separately tested.

The robot response producer also checks its already-declared exact one-step
clock and observed finite state/command before prediction. These checks run
once during observation encoding, not inside every ODE node. The per-node
consumer adds only tensor masking and structural shape checks. Invalid
observed data is rejected, never silently converted into a successful record.
No identity/physical-outcome or real-data training acceptance is inferred.
