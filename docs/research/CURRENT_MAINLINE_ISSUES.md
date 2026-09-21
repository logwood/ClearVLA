# ClearVLA current decision ledger

Updated: 2026-09-19 UTC

This file owns unresolved source/behavior questions and compact closure records
needed to avoid reopening resolved implementation defects. The active graph and invariants live in
[00_CURRENT_ARCHITECTURE_CONTRACT.md](00_CURRENT_ARCHITECTURE_CONTRACT.md),
the ordered work in
[CURRENT_MAINLINE_REPAIR_PLAN.md](CURRENT_MAINLINE_REPAIR_PLAN.md), and
dated run observations in
[auxiliary/ACTIVE_MAINLINE_HANDOFF.md](auxiliary/ACTIVE_MAINLINE_HANDOFF.md).
Completed repair narratives belong to Git history or the archive.

**Scope:** each observation applies to its named source, config, data and
checkpoint. The document update date is not a fresh run-status check. Current
workspace repairs must not be attributed retroactively to historical runs;
source closure and learned behavior are recorded separately below.

## Maintenance rule

Each issue has one question, one latest comparable observation, and one
condition for entering source work. Replace an observation when a newer
checkpoint has the same source, data, normalizer, decoder and evaluation
contract. Keep an older observation only when it explains an accepted or
rejected source decision, a contract transition or a causal sign reversal.
Raw rows, console excerpts and full probe dumps do not belong here.

At a decision checkpoint, read the complete available curve once, update this
ledger in place and choose the highest-priority unresolved behavior. Do not
repeatedly rescan an unchanged log to invent a different mechanism.

## Current evidence boundary

| Artifact | Comparable evidence | Boundary |
|---|---|---|
| Pen Schema28 anchor | Eight complete epochs at commit 097330a8 | Latest retained pre-recovery behavior anchor |
| Pen recovery | Eight complete epochs from 0973f192; full/arm/gripper physical RMSE 0.07291 / 0.05543 / 0.13702 | Current best comparable Pen curve; far/event closure remains open |
| Pen Schema29/cachefix | Eight complete epochs from d8a77a1 | Valid rejected-self-conditioning comparator; its cache/VJP lifecycle repair remains independently proven |
| Pen Schema32 matched baseline | Eight complete epochs from c7eccded; same 63/5/5 split, seed 0, batch 8 and action-normalizer fingerprint as the aggressive candidate | Direct baseline for the combined geometry/gripper intervention; E6 is the latest completed matched checkpoint shared by both runs |
| Pen Schema32 aggressive candidate | Six complete epochs plus a partial seventh from 014a26ec; `rms_floored_v1` geometry ingress and `six_channel_consensus_v1` gripper decode are both enabled | Rejected as a combined promotion candidate at matched E6; the two simultaneous interventions cannot provide single-factor attribution |
| RDT-8 Schema29/cachefix | Five complete epochs from d8a77a1 | Current adapter behavior reference |
| RDT-8 recovery | Two complete epochs plus a partial third from 0973f192 | Arm/RMSE are directional only; event/F1 is incompatible because loss/evaluation used the wrong row-zero gripper boundary |
| CALVIN direct-arm | Completed run from f9cee96a | Rejected comparator; it changed the shared core more broadly than the diagnosed outlet defects required |
| CALVIN outlet repair | Converter-v1 formal run rejected after source proof that the final 24 real actions were absent from direct policy targets and annotation rows, not source trajectories, owned split membership | Rebuild with converter-v2 trajectory-disjoint/absorbing-terminal data before any architecture attribution |
| CALVIN object-binding V1 E4 panel | Frozen `34ee78e98fd8cb891f30a4d23fe33ecfebd50271`, epoch 4 / step 105308, checkpoint SHA-256 `327e4a3d1a946568330b76c6b3fc5f57e67c2aa824d6c1c927759d47734d7119`; 34 independently reset trials under the original direct-command, uniform five-step contract | Official oracle success 6/34; push 2/18, target contact 11/18 (left 9/9, right 2/9); this is a frozen behavioral baseline and target-routing warning, not evidence for a unique G/S/W/P2 cause or a matched language counterfactual |
| LIBERO retarget overlay r3 + timing probe | One-epoch component/data-contract initialization from the E2 best weights, with four train-only simulator-retarget episodes; source-native validation full/arm/gripper RMSE 0.213688 / 0.131133 / 0.465255 and decoded event F1 0.1153. A fixed-snapshot 32-step baseline/delayed-close pair held arm[0:6] bit-exact (max delta 0), suppressed 12 close commands, and produced no gripper-object or object-plate contact in either branch. | Closed loop remains 0/9 plus the timing pair is 0/2; early-close timing is not sufficient to explain the failure and there is no evidence that gripper commands drag the arm in this scene. Do not promote the overlay or change the shared codec/loss before diagnosing arm reach/observation feedback. |

Smoke, checkpoint round-trip and read-only replay establish only the boundary
they exercise. Pre-cachefix Schema29 formal runs are invalid behavior evidence:
a CUDA BF16 autocast weight-cache interaction removed formal parameter-owner
VJP after the detached pass while forward values remained finite.

The recovery intake found no demonstrated future-target or split leakage.
Online and future-supervision types are disjoint, evaluation samples from
online inputs, and compared episode membership, normalizers and cache
identities match. This is not a byte-for-byte raw-HDF5 proof because the
current composite dataset inventory does not serialize a separate raw-file
digest.

## Source closure and behavior status

These rows separate implementation decisions from claims about trained models.
They do not refresh process liveness or select a new training/evaluation job;
dated run observations belong in the active handoff.

| Unit | Source / structural evidence | Behavior boundary | Work status |
|---|---|---|---|
| S/coarse/W producer-owned validity | Repaired; invalid-value/VJP isolation and all-invalid finite-zero checks recorded below | Real-data prevalence and task benefit are not established by those checks | Source closed; retain regression gates |
| CALVIN object-binding V1 E4 behavior | Original checkpoint/deployment ABI was verified; final 34-case artifact, trajectory, oracle and video inventory is hash-checked in `artifacts/calvin_e4_20260919_download/analysis/inventory.json` | Strong behavior-level asymmetry and collateral-object evidence are retained; no internal-module attribution is accepted from this rollout alone | Frozen baseline; do not promote or relabel as a matched Q5 comparison |
| Bottom `ffn_expansion` propagation | Repaired; `tests/test_mainline_amp_cache.py::test_decoder_config_propagates_bottom_ffn_expansion` checks the resolved value | The former default was numerically unchanged; no task-gain claim is needed to close the mapping defect | Source closed; supersedes the WIP snapshot's former test TODO |
| Shared S goal/history object queries | Implemented in the current source; the identified `1e51e28` E1 checkpoint already contains the query repair | Bounded color/target-position behavior remains weak; `635213d` findings do not describe that checkpoint | Behavior question open; do not repeat the old missing-query repair |
| W `coordinate_role_v1` camera condition | Opt-in implementation and restricted component-initialization checks passed | No matched learned improvement is established by source admission | Source unit closed; behavior comparison remains separate |
| W `sequence_prefix_v1` action condition | Opt-in implementation, prefix-causality, AMP identity and lifecycle checks passed | Visibility of reordered actions does not establish useful correction after training | Source unit closed; behavior comparison remains separate |
| P2 typed-S spatial/interval placement | Default `post_pool_only` remains intact. The opt-in `shared_target_prior_v1` now supplies one zero-start S-owned target-K address to both semantic K and geometry K*C before pooling, while post-pooling typed S retains only interval selection. Forward/reverse, K/C permutation, validity/non-finite, RNG, Q5 lifecycle, optimizer ownership and checkpoint gates pass locally. | No trained or closed-loop benefit is established; the next claim must measure target-relative action and contact, not merely nonzero internal tensors. | Source unit implemented opt-in; behavior admission open |
| Exact-order visual NPY reader | The reviewed `c77420a` mmap/pread transport was ported into the current source with default mmap identity preserved. Row order/repetition, camera order, token/image bytes, pickle/handle lifecycle and config gates pass locally. The explicit `p2_shared_target_prior_pread_v1` initialization permits only the shared-target unit plus the four reviewed I/O paths. | Full E1 index/cache/normalizer identity, production worker/RNG parity, real frozen-source changed-path admission and end-to-end throughput remain required. | Source integrated opt-in; production admission open |
| S common/residual reverse ownership | The exact `common + residual` route reconstructs one source tensor; its zero common-lane VJP is explained by that decomposition | Independent learned owners are not part of the accepted contract | Explained semantics; new design only if separately selected |

"Source closed" does not mean promoted to the default, fully trained or
behaviorally validated. A separate open behavior question does not reopen the
closed implementation defect without contrary evidence.

## Decision order

Severity and scheduling are separate. **P0 is an admission gate for the
selected run or claim, not an instruction to start another outlet's work.**
The sequence below reflects the current CALVIN/DataLoader scope. It changes no
experiment selection, process state or source ownership. Operational progress
remains in the dated handoff; implementation packages remain in the repair plan.

### Conditional P0 admission gates

| Gate | When it blocks a decision | Retained boundary |
|---|---|---|
| Source, checkpoint and continuation identity | A comparison/init/resume claim fails its applicable source/config/ABI, data, normalizer, split or continuation checks | Identify the exact artifact and lifecycle first. Component initialization is not exact resume; saved metadata alone is not proof of restored sampler/worker state. |
| Outlet and target validity | A claim relies on the known-invalid RDT gripper boundary or converter-v1 CALVIN target/split contract | RDT `0973f192` event/F1 cannot validate the corrected boundary. CALVIN converter-v1 omitted the final 24 real actions and split annotations rather than trajectories. A current corrected-contract claim requires its own admitted data/source; an original-protocol historical screen must retain those limitations. |
| Numerical and ownership integrity | Non-finite values, a failed loss ledger, vanished required parameter-owner VJP or checkpoint ABI failure invalidates the selected result | Apply the existing stop/closure rules to the identified run. Small geometry RMS, finite gradient spikes and uniform route mass are not such failures. |

Reuse valid admission evidence. These gates neither declare every retained run
invalid nor reopen a completed diagnostic round.

### Decisions in the current scope

| Order / state | Decision | Evidence and entry condition | Result to record |
|---|---|---|---|
| 1A — existing evaluation (closed) | Does the historical V1 merit further training time? | Final authorized E4 panel is complete under its original model/data/deployment contract: 6/34 official successes, 2/18 push successes, 11/18 push target contacts, and 0/4 lift or 0/4 drawer successes. | Keep the frozen E4 artifact as a diagnostic baseline and do not promote it as the target-binding solution. The panel does not authorize stopping any trainer; any continuation or stop remains a separate explicit user decision. |
| 1B — separate model comparison | Do B1 camera conditioning or the 24-row W condition improve target-relative behavior? | Source checks are closed. Use the owning task's admitted training/evaluation artifacts with the same E1 start, six-task split, normalizers, Q5 and layouts. A single-factor improvement claim also needs an equal-training-budget legacy control with a legal frozen checkout/init entry. | Keep camera and sequence conclusions separate. Additional training versus the original E1 supports checkpoint selection, not isolated structural causality. Missing controls limit the claim; this row launches no control run. |
| 2 — source integrated; production gate open | Can the visual reader reduce I/O while preserving the existing seeded training stream? | Optional pread is now present without changing the default mmap path. Production admission still requires frozen-source selection, full-index/cache/normalizer identity, full-batch bytes and sample/RNG checks at production worker settings. Six-episode I/O results are not training-throughput or recovery evidence. | End-to-end throughput and memory at equal data/order; enable only in an identified new initialization run. Do not change a running experiment's reader. |
| 3 — structural source implemented; behavior gate open | Does one shared S target-K prior move both P2 spatial reads to the instructed object strongly enough to improve action and contact? | The opt-in contract is implemented without changing W values/support, bottom language access or the interval owner. Local structural and migration gates are closed; real checkpoint admission and autonomous six-task behavior are not. | Require target-relative approach/action response, push direction, non-target contact and success under the identified Q5 run before promotion. |

1A and 1B are independent decisions; the old V1 is not the legacy control for
1B. Reader preparation can proceed alongside them, while production activation
keeps the integration milestone already agreed with the owning task. Details:
[CALVIN structural comparison](CURRENT_MAINLINE_REPAIR_PLAN.md#1a-resolve-calvin-target-routing-across-s-w-and-p2),
[reader integration](CURRENT_MAINLINE_REPAIR_PLAN.md#1b-integrate-the-exact-order-visual-reader-independently),
and [historical V1 evaluation](CURRENT_MAINLINE_REPAIR_PLAN.md#1c-evaluate-the-historical-v1-before-deciding-whether-to-retire-its-trainer).

### Deferred questions and reopening conditions

The following evidence remains relevant, but does not precede the current
sequence or create new diagnostics. An outlet-specific validity gate stays
mandatory if that outlet is selected; deferring its work does not lower its
correctness requirement.

| State | Question | Retained observation | Entry condition if selected |
|---|---|---|---|
| Outlet work, not selected here | Is the corrected RDT gripper contract behaviorally sound? | The historical recovery run declares previous-command ownership but uses current qpos in loss/evaluation. At matched E2 its full/arm/gripper RMSE is 0.29469 / 0.11260 / 0.72925 versus 0.27212 / 0.10324 / 0.67408 for Schema29. | Only a new initialization from the corrected boundary can replace the adapter reference or support event/F1 claims. |
| Outlet work, not selected here | What limits Pen far-horizon and gripper-event behavior? | At matched E6 the combined Schema32 candidate changes full/arm/gripper RMSE from 0.07315 / 0.05615 / 0.13616 to 0.07680 / 0.05789 / 0.14555, decoded F1 from 0.3694 to 0.3532 and event ratio from 0.444 to 0.406. First and band 1-4 improve slightly, but bands 5-12 and 13-24 regress. | Do not promote or tune the combined candidate. A later intervention must isolate one owner, improve far/event closure without worsening near or arm behavior, and show deployed-action responsibility. |
| Scoped engineering debt | Can data provenance and compatibility be read independently? | Metadata-only physical-chart changes alter the composite inventory digest. | If an identity refactor is selected, serialize separate raw inventory, split, profile metadata, normalizer/cache/language and continuation/ABI records. This broad refactor is not a prerequisite for the bounded reader unit, which must still pass its existing full-data identity gates. |
| Unproven behavior mechanism | Are S/W interval semantics under-used in a behavior-limiting way? | Pen recovery E8 public-S variation is 0.16284, typed semantic/appearance/geometry variation 0.01730 / 0.01334 / 0.00959; W variation is 0.07028 versus Teacher 0.12152. | Behavior or a matched intervention must tie the under-use to far/gripper error before changing chronology, bandwidth or loss. |
| Unproven behavior mechanism | Does geometry address or value have stable action utility? | At matched E6, RMS-floor raises W geometry common/interval contribution from 0.2095 / 0.0589 to 0.3311 / 0.0935, while P2 geometry effect falls from 0.02116 to 0.01648 and the combined geometry-zero far-gripper action delta falls from 0.00481 to 0.00432 at 8.94% coverage. Larger W ingress therefore did not create greater P2 or action utility. | If revisited, use a geometry-only matched unit and trace W ingress -> W prediction -> P2 selection -> deployed action. Do not add another gain merely because an upstream RMS is small. |
| Bounded representation risk | Does dynamic P1 precision lose useful basis detail at its two consumers? | Source trace confirms one tensor is intentionally consumed twice: full `[B,24,Q,H]` by ControlledTransition and a separate no-null basis read plus fixed `0.25` ingress by terminal RoleDelta; both paths have nonzero owner/VJP. The matched fixture's basis posterior is effectively uniform (`entropy 0.9999997`, max mass `0.2503`), so an equal-RMS basis-antisymmetric injection is attenuated to about `3.4e-5` at RoleDelta versus `2.5e-2` for a constant injection. The actual dynamic residual in the same fixture is already nearly basis-common (centered ratio `0.012`), so this proves a bounded cancellation risk, not a current learned-data defect. | Obtain the same basis-variation and dual-consumer trace on a real checkpoint before splitting the reader or changing its gain/entropy. Do not promote the synthetic antisymmetric injection to a structural repair. |
| Interface ownership decision | Does the public `FutureSupervision` type enforce the same horizon/offset contract as formal preflight? | `interfaces.py:214-268` accepts any support offsets with 48+ action rows; `teacher.py:275-287` silently falls back to the nearest support when an interval has no rows, while `intent.py:507-508` clips action/state to the shorter sequence. `training/engine.py:122-126` rejects noncanonical `4,8,...,48` only later. The focused boundary probe accepts sparse offsets and length 49 at the type layer, although formal preflight rejects the sparse case and the recognizer ignores row 49. | Decide whether this is intentionally a preflight-only contract or tighten the public validator. Until then, keep it as interface debt; do not infer a learned-model defect from the probe. |
| Unproven behavior mechanism | Does the bottom typed-P2 router need route selection? | Five route masses remain approximately uniform while the routed value update is nonzero. | An action-level intervention must show that equal mixture, rather than the values themselves, limits behavior. |

LIBERO arm reach/feedback and ManiSkill binary-base behavior retain their
[outlet-specific evidence](#outlet-specific-evidence); neither is silently
closed or scheduled by this CALVIN-focused order. B-spine remains in the
[optional experiment lane](#optional-experiment-lane).

Outer action/world closure is a regression guard, not a current repair queue.
The S common/residual VJP pattern is explained by the accepted lossless
coordinates; independent trainable owners would be a new design decision.
Both are kept outside the active ranking, alongside the source closures above.

<a id="p0-details"></a>

## Outlet-specific evidence

### Corrected RDT gripper contract

RDT action gripper is a command with a different scale and lag from qpos.
Previous executed command owns its continuous codec boundary, including row
zero, the full horizon anchor, cumulative decode, loss and evaluation. Pen uses
the current action-state boundary. Partial row-zero repair is invalid.

The 0973f192 recovery source materialized the RDT previous-command input but
used current qpos when computing loss/evaluation event boundaries. Its event
masks, balanced weights, target counts and F1 are therefore not comparable.
The current source correction cannot repair that trained artifact.

Recovery also logged 53 finite gradient-threshold crossings, 44 in epoch 1,
with maximum 262.63 and most owned by the target-DINO key projection. Loss
ledgers remain exact and E1-to-E2 validation improves. Treat this as numerical
health telemetry unless it becomes reproducibly coupled to optimizer damage,
non-recovery or validation regression.

### Pen far horizon and continuous gripper

The complete recovery curve improves full physical RMSE by about 4.8% over the
Schema28 anchor and 6.5% over Schema29/cachefix. Its bands 1-4 / 5-12 / 13-24
are 0.02474 / 0.05276 / 0.09258, and decoded precision/recall/F1 is
0.6091 / 0.3043 / 0.4059. This establishes a better Pen baseline, not a solved
task. Do not substitute event-head accuracy for decoded continuous-gripper
behavior or trade a better near band for a worse far band.

The Schema32 aggressive candidate is a controlled comparison only at the
combined-source level: data split, normalizer, seed, batch size and deployment
schedule match its c7eccded baseline, but geometry ingress normalization and
six-channel gripper consensus were enabled together.  At E6 it improves first
RMSE from 0.01369 to 0.01268 and band 1-4 from 0.02567 to 0.02532, while full,
arm, gripper, tail, bands 5-12/13-24, decoded F1 and event recall all worsen.
The consensus changes deployed absolute/delta operands by 0.0181 / 0.0135 RMS
and the gripper trajectory/persistence losses rise from 0.01716 / 0.01096 to
0.01997 / 0.01594.  This rejects the combination as a promotion candidate; it
does not assign the regression to either intervention individually.

### CALVIN action outlet

CALVIN rows are relative TCP commands. The accepted current boundary keeps the
shared value-plus-adjacent-difference 18-D core and isolates CALVIN semantics:

- the sampler scores normalized zero-centered command motion;
- the W adapter maps four centered command rows to cumulative canonical
  displacement plus command delta;
- the binary command head is isolated from the six compatibility-only future
  gripper coordinates.

The duplicated-direct-arm alternative is rejected because it changed shared
codec, decode and objective semantics. Initial equivalence, owner-VJP, smoke
and exact-zero binary-gripper conditioning do not establish learned behavior.
Use the formal curve and closed-loop task result for the decision.

The previous CALVIN data conversion is also rejected independently of the
network. It constrained centers by the 48-frame Teacher horizon but supervised
only the first 24 action rows, so terminal manipulation was systematically
absent from the direct policy objective. Converter-v2 restores centers through
`terminal-24`, uses official absorbing terminal padding only for the residual
Teacher future, excludes synthetic rows from normalization and groups splits by
the official source-trajectory inventory.

The v2 preflight also now rejects annotations whose start lies within the first
24 frames of their source trajectory.  An earlier eligibility predicate checked
only the terminal-side horizon and could admit such a row before failing during
episode materialization.  The rejection is performed before any HDF5 episode is
written, so boundary-short annotations cannot leave a misleading partial
dataset.

### ManiSkill StackCube binary gripper repair (2026-09-15)

The failed StackCube v2 closed loop exposed an interface mismatch, not a missing
binary label: source `action`/`action_state` values were already exactly
`{-1,+1}`, but the run still routed dimension 7 through the continuous six-row
gripper flow.  That let a future-gripper field participate in candidate/W
conditioning and made the final native command an accidental consequence of a
continuous decode.  The observed early-close/no-contact trace is therefore not
evidence that the demonstrations use the wrong sign or scale.

The accepted repair gives repaired ManiSkill v2 its own
`maniskill_binary_command` / `maniskill_7d_binary_v2` selection while retaining
the shared 18-D codec and relative-arm W conversion.  A private two-class head
owns the native command; all six compatibility gripper-field coordinates are
zeroed before every dynamic consumer; finalization and W rebuild explicitly
write `+1` (open) or `-1` (close).  Admission fails closed unless both action
arrays contain only those exact values.  This is an interface closure, not a
learned-behavior claim: existing continuous checkpoints are not relabelled, and
a fresh binary base plus closed-loop video/telemetry remains required.

### CALVIN language-to-object binding probe (2026-09-11)

The matched frozen-checkpoint probe in
`scripts/probe_calvin_object_binding_npz.py` (compact result retained at
`new_logs/current/calvin/object_binding_20260911.json`) held each exported
causal observation/history fixed and changed only the instruction across four
layouts.  For every same-layout language swap, the object tokens and both
object-read values/updates were bit-identical (maximum absolute difference
`0.0`).  The organizer's `interval_object` attention was also bit-identical;
its mean mass over K slots was approximately `[0.719, 0.060–0.074,
0.126–0.129, 0.080–0.095]`, with K-slot 0 receiving the largest share.  The
coarse object's attention output is a zero diagnostic placeholder because that
call does not request attention weights; it is not interpreted as a measured
posterior.

The visual object path is not dead: moving the scene changes object-token RMS
by `0.115–0.144` and changes the object-read values.  In contrast, the public
S interval carrier changes by about `0.285–0.294` for a left/right swap but only
`0.0058–0.0140` for red/pink/blue swaps; typed relevance-mass color swaps are
only `0.0011–0.0028`.  This is direct evidence that language reaches the
direction/public-intent path while the coarse/object read has no language-
conditioned pointer.  It is a structural CALVIN outer-path defect, not a T5
lookup failure, a dead visual representation, or evidence of annotation-label
misalignment.

The companion object-memory intervention reinforces the boundary diagnosis:
zeroing or K-averaging the visual object memory changes the coarse proposal by
`0.2169–0.2377` RMS and the one-node physical velocity by `0.0203–0.0331`, but
reversing all K rows changes both by exactly `0.0`.  The current coarse reader
therefore uses an order-invariant aggregate of object evidence; it does not
preserve an object identity that language can select.  This intervention is
read-only and low-coverage, so it establishes responsibility of the existing
object path, not a claim about population success rate.

The source-entry condition for a generic mainline repair is therefore met.  The
first source unit now uses the existing S interval reader with a
validity-masked, language/history-conditioned query and carries that query into
the coarse K read; no CALVIN-only pointer module, color label or new
`ObjectFactSet` is introduced.  The symmetric visual-route memory used by the
coarse read keeps semantic/appearance/geometry evidence on the same K axis.
Local fixed-scene intervention, K permutation, invalid-slot and owner-VJP
gates pass.  This is interface/causal closure only; it is not a learned
behavior claim until a fresh multi-task run is complete.

### CALVIN color-only follow-up (2026-09-14)

The follow-up closed-loop test of the color-only CALVIN structure was a
behavioral failure: it could not reliably push the addressed block.  This
retires color-only conditioning as a candidate, including the prior
outlet-scoped binding sidecar; increasing color loss or wiring its pointer into
the bottom flow is not an acceptable repair.  `ObjectFactSet` remains the
language-free visual fact producer.  A future binding repair must resolve
language to a validity-masked object role/affordance in S/P2, compile the
result into the existing physical action/consequence carriers, and keep raw
language, color labels and object pointers out of the bottom flow.  The bottom
step-conditioning experiment is a separate numerical change and must not be
used to rescue or explain this rejected semantic design.

### Mainline language binding source unit (2026-09-14)

The repaired path is deliberately integrated into `model/intent.py`: the
existing S goal/history innovations condition the interval object read, and
the resulting public K memory plus its producer-owned validity mask condition
`CoarseActionIntent`.  The three visual routes are combined with a shared
K-symmetric bounded sum for the coarse read, while the content-only interval
read remains the owner of the established typed S relevance decomposition.
Consequently language can alter object attention, typed policy context and the
physical proposal without reopening W or injecting semantics into the bottom
flow.  The bottom receives only the compiled physical consequence as before.

Focused CPU gates in
`tests/test_mainline_structural_contracts.py` cover language intervention,
K-permutation equivariance, invalid/all-invalid masking and nonzero gradients
to the language/object readers.  The next behavior gate is an ABC-D multi-task
CALVIN run with the same data/codec/evaluator identity; a single push task or
the rejected color-only rollout cannot certify general language grounding.

The 2026-09-18 intake of isolated source `635213d` does not supersede this
repair: it still uses `relative_command_direct` and a CALVIN-specific selected
context. A CPU language-swap probe leaves its original S object attention and
coarse object query exactly unchanged, whereas this workspace's repaired
queries respond. At that intake, its V2 run had no completed formal validation;
compared with the deployed V1 source `34ee78e`, it also omits two binding objectives, so lower
total loss is not an architecture gain. Seven emitted `calvin_binding_*`
diagnostics are filtered out by that source's logger. Any future promotion of
that frozen branch would require preserving the accepted codec/S boundaries,
closing this diagnostic gap and comparing matched objectives. This historical
intake is not a pending probe request. Detailed evidence and probes:
`artifacts/verified_integration_audit_20260918/REVIEW.md`.

The follow-up candidate `6a646dc` on `codex/shared-s-pilot-20260918` preserves
the accepted shared codec/S graph by default and adds a `history_only` research
control. Its prepared two-arm 5000-step training plan is no longer the current
priority and remains unstarted. The user requested a structure-first review.
The actual `新主执行者` CALVIN checkpoint is `1e51e28` / E1/11012 / Q5; its model
tree equals intake `22d2380` and already contains the shared S repair. The
missing-query finding for `635213d` cannot be attributed to this checkpoint.

The behavioral question is whether intent/object identity and physical action
order survive the S/W/P2 boundaries usefully. For the `1e51e28` legacy-mode
reference, source plus deterministic counterexamples confirm that direct
typed-S conditions the interval terminal after K/K*C pooling; W's 24-row
proposal is reduced to four clipped means
(rows 4–8, 8–16, 16–24, 24); and geometry value averages camera-plane transport
before the shared 2→H projection. P1/coarse indirect language paths and the
bottom's complete action path remain. These are representation limits, not
proof of the learned failure mechanism. The existing checkpoint's matched
16/256-batch panel gives arm RMSE 0.131928, versus 0.129268 with W dynamics or
the consequence effect neutralized; this supports reviewing utility, not
deleting W. Geometry value neutralization changes deployed arm actions by
0.001057 / 0.001599 / 0.002025 RMSE in the three horizon bands, so it is not a
disconnected lane.

The extended geometry check of that legacy source also finds that W's
per-camera producer conditions a repeated K geometry carrier only on camera
motion prior/support.
For equal valid priors its shared heads therefore predict equal transport and
covariance despite distinct camera coordinates. A nonzero-head fixture confirms
this to FP32 roundoff; it does not measure real-checkpoint prevalence. The final
two-snapshot result below bounds the available checkpoint evidence. Moving
the same shared linear value map before P2 pooling is algebraically equivalent
and cannot fix cancellation.

Design constraints retained from this review: a deployment-only 24-row W input
with the old four-row training coarse head would add a representation mismatch.
The implemented sequence and B1 units below address their respective producer
boundaries; they do not change S/P2 placement or prove behavioral benefit.
Any future typed-S pre-pooling design remains a separate unit, retaining
physical-value/support ownership, zero-effect preservation, K/camera
correspondence and the interval axis. The current post-pooling restriction is
deliberate and tested; revising it needs a replacement contract, not a claim
that a wire was missing. Preserve separate attribution of action,
camera-producer, camera-reader and S changes. The final evidence round below
is closed; these constraints do not request another export or diagnostic run.
Evidence: `artifacts/shared_s_candidate_20260918/STRUCTURAL_REVIEW.md` and
`structural_boundary_probe.json` / `geometry_design_probe.json`.
Execution state belongs in the active handoff.

The 2026-09-18 action-temporal observability probe closes the representation
question for the current Q5 checkpoint.  Across 12 language/task windows from
10 unique source trajectories, 35.822% mean / 35.830% median of paired
expert-minus-proposal arm error energy lies in the exact null space of the
four-mean W operator (range 18.035--58.898%).  Only about 18.4% of that null
energy is in rows 1--3; most is within-window temporal shape/order, so exposing
only the previously omitted first rows would not close it.  Exact same-RMS
null perturbations leave W and the refined action unchanged to numerical
precision, while matched visible perturbations change the refined action by
0.02282 mean horizon RMS.  Artifact:
`new_logs/current/20260918-action-temporal-observability-q5/report.json` with
the bounded interpretation in its sibling `ANALYSIS.md`.

The same probe does **not** localize a second failure to P2 or the bottom.
Replacing the proposal W condition with the full expert action changes W
semantic/transport outputs and the refined action, but mean expert RMSE
changes by -0.00362 (four of eight cases improve and four worsen; mean
correction cosine -0.0779).  This establishes that the current correction is
not reliably aligned with expert action, while leaving W prediction accuracy,
proposal-rebuild distribution shift, P2/bottom consumption and action
multimodality as live alternatives.  The 4.28% action-delta/error-scale ratio
is not an information-efficiency estimate.

The opt-in `sequence_prefix_v1` source unit is now structurally implemented,
but not yet behaviorally promoted.  It uses a distinct 24-row condition,
same-target coarse supervision, outlet-owned relative/absolute charts, a
causal 8/16/24/24 W encoder, action-independent W1 shared common and a separate
fail-closed initialization migration.  Non-degenerate regression verifies
that first-three-row changes and equal-mean reorderings are visible, and that
changing rows 9--24 leaves final interval 0 semantic/transport/covariance
exactly unchanged while changing rows 17--24 leaves intervals 0/1 unchanged.
The legacy mode retains the exact pre-unit state tensor digest and constructor
RNG digest.  These are implementation gates, not evidence of closed-loop gain;
B1, typed-S pooling, P2 and the bottom remain separate factors.

The post-implementation reverse audit found and repaired one real AMP blocker:
the first factory version cast normalizer metadata to the action dtype and then
recomputed its fingerprint, so an otherwise valid BF16/FP16 condition could
disagree with the OutletAdapter's FP32 identity.  The condition now keeps FP32
metadata/fingerprint and casts only the arithmetic view.  Pen and CALVIN
FP32/BF16/FP16 matrices, a real CPU-BF16 policy encode with non-unit
normalization, the full training-graph backward/optimizer ownership check, Q5
one-rebuild/two-pass lifecycle, epoch aggregation/logging, and fail-closed
cache/migration cases pass.  The default legacy constructor remains exactly
identical to the pre-unit snapshot (1,391 state entries plus equal state and RNG
digests).  These results remove implementation blockers but still do not claim
behavioral gain before matched training and autonomous rollout.

Cross-task evidence at the same `1e51e28` E1/11012/Q5 checkpoint now includes
four fixed-observation language panels: recomputed arm-chunk RMSE changes are
0.054747–0.057788 for direction versus 0.000210–0.000414 for color. This
supports weak target-identity use but does not isolate a structural cause.
Do not interpret `flow_jepa_typed_p2_*_route_mass≈0.2` as uniform K-object
selection: those are semantic/appearance/geometry/horizon local role deltas
plus null. Actual object P2 has no null; deployed spatial maximum weight is
0.640663 on average, which itself does not certify correct identity. See
`artifacts/shared_s_candidate_20260918/main_executor_comparison.json`.

The completed target-y v2 pair adds matched behavioral evidence: checkpoint and
snapshot hashes match, initial robot state/history are identical, and an 80.004
mm target shift produces at most 14.113 mm pre-contact TCP y response (0.1764).
Both 120-step runs fail; the near target has 20 contact frames but moves 13.953
mm left and 18.737 mm sideways. This supports weak target-relative routing in
this pair. It does not isolate S/W/P2 or rule out the bottom: high-alignment
snapshots are steps 27/28 with different histories, internal velocity is sampled
at fixed noise in one proposal context, and channel RMS ratios are not causal
utility. Reproducible pair audit: `artifacts/shared_s_candidate_20260918/target_y_pair_audit.json`.

The final two-initial-snapshot evidence round is now closed. Joint assignment
(not the normalized object-to-chart read posterior) gives red-region per-camera
argmax K3/K2 in both scenes; the reliable-entity gate fails and target-K oracle
interventions were skipped. Coarse color masks do not establish instance-level
grounding failure, and combined K2 target-mass numbers are not entity attribution.
Same-K cross-camera W transport differences are only 0.0628% / 0.0778% of W RMS,
despite differing coordinates/priors. P2 proposal cross-camera vector retention
is 0.999750 / 0.988400 (ratio of summed magnitudes), so strong camera cancellation
is not supported in these instances. BF16 source arithmetic reproduces all
9216 exported proposal geometry values exactly. Refined W and interval-terminal
weights remain absent; no additional export or diagnostic round is requested.
Full-lifecycle real geometry-read/effect transplants produce refined arm changes
around 0.0031 / 0.0022-0.00234, versus 0.1601 across the two baseline scenes;
this establishes bounded path sensitivity, not a unique cause or bottom immunity.
Audit: `artifacts/shared_s_candidate_20260918/causal_ladder_audit.json`.

A frozen E1/11012 checkpoint probe now isolates the direct typed-S→P2 edge on
one fixed initial scene (red versus blue instruction, same observation, noise,
W candidate and post-P1 query). The CPU-FP32 Q5 node uses identity-checked
BF16 DINO features; baseline replay and language-free G correspondence are
exact. Swapping only typed-S changes its selected context by RMSE 4.45e-5,
but semantic/geometry spatial posteriors and selected physical values remain
exactly unchanged; temporal posterior changes 7.43e-8, geometry effect
1.01e-9 and physical velocity 1.69e-7. The full-language swap changes that
node's physical velocity 1.29e-4, but this is not an additive contribution
decomposition. The typed-S donor itself differs only 9.46e-5 in common-value
RMSE, so the tiny downstream response cannot be attributed solely to P2's
post-pooling placement. This validates the live spatial-address limitation in
one checkpoint case, not a unique closed-loop failure cause or a predicted
benefit from moving S before pooling. Reproduction and compact report:
`artifacts/shared_s_candidate_20260918/probe_p2_typed_isolation.py` and
`p2_typed_isolation_report.json`.

The first source unit selected in the cross-task review was B1, opt-in W camera
conditioning using declared camera identity and existing current coordinates alongside the
original motion-prior branch. Its accepted contract preserves legacy by default
and zero-initializes the new condition branch for comparable initialization.
The review gates cover producer support,
invalid/NaN isolation, K/C-and-identity permutation, compact-belief/rebuild parity,
explicit migration/ABI and unchanged train/deploy call counts. S, P2 pooling,
coarse action and losses remain separate factors. The owning main-executor task
owns implementation and any separately authorized training; this selection
record is not default promotion or training authorization. Concrete interface
and acceptance criteria are in
`artifacts/shared_s_candidate_20260918/STRUCTURAL_REVIEW.md`.

B1 source review is closed at the implementation boundary. The opt-in branch
is exact-zero/legacy-equivalent at initialization, invalid rows remain finite
zero, K and joint camera-role permutations are covered, and train/deployment
materialization counts remain unchanged. The old-checkpoint path is an explicit
fresh component initialization whose source allow-list is the six B1/entry
files only; independently recomputed `compiler.py` and `transition.py` drift is
rejected before any live parameter changes. Review evidence:
`artifacts/shared_s_candidate_20260918/b1_reverse_review_checks.json` and
`artifacts/shared_s_candidate_20260918/b1_migration_review_checks.json`. This is
structural closure, not evidence of E1/CUDA optimization or closed-loop gain;
K/entity correspondence and W32-48 action coverage remain deferred rather than
silently attributed to B1.

### CALVIN binary versus Pen continuous gripper amplitude audit

The read-only `clearvla/tools/probe_gripper_scale.py` scan found no absolute
normalization defect.  Across the selected train charts, raw gripper RMS was
1.000 for CALVIN and 0.867 for Pen; after each outlet's train-only z-score the
RMS was 1.000 and 1.018 (ratio 0.982).  Adjacent-change RMS was 0.3297 for
CALVIN versus 0.0572 for Pen (ratio 5.77), which is the expected binary
transition jump rather than an affine-scale mistake.

This difference is already isolated by the accepted CALVIN contract: its
continuous six-channel gripper compatibility field is zeroed before dynamic
consumers and contributes no formal action-flow or trajectory loss; the binary
command head owns the CE term.  Therefore no shared normalizer, gripper target,
or network-core change is justified by amplitude alone.  CALVIN validation now
also exposes explicitly named binary-command normalized/source-native RMSE and
error-rate keys; the historical `...command_rmse_physical` key remains only as
an alias.  Compare CALVIN command accuracy/F1/error rate with Pen continuous
trajectory RMSE, not the two same-looking legacy RMSE labels.

Reopen this question only if a joint-outlet run shows a nonzero CALVIN
compatibility-field objective/gradient or a matched owner-gradient imbalance;
raw delta-RMS disparity by itself is not a reason to rescale the core.

### LIBERO retarget overlay and closed-loop alignment

The retained r3 artifact is a **component/data-contract initialization**, not
a fresh-from-scratch run and not an exact resume.  It initializes from the
terminal-suffix E2 best checkpoint, while optimizer, schedule, RNG and loader
state are all disabled.  The overlay is train-only: two source demonstrations
produce four admitted `+/-3 cm` Y-translation episodes, increasing the effective
train episode count from 40 to 44; validation/test membership and the action
and state normalizer SHA-256 values remain unchanged.  The serialized model
and data identities are in `artifacts/libero_retarget_20260910/training_r3/`.

At the completed one-epoch bounded run (120 train batches and four validation
batches), source-native validation RMSE is **0.213688 full / 0.131133 arm /
0.465255 gripper**; first, first-8 and tail RMSE are **0.169031 / 0.217737 /
0.211634**, with tail/first **1.252**.  The loss ledger assigns about **87.6%**
to action, **6.3%** to representation and **6.1%** to execution.  The weighted
gripper-trajectory contribution is **0.00805** (about **2.4%** of total loss),
so this run does not support the claim that gripper loss weight is dragging the
arm.  Decoded gripper events remain badly over-predicted (**425 vs 26**, ratio
**16.35**, precision **0.0612**, recall **1.0**, F1 **0.1153**).

The paired diagnostic closed loop has **9 rollouts and 0 successes**: all
8-step probes and all 32-step baseline/`+/-` target-shift probes fail.  Strong
close commands begin around steps **5--6**, while the three traces that show a
strong open command begin around step **21**; at the end of the long probes the
EEF is still roughly **0.171--0.202 m** from the bowl.  The target/plate show no
material XY scene drift, and the target-shift response has the correct sign in
**5/6** pairs, but the response is too small to establish useful closed-loop
control.  The probe records command timing only; it has no gripper-joint qpos
or contact-force measurement, so “early close” is not yet a physical-grasp
claim.

The bounded timing attribution is now complete at
`artifacts/libero_timing_20260911/timing_probe_v1.json` (videos are in the
adjacent `timing_probe_v1_videos/` directory).  One task-0/init-0 pair ran for
32 steps from one post-warmup snapshot.  The delayed branch replaced the first
12 positive close commands with zero while copying baseline arm commands
exactly.  Arm action delta max/RMS were **0/0**; gripper command delta RMS was
**0.44066** and 12 commands were suppressed.  Neither branch made a
gripper--bowl or bowl--plate contact, both remained unsuccessful, and bowl
displacement was only **5.02e-5 m**.  The post-step EEF trajectory difference
was **9.10e-6 m RMS**, with final EEF-to-bowl distance about **0.185 m** in both
branches.  This is strong negative evidence against “early close drags the arm”
for this state, but it does not establish behavior on a trajectory that reaches
the object.  Keep the next source-entry condition on arm reach and observation
feedback; do not increase gripper weight, add a fixed gain, alter the
seven-dimensional codec or start a longer formal run before that attribution.

<a id="p1-details"></a>

## Cross-cutting evidence

### Identity and continuation

The future identity surface must distinguish provenance from compatibility:
raw episode inventory, split, profile metadata, normalizers, caches, language,
component/deployment ABI, optimizer/scheduler state and sampler/named-generator
continuation state need separately named records. A saved field that is neither
validated nor restored is provenance, not resume state.

### S/W chronology and responsibility

Recovery returns Pen W and outer closure close to Schema28 scale, so a
disconnected W path is not demonstrated. RDT still presents the harder
question: state/history can identify tasks outside language, while online S is
asked to match a 48-step instance-specific recognizer and W receives a compact
goal-free physical condition. Existing evidence supports an information and
responsibility mismatch, not a specific repair.

The next comparable RDT checkpoint should use bounded goal zero/shuffle,
history shuffle and oracle-versus-online physical-condition probes. Do not add
goal directly to W, force typed amplitude or start a diagnostic training run
without those attribution results.

### Typed common/residual reverse responsibility

The current source deliberately keeps the Schema25 typed relevance value
lossless across the S-to-P2 dock:

```text
common   = source.mean(dim=interval)
residual = source - common
route    = common + residual
```

The source anchors are `clearvla/mainline/model/intent.py:40-47` and
`clearvla/mainline/model/compiler.py:600-608`. A deterministic local fixture
confirmed zero reconstruction error. With the live route, a downstream P2
loss produced an exactly-zero VJP at `typed_common_value` and a nonzero VJP at
`typed_interval_residual_value`; replacing the residual with a detached copy
preserved the same forward route and moved the VJP to common. The P2 typed-key
owner VJP and all forward outputs were unchanged by those ownership-only
controls.

This is not evidence that typed S is disconnected. It means the two stored
fields are coordinates of one source tensor, not independent learned lanes;
the residual's `SubBackward0` owns the local reverse path. R1C explicitly
defined S-02 as a lossless decomposition, so changing this algebra would be a
new semantic unit. Do not repair it with a hand-set amplitude, stop-gradient,
route quota or entropy target. If independent responsibility is later
required, first specify the producer/owner contract and pair it with a matched
deployed-action intervention.

### Geometry and bottom routing

Both paths are connected: geometry values and the bottom routed update can
change action. Their current metrics do not prove beneficial address
specialization. Small amplitude or near-uniform route mass alone does not
authorize deletion, gain, entropy pressure, quotas or forced specialization.

## Optional experiment lane

B-spine is an opt-in execution-bottom experiment, not the default architecture
and not a replacement action codec. Its current question is whether a fixed
coarse/detail view beside the unchanged raw lift adds Pen action utility
without suppressing first-step, far-horizon or gripper-event detail. Each
selected component's serialized run context owns its experiment identity.
The first coarse/detail run's early observations are retained in
[auxiliary/SCHEMA31_BSPINE_EARLY_AUDIT.md](auxiliary/SCHEMA31_BSPINE_EARLY_AUDIT.md)
as historical evidence; the repair plan owns future disposition. Do not infer
another candidate's status from that snapshot or the Schema31 label.

The selected `fixed_bspline_arm_coarse_context_v1` investigation intentionally
keeps the downstream arm/gripper route jointly trainable; it does not use a
stop-gradient bridge.  Diagnostic batches record the direct arm-flow and
gripper-flow pulls on the B-spine owner, their cosine/projection, and the
weighted action/representation/execution group pulls.  This is the evidence
for what training actually learns.  The evaluation-only `spine_zero` result
remains a dependency probe and cannot substitute for these training-time
gradients.

The new `fixed_bspline_arm_private_reader_v1` is a matched routing control for
that question. It uses the identical cubic arm coarse chart, while an explicit
`ArmPrivateBSpineReader` sends its output only to the first twelve physical
arm-velocity channels at the terminal controller. The raw lift and shared
MMDiT path remain unchanged; gripper physical channels, gripper state/command
and motion heads do not consume the correction. Its zero-initialized coarse
lifts preserve the baseline at construction, and its reader parameters are
owned by the same opt-in `bottom_spine` optimizer group. This branch is
implemented and ready for a matched Pen run, but has no learned-behavior claim
until the full curve, arm/gripper horizon bands, and matched-noise
`spine_zero` attribution are available. Read
`gradient_probe_bottom_spine_*` and
`gradient_probe_bottom_spine_private_reader_*` separately; the aggregate owner
gradient alone is insufficient to decide whether private routing helps.

## Closed facts

- W is the only future-world producer; ControlledTransition is a consumer.
- Semantic P2 has demonstrated far/gripper responsibility.
- The CUDA autocast-cache VJP break is closed by scoped cache isolation and a
  real parameter-owner gate.
- The default training path has one formal velocity/loss pass; deployment has
  one proposal ODE, one W rebuild and one refined ODE.
- Outer action/world closure remains a regression guard: recovery E8
  proposal/refined/final-interval mismatch was 0.02863 / 0.03212 / 0.01556,
  close to Schema28. Reopen only on contradictory comparable evidence.
- Capacity near full width is not a release failure or a hardware-rank claim.

Reopen a closed fact only with contradictory source or matched causal evidence.

### Producer-owned validity at S/W object reads — repaired

The source trace confirmed a local support-boundary omission: S/coarse object
reads and W generic/typed object-axis self-attention could reduce over invalid
K rows. The accepted repair keeps the fixed K axis and the producer-owned
authority. `StatelessObjectIntentOrganizer` now materializes a detached
`[B,K]` support mask beside its public memory and passes it to the interval
read; `CoarseActionIntent` carries the same mask through `ActionIntentDock` to
its object read. `_ObjectIntervalBlock` accepts the mask for both generic and
typed paths, zeroes invalid rows before normalization, masks them in the MHA
denominator, and zeroes them after the residual/FFN. An all-invalid row
unmasks one zero key only to avoid PyTorch MHA NaNs and returns an exact finite
zero block. The default no-mask call remains the inherited operator for old
structural fixtures.

The focused support gate in
`tests/test_mainline_structural_contracts.py::test_w_object_axis_support_isolated_forward_and_backward`
shows, with one invalid K, invariant valid outputs under a large invalid-value
intervention, exactly zero invalid-row VJP, and finite zero output for the
all-invalid generic and typed cases. The existing S/coarse invalid-slot test,
K permutation tests, full structural suite (90 tests), mainline top/config/
manifest suite (115 tests), policy suite (49 tests), and checkpoint/runtime
suite (41 tests) pass after the repair. This is a boundary guarantee, not a
claim about real-data invalid-slot prevalence; the current Pen logs still show
`object_grounding_validity_mean` near 1.0.

The repair does not add a learned gate, route quota, entropy target, sidecar,
or extra W pass. It is distinct from the lossless S common/residual coordinate
issue and from the independent flow-step schedule.

## Stop and closure rules

Hard-stop for non-finite values, identity/lineage failure, an open loss ledger,
vanished formal parameter-owner VJP, checkpoint ABI violation or process memory
above 22 GiB. Finite threshold crossings, early event F1, small geometry RMS
and warm-up capacity do not stop a run by themselves.

Close an implementation defect when its named boundary and applicable
forward, reverse-owner, identity or lifecycle checks pass at stated coverage.
Record any untested runtime boundary explicitly. Closing that defect makes no
claim of learned benefit. Close a behavior question only with the relevant
matched intervention or task evaluation; a source explanation alone cannot
close it. An unusual metric without attributed responsibility does not select
a structural edit or require another diagnostic round.
