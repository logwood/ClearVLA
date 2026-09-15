# R1/R2 closure index

Status: compact replacement index for the ten completed R1/R2 worksheets.
The active architecture and release truth remain in
`../00_CURRENT_ARCHITECTURE_CONTRACT.md`; this file is historical replay
memory, not an implementation authority.

The detailed worksheets were complete source-boundary records, but they
repeated the same forward/reverse audit, test gates and inventory tables. This
index keeps the decision-bearing result of each unit in one place. The exact
pre-edit narratives, full fingerprints and red-state transcripts remain
recoverable from Git commit `f60bd80`:

```powershell
git show f60bd80:docs/research/auxiliary/R1A_G01_G3_HANDOFF_WORKSHEET.md
```

Replace the path in that command for any unit listed below. Do not infer
current behavior from the old source snapshots; cross-check the current
contract, source and run context first.

## Reading contract

- `Status` describes the historical unit, not the current checkout's release
  status.
- `Boundary` is the producer/consumer ownership decision that survived review.
- `Evidence` records only decision-facing tests, inventories and validation
  facts; it is not a substitute for the original audit.
- `Rejected` lists mechanics explicitly ruled out so that a later reader does
  not mistake a donor idea for an adopted repair.
- Every R1 source slice was static/test-closed and explicitly did **not** run
  training, dataset, CUDA or checkpoint commands. R2-A01 was validation-only on
  an existing R1 checkpoint and changed no training graph.

## Closure matrix

| Unit | Boundary | Status | Evidence | Historical source |
|---|---|---|---|---|
| R1A / G-01 | one completed G3 rollout handoff | closed | focused 2/2; retained 123/123; -2,048 params | `policy.py`, `transition.py` |
| R1B / G-02 | conditional-K reconstruction ownership | closed | focused 6/6; retained 129/129; inventory unchanged | `grounding.py` |
| R1C / S-01,S-02 | one typed W ingress plus lossless common/residual coordinates | closed | focused 5/5; retained 134/134; inventory unchanged | `intent.py`, `types.py`, `dynamics.py` |
| R1D / W-01,W-02 | one-way common/near/far W ownership and camera ABI | closed | focused 10/10; retained 140/140; -3,075 params | `types.py`, `dynamics.py`, `teacher.py` |
| R1E / P1-01 | static factual read separated from dynamic P1 residual | closed | focused 3/3; retained 140/140; inventory unchanged | `policy.py`, `top.py`, `compiler.py` |
| R1F / P2-01 | independent spatial selection and physical interval terminal | closed | focused 5/5; retained 144/144; +32,768 params | `compiler.py`, `types.py`, `top.py` |
| R1G / P3-01,B-01 | unique private P3 lanes and protected bottom ingress | closed | retained 145/145; -1,573,376 params | `compiler.py`, `restored_bottom.py`, `top.py` |
| LC-01 | delete exact-zero trajectory aliases | closed | retained 145/145; -23,590 frozen params; -16 state keys | `policy.py`, `restored_bottom.py`, `layer_contracts.py` |
| R1H / N-01,D-01 | finite zero-preserving numerics and read-only diagnostics | closed | retained 155/155; R1 inventory/RNG exact | observation, grounding, routing, logging |
| R2-A01 | matched P2 value and gripper attribution | validation-only closed | 94 focused; 164 retained; 179 validation batches | `evaluation.py`, `train.py`, `logging.py` |

## Retired worksheet names

The following paths are intentionally no longer present in the working tree;
the unit-to-anchor map keeps old notes and issue references searchable:

| Retired filename | Index anchor |
|---|---|
| `R1A_G01_G3_HANDOFF_WORKSHEET.md` | `#r1a-g01` |
| `R1B_G02_CONDITIONAL_K_RECONSTRUCTION_WORKSHEET.md` | `#r1b-g02` |
| `R1C_S01_S02_TYPED_INGRESS_DECOMPOSITION_WORKSHEET.md` | `#r1c-s01-s02` |
| `R1D_W01_W02_CAUSAL_FIELD_ABI_WORKSHEET.md` | `#r1d-w01-w02` |
| `R1E_P101_STATIC_DYNAMIC_P1_WORKSHEET.md` | `#r1e-p1-01` |
| `R1F_P201_SPATIAL_PHYSICAL_TERMINAL_WORKSHEET.md` | `#r1f-p2-01` |
| `R1G_P301_B01_UNIQUE_LANE_BOTTOM_INGRESS_WORKSHEET.md` | `#r1g-p3-01-b-01` |
| `LC01_EXACT_ZERO_LAYER_TRAJECTORY_CLEANUP_WORKSHEET.md` | `#lc-01` |
| `R1H_N01_D01_FINITE_NUMERICS_DIAGNOSTICS_WORKSHEET.md` | `#r1h-n01-d01` |
| `R2_A01_MATCHED_P2_VALUE_GRIPPER_ATTRIBUTION_WORKSHEET.md` | `#r2-a01` |

## R1 source slices

<a id="r1a-g01"></a>
### R1A / G-01 — exact G3 handoff

Boundary: bind the final completed G3 rollout once and pass the same tensor
object to P1 and controlled-transition source construction. The carrier is
`[B, 4*C*8*8, H]` (512 rows under the locked manifest).

Accepted result:

- `policy.py` names one `g3_rollout`; `transition.py` stores it directly.
- The old `interval_identity` reconstruction and `ObjectFactSet` transition
  API are removed; no cast, clone, detach, projection or normalization is
  inserted.
- Exact-zero and sentinel-row identity survive, and a one-row cotangent maps
  back to the same G3 view.
- The only inventory change is removal of the `interval_identity` field:
  total/trainable parameters fall by 2,048 and one optimizer/state tensor is
  removed.

Evidence: focused G-01 tests 2/2, retained suite 123/123, compileall/Ruff/
Pyright pass. No training run. No unresolved assumption remains.

<a id="r1b-g02"></a>
### R1B / G-02 — conditional-K reconstruction

Boundary: keep the detached current-DINO chart as the reconstruction target,
let the physical binder preserve real/null mass, and change only `P(K | real)`.
The existing content residual is exported once as `ObjectFactSet.content`.

Accepted result:

- `_conditional_k_reconstruction_assignment` applies the retained local prior
  and observable validity outside the conditional-K softmax in FP32.
- Reconstruction uses the one exported content and a K-independent shared
  coordinate term, reduced only over observed cells.
- S, W and detached Teacher all read that same content; no second K-specific
  value, `decoded_slot`, or target gradient path remains.
- Parameters, optimizer groups, state keys and checkpoint fields are exactly
  unchanged from R1A.

Evidence: focused mechanisms 6/6, retained suite 129/129, compileall/Ruff/
Pyright pass. No training run. No unresolved assumption remains.

<a id="r1c-s01-s02"></a>
### R1C / S-01,S-02 — typed ingress decomposition

Boundary: retain the unchanged Schema25 typed-relevance operator, remove the
typed CoarseAction re-entry, and store the value as one interval common plus
four signed zero-sum residual rows:

```text
common = source.mean(dim=interval)
residual_i = source_i - common
source_i = common + residual_i
```

Accepted result:

- `ActionIntentDock` and CoarseAction are typed-free; `WorldIntentDock` is the
  sole typed W ingress.
- W reconstructs the former source exactly once at its named boundary, with
  ordinary autograd and exact-zero/constant residual semantics.
- K/type/interval axes and permutation behavior are retained; no new selector,
  gain, floor, loss, parameter, buffer or future-owner supervisor is added.
- Inventory is unchanged from R1B.

Evidence: focused mechanisms 5/5, retained suite 134/134, compileall/Ruff/
Pyright pass. No training run. No unresolved assumption remains.

<a id="r1d-w01-w02"></a>
### R1D / W-01,W-02 — causal field and camera ABI

Boundary: W1 owns protected common plus the two near innovations; W2 reads
completed W1 state and owns only the two far innovations. Appearance conditions
semantic state; geometry retains per-camera transport/covariance and no
predicted-status authority.

Accepted result:

- Common/innovation processing is one-way and zero-preserving via the existing
  relation `x + x * tanh(c)`; W2 cannot rewrite common or near state.
- Camera identity survives through Teacher targets, per-camera losses and P2;
  covariance remains FP32 PSD and C is reduced only after scoring.
- Status fields/heads and their loss path are removed; no alternate W decoder,
  unbounded scale competition or axis reconstruction remains.
- Inventory changes by -3,075 parameters, -7 parameter/state tensors, with 23
  optimizer groups and owner assignments unchanged.

Evidence: focused mechanisms 10/10, retained suite 140/140, exact inventory and
RNG sentinel checks pass. No training run. No unresolved assumption remains;
the retained flattened P2 terminal is explicitly R1F debt.

<a id="r1e-p1-01"></a>
### R1E / P1-01 — static/dynamic P1 ownership

Boundary: keep the cached observation-owned factual read `[B,24,Q,H]`
separate from the noisy-action/time-dependent dynamic residual.

Accepted result:

- `complete_p1_fact()` returns a typed static/dynamic state; the P2 query is
  `action_query + factual_base + policy_query_residual` exactly once.
- Protected consequence uses factual base plus P2 effect, never the dynamic
  residual. The same raw residual reaches the controlled transition and bottom
  once at their named consumers.
- Zero residual stays exact zero; no projection, gain, floor, null, scale,
  detach, clone, extra consumer or public ABI shortcut is introduced.
- Parameters, optimizer groups, state keys and RNG are identical to R1D.

Evidence: focused mechanisms 3/3, retained suite 140/140, compileall/Ruff/
Pyright and temporary R1D comparison pass. No training run. No unresolved
assumption remains; P2 terminal/P3 aliases are deferred to R1F/R1G.

<a id="r1f-p2-01"></a>
### R1F / P2-01 — spatial selection and physical terminal

Boundary: semantic selects K independently per interval; geometry selects K*C
independently per interval; the same W-owned posterior selects existing S
metadata, then each type removes the four physical intervals through a
no-null terminal before the unchanged caller contract.

Accepted result:

- Covariance and transport are scored together before camera C disappears;
  `SelectedIntervalEvidence` retains `[B,T,Q,I,2,H]` and support identity.
- Semantic and geometry are complementary owners. There is no flattened
  `[I*K]+null` competition, type softmax, type gain or S-owned support/value.
- With no observable support output is exact zero and gradients remain finite;
  arbitrary S/action cannot create an effect when W values are zero.
- Two existing projections add 32,768 parameters and two tensors; no new loss,
  buffer, quota or bottom change is made.

Evidence: focused mechanisms 5/5, retained suite 144/144, compileall/Ruff/
Pyright pass. No training run. No unresolved assumption remains.

<a id="r1g-p3-01-b-01"></a>
### R1G / P3-01,B-01 — unique lanes and bottom ingress

Boundary: retain exactly four meaningful values: protected consequence, raw
dynamic P1 precision, optional temporal innovation and optional observable
state-change innovation. The first two are protected direct reads; only the
last two own learned-null decisions.

Accepted result:

- Optional factual/static-precision/effect aliases are deleted; temporal and
  state-change retain separate private Q+null routing and zero semantics.
- Bottom reads two disjoint Q+null sources whose raw outputs add under the
  inherited lane contracts. The serialized optional source changes from 5*Q
  rows to Q rows; six H-to-H projections and 512 key parameters are removed.
- No cross-lane simplex, fixed attenuation, extra aggregate RMS, alternate
  bottom write or new event/motion loss is introduced.

Evidence: retained suite 145/145, inventory and RNG sentinels pass; no training
run. No unresolved assumption remains.

<a id="lc-01"></a>
### LC-01 — exact-zero trajectory cleanup

Boundary: deletion-only cleanup after R1G. Remove two source-proven exact-zero
trajectory aliases from terminal layer contracts while retaining every live
rollout/state/history/event and downstream decoder consumer.

Accepted result:

- Both depth-5/depth-6 adapters read the completed rollout/state canvas; the
  Evidence view still receives rollout, state, history and terminal event rows.
- Injecting arbitrary trajectory rows leaves retained contract results
  bit-identical; all 12 trainable adapter tensors retain finite nonzero VJPs.
- Total parameters fall by 23,590 frozen parameters and 16 state keys;
  trainable/optimizer tensors, groups and RNG remain exact.
- No replacement carrier, scale, gate, loss, diagnostic proxy or training run.

Evidence: retained suite 145/145, exact weight/decoder/RNG digests, compileall,
Ruff and Pyright changed-line gate pass. No unresolved assumption remains.

<a id="r1h-n01-d01"></a>
### R1H / N-01,D-01 — finite numerics and diagnostics

Boundary: preserve the R1G/LC-01 graph while making active address
variance-to-standard-deviation paths zero-preserving, retaining producer-owned
FP32 probability/log measures and adding read-only source-tensor diagnostics.

Accepted result:

- G2/G3 address paths keep exact zero/constant behavior and bounded local
  normalization; FP32 measures/logs travel through G to W/P2 without changing
  owners or support semantics.
- Forward diagnostics name live S/P1/W2/P2/P3 tensors; reverse diagnostics
  attribute finite raw-gradient spikes without adding a loss edge or update.
- Manifest/runtime ABI changes while parameter, optimizer, state-key and RNG
  inventories remain LC-01 exact.

Evidence: focused N-01/D-01 checks, retained suite 155/155, forward/reverse
review, exact inventory/RNG, compileall/Ruff/Pyright and diff checks pass. No
training run. No unresolved assumption remains.

## R2 validation slice

<a id="r2-a01"></a>
### R2-A01 — matched P2 value and gripper attribution

Boundary: validation-only intervention on an existing R1 checkpoint. Four
plain evaluation modes may zero only selected `[interval, type]` value cells
after interval posterior completion; primary validation decodes both deployed
gripper branches from the already returned physical field and partitions
event/post-event rows without another model forward.

Guardrails:

- Reuse the primary encoded cache and initial physical noise for each complete
  counterfactual; clear the mode in `finally`.
- No parameter, buffer, optimizer owner, checkpoint field, training loss,
  sampler draw, P2 caller contract or deployment pass changes.
- Console output is bounded; lossless metrics remain in JSONL.

Validation evidence: 94 focused runtime/structural tests, 164 retained tests,
179 validation batches / 1,432 samples, and 16 matched diagnostic batches.
Semantic intervals 2/3 are strongly required for far action; geometry is nearly
action-inert at learned R1 scale but must not be deleted; both absolute and
cumulative-delta gripper branches are already weak before their fixed blend.

This closes the attribution slice while leaving three distinct parent-plan
problems: weak W geometry supervision/output use, overlapping spatial-versus-
temporal P2 query ownership, and missing gripper-private closure between event
semantics and deployed continuous value/delta heads. It rejects a hard far-
interval schedule and a first codec-blend edit.

## Cross-unit invariants

These invariants were checked repeatedly across the worksheets and are the
reason the detailed records can be indexed together:

1. Producer ownership is reviewed forward and backward: every retained value
   has a named consumer, loss route and optimizer owner; no hidden duplicate
   path is accepted.
2. Exact-zero and zero-preserving relations are semantic contracts, not
   numerical patches. A zero owner cannot be revived by a free conditioner.
3. Physical validity/support stays producer-owned. Null mass is explicit where
   the source contract requires it; a null is never used as disappearance of a
   valid axis.
4. Camera identity is retained until the named geometry consumer. C reduction
   happens only after covariance/transport scoring and producer-owned validity.
5. No unit authorizes a gain, quota, entropy target, hard event gate, artificial
   gradient, extra clip, loss-weight bundle or whole-version rollback.
6. Schema/source/checkpoint identity is strict. A static closure does not make
   an older checkpoint an exact-resume source.
7. Historical donor code is evidence only. Current implementation must be
   rechecked against the active contract, source, serialized run context and
   supplied logs.

## Retrieval map

Use the compact current documents first:

- [`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md)
  — active graph and release truth;
- [`../CURRENT_MAINLINE_ISSUES.md`](../CURRENT_MAINLINE_ISSUES.md) — still-open
  active problems;
- [`../CURRENT_MAINLINE_REPAIR_PLAN.md`](../CURRENT_MAINLINE_REPAIR_PLAN.md) —
  current run/repair order;
- [`SCHEMA25_R1_IMPLEMENTATION_PROTOCOL.md`](../archive/replay/SCHEMA25_R1_IMPLEMENTATION_PROTOCOL.md)
  — adopted replay order and gates;
- [`ARCHITECTURE_REPLAY_SOURCE_UNITS.md`](../archive/replay/ARCHITECTURE_REPLAY_SOURCE_UNITS.md)
  — source boundaries and dispositions;
- [`ARCHITECTURE_REPLAY_PLAN.md`](../archive/replay/ARCHITECTURE_REPLAY_PLAN.md)
  — replay bundle and retained tests.

For a full historical worksheet, run:

```powershell
git show f60bd80:docs/research/auxiliary/<worksheet-name>.md
```

The compressed index intentionally omits raw logs, checkpoints, tensor caches,
full probe dumps and generated binaries. Reproduce those from the recorded
source/run paths when a new decision requires them.

## Maintenance rule

When a historical unit needs a correction, update this index and record the
exact source/log anchor. Do not recreate one new versioned worksheet per
conversation. If a future implementation decision changes active semantics,
update `../00_CURRENT_ARCHITECTURE_CONTRACT.md` in place and leave the
historical result here as provenance.
