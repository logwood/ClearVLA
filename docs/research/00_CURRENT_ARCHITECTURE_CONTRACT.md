# Current ClearVLA Architecture Contract

Status: source of truth for the V111 structured evidence-ownership redesign.
The source, launcher, serialized contract and CPU G/W/P autograd smoke are
aligned below; CUDA runtime acceptance remains pending. V110 is the exact
flags-off structural parent, while V107 supplies the finite-gain numerical
boundary inherited by the complete path.
Updated: 2026-07-28
Format: short current contract first, historical evidence below. Update this
file in place.

## 0. Agent quick contract

### 0.1 Reading rule

For an ordinary V111 implementation or review, read this section and Section 2
first, then verify the named source. Do not reconstruct the current graph from
experiment names or from the historical ledger in Section 3.

Read Section 3 only when diagnosing ancestry, explaining an old log, or
checking why a previous repair existed. Sections 4-10 are supporting design,
acceptance, and data evidence rather than a second source of current behavior.

If source, serialized run context, and this document disagree, source plus the
serialized run context describe what actually ran. Update this quick contract
before making the next architectural change.

### 0.2 Current identity

```text
target version:       V111
training topology:    one-stage end-to-end
top blocks:           G1 G2 G3 / W1 W2 W3 / P1 P2
launcher:             scripts/current_v111_structured_ownership_bottleneck.sh
smoke launcher:       scripts/current_v111_structured_ownership_bottleneck_smoke.sh
required contract:    v111
parent typed flag:    flow_jepa_coordinate_typed_raw_detail=1
new topology flag:    flow_jepa_structured_ownership_bottleneck=1
structural parent:    V110 coordinate-typed G/W/P path
numerical parent:     V107 complete finite-gain contract
flags-off behavior:   new flag 0 restores the V110 graph exactly
runtime status:       Python compile, CPU BF16 G1/G2/G3->W->P forward/backward
                      and exact-zero P2 value test pass; CUDA smoke pending
```

Do not call the V111 graph “validated” merely because it compiles or starts.
Validation requires the smoke, gradients, and causal probe in Section 0.8.

### 0.3 Current forward graph

```text
current observations only
  -> Pre-G observation bank
       exact current camera/coordinate/scale/time identity
       native-resolution literal bounded RGB and lower-resolution learned
       raw-detail charts kept separate; literal RGB is not pre-pooled
       DINO semantics, raw-pair correspondence and flow geometry kept separate
       no noisy action, policy query, or future teacher
  -> G1 semantic complete-chart multi-slot hypotheses
       DINO plus exact identity establishes observed current anchors
  -> G2 typed geometric rectification
       appearance verifies local correspondence; geometry owns centre/support
       semantic retains its own hypothesis posterior but cannot seize fine xy
       stable bounded coordinate update rematerializes current candidates
       camera x source-cell x slot x fine-candidate x type axes survive
  -> G3 typed canonical selector state
       semantic, appearance and geometry keys remain separately addressable
       emits one uniform-slot low-frequency camera-spatial public chart to W
       retains owner-specific slot posteriors and route-width keys in side state
       no high-resolution value read
  -> W1-W3 horizon/goal/phase/history-conditioned world state
       decomposes projected W queries into public/common plus exact innovations
       semantic+geometry own source selection; appearance stays a private verifier
       predicts bounded chronological interval transport from current anchors
       bounded transport compatibility directly changes JEPA relevance and
       the paired W->P source prior; it never creates a fictional raw crop
  -> P1 / W->P ingress A: first and only high-resolution read
       factorizes semantic+geometry source selection from
       appearance+geometry+transport fine selection
       exact current coordinates read a structured 3x3 literal-RGB micro-patch
       learned raw detail remains a separate companion lane
       W transport supplies a bounded soft fine-candidate likelihood
  -> P2 / W->P ingress B: owner-conditioned local patch operations
       RGB and learned detail remain separate value-token lanes
       policy/semantic/appearance/geometry/horizon perform separate reads
       fuses only their action-ready contributions
  -> existing P1-P2 policy DiT blocks
       consume the typed ingress update plus clean policy intent and write the
       trajectory; they do not reopen the observation bank
  -> existing bottom Evidence MMDiT/CVAE/workspace/adaptive-selection/execution
  -> deployed action

future DINO teachers -> losses only; never enter the online graph
```

### 0.4 Ownership table

| Boundary | Owns | Must not own |
| --- | --- | --- |
| Pre-G | deterministic current geometry; separately typed RGB, DINO, raw-pair and flow charts | learned final route, action, future teacher, mixed evidence soup |
| G1 | same-camera semantic complete-chart hypotheses per source cell/slot | raw-value aggregation, final geometry, camera fusion |
| G2 | geometry rectification/support, appearance verification and separately retained owner posteriors | one all-owner fine posterior as the only record, future pixels, value aggregation |
| G3 | one public spatial chart plus owner-specific slot posteriors/keys/summaries | private selector decisions in generic public memory, high-resolution value read |
| W1-W3 | public/common state, horizon innovations, semantic+geometry source relevance and chronological transport | cached values, future pixels, appearance-owned fine decisions |
| W->P ingress A / P1 | factorized source/fine decisions and 4-glimpse 3x3 RGB/detail read | all-owner logit soup, early local averaging |
| W->P ingress B / P2 | separate RGB/detail lanes and typed owner reads before final fusion | condition summation, a second bank or untyped global soup |
| Policy blocks P1/P2 | trajectory organization from the typed ingress and clean intent | reopening raw/DINO observation values |
| Bottom | action decoding and execution machinery | reinterpretation of teacher data as online evidence |

The order is dependency driven and must not be casually reversed: G2 needs G1
hypotheses to rectify; G3 needs the rectified state to canonicalize; W needs the
canonical basis to form horizon selection; P needs that selection to read
values.

### 0.5 Main tensor/state contract

```text
observation bank:
  horizon independent
  camera identity retained
  exact normalized current coordinates retained
  native-resolution literal RGB, learned raw detail, DINO, raw-pair and
  geometry charts typed

progressive address state:
  G1: coarse logits/probability, centres, variance, aligned keys
  G2: current centres/support, typed keys, RGB/detail values, coordinates,
      validity and semantic/appearance/geometry probabilities
  G3: semantic/appearance/geometry slot keys and owner slot probabilities;
      one public spatial chart plus typed owner side state
  W:  teacher relevance    [B,A,C,G,G]
      current-source prior [B,A,C,G,G,M]
      public query + horizon innovation (exactly reconstructs W query)
      owner source priors retained separately
      chronological interval transport deltas
      future centre/offset/scale/visibility/uncertainty [B,A,C,G,G,M,*]

P input candidates:
  typed keys     [B,C,G,G,M,K,key_dim]
  RGB values     [B,C,G,G,M,K,3]
  learned detail [B,C,G,G,M,K,raw_value_dim]
  current xy     [B,C,G,G,M,K,2]

W->P ingress A performs the first query-dependent selection over C/G/G/M/K but
retains a 3x3 micro-patch axis. Ingress B compresses local structure into one
fixed-width trajectory update. The actual policy blocks P1/P2 then organize
that update; they cannot read the observation bank again.
```

An `8x8` cell is a query/source lattice cell, not a hard pointer to one image
patch. Slots and fine candidates remain soft and differentiable.

### 0.6 Forbidden shortcuts

- no future-teacher tensor in the observation bank, G/W queries, or action path;
- no noisy-action -> observation-bank -> action echo;
- no detach followed by a gradient patch;
- no hard top-k, argmax, straight-through selector, or non-differentiable crop;
- no fixed entropy target, forced route mass, or artificial nonzero-flow quota;
- no early mean over cameras, slots, fine candidates, or spatial W state;
- no pooled raw-detail duplicate that bypasses the P read;
- no fusion that irreversibly erases RGB/DINO/raw-pair/geometry type identity
  before P2;
- no future offset used to sample nonexistent future RGB;
- no immediate mean of the P1 fine-candidate lattice into one vector;
- no log-only posterior presented as a connected action mechanism;
- no optimizer-owned trainable module absent from the active loss graph.

### 0.7 Authoritative source map

```text
config and validation:
  clearvla/policy/config.py
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py

observation bank, G1/G2/G3 and W posterior:
  clearvla/policy/flow_dino_evidence.py
    SoftAddressLatticeBank
    ProgressiveGroundingAddressState
    ProgressiveFineCandidates
    _SoftMultiResolutionAddressCompiler.progressive_fine_candidates
    _ProgressiveGroundingAddressOrganizer

G/W/P placement and sole P value read:
  clearvla/policy/trunk.py
    LateRawDetailPolicyReader
    V39PolicyTrunk.forward

launcher and frozen-checkpoint probe:
  scripts/current_v111_structured_ownership_bottleneck.sh
  scripts/current_v111_structured_ownership_bottleneck_smoke.sh
  scripts/run_v111_model_path_probe.sh
```

V111 retains V110/V109's removal of the retired V108
`_HorizonSoftAddressJEPA`, the compiler-only target-value projection, or the
compiler-only coarse-geometry projection. They would otherwise be trainable
dead branches. The serialized V108 online flag remains only to preserve parent
identity and is suppressed by the progressive topology.

### 0.8 Required next evidence

Before a long run:

1. CUDA smoke completes preflight, forward, total-loss backward, optimizer
   step, short validation, and five-step deployment sampling.
2. All V111 metrics are finite; G2 dynamic-candidate validity is nonzero; the P
   read occurs exactly once; the old V108/V107 readers occur zero times.
3. Action loss gives finite nonzero gradients to progressive G1, G2, G3, W
   query/transport, typed key projections, literal-RGB local refiner, learned
   raw-detail lane and final P2 projection.
4. Zero RGB and zero learned detail are independent interventions; jointly
   zeroing both produces an exact zero protected detail update.
5. Current-coordinate reconstruction is exact up to bilinear sampling; future
   transport stays finite, horizon-varying and never changes the cached bank.
6. A V111 checkpoint passes its model-path probe, including owner-specific,
   current-coordinate, future-transport, RGB, detail and world interventions.
7. The V110 parent flag-off path is numerically identical under matching
   weights and inputs.

Interpretation rule: an internal representation delta with negligible action
delta is evidence of downstream attenuation or compensation, not proof of
useful control.

### 0.9 Maintenance rule

When the architecture changes, update Sections 0.2-0.8 in the same change as
the code. Record only decision-changing measurements in the historical ledger.
Do not paste logs, checkpoint contents, tensor dumps, or conversation summaries
into this file. Do not create `contract_v2`, `contract_v3`, and similar copies.

### 0.10 V110 design lock and implementation order

This is the compact continuity record for the current refactor. It is not
permission to modify the bottom network.

1. Preserve the V109 selector skeleton and V107 finite-gain normalization.
2. Replace the learned-feature-only detail claim with a native-resolution,
   literal bounded current-RGB chart plus a separate learned-detail chart.
   Never area-pool the RGB owner to the learned-feature side before P.
3. Store semantic, raw-pair and geometry components separately; combine only
   their scalar compatibility logits, with each contribution logged.
4. Replace the edge-sensitive divided coordinate limiter with a smooth
   source-relative update whose forward and reverse gains stay finite at image
   boundaries.
5. Let W predict future transport as a distribution attached to exact current
   anchors. Its bounded geometry must participate in the JEPA relevance, the
   paired W->P source prior, P1 fine routing and P2 conditioning; it never
   indexes a nonexistent future image.
6. Let W->P ingress A retain a 3x3 micro-patch layout from RGB and learned
   detail. Let ingress B use a local DINO-conditioned refiner before writing
   the unchanged `[B, action_horizon * basis, hidden]` policy input. The real
   P1/P2 DiT blocks then organize that input without reopening the bank.
7. Keep Evidence MMDiT, CVAE/workspace, adaptive selection and execution
   unchanged. Bottom information ownership is a later controlled change.
8. Audit in three passes: tensor/causality ownership; natural-autograd and
   numerical gain; flags-off/runtime/log/probe coverage.

### 0.11 V111 structured ownership bottleneck (implemented; CUDA pending)

This section preserves the V110 source audit and the design implemented by
V111.  The opt-in flag, launcher, serialized contract, CPU BF16 forward/
backward smoke, exact-zero value test, metrics and owner interventions now
exist.  CUDA smoke and checkpoint-level causal acceptance remain pending.

#### 0.11.1 Decision-changing evidence

The V110 batch-eight run through epoch-1 batch 1760 is numerically healthy and
shows useful action, flow and literal/detail learning.  It also isolates an
ownership problem that is not explained by the earlier batch-one optimizer
schedule:

- the late world anchor/common residual is about `65.36`, while its spatial
  residual is about `8.81`;
- the W3 horizon adjacent cosine is about `0.987`;
- future transport has an offset RMS near `0.14`, but adjacent-horizon
  variation near `0.0016`;
- the P2 local refiner gradient is about `4.08e-2`, while the combined typed
  condition gradient is about `1.8e-5`;
- four policy glimpses are active, but their route variation is only about
  `0.0023`.

These values do not say that the important information is absent.  They say
that the common carrier has much higher usable bandwidth than the typed,
horizon-specific and spatial innovations, so the latter survive mostly as
small perturbations that downstream blocks can ignore or compensate.

#### 0.11.2 Static findings in the current V110 source

`SB01 shared posterior at G2`: `flow_dino_evidence.py` builds independent
semantic, appearance and geometry keys and queries, but sums their three
compatibility logits before the only fine posterior (`4073-4082`).  Type
tensors survive in side state, while the selection decision no longer records
which owner established it.

`SB02 shared slot ownership at G3`: the three typed slot scores are summed
into one `slot_score` (`4247-4257`), and the resulting single `slot_weights`
aggregates every typed summary (`4370-4384`).  The summary tokens are typed by
projection, but every type inherits the same slot decision.

`SB03 generic public-memory handoff`: G3 flattens its typed
`[camera,xy,type]` summaries and appends them to the ordinary visual key and
value memories (`trunk.py:4710-4727`).  Every later non-policy block consumes
that bank through the same generic visual cross-attention
(`trunk_primitives.py:929-952`).  The type axis exists, but it has no protected
consumer or write responsibility.

`SB04 shared world posterior and transport`: W independently projects typed
queries, then sums all typed logits into one posterior
(`flow_dino_evidence.py:4685-4698`).  Its transport network concatenates the
query and all three owners into one MLP (`4518-4530`) and directly emits one
five-channel transport per horizon.  There is no explicit current/common
state plus chronological interval-innovation decomposition.

`SB05 public role bridges`: Ground-to-world AttnRes routes block deltas after
pooling the complete xy chart and broadcasts the result back over xy
(`trunk.py:3423-3448`).  World-to-policy AttnRes routes depth/camera values but
not semantic/appearance/geometry ownership (`3666-3778`).  These bridges are
valid depth selectors; they are not evidence-ownership bottlenecks.

`SB06 shared P1 decisions`: the policy reader separately computes typed fine
and coarse logits, but sums them before each posterior
(`trunk.py:1126-1145`, `1237-1256`).  Semantic, appearance and geometry
contexts are then all reduced with those same route/fine weights
(`1374-1387`).  Thus the stored types cannot request different candidate
support.

`SB07 P2 condition soup`: the local refiner preserves a 3x3 value layout, but
adds policy query, semantic, appearance, geometry and future transport into
one condition token (`trunk.py:232-245`).  The refiner can be active while the
typed condition remains causally weak, which is exactly what the V110 log
shows.

`SB08 test-coverage boundary`: current V110 tests establish dtype correctness,
one value read, exact-zero value behavior, attachment and nonzero gradients.
They do not require an owner-specific intervention to affect its assigned
posterior, local read or action.  Attachment is therefore tested; functional
ownership is not.

#### 0.11.3 Core design rule

The new bottleneck constrains **access, provenance and responsibility**, not
raw information capacity:

```text
common information -> structured public carrier
important/selective information -> structured owner evidence bundle

deployed state = public carrier + bounded owner innovations
```

The public carrier must not be one global mean.  It has two useful scopes:

```text
global public tokens:
  task identity, current robot state, compact history, phase uncertainty

camera-spatial public chart [B,C,G,G,R_public]:
  current low-frequency scene layout, camera identity, stable motion context
```

Exact coordinates, camera/source identity, validity and scale are shared
factual metadata rather than learned content owners.  Candidate-specific
semantics, appearance, high-resolution values, future transport and
horizon-specific change are not public carrier values.

The public carrier always remains outside owner softmaxes.  An owner router may
decline an optional innovation, but it may neither delete the carrier nor make
a mandatory precision path compete for survival against the carrier.

#### 0.11.4 Implemented side-car contracts

Use a side-car data contract rather than adding every source as another
full-hidden canvas token.  Names below are semantic contracts; implementation
may chunk camera, horizon, slot or candidate axes.

```text
ObservedEvidenceBundle:                       # observation-only, ODE-cacheable
  semantic_key       [B,C,G,G,M,K,R]
  appearance_key     [B,C,G,G,M,K,R]
  geometry_state     [B,C,G,G,M,K,Dg]
  current_xy         [B,C,G,G,M,K,2]
  source_xy          [B,C,G,G,M,K,2]
  validity           [B,C,G,G,M,K]
  confidence/uncertainty/visibility
  literal_rgb_chart  [B,C,3,Rgb,Rgb]          # referenced, not copied per slot
  learned_detail_map [B,C,Cr,Rd,Rd]           # referenced, not full-hidden
  provenance         camera/source/slot/candidate/type identity

HorizonOwnershipBundle:                       # W-owned, not cacheable
  public_world       [B,C,G,G,R_public]
  interval_delta     [B,I,C,G,G,M,R_h]        # I=(4-8,8-16,16-32,32-48)
  semantic_relevance [B,A,C,G,G,M]
  transport_state    [B,A,C,G,G,M,5]
  owner_logit_delta  semantic/appearance/geometry, retained separately
  reliability/provenance

PrecisionEvidenceBundle:                      # P-owned transient result
  rgb_micro          [B,T,basis,L,9,3]
  detail_micro       [B,T,basis,L,9,Cr]
  micro_xy           [B,T,basis,L,9,2]
  semantic_context   [B,T,basis,L,R]
  appearance_context [B,T,basis,L,R]
  geometry_context   [B,T,basis,L,Dg]
  horizon_context    [B,T,basis,L,R_h]
  posterior/provenance/reliability
```

`M` remains a soft multi-modal slot axis, `K` remains a continuous fine
candidate axis, `L` remains several autonomous glimpses, and the 3x3 micro
layout remains explicit.  This is why the owner bottleneck is not a global
bottleneck: it restricts the only legal high-resolution access interface while
retaining multiple modes, cameras, positions, time scales and local pixels.

Do not cache `[...candidate..., hidden_size]` values.  Preserve low-rank keys,
small-channel values and chart references, aggregate values in small-channel
space, and project only after the query-dependent read.

#### 0.11.5 Functional factorization through 3-3-2

The target is one connected prediction path, not separate action heads.

`Pre-G`:

- builds the public camera-spatial chart and `ObservedEvidenceBundle`;
- keeps literal RGB, learned detail, DINO semantics, raw-pair appearance and
  geometry factual lanes separate;
- provides no policy query, noisy action or future teacher to cached values.

`G1 -- semantic hypothesis`:

- forms complete-chart, multi-slot semantic hypotheses;
- may use coordinate validity and a bounded flow proposal as priors;
- does not read or aggregate high-resolution values.

`G2 -- geometric rectification and local verification`:

- geometry owns centre, support, uncertainty and transport-ready metadata;
- appearance owns local source/current verification logits;
- semantic, appearance and geometry logit deltas remain named fields;
- no single posterior is allowed to become the only surviving record of all
  three owners.

`G3 -- canonical evidence compilation`:

- emits a small public low-frequency summary to ordinary visual memory;
- emits the full structured owner bundle through a typed side-car boundary;
- does not use one shared slot distribution to aggregate every owner;
- preserves explicit type and provenance embeddings rather than relying on
  flattened token order to communicate identity.
- its typed slot posteriors condition the matching W owner-source posterior;
  no unconsumed full-hidden owner-summary copy is retained for diagnostics.

`W1-W3 -- current base plus horizon innovation`:

- reconstruct every world state as `public_world + horizon_innovation`;
- obtain `public_world` from current observation/history, never by averaging
  future-owner states after they have mixed;
- represent future motion as chronological interval increments attached to the
  exact current anchors, then compose them to +4/+12/+24/+48 distributions;
- let W1 emphasize near change, W2 intermediate consequence and W3 far
  phase/goal consequence, while later blocks may read earlier typed deltas;
- keep owner innovations outside the generic public visual-value memory.

The common/innovation representation must be information preserving.  Where a
linear decomposition is used, it must have an exact reconstruction test.  Do
not normalize a near-zero innovation to unit scale merely to make it visible.

`P1 -- factorized address decision`:

Use one joint address with explicitly factored responsibilities rather than
three independent action heads or one all-owner soup:

```text
p(source,slot,fine | query)
  = p(source,slot | semantic, horizon, public world, geometry prior)
    * p(fine | source,slot, appearance, geometry, transport)
```

Geometry changes coordinates/support and supplies bounded likelihoods;
semantic evidence establishes task/object relevance; appearance resolves local
correspondence and precision; horizon innovation changes which observed anchor
is useful now.  These factors remain inspectable even though their normalized
product is the single posterior used for the value read.

`P2 -- owner-conditioned local read`:

- retain RGB and learned-detail micro tokens as distinct value types through
  local spatial processing;
- use geometry as coordinate/key/bias information, not as a value substitute;
- present policy, semantic, appearance and horizon information as typed
  condition tokens or typed sublayers, not one five-way sum;
- allow the final action-ready precision residual to fuse owners only after
  each owner has performed its assigned operation;
- retain per-owner contribution handles for intervention even after the final
  fusion.

`P2 -> bottom`:

- preserve the existing protected additive detail route and existing
  `PolicyRoleDeltaBank` in the first implementation;
- keep Evidence MMDiT, CVAE/workspace, adaptive selection and execution
  unchanged;
- do not expose the complete observation bank to the bottom as a workaround.

#### 0.11.6 Collapse and shortcut prevention

- No hard gate, argmax, top-k, straight-through estimator, route-mass quota,
  fixed entropy target, owner-balance loss or artificial nonzero-flow loss.
- No pooled RGB/detail duplicate in the public carrier.
- No future teacher in either the public carrier or owner bundle.
- No shared posterior may be the only stored result of several owners.
- No owner innovation may be reconstructed from an affine bias when its source
  value is zero.
- Fixed horizon identity/position metadata may remain public, but the public
  carrier may not hide learned horizon-specific evidence values; such content
  enters through interval/horizon innovations.
- Apply variance-floor normalization and smooth RMS contracts per owner before
  fusion.  Bounding only the final sum allows one owner to cancel or dominate
  another invisibly.
- Preserve ordinary autograd.  Do not detach an owner and patch its gradient.
- A zero owner update is allowed when the sample does not need it.  Collapse is
  diagnosed by matched interventions and data-conditioned variation, not by
  forcing every sample to use every owner.

#### 0.11.7 Supervision policy

The first implementation does not add generic ownership, entropy, diversity
or usage losses.

- action flow matching supervises the final precision residual and all natural
  upstream paths;
- warp/cycle/uncertainty supervise geometry and learned flow;
- JEPA future change supervises the semantic/horizon prediction;
- interval-stage teacher targets attach to the actual interval innovations,
  not to a public carrier after fusion;
- spatial relevance supervises the corresponding semantic/source factor, while
  appearance and geometry retain their factual local objectives;
- an additional loss is considered only after a valid intervention proves the
  intended boundary moves but the relevant factual target remains
  underdetermined.

#### 0.11.8 Required metrics and probes

Add a boundary ownership matrix rather than only aggregate gradients:

```text
boundary rows: G1, G2-geometry, G2-appearance, G3-semantic,
               W-interval, W-transport, P1-source, P1-fine,
               P2-RGB, P2-detail, P2-condition, deployed action
intervention columns: semantic zero/shuffle, appearance zero/spatial-shuffle,
                      geometry neutral/spatial-shuffle, interval zero/shuffle,
                      RGB zero/shuffle, learned-detail zero/shuffle
```

For every boundary report source delta, posterior delta, value-read delta and
deployed-action delta.  Also report:

- public RMS, each owner-innovation RMS and owner/public ratio;
- exact common-plus-innovation reconstruction error;
- owner logit RMS/span before combination and effective posterior change;
- horizon innovation variation and adjacent cosine before/after each W block;
- per-glimpse posterior distance, coordinate distance and value distance;
- per-owner P2 condition/output gradient and intervention sensitivity;
- RGB-only, detail-only and joint-value interventions;
- episode/horizon/camera variation, not just global means.

Acceptance is causal rather than cosmetic.  Nonzero gradients, nonuniform
attention or a larger owner norm do not prove useful ownership.  A valid route
must change its assigned boundary, and the relevant joint intervention must
reach action.  Utility still requires validation error or a
precision-critical subset to worsen when useful evidence is removed.

#### 0.11.9 Implementation and review order

Implement behind one opt-in top-structure flag whose off state exactly
reproduces V110.  Work in one coherent branch, but keep the following internal
checkpoints so faults can be localized without running unrelated experiment
variants:

1. Freeze a V110 golden forward and serialized run-context check.
2. Introduce the bundle dataclasses, public/owner decomposition and shadow
   metrics with no action change; prove exact reconstruction and cache safety.
3. Factor G2/G3 ownership and add the typed G->W side-car while preserving the
   existing public carrier.
4. Replace direct common-mode future transport with current-anchor plus
   chronological interval innovations; attach existing interval supervision to
   the actual online write.
5. Replace the two shared P1 posterior sums with the source/fine factorization,
   preserving all candidates and the one-read rule.
6. Replace the P2 summed condition with typed local operations while retaining
   the 4-glimpse x 3x3 value bandwidth and protected bottom route.
7. Run three static reviews: tensor/causality/provenance; dtype/autograd/
   normalization/amplitude; flags-off/cache/ODE/log/probe coverage.
8. Only after the complete path passes CPU tests and CUDA smoke, run the owner
   intervention matrix and one full validation.  Do not redesign the bottom in
   the same change.

Implementation status (2026-07-28): steps 1-7 are present in source. The CPU
BF16 synthetic path traversed G1/G2/G3, W posterior/transport, P1/P2 and total
detail backward with nonzero gradients for all G2 queries, all G3 typed slot
scorers, all W owner queries, future transport and P2 owners. Python
compilation, exact V110 flags-off state construction and the exact-zero
structured P2 value test pass. Step 8 remains gated on the production CUDA
smoke and a trained V111 checkpoint; no runtime result is claimed yet.

The implementation is incomplete if it merely adds more typed names, logs or
auxiliary losses while retaining the shared G2/G3/W/P1 posteriors and summed P2
condition described in `SB01-SB07`.

## 1. Objective

The architectural objective is not to force every condition or visual source
to be used on every sample. It is to keep every potentially useful source:

- available at sufficient resolution;
- selectable by the appropriate query;
- differentiable through its natural forward path;
- causally able to reach the deployed action;
- recoverable later instead of being irreversibly compressed early.

Data identifiability and coverage are separate experiment tracks. The network
must not impose an avoidable bottleneck before those tracks can be studied.

## 2. Non-negotiable invariants

1. Raw RGB, DINO, flow, camera identity, address uncertainty, and multiple
   spatial candidates survive until the final precision read.
2. An 8×8 cell is a query cell with several soft address slots. It is not a
   fixed pointer to one high-resolution patch.
3. Flow is a geometric proposal/prior. It never uniquely determines the final
   address.
4. Goal and phase condition address queries; they do not contaminate cached
   observation values.
5. Observation banks contain no noisy action and no future teacher target.
6. G/W/P roles are typed responsibilities on a shared persistent state, not
   isolated information silos.
7. Action loss reaches G/W queries, flow, DINO/raw keys, raw values, and the
   final reader through ordinary autograd.
8. No detach-followed-by-gradient-patch, hard top-k, argmax, straight-through
   estimator, forced route mass, fixed entropy target, or artificial nonzero
   flow quota.
9. High-frequency detail has a protected additive route to action, but the
   architecture does not force a sample to depend on irrelevant detail.
10. There must be no pooled duplicate that lets the action path reconstruct
    the same high-frequency information while bypassing the address read.
11. The existing bottom Evidence MMDiT, CVAE/workspace, adaptive layer
    selection, and execution machinery remain available unless a separate
    controlled change explicitly replaces them.
12. New modules initialize near identity or zero-delta and preserve their
    immediate parent path. Disabling the V111 structured-ownership flag
    reproduces V110; disabling V110 coordinate-typed raw detail reproduces
    V109. Disabling the four V107 top-path flags reproduces
    V106; disabling both V106 interval-stage and variance-safe-routing flags
    reproduces V105; disabling the V105
    horizon-address flag and reliable-loss option reproduces V104; disabling
    the three V104 repair flags reproduces V103; the older V103 flags still
    preserve their documented V102 parent path.

## 3. Historical ancestry and static audit ledger (not a default read)

This section preserves implementation evidence for old versions. It is not the
current topology contract. Read it only for ancestry, old-log interpretation,
or the rationale behind a named repair; Section 0 and current source remain
authoritative.

### Language

- One precomputed T5 condition is loaded as the default for the entire
  single-task run.
- A GoalTokenResampler produces a few task tokens.
- Those tokens enter the top canvas and the bottom clean-intent organizer.
- The current raw/Flow-DINO observation pack is built before goal resampling.
- Goal is not an explicit input to the late high-resolution address query.
- Goal can affect that read only indirectly through the world path.
- Current goal dropout zeros T5 content before the resampler but retains a
  learned null-goal query/template.

Therefore the V96-V102 tensor path exists but does not provide a strong
goal-conditioned spatial, camera, horizon, or phase interface. The opt-in
post-V102 repair adds exact-null conditioning and selector-only world/detail
query deltas without modifying observation values.

### Long-horizon phase

V96 through V102 explicitly use:

```text
flow_jepa_late_bottleneck = 1
flow_jepa_stage_offset = 0
flow_jepa_stage_loss_weight = 0
flow_jepa_stage_tokens = 0
```

The V96-V102 reference has 4/12/24/48 future queries, not a stateless
task-phase estimator. The opt-in post-V102 path adds a stateless soft phase
belief without reinstating a separate stage token.

The older V95 `stage` token predicted one pooled far-future DINO change. It was
not a semantic task-stage belief and did not represent arbitrary long-task
progress.

### Flow and high-resolution read

- The coarse DINO flow uses one soft expected displacement.
- Ambiguous or multimodal correspondence can therefore average toward zero.
- The identity-centred prior intentionally maps uninformative correspondence
  to zero flow.
- Default raw refinement radii are 2 on the mid chart and 1 on the high chart.
- If the coarse seed is wrong or near zero, the fine stages may not cover the
  true displacement after scale conversion.
- The current raw refiners borrow SEA-RAFT ideas but are compact one-pass
  scale refiners, not a complete recurrent SEA-RAFT implementation.
- V102 late reading preserves per-camera charts during attention, then
  combines camera contexts with a fixed equal variance-preserving sum.
- V102 pre-world raw compilation still compresses local high-resolution
  candidates before the world/policy query can choose among all candidates.
- The opt-in soft lattice retains multiple modes and continuous fine
  candidates, performs query-conditioned camera selection, and keeps a
  positive soft flow-prior floor in the repaired predictive contract.

### Top-to-bottom gradient path

- Only the final policy blocks may directly write trajectory tokens.
- World blocks write only stage/rollout; grounding writes clean context and
  spatial evidence.
- The bottom decoder also has direct noisy-action and global clean-intent
  inputs.
- Consequently some policy-gradient dominance is expected, but persistent
  near-null world/detail interventions indicate bypass or attenuation rather
  than merely different module sizes.
- Do not equalize gradient norms with an auxiliary gradient loss. Shorten and
  type the forward paths, then judge them with causal interventions.

### Model-only static audit ledger

Architecture ancestry anchor: V96, where the late-bottleneck representation
first became explicit. Functional comparison baseline: V98, the last broadly
usable raw/detail mainline before the successive V99-V102 constraints.
Historical note: when this ledger section was written, its implementation
target was V107 with V106 as the frozen parent baseline. It is not the current
V110 target. The audit did not copy one old version wholesale; it checked
every V99-V107 delta against both V96's intended information contract and
V98's demonstrated usable action path. This ledger
excludes run management, logging, speed, cache, Git, and other non-model
concerns.

Audit findings, in dependency order:

1. `M01 premature flow expectation` (V96-V102 defect; structural repair
   implemented behind an off-by-default flag, empirical validation pending).
   `LatentSeaRaft` converts the complete correspondence distribution to one
   expected coordinate at `flow_dino_evidence.py:482`. Both the sparse DINO
   refiner and dense raw refiners then take another expectation over local
   offsets. Distinct plausible modes can therefore cancel into a near-zero
   displacement before a semantic query sees them.
2. `M02 premature precision-read collapse` (V96-V102 defect; structural repair
   implemented behind an off-by-default flag, empirical validation pending).
   `_SoftFlowAddressReader` and `_RawDeformableAddressReader` softmax all
   identity/flow/fine candidates and emit one key/value per 8x8 cell
   (`flow_dino_evidence.py:2034-2051` and `1537-1553`). G/W/P, goal, phase,
   horizon, and action-basis queries cannot recover the discarded modes.
3. `M03 historical value deletion by confidence` (V96 only, fixed later).
   V96 multiplies the late aligned value itself by `fine_confidence`
   (`flow_dino_evidence.py:3620-3626`). With observed confidence near 0.003,
   this almost removes the precision value lane. Raw complementary paths no
   longer use this exact amplitude deletion.
4. `M04 historical direct visual bypass` (V96/V98/V99, fixed by V100 strict
   ownership). Policy blocks and the final decoder could re-read visual
   selector/value memory directly (`trunk_primitives.py:461`,
   `trunk.py:1349-1367`, `1749-1758`). This made the JEPA/world route optional.
   Keep the strict boundary; do not reintroduce the bypass while repairing
   bandwidth.
5. `M05 3-3-2 depth semantics are not preserved as values` (V98-V102 defect;
   typed-delta repair implemented behind off-by-default flags, empirical
   validation pending). Role write masks are real, but the old bottom strict path retains
   only terminal P layer contracts (`trunk.py:1811-1817`). In the Evidence
   adapter, actual layer tokens provide retrieval geometry while layer values
   are freshly synthesized from clean intent
   (`time_domain_mmdit.py:246-266`). Consequently G/W/P depth increments are
   not available to the bottom as typed value evidence.
6. `M06 fixed policy-workspace superhighway` (V101-V102 defect; replacement
   implemented behind an off-by-default flag, empirical validation pending).
   Fixed fusion layer-normalizes the P workspace, matches it to action RMS,
   and mixes both branches with `sqrt(0.5)`
   (`time_domain_mmdit.py:2436-2447`). It removes the old 0.10 bottleneck but
   also forces an equal-energy, provenance-free P route. The V101 action-path
   probe found policy-zero action delta RMSE 0.1331 and +293.2% MSE, while raw
   address/detail interventions changed action by only about 0.00084-0.00087.
7. `M07 late detail is late only after an early bottleneck` (V102 defect;
   structural repair implemented with the soft address lattice, empirical
   validation pending). The observation-only compile correctly prevents action echo,
   but it sets the grounding query to zero
   (`flow_dino_evidence.py:3018`) after candidates have already collapsed to
   one 8x8 vector. The late policy reader can select among 8x8 cells, not among
   the original high-resolution candidates.
8. `M08 fixed camera fusion` (V102 defect; repaired in the lattice policy
   reader, empirical validation pending). Per-camera attention is
   preserved, then camera contexts are always summed with equal
   variance-preserving weights (`trunk.py:233-236`). Dataset evidence shows
   camera usefulness varies with motion, horizon, and task, so camera
   combination must be query-conditioned.
9. `M09 goal/history lack an explicit spatial and phase interface` (V96-V102
   defect; structural repair implemented behind off-by-default flags,
   empirical validation pending). T5 and action-history tensor paths exist, but neither directly
   conditions the clean address posterior. Goal affects the late detail read
   only indirectly through canvas/world state. There is also no stateless
   phase-belief adapter.
10. `M10 condition-dropout null semantics are inexact` (V101-V102 defect;
    exact-null repair implemented behind off-by-default flags, empirical
    validation pending). Language is zeroed before a biased resampler while its mask
    remains valid (`system.py:591-603`), so dropout yields a learned template
    rather than an explicit post-resampler null/content-delta condition. The
    same class of constant survives in history-side biased projections.
11. `M11 current JEPA mask is not early I-JEPA masking` (V96-V102 defect;
    raw pre-mixing boundary implemented, cached-DINO limitation remains
    explicit). Cached final-layer DINO has already mixed same-frame patches before
    `_structured_mask`; the source itself identifies this as context dropout
    (`flow_dino_evidence.py:19-24`, `2398-2430`). Static semantic content can
    therefore remain available through unmasked tokens.
12. `M12 future loss does not make predictive change causally necessary`
    (V96-V102 defect; explicit delta-prediction repair implemented behind an
    off-by-default flag, empirical validation pending). Absolute future DINO prediction is dominated by
    static content; its internal change-direction term is only 0.10 of the
    future loss (`policy_runtime_v39.py:790-836`). V100 adds a separate
    change-weighted objective, but action can still succeed through the P
    workspace without using future/world evidence.
13. `M13 flow objectives identify correspondence, not task usefulness`
    (V96-V102 defect; structural gradient path and nonzero soft-prior floor
    implemented, empirical validation pending). Warp/cycle/smoothness/uncertainty losses can improve
    physical alignment while the action decoder ignores that alignment. They
    are valid proposal objectives, but the final query-conditioned address
    posterior also needs ordinary action-gradient access through a
    non-collapsed path.
14. `M14 compounded typed-role attenuation` (initial V103 static defect;
    repaired, empirical validation pending). The original bridge return values
    already included the `0.10` carrier step, so G evidence reaching the bottom
    was multiplied by `0.10 * 0.10 * 0.25`, W by `0.10 * 0.25`, and P only by
    `0.25`. Carrier writes remain bounded, but typed boundary values now retain
    the unscaled routed delta and receive the one common bottom scale.
15. `M15 +48 loses typed W->P identity` (initial V103 static defect; repaired,
    empirical validation pending). The legacy action-alignment helper correctly
    selected only 4/12/24, but the same helper was also used to build every W->P
    typed candidate. +48 remained present in the full rollout bank but could not
    be selected or intervened as a provenance-preserving far-context delta.
    Each source/camera now exposes +48 as a separate context candidate without
    assigning it an action-time label.
16. `M16 nominal flow floor can be flattened by uncertainty` (initial V103
    static defect; repaired, empirical validation pending). The floor
    coefficient previously multiplied the same adaptive-width Gaussian as the
    learned prior, so large uncertainty/occlusion could make its spatial logits
    nearly constant. The floor expert now has a fixed coarse-cell width; a
    separate learned expert keeps the uncertainty-adaptive width.
17. `M17 world blocks cannot write spatial change` (V102 inheritance in the
    initial V103 path; repaired, empirical validation pending).
    `flow_jepa_world_anchor_write_only=1` averaged every W-block residual over
    the 8x8 chart and broadcast one vector per anchor/camera. W blocks could
    read spatial evidence but could not express a spatially varying near/mid/far
    prediction, forcing that work back into G. V103 now permits W to write the
    coarse 8x8 DINO/flow world chart. This does not reopen the raw high-frequency
    bypass: raw residual values remain excluded from the G/W visual bank and
    enter only through the late policy address reader.
18. `M18 world spatial state is averaged before precision addressing`
    (initial V103 defect; repaired, empirical validation pending). Restoring
    W1-W3 spatial writes was insufficient because
    `LateRawDetailPolicyReader` immediately averaged the W chart over xy before
    forming its high-resolution address query. The policy could read a spatial
    W chart later, but that chart could not decide which raw candidates entered
    the protected detail update. The soft-lattice reader now aligns the complete
    W chart to action time and adds a per-camera/per-xy world compatibility term
    to the address logits. This term is selector-only: it never enters or
    rewrites the cached raw value bank.
19. `M19 grounding rollout reads the candidate action` (V98-V103 defect;
    repaired, empirical validation pending). Role write masks prevented G
    blocks from writing trajectory, but G rollout queries could still attend
    both the flow-matching `x_t` trajectory and the history proposal through
    canvas self-attention and the explicit rollout cross-attention sublayer.
    G1-G3 could therefore become an early action-denoising route instead of a
    clean alignment/canonicalization stream. Grounding rollout now excludes
    proposal and noisy trajectory from self-attention, dynamics context, and
    block modulation. W1-W3 remain the first blocks that combine the candidate
    action with the grounded spatial state.
20. `M20 JEPA context mask and supervised target mask disagree` (V96-V103
    defect; repaired, empirical validation pending). The early RGB encoder and
    DINO organizer used `context_dropout`, while the future loss used a second
    independently sampled `future_target_mask`; V101 teacher balancing then
    replaced that second mask after the online forward. A supervised future
    coordinate could therefore remain visible in the latest online context.
    Predicting teacher deltas weakened absolute-scene copying but did not make
    this a valid masked-prediction boundary. Predictive-change mode now samples
    one mask from observation-only motion/random evidence, reuses it across all
    real horizons, applies it to the latest RGB/DINO context before trainable
    spatial mixing during training, and uses exactly the same coordinates for
    the future loss. Teacher-derived target selection is forbidden in this
    mode; teacher features remain targets only. Deployment keeps the complete
    current observation because masking is a training objective, not an
    inference-time information deletion.
21. `M21 source raw appearance is absent from the soft address compiler`
    (initial V103 defect; repaired, empirical validation pending).
    `_SoftMultiResolutionAddressCompiler.forward` accepted `source_raw` but
    used it only for shape validation. Coarse slots could use source DINO and
    flow, while fine keys contained target raw/target DINO only. A weak flow
    therefore remained the sole source-side high-resolution cue and local raw
    appearance could not correct its fine address. The compiler now samples a
    source-side sub-cell lattice, constructs a learned source/target raw pair
    key from both features, their product, and their difference, and adds that
    key only to fine selector evidence. High-frequency values remain sampled
    exclusively from the target raw residual; source appearance cannot rewrite
    them or create an action echo.
22. `M22 formal V103 identity and optimizer ownership are not enforced`
    (initial V103 integration defect; repaired). The launcher previously
    advertised V103 while only the later evaluation probe checked the resolved
    serialized graph. A wrapper-order regression, zero-scale bridge, disabled
    native execution controller, or missing observable-flow objective could
    therefore begin training under the same label. The formal launcher now
    requires the complete V103 contract before optimization and the frozen
    checkpoint probe reuses the same validator. Optimizer construction also
    excludes the intentionally frozen legacy action-history positional key;
    an exact ownership test now proves that every trainable parameter has one
    optimizer owner and no frozen parameter has one.
23. `M23 optimizer-owned V103 parameters are absent from the deployed loss
    graph` (initial V103 integration defect; repaired). The single-stage
    strict-role model still instantiated and optimized the historical mid-cut
    heads, G/W layer-contract probes, disabled P-block rollout-transition
    sublayers, strict-mode visual intent projection, and raw-mode semantic mask
    token. A complete V103 loss backward left 275 of 1169 trainable parameter
    tensors with `grad=None`; these were not weak gradients but unreachable
    branches. Strict Evidence mode now retains only the two terminal P-layer
    adapters required by the final decoder, freezes their unused readouts,
    freezes all structurally unreachable compatibility modules, and does not
    execute G/W layer-contract heads. The exact-null shared-condition output
    bias is also frozen because `f(condition)-f(0)` cancels it identically.
    With the complete action, JEPA, observable-flow, and execution-value
    objective, every remaining trainable parameter is graph-reachable on the
    first backward and has nonzero gradient after one ordinary optimizer
    update. The formal V103 validator additionally rejects the old shared
    layer-FM probe and recurrent consequence branch.
24. `M24 history proposal is action-gradient detached` (V96-inherited V103
    defect; repaired). The planner received `proposal.tokens.detach()` during
    training even though the same causal history proposal is available during
    deployment. Direct history memory remained attached, so history was not
    globally disconnected, but the proposal future queries and proposal blocks
    could learn only from their auxiliary regression head and could not adapt
    to final action utility. V103 now sets
    `action_history_proposal_detach=0`; the historical detached behavior
    remains an explicit compatibility option. An action-only backward gives
    nonzero gradients to every proposal future-query/block parameter with the
    V103 setting and `grad=None` with the legacy setting, while both settings
    produce bitwise-identical forward values before backward.
25. `M25 nominal multi-slot and typed-role tensors can collapse invisibly`
    (V103 acceptance defect; factual instrumentation repaired, trained
    validation pending). A configured slot count greater than one did not prove
    that coarse centers or full posteriors differed, that the final policy used
    more than one slot, or that its slot choice varied by query. Likewise,
    globally averaged AttnRes source mass could hide identical routing at every
    anchor, camera, horizon, or action basis. The compiler now reports
    normalized pairwise center distance and posterior Hellinger separation; the
    late policy reader reports effective slot count and query-conditioned slot
    variation. G->W, W->P, P->MMDiT, and protected-detail routing now separately
    report non-null source effective count and semantic-axis route variation.
    The frozen deployment probe captures these values even on the minimal
    tensor-only sampling path and evaluates only numerical identity here. No
    diversity loss, balance loss, hard routing, gradient override, or arbitrary
    utility threshold was added. Whether trained V103 routing is useful still
    requires matched causal ablations and error intervals.
26. `M26 formal V103 launcher inherits the retired recurrent consequence
    branch` (V103 launch integration defect; repaired before training). The
    oldest V48 base script still appends
    `--layer-recurrent-consequence 1`. The formal V103 validator correctly
    rejected the resolved graph before the first optimizer step, but the V103
    wrapper had not explicitly overridden that inherited value. The formal
    launcher now sets both `--layer-shared-fm-probe 0` and
    `--layer-recurrent-consequence 0`, leaving the terminal P contracts and
    typed P-to-MMDiT bridge as the single deployed top-to-bottom action path.
    The launcher regression test records the conflicting base value and the
    required V103 override so the inheritance error cannot silently return.
27. `M27 learned flow can erase its own supervision by leaving the image`
    (trained V103 failure; repaired in V104, empirical validation pending).
    The recurrent coarse flow and both dense raw refiners accumulated
    unconstrained `flow + delta`. Photometric, cycle, and correspondence terms
    then multiplied by an in-bounds validity mask. Once enough coordinates
    left the image, the model both lost valid supervision and fed very large
    grid-unit flow/velocity directly into the learned motion key/value lane.
    The completed V103 run exposed this exact sequence: raw flow magnitude rose
    from 5.709 at batch 1020 to 52.918 at batch 1560, valid fraction fell from
    0.819 to 0.250, and moving-region warp gain changed from +0.0079 to
    -0.0802. V104 represents every coarse, recurrent, mid, and high proposal
    in a smooth source-relative in-image chart. Geometry remains in grid units
    for address compilation, while learned motion keys/values receive
    resolution-normalized displacement and velocity. There is no clamp,
    invalid-route reward, detached repair, or nonzero-flow quota.
28. `M28 pre-norm role residuals and AttnRes values have no amplitude
    contract` (trained V103 failure; repaired in V104, empirical validation
    pending). Layer normalization stabilized each sublayer input but did not
    limit the residual written back to the persistent carrier. Delta AttnRes
    normalized values only for route keys and then routed the original
    full-width values. The V103 log therefore showed a coherent amplitude
    escape rather than a mere role-share change: world anchor residual reached
    4991 and world xy residual 741 by batch 1020, then 7476/1565 by batch 1560;
    late detail update remained numerically nonzero but its trajectory ratio
    fell to 0.006. V104 applies one smooth, fixed normalized-chart RMS contract
    to every actual G/W/P sublayer write and to full-width values crossing
    G->W, W->P, and P->MMDiT/protected-detail AttnRes boundaries. The carrier
    is not part of the limit, so carrier inflation cannot evade it. Small
    writes are nearly identity, direction is preserved, and gradients remain
    ordinary autograd.
29. `M29 nominal +4/+12/+24/+48 queries are parallel common-mode predictions`
    (V96-V103 structural defect; repaired in V104, empirical validation
    pending). Every horizon was independently seeded from the same latest
    motion and current context, with only a horizon embedding and scalar
    magnitude change. The trained V103 adjacent-horizon cosine reached 1.000,
    so +48 did not structurally inherit a progression through the observed
    sequence or the nearer predicted states. V104 first reads all observed
    pairwise perceptual-motion states into one local memory, then applies one
    shared full-width transition in chronological order
    `history -> +4 -> +12 -> +24 -> +48` during a single forward call. The
    state is stateless across samples and calls, but attached within the call:
    no LSTM training pass, persistent episode cache, detach, target future
    feature, labelled action, or teacher-derived mask enters the transition.
    The +48 typed and bottom routes remain distinct after this change.
30. `M30 one shared online mask does not provide a horizon-specific spatial
    address, and normalized delta loss overweights weak teacher jitter`
    (V104 residual defect; repaired in V105, empirical validation pending).
    V104 correctly shares one observation-only context/target mask across all
    horizons, but the final JEPA predictor still receives each W cell without
    a +4/+12/+24/+48-specific precision read. The mask says which current
    locations are hidden; it is not a claim that equal-size future pixels map
    to the same source patch. V105 reuses the observation-only soft
    multi-resolution bank and lets every horizon/camera/W query cell softly
    select all same-camera source cells, slots, and continuous fine candidates.
    The resulting bounded, fixed-scale residual enters only the future JEPA
    predictor. It does not write the action trajectory or create a pooled raw
    action bypass. Frozen future change supervises only the predicted spatial
    relevance distribution in the loss; neither teacher values nor a
    teacher-derived mask enter the reader. The distribution is a softmax, so
    its mass cannot collapse to zero, while no top-k, route quota, entropy
    target, or learned close gate is added. V105 also retains raw delta
    Smooth-L1 and weights normalized magnitude/direction by detached
    teacher-change reliability using the fixed valid-position denominator and
    a smooth current-teacher-relative normalization scale. Low reliability
    therefore attenuates weak jitter instead of being divided away by a
    reliability-sum denominator or cancelled by using only the weak delta as
    its own reference. A bias-free raw-value path also prevents an
    address-independent constant residual.
31. `M31 normalize-after-cancellation makes the V105 address backward
    ill-conditioned` (trained V105 failure; V106 repair implemented, runtime
    confirmation pending). V105 first
    forms a diffuse weighted average of observation-owned high-frequency raw
    residuals and then applies affine-free LayerNorm to that aggregate. Broad
    posteriors can cancel positive and negative residuals to a very small but
    nonzero channel variance. LayerNorm restores a unit-scale forward value but
    exposes an inverse-standard-deviation backward Jacobian. The following
    fixed update coefficient and smooth RMS bound hide the failure in forward
    loss and activation magnitudes. In the completed V105 epoch-1 log, global
    preclip norm reached 4.31e6, grounding reached 2.49e6, and Flow-DINO reached
    2.09e5, while the controlled V104 parent remained below 1.72 globally.
    Static audit also found an independent inverse-norm singularity in the
    inherited predictive-change direction loss at a near-zero delta
    prediction, so the log does not justify assigning all of the explosion to
    the address LayerNorm alone. V106 repairs both mechanisms. It must never
    expand a small routed/address value merely to normalize its direction and
    uses a zero-preserving centered RMS normalization with a
    positive fixed denominator floor for selector keys and an upper-bound-only
    RMS contract for address values. The same safe selector normalization is
    applied to Delta AttnRes so a V105 address instability cannot be amplified
    at G->W, W->P, or P->MMDiT. Exact-zero values remain exact zero; there is no
    affine substitute, gradient patch, detach, hard route, or learned close
    gate.
32. `M32 point anchors and global phase modulation do not constitute interval
    stage progression` (V104/V105 semantic defect; V106 repair implemented,
    runtime confirmation pending).
    Point targets at 4/12/24/48 can be predicted as four static future states,
    and a single global phase vector can modulate every spatial query in the
    same direction. Neither operation requires a signed within-stage
    progression, and an unrestricted phase/value fusion could wash out the
    high-resolution address lane. V106 changes teacher semantics to four
    contiguous supports with shared boundary observations,
    `H4=[4,8]`, `H12=[8,16]`, `H24=[16,32]`, and
    `H48=[32,48]`. For every camera and 8x8 query cell it constructs a robust
    interval-content target, a signed least-squares progression target, and a
    separate endpoint increment. It is not a plain temporal frame mean.
    Online interval organization is chronological and observation-only. Its
    bounded delta is written only to the coarse W rollout/query chart at the
    W->P boundary, before typed depth selection and the precision read. It may
    therefore condition world-to-policy routing, the late spatial posterior,
    P blocks, and the bottom rollout path, but it never enters, averages, gates,
    or rewrites the continuous fine-value bank. The protected high-resolution
    value lane remains additive and outside every stage/depth source-survival
    softmax.
33. `M33 V106 future-teacher construction was absent from policy preflight`
    (runtime failure found before the first optimizer step; repair
    implemented, server confirmation pending). The interval teacher's local
    `flatten()` referenced `FlowDINOEvidenceEncoder.self.anchors`, although that
    outer encoder never owned such an attribute. The existing unit test would
    exercise this path in a PyTorch-capable environment, but the policy
    preflight sampled only deployment action and passed
    `include_target_visual=False`; it therefore printed a misleading V105
    preflight success before the first training batch entered the broken
    teacher path. The repair derives the anchor count from the stacked teacher
    tensor and validates it against the configured window contract. Formal
    V106 preflight now prepares all 12 future supports, builds the real target
    pack, validates FP32 content/progression/endpoint/current geometry, boolean
    mask geometry, support count, effective-support bounds, and finiteness, and
    prints V106 success only after both deploy sampling and interval teacher
    construction pass.
34. `M34 V106 bounded forward residuals did not bound the active backward
    interfaces` (delayed trained-run failure; complete numerical repair
    implemented, new smoke pending). The first V106 repair bounded routed
    values and near-zero direction objectives, but three upstream interfaces
    remained outside that contract: learned DINO/raw correlation used
    scale-invariant `F.normalize` with its default `eps=1e-12`; raw
    forward/backward consistency became a boolean visibility/occlusion value
    at one threshold; and every G/W/P block still used ordinary LayerNorm
    inside self-attention, visual cross-attention, rollout dynamics, and FFN
    before the already-bounded residual write. The resulting forward values
    could remain ordinary while their chained Jacobian was not. This matches
    the delayed transition in the supplied V106 run: at batch 600 global
    preclip was 1.15 with raw grid-flow 0.111, cycle 0.181, and occlusion
    0.008; by batch 620 they were 1.96/0.186/0.323/0.077; at batch 640 the
    same state crossed to 3.74e3/0.514/0.941/0.577, and the later preclip norm
    reached 2.76e7. There is no optimizer or loss schedule boundary at that
    location. The complete V106 numerical contract now applies one
    width-aware RMS-floored L2 normalization to every active learned
    correlation, a continuous cycle-visibility transition to online
    occlusion evidence (hard visibility is audit-only), and the existing
    centered variance floor to all G/W/P internal normalization sites. The
    learnable LayerNorm affine scale remains useful but is smoothly bounded
    at four, so the role-normalization maximum local gain is explicit rather
    than merely observed. The observation/address selector and protected raw
    value normalizers use the same zero-preserving rule. None of these changes
    detach gradients, add an auxiliary gradient, hard-select a route, or alter
    the flags-off V105 arithmetic.
35. `M35 the active soft-lattice policy reader is nominally multi-head but
    collapses every candidate before the value projection` (V103-V106 static
    defect; V107 repair implemented, runtime smoke pending). `LateRawDetailPolicyReader` validates and
    stores `flow_jepa_raw_reader_heads`, but its soft-lattice branch forms one
    route query, one fine/coarse/camera/slot posterior, and one signed
    high-frequency expectation before `lattice_value_out`.  The configured
    heads are used only by the retired dense-reader compatibility branch.
    Multimodal positive/negative detail can therefore cancel before a
    nonlinear policy projection, contradicting the requirement that several
    soft modes survive until the P-side query.  The repair must implement real
    independent soft glimpses, keep every selection differentiable, preserve
    exact-zero values, and concatenate the per-glimpse value channels only
    after each glimpse has read the observation bank.
36. `M36 the auxiliary horizon reader removes target-cell identity from its
    fine address and starts its value owner near numerical silence`
    (V105-V106 static defect; V107 repair implemented, runtime smoke pending).
    `_HorizonSoftAddressJEPA.forward` averages the query over all target 8x8
    cells before forming `fine_logits`; consequently fine sub-cell selection
    may vary by horizon/camera/source slot but not by the target cell doing the
    read.  Its bias-free value projection is also initialized with
    `std=1e-3`, then passes through the fixed 0.10 update scale next to an
    unchanged full future carrier.  The KL can train relevance logits while
    the predictive value route remains structurally optional.  The repair
    keeps target-cell-specific fine queries, uses chunking rather than a
    premature mean to bound memory, and gives the zero-preserving value
    projection a variance-preserving initialization.  It must not copy frozen
    teacher values or create a new action input.
37. `M37 the supervised interval-stage increment has no provenance-preserving
    W->P value lane` (V106 static integration defect; V107 repair implemented,
    runtime smoke pending).
    W1-W3 deltas are captured before `_IntervalStageDeltaOrganizer` runs.  The
    interval increment changes the coarse rollout and therefore can weakly
    perturb the W->P query, late-detail query, and generic bottom rollout, but
    it is absent from `world_bridge_values`.  Thus its signed progression is
    supervised without becoming a separately selectable typed W->P source;
    the supplied V106 run correspondingly reports interval write/carrier near
    0.001.  The repair appends the bounded interval increment as an explicit
    W->P typed candidate while retaining its spatial coarse write and keeping
    it outside the protected fine-value bank.
38. `M38 the role residual safety contract is applied before, rather than at,
    the gated write boundary` (V104-V106 static/numerical defect; V107 repair
    implemented, runtime smoke pending).  Each active G/W/P sublayer currently computes
    `gate * smooth_rms_contract(raw)`.  A partially closed gate encourages the
    optimizer to drive the raw branch deep into cap saturation in an attempt
    to recover write amplitude, although the gate still prevents recovery.
    This explains a coherent V106 state in which raw role residual RMS is
    about 4.8 while the bounded residual stays near 0.49.  The safety contract
    must instead receive the actual gated proposal and its output must be the
    tensor written to the carrier.  The upper bound and ordinary autograd are
    unchanged; flags-off arithmetic remains available for matched ablation.
39. `M39 aggregate role-residual telemetry hides the block and sublayer that
    owns saturation` (V104-V106 acceptance defect; V107 repair implemented,
    runtime smoke pending).
    Current logging averages raw/bounded/compression metrics across all eight
    role blocks before exposing them.  It cannot distinguish a G alignment
    failure from W world growth, P policy growth, or one self/visual/dynamics/
    FFN branch.  The repaired path reports the actual gated proposal and final
    write RMS per role block and sublayer, plus compact G/W/P maxima.  These
    values remain audit-only: no loss, gradient scaling, route quota, or
    automatic gate is derived from them.
40. `M40 the V105-V107 horizon address is supervised after the deployed action
    has already been decoded` (V105-V107 structural integration defect; V108
    repair implemented, runtime smoke pending).  The role stack first
    completes G/W/P and decodes the action, then
    `predict_future_with_address()` rereads the final rollout only for the JEPA
    prediction head.  Ordinary deployment with diagnostics disabled skips this
    read.  Its representation loss and combined-loss gradient can therefore
    train the address owner without proving that the selected evidence ever
    participates in the deployed action.  V108 has four indivisible repair
    tasks: (1) execute the horizon-address organization exactly once after G3
    has compiled/refined the observation-owned bank and before W1, then write
    its bounded residual into the existing `[horizon,camera,xy]` rollout;
    (2) preserve the existing rollout decoder path and strict raw/DINO/teacher
    ownership--there is no new decoder input, direct raw-to-action lane, or
    target-dependent value; (3) make the future predictor consume that same
    final rollout without a second late address read; and (4) place the whole
    topology behind one serialized V108 flag whose disabled branch exactly
    reproduces V107.  Structural execution must never depend on
    `collect_diagnostics`.  The repair must be accepted by an action-only
    backward to horizon-address query/key/value owners, teacher invariance,
    diagnostics-on/off action parity, exact-zero value behavior, V108 flag-off
    V107 equality, and matched online-address zero/episode-shuffle action
    interventions.  Loss weights, write caps, flow training, language/history,
    interval organization, execution control, and the bottom decoder remain
    unchanged in this version.

Verified non-issues:

- The +48 future anchor is not incorrectly forced into the 24-step action
  partition. `flow_jepa_action_offsets` explicitly keeps only 4/12/24. In the
  repaired typed path, every W->P source also retains +48 as a separate
  horizon-constant, camera-specific far-context candidate. The action
  time/basis query may select it, but it is never relabelled as action step 24.
- Goal and action history are not statically disconnected. Attached direct
  history tokens and the attached causal proposal both reach the deployed
  action path. The proposal retains its own auxiliary action objective, but it
  is no longer restricted to that objective in V103.
- The Evidence action field has no target-action posterior leak. At fixed
  `x_1`, time, observation, goal, history, and proposal, changing
  `target_action` leaves the predicted velocity and reconstructed action
  bitwise unchanged. A one-step training reconstruction also matches a
  one-step deployment integration from the same action noise. Disabling
  auxiliary layer diagnostics at deployment therefore does not alter the
  V103 action graph.
- Future DINO teacher features are target-only. At fixed current observation
  and action input, changing the future teacher changes the JEPA target but
  leaves the predicted action velocity and reconstruction bitwise unchanged.
- A second active-graph detach audit found no remaining hidden V103 action
  cut. `latent_cvae_transition_detach=0` and
  `action_history_proposal_detach=0` keep both transition evidence and the
  causal proposal attached. The detached `rollout_seed` is diagnostic-only;
  the detached `trajectory_seed` constructs the legacy workspace that is not
  consumed when the typed P-to-MMDiT bank is active. Native execution argmax
  sites belong to explicit hard-evaluation/audit views; the V103 training and
  default soft-deployment state uses continuous candidate probabilities.
- Formal loss ownership is one-way: labelled actions supervise the physical
  flow field, decoded-action geometry, proposal/event/motion readouts, and the
  detached execution-value target; future visual frames supervise only the
  frozen-DINO JEPA target. Neither labelled source is present in the reusable
  observation bank or clean condition memory. Current RGB/DINO, state,
  executed history, proposal, goal, and stateless phase are the deployed
  conditions.
- The bottom rollout/transition and terminal P contracts do not reopen the
  historical high-resolution visual bypass. In strict V103, raw visual
  selector/value tensors are not passed to the final Evidence adapter and
  visual intent is removed. Policy blocks can read but cannot write the coarse
  rollout chart, while protected high-resolution detail is written only into
  trajectory and then passed through its typed bottom lane. Terminal P layer
  states may select evidence, but their value rows are rebuilt from clean
  intent memory. The remaining coarse bottom-rollout path is therefore an
  intentional complementary route, not a duplicate raw-value route; +48
  typed-only, bottom-only, and joint interventions remain required to determine
  whether either route is ignored or compensates for the other.
- The hard top-k in `_structured_mask` selects context-dropout locations; it is
  not a hard action/address router. The defect is the late masking boundary,
  not a missing gradient through mask indices.
- A negative uncertainty NLL is mathematically possible because of
  `error / sigma + log(sigma)` and is bounded by the configured uncertainty
  floor. It can obscure loss-budget interpretation but is not the source of
  the representation disconnect.

Repair order:

1. Preserve several spatial hypotheses and their uncertainty through a clean,
   observation-only multi-resolution address lattice (`M01`, `M02`).
2. Make G/W/P depth deltas real typed values and replace the fixed P
   superhighway with a residual/provenance-preserving bottom bridge
   (`M05`, `M06`).
3. Move the final precision and camera choice to horizon/basis queries without
   losing protected raw detail (`M07`, `M08`).
4. Add exact goal-null semantics and a stateless soft phase belief, then use
   them only as address/world/policy queries (`M09`, `M10`).
5. Move masking to a pre-mixing trainable representation boundary and make
    predictive change useful to action through the repaired path
6. Restore coarse spatial write capacity to W1-W3 while keeping raw
   high-frequency values behind the late address reader (`M17`).
7. Preserve the W chart through the late precision selector instead of
   reducing it to an anchor/camera mean before address selection (`M18`).
8. Keep G1-G3 free of candidate-action values; introduce action-conditioned
   consequence organization at W1 rather than inside grounding (`M19`).
9. Couple the predictive JEPA online context mask and target mask without
   allowing future-teacher-derived mask patterns into the forward pass
   (`M20`).
10. Let source/target raw correspondence correct the fine soft posterior while
    preserving target-only high-frequency value ownership (`M21`).

Implementation checkpoint (2026-07-26):

- `flow_jepa_soft_address_lattice=1` keeps multiple same-camera coarse slots
  and continuous fine raw candidates until a horizon/basis policy query. The
  observation bank is compiled once and reused across ODE steps; raw values
  remain narrow until after the fine weighted read.
- Each fine candidate carries a source/target raw pair key. Source sub-cell
  features, target candidate features, their channelwise interaction, and
  their difference are encoded as selector evidence. Changing only source raw
  appearance changes fine keys and receives ordinary address-gradient, while
  the target high-pass value tensor remains exactly unchanged.
- `role_attnres_*` records real per-block deltas. Grounding and world values
  preserve `[anchor,camera]`; policy values preserve `[time,basis]`.
- V103 sets `flow_jepa_world_anchor_write_only=0`. W1-W3 retain the complete
  coarse 8x8 rollout state and may write spatially varying prediction
  residuals. The G->W and W->P AttnRes values remain typed
  `[anchor,camera]` summaries; they select depth/role provenance and do not
  replace the persistent spatial world state.
- The late soft-lattice reader aligns that persistent W chart to the
  4/12/24 action bands without averaging xy. A bias-free world-key projection
  contributes local camera/xy compatibility to the normalized address
  posterior; DINO/flow/raw keys still select slot and fine offset, and only
  observation-owned high-frequency raw values are read. A mean-preserving
  spatial permutation of W therefore changes the precision read, and action
  loss reaches the non-uniform part of the W chart through ordinary autograd.
- In the directed 3-3-2 canvas, G rollout queries cannot read `proposal` or
  noisy `trajectory`, and those sources are absent from the G dynamics and
  AdaLN modulation contexts. G still reads the observed visual bank plus
  goal/state/history context. W retains proposal/trajectory access and is the
  explicit action-conditioned consequence boundary. An exact unit check
  changes only proposal/noisy action and verifies that G rollout is bitwise
  invariant while W rollout changes.
- G->W and W->P bridges use low-rank continuous routing over full-width delta
  values. The cumulative role carrier stays outside softmax. The G->W bridge
  write is broadcast over xy because its typed value is an
  `[anchor,camera]` ownership summary; W1-W3 themselves retain and update the
  complete coarse xy chart. W->P keeps the 4/12/24 action-aligned candidates
  and the +48 far-context candidates distinct by source and camera; the latter
  are expanded only as context for each action query, not as an action-time
  target.
- Predictive-change masking also owns the late precision bank. Before the
  source/target raw pair and native DINO charts enter the soft-address
  compiler, the same latest-context mask replaces raw features and removes
  DINO cells. The compiler therefore cannot restore hidden appearance after
  the G boundary. Observation-only flow geometry remains available by design;
  no future-teacher value or teacher-derived mask pattern enters the bank.
- The conservative G->W and W->P fixed scales apply only to writes into the
  shared top carrier. Typed values crossing the next ownership boundary retain
  the routed delta before that carrier scale. Thus the bottom bank applies its
  one common P->MMDiT scale to G/W/P evidence; it does not compound
  `0.10 * 0.10 * 0.25` for G while applying only `0.25` to P.
- `role_attnres_policy_to_mmdit=1` removes the normalized
  `sqrt(0.5)` policy-workspace fusion. The bottom action seed remains direct,
  policy-approved world/P deltas are selected per action horizon, and the
  already scaled high-resolution detail update is a separate protected
  addition outside the depth route.
- No route entropy/balance loss, hard selection, detach-and-repair gradient, or
  learned amplitude gate was added. Local full-policy forward/backward tests
  reach G->W, W->P, P->MMDiT and all G/W/P block groups; every new parameter has
  exactly one optimizer owner. These checks prove connectivity and ownership,
  not task benefit; zero/shuffle interventions and a full validation run are
  still required.
- `goal_condition_exact_null=1` and
  `action_history_condition_exact_null=1` express condition content as
  `f(x)-f(0)` before the explicit keep mask. Dropped conditions therefore
  contain no biased content template.
- `stateless_phase_enabled=1` produces a soft ordered phase belief plus a
  separate goal/history selector context. Both modulate each W block's
  organization query and the W->P/late-detail selector queries; neither is an
  observation value or direct action writer.
- `flow_jepa_predictive_change_contract=1` masks raw RGB in image coordinates
  before any trainable cross-cell mixing and uses that chart instead of the
  direct latest-DINO value seed. The future head emits
  `future_teacher-current_teacher`; absolute scene reconstruction is no longer
  the primary objective.
- Under the same predictive contract, one observation-only spatial mask is
  shared by +4/+12/+24/+48, written into the latest online context mask, and
  retained as the training loss mask. `flow_jepa_teacher_balanced_target_mask`
  must be zero because selecting a different mask from future-teacher change
  after the forward would leave supervised cells visible; feeding that
  teacher-selected mask back into the forward would instead leak future-change
  locations. In evaluation/sample mode the context mask is empty so the action
  path never loses current observation detail.
- The same contract requires a positive
  `flow_jepa_address_flow_prior_floor`. Its floor expert has a fixed width of
  roughly one coarse cell, so learned uncertainty/occlusion cannot flatten the
  nominal nonzero coefficient into a spatially constant prior. A separate
  learned expert may widen under ambiguity. Flow remains a soft geometric
  expert, not a hard selector, but its minimum spatial contribution cannot be
  learned down to zero. Full-policy backward reaches both the early masked RGB
  encoder and the soft address compiler from the shared JEPA/action graph.
- Exact unit checks prove that changing only hidden RGB pixels leaves the early
  context bitwise unchanged, and that a correct delta receives near-zero loss
  while a zero delta receives nonzero gradient. These are structural checks;
  causal flow/goal/phase interventions still decide empirical usefulness.
- `scripts/run_v103_model_path_probe.sh` now provides one frozen-checkpoint,
  matched-noise causal audit for the active V103 graph. It covers exact goal
  null, goal batch permutation, full-history null/permutation/truncation,
  condition-only history null, proposal-only null/permutation,
  phase/condition queries, flow zero/episode/spatial mismatch, target-DINO key
  mismatch, source-raw pair-key zero/spatial mismatch, raw-value zero/spatial
  mismatch, a joint flow/DINO/source-raw key mismatch, joint
  address/fine/camera posterior interventions, camera-bank swap, isolated
  W-spatial-query zero/shuffle, every typed G/W/P candidate, the independent
  +48 far-context bank, the protected detail candidate, and the aggregate
  policy/world residual controls. +48 is tested
  three ways: typed W->P far only, full bottom-rollout far only, and both
  together. A null single-route result therefore cannot hide compensation by
  the other legal far-evidence path.
- The probe records the tensor delta at the intervened boundary, compact
  address/fine-posterior and final-detail deltas relative to the same-batch
  baseline, final action delta, validation error change by horizon, and an
  episode-cluster bootstrap interval. Its transient state is not serialized;
  unit tests verify that clearing every intervention exactly restores the
  ordinary forward and leaves the model state dictionary unchanged.
- Probe v3 emits a factual acceptance matrix: replay identity, boundary
  accessibility, action reachability, paired utility direction, and direct
  action differences between joint +48 removal and each single +48 route.
  Nonzero action change is never promoted to utility without the paired error
  interval. Full-history interventions intentionally include their
  history-derived proposal route. The condition-only history null retains the
  proposal and isolates the attached direct-history lane; the proposal-only
  modes retain the direct compressed history memory and either remove proposal
  content at its exact keep boundary or episode-mismatch only its future
  tokens. Separate acceptance fields therefore report direct-history and
  proposal action reachability instead of allowing one route to hide the
  other.
- The `--model-path-intervention-batches` entry now rejects a merely
  shape-compatible or partially enabled checkpoint. It requires the serialized
  3/3/2 typed graph, predictive shared-mask contract, source/raw soft lattice,
  exact goal/history nulls, stateless phase, attached transition path, positive
  primary JEPA loss, and zero duplicate/legacy rollout objectives. A successful
  result records `complete_v103_contract_verified=true` in both the probe and
  run identity; the schema name alone is not accepted as proof. Historical
  `--action-path-intervention-batches` remains available for old V101-style
  compatibility probes and does not claim V103 identity.
- The formal V103 launcher exports
  `CLEARVLA_REQUIRED_MODEL_CONTRACT=v103`; the training entry validates the
  resolved policy and trainer objects before optimization. This catches
  inherited argument-order regressions, one-slot pseudo-lattices, zero-scale
  bridges, detached layer/transition paths, missing observable-flow objectives,
  and legacy duplicate objectives. The same validator is reused by the frozen
  checkpoint probe, so training and evaluation cannot silently assign V103 to
  different graphs.
- Strict V103 Evidence mode executes only the two terminal P-layer adapters
  consumed by the final decoder. Historical mid-cut heads, G/W diagnostic
  layer probes, disabled P rollout-transition sublayers, and other
  compatibility-only branches are frozen and excluded from optimization.
  After one ordinary optimizer update, a complete V103 objective gives a
  nonzero gradient to every remaining trainable parameter; this establishes
  loss-graph ownership, not empirical task utility.
- With the execution schedule fully open and after one ordinary update, the
  primary action field reaches every remaining trainable parameter except the
  exact declared auxiliary/deployed-readout set: proposal action regression,
  JEPA future prediction, event classification, and motion classification
  heads. The complete objective reaches those heads through their respective
  real targets. Any future parameter outside this exact set that is absent
  from the action field is a contract failure.
- A complete CPU BF16-autocast V103 pass, including raw/Flow-DINO, predictive
  JEPA, typed 3/3/2 bridges, goal/history/phase conditioning, native Evidence
  execution, and the formal total loss, produces BF16 model predictions, an
  FP32 finite scalar loss, and finite gradients for every trainable parameter.
- The history proposal reads only past executed actions and is available in
  exactly the same form at deployment. V103 therefore keeps it attached to the
  final action loss; history/proposal dropout, rather than a gradient cut,
  provides conditional robustness. Static action-gradient reachability is not
  treated as empirical use: the frozen-checkpoint v3 probe must show a changed
  proposal boundary, a changed deployed action, and then use the paired
  episode-bootstrap error interval to decide whether the learned proposal is
  helpful, harmful, or inconclusive.
- A batch permutation is not a semantic goal-mismatch experiment when every
  sample uses the same cached T5 condition. In the current single-task data it
  may correctly report zero goal-input delta. A real goal-mismatch conclusion
  still requires per-sample alternative goal tokens or a multi-goal
  validation set; the probe does not manufacture a fake language effect.
- These additions make the causal questions measurable but do not constitute
  empirical evidence that V103 uses the repaired paths. That claim still
  requires a trained V103 checkpoint and the probe result.

V104 structural checkpoint (2026-07-26):

- V103 is not silently redefined. The three new configuration fields default
  to zero, the V103 launcher still requires the complete V103 contract, and
  `scripts/current_v104_sequential_bounded_flow_jepa.sh` is the only formal
  launcher that enables all three repairs and requires the V104 validator.
- `flow_jepa_bounded_flow_coordinates=1` applies the source-relative smooth
  chart to the initial DINO proposal, every recurrent DINO update, and the
  final mid/high raw proposals. It reports mid/high boundary compression and
  the normalized motion-evidence magnitude. Address geometry, warp metrics,
  and raw flow magnitude retain their native grid-unit meanings.
- `role_residual_amplitude_contract=1` applies a per-token soft RMS contract to
  self-attention, visual cross-attention, stage, stage-to-window, rollout, and
  FFN writes before each legal role mask/addition. The same contract is applied
  to full-width Delta AttnRes values at every G/W/P/bottom ownership boundary.
  Raw RMS, written RMS, and compression are logged separately; compression is
  diagnostic and is not an objective or learned gate.
- `flow_jepa_sequential_horizon_memory=1` constructs the observed perceptual
  history memory and performs the four ordered transitions. It reports history
  entropy/latest-pair mass, per-horizon update/state change, and adjacent-query
  cosine. These measurements distinguish "transition exists" from "all four
  horizons still became the same representation."
- The formal V104 validator first requires the entire V103 graph, then requires
  all three V104 flags and positive residual/value RMS limits. The CLI records
  these fields in serialized run context, hashes the V104 launcher and new
  routing source, and the compact logger identifies V103 and V104 by active
  structural fields instead of falling back to a V102 label.
- `scripts/run_v104_model_path_probe.sh` reuses the matched V103 causal
  intervention matrix but requires the complete V104 checkpoint contract and
  emits a distinct V104 schema/verification field. A partial V104 ablation is
  deliberately not named as the formal V104 model.
- Static unit coverage verifies in-image coordinates and finite gradients under
  extreme proposals, source-resolution normalization, bounded but
  direction-preserving role writes, chronological history influence, attached
  transition gradients, complete formal forward construction, BF16
  forward/backward, launcher inheritance, and V103 flags-off compatibility.

What V104 structurally resolves:

- flow cannot make most of its warp supervision disappear by moving its own
  sampling coordinates outside the image;
- grid resolution and an exploding pixel displacement cannot directly inflate
  learned motion keys/values;
- role carriers and typed values cannot grow without a normalized-chart
  amplitude limit and numerically drown the protected detail lane;
- +48 now has an actual chronological, observation-history-dependent
  computation rather than only a larger embedding on the same latest-motion
  seed;
- the new contracts have distinct serialized identity and factual diagnostics.

What V104 does not prove or manufacture:

- that the dataset contains enough observable, task-relevant motion for learned
  flow to improve action;
- that a single cached T5 condition provides identifiable language variation;
- that stateless phase beliefs align with semantic task stages;
- that address, raw detail, world, goal, phase, history, or +48 causally improve
  validation action error;
- that validation plateau, generalization, or deployment latency improves.

Those are trained-checkpoint questions. They require matched interventions,
per-horizon validation, representation-boundary deltas, and episode-cluster
uncertainty. V104 removes structural excuses and records the required evidence;
it deliberately does not add route quotas, gradient balancing losses, entropy
targets, artificial motion, or teacher leakage to force a positive result.

V105 horizon-address checkpoint (2026-07-27):

- V104 is not silently redefined. `flow_jepa_horizon_soft_address=0`,
  `flow_jepa_future_reliable_normalization=0`, and address loss weight zero
  preserve the V104 graph/objective. The formal launcher is
  `scripts/current_v105_horizon_addressed_flow_jepa.sh`, and the V105 validator
  first requires the complete V104 contract.
- Compact logs call a run V105 only when the horizon reader, reliable
  normalization, and positive address supervision are all active; partial
  combinations remain labelled V104 ablations.
- `_HorizonSoftAddressJEPA` consumes only the final W rollout chart and the
  already compiled observation bank. Fine-offset selection is factorized by
  horizon/camera to bound memory; coarse routing remains target-cell specific
  over every same-camera source cell and slot. Continuous high-resolution
  candidates and raw values are not pooled before this read.
- The 8x8 target chart is only the query lattice. It may attend across source
  cells and continuous raw coordinates, so the mechanism does not assume
  one-to-one, equal-area, or equal-pixel source/future correspondence.
- The read is smoothly RMS-bounded and added with one fixed small coefficient
  to the future-prediction input. It never mutates rollout, trajectory,
  protected policy detail, bottom Evidence MMDiT, CVAE/workspace, adaptive
  layer selection, or execution control.
- The raw-value normalization has no learnable affine term and the value
  projection has no bias. An all-zero observation detail bank therefore gives
  an exactly zero update; the branch cannot replace its address read with a
  learned constant future residual.
- Normal action deployment skips the auxiliary horizon read when diagnostics
  are disabled. Training and representation validation execute it. Therefore
  V105 does not add an unused high-resolution reader to ordinary action
  sampling or turn the auxiliary future head into an action owner.
- Address supervision is a reliability-weighted KL from frozen
  `future_teacher-current_teacher` spatial strength to the forward-produced
  relevance logits. Teacher tensors are constructed after policy forward and
  are absent from every address-reader argument. The predicted distribution
  is never used to weight its own future prediction loss, preventing an
  easy-cell self-routing shortcut.
- The predictive delta objective is
  `raw Smooth-L1 + reliability-weighted normalized Smooth-L1 + 0.1 *
  reliability-weighted direction`. All terms use the existing
  observation-only shared target mask. The reliable normalization scale is
  `sqrt(delta_rms^2 + (0.05 * current_teacher_rms)^2)`: unlike normalization
  by the delta itself, it does not cancel when the entire teacher change
  becomes weak. Reliability is detached, continuous, has no hard motion
  threshold, and is divided by the valid-position count rather than its own
  sum.
- Required diagnostics include raw delta loss, reliable normalized loss,
  current-reference/delta/normalization scales, change reliability, address
  KL/teacher reliability/teacher and predicted entropy, route/fine entropy and
  maxima, applied update RMS/ratio, horizon-posterior variation, cross-cell
  distance, per-horizon scale and reliability, and the horizon-address
  gradient norm.

What V105 structurally resolves:

- each real future horizon can request a different spatial precision read
  after G/W organization rather than inheriting one undifferentiated mask;
- non-equal source/future spatial correspondence remains representable through
  cross-cell, multi-slot, continuous-coordinate attention;
- address mass cannot disappear through a close gate, while relevance is not
  forced to be sharp or to allocate a hand-written quota;
- weak normalized-future jitter is attenuated without deleting the raw static
  anchor;
- future-teacher supervision cannot leak into action or the online address
  bank.

What V105 still does not prove:

- that predicted horizon addresses match task-relevant motion on this dataset;
- that the soft read improves DINO prediction, action generalization, or +48
  causal use;
- that learned flow itself is accurate rather than being corrected mainly by
  DINO/raw keys;
- that language or stateless phase is identifiable in a single-goal dataset.

Those remain empirical questions. Judge them from trained per-horizon logs and
matched interventions, not from a nonzero gradient or a sharper posterior.

V106 interval-stage decision ledger (2026-07-27, implemented candidate; not retroactive to
V104/V105 checkpoints):

- Replace the semantic meaning of four isolated future points with four
  contiguous temporal supports whose interiors do not overlap and whose
  boundary observations are shared: `H4=[4,8]`, `H12=[8,16]`,
  `H24=[16,32]`, and `H48=[32,48]`. The names remain compact horizon labels;
  they no longer mean supervision by exactly frame 4/12/24/48.
- A window target preserves both interval content and progression. A plain
  frame mean is forbidden because it can erase direction and short events.
  The teacher keeps camera and 8x8 cell ownership and uses three temporal
  quantities at each cell: a robust trapezoid-supported content summary, a
  signed least-squares progression over real frame offsets, and an explicit
  endpoint increment. The content target remains the primary future-chart
  target. A separate progression prediction is supervised by the signed
  target plus a weak endpoint-consistency term.
- The four online queries are still formed chronologically from observable
  history, learned flow, DINO/raw evidence, and G/W organization. No target
  frame, teacher-derived mask, or labelled future action may enter their
  forward construction.
- The interval organizer extends the existing W chart instead of creating a
  second stage tower. It performs causal horizon-only organization separately
  for every camera/xy cell, emits a smoothly bounded near-zero-initialized
  delta, and applies one fixed small write at the W->P boundary. The written
  delta is a selector/coarse-world carrier: it is visible to W->P, the late
  address query, P1/P2, and the legal bottom rollout route.
- Stage information and precision values remain orthogonal interfaces. Stage
  deltas may change where and when the soft lattice reads, but cannot become
  candidate raw values, pre-average fine candidates, select a hard crop, or
  compete with protected detail for survival. Disabling the interval-stage
  flag restores the V105 carrier and point-target behavior; disabling the
  interval, variance-safe-routing, and complete-numerics flags restores the
  exact V105 normalization path as well.
- V106 replaces post-cancellation LayerNorm in the active horizon address and
  role-depth routing sites with zero-preserving, variance-floored selector
  normalization. Address values use only an upper-bound smooth RMS contract;
  small evidence is never expanded to unit variance. Required diagnostics are
  pre-contract RMS, channel standard deviation, applied contraction, update
  RMS, per-boundary selector scale, and preclip gradients.
- The complete numerical sub-contract extends this rule to the interfaces
  missed by the initial V106 candidate. Learned correlation uses an RMS floor
  of `0.10`, which changes nominal RMS-one cosine features by less than one
  percent and limits inverse-RMS gain to 10. G/W/P and address routing use a
  centered variance floor of `0.25`; non-affine gain is at most 4 and the
  smoothly bounded affine scale makes the complete role-normalization gain at
  most 16. Online visibility crosses the unchanged local consistency
  threshold over 10 percent of that detached threshold. Since the threshold
  has a `0.5` floor, its derivative with respect to squared cycle error is at
  most 5. These are coordinate-system contracts, not observed-value clipping.
- Old V104/V105 checkpoint probes must report their actual
  `point_anchors=(4,12,24,48)` semantics. They may diagnose sampling-path or
  call-contract mismatch but cannot be presented as evidence for the planned
  interval target.

V106 implementation and acceptance boundaries:

- New fields default off. With the interval-stage, variance-safe-routing, and
  complete-numerics fields off, the serialized graph and objective remain
  V105.
- Required interval supports are the real dataset offsets
  `(4,8,12,16,20,24,28,32,36,40,44,48)` and the boundaries are
  `(4,8,16,32,48)`. Cached and uncached target loaders must return the same
  ordered support.
- The teacher projection remains frozen and no-grad. Future supports may be
  loaded only for target construction and must be absent from every online
  module argument.
- The same observation-only spatial mask is reused for content, progression,
  endpoint, and address supervision. No future-change-derived mask is allowed.
- The interval loss is registered once in the representation ledger. Raw
  progression and endpoint diagnostics are not additional objectives unless
  their explicit weight is positive.
- Training and representation-validation logs retain per-interval loss,
  reliability, and actual write RMS as compact
  `interval_h4/h12/h24/h48=l:/r:/w:` fields, so a healthy mean cannot hide one
  collapsed temporal range.
- A near-zero nonzero address-value stress test, a cancelled multi-candidate
  stress test, and a complete G->W->P->MMDiT backward must remain finite and
  bounded before training.
- The formal V106 validator requires the complete V105 parent, interval
  supports/boundaries, positive interval loss weight, variance-safe address
  and role routing, the complete numerical contract, floors no smaller than
  `0.25/0.10/0.10` for role/correlation/visibility respectively, and positive
  fixed write scales. It rejects partial graphs before the optimizer is built.
- The supervised progression tensor is the exact bounded delta written to the
  W->P carrier before multiplication by the fixed update scale. There is no
  independent trainable progression readout that can fit the teacher while
  leaving the deployed write near zero.
- Interval-duration encodings participate only in causal query/key selection.
  The attention value and both downstream projections are bias-free and read
  the observable W chart, so an exact-zero W chart produces an exact-zero
  stage write instead of a learnable horizon-only template.
- Signed-direction supervision uses a frozen teacher-scale denominator floor,
  not `F.normalize` at the near-zero-initialized prediction. This keeps the
  loss derivative bounded at zero and logs
  `flow_jepa_interval_stage_direction_floor_min`. The inherited
  predictive-change direction term uses the same V106-only rule and logs
  `flow_jepa_future_direction_floor_min`; V105 flags-off arithmetic remains
  unchanged.
- Implementation owners are
  `clearvla/policy/flow_dino_evidence.py` (teacher, safe address, causal
  organizer), `clearvla/policy/trunk.py` (W->P insertion),
  `clearvla/policy/role_delta_attnres.py` and
  `clearvla/policy/time_domain_mmdit.py` (safe typed routing),
  `clearvla/policy/system.py` (teacher-only pack), and
  `clearvla/experiments/observed_state_lab/policy_runtime_v39.py`
  (objective, ledger, gradients, logs, validator).
- The formal launcher is
  `scripts/current_v106_interval_stage_flow_jepa.sh`; the matched checkpoint
  entry is `scripts/run_v106_model_path_probe.sh`. The V106 probe adds matched
  zero and episode-shuffle interventions at the exact bounded W->P stage-write
  boundary. The V105 wrapper now preserves an already-selected child contract
  instead of overwriting V106.
- Static verification completed: all modified Python/test files parse and
  compile, launcher/probe scripts pass Bash syntax validation, and source
  checks cover target scheduling, exact loss accounting, serialized contract
  selection, and model-path schema recognition. PyTorch/pytest are not
  installed in this Windows inspection environment, so tensor/backward tests
  remain pending for the server smoke rather than being reported as passed.

V106 two-round static audit ledger:

- Round 1, ownership/integration: found and removed an independent trainable
  progression readout, registered interval loss and target scheduling in both
  Stage1/main ledgers, based log identity on forward-observed flags, and fixed
  the V105 wrapper so it cannot overwrite a requested V106 contract.
- Round 2, shortcut/numerics/reverse path: followed launcher -> serialized
  config -> support loader -> frozen teacher -> online W organizer -> W->P
  bridge -> precision query/P blocks -> bottom action/loss. It replaced the
  remaining V106 future-prediction LayerNorm with variance-floored
  normalization, made interval values evidence-only and bias-free, removed
  zero-vector cosine singularities from both the inherited predictive-change
  and new interval-direction terms, and added exact-boundary zero/shuffle
  interventions. No target teacher, detach, hard route, fine-bank rewrite, or
  second loss contribution was found on the resulting online path.
- Post-run numerical audit round 1, activation and threshold ownership:
  followed the batch-600/620/640 state transition through learned semantic and
  raw correlations, forward/backward cycle visibility, occlusion-conditioned
  address sigma/metadata/motion keys, and every G/W/P internal normalization.
  It separated inactive semantic-only/legacy readers from the actual raw V106
  path and found no schedule event at the onset.
- Post-run numerical audit round 2, reverse gradient and flags-off behavior:
  traced action/JEPA gradients back through P/W/G and the observation bank,
  placed finite-gain contracts before each amplification rather than after the
  residual, retained hard visibility only as a detached diagnostic, preserved
  old code in explicit disabled branches, serialized every new threshold, and
  added correlation, visibility, near-constant role-block, validator, launcher,
  and compact-log contract tests.

V107 complete top-path repair ledger (2026-07-27, implemented candidate; not
retroactive to V106 checkpoints):

- `flow_jepa_policy_multi_glimpse_address=1` makes the protected P-side
  precision reader genuinely multi-glimpse. Each glimpse owns its query and
  fine/coarse/camera/slot posterior, reads the observation-owned value bank,
  and passes through a narrow bias-free value projection before the glimpse
  channels are concatenated. Candidate modes can no longer cancel before the
  first nonlinear policy projection. The reader remains entirely soft and an
  exact-zero value bank still produces an exact-zero policy update.
- `flow_jepa_horizon_cell_fine_address=1` preserves the target 8x8 cell axis
  while selecting fine source locations in the auxiliary horizon reader.
  Target cells are evaluated in bounded chunks rather than averaged into one
  global fine query. The bias-free value owner uses variance-preserving
  initialization instead of the V106 `1e-3` initialization; zero input still
  maps to zero, and future teacher values never enter the online reader.
- `flow_jepa_interval_stage_typed_value=1` appends the actual bounded
  post-intervention interval-stage increment as a named W->P value source.
  The source preserves horizon/camera/channel provenance, participates in the
  existing typed W->P selection, and does not enter or duplicate the protected
  raw/DINO precision bank. Interval zero/shuffle intervention is applied once
  at the owned write boundary, not again inside the generic role-value probe.
- `role_residual_contract_after_gate=1` changes the active arithmetic from
  `gate * contract(raw)` to `contract(gate * raw)` for every G/W/P self,
  visual, dynamics, and FFN write. Thus the finite write contract now bounds
  the tensor actually added to the carrier and no longer rewards a raw branch
  for driving into saturation behind a partly closed gate. The disabled branch
  retains exact V106 arithmetic for matched ablation.
- Role telemetry now exposes raw, gated-proposal, bounded, and written RMS by
  role block and sublayer, plus compact G/W/P written maxima and means. Policy
  logs expose glimpse count/route variation, target-cell fine-address status,
  and actual interval-stage W->P source mass. These diagnostics are audit-only.
- The complete V107 validator requires the complete V106 parent plus all four
  repairs, at least two policy glimpses, and a positive target-cell chunk size.
  The launcher and serialized run context cannot silently downgrade the child
  contract: `current_v106_interval_stage_flow_jepa.sh` now preserves an
  already-selected `FLOW_JEPA_PARENT_VERSION` and
  `CLEARVLA_REQUIRED_MODEL_CONTRACT`.
- Formal entries are `scripts/current_v107_complete_top_path_flow_jepa.sh` and
  `scripts/run_v107_model_path_probe.sh`. The matched probe schema is
  `clearvla-v107-model-path-intervention-v6`; automatic evaluation recognizes
  V107 only when the complete serialized contract is present.

V107 three-round review record:

- Round 1, top-down ownership and shortcut audit: followed raw/DINO/flow banks
  through G1-G3, W1-W3, interval organization, typed W->P selection, P1-P2,
  late precision read, and the retained bottom Evidence MMDiT/CVAE/workspace.
  It recorded M35-M39 before editing. No future target, labelled future action,
  teacher feature, pooled high-frequency duplicate, hard address, detach patch,
  or second action decoder was found on the repaired online path.
- Round 2, amplitude and numerical audit: checked every new projection for
  bias and initialization, exact-zero behavior, candidate/camera/cell axes,
  and the legal residual write boundary. A complete CPU forward/backward had
  finite G/W/P written maxima `0.02072/0.02564/0.02628`, below the configured
  `0.5` cap; the actual interval source was selectable with observed camera-0
  mass `0.04763`. These numbers are implementation stress-test evidence, not
  trained-model quality claims.
- Round 3, reverse reachability, compatibility, and observability audit:
  action-only loss produced finite nonzero gradients in the multi-glimpse
  query, interval organizer output, and W->P query route. Combined action and
  representation losses also reached the target-cell horizon query/value
  owners. Targeted inherited V103/V105/V106 tests and new V107 contract,
  zero-value, cell-identity, post-gate-bound, launcher, log-schema, and
  intervention-schema checks pass in local PyTorch execution. One inherited
  `1e-7` exact-equality assertion differs by `2.842e-14` under local PyTorch
  2.11 centering arithmetic; it is unrelated to the V107 branches and is not
  treated as a model failure. Full CUDA smoke and trained interventions remain
  required before empirical acceptance.

V108 online horizon-address topology plan (2026-07-27, implementation record;
not retroactive to V107 checkpoints):

- The single owned address read is at the G3->W1 boundary.  Its query is the
  chronological horizon rollout produced by the existing online path; its
  keys/values are the observation-only bank compiled/refined by G1-G3.  It
  preserves horizon, camera, and target-cell identity and returns a bounded,
  differentiable residual on the existing rollout carrier.
- W1-W3, interval organization, typed W->P selection, P1-P2, the protected
  late-detail lane, and the retained bottom Evidence MMDiT/CVAE/workspace all
  consume the ordinary carrier exactly as before.  V108 adds no action-decoder
  argument and grants no raw/DINO/teacher tensor a new downstream ownership.
- The JEPA future head consumes the final post-P rollout and reuses the online
  address telemetry saved at G3->W1.  It must not invoke a second address
  selection or create two independently optimized versions of the future
  carrier.
- `collect_diagnostics` controls only metric materialization.  The V108 online
  address computation is part of both training and deployment whenever its
  serialized topology flag is enabled.  Disabling the flag retains the exact
  V107 late auxiliary behavior for checkpoint-compatible matched comparison.
- Boundary telemetry records seed, post-G3, post-address, post-W1/W2/W3, and
  post-interval horizon similarity plus address-delta RMS and downstream
  retention.  These values are audit-only and must not become a balancing,
  entropy, route-mass, or gradient-shaping objective.
- Static/runtime acceptance is bidirectional: a forward ownership audit must
  show one observation-bank read and no target/bypass; an action-only backward
  must reach the online address owners without representation loss; zero and
  episode-shuffle interventions must alter the owned boundary and report their
  deployed-action effect.  Teacher invariance, diagnostics parity, exact-zero
  values, flags-off V107 equality, finite BF16/FP32 routing, and multi-step
  deployment reuse are mandatory.

V108 implementation and two-round static review (2026-07-27):

- Round 1, forward ownership/call/loss review: the only online call is
  `organize_horizon_address()` after the G3 refinement boundary and before W1.
  It writes the existing rollout slice.  The V108 future head calls only
  `predict_future(final_rollout)`; `predict_future_with_address()` remains only
  in the disabled-flag V107 branch.  The saved online logits have one existing
  horizon-address loss owner, and no decoder argument, teacher-conditioned
  value, duplicate loss, or raw-to-action bypass was added.  Config, run
  context, resume validation, required-contract validation, logging, and the
  paired probe all serialize the same single flag.
- Round 2, reverse gradient/numerical/compatibility review: the addressed
  rollout reaches the bottom decoder without detach; the historical
  transition-memory detach does not wrap that direct carrier.  Query/key/value
  routing uses the inherited variance floors, the bias-free zero-preserving
  value path, and the inherited `0.10 * 0.50 = 0.05` per-token write-RMS upper
  bound.  Audit-only boundary reductions are skipped when diagnostics are
  disabled, while the address write itself is not.  The V107 flag-off branch
  allocates no new parameters and retains its old late auxiliary call.
- Acceptance tests now cover action-only gradients through address query,
  compiled keys, and value projection; future-teacher invariance; exact-zero
  observation values; diagnostics-on/off multi-step action equality and one
  address read per model evaluation; optimizer one-owner coverage; V107
  flag-off topology; launcher inheritance; logs; and zero/shuffle probe schema.
  Python compilation, shell syntax, and whitespace checks pass locally.  The
  local Windows environment has neither PyTorch nor pytest, so tensor/backward
  execution is explicitly unverified here and remains a required CUDA smoke
  gate rather than an inferred success.

V108 early trained diagnosis and V109 design premise (2026-07-27; this is the
authoritative replacement for treating the single G3->W1 read as the intended
final design):

- The V108 batch-220 run proves topology and gradient reachability, but not
  useful information bandwidth.  The horizon carrier adjacent cosine changes
  from about `0.906` at the seed to `0.981` after G3, remains `0.981` after the
  address write, and reaches `0.995` after W3.  The online write is about
  `0.002` RMS (`~0.001` of its carrier), while its route entropy is about
  `0.968`.  V108 therefore moves the inherited V105 auxiliary read into the
  deployed path without making grounding progressively own address formation.
- V108 remains a reproducible diagnostic baseline.  V109 is an opt-in
  child whose flag-off branch is numerically and structurally V108.  It must
  not silently reinterpret a V108 checkpoint or remove the old single-read
  topology before matched comparison is possible.
- Deterministic coordinate scaffolding precedes all grounding blocks.  It
  establishes continuous source/current coordinates, scale transforms,
  camera identity, validity, and candidate geometry; it performs no learned
  camera fusion and no final address selection.
- G1 owns candidate-hypothesis establishment.  It updates per-camera,
  per-cell, per-slot coarse semantic/correspondence evidence over the complete
  same-camera chart.  Flow may contribute a soft prior, but G1 never reduces
  multiple slots to one expected point or combines camera values.
- G2 owns geometric rectification.  It consumes the G1 posterior plus learned
  flow, raw/DINO correlation, cycle consistency, occlusion, and uncertainty;
  it updates posterior geometry, variance/support, validity, and continuous
  residual offsets while preserving every camera/slot/candidate identity.
  Because a complete-chart G1 update may move a mode outside the old
  compiler-centred neighbourhood, the V109 bank retains the dense projected
  source/target raw keys, target DINO keys, and narrow target-detail chart.
  G2 differentiably rematerializes candidate keys and values around its
  corrected centres.  This is candidate construction, not value aggregation;
  otherwise G1/G2 could report changed coordinates while P still read only
  the old location.
- G3 owns canonical compilation and handoff, not value readout.  It combines
  the typed G1/G2 increments into a clean observation-owned address basis and
  emits both the complete candidate bank and typed grounding summaries.  It
  must not soft-average high-resolution values into the rollout carrier.
- Horizon-specific selection begins in W, not in the clean bank.  W1/W2/W3
  use chronological JEPA/history/goal/phase queries to form near/mid/far
  posterior deltas over the G3 basis.  The observation keys/values stay
  horizon independent; horizon posterior state stays query owned and is never
  cached as observation data.  At the W3->P1 boundary, one compatibility
  tensor over `[horizon,camera,W-cell,G3-source-cell,slot]` produces two
  marginals: a target-cell chart for the existing teacher-only relevance loss
  and a bounded source-cell/slot prior consumed by the P reader.  Computing the
  teacher chart after P, or logging it without feeding the paired source prior
  to P, is not an implementation of this contract.
- P1/P2 own the first final high-resolution value read.  Camera relevance,
  spatial modes, fine candidates, action horizon/basis, goal, and phase remain
  available until this read.  The result enters the existing protected detail
  and typed policy paths; no pooled visual duplicate may bypass it.
  The G2-rematerialized candidate values remain unaggregated and retain their
  full camera/source-cell/slot/fine axes until this boundary.
- Each G block performs a distinct state transition rather than repeating the
  same reader.  The persistent address state contains at minimum coarse
  logits, mode center/covariance, fine support/offset, validity/uncertainty,
  and typed depth deltas.  G1/G2/G3 updates remain individually intervenable
  and observable.
- The V105 outer relevance KL remains a weak, teacher-only diagnostic/objective
  for changed target regions; it is not sufficient supervision for
  source-cell/slot/fine correspondence.  V109 must not respond by imposing a
  fixed entropy target, hard selector, forced route mass, artificial nonzero
  flow, or gradient patch.  Geometry consistency, JEPA prediction, and action
  utility reach the progressive posterior through ordinary forward edges.
- The inherited fixed `0.10` auxiliary address scale is not promoted into a
  main-path bandwidth fix.  Residual bounds protect actual writes, but useful
  contrast must be formed before amplitude.  Broad/common-mode reads must not
  be made influential merely by increasing their scale.
- Required acceptance is stage specific: G1 intervention changes coarse
  hypothesis evidence; G2 intervention changes geometry/support; G3
  intervention changes the canonical basis/summary; W interventions change
  horizon posterior without modifying observation values; the joint P read
  changes deployed action on relevant samples.  Boundary change without
  action change is attenuation/compensation evidence, not a pass.
- V109 source implementation status: the pre-G bank retains dense charts; G1,
  G2, and G3 advance one typed mutable selector state after their corresponding
  blocks; G2 rematerializes rather than reusing old-centre fine candidates; W
  forms both posterior marginals before P; and P performs the sole final raw
  value aggregation.  The progressive graph does not instantiate the retired
  `_HorizonSoftAddressJEPA` or the two compiler-only projections that would be
  trainable dead branches.  The launcher, v8 checkpoint probe, serialized
  contract, compact/epoch logs, gradient groups, and zero/shuffle interventions
  are wired.  Python syntax and three static connection reviews pass locally;
  this Windows environment lacks PyTorch/pytest, so CUDA forward/backward and
  flags-off numerical identity remain mandatory server smoke checks.

V109 completed-log audit and V110 base selection (2026-07-28):

- The supplied/local V104, V106 and V107 logs contain all eight completed
  epochs. V109 contains three completed epochs and then a delayed instability.
  The action/upstream gradient rise is already visible around epoch 3 / batch
  1020; raw-flow cycle, occlusion and other upstream representation metrics
  deteriorate later. Therefore the raw branch participates in the eventual
  failure, but the logs do not support claiming it is the initiating cause.
- V109's progressive selectors remain the best current ownership skeleton,
  but its P-side detail ratio weakens and the W posterior remains diffuse. The
  older V98 zero/shuffle checkpoint probe changed action by only about
  `0.000699` RMSE and found no significant action gain, establishing that a
  represented address can still be downstream-ignored.
- V110 consequently does not copy one checkpoint wholesale. It retains
  V109's progressive G1/G2/G3/W/P state topology, restores V107's demonstrated
  finite-gain numerical rules at sensitive boundaries, removes V109's divided
  edge limiter, and replaces the learned-feature-only P read by typed current
  RGB/detail plus a spatially consequential future transport.
- A higher learning rate/longer V109 schedule may change when instability is
  observed, but it cannot explain the static shared-scorer,
  learned-feature-only value ownership, or transport-as-optional-condition
  defects. Those are source-level reasons for V110 and must not be deferred to
  a scheduler-only experiment.

V110 multi-round static review record (2026-07-28):

- Verified the 3/3/2 boundary hooks run before the next block: `index=4` is the
  post-G3 G->W entry and `index=7` is the post-W3 W->P entry. There is no
  off-by-one P contamination in the W posterior.
- Removed the pre-P RGB area-resize. `dense_current_rgb` keeps the input image
  side and is sampled only by normalized current-anchor coordinates; the
  learned-detail chart may remain lower resolution.
- Replaced the single fused G3 summary with separate semantic, appearance and
  geometry summary-token families. W attention may organize them, but G3 no
  longer destroys type identity at the boundary.
- Restored the typed G2 rectifier to four consumed channels: x/y correction,
  support and geometric-prior strength. A per-output gradient assertion guards
  this contract. The earlier three-output edit would have indexed past the
  tensor at runtime.
- Removed a second addition of the G1 aligned semantic key in G3. Dynamic G2
  semantic candidates already contain G1 coarse plus local DINO information;
  the duplicate silently biased semantic amplitude over appearance/geometry.
- Distinguished W->P ingress A/B from the actual policy DiT blocks P1/P2. The
  ingress selects and locally refines the 3x3 values once; P1/P2 then organize
  trajectory state without reopening the observation bank.
- Restricted `flow_jepa_raw_micro_grid` validation to the V110 flag so a new
  inactive field cannot narrow the V109 flags-off configuration domain.
- Added separate natural-gradient diagnostics for the P-ingress selector,
  literal RGB value projection, learned-detail value projection, conditional
  local organizer and aggregate refiner. No auxiliary/forced gradient was
  introduced.
- The first CUDA BF16 preflight reached G2 and exposed a mixed-dtype boundary:
  a Float32 typed query projection was called with a BF16 activation inside an
  explicit `autocast(enabled=False)` numerical island. Auditing the complete
  path found the same latent defect at W and P. G2/W/P learned projections and
  their adjacent key normalization now execute in the surrounding autocast
  domain; only their projected activations enter FP32 similarity, geometry and
  probability arithmetic. This is a dtype-boundary repair, not an architectural
  or gradient-path change. The full typed G/W/P test now runs under CPU BF16
  autocast so this class of error is covered before another CUDA launch.
- The first full training epoch exposed a memory implementation defect at the
  typed P ingress. With the formal batch-8 shape, the original code created a
  full candidate-by-3x3 micro posterior and then separate RGB/detail/coordinate
  state-by-micro tensors before applying the coarse route. That is about 266
  MiB of explicit dominant materialization per four-query chunk and leaves 24
  chunk graphs for backward. The repaired contraction keeps the exact same
  route, 49 fine candidates, nine micro cells and typed values, but streams one
  micro cell at a time, contracts candidates before coarse states, and uses the
  existing raw activation-checkpoint contract to recompute each chunk during
  backward. Typed logit families are accumulated sequentially while retaining
  their separate scalar diagnostics. A small-tensor reference test compares
  both outputs and natural gradients against the former fully materialized
  formula.
- Targeted `py_compile`, trailing-space and `git diff --check` reviews pass.
  Nine related V109/V110 test functions pass under the local PyTorch 2.11
  environment, including CPU BF16, the full G1/G2/G3->W->P path, exact
  zero-value ownership, flags-off ancestry, output equivalence and gradient
  equivalence. A formal-shape CUDA micro-read benchmark reduced the local
  forward peak delta from 380.1 MiB to 91.2 MiB (76.0%) with maximum output
  difference `8.94e-08`; the small reference gradient maximum difference was
  `1.19e-07`. These are subpath measurements, not a claim that complete-system
  CUDA memory falls by the same percentage. The server smoke and causal
  checkpoint probe remain required.

Frozen sampling-path probe contract (implemented for the retained V104
checkpoint):

- `scripts/run_v104_sampling_path_probe.sh` holds the validation sample,
  initial physical noise, visual context, language/history conditions, and
  checkpoint fixed.
- At the five deployment times it separately measures: the
  teacher-conditioned training call versus the deploy call on the same exact
  flow-matching bridge; the deploy call on the bridge versus the recursively
  reached state; and the recursive/bridge state distance.
- The same noise is then integrated with 5/10/20 steps. Improvement from more
  steps is solver/discretization evidence; a deploy-bridge penalty is
  call-contract evidence; an additional recursive penalty is off-path
  exposure/recovery evidence. Signed excess MSE may be negative under
  compensation and must not be converted into an absolute "problem score."
- The probe also reports old-checkpoint point-anchor prediction error at each
  matched action-flow time, including H48, so a far-horizon degradation can be
  localized without claiming that V104 was trained on interval targets.

Each repair must keep the strict no-bypass boundary, preserve the flags-off
reference, and pass a static review for candidate count, camera identity,
detach sites, dtype, and action-gradient reachability before activation.

## 4. Dataset motion evidence that the architecture must preserve

The radius-bounded DINO dataset probe is an oracle/headroom measurement, not a
learned-flow evaluation.

For motion-top patches, the local oracle reduces zero-flow warp error by
approximately:

```text
delta 4:  22.9%
delta 12: 31.7%
delta 24: 31.3%
delta 48: 28.6%
```

The corresponding simple soft matcher is poor at short offsets and reaches
about 13–15% gain at longer offsets. This means:

- useful nonzero correspondence exists;
- a single diffuse/global soft expectation is insufficient;
- multimodal local candidates and uncertainty must survive;
- static background must not set the whole-image solution;
- camera relevance depends on time and task.

At delta 4, motion-top oracle gain is about 6.7% for the top camera and 28.7%
for the wrist camera. Fixed equal camera fusion therefore discards useful
conditional choice.

Motion-top displacement at delta 4 on the 8×8 chart is approximately:

```text
mean 0.77 cells
p90  1.41 cells
p95  2.00 cells
```

The mean becomes roughly 4.5 cells on a 42×42 chart. A radius-2 mid refiner
cannot reliably recover it when the coarse seed remains near zero.

## 5. Target module interfaces

### 5.1 Clean observation bank

Inputs:

- raw 84/42-scale value maps;
- native/coarse DINO keys and values;
- forward/backward flow proposal;
- confidence, uncertainty, occlusion, and cycle evidence;
- camera and continuous coordinate identity.

Excludes:

- noisy action;
- proposal/future action tokens;
- future teacher features;
- goal/phase content as values.

It may be cached across ODE steps because it is observation-only.

### 5.2 Goal tokens

- Preserve several T5-derived tokens rather than only one global mean.
- Keep a clean global goal path for broad task identity.
- Also expose typed goal queries to semantic address organization.
- Goal should influence what is read, not rewrite the raw observation bank.
- Support real per-sample goal tokens in the training batch API.
- Represent missing language with an explicit learned null condition and apply
  dropout to the post-resampler content delta so dropout semantics are exact.

### 5.3 Stateless phase-belief adapter

Implemented behind `stateless_phase_enabled=1`; empirical validation remains.

Inputs may include:

- Goal Tokens;
- current state and state history;
- compressed action history;
- current visual summary;
- JEPA horizon-change tokens;
- flow/motion/confidence summaries.

Outputs:

- several soft Phase Belief Tokens;
- uncertainty/confidence;
- optional time-scale-specific query deltas.

Constraints:

- no recurrent hidden state is required;
- no hard phase classification is required;
- it must support ambiguity between plausible phases;
- no extra phase label is required initially;
- action and JEPA gradients train it naturally;
- it conditions W/P/address queries through explicit typed residuals;
- zero/shuffle intervention must reach address and action.

### 5.4 Soft multi-resolution address lattice

For every query cell and camera:

1. form multiple coarse address slots over the complete same-camera DINO chart;
2. add a soft flow-derived geometry prior, never a hard mask;
3. retain posterior mean, covariance/uncertainty, and multiple modes;
4. construct continuous fine candidates around each mode;
5. scale fine search support using uncertainty and resolution conversion;
6. permit overlapping and cross-cell support;
7. read small-channel raw values first, then project to hidden width;
8. preserve camera identity until the policy chooses camera relevance.

The spatial posterior is over camera × slot × xy. Camera combination must be
query-conditioned and normalized, not a fixed equal sum and not an
unconstrained amplitude gate.

### 5.5 Explicit 3-3-2 responsibilities

The eight blocks remain connected through one persistent main state.

```text
G1: cross-camera and cross-scale alignment
G2: flow rectification and multi-hypothesis update
G3: canonical coordinates and observation-bank address compilation

W1: near-horizon scene/change organization
W2: intermediate-horizon consequence organization
W3: far-horizon/phase-conditioned organization

P1: horizon- and basis-specific high-resolution read
P2: action-ready precision residual and terminal policy workspace
```

These are functional contracts, not eight different toy block classes. Blocks
retain full-width attention/FFN capacity and exchange typed deltas. W blocks
may update the coarse spatial chart; they are not restricted to an
anchor/camera-constant broadcast. High-resolution raw residuals remain outside
that chart until the P-boundary address read.

### 5.6 Delta AttnRes

Use role-constrained low-rank residual selection to preserve useful depth
increments:

```text
G-depth deltas -> G/W address queries
W-depth deltas -> P horizon/detail queries
P/workspace deltas -> bottom Evidence MMDiT
```

AttnRes selects depth/role information. The address lattice selects spatial
camera/slot/xy information. Do not merge these into one opaque router.

Begin as a bridge on top of the existing main path. Consider replacement only
after intervention evidence proves the bridge is useful.

### 5.7 Bottom action interface

Keep:

- direct noisy-action state;
- clean global goal/intent condition;
- bottom Evidence MMDiT;
- existing lower evidence and execution machinery.

Add:

- typed world/address/detail evidence;
- protected policy precision residual;
- explicit goal/phase-conditioned query deltas.

The bottom must retain an easy path for easy samples while still being able to
respond causally to detail on precision-critical samples.

## 6. Required causal evidence

Every representation intervention reports both the intended boundary delta
and the final deployed-action delta.

Required matched probes:

```text
goal zero / goal mismatch / goal episode-shuffle
action-history zero / shuffle / truncate
phase-belief zero / shuffle
interval-stage write zero / episode-shuffle
flow zero / episode-shuffle / spatial-shuffle
DINO key shuffle
source-raw pair key zero / spatial-shuffle
joint flow + DINO + source-raw key spatial-shuffle
raw value zero / spatial-shuffle
camera posterior uniform / camera swap
world address query zero / spatial-shuffle
G1/G2/G3 delta zero / shuffle
W1/W2/W3 delta zero / shuffle
W->P +48 zero/shuffle; bottom-rollout +48 zero/shuffle; both together
address posterior uniform
fine offset zero
late detail zero / spatial-shuffle
```

Report:

- representation delta at the intervened interface;
- address-posterior delta;
- final detail delta;
- action delta RMSE;
- validation MSE/RMSE change;
- horizon bands;
- arm and gripper separately;
- event precision/recall/F1 and event counts;
- episode-cluster bootstrap interval.

A representation change with a null action change means downstream filtering
or compensation. It is not evidence that the source is useful.

Model acceptance gates after training:

1. Probe validity: matched baseline replay is numerically identical, every
   requested boundary reports a nonzero intervention delta and positive apply
   count, and clearing transient state restores the ordinary forward.
2. Spatial accessibility: flow/DINO/raw/posterior interventions must move the
   intended address or detail boundary. If they do not, the address path is
   still structurally or numerically collapsed regardless of validation RMSE.
3. Action use: at least the aggregate spatial/detail interventions must produce
   a non-null deployed-action delta. Individual redundant experts may be
   compensated, so interpret their single-route result together with joint
   interventions rather than demanding every ablation be independently large.
4. Long-horizon use: the joint +48 intervention must be distinguishable from
   both its typed-only and bottom-rollout-only interventions. Report action
   changes and error changes by horizon band; do not infer far-horizon use from
   a representation delta alone.
5. Utility: action change is not automatically benefit. A path is helpful only
   when its matched zero/shuffle ablation worsens error with an
   episode-cluster interval that does not support a practically null effect, or
   when an agreed precision-critical subset shows the same directional result.
6. Conditioning: goal/history/phase boundary deltas must reach action. A
   single-task batch with identical cached T5 tokens cannot validate semantic
   goal mismatch and is reported as an evidence limitation, not a model pass.

## 7. Implementation order

The historical V103 migration used the following phased order. V104 keeps its
three repair flags independent for ablation, and V105 adds one independently
switchable horizon-address/read-loss contract on top of the complete V104
parent:

1. Preserve a flags-off V102 reference and golden forward.
2. Add observation-bank shape/cache/dtype tests.
3. Run the soft lattice in shadow mode without changing action.
4. Replace only the late raw read and run causal probes.
5. Add query-conditioned camera selection.
6. Add G1/G2/G3 clean address stream.
7. Add the stateless phase-belief adapter.
8. Add goal/phase conditioning to W/P/address queries.
9. Add W1/W2/W3 horizon queries.
10. Add G→W and W→P Delta AttnRes.
11. Add the protected typed bridge to bottom Evidence MMDiT.
12. Remove an old reader only after validation, intervention, ODE replay,
    memory, and speed checks pass.

Do not infer that one aggregate V104 result identifies which repair mattered.
Keep the independent flags for matched ablations after the complete structural
path passes smoke validation.

## 8. Static and runtime acceptance criteria

- No future-teacher input leakage.
- No action -> observation bank -> action echo.
- No hidden detach on an active condition path.
- No fixed patch ID masquerading as a continuous address.
- No camera identity loss before the final read.
- No premature mean over address slots or high-resolution candidates.
- No hard selection or hand-written gradient compensation.
- FP32 address/correlation logits under BF16 model execution.
- No `inf`/`NaN` for invalid candidates.
- Cache and non-cache outputs match.
- ODE multi-step sampling retains the observation-only bank correctly.
- Disabling the V105 horizon-address/reliable-loss options reproduces V104;
  disabling only the three V104 repair flags then reproduces V103; disabling
  the earlier V103 feature set retains its separately tested V102 parent path.
- Every learned flow proposal is finite and maps source coordinates inside its
  current chart; boundary compression is visible and never used as a loss.
- Motion K/V receives normalized displacement/velocity, while address and warp
  geometry retain native units.
- Every active role sublayer and typed AttnRes boundary reports raw and written
  RMS; the written RMS respects its configured smooth contract.
- +4/+12/+24/+48 are computed in chronological order from attached perceptual
  history, without future-teacher or labelled-action input.
- Every active horizon address is produced before teacher attachment from W
  queries plus the observation-only bank, retains same-camera identity and
  continuous fine candidates, and normalizes to unit spatial mass without a
  hard selector or learned close gate.
- Replacing every fine value by zero produces an exact zero future-address
  residual; no affine normalization or projection bias may synthesize a
  constant substitute.
- Reliable normalized future terms are divided by valid-position count, not
  reliability mass; their smooth current-teacher-relative scale floor prevents
  weak changes from cancelling their own attenuation, while raw delta
  supervision remains active on static cells.
- Under the V105-V107 parent contract, ordinary action sampling with
  diagnostics disabled does not execute the auxiliary horizon reader.  Under
  V108, the single G3->W1 online address write executes independently of
  diagnostics but still creates no new action input/output boundary; disabling
  the V108 topology flag exactly restores the V107 behavior.
- Peak memory remains bounded using low-rank keys, small-channel raw values,
  chunked camera/slot/horizon reads, and activation checkpointing.
- Policy gradient may remain larger than world/detail gradient, but causal
  action sensitivity from world/detail must be measurable where the
  corresponding evidence is relevant.

## 9. Memory and cache hygiene

This contract stores conclusions, not artifacts.

Do not embed or duplicate:

- `.pt`, `.pth`, `.ckpt`, optimizer state, or model tensors;
- DINO/raw decoded caches;
- run directories;
- full nohup logs;
- full JSON probe outputs;
- images/contact sheets;
- repeated source excerpts.

For a new result, append only:

- artifact identity/path outside this document;
- run contract and source fingerprint;
- the few decision-changing statistics;
- the architectural conclusion;
- the next falsifiable probe.

Keep one current contract. Move superseded detail into an existing research
document instead of creating `v2`, `v3`, and conversation-specific copies.

## 10. Detailed references

Read only when the task requires implementation detail:

- `docs/research/soft_multires_address_lattice.md`
- `docs/research/attnres_332_migration_plan.md`

The present file overrides conflicting older prose. Source and serialized run
context override all documents when describing what an existing experiment
actually executed.
