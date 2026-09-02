# Schema25 base version card

Status: read-only base audit complete on 2026-08-26; disposition `BASE`.

This card closes the source/log reconstruction required before the first donor
review. It does **not** claim that Schema25 is architecturally closed, does not
authorize a source checkout or model edit, and does not promote this historical
graph into the checked-out mainline contract. Schema25 is the fixed comparison
base because it is the earliest locally recoverable graph with a complete run
and relatively little accumulated hardening, not because every inherited
operator is correct.

The replay procedure and future unit dispositions remain owned by
`ARCHITECTURE_REPLAY_PLAN.md`.

## 1. Base verdict

```text
Source base:       6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6
Historical parent:32d969fccecca35a641382888f6f4681221c9c24
Commit title:      fix: preserve S object ownership in schema 25
Behavior record:   schema25_s_owned_typed_b8.log
Disposition:       BASE
Next donor:        Schema26 / caa7e3315e85e5f4119fe3174e86037b47a5903c
```

Schema25 is suitable as the replay base for four source-backed reasons:

1. its online and training planes are type-separated and Teacher is absent from
   deployment;
2. S owns typed relevance once, preserves `[interval,K,type]` for W, and gives
   CoarseAction a separately named typed-action aggregate;
3. the complete eight-epoch log shows a trainable end-to-end graph, the best
   local physical/gripper point in the recovery island, nontrivial object and
   interval differentiation, and no fatal numerical event;
4. later changes such as shared type/time posteriors, stronger dustbin
   authority, bilateral isolation, per-lane bottom repair and terminal
   reconstruction have not yet accumulated.

The base remains structurally open. In particular, Schema25 still contains
fixed averaging and attenuation, repeated typed influence, a joint P2
interval-object-null posterior, complementary value competition, dynamic/static
P1 mixing, reconstructed transition interval identity, and an inherited joint
P3-lane/basis/null bottom route. Those are recorded as debts rather than silently
accepted as replay invariants.

## 2. Evidence anchors and confidence

| Evidence | Exact anchor | What it establishes | Limitation |
|---|---|---|---|
| Source | commit `6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6` | Exact serialized source tree, manifest, config, interfaces, losses, optimizer ownership, runtime and checkpoint rules | It does not prove that the copied console log was produced from this exact tree |
| Historical delta | `32d969f..6a6c1bf` | What Schema25 itself changed relative to corrected Schema24 | The parent already contains several V120 fidelity corrections inherited by the base |
| Behavior | repository-root `schema25_s_owned_typed_b8.log` | One uninterrupted eight-epoch run, 22,768 updates, 1,136 logged training windows, full epoch validation and selected ablations | The copied console log has no `run_context`, source digest, normalizer fingerprint, checkpoint or RNG manifest |
| Historical contract | `6a6c1bf:docs/research/00_CURRENT_ARCHITECTURE_CONTRACT.md` | Contemporary intended ownership, parameter inventory and known transition-anchor debt | Intent is secondary to the executable source |
| Historical repair plan | `6a6c1bf:docs/research/CURRENT_MAINLINE_REPAIR_PLAN.md` | Original S-ownership defect and intended repair boundary | It is not a reason to preserve every chosen operator |

The log-to-source association is strong but not bit-exact: the log names
`object_intent_dynamics_323`, writes to `runs/schema25_s_owned_typed_b8`, uses
the Schema25 metric vocabulary, reports BF16/five-step/Teacher-isolated startup,
and contains only one launcher/preflight header with no resume header. Because
no source identity was serialized, source claims and behavior claims remain
separate throughout this card.

The active log path is also easy to confuse with a same-named historical
capability package. The run header comes from `scripts/train_mainline.sh`, whose
entry point is `clearvla.mainline.train`. Its executed policy graph is therefore
under `clearvla/mainline/model/`. Files under
`clearvla/policy/object_intent_dynamics_323/` belong to a different historical
launcher path and are not evidence for this mainline run unless an active import
explicitly reaches them.

## 3. Identity and launch contract

The source manifest at `6a6c1bf` is:

```text
schema:       25
capability:   object_intent_dynamics_323
topology:     G1/G2/G3, W1/W2, P1/P2/P3
observation:  restored_v120_three_frame_flow_dino_progressive_g123_bank
top:          v120_progressive_g123_dense_grounder_exact_p1_s_owned_k_typed_relevance_four_interval_w_five_lane_p3
bottom:       restored_v120_shared_seed_dynamic_p1_p1_p2_contracts_evidence_mmdit_dense512_execution
training:     v120_mirrored_physical_flow_exact_teacher_current_support_event_boost_v120_decay_local_global_clip
runtime:      cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated
```

The formal configuration fixes:

```text
current visual history:     -8, -4, 0
cameras / grid:             C=2 / 8x8
local hypotheses / objects: M=4 / K=4 plus null
future intervals:           I=4: [4,8], [8,16], [16,32], [32,48]
future supports:            offsets 4,8,...,48 (12 supports)
action horizon / bases:     T=24 / Q=4
hidden / DINO / route:      H=512 / D=768 / R=32
language:                   4096-wide T5, at most 32 tokens, required
runtime:                    BF16, five Euler updates
training:                   batch 8, eight epochs, AdamW, base LR 8e-5
parameters:                 169,981,895 total / 153,587,574 trainable
```

Schema24 and older checkpoints cannot exact-resume this graph. Schema25 changes
the model and state-dict fingerprints and requires a fresh run unless a complete
Schema25 checkpoint identity matches.

## 4. What Schema25 actually changed

The historical parent is corrected Schema24 commit `32d969f`, titled
`fix: correct V120 geometry and validation diagnostics`. The Schema25 commit
changes 18 mainline/docs/test files, with `+1,158/-340` in that scoped diff.
Its semantic model delta is narrower than the raw file count.

### 4.1 Unit S25-A: one S-owned typed relevance field

Disposition: `BASE`; ownership invariant protected, exact gating formula open.

Schema24 had three typed `_CrossRead`s and a shared learned-null
`RoleDeltaAttnRes` inside S. Schema25 replaces them with:

```text
public interval carrier                         [B,I,H]
per-type route queries                          [B,I,type,R]
G semantic/appearance/geometry routes           [B,K,type,R]
bounded dot relevance score                     [B,I,K,type]
sigmoid signal coefficient against zero baseline[B,I,K,type,1]
typed relevance value                           [B,I,K,type,R]
typed action components                         [B,I,type,H]
policy interval = public + typed context        [B,I,H]
```

Each type has an independent multiplicative coefficient against a fixed-zero
baseline. Types and objects do not compete in a shared softmax at this S
boundary, and there is no learned null value. Physical object validity
multiplies relevance mass; an invalid or algebraically zero route therefore
stays exactly zero and no selected-mass renormalization recreates it. The
bounded sigmoid coefficient is nevertheless strictly positive for a valid
nonzero route and cannot choose literal zero. That residual floor is a base
mechanic, not proof of complete optionality.

The implementation also uses a `0.25` norm floor, temperature constrained to
`[0.25,4]`, a fixed mean across K, `/sqrt(3)` type aggregation and two `0.35`
smooth RMS contracts. Those are base mechanics, not protected truths.

### 4.2 Unit S25-B: remove downstream raw typed rereads

Disposition: `BASE`; single-owner invariant protected, current consumer algebra
open.

CoarseAction drops three typed `_CrossRead`s and its learned-null router. It
reads S's public carrier, public object memory, observable history and
`typed_action_context` through an `ActionIntentDock`.

W drops its learned-null router. It reads public object content and transport,
S's public interval and protected goal, CoarseAction tokens, and S's
`[I,K,type]` relevance values through a `WorldIntentDock`. It no longer selects
raw semantic/appearance/geometry facts independently of S.

P1/P2/P3 receive named restricted docks rather than the monolithic S state, but their
internal algorithms are unchanged by Schema25. P2 receives a
`PolicyIntentDock`; P1 receives a `FactualIntentDock`.

### 4.3 Unit S25-C: typed interfaces and observability

Disposition: `BASE`.

`ObjectIntentState` is split into public, action, world, factual and policy
docks. Diagnostics separately report public/policy interval variation,
per-type mass/null, object variation, interval variation and W contribution.
The interface split is useful because it makes later freedom loss observable;
the metric names themselves are not architecture.

### 4.4 Exact parameter delta

Relative to corrected Schema24, trainable parameters decrease by
`12,731,133`:

| Owner | Change |
|---|---:|
| S: remove three typed cross-reads and one shared router; add three route projections and temperatures | `-6,308,093` |
| CoarseAction: remove three typed cross-reads and one router | `-6,357,248` |
| W: remove one router | `-65,792` |

Bottom and exact P1 are unchanged by the Schema25 semantic delta.

### 4.5 Inherited corrected-Schema24 facts

The following belong to the fixed base but are not Schema25 S changes:

- active V120 pre-G/address/future-query parameters are trainable; only the
  unconsumed generic G3 route query remains frozen;
- Teacher displacement and covariance moments are formed inside each camera
  before object reduction;
- the global-K binder does not add the same public chart to every private
  candidate key;
- validation sampling/proposal/execution probes have separate coverage and
  matched physical-noise streams.

They must be evaluated as inherited units if a donor touches them; the Schema25
commit title is not evidence that they were introduced here.

## 5. Complete forward boundary

Axis key used below:

```text
B batch, C=2 cameras, M=4 local hypotheses, K=4 objects, I=4 intervals
L=3 typed roles, T=24 action steps, Q=4 action bases, H=512 hidden
D=768 full DINO width, R=32 typed route width, Z=512 transition rows
```

| Owner/boundary | Producer and transformations | Output / zero and scale semantics | Consumers and frequency |
|---|---|---|---|
| Online input | Raw RGB and DINO at `-8,-4,0`; current/state history; eight executed actions; required T5 goal | Future supports, target action/state and audit metadata are different interface types and cannot enter online policy input | Observation, S and shared seed; once per observation |
| Observation + progressive G | V120 raw/DINO compiler, two learned adjacent flows, G1 update, G2 `N=49` rematerialization, G3 bounded owner correction | Completed `[B,C,8,8,M,*]` chart; G is current-only and uses a clean static endpoint; default online path is BF16 with selected score/reduction work in FP32 | Dense global grounder and static P1; once per observation |
| Dense global grounder | Competitive K+null owner posterior followed by object-specific reads; local prior and physical validity remain outside the conditional owner softmax | `ObjectFactSet`: content `[B,K,D]`, typed routes `[B,K,R]`, camera coordinates/transport/support/validity, reversible object-to-chart assignments and one reconstruction scalar | S, W, Teacher target construction, static transition source and reconstruction loss; once |
| S | T5 protected goal, observable state/action history and current ObjectFactSet create public intervals; S-owned typed signal-vs-zero relevance creates `[B,I,K,L,R]`; policy interval adds typed context | Public carrier is future-recognizer supervised; the null baseline has no learned value and invalid/zero routes stay zero, but a valid sigmoid gate is strictly positive; state-change evidence is current-observable; noisy action, Teacher and future target are forbidden | CoarseAction, W, static P1 docks, dynamic P2/P3 docks; once |
| History module | Eight executed actions become three summary plus four recent history rows and a separate 24-step action prediction | The 24-step prediction is auxiliary; the seven compressed history rows are a real deployed condition and are not auxiliary. `proposal_condition_dropout` is sampled/logged but never applied; deployed history uses the separate action-history keep mask | Compressed rows enter the shared seed/bottom every dynamic call; prediction receives proposal loss only |
| CoarseAction | Public S intervals, public object memory, S typed context and observable history produce four action-intent tokens and a seven-dimensional interval action prediction | No raw typed reread; online target/loss placeholder is zero; training target comes from future action | W once; separate supervised coarse loss |
| W1/W2 | Current content/transport, public S interval, CoarseAction, goal and S-owned typed relevance form W base; W1 owns near intervals and W2 reads W1 to produce far intervals | `FutureObjectDynamics [B,I,K,*]`: successor, semantic delta, transport/covariance, visibility/persistence/uncertainty/reliability and selector validity. Current reference is detached. Zero-init effect heads coexist with positive softplus covariance/uncertainty floors and initial visibility probability `0.5` | P2 and future-dynamics loss; once per observation |
| Teacher/recognizer plane | No-grad Teacher associates current typed object facts with 12 future DINO supports using semantic, appearance, camera/geometry and one null candidate; it runs with autocast disabled in FP32. Recognizer reads future action/state and detached Teacher effect | Teacher target stays `[B,I,K,*]`; null content/address falls back once to the current fact; per-camera displacement moments precede object reduction | W/recognizer/coarse/intent losses only; once per training batch, zero times in deployment |
| Static P1 | Clean `[T,Q]` bases query the completed progressive chart using S public phase, protected goal and last observable-history context | Cached protected detail `[B,T,Q,H]`; P1 is the sole high-resolution visual read | Dynamic P1 at every velocity call; built once |
| Dynamic P1 | Shared noisy-action query plus cached protected detail passes one compact V120 policy block | Completed P1 fact `[B,T,Q,H] = protected_detail + dynamic_delta`; static and dynamic writes share one factual identity in Schema25 | P2, P3, transition layer contracts and bottom; every dynamic call |
| P2 + consequence | Completed P1 trajectory queries a joint `[I,K]` W field with content, S-policy-intent and coordinate scores plus predicted selector validity; one extra null competes. Semantic, transport and status values then compete through a three-way type softmax. Caller applies a `0.35` RMS contract | P2 effect `[B,T,Q,H]`; zero effect makes consequence exactly P1 through `P1 + effect + interaction` | P3, controlled transition and protected bottom path; every dynamic call |
| P3 | Factual, precision, effect, temporal and state-change lanes are derived from P1/consequence/S/noisy action; each lane is contracted at `0.35`, while state-change also has a fixed `0.05` multiplier | Five optional `[B,T,Q,H]` lanes plus protected P1+P2 consequence | V120 bottom role bank and layer contracts; every dynamic call |
| Controlled transition | Static source repeats the public G3 `[C,8,8]` chart under four learned interval identities to make `Z=512` selectors. Dynamic V120 real-minus-learned-neutral coefficients read all 96 post-P1/P2 action tokens, current state/history and plan | Selector/value `[B,512,H]`; no spatial pooling before bottom, but the interval axis is reconstructed rather than inherited from G3 | Bottom and event context; static source once, dynamic value every velocity call |
| V120 bottom | Protected consequence has a separate per-basis, no-null reader. The five P3 lanes become `5*Q=20` candidates in one `RoleDeltaAttnRes` with one null, `0.35` value contract, `0.25` normalization floor and fixed `0.25` write scale. Generic trajectory is exact zero. Full transition rows, state/history, layer contracts and shared noisy seed enter the restored organizer/Evidence-MMDiT | Three MMDiT blocks, ordered low-rank capacity/execution control, physical velocity, event logits and motion logits. RGB/DINO and Teacher cannot be reopened | Five integration calls at `t=0,.2,.4,.6,.8`; one additional `t=1` call produces heads only and cannot update action |

The complete runtime therefore has one static evidence build and six dynamic
forwards: five velocity updates and one endpoint-head read. Any donor that
moves a static owner into the dynamic path multiplies its effect and memory by
six; any donor that caches a dynamic owner changes action/time semantics.

## 6. Backward, loss and optimizer boundary

### 6.1 Source-exact loss ledger

| Group | Source weight and target | Ordinary gradient path |
|---|---|---|
| Action | physical flow `1.0`; decoded action `0.08`; event `0.03`; motion `0.03`; smooth delta `0.02`; physical-delta consistency `0.03` | Heads/bottom -> transition and P3/P2 -> dynamic/static P1 -> W/S/Coarse/G/observation wherever the forward path is not detached |
| Proposal | `0.05 * history_proposal_loss` | Auxiliary action-prediction head and shared history encoder; the same encoder can also receive deployed action-path gradients through compressed history rows |
| Future dynamics | `0.10 *` weighted successor/semantic/transport/covariance/visibility/persistence/uncertainty loss | W -> S-owned docks, CoarseAction and current G facts. Teacher tensors and current-loss support are detached targets |
| Intent structure | total `0.02`; half adjacent-W transition loss, half scaffold: object reconstruction `.25`, online public-intent match `.35`, recognizer `.20`, coarse action `.20` | W interval differentiation; G reconstruction; S public carrier; recognizer; CoarseAction and its S inputs. Recognizer target is detached from the online-intent match |
| Flow geometry | warp `.03`, identity advantage `.02`, static identity `.01`, cycle `.01`, smoothness `.002`, uncertainty `.005`, refinement sequence `.02` | Observation/learned-flow path; literal RGB targets prevent a trainable target shortcut |
| Execution value | `0.05` centered physical candidate-value regression | Controller typed value field; candidate action predictions are detached targets |

The source loss algebra is exact. The copied console is not a complete
machine-readable loss ledger, so the audit tool labels its reconstructed loss
budget `estimated-known-terms`; this does not invalidate the source weights but
does prevent reverse-engineering every logged total from recognized console
fields alone.

### 6.2 Gradient lifecycle and ownership

Every trainable parameter must map to exactly one named optimizer role.
Schema25 uses AdamW, explicit role LR scales, decoder-local clip `1.0`, then
global clip `1.0`, with a non-finite owner report before mutation. Teacher
projection parameters are frozen and excluded. Scale-invariant ordered
contraction coordinates are the only explicit no-decay exception.

Late-window median raw gradient L2 values from the log include:

| Role | Tail median raw L2 |
|---|---:|
| global preclip | `0.400288` |
| observation / grounding / grounder | `0.090850 / 0.012695 / 0.004050` |
| S / CoarseAction / W | `0.061315 / 0.030765 / 0.072445` |
| static+dynamic P1 / P2 / P3 | `0.041660 / 0.030910 / 0.009321` |
| controlled transition | `0.005429` |
| bottom query / adapter / organizer / MMDiT / heads | `0.132750 / 0.089535 / 0.170950 / 0.078045 / 0.274100` |
| bottom execution / capacity | `0.026660 / 0.000000` |
| recognizer / history proposal | `0.0000717 / 0.004959` |

The zero late capacity gradient is not proof that the capacity path never
worked: it is nonzero around the opening transition, but capacity reaches
approximately one and the late low-rank contraction coordinates lose leverage.
This is a saturation debt, not an optimizer-coverage failure.

### 6.3 Checkpoint and runtime state

Checkpoint schema `clearvla-mainline-checkpoint-v4` serializes:

```text
architecture identity and manifest digest
path-independent typed config
complete model state
named optimizer groups and AdamW state
scheduler state
epoch, global step and best metric
data state
Python, NumPy, CPU/CUDA RNG and named generator states
```

Runtime evidence caches are deliberately not serialized. Exact resume checks
model names/shapes/dtypes, optimizer ownership, schedule/global-step agreement,
config digest, identity and RNG ownership before mutating live state. Bottom-only
migration is allowed only when the complete bottom ABI and every bottom state
entry match; shape-only partial loading is rejected.

## 7. Existing run behavior

### 7.1 Coverage and performance

```text
log size:               15,260,313 bytes
training windows:       1,136 (epoch 1 batch 20 -> epoch 8 batch 2840)
epoch records:          8
global steps:           22,768
traceback/fatal rows:   0
median / p90 s-batch:   1.8527 / 1.8943
median throughput:      4.318 samples/s
peak process estimate:  11.6464 GiB
```

### 7.2 Full validation trajectory

| Epoch | Step | Physical full | Arm physical | Gripper physical | Normalized full | First normalized | Tail normalized | Decoded gripper F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,846 | `0.104658` | `0.0872082` | `0.176187` | `0.287403` | `0.100460` | `0.336772` | `0.333713` |
| 2 | 5,692 | `0.103847` | `0.0806469` | `0.190960` | `0.273969` | `0.0737977` | `0.323764` | `0.351843` |
| 3 | 8,538 | `0.0898043` | `0.0687866` | `0.167524` | `0.232578` | `0.0526690` | `0.274507` | `0.377506` |
| 4 | 11,384 | `0.0827373` | `0.0649804` | `0.150278` | `0.215329` | `0.0535680` | `0.253158` | `0.429153` |
| 5 | 14,230 | `0.0796621` | `0.0622404` | `0.145530` | `0.212973` | `0.0399821` | `0.251908` | `0.419038` |
| **6 best** | **17,076** | **`0.0772445`** | **`0.0618620`** | **`0.137133`** | **`0.207398`** | **`0.0331819`** | **`0.245300`** | **`0.421929`** |
| 7 | 19,922 | `0.0783049` | `0.0624389` | `0.139749` | `0.209262` | `0.0300607` | `0.247717` | `0.415113` |
| 8 final | 22,768 | `0.0788740` | `0.0626888` | `0.141310` | `0.209929` | `0.0249356` | `0.248711` | `0.418327` |

The best point is epoch 6, not the last epoch. Epochs 7-8 slightly regress in
physical and gripper error while the first-step error continues to improve.
The remaining normalized error is concentrated in the long tail, so the run
does not prove long-horizon phase closure.

Across the first and last audit windows, total loss falls `1.138515 ->
0.0493014`, physical flow `1.02820 -> 0.0165633`, native flow `0.904271 ->
0.0222098`, decoded action `0.235403 -> 0.00208321`, future prediction
`0.0289092 -> 0.0202167`, flow warp `0.111778 -> 0.0868690`, and cycle
`0.308813 -> 0.0214028`. This is strong evidence of optimization and weak
evidence of ownership correctness.

### 7.3 Epoch-8 structural snapshot

| Boundary | Decision statistic |
|---|---|
| G | object content pair cosine `0.636230`; chart overlap `0.370972`; camera-coordinate variation `0.129576`; G3-parent L1 `0.0108606` |
| S public vs policy | interval variation `0.147413` vs `0.147680`; typed context RMS `0.141868` |
| S semantic | mass `0.495881`; object variation `0.249564`; interval variation `0.0142518` |
| S appearance | mass `0.437282`; object variation `0.092534`; interval variation `0.00670116` |
| S geometry | mass `0.530304`; object variation `0.0490525`; interval variation `0.0121672` |
| Teacher/W | Teacher null `0.0421324`, reliability `0.244364`, semantic delta RMS `0.349925`, interval variation `0.134284`; W interval variation `0.0857136`, adjacent cosine `0.908716`, object cosine `0.486572` |
| P1 | protected static detail `0.0402273`; dynamic delta `0.252067`; completed fact `0.256374` |
| P2 | posterior entropy `0.815086`; null `0.172445`; semantic/transport/status type mass `0.389363 / 0.0717014 / 0.538935`; effect RMS `0.0826546` |
| P3 | factual/precision/effect/temporal/state-change RMS `0.205785 / 0.240617 / 0.131328 / 0.291250 / 0.0444467` |
| Transition/bottom | transition value RMS `1.08561`, spatial variation `0.280166`, rows `512`; capacity `0.999994` |

The log therefore shows real object differentiation and W interval structure,
but S's typed values vary much more across objects than across intervals, and
the public-to-policy interval variation barely changes after typed context is
added. Those observations motivate review of fixed K/type aggregation; they do
not identify a unique causal defect.

### 7.4 Decoded action and intervention limits

At epoch 8 the decoded gripper event has precision `0.645161`, recall
`0.309506`, F1 `0.418327`, predicted/target event ratio `0.479735` and timing
MAE `0.947619` steps. The separate event head has F1 `0.145852`; it must not be
substituted for decoded gripper behavior. Motion-head F1 is `0.822459`.

Only `16/179` validation batches (`0.0893855`) receive sampling/proposal
diagnostics and `8/179` (`0.0446927`) receive execution interventions.
Consequently:

- proposal primary and zero are both `0.0774322` because the Schema25
  `proposal_ablation_cache` deliberately returns the unchanged cache; this is
  an algebraic identity, not causal evidence that compressed history is useless;
- learned execution primary is `0.0763526`; hard `0.0877128` and neutral
  `0.0912800` are worse on the matched subset;
- full capacity is `0.0763544`, as expected when learned capacity is already
  almost one;
- three-basis reduction is `0.0751673`, a small low-coverage improvement that
  is insufficient for a structural verdict.

No new training run is warranted by these observations alone.

## 8. Replay invariants versus base mechanics

This distinction is the principal guard against over-hardening.

### 8.1 Protected semantic invariants

Until a donor supplies a complete counterargument, replay must preserve:

1. online current-observable inputs and future/Teacher supervision are
   different types;
2. G is current-only and progressive G1/G2/G3 boundaries remain real;
3. object, interval, type, action-step and basis axes remain explicit until a
   named consumer performs a justified reduction;
4. S is the sole owner of typed relevance; CoarseAction and W cannot create
   independent raw typed selectors under new names;
5. public interval prediction and optional typed relevance remain separable;
6. a null branch carries no learned value; algebraically zero or invalid
   evidence remains zero and is not resurrected by selected-mass normalization;
7. Teacher is no-grad, FP32 target construction and absent from deployment;
8. static evidence is built once, dynamic action-conditioned owners run at
   every velocity call, and the endpoint head call cannot change action;
9. P1 is the only high-resolution visual read and bottom cannot reopen vision;
10. zero P2 effect leaves the factual consequence unchanged;
11. generic trajectory aliases stay neutral and the protected consequence is
    not written under a second semantic name;
12. optimizer, checkpoint and runtime identity cover every trainable or
    serialized owner explicitly.

These are ownership and information-preservation rules. They do not prescribe
a particular attention, posterior, norm or gain.

### 8.2 Mechanics explicitly **not** locked

The following enter replay as `BASE/OPEN`, not `KEEP`:

```text
0.25 normalization floors
[0.25,4] temperature bounds
0.35 RMS contracts and the P3 state-change 0.05 multiplier
fixed mean across K and /sqrt(3) across types
W object self-attention and W1/W2 internal residual form
positive covariance/uncertainty floors and predicted selector validity
joint interval*object P2 posterior and its null
P2 semantic/transport/status type softmax
dynamic P1 folded into the factual owner
five optional P3 lanes around an already protected consequence
reconstructed transition interval identities
joint P3-lane*basis*null bottom routing and its fixed 0.25 write scale
ordered capacity/execution controller parameterization
```

A donor may preserve, soften, reimplement or reject any of these mechanics,
but only as a separately reversible semantic unit with its lost/preserved
information written down.

## 9. Retained base debts

| Debt | Source-certain issue | Log evidence | Replay consequence |
|---|---|---|---|
| `S25-D01` | S's bounded sigmoid leaves a strictly positive coefficient on every valid nonzero route; S then averages four object values with a fixed divisor, sums three type components with `/sqrt(3)`, and applies repeated contracts | Typed interval variation is only `0.0067-0.0143`, much smaller than object variation; public/policy interval variation is nearly equal | Review optionality, attenuation and consumer need before importing any stronger pooling or shared posterior |
| `S25-D02` | S typed context reaches W directly through relevance values and indirectly through CoarseAction; P2 then sees S policy interval beside W fields | Both routes are active, but the log cannot isolate their marginal effects | Map selector conditioning versus value prediction before calling the routes duplicate or deleting either |
| `S25-D03` | W mixes objects through self-attention, detaches current reference, initializes covariance/uncertainty positive, and lets predicted visibility control P2 selector validity | Prediction validity `0.950166` is close to Teacher `0.959591`, but agreement does not prove rightful amplitude ownership | Predicted existence/status must not gain stronger physical suppression authority without a separate unit |
| `S25-D04` | Static detail and a much larger dynamic P1 delta are merged into one factual identity | Epoch-8 RMS `0.0402` static vs `0.2521` dynamic | Later static/dynamic split is a plausible donor, but must preserve both gradient paths and addition frequency |
| `S25-D05` | P2 makes all interval-object candidates compete with one null, then forces semantic, transport and status values to compete | Far interval 3 owns `0.438278` posterior mass; status owns `0.538935` type mass | Factorization/status removal is not pre-approved; review exclusivity and zero semantics unit by unit |
| `S25-D06` | P3 reprojects factual/effect values already present in protected consequence and fixes state-change scale at `0.05` | Optional lanes are nontrivial, but causal necessity is unmeasured | Treat duplicate semantic names and constants as open; do not preserve them merely for V120 ancestry |
| `S25-D07` | Transition creates four interval identities by adding learned labels to one public G3 chart; no exact G3 anchor axis is transmitted | Transition value RMS `1.08561` exceeds the top lane contracts and downstream relative scale is not isolated | Schema26 exact-anchor material is high priority; map bottom normalization and all 512 consumers first |
| `S25-D08` | Bottom forces `5 lanes * 4 bases` plus one null through one route and then multiplies the result by fixed `0.25`; protected detail uses a separate no-null basis route | The full-capacity controller is already saturated; the log has no per-lane causal ablation | Per-lane routing is a later donor candidate; do not inherit the joint posterior as truth |
| `S25-D09` | Proposal-zero cannot intervene on deployed compressed history and returns an unchanged cache; `proposal_condition_dropout` is also sampled/logged without entering the forward value | Exact equality between primary and zero | Repair the diagnostic/config semantic before using either to delete or validate history conditioning |
| `S25-D10` | Late capacity coordinates have zero gradient at nearly full capacity | Capacity `0.999994`; late raw capacity gradient `0` | Determine whether adaptive capacity has a real task degree of freedom before preserving or removing it |
| `S25-D11` | Best physical behavior occurs at epoch 6, while first-step error keeps improving and tail error plateaus/regresses | e6 `0.0772445`; e8 `0.0788740`; e8 first/tail `0.02494/0.24871` | Do not optimize donors against final-only or first-step-only metrics; keep horizon bands separate |
| `S25-D12` | Copied log lacks exact source/config/normalizer/checkpoint identity | Auditor manifest fields other than capability are absent | Use behavior directionally; never claim a bit-exact controlled win or resume source from this console alone |

These debts are the initial unresolved-assumption register. None authorizes an
immediate fix. They bound what the Schema26 card must inspect.

## 10. Base acceptance and next gate

The Schema25 audit is complete enough to begin Schema26 review because all
active producers, transformations, consumers, ordinary loss paths, optimizer
owners, checkpoint fields and runtime repetitions touched by the expected donor
have been located. Functional closure is deliberately **not** claimed: the
debts above identify remaining axis, scale, duplication and null-authority
questions.

Before any Schema26 unit receives `KEEP`, `SOFTEN` or `REIMPLEMENT`, its version
card must:

1. diff Schema25 -> historical Schema26 and split the commit into independent
   semantic units;
2. identify whether each claimed Schema26 defect actually exists in this base;
3. trace exact G3 anchor production, S boundary changes, W common/residual
   construction and any typed re-entry removal through P2, transition and
   bottom;
4. state which Schema25 invariant is preserved and which `BASE/OPEN` mechanic
   changes;
5. record the information/freedom delta, zero behavior, scale, gradient path,
   runtime frequency, checkpoint impact and rollback boundary;
6. use existing source/log evidence first. No experiment is implied by opening
   the Schema26 card.

## 11. Reproduction commands

```powershell
git show 6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6:clearvla/mainline/manifest.py
git show 6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6:configs/mainline/object_intent_dynamics_323.json
git diff 32d969fccecca35a641382888f6f4681221c9c24 6a6c1bfb164e0013a4f5e6b4303d296f4de5b2d6 -- clearvla/mainline
python -m clearvla.tools.audit_policy_logs --format text schema25_s_owned_typed_b8.log
python -m clearvla.tools.audit_policy_logs --format json schema25_s_owned_typed_b8.log
```

The audit commands are read-only. This card intentionally stores selected
decision statistics rather than copying the context-scale console log.
