# ClearVLA as a lightweight WAM: literature integration and validation ledger

Updated: 2026-09-04

This is an auxiliary research ledger. It is not an architecture contract, an
implementation authorization, or a replacement for
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md).
It records literature-derived hypotheses, their current source boundaries and
the smallest tests that could reject them. Architectural decisions still belong
in the compact contract only after they have passed the mainline release and
causal/lifecycle gates.

## Working conclusion

It is useful to treat the whole ClearVLA policy as a **lightweight world-action
model (WAM)**, provided that the term is used functionally rather than as a
claim that ClearVLA is a video generator:

```text
observed object/history evidence
  -> intent S
  -> physical action proposal
  -> action-conditioned object future W
  -> factual/consequence/transition consumers
  -> action-generating MMDiT bottom
  -> deployed physical action
```

This view changes the priority of the research. The central question is not
whether B-spline, MIP, JEPA or a larger world model is individually attractive.
It is whether action, predicted future and final action stay mutually useful and
consistent through the current producer -> consumer -> deployment lifecycle.
B-spine is therefore one candidate internal action representation, not the
organizing principle for the whole model.

The present evidence supports four conclusions:

1. **Do not optimize for two denoising steps.** MIP-style intermediate
   supervision is worth testing, but the number of refinement stages must be
   selected by complete behavior, especially far-horizon/contact/gripper
   performance. Two is a control point, not a goal.
2. **Keep continuous flow time.** In the current linear noise-to-clean bridge,
   `t` is the exact corruption coordinate and is better founded than call index,
   action-state norm or a guessed task phase. If the solver becomes non-uniform
   or iterative, add step size/next time and refinement-stage identity as
   separate conditions; do not ask one scalar to represent all clocks.
3. **The last integration interval deserves an audit.** Deployment uses uniform
   Euler updates at `t={0,.2,.4,.6,.8}`, while training samples the mirrored
   Beta distribution that concentrates more heavily near the noisy end. The
   final `.8 -> 1.0` update is therefore a relatively large left-endpoint step
   in the least-sampled fifth of the training chart. This is a source-derived
   risk, not yet evidence that the step is wrong.
4. **The next work should mostly be read-only causal diagnostics.** A solver
   oracle, per-time curvature/error, cross-chunk seam measurements and matched
   W/route interventions can decide whether the next learned unit belongs in
   the final DiT, the Teacher/W path, the solver, or the outlet adapter.

## Current source boundary used for this review

The literature is mapped to the active Schema30 ABI / Schema28-core recovery,
not to an old experiment name:

| Concern | Current owner/consumer boundary | Source anchor |
|---|---|---|
| online evidence and static cache | one `encode_online` per observation; future evidence is training-only | `clearvla/mainline/model/policy.py::ClearVLAMainlinePolicy.encode_online` |
| S intent and coarse action | object/language/history -> four interval carrier -> typed-free physical proposal | `clearvla/mainline/model/intent.py::StatelessObjectIntentOrganizer`, `CoarseActionIntent` |
| action-conditioned W | `PhysicalActionCondition` -> W1 intervals 0/1 -> W2 intervals 2/3 | `clearvla/mainline/model/dynamics.py::ObjectFutureDynamicsCompiler` |
| bounded outer correction | proposal ODE -> one W rebuild -> refined ODE from identical noise | `clearvla/mainline/runtime/sampling.py::sample_refined_cached_action_with_cache` |
| flow training | one sampled linear bridge state and constant target velocity | `clearvla/mainline/training/losses.py::sample_flow_matching` |
| deployed solver | five Euler updates per pass, plus a non-updating endpoint head call | `clearvla/mainline/runtime/sampling.py::_integrate_cache` |
| final action generator | noisy physical field plus evidence -> three MMDiT blocks -> physical velocity head | `clearvla/mainline/v120_core/time_domain_mmdit.py::EvidenceLatentMMDiTActionDecoder` |
| outlet semantics | physical chart/profile and explicit continuous versus CALVIN binary gripper boundary | `clearvla/data/action_chart.py`, `clearvla/data/physical_chart.py`, `clearvla/mainline/model/action_codec.py` |

The modeling problem contains at least four clocks. The current generator
realizes flow time and horizon row explicitly, represents task phase only
indirectly, and does not yet expose physical elapsed time across outlets:

| Clock | Meaning | Current realization |
|---|---|---|
| `t` | noise-to-clean flow coordinate | continuous sinusoidal embedding plus MLP |
| `tau` | row position inside the 24-step action chunk | persistent horizon sinusoid/query |
| `t_phys` | physical elapsed time of a row (`tau / control_hz`) | not an explicit cross-outlet coordinate today |
| task phase | contact/transition progress in the environment | only indirectly represented through observed history/S/W; not the flow clock |

Any MIP or adaptive solver additionally introduces refinement stage `k` and
possibly requested step `dt=t_next-t`. These should be named separately.
Replacing `t` with `k` would make the same call mean different corruption
levels under different schedules.

## Evidence grading

The papers below are not given equal authority.

- **R1 — direct robot evidence:** physical or established manipulation
  benchmarks test the method's deployed action behavior.
- **R2 — mechanism evidence:** the work isolates a mechanism close to a
  ClearVLA boundary, but the model/data/runtime differs materially.
- **D — diagnostic method:** useful for deciding whether a path is causal; it
  does not itself establish a better robot policy.
- **N — new preprint caution:** 2026 work checked against the primary abstract
  but too recent to treat as replicated. `N` is combined with R1/R2/D as
  appropriate.

Internal activation RMS, latent proxy scores, reconstruction quality or a
paper's average benchmark gain are never sufficient to authorize a ClearVLA
source change. The required bridge is a current-checkpoint action intervention
or a fresh matched run under the complete ABI.

## Research axis A — object-latent future and Teacher supervision

### What the literature supports

| Work | Grade | Useful evidence | What it does not prove for ClearVLA | Smallest relevant test |
|---|---:|---|---|---|
| [SlotFormer](https://arxiv.org/abs/2210.05861) | R2 | autoregressive dynamics over learned object slots can predict long object futures and support planning without object labels | that slot identity is physically stable in manipulation, or that ClearVLA should replace its explicit K=4/null object contract | measure current-object -> future-object identity/transport consistency per interval, then shuffle only the predicted slot identity seen by P |
| [DINO-WM](https://arxiv.org/abs/2411.04983) | R2 | future prediction in DINO patch space can support planning without pixel reconstruction | that test-time action optimization or a free patch grid should replace the policy-integrated object W | compare DINO semantic/transport target error with downstream action utility under matched W zero/shuffle interventions |
| [V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) | R2 | a video-pretrained latent model can be post-trained with actions and used for zero-shot image-goal planning | that web-scale video pretraining or online planning is affordable or necessary for this lightweight policy | freeze the current vision backbone and test a small action-conditioned latent target only as Teacher supervision; no planner in the first experiment |
| [FLARE](https://arxiv.org/abs/2505.15659) | R1/R2 | future-latent alignment can be applied inside an action DiT; its main setting uses layer 6 of 8 and the paper reports a notable drop when applying it earlier at layer 4 | that layer 6 is globally optimal, or that future observation may enter deployment | put a stop-gradient future probe/alignment only on a late action hidden state after causal use is established |
| [ForeTime-VLA](https://arxiv.org/abs/2608.20735) | N/R1 | privileged future WAM features, phase and time-to-transition can be distilled into a causal VLA with small reported latency overhead | that a second video WAM should be deployed, or that phase labels may enter the online graph directly | train-only target probe: predict compact future phase/transition time from the same S/W consumer and measure contact/far action intervention |
| [PHR-VLA](https://arxiv.org/abs/2608.27609) | N/R1 | privileged future representation is especially useful when supervision is contact-centric and wrist-camera specific | that all cameras should be pooled into one stronger future target | report Teacher loss and action utility separately for wrist contact patches versus global/high camera evidence |
| [Temporal Forcing](https://arxiv.org/abs/2608.30643) | N/R1 | an explicit history path aligned to a pretrained 4D representation can reduce observation aliasing | that ClearVLA needs another 4D foundation model beside its existing `-8/-4/0` raw/DINO/flow history | first shift/drop one existing history time or one learned flow at a time and measure action/phase effects |
| [TrAct](https://arxiv.org/abs/2608.24101) | N/R2 | visual tracks can act as an embodiment-agnostic interface between candidate action and future prediction | that candidate rollout plus a reward model fits the present runtime budget | derive sparse tracks from existing Flow-DINO only for a target/intervention control; do not add candidate planning |
| [DreamZero / World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) | N/R1 | joint video/action world-action training can itself yield a cross-embodiment policy | that a 14B video diffusion WAM, video reconstruction target or 7 Hz system is the right scale for ClearVLA | use only as support for the system-level WAM framing; no direct transplant gate |
| [ImageWAM](https://arxiv.org/abs/2606.19531) | N/R1 | an image-edit model can supply compact action-relevant denoising KV context without decoding a target image; the abstract reports one-sixth the FLOPs and one-quarter the latency of video WAM controls | that ClearVLA should replace its object-latent W with an image editor, or that attention maps prove action utility | compare the present W with an action-relevant current-to-target change target; keep the deployed object owner unchanged in the first probe |
| [SLIM-0.5B](https://arxiv.org/abs/2608.09771) | N/R1 | joint inverse dynamics from current/future latent and forward dynamics from current latent/clean action can make future representation action-grounded; its EMA ablation shows lower future MSE can coexist with much lower effective rank | that ClearVLA should move to two-stage training or remove its explicit W/outer refinement | add rank/energy/cosine collapse metrics beside target error and test both action-to-W and W-to-action reachability |
| [LaWAM](https://arxiv.org/abs/2606.15768) | N/R1 | a predicted latent action conditions a latent world model whose visual subgoal is consumed by a late action DiT; removing the world model reportedly hurts long-horizon behavior most | that a frozen/gradient-insulated W is compatible with ClearVLA's ordinary end-to-end online autograd | map latent action to `PhysicalActionCondition`, latent subgoal to `FutureObjectDynamics`, and intervene at the terminal consumer before copying any training isolation |
| [AcrossWAM1.0](https://arxiv.org/abs/2608.29937) | N/R2 | training-only teacher/state modules can be removed from an exported WAM graph and deployment reachability can be accounted separately from total parameters | that its cross-backbone smoke proves behavioral portability or that its billion-scale export is lightweight here | keep Teacher reachability at zero in deployment and report parameters/calls by inference reachability |
| [Causal World Modeling for Robot Control / LingBot-VA](https://arxiv.org/abs/2601.21998) | N/R1 | causal chunk caching, real-observation refresh and asynchronous execution can close a large video/action WAM loop; its video stream is only partially denoised while action completes | that a partial video denoising result authorizes shorter ClearVLA action integration, or that an internal W rebuild equals a new real observation | compare true next-observation closure separately from proposal-to-W internal correction |
| [Prediction with Action (PAD)](https://arxiv.org/abs/2411.18179) | R1/R2 | joint RGB/action/depth DiT denoising is an early explicit world/action formulation and demonstrates shared generative physical representations | that pixel generation, one shared modality time or low control frequency is desirable | use as a negative scale/control-frequency reference supporting latent rather than pixel W |

### Integration judgment

The most compatible direction is **better action-grounded latent future and a
proved terminal consumer, not a larger deployed W**. ClearVLA already owns
object identity, current validity, camera-resolved transport and four future
intervals. The missing evidence is whether action changes W and whether these
fields in turn carry useful contact/phase information into final action.
ForeTime-VLA and PHR-VLA suggest compact time-to-transition/phase and
camera-specific contact targets. SLIM and LaWAM add the important bidirectional
test; PAD, ImageWAM, SlotFormer, DINO-WM and V-JEPA 2 support latent prediction
without overriding the current owner/null/camera contracts.

A functional WAM claim therefore requires both matched arrows under the same
observation, history and initial ODE noise:

```text
PhysicalActionCondition --zero/shuffle/scale--> W / FutureObjectDynamics
W / FutureObjectDynamics --zero/shuffle------> final deployed action
```

If only the first arrow is present, W predicts but is optional to the policy.
If only the second is present, the terminal action block uses a future carrier
that is not demonstrably action-conditioned. Neither case establishes the
intended world-action closure.

## Research axis B — action <-> world consistency and perception-to-action use

| Work | Grade | Useful evidence | What it does not prove | Smallest relevant test |
|---|---:|---|---|---|
| [V-Link](https://arxiv.org/abs/2608.25308) | N/R2 | semantic and spatial beliefs inside a VLM may be inaccessible to the Action DiT; query-based asymmetric injection can improve reachability | that another visual query branch should be added before proving the present G/P/W path is unused | matched zero/shuffle/swap interventions at G address, P value and final DiT consumption, with action and MSE effects |
| [CARD](https://arxiv.org/abs/2608.20763) | N/D | an internal belief is not evidence that another prediction/action axis consumes it; cross-axis interventions expose routing failure | that a grid/VLM diagnostic predicts robot success | construct a ClearVLA cross-axis routing matrix: perturb S interval, W object/interval, camera geometry, P2 route and observe action horizon/arm/gripper response |
| [Action-JND](https://arxiv.org/abs/2608.21247) | N/D | visual-token compression should be judged by the resulting language-conditioned action response, not similarity or attention alone | that a reported tolerance is a robot-safety threshold | measure final action and task error while removing camera/patch evidence, stratified by contact and horizon |
| [GUARD](https://arxiv.org/abs/2608.04510) | N/D | token-indexed counterfactual ablation at the final VLM KV boundary can reveal denoising-response failures across tasks and embodiments | that a runtime alarm repairs a missing producer-to-consumer path | use the intervention pattern offline to localize which evidence reaches the final action block |
| [AtVLA / Look Where It Matters](https://arxiv.org/abs/2608.02197) | N/R1 | low-resolution global vision can miss localization/contact detail; uncertainty-triggered local high-resolution re-encoding can improve fine manipulation | that action-conditioned crops or multiple action samples preserve ClearVLA's static P1 and dynamic action-query boundary | first compare high versus wrist camera and raw/DINO/P1 ablations in contact windows; add no crop path until a missing detail owner is demonstrated |
| [Trajectory-Consistent Flow Matching](https://arxiv.org/abs/2605.08511) | N/R1 | pointwise velocity training can disagree with integrated deployment; the reported improvement requires trajectory consistency, velocity smoothness, time coverage and higher-order integration together | that an extra loss or RK4 alone repairs ClearVLA's outer ownership gap | read-only solver oracle and local step-doubling first; only then decide whether the defect is solver, time coverage or learned field |

The current outer loop already supplies a decisive metric: the final action can
move away from the action used to rebuild W. Literature does not justify a
third unconsumed `W(final)` call. The useful next measurement is a causal
triangle under identical initial noise:

```text
proposal action -> W(proposal) -> refined final action
       |                |                 |
       + interval/delta + W field change  + task/action error
```

Shuffle or freeze one vertex at a time. If final action quality improves while
the W-condition residual grows, the problem is not simply numerical closure.
If a higher-accuracy solver reduces both residual and behavior error, the
integrator is implicated. If changing W has little action effect, another
outer pass cannot help.

## Research axis C — action generation: flow, MIP and direct controls

| Work | Grade | Useful evidence | What it does not prove | Role in the ClearVLA matrix |
|---|---:|---|---|---|
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | R1 | iterative stochastic action generation plus receding-horizon execution is strong for multimodal high-dimensional control | that every task needs many stochastic denoising steps or that diffusion beats a matched continuous head here | established iterative/chunk baseline |
| [Much Ado About Noising / MIP](https://arxiv.org/abs/2512.01809) | N/R1 | much of a generative policy's gain can come from iterative computation with intermediate supervision and appropriate stochasticity; two-step MIP is often competitive in its setting | that two is optimal, that fewer calls outrank success, or that MIP should be stacked on `2 x 5` ClearVLA updates | candidate replacement action backend, with prefix `K=1..Kmax` behavior curves |
| [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | R1 | parallel decoding, action chunking, continuous actions and a simple L1 objective can produce very strong LIBERO behavior and throughput | that L1/direct regression can express every multimodal ClearVLA outlet | mandatory non-iterative control; prevents assuming flow/MIP is inherently better |
| [DriftingVLA](https://arxiv.org/abs/2608.29749) | N/R1 | a distribution-drifting training objective can retain joint chunk generation while using one action-expert forward | that its per-dimension training drift transfers to the current physical field/outer W contract | one-forward generative control after the direct head, not an immediate mainline change |
| [$\pi_0$](https://arxiv.org/abs/2410.24164) | R1 | flow matching can scale across single-arm, bimanual and mobile platforms when paired with a large VLM and diverse physical data | that a large VLM or shared normalization solves outlet physics automatically | supports flow as a viable family; adapter and data scale are confounders, not a solver verdict |
| [AdaFlow](https://arxiv.org/abs/2402.04292) | R1/R2 | a trained conditional-variance estimate can allocate a large step to low-variance states and more steps to high-variance states | that variance is available training-free, or that its assumptions hold across two different ClearVLA W conditions | later learned step-controller control, only after a fixed-checkpoint solver audit |
| [FlowPolicy](https://arxiv.org/abs/2412.04987) | R1 | consistency flow matching can produce a one-step 3D manipulation policy with a reported large speedup | that shortcut/consistency distillation preserves ClearVLA performance or world/action closure | one-step efficiency reference, not a reason to remove the flow teacher/control |
| [AdaVLA](https://arxiv.org/abs/2608.29208) | N/R1 | two velocity evaluations estimate relative flow-trajectory curvature and adapt the next step; the paper reports speedups on flow VLAs with near-preserved success | that its searched threshold improves performance, or that reuse is valid across the ClearVLA W rebuild | use its curvature quantity as P0 telemetry before considering adaptive inference |

### Recommended MIP boundary

If MIP reaches implementation, it should be an **alternative terminal action
controller inside the final action-generating DiT boundary**, not a new top
module, not a second W, and not an extra loop appended to the existing two full
ODE passes. It should keep G/S/W/P/transition evidence fixed within one outer
condition and expose:

```text
x^(k), continuous flow time t, requested next time t_next,
refinement stage k/K, fixed evidence cache
  -> final action DiT/refinement tail
  -> complete predicted clean physical action
  -> feedback as x^(k+1) for the next supervised call
```

Use one maximum-stage model and evaluate every prefix `K=1,2,3,...`; select K
by the complete Pen/RDT/CALVIN behavior/cost frontier. The staged denoising
range should be chosen from measured local error/curvature, not divided into
equal ranges merely for symmetry. Later stages may emphasize smaller
near-clean/contact corrections, but must retain all 24 rows and the complete
18-D field. Intermediate supervision is part of the MIP hypothesis and must be
introduced under a new schema as such, not used as a numerical patch for a
disconnected owner.

This is a ClearVLA-specific hypothesis, not something established by the MIP
paper. That paper repeats a shared policy call with time conditioning; it does
not prove that repeating only this model's last DiT block retains all dynamic
P/transition computation. A bounded comparison must therefore contrast a
repeated terminal block with a repeated complete dynamic policy call. In code,
the change cannot be limited to the final linear head: every execution
candidate that currently reads the V120 velocity reader must be routed through
the selected `TerminalActionController`.

Before this source unit is justified, two cheaper questions must be answered:

1. Does a high-accuracy integration oracle improve the current checkpoint?
2. Does a matched direct continuous/chunk head approach or beat the iterative
   policy on the same evidence and action ABI?

If the second answer is yes, adding iterative stages is unlikely to be the best
performance-first move.

The MIP ablations sharpen the time-condition decision: removing timestep
conditioning or intermediate supervision made the method worse than the simple
regression control in that study. Its selected `t*=.9` and two-step result show
that a small minimum can work; they do not show that time may be removed or
that two steps maximize ClearVLA performance. OpenVLA-OFT supplies a similarly
important control: in its matched table continuous diffusion and continuous L1
were approximately tied (`95.4` versus `95.3`), so the correct conclusion is
that a direct head can be competitive and faster, not that L1 has already been
proved superior.

## Research axis D — chunk continuity, long horizon and contact

| Work | Grade | Useful evidence | What it does not prove | Smallest relevant test |
|---|---:|---|---|---|
| [ACT](https://arxiv.org/abs/2304.13705) | R1 | action chunks reduce compounding error and can learn precise bimanual behavior from limited demonstrations | that a CVAE Transformer or temporal ensemble should replace the current bottom | retain as a simple chunk/bimanual reference and compare overlap/seam behavior |
| [ChunkFlow](https://arxiv.org/abs/2607.12992) | N/R1 | frozen/editable/future zones, overlap blending, continuity objectives, corrupted history and scheduled sampling target seams between consecutive chunks | that an additional continuity loss is safe before the current seam error is measured | on consecutive deployed chunks, report value/velocity/acceleration discontinuity and overlap-conditioned action error by 1/4/8 rows |
| [ForeTime-VLA](https://arxiv.org/abs/2608.20735) | N/R1 | future phase and time-to-transition targets can improve dynamic contact timing | that predicted phase should gate or hard-switch the controller | correlate a detached phase/ETA probe with gripper event latency; then intervene on its consumer |
| [PHR-VLA](https://arxiv.org/abs/2608.27609) | N/R1 | wrist-centric privileged future targets can concentrate supervision at contact | that global future loss should be replaced | compare contact-window and non-contact-window Teacher/action utility separately |

The immediate gap is measurement: the repository reports within-chunk
near/mid/far bands and decoded gripper events, but not the seam between the
executed prefix of one prediction and the next prediction. A zero-training
replay should measure both. This can distinguish a weak far-row representation
from a receding-horizon distribution mismatch.

## Research axis E — numerical action representation

| Work | Grade | Useful evidence | What it does not prove | Current interpretation |
|---|---:|---|---|---|
| [FAST](https://arxiv.org/abs/2501.09747) | R1 | DCT compression is an effective tokenizer for high-frequency actions and can match diffusion VLA performance with faster training in its autoregressive setting | that DCT should replace the continuous flow field or that low frequencies may suppress events | strongest global-support control for B-spine Gate A |
| [B-spline Policy](https://arxiv.org/abs/2607.09648) | N/R1 | fixed control points/local support can expose a continuous action trajectory and local modification | that spline control points should become the ClearVLA action ABI | supports a parallel fixed chart only |
| [Spline Policy](https://arxiv.org/abs/2606.07386) | N/R1 | spline output can support continuous resampling and local correction | that its vector-field projection assumptions hold for noisy `x_t` | later output/interface evidence, not a first bottom implementation |
| [BEAST](https://arxiv.org/abs/2506.06072) | R1 | fixed-length B-spline action tokens can represent manipulation trajectories | that learned/adaptive knots are compatible with current field roles and outlet units | candidate chart evidence only |

B-spine-0 remains a sensible bounded experiment only after its existing release
gates. It should not be coupled to MIP in the first run: otherwise a gain cannot
be attributed to numerical representation, iterative depth or solver changes.

## Research axis F — lightweight runtime and outlet adaptation

| Work | Grade | Useful evidence | What it does not prove | ClearVLA consequence |
|---|---:|---|---|---|
| [SmolVLA](https://arxiv.org/abs/2506.01844) | R1 | a small VLA can train on one GPU and deploy on consumer hardware; asynchronous inference decouples prediction from execution | that action-solver calls are the only latency bottleneck | measure encode/static, each dynamic call, W rebuild and robot wait separately; test asynchronous execution before cutting solver quality |
| [Mamba-based SmolVLA action expert](https://arxiv.org/abs/2608.21407) | N/R1 | an SSM expert reportedly retains more behavior over long execution horizons with fewer parameters than the Transformer control | that an SSM should be mixed with B-spine/MIP or that N=1 comparisons predict ClearVLA's 24-row errors | independent later replacement control if sequence modeling, not evidence routing, is proven limiting |
| [TinyVLA](https://arxiv.org/abs/2409.12514) | R1 | a compact multimodal backbone plus diffusion action decoder can be data-efficient and fast in simulation and real-robot experiments | that small parameter count implies good use of explicit world evidence | architecture-size control only; profile evidence flow and wall-clock separately |
| [C³ache](https://arxiv.org/abs/2606.08962) | N/R1 | Fast-WAM residuals at the same denoising step can be reused across smooth consecutive chunks; the abstract reports up to `2.5x` wall-clock speedup | that a cache remains valid across a ClearVLA W rebuild or changed action condition | consider reuse only when CandidateWorld/action-condition identity is exactly unchanged; otherwise fail closed |
| [Reflex](https://arxiv.org/abs/2607.14695) | N/R1 | global timestep injection invalidates ordinary KV caching; static/sliding/dynamic partitioning plus asynchronous visual/action execution reports `2.58x` and stable 50 Hz | that dynamic P/W/transition tensors are cacheable or that phase-gated normalization should be copied | exploit the already explicit one-observation static cache; prove full-batch equivalence for every reused tensor |
| [RDT-1B](https://arxiv.org/abs/2410.07864) | R1 | bimanual diffusion benefits from a physically interpretable unified action space and heterogeneous multimodal inputs | that projecting native 14-D data to right-arm 7-D already tests bimanual modeling | native bimanual expansion starts at the action/state ABI, not at an extra camera or hidden task token |
| [Octo](https://arxiv.org/abs/2405.12213) | R1 | a shared transformer policy can adapt across many platforms when observation/action interfaces are modular | that task id should enter the shared core or physical coordinates can be silently normalized together | supports explicit `OutletActionAdapter` and task identity remaining data/sampling metadata |
| [$\pi_0$](https://arxiv.org/abs/2410.24164) | R1 | a common flow policy can span single-arm, dual-arm and mobile embodiments with sufficient data and interface design | that cross-embodiment scale eliminates controller semantics and calibration | compare shared core only after each outlet's native physical chart is explicit |
| [What Matters for Latent Actions in Robot Learning](https://arxiv.org/abs/2608.19613) | N/D | a broad unified comparison shows that latent-action proxy metrics need not predict downstream manipulation | that latent actions should replace the physical action ABI | use downstream physical action/closed-loop metrics for every representation decision |

The typed-P2 wording also needs a current-source qualification. There are five
named owners (`policy`, `semantic`, `appearance`, `geometry`, `horizon`), but
`policy` is now a protected `policy_carrier`; the live optional
`RoleDeltaAttnRes` routes four deltas (`semantic`, `appearance`, `geometry`,
`horizon`). The “five routes” wording in the current issue ledger describes the
historical observation series. A present-day intervention must therefore test
the protected carrier separately from the four-way optional delta mixture.

Likewise, lightweight WAM papers point to the existing static/dynamic split,
not to indiscriminate caching. ClearVLA may reuse observation-static evidence
within one observation. It must treat the cache before and after the W rebuild
as different unless `CandidateWorld` and action-condition identity are proven
equal. Global flow time remains dynamic and cannot be hidden inside a reused KV
state without an equivalence proof.

The current outlet boundary is narrower than the generalist papers. The shared
policy consumes a native 7-D action and a 24 x 18 physical flow field. The RDT
profile selects native indices `7..13` and two cameras (`high`,
`right_wrist`); current config/transition validation still requires exactly
two cameras. It is therefore a right-arm/two-view adapter result, not native
three-camera or bimanual evidence. A true two-arm chart cannot be obtained by
changing `action_dim` to 14: it must preserve one six-joint plus one-gripper
chart per limb, likely yielding two limb-local 18-D physical fields, and carry
that meaning through codec, lift, heads, loss, decode, checkpoint and per-limb
validation. The exact 36-D design remains an ABI decision, not an inference
from RDT-1B.

Task identity remains sampling/split/logging metadata; instruction/T5 is the
model condition. A cheap invariant is to rename or reorder only the CPU task
registry while holding the tensors fixed: the shared model output must remain
bit-identical even though the statistics row may change. CALVIN's binary
command likewise remains an outlet head with its six compatibility-only future
gripper coordinates zeroed before dynamic consumers; continuous-action papers
do not authorize merging it back into the Pen/RDT gripper field.

## The late-step and time-conditioning question

Current source facts:

```text
training bridge: x_t = (1-t) * noise + t * target
training t:       mirrored Beta(1.5,1.0), approximately Beta(1,1.5)
deployment nodes: 0.0, 0.2, 0.4, 0.6, 0.8
update:           x <- x + 0.2 * v(x,t)
endpoint:         t=1.0 head call, no physical-field update
```

For the idealized Beta(1,1.5) density, about 28.4% of samples lie in
`[0,.2]`, while only about 8.9% lie in `[.8,1]` and about 3.1% in
`[.9,1]`. Thus the last Euler update
uses one velocity evaluation at `.8` to cross the whole least-sampled fifth of
the chart. This does **not** prove that late curvature is larger, but it makes
the user's concern technically well founded.

There are two further source details. Every deployed action is ten physical
Euler updates, not five total: proposal `5` + one W rebuild + refined `5`, with
an additional non-updating `t=1` head forward after each pass. Also,
`ActionQueryEncoder` validates and then drops time; the time value reaches
dynamic P1 modulation and the V120 bottom/MMDiT, but not every object called at
an ODE node. A time probe must therefore be localized by consumer rather than
claiming that a time argument in a function signature proves its use.

The first audit should keep the checkpoint, initial noise and W cache fixed and
record, separately for proposal/refined passes and action bands:

1. `||v(t+.05, x+.05v)-v(t,x)||/.05` as a local field-change proxy;
2. one full step versus two half steps for local truncation error;
3. 5-step Euler versus 10/20-step Euler, Heun and a high-accuracy RK reference;
4. final action error, W-condition residual, gripper event timing and runtime;
5. error contribution from each interval after holding the other intervals to
   the higher-accuracy trajectory.

The existing `t=1` endpoint forward makes a cheap late-step diagnostic
possible without changing the returned action: compare the `.8` velocity with
the endpoint velocity evaluated on the Euler endpoint, approximately
`.1 * ||v(x_E,1)-v(x_.8,.8)||` for a Heun correction magnitude. Then run a
matched convergence panel: existing five-step Euler, only the last interval
split into two half steps, 10/20-step Euler, and matched-NFE midpoint/Heun. A
separate causal probe must hold `x_t` and all caches fixed while using correct,
shuffled, constant and zero time at dynamic P1 and at the terminal block.

Only after this oracle should a schedule be selected. Possible outcomes are:

- late error dominates -> keep continuous `t`, allocate more nodes near 1 or
  use a higher-order final interval;
- error is broad -> train-time trajectory consistency/time coverage is more
  plausible than a late-only schedule repair;
- solver changes barely move action -> the bottleneck lies in W/evidence/action
  modeling, and more integration is wasted;
- a direct head matches behavior -> prefer the simpler controller unless flow
  wins a clear multimodal/contact case.

If a non-uniform or adaptive scheme is tested, condition the dynamic block on
`(t, t_next)` or `(t, dt)` in addition to any MIP stage id. Call count is not a
substitute for time, and action-state norm is not a stable time proxy across
outlets. `logSNR` or remaining-time are only monotone reparameterizations of
the same corruption coordinate and add no information by themselves. Also do
not add proposal/refined pass identity now: training has one formal forward and
no second-pass target, while the two deployed passes are already distinguished
by different W caches. A new pass token would create a fresh train/deploy gap.

Physical action time is the genuinely missing candidate, not a replacement for
flow time. LaWAM reports constructing horizon rows from physical interval and
control frequency and encoding `token_index / control_frequency`; its mixed
5/10/20 Hz experiments support distinguishing equal row index from equal
elapsed seconds. If Pen/RDT/CALVIN are later trained under mixed rates, add a
separate physical-time coordinate at the outlet/action boundary. Do not reuse
`t`, which still denotes corruption, or a task-phase token, which denotes
environment progress.

## Ordered experiment ledger

### P0 — no training and no architecture change

1. **Solver oracle and time audit:** the matched comparison described above.
2. **Outer-consistency triangle:** proposal/refined/final-W-condition residuals
   with W fixed, shuffled and rebuilt exactly once.
3. **Future consumer matrix:** Teacher target quality, W field amplitude and
   action intervention for every interval/type/camera role on the same rows.
   Include both action-condition -> W and W -> final-action arrows, plus W
   effective rank, top energy, token cosine and target error so low-MSE collapse
   is not mislabeled as improvement.
4. **Cross-chunk seam replay:** consecutive predictions under the real executed
   history, including first/second differences and gripper event latency. Add
   `-8/-4/0` history permutation, duplicate-current and future-leak guards.
5. **Perception/action routing matrix:** G address, W semantic/transport, P2
   value/route and final action axes, CARD-style. For current bottom typed-P2,
   compare each of the four optional deltas with exact equal mixture and
   per-source one-hot controls while keeping the protected policy carrier fixed.
6. **Latency decomposition:** static vision/encode, W build, one dynamic call,
   complete `2 x 5` solver and robot execution overlap.

P0 writes compact decision tables only. It does not change a loss, add a W
pass, alter the time grid in production or copy raw probe dumps into memory.

### P1 — after the complete Schema30 release decision

Run isolated, one-question controls rather than a bundle:

- read-only non-uniform/higher-order solver candidate if P0 implicates
  integration;
- a fresh current-mirrored-Beta versus uniform-time versus mixture control if
  P0 shows late under-training rather than only integration error;
- compact Teacher-only phase/time-to-transition or wrist-contact target if P0
  shows missing future bandwidth at contact;
- a stop-gradient four-interval future/object probe on the existing W output
  and on a late action-DiT hidden state, with action/time/K shuffles and a
  detach-consumer negative control;
- direct continuous/chunk action head using the identical evidence/ABI,
  compared both as a single direct pass and as direct proposal -> one W rebuild
  -> direct refinement;
- asynchronous execution if latency, rather than policy quality, is limiting;
- exact static/sliding/dynamic cache partitioning if profiling shows repeated
  static work; never reuse a pre-rebuild W condition after its identity changes;
- existing B-spine Gate A as a representation-only study.

### P2 — new schema, one semantic unit per experiment

- one explicit `TerminalActionController = {flow_velocity, direct_l1, mip_k}`
  selection boundary;
- final-DiT MIP with `K=2/3/4`, complete-clean-action supervision at every
  stage and a prefix behavior curve; performance, not smallest K, selects it;
- trajectory-consistent training only if the solver oracle shows an integrated
  versus pointwise gap;
- history corruption/scheduled sampling or overlap-aware chunk generation only
  if the seam replay proves exposure error;
- SSM action expert only if the error follows long sequence modeling after
  evidence/solver controls;
- native three-camera/depth/14-D bimanual outlet only after the physical ABI is
  explicit;
- visual-track or action-conditioned JEPA target only after existing Flow-DINO
  future evidence is shown insufficient;
- FLARE/LaWAM-style late-DiT future tokens or alignment only after the two WAM
  reachability arrows are both demonstrated; keep Teacher construction at zero
  in deployment and do not claim layer 6 is universally optimal.

Do not combine B-spine, MIP, a new solver, a new Teacher target and native RDT
adaptation in one run. They change different owners and would make a positive
or negative result uninterpretable.

## Practical priority decision

The most likely high-value order, based on current evidence, is:

```text
complete Schema30 behavior gate
  -> solver/time + seam + outer-consistency diagnostics
  -> choose the demonstrated owner:
       solver defect      -> non-uniform/higher-order integration candidate
       action compute     -> direct head versus final-DiT MIP
       future bandwidth   -> phase/contact Teacher target
       sequence exposure  -> chunk-history/overlap treatment
       outlet semantics   -> adapter/physical ABI unit
  -> only then revisit B-spine or a larger WAM representation
```

My present feasibility estimate is therefore not one number for “MIP +
B-spine”. The bounded components have different odds:

| Candidate | Feasibility | Reason |
|---|---|---|
| P0 diagnostic package | high | uses existing checkpoints/owners and can reject several expensive ideas |
| solver schedule/order improvement | medium-high if late local error is measured | minimal architectural intrusion, but current checkpoint may be solver-insensitive |
| Teacher-only phase/contact supervision | medium | strong literature fit and causal deployment boundary, but usefulness must reach action |
| final-DiT MIP alternative | medium | clean insertion boundary and credible method evidence, but cost and `2 x 5` interaction are unresolved |
| B-spine-0 alone | medium | bounded and cheap, but current evidence does not show representation is the dominant bottleneck |
| B-spine + MIP together | low as a first experiment | attribution is lost and the performance source cannot be identified |
| deploy another video WAM/planner | low | incompatible with the lightweight runtime and duplicates the present object W owner |

The performance-first implication is simple: preserve two, three or more
refinement stages whenever they buy complete behavior; remove calls only after
the behavior/cost frontier shows they are redundant.

## External source trace

Two WAM papers in this review expose a directly traceable official repository:

- [Prediction with Action code](https://github.com/Robert-gyj/Prediction_with_Action):
  the reported implementation concatenates RGB/action/depth tokens in one DiT
  and uses one sampled timestep for the three noising paths. This is evidence
  for a shared-time joint generator, not independent modality clocks.
- [LingBot-VA code](https://github.com/robbyant/lingbot-va): supports the
  causal-history/cache and asynchronous-execution reading of the paper.

The arXiv metadata inspected for LaWAM and AcrossWAM did not expose a separately
auditable official code link. Their architecture claims remain paper-level
evidence in this ledger rather than reproduced implementation facts.

## Primary references checked

- [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705)
- [AdaFlow](https://arxiv.org/abs/2402.04292)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864)
- [$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [DINO-WM](https://arxiv.org/abs/2411.04983)
- [Prediction with Action](https://arxiv.org/abs/2411.18179)
- [FlowPolicy](https://arxiv.org/abs/2412.04987)
- [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
- [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)](https://arxiv.org/abs/2502.19645)
- [FLARE: Robot Learning with Implicit World Modeling](https://arxiv.org/abs/2505.15659)
- [SmolVLA](https://arxiv.org/abs/2506.01844)
- [TinyVLA](https://arxiv.org/abs/2409.12514)
- [BEAST](https://arxiv.org/abs/2506.06072)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [Much Ado About Noising](https://arxiv.org/abs/2512.01809)
- [DreamZero / World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
- [Trajectory-Consistent Flow Matching](https://arxiv.org/abs/2605.08511)
- [C³ache: Accelerating World Action Models with Cross Inference Chunk Cache](https://arxiv.org/abs/2606.08962)
- [LaWAM](https://arxiv.org/abs/2606.15768)
- [ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?](https://arxiv.org/abs/2606.19531)
- [Spline Policy](https://arxiv.org/abs/2606.07386)
- [B-spline Policy](https://arxiv.org/abs/2607.09648)
- [Reflex: Real-Time VLA Control through Streaming Inference](https://arxiv.org/abs/2607.14695)
- [ChunkFlow](https://arxiv.org/abs/2607.12992)
- [AtVLA / Look Where It Matters](https://arxiv.org/abs/2608.02197)
- [GUARD](https://arxiv.org/abs/2608.04510)
- [What Matters for Latent Actions in Robot Learning](https://arxiv.org/abs/2608.19613)
- [SLIM-0.5B](https://arxiv.org/abs/2608.09771)
- [ForeTime-VLA](https://arxiv.org/abs/2608.20735)
- [Action-JND](https://arxiv.org/abs/2608.21247)
- [Mamba-based SmolVLA action expert](https://arxiv.org/abs/2608.21407)
- [TrAct](https://arxiv.org/abs/2608.24101)
- [V-Link](https://arxiv.org/abs/2608.25308)
- [PHR-VLA](https://arxiv.org/abs/2608.27609)
- [AdaVLA](https://arxiv.org/abs/2608.29208)
- [DriftingVLA](https://arxiv.org/abs/2608.29749)
- [CARD](https://arxiv.org/abs/2608.20763)
- [Temporal Forcing](https://arxiv.org/abs/2608.30643)
- [AcrossWAM1.0](https://arxiv.org/abs/2608.29937)
- [Causal World Modeling for Robot Control / LingBot-VA](https://arxiv.org/abs/2601.21998)
