# MIP / B-spine auxiliary research ledger

Updated: 2026-08-21

Status: recovered prospective research; not implemented, not an active
mainline issue, and not part of the adopted Schema25 replay. The former
Schema28 smoke/behavior prerequisite is a historical condition, not a current
activation gate. MIP enters active consideration only after explicit user
promotion; it must not be mixed into chronological architecture replay by
default.

## Research premise

Treat the whole ClearVLA policy as a hierarchical, object-centered WAM. The
static observation path constructs facts and predicted consequences once;
the dynamic policy evaluates an action hypothesis and emits an updated action
field. MIP therefore belongs at the final action-generation boundary rather
than in a second top-level world model.

The desired ownership split is:

```text
static WAM stem (once per observation)
  policy.encode_online
    -> Observation / G / S / W
    -> cached P1 factual detail
    -> cached G3 transition source and role table

dynamic B-spine (once per action stage)
  policy.velocity(noisy_physical, time, cache)
    -> action query
    -> dynamic P1
    -> P2 / P3
    -> controlled transition
    -> restored bottom
    -> active EvidenceLatentMMDiTActionDecoder
    -> physical velocity plus event/motion heads
```

Only the current 18-D physical action field and stage time cross from one stage
to the next. G/S/W, Teacher outputs, rollout posterior state, and new hidden MIP
state do not feed back across stages.

## Verified active source boundary

- `policy.encode_online` is the static cache builder.
- `policy.velocity` is the smallest correct per-stage execution unit. It
  recomputes dynamic P1, P2/P3, transition, layer contracts, and the complete
  bottom under the current physical action hypothesis.
- `ActionQueryEncoder` validates `time` and then discards it. The query directly
  depends on the physical field, history, and role. Time directly conditions
  dynamic P1 and the active Evidence decoder, and reaches other dynamic paths
  indirectly.
- The active decoder is imported by `restored_bottom.py` from
  `v120_core/time_domain_mmdit.py`. Its three `TimeDomainMMDiTBlock` instances
  feed the final action normalization and the velocity/event/motion readouts at
  `time_domain_mmdit.py:3090-3092`.
- `v120_core/decoder.py::_run_refinement/_velocity_prediction` is not the active
  Evidence decoder route. MIP must not be attached there.

Consequently, "integrate at the final DiT" means that parameter ownership stays
at the final action decoder, while the stage loop remains outside the decoder
and repeats the full dynamic B-spine. Repeating only `blocks[-1]` would reuse
upstream evidence constructed for an obsolete action hypothesis.

## First training adaptation

Keep the existing physical velocity head. Do not combine the first port with
direct-action regression, a second action network, copied DiT blocks, or a new
stage embedding.

For target physical action `a` and stage times
`T={t_0,...,t_{K-1}}`:

```text
z ~ N(0,I)                    # full batch, owned train_flow_generator
i ~ Uniform({0,...,K-1})      # one scalar stage shared by the whole batch
t = T[i]
source = 0 if i == 0 else z
x_t = (1-t) * source + t * a
u_t = a - source

prediction = policy.velocity(cache, x_t, t)
clean_estimate = x_t + (1-t) * prediction
```

The first ClearVLA-compatible version should sample one stage for the whole
batch, not mix stages per sample. `execution_value_terms` currently derives
`target_spread`, `reliability_scale`, and `normalization_scale` nonlinearly from
the whole batch. Mixing the stage-0 target `a` with later targets `a-z` changes
each other's controller objective and is not an unbiased stage-loss average.

Per-sample categorical sampling can remain a later fidelity experiment only
after execution loss is grouped by stage. With batch 8 and K=4 this would leave
roughly two samples per group, so its batch statistics are expected to be noisy.

If `L_i` is a stage loss, `w_i` its desired weight, and `p_i` its sampling
probability, the normalized objective is:

```text
L_stage = sum_i(w_i * L_i) / sum_i(w_i)
sampled_loss = w_i / (p_i * sum_j(w_j)) * L_i
```

Uniform sampling with all weights equal to one therefore needs no extra sample
multiplier. A fully expanded K-stage loss must divide by K. The first version
should omit configurable stage weights: the current action group also includes
static `history_proposal_loss`, so weighting the whole group would incorrectly
reweight a non-stage loss. Observation/G/S/W/Teacher/JEPA/proposal losses remain
once per batch; dynamic action/execution loss contributes once.

## MIP-2 fidelity and K greater than 2

The mirrored official MIP-2 velocity port uses training nodes `t={0,.9}`:

- at `t=0`, source is zero and the velocity target is `a`;
- at `t=.9`, source is Gaussian and the velocity target is `a-z`;
- deployment starts from a deterministic physical-origin field.

For predicted velocity `v_i`, define the clean estimate:

```text
a_hat_i = x_i + (1-t_i) * v_i
```

Two deployment transitions must be kept distinct:

```text
euler:
  x_next = x_i + (t_next-t_i) * v_i

clean_rescale:
  x_next = t_next * a_hat_i
```

They are identical for the two calls at `[0,.9]`, including the final move to
one, but differ for K greater than 2 when the model has prediction error.
`clean_rescale` is the natural extension of the paper's clean estimate;
multi-stage Euler is a separate solver experiment.

The first falsifiable comparison matrix is:

| Identity | Call times | Transition | Evidence status |
|---|---|---|---|
| Schema28 five-step control | `[0,.2,.4,.6,.8]` | Euler | active V120-compatible runtime |
| `mip2_velocity_reference` | `[0,.9]` | equivalent | direct MIP-2/official-port evidence |
| `mipk_clean_rescale_experimental` | `[0,.55,.75,.9]` | clean-rescale | preferred K>2 hypothesis; unvalidated |
| `mipk_euler_experimental` | `[0,.55,.75,.9]` | Euler | same-node solver control; unvalidated |

The `.55` training node still contains 45 percent Gaussian source. This schedule
changes training corruption as well as deployment step size; it is not merely a
finer late solver. Do not push the last action call to `.95` on the assumption
that a smaller final step must be better. Node choice, solver, model, and data
must be evaluated separately. Performance, not fewer calls, is the objective.

## Endpoint-head adaptation

The active runtime performs a complete dynamic call at `t=1` for event/motion
heads and discards its velocity. Discrete MIP action training ending at `.9`
would otherwise leave this deployment readout out of distribution.

Recommended first contract: `t1_explicit_supervision`.

- Make one additional complete dynamic forward using the clean target field at
  `t=1` during training.
- Train only event/motion heads from this call. Do not repeat velocity/action,
  execution, representation, Teacher, or other static losses.
- Normalize the event/motion budget to the existing one-supervision budget.
- Continue reading deployment event/motion heads at `t=1`.

An alternative `last_action_stage` ABI may train and deploy the heads from the
last action stage, saving the endpoint call. It must be an explicit paired
ablation, not a silent runtime substitution.

Report NFE as three values:

- action-update NFE: K;
- endpoint-head NFE: 1 when the `t=1` contract is used;
- total dynamic B-spine NFE: K+1.

The paper's MIP-2 NFE of two does not include ClearVLA's extra endpoint readout.

## Initialization and RNG contract

"Zero start" is ambiguous in the 18-D chart. Decoding an all-zero physical
field retains approximately `0.25 * action_state` through the delta path, so a
zero tensor is not a native zero action. Keep these initialization identities
separate:

- `physical_origin_zero`;
- `encoded_native_zero`;
- `encoded_hold`.

Use `physical_origin_zero` for the strict first velocity port, but record its
decoded initial action. The official port draws noise and then overwrites it
with zero at inference; strict `mip2_velocity_reference` should preserve that
draw cadence. A `no_draw_zero` runtime is a separate RNG ablation and ABI.

Training should reuse the serialized `train_flow_generator` with the fixed
order "draw full-batch z, then draw scalar stage". Formal validation and
rollout-gap diagnostics need independent generators so diagnostics cannot
perturb validation results.

## Proposed typed configuration and identity

```text
action_policy_mode:              v120_flow | mip_velocity
action_training_distribution:    v120_mirrored_beta_1_5_1 | mip_discrete_stage
action_stage_times:              tuple[float, ...]
action_stage_sampling:           continuous | batch_uniform_categorical
action_stage_transition:         euler | clean_rescale
action_initialization:           gaussian | physical_origin_zero | encoded_native_zero | encoded_hold
action_endpoint_head_source:     t1_explicit_supervision | last_action_stage
action_runtime_rng:              v120_noise | mip_reference_draw_then_zero | no_draw_zero
```

One field must own the training distribution; the config cannot simultaneously
claim the active mirrored Beta distribution and a MIP categorical distribution.
The canonical schedule owns runtime calls; `inference_steps` should be derived
or used only as a redundant cross-check. JSON list-to-tuple restoration,
finite/strictly increasing times, first time exactly zero, last time below one,
positive updates, and every cross-field combination must validate fail-closed.

Any executable MIP mode must receive its child architecture schema,
training/runtime ABI, formal config, run-context fields, and exact-resume
rejection in the same atomic identity change. Identity cannot be added after a
runnable branch exists. A Schema28 checkpoint must not exact-resume as MIP even
if parameter shapes match.

Fresh end-to-end training owns the primary result. Bottom-only migration can be
an initialization ablation only when run context records initialization mode,
source checkpoint identity/hash, and the actually loaded prefixes.

## Metrics and test gates

Per-stage metrics must aggregate numerator and sample count/support. Do not log
zero for a stage absent from the current batch. The existing accumulator only
supports a uniform whole-batch weight, so it cannot directly aggregate mixed
stage means.

Required diagnostics include:

- per-stage teacher and rollout input error in physical/native, arm/gripper,
  first/tail, and horizon bands;
- input/output physical RMS, update RMS, and per-stage gain or regression;
- last-stage versus endpoint event/motion disagreement;
- low-frequency fixed-probe gradient norm/cosine before and after decoder and
  execution-controller clipping;
- dynamic B-spine wall time, peak memory, and action/endpoint/total NFE.

Required tests include:

- unchanged Schema28 five-step config, RNG order, and call cadence;
- schedule tuple/list round-trip and config-digest changes;
- invalid mode/distribution/transition/initialization/endpoint/RNG combinations;
- K=2 Euler and clean-rescale numerical equivalence; K>2 identity separation;
- static stem once, action dynamic path K times, optional endpoint once, Teacher
  zero times during deployment;
- action query invariant to time while dynamic P1 and final velocity vary;
- save/restore of stage and noise draws; Schema28-to-MIP exact-resume rejection;
- static loss budget independent of K and dynamic loss contributing once;
- endpoint training source matching deployment source;
- execution ablations reusing the same explicit initial physical field;
- stage numerator/support aggregation and diagnostics not altering validation.

## Expected implementation surface

Semantic changes are expected in:

```text
clearvla/mainline/config.py
configs/mainline/<new-mip-config>.json
clearvla/mainline/training/losses.py
clearvla/mainline/training/engine.py
clearvla/mainline/train.py
clearvla/mainline/runtime/sampling.py
clearvla/mainline/runtime/evaluation.py
clearvla/mainline/runtime/logging.py
clearvla/tools/audit_policy_logs.py
clearvla/mainline/manifest.py
corresponding config/action/policy/runtime/manifest/checkpoint tests
```

The first port should not change the forward semantics of Observation/G/S/W,
the factual P1 reader, P2/P3, transition, restored bottom, active Evidence
MMDiT, `ActionOnlyPhysicalVelocityHead`, CVAE/workspace, or execution controller.

## Staged research plan

0. Finish Schema28's current implementation, fresh CUDA smoke, and early/late
   behavior gates. Until then, change only this auxiliary research ledger.
1. Freeze MIP-2 math, K>2 transition identities, endpoint, initialization, RNG,
   loss normalization, and their test vectors.
2. Create one atomic child identity with schema/ABI/config/run-context/rejection
   before enabling any MIP execution path.
3. Add batch-stage training with the fixed noise/stage RNG order, static losses
   once, dynamic losses once, and the matched endpoint-head budget.
4. Add physical-origin deployment, strict MIP-2, and the two K>2 transitions;
   pass finite/call-count/cache/RNG/poison-endpoint tests before claiming quality.
5. With matched data, seed, batch, optimizer, and training budget, compare the
   Schema28 five-step control, MIP-2, K>2 clean-rescale, and necessary Euler
   control.
6. Change K or nodes only when stage gain, rollout gap, endpoint behavior,
   gripper rebound, and gradient-conflict evidence jointly support the change.

## External research references

- MIP paper: `https://proceedings.iclr.cc/paper_files/paper/2026/file/9299ca9ed58731945e934adb5b71728c-Paper-Conference.pdf`
- Official pi0 velocity-port diff: `https://github.com/simchowitzlabpublic/much-ado-about-noising/blob/main/examples/pi0_pytorch.diff`
- Dense-Jump counter-control: `https://arxiv.org/abs/2509.13574`
- Align Your Steps secondary schedule-selection principle: `https://arxiv.org/abs/2404.14507`
