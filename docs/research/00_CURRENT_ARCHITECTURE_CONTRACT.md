# Current ClearVLA Architecture Contract

Updated: 2026-08-01

This file is the compact source of truth for the active top representation.
Run labels describe experiments; they do not select source semantics. Historical
details belong in Git history and focused research notes, not in this contract.

## Agent quick contract

```text
capability:             object_intent_dynamics_323
manifest schema:        2
default run label:      v120
topology:               G1 G2 G3 / W1 W2 / P1 P2 P3
training:               fresh, single-stage end-to-end
future intervals:       4-8 / 8-16 / 16-32 / 32-48
formal language:        precomputed T5 .pt required
bottom compatibility:   evidence_mmdit_cvae_workspace_v1
launcher:               scripts/current_object_intent_dynamics_323.sh
smoke launcher:         scripts/current_object_intent_dynamics_323_smoke.sh
required capability:    object_intent_dynamics_323
```

`v120` is only the default log/run label. V119 remains reproducible ancestry.
The new top starts fresh; unchanged bottom migration is allowed only through an
explicit migration report. Direct `--resume` is rejected by the launcher.

The serialized `ArchitectureManifest` contains only capability, schema,
topology, intervals, language requirement, and bottom compatibility. Do not add
another version-wide `_validate_vXXX_*` contract.

## Active graph

```text
current RGB / DINO / raw pair / learned flow
  -> unchanged Pre-G observation bank
  -> G1-G3: dense current facts -> K=4 competitive global objects
  -> S: stateless object-intent organizer
       full T5 + ordered state/executed-action history + typed G objects
       zero-centred observed state/transport change (not completion/progress)
  -> online CoarseActionIntent
       online evidence only; its sole exit is W
  -> W1: object effects for 4-8 and 8-16
  -> W2: object effects for 16-32 and 32-48
       reads the full two-token W1 interval sequence
  -> FutureObjectDynamics
       the same directly supervised object is consumed by P2
  -> P1: one current high-resolution factual read
  -> P2: bounded, null-capable, exact-zero future-effect read
  -> zero-preserving consequence
  -> P3: factual / precision / effect / temporal / state-change compiler
  -> restricted bottom ingress
  -> unchanged Evidence MMDiT / CVAE / workspace / adaptive execution
```

Training-only path:

```text
future DINO supports
  -> no-grad Teacher-G object association
  -> four detached FutureObjectDynamics targets

future state/action + detached teacher effects
  -> FuturePlanRecognizer
  -> detached targets for online S
```

Teacher-G and the recognizer never enter deployment values. Replacing future
supports may change targets and losses only; it must not change deployment
actions.

## Non-negotiable invariants

1. Local `[B,C,8,8,M]` hypotheses are not persistent objects. A competitive
   binder creates global `K=4` objects while retaining camera and chart support.
   The local `M` owner prior is a conditional mixture: its complement is not
   null. Only physical candidate invalidity may transfer prior mass to null.
2. G2 to G3 ownership is `log(parent posterior) + bounded residual`. A zero
   residual exactly preserves the parent posterior.
3. Object, camera, space, type, and interval axes remain real until their named
   consumer. No reduction followed by `expand` may recreate an identity axis.
4. S reads the complete T5 sequence and ordered observable history. It receives
   no frame position, scalar progress, phase label, noisy action, or teacher.
   Its state-change value is built only from observed state deltas and current
   transport through bias-free projections; zero deltas and zero transport
   produce exact zero.
5. `CoarseActionIntent` is the only action-conditioned W input. It receives no
   target/noisy action and has no direct bottom bypass.
6. W1 owns the two near intervals; W2 owns the two far intervals. W2 may read
   W1 causally but may not replace its interval sequence with a mean.
7. The only W-to-P value is the directly supervised
   `FutureObjectDynamics`. No public world residual or hidden W carrier crosses
   this boundary.
8. Visibility, persistence, uncertainty, and validity describe selection or
   calibration. They do not repeatedly attenuate semantic effect values.
9. P1 performs one high-resolution current-fact read. P2/P3 cannot reopen the
   RGB/DINO bank, and P1 never reads future teacher values.
10. P2 content, intent, and coordinate scores are each bounded to `[-1,1]`;
    their temperatures are bounded to `[0.25,4]`. Its zero-vector normalization
    uses a `0.25` norm floor.
11. A neutral future field yields exact algebraic zero:
    `effect=0`, `interaction=0`, and `consequence=P1_fact`.
12. P3 precision and temporal lanes both consume consequence and the current
    action query. The temporal lane additionally consumes S temporal control.
    Its fifth lane is a `0.05`-scale state-change modulation: temporal/action
    context may shape a nonzero observed change but cannot synthesize one.
13. The bottom receives current state, the last typed state/action history, and
    the five-lane P3 bank. Generic task, full-history, proposal, and duplicated
    owned-trajectory aliases are forbidden. Object S does not provide an
    external completion probability or execution-terminal logit bias.
14. All online paths use ordinary autograd. No artificial gradient, hard gate,
    entropy target, route quota, progress loss, forced slot diversity, or
    forced nonzero flow is allowed.
15. Flags-off historical paths remain reproducible; the new capability must not
    silently mutate their semantics or load their top weights.

## Typed boundaries

```text
DenseFactChart
  public_scene_base
  semantic_chart / appearance_chart / geometry_chart
  coordinates / support / validity

ObjectFactSet
  object_content [B,K,D]
  typed_object_content
  object_to_chart [B,K,C,8,8]
  object_coordinates / transport_prior / support
  existence  # object-vs-null confidence on the object's own read support
  validity   # physical candidate support; the only legal W/P2 support mask

OnlineIntentState
  protected_goal_set
  observable_history_state
  interval_queries [B,4,D]
  temporal_queries [B,24,D]
  state_change_evidence  # zero-centred observed delta, not terminal evidence

FutureObjectDynamics
  current_content [B,K,D_dino]
  successor_content [B,4,K,D_dino]
  semantic_effect_delta [B,4,K,D_dino]
  future_address [B,4,K,C,8,8]
  transport_mean / transport_covariance
  visibility_change / persistence_change
  association_uncertainty / validity

PolicyPlanDeltaBank
  factual / precision / effect / temporal / state_change
```

Each type has one definition in
`clearvla/policy/object_intent_dynamics_323/`. Do not duplicate these semantic
types in `trunk.py` or select them by experiment version.

## Provenance table

| Module | Legal inputs | Forbidden inputs |
| --- | --- | --- |
| G | current observation bank, coordinates, flow, observable state/registers | task/language, history, proposal, noisy action, teacher |
| S | full T5, state history, executed-action history, typed G objects | frame position, scalar progress, noisy action, teacher |
| recognizer | future action/state and detached teacher effects; training only | deployment value path |
| W | typed G objects, S, one online CoarseActionIntent | target/noisy action, generic aliases, teacher, old world residual |
| P1 | G3 address, S query, current action-basis query | W effect, teacher, second visual read |
| P2 | supervised FutureObjectDynamics, S interval query, post-P1 action query | raw/DINO reopen, public W hidden, default geometry value |
| P3 | P1 fact, protected consequence, S temporal control, zero-centred observed state change, action query | unsupervised W carrier, RGB/DINO, second future head, completion/progress terminal bias |

## Loss ownership

- Action flow matching remains the dominant objective.
- Existing `flow_jepa_future_loss_weight` owns successor content, scale-aware
  semantic delta, transport/covariance, visibility/persistence, and detached
  uncertainty calibration.
- Existing `flow_jepa_interval_stage_loss_weight` owns chronological transition,
  object reconstruction, online-S matching, recognizer reconstruction, and
  coarse-action structure.
- Current-object existence is the read-conditioned object-vs-null confidence,
  never the fraction of total chart area allocated to an object and never a
  W/P2 gate. Physical current validity defines semantic target support;
  predicted future visibility cannot erase the W training target.
- Confidence may weight a target once. It never scales the P2 effect value.
- Retired global-S, current-reference, proposal, slot-reduced future, and free
  uncertainty-NLL aliases are not emitted by this capability.
- Deleted auxiliary weights are not reassigned to hide connection defects.

## Runtime contract

- Teacher association is built once per teacher-forced training or
  representation-validation batch and zero times in deployment. Permission to
  carry `future_training_pack` is owned by the explicit teacher-forced forward
  entry point, not by mutable `train()`/`eval()` module mode. The training-only
  recognizer is also absent from deployment sampling.
- Five-step deployment caches visual/G/S/W and P1 once. P2, P3, and the bottom
  action tower remain ODE-step dependent.
- Generic W1/W2, generic P2, and generic P3 are skipped; unused ancestral heads
  are frozen.
- Every trainable system parameter occurs in exactly one optimizer group.
- The inherited old Stage1 path is disabled by
  `stage1_initialization_enabled=0`; the CLI therefore ignores any historical
  `--stage1-checkpoint` argument carried by the V48 base launcher.
- Formal training requires the configured T5 file. Only a deliberately separate
  null-goal smoke may omit language.

## Current evidence and pending gates

Local CPU evidence currently passes:

- object permutation and G ownership-continuity tests;
- local-hypothesis prior/null separation, per-cell probability conservation,
  and read-conditioned existence tests;
- teacher isolation and target-only replacement tests;
- exact-neutral P2/consequence and bounded zero-Jacobian tests;
- W1 interval-order preservation and W2 sequence sensitivity;
- P3 precision/temporal consequence-dependency tests;
- exact-zero state-change evidence/lane and fifth-lane non-interference tests;
- complete-system BF16 forward/backward and optimizer ownership;
- deterministic eval/no-grad teacher-forced forward with FP32 targets, while
  the ordinary policy/deployment entry point rejects the same future pack;
- five-step cache behavior: eight top blocks execute
  `[1,1,1,0,0,5,0,0]`, with G/S/W/P1 once and Teacher zero times;
- flags-off sibling regressions and V120 log parser coverage.

The first V120 long run reached epoch 1 batch 640 with finite gradients and
about 2.05 s/batch, but it exposed a source error: local owner-prior mass was
passed as physical validity, and chart allocation share was exported as object
existence. That run is evidence for the old faulty semantics and must not be
continued as the repaired baseline. The source now keeps prior, validity,
cell-null mass, allocation share, read-conditioned existence, and physical
object validity separate;
a fresh completed server smoke is required. No frozen-checkpoint intervention
has been observed. Do not claim action improvement, useful W intervention, or
batch-eight memory below 22 GiB until new server evidence exists.

A later run also showed that the unsupervised object-S completion scalar moved
from about `0.99` to `0.001`, changing the external execution bias from roughly
`+0.08` to `-0.09` while the independent execution controller compensated.
That quantity was not identifiable as completion. The active source removes
both its direct terminal bias and its P3 terminal lane; the same fifth-lane
capacity now carries only algebraically zero-centred observable state change.
This semantic repair also requires a fresh run.

## Authoritative source map

```text
typed G/S/Teacher/W/P2/P3:
  clearvla/policy/object_intent_dynamics_323/
ancestral observation bank, dense G chart and P1:
  clearvla/policy/flow_dino_evidence.py
placement, role schedule and bottom ingress:
  clearvla/policy/trunk.py
provenance masks:
  clearvla/policy/trunk_primitives.py
losses, runtime cache, optimizer and logs:
  clearvla/experiments/observed_state_lab/policy_runtime_v39.py
configuration and manifest:
  clearvla/policy/config.py
  clearvla/cli/train_v40_policy.py
active launchers:
  scripts/current_object_intent_dynamics_323.sh
  scripts/current_object_intent_dynamics_323_smoke.sh
tests:
  tests/test_object_intent_dynamics_323.py
  tests/test_flow_dino_evidence.py
  tests/test_audit_policy_logs.py
active issue ledger:
  docs/research/TOP_ARCHITECTURE_ISSUE_LEDGER.md
```

## Historical boundary

V98, V107, V113, V114, V117, V118, and V119 remain ancestry and comparison
experiments. Their detailed decisions are recoverable from Git history,
versioned launchers, serialized run contexts, and focused research documents.
Do not rebuild the active graph from their names or paste their old contracts
back into this file.
