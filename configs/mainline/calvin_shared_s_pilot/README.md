# CALVIN shared S query pilot

This pair tests only goal innovation in the interval K query. `goal_history`
is the accepted graph; `history_only` retains history and every other language
path. Both use the shared 24x18 codec, `relative_command_adapter`, binary
gripper command, identical parameters at initialization, data, seed, sampler,
objectives and optimizer. Neither consumes the retired color-slot sidecar.

Both are fresh five-epoch experiments capped at 1000 batches per epoch. This
reshuffles training at each short epoch; it is not the first 5000 consecutive
batches of the old full-epoch schedule. Validation selects the same 64 evenly
spaced batches across the validation windows. This gives coverage across the
dataset but does not claim task stratification. Four sampling, four proposal
and two execution diagnostic batches are fixed for both arms.

Launch only from the verified isolated candidate commit and a supported
Python 3.12 / Torch 2.11 environment. The standard workspace launcher accepts
`--retain-checkpoint-epochs 2 5`; this preserves immutable copies at steps
2000 and 5000 when each epoch consumes its specified 1000 batches. Latest/best
remain available. The schedule is fixed at 5000 steps, with the usual 500-step
warmup. Record the GPU UUID and concurrent load; do not use differing schedules
or initialize one arm from a legacy direct-codec checkpoint.

Before comparing, check `run_context.json` source, split, dataset/cache and
language identities against `pilot_manifest.json`. The helper
`scripts.prepare_calvin_shared_s_pilot.compare_contexts` rejects extra config,
source, data or normalizer differences. Also verify both actual epoch batch
counts and interpreter/library versions; prepared files alone are not proof
that a run used the intended environment.

At retained checkpoints, use the existing fixed-observation NPZ probe:

```bash
python -m scripts.probe_calvin_intent_intervention_npz \
  --checkpoint CHECKPOINT --observation-dir VERIFIED_NPZ_DIRECTORY \
  --output PROBE_JSON --device cuda --dinov2-local-files-only \
  --run-velocity --run-action
```

Keep observation, history and noise fixed while changing instructions. Inspect
G invariance; S attention; coarse query/action; W; P2 posterior/consequence;
and the full two-pass action. Check K permutation invariance and finite all-null
fallback. A nonzero tensor/action change proves sensitivity only. Follow with
the same-initial-state CALVIN color/direction and drawer/slider closed-loop
panel, scoring target contact/displacement, non-target disturbance and success.
Do not infer target correctness from slot number or total loss alone.

This is an experiment specification, not a completed training or behavior
result. The original V1/V2 jobs are separate references.
