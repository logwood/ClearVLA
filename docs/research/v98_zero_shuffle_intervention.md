# V98 zero/shuffle flow-address intervention

## 1. Experiment goal

Use a frozen V98 checkpoint to answer one narrow causal question:

> Does the V98 predicted raw flow provide a sample-specific spatial address that changes the raw reader output and improves the deployed action prediction?

This is an evaluation-only experiment. Do not train, resume, alter weights, enable the V99 zero-flow guard, or overwrite the original V98 run.

The supplied V98 log identifies the run as:

~~~text
run_dir=runs/v98_bs8_formal_20260722_193507
data_root=/data/liang.zhang/dataset/grab_pen_single/grab_pen_single
decoded_image_cache_dir=/home/sen.wang/workspace/robotics/clear/data/cache_336
dinov2_token_cache_dir=/home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336
batch_size=8
condition_mode=dinov2-cache
eval_inference_steps=5
seed=0
~~~

The log contains one completed checkpoint boundary at epoch 1, global step 2846. It later reaches epoch 2 batch 2380, but the trainer only writes "latest.pt" at epoch boundaries. Inspect the checkpoint metadata first. If it still says epoch 1 / step 2846, this probe evaluates the already-degrading epoch-1 model, not the exact epoch-2-batch-2380 state. Do not describe it as an epoch-2 intervention unless checkpoint metadata proves that epoch 2 was saved.

## 2. Why a direct evaluator command is insufficient

Two current implementation details must be handled:

1. In "clearvla/policy/flow_dino_evidence.py", the existing address zero/shuffle reader diagnostics run only when "flow_jepa_zero_flow_guard=1". V98 has "flow_jepa_zero_flow_guard=0", so it currently emits neither diagnostic.
2. In "clearvla/cli/eval_v39_policy.py", "dinov2-cache" currently forces "return_images=False". A V98 raw-image checkpoint still requires "history_obs_image". The standalone evaluator must use the same condition as the training entry point:

~~~python
return_images = (
    args.condition_mode != "dinov2-cache"
    or bool(int(policy_config.flow_jepa_raw_image_enabled))
)
~~~

Do not solve item 1 by changing the loaded V98 policy config to "flow_jepa_zero_flow_guard=1". That also changes the motion descriptor, uncertain-seed behavior and fallback-reader semantics, turning the experiment into V99 rather than a V98 intervention.

## 3. Required intervention semantics

Implement a transient, evaluation-only mode on "FlowDINOEvidenceEncoder":

~~~text
none | zero | shuffle
~~~

It must not be a parameter, buffer, policy-config field or checkpoint key. Provide a setter and a clear method, and always clear it in "finally".

Apply the intervention only to the flow tensor passed into "_RawDeformableAddressReader" inside "_read_raw_chart":

- "none": predicted V98 flow unchanged;
- "zero": pass "zeros_like(predicted_flow)";
- "shuffle": replace each sample's flow with another validation sample's flow while preserving camera identity.

For normal "B > 1", reshape to "[B,C,2,H,W]", deterministically roll along "B", then flatten back to "[B*C,2,H,W]". Do not roll the already flattened "B*C" axis because that can exchange top and wrist cameras. For "B == 1", use a deterministic spatial roll and label this fallback in the result.

Keep the original predicted flow for the motion/detail gate and every other consumer. Replace only the address coordinates. This isolates the claim “flow is a spatial address” rather than simultaneously changing masking, confidence or JEPA organization.

The existing audit-only reader metrics should also be computed for V98 in evaluation mode:

~~~text
flow_jepa_raw_address_zero_flow_value_delta
flow_jepa_raw_address_shuffled_flow_value_delta
~~~

The extra reader calls must not alter the baseline selector/value returned downstream.

## 4. End-to-end paired action ablation

Extend standalone V39/V40 evaluation with an explicit option such as:

~~~text
--flow-address-intervention-batches N
~~~

Select deterministic, evenly spaced validation batches using the evaluator's existing diagnostic-budget method. Use "N=4" for smoke and "N=32" for the full probe.

On every selected batch, run three deploy-style samples:

1. baseline "none";
2. "zero" address flow;
3. "shuffle" address flow.

All passes must use:

- the same frozen checkpoint and input batch;
- "system.eval()" and no gradient recording;
- the same pre-sampled noise tensor;
- the same five inference steps;
- proposal enabled;
- no execution-controller ablation;
- no V99 guard or loss changes.

Follow the existing execution-ablation pattern in "evaluate_v39_policy": set the transient mode, call "system.sample(...)", and clear it in "finally". Decode all predictions with the checkpoint action normalizer before computing metrics.

## 5. Required JSON output

Write one JSON result and one plain-text log into a new diagnostics directory.

### Run identity

~~~text
git_commit
git_dirty
checkpoint_path
checkpoint_schema
checkpoint_epoch
checkpoint_global_step
flow_jepa_raw_image_enabled
flow_jepa_zero_flow_guard
action_normalizer_fingerprint
split episode IDs
planned/finished/intervention batches
intervention samples and coverage
~~~

Fail immediately unless:

~~~text
flow_jepa_raw_image_enabled=1
flow_jepa_zero_flow_guard=0
~~~

### Baseline representation evidence

Aggregate on the same selected batches:

~~~text
flow_jepa_raw_flow_grid_magnitude
flow_jepa_seed_reliability
flow_jepa_raw_correlation_entropy
flow_jepa_raw_correlation_margin
flow_jepa_raw_address_flow_mass
flow_jepa_raw_address_fallback_mass
flow_jepa_raw_address_entropy
flow_jepa_raw_address_zero_flow_value_delta
flow_jepa_raw_address_shuffled_flow_value_delta
~~~

Keep V98 address semantics. Do not compare its absolute address mass directly with V99, whose fallback lane is defined differently.

### Paired action metrics

For "baseline", "zero" and "shuffle", report:

~~~text
full_rmse
first_rmse
first8_rmse
tail_rmse
tail_first_ratio
arm_full_rmse
gripper_full_rmse
decoded gripper precision/recall/F1
decoded gripper predicted/target event counts
~~~

Also report:

~~~text
zero_action_delta_rmse
shuffle_action_delta_rmse
zero_mse_delta_vs_baseline
shuffle_mse_delta_vs_baseline
~~~

"action_delta_rmse" compares the intervened prediction against the baseline prediction. "mse_delta_vs_baseline" means "MSE(intervention,target) - MSE(baseline,target)": positive means the intervention harmed prediction; negative means it improved prediction.

Store per-sample squared-error differences and compute paired bootstrap 95% confidence intervals with a fixed seed and at least 1000 resamples. Do not infer causality from a rounded aggregate whose interval crosses zero.

## 6. Interpretation

| Reader value deltas | Action MSE deltas | Interpretation |
|---|---|---|
| both approximately zero | both approximately zero | V98 flow address is ignored or already indistinguishable from zero |
| zero nonzero, shuffle approximately zero | little/no degradation | reader reacts mainly to magnitude/identity removal, not sample-specific correspondence |
| zero and shuffle nonzero | positive degradation with CI above zero | causally useful, sample-specific address |
| nonzero | negative degradation | learned V98 address is harmful; bypassing it helps |
| reader changes | action does not | downstream policy ignores or cancels raw-reader changes |

As a reporting band, absolute RMSE change below 0.001 and relative change below 1% is small. An effect above 3% with a paired confidence interval excluding zero is material. These are evidence bands, not training targets.

## 7. Checkpoint preflight

Run from the server repository root:

~~~bash
python -c "import torch; p=torch.load('runs/v98_bs8_formal_20260722_193507/checkpoints/latest.pt',map_location='cpu',weights_only=False); c=p['policy_config']; print({'schema':p.get('schema'),'epoch':p.get('epoch'),'global_step':p.get('global_step'),'raw':c.get('flow_jepa_raw_image_enabled'),'guard':c.get('flow_jepa_zero_flow_guard',0)})"
~~~

Expected minimum contract:

~~~text
schema=clearvla-v40-policy-checkpoint-v1
raw=1
guard=0
~~~

If "latest.pt" is epoch 1, optionally confirm that "best_full.pt" is the same epoch/step. Do not pretend the unsaved epoch-2-batch-2380 weights can be recovered from the log.

## 8. Four-batch smoke

After implementing the evaluator option:

~~~bash
mkdir -p runs/diagnostics/v98_zero_shuffle

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m clearvla.cli.eval_v39_policy \
  --checkpoint runs/v98_bs8_formal_20260722_193507/checkpoints/latest.pt \
  --data-root /data/liang.zhang/dataset/grab_pen_single/grab_pen_single \
  --decoded-image-cache-dir /home/sen.wang/workspace/robotics/clear/data/cache_336 \
  --condition-mode dinov2-cache \
  --dinov2-token-cache-dir /home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336 \
  --cache-resize 336 336 \
  --cameras top wrist \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --dtype bf16 \
  --eval-inference-steps 5 \
  --max-val-batches 4 \
  --flow-address-intervention-batches 4 \
  --out-json runs/diagnostics/v98_zero_shuffle/smoke.json \
  > runs/diagnostics/v98_zero_shuffle/smoke.log 2>&1
~~~

Smoke acceptance checks:

- checkpoint remains V98 ("raw=1", "guard=0");
- raw history RGB is present;
- exactly four paired baseline/zero/shuffle batches complete;
- required metrics are finite;
- the intervention reaches the reader;
- patched baseline matches the ordinary baseline within numerical tolerance;
- checkpoint hash and mtime do not change.

No T5 path is needed: the checkpoint contains the default goal-language tokens used by the trained policy.

## 9. Full 32-batch probe

~~~bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -m clearvla.cli.eval_v39_policy \
  --checkpoint runs/v98_bs8_formal_20260722_193507/checkpoints/latest.pt \
  --data-root /data/liang.zhang/dataset/grab_pen_single/grab_pen_single \
  --decoded-image-cache-dir /home/sen.wang/workspace/robotics/clear/data/cache_336 \
  --condition-mode dinov2-cache \
  --dinov2-token-cache-dir /home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336 \
  --cache-resize 336 336 \
  --cameras top wrist \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --dtype bf16 \
  --eval-inference-steps 5 \
  --max-val-batches 0 \
  --flow-address-intervention-batches 32 \
  --out-json runs/diagnostics/v98_zero_shuffle/full_32.json \
  > runs/diagnostics/v98_zero_shuffle/full_32.log 2>&1 &
~~~

Avoid sharing the GPU if possible. Contention should not systematically change deterministic paired metrics, but it complicates OOM and runtime diagnosis.

## 10. Deliverables

Return:

1. exact checkpoint metadata;
2. git commit and diff used for the audit-only intervention;
3. smoke command, log and JSON;
4. full command, log and JSON;
5. baseline/zero/shuffle table and paired confidence intervals;
6. a conclusion choosing among useful address dependence, magnitude-only sensitivity, downstream bypass, or harmful address;
7. the limitation if only an epoch-1 checkpoint exists.

Do not conclude merely that the probe ran normally. Tie the conclusion to paired deltas, confidence intervals and coverage.
