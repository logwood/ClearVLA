# DINOv3 online / aligned multiview integration

## Source scope

Continue `codex/dinov3-online-aligned-vision-20260928` from encoder-boundary
commit `96733915f786379b6982515c6fb688e180cfc010`, which inherits M10a
`3db5760afa565b9bff911d341450f021c735ccd8`. No original branch is changed.
The historical cached DINOv2 configs remain explicit frozen controls.

This source checkpoint integrates the production routes below. The source is
intended for actual-data/pretrained-GPU interface admission, not a trained
policy-quality claim. The local 4GiB container could not finish integration
backward; complete CPU groups are run on a separate supported runner.
The policy integration fixture uses H32, 336px RGB and the production visual
width768/256-token boundary. It does NOT establish H512 GPU memory or speed.

## Selected new experiment

Main: `configs/mainline/dinov3_online_cumulative_calvin.json`.
M6n comparison: `configs/mainline/dinov3_online_m6n_calvin.json`.
Pooled-camera comparison: `configs/mainline/dinov3_online_alignment_control_calvin.json`.
All retain their original data selection, objectives, optimizer and cumulative
mechanisms. The new default uses raw HDF5 image reading. Existing decoded RGB
can still be selected explicitly; full dataset DINO tokens are not required.

Install the repository's supported environment (Python3.12, Torch2.11). The
existing locked `reference-models` extra selects Transformers4.57.6, which
contains DINOv3. No new unresolvable extra or silently stale lockfile is added.
Supply authorized **Hugging Face format** ViT-B/16 safetensors. Original Meta
`.pth` and timm-format weights are not silently interpreted as HF weights.
No authentication token should be placed in source/config/logs. Remote loading
requires explicit permission plus an immutable 40-character revision. Local
weight/config bytes are hashed, allowing identical files to be relocated.

## What is connected

* CPU workers read source-timed uint8 RGB for current history, future supports,
  instruction start, executed-world history and annotation endpoint. Unsupported
  future/endpoint rows stay unsupported and are not encoded. All provenance and
  split ownership are retained. CPU workers contain no CUDA neural module.
* One process-resident frozen DINOv3 online producer deduplicates compatible
  frame/camera requests in each batch, uses bounded microbatches and normal
  detached tensors, and scatters causal conditions and supervision separately.
  Future images never enter current policy conditions. `no_grad` does not block
  downstream gradients. No persistent dataset feature cache is introduced.
* Native 16x16 patch centers are transformed ONCE into a **derived full-image
  endpoint raster**. This is interpolation, not relabelling patch tokens or
  asserting invertible/no-loss resampling. At outer edges it repeats the nearest
  patch-center feature; it does not create new edge pixels. The half-pixel
  full-frame resize convention is explicit. RGB pyramids' padded stride4/8
  centers and the early masked context's stride2 centers are likewise mapped
  to full-image endpoints. DINO coarse/Teacher sampling uses endpoint-preserving
  interpolation. Existing G1/G2/P1/Teacher/reference/transport consumers receive
  the same declared chart, not mixtures of crop and full image.
  The new chart canonicalizes RGB/DINO memory layout after time gathering;
  the producer exports a contiguous raster and declares this in its identity.
  Tensor strides are not allowed to silently choose different train/adapter
  convolution layouts. Legacy charts retain their original arithmetic.
* G3 retains four actual value types per camera before fusion, using its single
  object law. New S binding compares task with those per-view values. Joint
  relations consume these values without expanding a pooled global attribute
  along C. Same K+null binding, no camera quota/nearest-object rule. Compact
  physical W remains language-free; a separate global fact summary still exists.
* Formal train/validation prepare online batches. The checkpoint includes exact
  encoder, raster, camera, dtype and microbatch identities. The real deployment
  adapter restores the same producer and rejects cross-family or changed visual
  identities. A new frozen backbone does not make old DINOv2 policies compatible.
  Existing two-pass action sampling, gripper/outlet conventions and causal clocks
  are unchanged. Legacy default serialization omits new default-only fields.

## Real interface admission (does NOT measure learned task success)

```bash
python -m scripts.probe_dinov3_deployment_ready \
  --config configs/mainline/dinov3_online_cumulative_calvin.json \
  --model /path/to/authorized/hf-dinov3-vitb16 \
  --device cuda --dtype bf16 --microbatch 2 --batch-size 1 --steps 2 \
  --output runs/dinov3_fullsize_interface_smoke
```

This loads actual data and pretrained weights, updates the actual policy,
saves and validates exact resume, restores the real deployment adapter, and
compares decoded native actions for a matched causal history. Result JSON
records stages, producer identity, request counts and failure details.
The smoke checkpoint is NOT a trained policy-quality checkpoint. `--compact`
is an explicit H32 interface diagnostic, never full-size acceptance.

Main matched training, after full-size interface admission:

```bash
python -m clearvla.mainline.train \
  --config configs/mainline/dinov3_online_cumulative_calvin.json \
  --dinov3-model /path/to/authorized/hf-dinov3-vitb16 \
  --device cuda --dtype bf16 --epochs 2 --batch-size 8 \
  --output-dir runs/dinov3_cumulative_new_run
```

Select an empty GPU with CUDA_VISIBLE_DEVICES in the shell. Batch8 is a proposed
experiment setting, NOT locally measured memory capacity. Begin with batch1
smoke, then qualify the intended batch. Start a new graph; do not resume old
DINOv2 checkpoints or overwrite caches/running experiments.

Deployment server for a **matching trained** new checkpoint:

```bash
python -m clearvla.benchmarks.bridge \
  --checkpoint /path/to/matching-new-checkpoint.pt \
  --dinov3-model /path/to/identical-hf-dinov3-vitb16 \
  --device cuda --host 127.0.0.1 --port 8765
```

The existing CALVIN client can connect; retain replan8 and the same scenario
panel when comparing behavior. No simulator, collision assets or controller
commands were changed. Check for target-first approaches with near distractors,
progress reversals, object drops and intermediate button toggles, not merely
attention weights or overall success count.

## Validation levels

Local source tests use explicitly marked artificial features, never a random
fallback in production. Full model-shape HF safetensors/load/forward is an
opt-in test using explicitly random weights (`CLEARVLA_TEST_HF_DINOV3=1`); it
checks library integration, not pretrained semantics. GPU/authorized weights
and real CALVIN closed-loop results must be recorded separately.

## Interrupted acceptance and exact-layout repair (2026-09-29 UTC)

Source was already published at `a183c777a29fd33b4993cb614367a6b57d63ce05`
when the conversation stopped. This was not another empty-branch recovery.
Run `36523911471` had one failing original adapter equality assertion (5/168
native action elements, maximum absolute difference 9.313225746154785e-9),
with 9 other production tests, 31 visual tests, 1 HF interface test and 90
regressions passing. Run `36524358555` passed the formal CLI test plus 89 runtime
tests, but did not rerun or supersede that failing adapter test.

Read-only diagnosis `36526758966` compared exact input values and parameter
hashes. They matched. Current DINO/RGB and reference tensors had different
memory strides; canonical contiguous inputs made the sampled actions exactly
equal. The controlled seed44 replay otherwise showed maximum native difference
2.2351741790771484e-8. A small real RGB-pyramid probe independently reproduced
the layout-dependent numerical effect. These tests use artificial visual
markers and do not explain the old policy's centimeter-scale behavior.

The fix in `de706c296bf4897ba6c9269f4bd056122a5ef876` preserves values, source
clocks and gradient paths; it adds new-chart-only contiguous observation ingress
and a contiguous online visual raster. Original `atol=0, rtol=0` adapter tests
are unchanged. A layout identity change intentionally invalidates intermediate
smoke checkpoints with the earlier raster identity; do not silently migrate
those or any cached DINOv2 policy.

Final acceptance run `36527063256` tested source
`de706c296bf4897ba6c9269f4bd056122a5ef876` with Python3.12/Torch2.11 CPU
and Transformers4.57.6. All six groups completed: 228 passed, zero failures,
errors or skips. Breakdown: production 10, visual
37, HF architecture interface 1,
formal CLI 1, related regression 90,
runtime/bridge 89. The original zero-tolerance deployed
action assertion passed unchanged. Final cleanup only updates this status and
removes temporary publication/diagnostic workflows; tested production source
is unchanged. Logs, JUnit and environment are retained in the delivery archive.


The first diagnosis helper mistakenly called `to_dict` instead of `as_dict`;
that helper error and the local OOM attempts are retained in delivery evidence,
not counted as passing tests or attributed to production DINOv3 behavior.
No authorized pretrained/CUDA/full-H512/robot closed-loop result is claimed.
