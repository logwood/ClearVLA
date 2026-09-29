# DINOv3 integration status

## Recovery checkpoint (2026-09-29 UTC)

Branch: `codex/dinov3-online-aligned-vision-20260928`.
Exact inherited source: `3db5760afa565b9bff911d341450f021c735ccd8`,
tree `9da55a749f4667461da4c953cf93814c29ef29ca`.
At recovery, the new branch still pointed to that same base, with zero
branch-associated Actions runs. No integration patch from the interrupted
attempt was found in the available attachments. This locates the publication
boundary, not the internal cause of the interrupted assistant session.

## Preserved source unit: encoder and coordinate boundary

`clearvla.vision.dinov3_online` provides a resident frozen Hugging Face DINOv3
ViT-B/16 encoder with strict safetensors loading, immutable remote revision or
local content hashes, full-FOV 256px preprocessing, explicit CLS/register
removal, fixed microbatch padding, and preserved leading time/camera axes.
No precomputed dataset token store is needed by this module. Output is the
native 16x16 patch chart, not a relabelled old endpoint-aligned chart.

The explicit sampler maps RGB pixel-center align-corners coordinates to patch
centers including resize half-pixel geometry. Reads outside the first/last
patch center use declared border replication, not newly observed edge detail.
These helpers are NOT silently substituted into historical consumers.

22 CPU source-contract tests passed. Their injected marker model tests axes,
token ownership, freezing and downstream gradients. It is not pretrained DINO.
The probe can run preprocessing only, or an actual user-supplied authorized
HF checkpoint. No pretrained encoder, full policy training, CUDA performance
or robot behavior acceptance is claimed for this checkpoint.

## Not ready for the requested training experiment

The existing training configs STILL use the old cached-token producer. Do not
set dino_cache to an empty path and assume this unit makes training online.
Remaining integration must land together with source tests:

1. CPU workers expose source-timed RGB requests, including history, future
   supervision, instruction reference, executed-world history and endpoints.
   A training-process encoder handles valid requests and compatible duplicates;
   future supervision remains disjoint from causal conditions.
2. All spatial consumers must adopt one explicit chart. Merely fixing a single
   G1/G2 lookup does not fix Teacher, pushforward, P1/P2 or reference reads.
3. Preserve per-camera object visual values before task-dependent fusion;
   expanding globally pooled attributes over C is not camera evidence.
4. Connect online train/validation and checkpoint/deployment ABI to the same
   weight/preprocessing/chart identity, add explicit new presets, and test
   loading, one real update, sampling and strict resume.
5. Run authorized pretrained forward, full-size CUDA smoke and matched robot
   experiments. Existing DINOv2 caches/weights remain frozen controls.

No task heuristics, labels, optimizer, physical action conventions, old
workflows or original branches are changed by this source unit.

## Reproduce this unit

```bash
python -m pytest tests/test_dinov3_online.py -q
python -m scripts.probe_dinov3_boundary \
  --images /path/to/static.png /path/to/wrist.png \
  --model /path/to/authorized/hf-dinov3-vitb16 \
  --device cuda --dtype bf16 --microbatch 2 \
  --output runs/dinov3_encoder_boundary
```

This is an encoder probe, not the formal training command. Install a
Transformers release containing DINOv3 in the experiment environment; there is
no random-weight or DINOv2 fallback. Private images are not uploaded to GitHub.

Interface references: official Transformers DINOv3 documentation and
facebookresearch/dinov3 image-transform documentation. No copied weights are
included in the source tree.
