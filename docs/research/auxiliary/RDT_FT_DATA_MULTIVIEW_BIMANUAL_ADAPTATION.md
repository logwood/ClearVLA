# RDT fine-tuning data: multiview and bimanual adaptation boundary

Status: active isolated compatibility line.  Its generic hierarchical-identity
and per-sample language-loader boundary is implemented locally; its RDT split,
camera, action, and experiment contracts are not yet selected by the formal
mainline configuration.

This document defines the isolated compatibility line for
`/data/rdt-ft-data/`.  It does not change the interpretation or acceptance of
the active Schema28 pen run.  The directory name `rdt-ft-data` identifies a
dataset; it does not select or reactivate any retired RDT/RDT2 policy code in
this repository.

## Line isolation

Two experiment lines have distinct responsibilities:

1. The Schema28 pen line diagnoses the current architecture on the established
   two-camera, seven-dimensional, single-task contract.
2. The RDT-data line establishes a new data/language/camera/action ABI and only
   then measures the same architecture on multiview bimanual tasks.

The lines may advance in source work at the same time, but formal GPU training
must not run concurrently on the same device.  A structural result from one
line is transferred to the other only as an explicit semantic commit.  Data
domain, camera count, action layout, and architecture are never changed in one
unattributable formal comparison.

The adaptation branch starts at commit
`76caa4846bee6bf36f96e32af4ad1aa5f561cccd`.  Existing Schema28 checkpoints
are not resume sources after camera or action ABI changes.

## Read-only dataset inventory (2026-08-31)

The initial remote inventory found approximately 1.7 TB.  The root has two
source partitions rather than one undifferentiated pool:

- `rdt_data/`: 302 task directories and 6,105 HDF5 episodes;
- `test/`: one task directory and 26 HDF5 episodes.

A subsequent all-file header scan covered all 6,131 episodes without error and
found:

- `action [T,14]` and `qpos [T,14]` in all 6,131 episodes;
- the canonical native order is left six joints, left gripper, right six
  joints, right gripper;
- three RGB streams in every sampled task:
  `cam_high`, `cam_left_wrist`, and `cam_right_wrist`;
- RGB rows are JPEG byte payloads stored in HDF5, not uncompressed
  `[T,H,W,3]` arrays;
- all three depth streams in 5,014 episodes and no depth stream in the other
  1,117; a one-episode-per-task view is 237/302 present in `rdt_data/`;
  depth rows are TIFF byte payloads;
- `base_action [T,2]` present but all-zero in the sampled inventory;
- both arms materially active in 287 of 302 sampled episodes, with 15
  left-near-static examples and no right-near-static examples;
- an HDF5 scalar `instruction` stored as UTF-8 bytes per episode, plus task JSON
  language variants;
- task-directory names are not a safe goal source.  At least one directory
  name and its HDF5/JSON instruction describe different tasks.

HDF5 instructions are internally constant within every one of the 303 task
directories.  They match the declared `instruction` in the available task
JSON files exactly.  There are 271 distinct original instructions, so the
deterministic T5 cache deduplicates exact UTF-8 instruction content while the
sample still retains its task and episode identities.

The official RDT implementation narrows two previously unresolved source
contracts:

- its `T5Embedder` is `google/t5-v1_1-xxl`, uses `AutoTokenizer` and
  `T5EncoderModel`, tokenizes with special tokens, attention mask, longest
  padding, truncation, and source maximum length 120;
- its HDF5 example divides left/right qpos grippers by `4.7908/4.7888` but
  left/right action grippers by `11.8997/13.9231`.  Therefore the state and
  command gripper coordinates are intentionally different source charts.

Sources:

- <https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/models/multimodal_encoder/t5_encoder.py>
- <https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/scripts/encode_lang_batch.py>
- <https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/data/hdf5_vla_dataset.py>

The source implementation also samples original, simplified, or expanded JSON
instructions.  The first ClearVLA compatibility run deliberately uses only the
exact HDF5 original instruction so the language distribution is deterministic;
language augmentation is a later controlled change.

Discovered episode lengths span 102--2,000 frames, with sampled median 549 and
sampled p10/p90 approximately 275/1,132.  Official configuration ancestry
indicates 25 Hz.  The sampled `action[t]` values are desired joint positions,
but are not numerically identical to `qpos[t+1]` (sampled median RMSE about
0.105).  The first implementation therefore treats `qpos[t]` as the observed
boundary and `action[t]` as the command/target chart; it must not silently
replace either with the other.

The all-episode numeric scan covered 3,222,474 rows.  Left/right gripper
commands have continuous ranges up to approximately 11.90/13.92 rather than a
shared binary scale.  Their boundary-relative absolute deltas have p95
approximately 0.0458/0.1373 and p99 approximately 0.490/0.618; a raw 0.10
cut selects about 4.04%/5.97% of rows.  This does not establish that 0.10 is a
correct event boundary.  It instead shows that the RDT grippers contain
continuous motion segments and require a source-backed transition/activity
definition before the Pen event semantics can be reused or replaced.

The global `action[t]-qpos[t+1]` RMSE is dominated by the gripper coordinates.
Per-coordinate-group values are approximately 0.0171 left joints, 0.0249
right joints, 0.592 left gripper and 0.656 right gripper.  Aggregate RMSE must
therefore never be used to infer that the joint-position timing contract is
uniform across semantic groups.

These are inventory facts, not behavior evidence.  Before a formal run they
must be regenerated by the repository audit command rather than copied from
this document.

## Current-source dataflow map

### Episode, split, language, and cache plane

| Boundary | Current producer/consumer | Blocking assumption |
|---|---|---|
| Episode discovery | `clearvla/data/hdf5_episode.py::find_hdf5_files` | recursive when its explicit glob is recursive; formal Pen config still supplies its existing flat glob |
| Episode identity | `LoadedEpisode.episode_id/cache_key` | validated suffix-free root-relative identity; flat data remains stem-compatible |
| Split | `clearvla/data/split.py` plus `tools/build_rdt_split_manifest.py` | deterministic per-task manifest exists, but formal loading still selects the Pen ordered 63/5/5 rule |
| Goal | `mainline/data/loading.py::GoalTemplate` | legacy one-row artifact or exact episode-to-instruction-bank row selected on CPU |
| Decoded RGB cache | `vision/decoded_image_store.py` | directory, metadata, and mmap lookup use `cache_key` |
| DINO cache writer | `cli/build_dinov2_token_cache.py` and `experiments/classic_policy_lab/rdt2_dinov2_cache.py` | writer/reuse lookup use `cache_key`; full RDT caches are not built yet |
| DINO mainline reader | `mainline/data/token_store.py` | reader/meta lookup use the same `cache_key` |
| Run identity | `mainline/runtime/identity.py` | episode/partition/task/instruction inventory plus decoded/DINO metadata digests are serialized |
| Camera config | `mainline/config.py::DataConfig` | exactly `top,wrist` and two named key fields |

The D0 unit must close every row together.  Fixing recursive discovery without
fixing cache and run identity would create silent cross-task cache reuse.
Fixing per-episode language without serializing its mapping would make a run
irreproducible.

### D0a implementation status: hierarchical identity and cache addressing

The first bounded D0 unit now gives each loaded episode a validated,
suffix-free root-relative `episode_id`.  For the observed RDT hierarchy this is
`source_partition/task/episode`, while `source_partition` and the task-local
`task_id` remain separate fields.  Flat Pen episodes retain exactly their old
stem as `episode_id` and cache directory.

The active decoded-image writer/reader, DINO preparation writer/reader,
mainline DINO reader, ordered split name, and checkpoint dataset identity all
use that same key.  Serialized v1 cache metadata deliberately retains the old
field name `episode_stem`; its value is the complete cache identity, which is
byte-for-byte unchanged for flat data.  Duplicate identities after removing
`.h5`/`.hdf5` fail before cache construction.  Synthetic regressions cover two
tasks with the same `episode_0`, both cache layers, both DINO readers, run
identity, and flat-v1 compatibility.

This unit changes no tensor axis, normalization, model forward, objective,
optimizer owner, checkpoint parameter, or gradient path.  D0b below implements
the source-partition-aware manifest and serialized per-instruction T5 mapping,
but formal configuration selection and real artifact materialization remain
separate execution gates.  Stem-keyed caches in inactive historical labs are
not an RDT-data execution path and have not been promoted into this
compatibility line.

### D0b implementation status: split and language artifacts

The offline per-task split builder now emits four disjoint identity lists:
`train`, `val`, known-task `test`, and isolated `external_test`.  Membership is
shuffled independently inside each `rdt_data` task by a stable task-local seed;
tasks with fewer than three episodes remain training-only because they cannot
support all three internal lanes.  Unknown source partitions fail closed.  The
manifest stores episode IDs rather than machine-local indices and excludes the
absolute root from its semantic content.

The language preparation tool reads the exact scalar HDF5 instructions,
deduplicates exact UTF-8 content, applies the official 120-token T5-v1.1-XXL
encoding contract, and stores the first 32 ClearVLA policy rows plus a real
mask in one typed cache.  The mainline loader accepts either that bank or the
legacy one-condition `.pt`; an episode selects its row before transfer to the
device.  Masked rows are exact zero, missing instruction mappings fail before
training, and the existing Pen path remains a one-row bank.  S, its condition
dropout, all losses, optimizer ownership, and checkpoint parameter shapes are
unchanged.

The split builder is not yet selected by the formal mainline config, and the
real T5 bank has not been encoded while the Schema28 GPU run is active.  Those
are execution gates, not permission to substitute another language encoder.

### Camera forward/backward plane

The typed observation interfaces already retain `[B,H,C,...]`, and most of the
observation, G, Teacher, W geometry, and P2 code reads `num_cameras` rather than
hard-coding two.  Camera identity is learned and ordered; no camera axis may be
pooled and reconstructed later.

Known non-generic boundaries are:

- `mainline/config.py` rejects any camera list/count except two;
- the loader owns only `top_camera_key` and `wrist_camera_key`;
- cache metadata and build CLIs own the same two-key convention;
- `model/transition.py::ControlledTransitionDynamics` explicitly rejects
  camera counts other than two, even though its protected G3 row shape is
  otherwise computed from `C`;
- all camera-count-dependent parameters and cache tensors require a fresh
  checkpoint;
- three cameras increase the RGB/DINO/observation/P1 working set.  Actual
  batch-one and batch-four CUDA peaks must be measured before selecting a
  formal batch size.

The first multiview contract is the ordered tuple
`(high, left_wrist, right_wrist)`.  It uses named, uncalibrated views unless a
later audit finds trustworthy intrinsics/extrinsics.  Camera embeddings and
per-camera 2-D geometry are allowed; invented calibration is not.

### Action forward/backward plane

The current seven-dimensional action path is semantically, not merely
dimensionally, single-gripper:

```text
native [6 joints, 1 gripper]
  -> PhysicalActionFieldCodec
  -> [6 joint absolute, 6 joint delta, 6-channel gripper field] = 18
  -> V120 physical lift / dynamic policy / execution decoder
  -> physical flow, decoded, delta, continuous gripper and motion objectives
  -> deployment decode
  -> arm/gripper/event/motion validation and matched interventions
```

The following active owners encode that meaning:

- `model/action_codec.py` defines `arm_dim=action_dim-1`, reads only the last
  native coordinate as gripper, and requires `action_dim=7`;
- `v120_core/config.py` derives the same single-gripper physical width;
- `v120_core/codec.py`, `v120_core/decoder.py`, and
  `v120_core/time_domain_mmdit.py` own the physical lift, output heads,
  execution candidate value field, and decode path;
- `training/losses.py` gives one gripper field/event trajectory and one
  aggregate arm-motion target;
- `data/dataset.py` and `data/loading.py` stratify windows using one
  `gripper_index=-1`;
- `runtime/evaluation.py`, `train.py`, and matched P2/execution diagnostics
  split `[..., :-1]` from `[..., -1:]` throughout;
- optimizer ownership, checkpoint shape validation, manifest identity, and
  logging names all observe the resulting tensor inventory.

Consequently, setting `action_dim=14` alone would misclassify the left gripper
as joint 7 and supervise only the right gripper.  Passing shape checks would be
a false implementation.

The intended bimanual native layout is explicit:

```text
limbs:
  left:  joint indices 0..5, gripper index 6
  right: joint indices 7..12, gripper index 13
shared carrier:
  both limbs remain visible to the same policy/world/action trajectory
typed closure:
  each limb owns its joint absolute/delta and gripper absolute/delta field
```

The minimally faithful physical field is two limb-local copies of the current
`6 joint absolute + 6 joint delta + 6 gripper` chart, for 36 channels total.
Whether execution value remains a two-family aggregate (joints/grippers) or
becomes limb-resolved is unresolved until the complete V120 execution
producer/consumer review closes.  Per-limb and per-gripper validation is
mandatory either way, so one failing limb cannot be hidden by the other.

## Staged implementation and experiment gates

### D0 -- data identity and language, no model change

- recursively enumerate task/episode HDF5 files;
- retain the top-level source partition (`rdt_data` or `test`) as explicit
  metadata rather than pooling it away;
- assign a normalized root-relative episode ID such as
  `rdt_data/task_name/episode_0`, while preserving the old flat ID for pen
  data;
- use that ID for decoded/DINO cache paths, metadata, run inventory, and split
  manifests;
- read the HDF5 instruction as the first deterministic language source;
- bind each sample to a precomputed T5 entry by instruction identity;
- support the provided external `test/` partition plus two explicit split
  meanings inside `rdt_data/`:
  per-task episode split (known-task new trajectories) and held-out-task split
  (new-task generalization);
- retain current flat-data behavior and existing v1 caches exactly when the
  root is flat and the episode ID equals its stem.

Acceptance is offline: complete inventory, no duplicate IDs, no cache-path
alias, no split overlap, every sample has one instruction/T5 key, and repeated
construction yields the same identities.  D0 authorizes no training claim.

### D1 -- right-arm, two-camera plumbing smoke

Use quantitatively selected left-near-static episodes, project only native
indices `7..13`, and retain `(high,right_wrist)`.  This is a loader/cache/model
plumbing check, not a representative dataset experiment and not evidence for
bimanual task quality.  Run dataset/static tests and one fresh batch-one
forward/backward only.

### D2 -- three-RGB-camera closure

Keep the D1 action projection fixed and change only the ordered camera tuple to
`(high,left_wrist,right_wrist)`.  Verify cache identity, every retained camera
axis, G/Teacher/W/P2 shapes, transition rows, gradients, five-step cache reuse,
and CUDA memory at batch one then batch four.  Do not add depth here.

### D3 -- native 14-D bimanual closure

With the three-camera path already closed, introduce the explicit two-limb
layout, two gripper fields/private trajectories, union event sampling, and
per-limb/per-gripper loss and validation accounting.  Preserve one common
policy carrier so cross-arm coordination is learnable.  Re-audit the graph in
both directions:

```text
qpos/action -> normalization -> layout -> physical field -> bottom heads
-> losses/decode -> validation/runtime/checkpoint/logging

loss/deployment action -> decoder -> physical field -> layout -> both native
limbs -> exact source qpos/action and per-limb diagnostics
```

Only fresh checkpoints are legal.  Static tests, one CPU/BF16 batch, one CUDA
batch-one backward, five-step deployment, and batch-four memory must close
before a formal run is proposed.

### D4 -- optional depth sidecar, deferred

Depth is not part of D0--D3.  A later depth design must decode the TIFF payload,
establish units and invalid-pixel semantics, and carry a real availability
mask.  Missing depth cannot be replaced by zeros and presented as an observed
modality.  RGB-only tasks must remain a first-class legal path.

## Formal experiment policy

There are two formal experiment families, but not two simultaneous GPU jobs:

- Schema28 pen remains the controlled architecture/recovery reference.
- The first RDT-data formal run occurs only after D0--D3 gates pass.  Its first
  split is per-task episode generalization; held-out-task generalization is a
  separate later experiment and must not reuse the same label.

No long run is spent on D0, D1, or D2.  These units consume only offline tests,
cache probes, and bounded batch-one/batch-four smoke checks.  The first RDT
long run changes the data domain relative to pen, but starts from a source
whose camera and action ABIs have already been verified independently.

## Unresolved assumptions blocking model edits

1. Encode and verify the real 271-row T5-v1.1-XXL cache after the current GPU
   run releases enough memory; no substitute embedding width is legal.
2. Confirm gripper direction and define continuous activity/transition
   semantics for both native gripper coordinates.  The Pen threshold `0.10`
   is an audit cut only and is not inherited as an objective boundary.
3. Finish the complete V120 decoder/execution/checkpoint review before choosing
   the bimanual execution-value field ABI.
4. Measure three-camera CUDA memory; do not infer the formal batch size from
   two-camera arithmetic alone.
5. Materialize the implemented per-task manifest on the full dataset; the
   later held-out-task rule remains a separate decision and must group
   duplicate instruction identities before claiming language generalization.
6. Determine how the provided 26-episode `test/` partition relates to the
   internal validation and held-out-task protocols; its name alone does not
   define the scientific claim.
7. Determine whether trustworthy camera calibration and depth units exist.
   Until then, neither may be inferred from task names or image dimensions.

No active model source may be edited while any assumption required by that
edit remains unresolved.
