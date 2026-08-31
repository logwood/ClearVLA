# RDT fine-tuning data: multiview and bimanual adaptation boundary

Status: active isolated compatibility line.  The complete algorithm-external
boundary is implemented locally through a finite typed-batch loader smoke:
hierarchical identity, manifest split, external-test isolation, per-sample
language, ordered RGB/cache access, and explicit qpos/action chart profiles.
No RDT artifact has been materialized on the real server yet, and no model,
loss, optimizer, three-camera consumer, bimanual codec, or formal experiment
has been declared adapted.

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

The complete 2026-08-31 header scan found 6,131 episodes: 6,105 under
`rdt_data/` and 26 under the isolated `test/`, with 3,222,474 total frames.
Lengths span 30--2,864 with median 460.  Eleven `rdt_data/` episodes are
shorter than the 73 rows required by the fixed `-24/+48` typed window.  They
remain in the source inventory and are recorded by identity and length in the
split manifest, but cannot enter a train/validation/test lane.  The earlier
numeric sample had median 549 and p10/p90 approximately 275/1,132.  Official
configuration ancestry indicates 25 Hz.  The sampled `action[t]` values are
desired joint positions, but are not numerically identical to `qpos[t+1]`
(sampled median RMSE about 0.105).  The first implementation therefore treats
`qpos[t]` as the observed boundary and `action[t]` as the command/target chart;
it must not silently replace either with the other.

The all-episode numeric scan covered those 3,222,474 rows.  Left/right gripper
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
| Split | `clearvla/data/split.py` plus `tools/build_rdt_split_manifest.py` | Pen keeps ordered 63/5/5; the isolated RDT preset selects a verified four-lane manifest |
| Goal | `mainline/data/loading.py::GoalTemplate` | legacy one-row artifact or exact episode-to-instruction-bank row selected on CPU |
| Decoded RGB | `vision/decoded_image_store.py` or `vision/online_store.py` | strict mmap cache or direct JPEG decode with bounded process-local LRU |
| DINO cache writer | `cli/build_dinov2_token_cache.py` and `experiments/classic_policy_lab/rdt2_dinov2_cache.py` | arbitrary ordered cameras, direct-HDF5 ingress, and verified manifest-lane selection; full real cache is not built yet |
| DINO mainline reader | `mainline/data/token_store.py` | root-relative key plus explicit ordered subset of cached cameras |
| Run identity | `mainline/runtime/identity.py` | episode/partition/task/instruction inventory plus decoded/DINO metadata digests are serialized |
| Camera config | `mainline/config.py::DataConfig` | ordered names plus exact matching key map; active model consumption remains two-camera |
| Action/qpos chart | `data/action_chart.py` plus `mainline/data/dataset.py` | named source projection and explicit qpos-to-command boundary scale |

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
optimizer owner, checkpoint parameter, or gradient path.  D0b--D0d below
complete the source-partition manifest, serialized per-instruction T5 mapping,
chart profiles and cache ingress; real artifact materialization remains a
separate execution gate.  Stem-keyed caches in inactive historical labs are
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

The released corpus already contains one task-local `lang_embed_0.pt` beside
each task's episodes.  The official `encode_lang_batch.py` constructs original,
simplified and expanded instructions in that order and writes
`lang_embed_{i}.pt`, so index zero is the exact original HDF5 instruction.  The
language preparation tool therefore does not load or download T5: it reads the
exact scalar HDF5 instructions, packages the existing BF16 `[L,4096]` rows,
and stores the first 32 ClearVLA policy tokens plus a real mask in one typed
cache.  Local tokenizer verification checks each full source row length against
the official T5-v1.1-XXL contract without loading encoder weights.

The observed corpus has 6,131 episode references, 303 task-local candidate
files and 271 distinct original instructions.  Eight repeated instruction
texts have more than one numerically distinct task-local embedding.  These are
not averaged or normalized: the adopted `lexicographic` policy chooses the
first root-relative candidate while the bank records every candidate path,
file hash, tensor-storage hash and the exact retained-row hash.  The loader
revalidates that provenance and fails if the selected policy row is changed.
The generic encoder builder remains an explicit fallback, is local-only by
default, and may use the network only when `RDT_ALLOW_T5_DOWNLOAD=YES` is also
provided.

The generic loader accepts either that bank or the legacy one-condition `.pt`,
but manifest-backed RDT loading requires the typed bank.  It rederives the
complete source episode count and instruction
multiset—including typed-window-short exclusions—and compares both with the
bank metadata before selecting an episode row.  Masked rows are exact zero,
missing or stale instruction mappings fail before training, and only the
existing Pen path may remain a one-row bank.  S, its condition dropout, all
losses, optimizer ownership, and checkpoint parameter shapes are unchanged.

The isolated RDT data preset now selects that manifest and instruction-bank
ABI.  The real full source audit and v2 split manifest have been materialized;
the typed bank and bounded DINO/loader smoke remain execution gates, not
permission to substitute another split or encoder.

### D0c implementation status: selected split and source-chart profiles

`DataConfig.split_mode=manifest` now loads the content-digested per-task
manifest by root-relative episode identity.  The manifest separately digests
the complete source inventory and the typed-window-eligible inventory.  Every
source episode below the fixed 73-row minimum is recorded by identity and
length; the four exact split names must then cover every eligible episode once
and only once.  The loader rederives the deterministic per-task policy,
partition isolation, task counts, exclusions, glob and both inventories before
normalization.  A malformed episode, stale file list, unknown episode, changed
glob, policy drift, duplicate membership, or missing `external_test` therefore
fails closed.  `external_test` becomes a real fourth dataset lane but is never
iterated by ordinary training or validation.

The action/qpos boundary is an explicit versioned profile rather than a shape
heuristic:

```text
identity_7d_pen
rdt_right_arm_action_chart_v1
rdt_left_arm_action_chart_v1
rdt_bimanual_action_chart_v1
```

The right-arm profile selects source coordinates `7..13`; the left selects
`0..6`; the bimanual profile retains all 14.  Command targets remain in native
RDT action units.  Native qpos remains in its own state chart for the state
normalizer.  Only the observed boundary sent through the action normalizer is
converted into command units: left gripper is multiplied by
`11.8997/4.7908`, right by `13.9231/4.7888`.  Thus action delta, physical
decode and raw-unit evaluation share one command chart without erasing the
different qpos chart.  The profile identity, source widths, projection,
scales, gripper indices and digest enter run/data identity.

The bimanual profile prepares a truthful 14-D typed external representation;
it does not make the current single-gripper 7-D model codec consume it.

### D0d implementation status: ordered RGB and scalable cache ingress

Camera names are now an ordered unique tuple with either alias resolution or
an exact ordered `camera_key_overrides` mapping.  A custom name is legal only
with an explicit HDF5 key.  Decoded and DINO cache CLIs accept repeatable
`--camera-key NAME=HDF5/PATH` assignments and recursive globs.  Hierarchical
cache identity remains root-relative.

One cache produced in `(high,left_wrist,right_wrist)` order can serve an
ordered subset such as `(high,right_wrist)`: decoded RGB is camera-file local,
and the mainline DINO reader performs an explicit cached-camera index select.
No camera is pooled, duplicated, or inferred from position.  This lets a
future three-view algorithm reuse the same external cache instead of copying
the two-view cache.

Persisting decoded RGB is no longer mandatory.  `image_store_mode=hdf5-direct`
decodes JPEG rows through a bounded process-local frame/file LRU and returns
the same preprocessed RGB contract.  DINO preparation can use the same direct
path.  The full audit reports exact byte estimates for two- and three-camera
336-RGB and float16 256x768 DINO caches.  The preparation launcher refuses a
potentially multi-TB DINO build until
`RDT_CONFIRM_MULTI_TB_DINO_CACHE=YES` is explicitly supplied; decoded
materialization has a separate confirmation.

Both RGB and DINO cache writers can take the verified split manifest.  They
first validate the complete source inventory, eligible inventory, exact short
exclusions and deterministic split policy; only then do they select one named
lane and apply `--max-episodes`.  The loader-only smoke mirrors that scope:
normalizer and language identity remain global, but only the selected episode
needs DINO rows.  The formal `load_mainline_data` path is unchanged and still
requires cache metadata/arrays for every eligible episode, so a bounded cache
cannot accidentally become a training cache.

For the complete current inventory, the uncompressed estimates are about
1.985/2.978 TiB for two/three-camera decoded RGB and 2.305/3.457 TiB for
two/three-camera float16 DINO tokens.  These are storage bounds, not permission
to materialize either cache.

The selected RDT profile deliberately has no default sampling event threshold.
An unshuffled loader-only smoke is legal, but creation of the shuffled formal
train sampler fails until a source-chart threshold is explicitly adopted.
Pen's raw `0.10` therefore cannot leak into RDT merely because both profiles
place a gripper last.

### Camera forward/backward plane

The typed observation interfaces already retain `[B,H,C,...]`, and most of the
observation, G, Teacher, W geometry, and P2 code reads `num_cameras` rather than
hard-coding two.  Camera identity is learned and ordered; no camera axis may be
pooled and reconstructed later.

Known non-generic boundaries are:

- `ModelDimensions` still rejects model camera counts other than two, although
  `DataConfig`, the loader, and both cache layers now accept explicit ordered
  camera names and keys;
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
- support the provided external `test/` partition plus the adopted per-task
  episode split inside `rdt_data/` for known-task new trajectories;
- keep held-out-task generalization as a separately adopted future manifest
  policy rather than silently deriving it from the per-task split;
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

The algorithm-external portion is now implemented and covered by a synthetic
end-to-end HDF5 -> manifest -> three-camera DINO subset -> exact T5 row ->
finite right-arm typed-batch test.  The real artifact smoke and the model
forward/backward remain gates; neither is claimed by that synthetic test.

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

## Preparation and acceptance commands

Metadata-only preparation is the default and starts no encoder or training:

```bash
RDT_PREPARE_THROUGH=manifest bash scripts/prepare_rdt_ft_data.sh
```

Language preparation is explicit but uses the corpus-provided
`lang_embed_0.pt` rows, so it has no T5 encoder-weight or GPU requirement:

```bash
RDT_PREPARE_THROUGH=language bash scripts/prepare_rdt_ft_data.sh
```

Fresh re-encoding is not part of the acceptance path.  If it is deliberately
requested, cached local weights remain the default; allowing a model download
requires both explicit switches:

```bash
RDT_T5_SOURCE=encoder \
RDT_ALLOW_T5_DOWNLOAD=YES \
RDT_PREPARE_THROUGH=language \
bash scripts/prepare_rdt_ft_data.sh
```

The first real acceptance should use the bounded end-to-end launcher.  It
builds or reuses the global audit, v2 manifest and full instruction bank,
encodes at most one deterministic `val` episode into a separate three-camera
DINO cache, then constructs one finite right-arm/two-camera typed batch.  It
does not require the multi-TB acknowledgement because its limit is enforced
before encoding, and it never constructs a model or optimizer:

```bash
RDT_SMOKE_SPLIT=val \
RDT_SMOKE_EPISODE_LIMIT=1 \
bash scripts/prepare_rdt_ft_data_bounded_smoke.sh
```

After inspecting `audit_full.json`, choosing storage scope and releasing the
experiment GPU, the DINO stage requires an explicit multi-TB acknowledgement:

```bash
RDT_PREPARE_THROUGH=dino \
RDT_CONFIRM_MULTI_TB_DINO_CACHE=YES \
bash scripts/prepare_rdt_ft_data.sh
```

Once a compatible bounded or full cache already exists, the loader-only
acceptance command can be repeated without encoding.  It constructs no model
or optimizer and performs no backward or checkpoint write:

```bash
bash scripts/smoke_rdt_ft_data.sh
```

It validates one selected `val`, `test`, or `external_test` typed batch, all
manifest/language/cache identities, tensor shapes/dtypes, future offsets and
finite values.  It does not authorize `scripts/train_mainline.sh` with the RDT
config.

## Unresolved assumptions blocking model edits

1. Package and verify the real 271-row typed language bank from the existing
   task-local `lang_embed_0.pt` files; no substitute embedding width or silent
   re-encoding is legal.
2. Confirm gripper direction and define continuous activity/transition
   semantics for both native gripper coordinates.  The Pen threshold `0.10`
   is an audit cut only and is not inherited as an objective boundary.
3. Finish the complete V120 decoder/execution/checkpoint review before choosing
   the bimanual execution-value field ABI.
4. Measure three-camera CUDA memory; do not infer the formal batch size from
   two-camera arithmetic alone.
5. Materialize and validate the implemented per-task manifest on the full
   dataset; the later held-out-task rule remains a separate decision and must
   group duplicate instruction identities before claiming language
   generalization.
6. Determine how the provided 26-episode `test/` partition relates to the
   internal validation and held-out-task protocols; its name alone does not
   define the scientific claim.
7. Determine whether trustworthy camera calibration and depth units exist.
   Until then, neither may be inferred from task names or image dimensions.

No active model source may be edited while any assumption required by that
edit remains unresolved.
