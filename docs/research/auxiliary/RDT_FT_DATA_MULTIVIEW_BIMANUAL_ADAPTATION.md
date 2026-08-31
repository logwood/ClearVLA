# RDT fine-tuning data: multiview and bimanual adaptation boundary

Status: active isolated compatibility line.  The complete algorithm-external
boundary is implemented and accepted on the real server through a finite
typed-batch loader smoke: hierarchical identity, manifest split, external-test
isolation, per-sample language, ordered RGB/cache access, and explicit
qpos/action chart profiles.  This acceptance constructs no model or optimizer
and does not declare the loss, three-camera model consumer, native bimanual
codec, depth path, or formal experiment adapted.

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
absolute root and filesystem discovery order from its semantic content.  Both
producer and loader canonicalize the flat root-relative POSIX identity; the
loader maps verified identities back to the caller's current episode indices.
This distinction is required by real task names such as `write_board_1` and
`write_board_1+1`: component-wise `pathlib` order and serialized string order
are not equivalent.

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
the typed bank and bounded DINO/loader smoke have now also passed on the real
server.  Their acceptance is not permission to substitute another split or
encoder and does not promote any model-side change.

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

The algorithm-external portion is now implemented and covered by both a
synthetic end-to-end test and the real bounded acceptance below.  A model
forward/backward remains a later model-side gate and is not claimed by either
loader test.

### Real bounded external acceptance (2026-08-31)

Commit `9a5611ede2133a5365d02e3a73b1a1fe5a6eb841` completed the real
`val`, one-episode acceptance without constructing a model, optimizer,
backward graph or checkpoint.  The full 6,131-episode inventory produced 6,120
eligible identities and 11 explicit short-trajectory exclusions.  The four
split identity sequences stayed byte-for-byte identical when the manifest's
canonical ordering bug was repaired; its corrected semantic SHA-256 is
`2442ecd9c382d14123449a5b72d408bad4bcf84b164f6104a0d615cf5925212b`.

The language artifact was assembled from 303 corpus-provided
`lang_embed_0.pt` files, not a downloaded encoder.  It contains 271 BF16
`[32,4096]` policy rows, records all eight repeated-text variant groups, is
71,241,050 bytes, and has file SHA-256
`e3870e6e83fa354e3fdd6f1eaecb7bf9b72c958eadfa8e01b10a830870d57c65`.
The selected real trajectory was
`rdt_data/airpods_on_second_layer/episode_13` (`T=609`).  Its bounded cache
materialized all three RGB cameras as `[609,3,256,768]` FP16 DINO tokens; the
unchanged model-facing adapter selected the ordered `(high,right_wrist)`
subset.

The accepted typed batch contained DINO history `[1,3,2,256,768]`, future DINO
`[1,12,2,256,768]`, raw RGB `[1,3,2,3,336,336]`, goal tokens
`[1,32,4096]`, right-arm action target `[1,24,7]`, future action/state
`[1,48,7]`, and executed-action history `[1,8,7]`.  The loader rederived the
complete source, manifest, language, cache and train-normalizer identities and
validated all values as finite.  The serialized report is
`/data/senwang/data/rdt_ft_data/bounded_smoke_val_1/typed_batch_smoke_20260831_114959.json`
with SHA-256
`dd1e8effc3dca0c8be1d48d8cb52f1131a81cb633614b2f4808e5549e88d9dc3`.
This closes the algorithm-external D0/D1 loader boundary only: the cache may
hold three cameras while the current model input still consumes two, and the
source may hold 14 coordinates while this accepted profile still consumes the
right-arm seven.

### First-round exact eight-task preparation (2026-08-31)

The first formal multitask candidate is deliberately bounded to eight task
directories after an offline audit of all 302 `rdt_data` tasks. The audit
aggregates native action/qpos and both grippers over every available episode,
uses the scalar HDF5 instruction as the only semantic source, checks the
high/right-wrist camera headers, and decodes deterministic audit frames. The
selection producer then reopens every selected episode, checks finite aligned
native 14-D arrays and all fixed typed windows, and decodes every selected
high/right-wrist RGB row. Five uniformly spaced high-camera frames from every
episode were also reviewed in task-complete contact sheets. The selected
behaviors are right-arm-only in the observed corpus: objects or receivers are
table/fixed-surface supported, with no left-arm brace, hold, alignment or
handoff required.

The deterministic task order and exact HDF5 instructions are:

1. `put_cherry_bowl` -- "Pick up the cherry on the right side of the glass and
   put it inside the glass."
2. `stack_tomato_cans` -- "Stack a can of tomatoes on the right onto the can of
   tomatoes on the left."
3. `draw_triangle` -- "Pick up the pen on the shelf to the right of the
   whiteboard and draw a triangle."
4. `wipe_glass_water` -- "Pick up the sponge from the table, wipe the water off
   the glass, and finally place the sponge back on the table."
5. `press_stapler` -- "Press the stapler down."
6. `grab_stick_into_bottle` -- "Grab a stick from the table and stick it in a
   plastic bottle."
7. `shake_glass` -- "Gently pick up the goblet, rotate it counterclockwise for
   two full turns (viewing from above), and then place it back on the table."
8. `pour_water_can2cup` -- "Pick up the metal can, pour some water into the
   cup, then place the can back on the table."

The eight tasks contain 179 eligible episodes and 81,237 frames. Their
manifest lanes contain `143/18/18` train/val/test episodes and
`54,648/6,711/6,990` valid typed windows. The source `/test/` partition stays
as a separate 26-episode `external_test` lane: it is neither selected nor
materialized and was not read for threshold choice. Task IDs control only
CPU-side audit, split ordering, sampling and logging metadata; the typed model
sample has no task ID/index/embedding field.

The small artifact identities are:

- 302-task audit: file
  `27617de19b4feff5a1c4a5f2afd4f2ad146ce322129f0e968a5f5bc45a7e4218`,
  canonical
  `adbf96943a8dbee1d37188a1bf9f3afc84b1fc003a806c95737082b809d8dc40`;
- selection manifest: file
  `c905730106e31b35dc89c38eea5b1585faa391cc7f66cc2bc1e77db8f21721b5`,
  canonical
  `99f082028be7d9b92f0be4ed02ce22b5367f5f4c7274bbb1064b403770d1fd6f`;
- corpus-derived typed language bank: 71,241,050 bytes, file
  `e3870e6e83fa354e3fdd6f1eaecb7bf9b72c958eadfa8e01b10a830870d57c65`;
- one shared train-only normalizer over 143 episodes/64,944 rows: file
  `8c0e6ff4ef6692abb7cc80b7b60474ac642bb44541a40f78333390ff56b9c899`,
  canonical
  `1aa44936eb3fa659270a2dcc2a0258fa1e888332a37de5034a56ccef09320e0c`;
- train-only right-gripper candidate audit: file
  `729b9f47328e53c81f54e85be78e795f80f74c5b0de6ba922a98b20dec519a6f`,
  canonical
  `3fb2bd082707a508d6c2eaff0d6edcc3bfce4728f495e9cfb07f00e355dfe9f0`.

No T5 encoder was loaded or downloaded. The 271-row language bank reuses the
corpus-provided `lang_embed_0.pt` rows and retains their provenance. One copy
of that bank, and no T5 model/checkpoint, was transferred to the local audit
directory.

The reusable DINO cache reads JPEG payloads directly from HDF5 and stores
three-view FP16 patch tokens in fixed
`(high,left_wrist,right_wrist)` order. The first-round model selects only
`(high,right_wrist)`. The exact estimate and realized token-array size are
both 95,831,087,488 bytes (89.25 GiB), consisting of 95,831,064,576 raw token
bytes plus 22,912 bytes of NPY headers. Including 179 episode metadata files,
the report and directories, the completed cache uses 95,832,104,978 logical
bytes and 95,833,804,800 allocated bytes. Its cache-report file SHA-256 is
`c47c8d3be8aa6f002d91c1969a82361bd5932b55637efbfdaa8e4da721d494b6`;
the full token-file inventory digest is
`2d9379804effa65968e4e8b19b032acd0b3e353f6dd3c468b774b9e26ee1833d`,
and the complete inventory canonical digest is
`2846e5b3b08e9568819e806218587616022e495bf676ed44218f89124110f847`.
The encoder inventory binds the already-local `facebook/dinov2-base` revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`; no encoder or checkpoint was
copied to the local workspace.

One unresolved gate remains deliberately visible. The right-gripper audit
uses only the selected train lane and reports continuous command values,
command deltas, exact sampler-boundary deltas, activity/persistence run
lengths and candidate quantiles. It adopts no event threshold, does not
inherit Pen's `0.1`, and marks shuffled formal training not ready because the
source does not define which continuous transitions own semantic event or
activity labels. Validation, known-task test and external test rows did not
participate in this decision. Until this source-semantic definition is
adopted, the formal train loader fails closed; the completed CPU/cache
preparation is not permission to start a long run.

### Eight-task experiment interface closure (2026-08-31)

The first D1 experiment outlet is implemented independently of the model
graph. A manifest-derived immutable CPU registry maps loaded episode indices to
the exact eight-task order. It has no sample field of its own: the model still
receives only real per-episode T5, observation/history and supervision.

Training now uses task-first information sampling. Batch slots rotate over the
task registry and only then receive the unchanged uniform/event/motion lane
ratio. With the adopted batch size eight, every batch owns one row per task.
Uniform traversal is task-local and without replacement until rollover;
event/motion repetition and fallback are also task-local, so a data-rich task
cannot silently fill another task's slot. The actual epoch sample count and
fraction for each task are archived.

Bounded validation uses a deterministic equal-count panel per task. The model
and deployment sampler execute once per row, after which detached CPU task
indices partition the existing prediction for task metrics. Each observed task
archives sample count, full physical RMSE, horizon bands `1..4`, `5..12` and
`13..24`, arm/gripper RMSE, decoded event precision/recall/F1 and event counts.
The same record includes sample-weighted micro and equal-observed-task macro
values plus expected/observed task coverage. An absent task is named as absent;
no zero score is synthesized. The console emits one bounded line per task and
one micro/macro line, while the complete nested record remains in
`metrics.jsonl`.

The interface adds no model parameter, optimizer owner, differentiable task
tensor or checkpoint tensor. Registry, sampler, validation panel, shared
normalizer/language identity and the one adopted event threshold are recorded
in run/data state. The threshold is a single CLI value bound to all three
consumers: sampler strata, continuous gripper-trajectory supervision and
decoded validation. The Pen path remains fixed at `0.10`; RDT accepts no
implicit or mismatched value. Local sampler/config/runtime regressions pass
`40/40`; remote mixed-model backward and deployment smoke remain pending.

The threshold gate is intentionally still open. Candidate train-only exact
sampler-boundary quantiles are:

| Candidate | Raw threshold | Event windows | Window fraction |
|---|---:|---:|---:|
| p90 | 0.146484375 | 54,250 | 0.992717 |
| p95 | 0.40283203125 | 44,724 | 0.818401 |
| p97.5 | 0.7599645257 | 27,502 | 0.503257 |

These high fractions are not evidence that most windows contain semantic
open/close events. They arise because the first policy row compares a
continuous command with observed qpos converted into the action chart. The
source supplies neither their equivalence nor an event annotation. A formal
run therefore requires an explicit decision about whether event ownership is
defined by command change, executed-state change, or another sourced rule;
choosing a convenient quantile alone is not sufficient.

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

There are two experiment families, but not two simultaneous jobs on one GPU:

- Schema28 pen remains the controlled architecture/recovery reference.
- The first RDT-data run is the adopted D1 eight-task/two-camera/right-arm 7-D
  interface experiment. It may start only after the gripper semantic boundary
  is adopted and the mixed batch-eight model smoke passes. Its claim is
  per-task episode generalization and outer-interface health, not three-camera
  or bimanual closure. Held-out-task generalization remains a separate later
  experiment and must not reuse the same label.
- D2 three-camera and D3 native 14-D bimanual experiments remain later,
  independently gated stages; their absence does not get relabeled as D1
  support.

No long run is spent on D0 or D2 preparation itself. The D1 outlet first uses
offline tests and a bounded mixed smoke, then becomes the initial formal
multitask run once its semantic gate closes. That run changes the data domain
relative to Pen but keeps the already verified two-camera/right-arm ABI.

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

The train-only adjacent-command audit adopts p95 `0.18310546875` raw command
units. Its total activity-window fraction is `0.269379`; the lowest nonzero
per-task coverage is `0.123409`, while `press_stapler=0` is retained as a
constant-command data fact. P97.5 would reduce `draw_triangle` coverage to
`0.03646`; p90 would broaden total activity to `0.326618`. The eight-task
config and formal launchers pin this value across sampler, continuous
trajectory supervision and decoded validation. The mixed model smoke and
formal entry points are therefore:

```bash
OUT_DIR=runs/clearvla_rdt_multitask8_smoke_TIMESTAMP \
bash scripts/smoke_rdt_multitask.sh

RDT_MAX_VAL_BATCHES=64 \
OUT_DIR=runs/clearvla_rdt_multitask8_TIMESTAMP \
bash scripts/train_rdt_multitask.sh
```

Both commands fail before data/model construction if an environment or CLI
override attempts to change the adopted threshold.
The formal default validation panel is 64 batches at batch eight, i.e. 64
deterministic rows per task; an explicit CLI override remains recorded.

## Unresolved assumptions blocking model edits

1. Audit both native gripper coordinates before D3. D1 already uses the
   selected right-gripper adjacent-command boundary and its train-only p95;
   the Pen threshold `0.10` is not inherited.
2. Finish the complete V120 decoder/execution/checkpoint review before choosing
   the bimanual execution-value field ABI.
3. Measure three-camera model-side CUDA memory; do not infer the formal batch
   size from the bounded cache/loader result or two-camera arithmetic alone.
4. Determine how the provided 26-episode `test/` partition relates to the
   internal validation and held-out-task protocols; its name alone does not
   define the scientific claim.  Any later held-out-task policy must group
   duplicate instruction identities before claiming language generalization.
5. Determine whether trustworthy camera calibration and depth units exist.
   Until then, neither may be inferred from task names or image dimensions.

No active model source may be edited while any assumption required by that
edit remains unresolved.
