# Causal unified CALVIN B-structure run profile (2026-10-10)

This record selects the CALVIN structure that has direct short-run evidence and
combines it with the repairs already admitted on
`codex/causal-unified-refactor-20261008`. It does not promote a 17/18 result
from another checkout as a result for this branch.

The retained B evidence is the fixed ABC-D push-color6 panel: the earlier B
short used canonical image ownership, RGB-D temporal correspondence
supervision, the mature 11012-update source checkpoint, BS8, a fresh Adam
state, learning rate (2e-5), and a 100-update warmup. Its standard closed
loop was 17/18, while the same evidence warns that this fixed panel can
overstate generalization. The target branch keeps the source-matched CALVIN
profile and adds its newer contracts:

- `entity_ownership_mode=canonical_image_v1`
- `identity_supervision_mode=rgbd_temporal_conditional_v2`
- `identity_correspondence=0.02` and `identity_source_prediction=0.02`
- `typed_object_value_mode=conditional_object_v1`
- `observed_outcome_mode=robot_world_before_proposal_v2`
- `typed_interval_gradient_mode=ordinary_v1`
- motion/event CALVIN frame weighting and positive gripper transition loss

The data remains source matched and is declared explicitly in both configs:
ABC-D HDF5 at `/data/senwang/data/calvin/converted/abc_d_full`, split
manifest
`/data/senwang/data/calvin/converted/abc_d_push_color6_raw_overlay_v2_splits_20260915.json`,
T5 bank `/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt`,
DINOv3 local model
`/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m`, and raw
CALVIN source `/data/senwang/data/calvin/raw/task_ABC_D`. The loader uses
both `top` and `wrist` cameras, direct HDF5 reads, online DINOv3, 7D
relative actions, three visual/state history frames, and eight executed action
history rows.

The reproducible profiles are:

- [causal_unified_calvin_b_short_20261010.json](../../configs/mainline/causal_unified_calvin_b_short_20261010.json):
  1024 BS8 updates, 256 validation batches, source clock 11012.
- [causal_unified_calvin_b_long_20261010.json](../../configs/mainline/causal_unified_calvin_b_long_20261010.json):
  eight full epochs, 256 validation batches, source clock 11012.

Both initialize only from the admitted checkpoint
`/data/senwang/clearvla/experiments/dinov3-gslot-identity-carrier-20261005/train-bs8-gpu0-r1/checkpoints/best.pt`
with `causal_unified_outcome_v1`, fresh optimizer moments, and the checkpoint
clock. A pre-existing target-branch long run was already launched before these
profiles were committed; its receipt and metrics remain the source of truth for
that run. Do not report behavior or 17/18 for the new profiles until a
source-matched closed-loop panel is run.
