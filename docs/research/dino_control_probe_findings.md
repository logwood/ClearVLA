# DINO final-layer control probe: first-pass findings

## Scope

- Dataset: 73 `grab_pen_single` episodes, split by episode as 63 train / 5 validation / 5 test.
- DINO cache: `facebook/dinov2-base`, final-layer patch tokens only, two cameras (`top`, `wrist`), 256 patches × 768 dimensions per camera.
- Sampling: stride 4, 7,663 samples; future/difference horizon 4 frames.
- Readouts: frozen cached DINO features with linear heads only; seed 0; validation early stopping.
- Interfaces: patch mean and fixed 2×2 spatial pooling, for top/wrist/both cameras.

The cache does not contain CLS or intermediate-layer tokens. The HDF5 files do not expose contact or object-pose labels, so those questions are not tested here.

## Data audit

- `action` and `qpos` are exactly identical in this dataset (`max |action-qpos| = 0`). Current-frame action readout is therefore not independent evidence of action-command information.
- Single-step gripper events (absolute change ≥5°) are sparse: 3.03% event versus 96.97% hold.
- For the 4-frame probe horizon, the event rate is 11.03% overall and 11.44% on the test split.

## Results

### Absolute state and future change from current DINO

| Interface | Current arm R² | Current gripper R² | Future arm Δ R² | Future gripper Δ R² |
|---|---:|---:|---:|---:|
| mean top | 0.651 | 0.619 | 0.299 | 0.250 |
| mean wrist | 0.659 | 0.766 | 0.445 | 0.223 |
| mean both | 0.840 | 0.784 | 0.442 | 0.312 |
| spatial top | 0.820 | 0.725 | 0.257 | 0.289 |
| spatial wrist | 0.709 | 0.787 | 0.493 | 0.268 |
| spatial both | **0.856** | **0.795** | 0.474 | **0.318** |

The final DINO patch cache clearly retains absolute robot and gripper state. Keeping coarse spatial layout improves the best two-camera result only modestly over patch mean (+0.016 arm R², +0.011 gripper R²).

Current-DINO-to-future-change performance may exploit task/trajectory priors. It is not by itself evidence that DINO encodes the realized motion.

### DINO difference to realized 4-frame motion

| ΔDINO interface | Arm Δ R² | Gripper Δ R² |
|---|---:|---:|
| mean top | 0.343 | 0.028 |
| mean wrist | 0.288 | **0.054** |
| mean both | **0.397** | 0.017 |
| spatial both | 0.331 | -0.097 |

Arm motion is moderately linearly readable from DINO changes. Uniform gripper-delta regression is nearly uninformative, but this metric is strongly affected by the 97% single-step hold rate.

Feature-change magnitude correlations support the same asymmetry: at lags 4–8, arm-motion Spearman reaches about 0.48–0.65, while gripper-motion Spearman remains about 0.01–0.11.

### Class-balanced gripper event detection from ΔDINO

The linear event detector predicts whether the gripper changes by at least 5° within four frames. Test positive rate is 11.44%; random ranking would have AP near 0.114.

| ΔDINO interface | AUROC | AP | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| mean top | 0.607 | 0.195 | 0.155 | 0.580 | 0.245 | 0.586 |
| mean wrist | 0.723 | 0.262 | 0.196 | 0.620 | 0.298 | 0.646 |
| mean both | 0.705 | 0.255 | 0.177 | 0.640 | 0.277 | 0.628 |
| spatial both | **0.736** | **0.288** | **0.209** | **0.660** | **0.317** | **0.669** |

The event signal is present and benefits from spatial patch layout, but remains noisy: the best AP is about 2.5× the event prior, while precision is only 0.21 at the class-weighted 0.5 threshold.

## First-pass conclusion

DINO has not discarded control information wholesale. Its final-layer tokens preserve absolute arm/gripper state well and realized arm motion moderately well. The weak point is fine gripper dynamics: events are detectable above chance, especially from wrist/spatial features, but the signal is substantially less accessible than absolute state or arm motion.

This does **not** yet prove that DINO itself removed the missing gripper detail. A matched raw-image event probe is required to separate DINO invariance from camera observability, label timing, and event sparsity.

## Highest-value next experiments

1. Run the same class-balanced 4-frame gripper event probe on decoded raw-image features. If raw image is much stronger than DINO, add a lightweight local image-detail branch; otherwise focus on labels/temporal alignment and downstream readers.
2. Use a one-query patch reader on the full 16×16 cached tokens, prioritizing the wrist camera. This tests whether 2×2 pooling still hides local gripper evidence without requiring a DINO model.
3. When the DINO model becomes available, cache layers 6/9/12 and compare them on the same event probe. Do not change the main model before this comparison.
