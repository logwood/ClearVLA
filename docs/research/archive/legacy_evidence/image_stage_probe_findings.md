# Image-side implicit stage probe

## Question

Does the fixed `grab_pen_single` validation split contain implicit stage information that can be decoded from images, independently of the policy and training loss?

## Protocol

- Read-only probe on the existing image and DINO caches.
- Episode-disjoint fixed split: 63 train / 5 validation / 5 test episodes.
- Linear readout heads only; no policy checkpoint, loss, cache, or mainline implementation was modified.
- Image interfaces:
  - final-layer DINO mean tokens from top, wrist, or both cameras;
  - 2x2 spatially pooled DINO tokens from both cameras;
  - 12x12 average-pooled raw RGB from top, wrist, or both cameras.
- Targets:
  1. normalized episode progress regression;
  2. five equal-width progress bins;
  3. three event-relative stages (`pre`, `event`, `post`) around each episode's largest gripper-close event.
- The episode is the statistical unit. Reported intervals are 2,000-replicate episode bootstrap 95% confidence intervals. Nulls use 500 within-test-episode prediction permutations; the smallest possible permutation p-value is 1/501 = 0.002.

## Results

### Global trajectory stage

| Interface | Progress MAE | Progress Spearman rho | Five-bin accuracy | Permutation null accuracy |
|---|---:|---:|---:|---:|
| DINO top mean | 0.106 [0.088, 0.125] | 0.898 [0.842, 0.937] | 0.808 [0.776, 0.840] | 0.200 |
| DINO wrist mean | **0.096 [0.078, 0.114]** | **0.925 [0.890, 0.957]** | 0.777 [0.710, 0.845] | 0.200 |
| DINO both mean | 0.096 [0.072, 0.121] | 0.895 [0.795, 0.956] | 0.784 [0.738, 0.839] | 0.199 |
| DINO both 2x2 spatial | 0.105 [0.080, 0.131] | 0.912 [0.833, 0.957] | 0.803 [0.767, 0.839] | 0.201 |
| Raw RGB top | 0.215 [0.183, 0.275] | 0.495 [0.252, 0.653] | 0.647 [0.537, 0.734] | 0.199 |
| Raw RGB wrist | 0.223 [0.200, 0.249] | 0.415 [0.263, 0.579] | 0.671 [0.583, 0.760] | 0.199 |
| Raw RGB both | 0.221 [0.176, 0.287] | 0.594 [0.390, 0.717] | 0.691 [0.586, 0.796] | 0.199 |

All progress and five-bin results have permutation p = 0.002. On the five held-out test episodes, DINO-wrist progress Spearman rho is 0.858, 0.925, 0.915, 0.965, and 0.965.

### Stage around the close event

| Interface | Three-stage accuracy | Macro F1 | Permutation null accuracy |
|---|---:|---:|---:|
| DINO top mean | 0.492 [0.300, 0.733] | 0.399 [0.182, 0.705] | 0.333 |
| DINO wrist mean | **0.633 [0.475, 0.767]** | **0.554 [0.358, 0.728]** | 0.334 |
| DINO both mean | 0.600 [0.458, 0.733] | 0.521 [0.331, 0.701] | 0.333 |
| DINO both 2x2 spatial | 0.508 [0.417, 0.567] | 0.412 [0.289, 0.481] | 0.336 |
| Raw RGB top | 0.383 [0.333, 0.450] | 0.245 [0.167, 0.343] | 0.334 |
| Raw RGB wrist | **0.608 [0.492, 0.717]** | 0.523 [0.366, 0.680] | 0.330 |
| Raw RGB both | 0.542 [0.433, 0.625] | 0.438 [0.301, 0.527] | 0.331 |

All event-stage interfaces have within-episode permutation p = 0.002, although the effect is much weaker and more episode-dependent than global progress.

### Time-only control for event stages

A scalar normalized-progress classifier was fit from the same training anchors. Its mean test-episode accuracy is 0.417. The image readouts improve over this time-only baseline as follows:

| Interface | Accuracy gain over time only | Paired episode-bootstrap 95% CI |
|---|---:|---:|
| DINO wrist mean | +0.217 | [+0.017, +0.417] |
| Raw RGB wrist | +0.192 | [+0.025, +0.358] |

This control matters because the close event occurs in a fairly narrow part of the trajectory. The positive paired intervals indicate that the wrist image contains information beyond a fixed normalized-time schedule, but the estimate is based on only five test episodes.

## Conclusion

The dataset does contain implicit image-side stage features.

1. **The strongest signal is a global trajectory clock.** A linear readout from frozen DINO reaches rho about 0.93 for normalized progress and about 0.81 accuracy on five progress bins, despite episode-disjoint evaluation. This is strong enough to serve as a shortcut for any stage-conditioned model.
2. **The signal is present in the pixels, not invented by DINO.** A linear classifier on only 12x12 raw RGB reaches about 0.69 five-bin accuracy versus 0.20 chance.
3. **There is also weaker action-local stage information.** Wrist images distinguish pre-close, close-event, and post-close states at about 0.63 DINO / 0.61 raw-RGB accuracy versus 0.33 chance, and outperform a normalized-time-only control.
4. **The cue is mostly coarse/global.** DINO mean pooling is as good as or better than the 2x2 spatial readout. Wrist is best around the manipulation event; top view is best for coarse five-bin progress.

The safe interpretation is therefore: **strong implicit phase/trajectory information, plus moderate wrist-centered action-state information**. The probe does not establish externally annotated semantic task stages, and the local-event conclusion should be treated as preliminary because the held-out test set contains five episodes.

## Artifacts

- Probe: `clearvla/tools/probe_image_stage_readout.py`
- Full per-episode result: `docs/research/image_stage_probe_fixed_63_5_5_seed0.json`
- Remote result: `/home/sen.wang/workspace/robotics/clear/image_stage_probe_runs/fixed_63_5_5_seed0.json`
