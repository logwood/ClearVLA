# V87 / V89 / V90 log comparison

## Scope and version identity

The middle attachment is not a plain V88 run. Its own banners identify it as
V89 typed-block execution on top of V88 learned dwell:

- V87: DCT state, `rho=0`, legacy execution/contraction behavior.
- V89: typed block budget, learned dwell, operation-value probes/loss enabled.
- V90: typed block budget, fixed dwell, operation-value probes/loss disabled,
  boundary-multiscale arm source enabled.

All three use seed 0, batch size 8, eight requested epochs, and the same stage-I
checkpoint path. V87 reports 877 missing checkpoint keys; V89/V90 report 892,
so V87 is a historical architecture comparator, while V89/V90 are the closer
pair. V89 and V90 still differ in both source process and execution control, so
they are not a clean one-variable source A/B.

Log completeness:

| Run | Batch rows | Complete validation epochs | Incomplete tail |
|---|---:|---:|---:|
| V87 | 551 | 1-3 | epoch 4, batch 2140/2972 |
| V89 learned | 404 | 1-2 | epoch 3, batch 2160/2972 |
| V90 fixed | 428 | 1-2 | epoch 3, batch 2640/2972 |

No run contains NaN, traceback, or a non-finite scalar.

## Main conclusion

V90 is numerically stable, but the current evidence does not show a net model
improvement. It makes the bridge much shorter and therefore makes early pflow
look excellent, but it removes a smaller fraction of that easier bridge and is
already plateauing above V87/V89. At the same epoch, validation is not better.

V89 is currently the most balanced same-epoch result. Its learned dwell reader
has real ranking signal and avoids a weak block, but its controller state is
collapsing toward one private direction and its contraction controls saturate
at full capacity. It selects *which function* to repeat, but does not yet save
capacity.

## Comparable validation results

Lower is better for MSE/RMSE. Precision/recall are continuous-gripper event
probes, not replacements for gripper trajectory RMSE.

| Epoch | Run | full MSE | arm RMSE | gripper RMSE | first8 RMSE | tail RMSE | grip P/R |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | V87 | 0.01310 | 0.09161 | 0.20327 | 0.08735 | 0.12581 | 0.093/0.388 |
| 1 | V89 | 0.01364 | 0.09126 | 0.21333 | 0.09078 | 0.12783 | 0.126/0.307 |
| 1 | V90 | 0.01487 | 0.09735 | 0.21737 | 0.09044 | 0.13498 | 0.189/0.333 |
| 2 | V87 | 0.01080 | 0.08239 | 0.18670 | 0.07279 | 0.11639 | 0.120/0.280 |
| 2 | V89 | **0.01037** | **0.07937** | **0.18653** | **0.07298** | **0.11354** | 0.159/0.336 |
| 2 | V90 | 0.01090 | 0.08060 | 0.19317 | 0.07396 | 0.11667 | 0.154/0.505 |

V90 does have the best epoch-2 arm-first8 RMSE (`0.04883`, versus V89
`0.05369`), but its arm-tail RMSE is worse (`0.09248` versus `0.08949`). The
event recall improvement therefore coexists with worse continuous gripper fit
and a stronger near-horizon bias.

V87 epoch 3 reaches `full_mse=0.00993`, `arm_rmse=0.08013`, and
`gripper_rmse=0.17600`. V89/V90 do not yet have an epoch-3 validation record,
so that row is not a fair cross-run comparison.

## Why V90 pflow is misleading

| Epoch | Run | initial refine error | final/native error | residual fraction |
|---:|---|---:|---:|---:|
| 1 | V87 | 1.9910 | 0.5565 | 0.280 |
| 1 | V89 | 1.9866 | 0.5261 | **0.265** |
| 1 | V90 | 0.9280 | 0.3018 | 0.325 |
| 2 | V87 | 1.9937 | 0.1371 | 0.069 |
| 2 | V89 | 1.9845 | 0.1354 | **0.068** |
| 2 | V90 | 0.9449 | 0.1545 | 0.163 |

V90's initial bridge error is less than half of V89's. Its DCT flow target
energy is also `0.576`, versus `1.441` for V87/V89. Absolute pflow therefore
cannot be treated as an architecture score after changing the source.

The threshold trajectory shows the same pattern. V90 reaches pflow `<0.5` at
epoch-1 batch 300, much earlier than V89 batch 640, but reaches `<0.1` only at
epoch-2 batch 240; V89 reaches it in epoch 1 batch 2240. Median pflow over
successive 25-row windows is:

- V87 epoch 4: `0.0948, 0.0695, 0.0716, 0.0608, 0.0545`.
- V89 epoch 3: `0.1159, 0.0836, 0.1150, 0.0923, 0.0913`.
- V90 epoch 3: `0.1366, 0.1161, 0.1219, 0.1266, 0.1490, 0.1361`.

The V90 curve is an early shortcut followed by a real plateau, not simply a
noisy continuation of the old descent.

## Source geometry

V90's sampler is behaving as coded:

- stochastic RMS/delta/acceleration: expected `0.800/0.803/1.369`, observed
  values track those numbers;
- covariance effective dimension `4.72/24`, condition number `21.0`;
- first-step standard deviation `0.581`, terminal standard deviation `1.048`.

Two experimental confounds matter:

1. V87/V89 use `rho=0` native white arm noise (`delta std=1.413`), whereas V90
   mixes a scale-0.8 multiscale covariance. It is not the advertised comparison
   against a rho-0.95 AR source.
2. Boundary-multiscale repeats `action_state` as the conditional mean at every
   horizon position, while rho-0 AR has zero conditional mean. V90 therefore
   changes both covariance and bridge mean. The lower initial refine error is
   the measured consequence.

The old source has a `1.413/0.0312 = 45.3x` arm delta-statistics mismatch. V90
reduces this to `0.803/0.0312 = 25.7x`, which is directionally useful but still
large. It also introduces a `1.80x` terminal/first marginal variance ratio. The
training flow reflects this directly: at epoch 2 V90 tail/first8 flow is
`0.1834/0.1036 = 1.77`, while V89 is `0.1323/0.1383 = 0.96`.

## DCT and manifold health

The coordinate implementation itself is healthy in all three runs:

- pflow and native pflow remain nearly identical;
- train-time arm/gripper output null fractions are effectively zero;
- round-trip error is about `1e-14` and tangent null is about `1e-5`;
- sampling pre-project null remains small.

V90's arm sampling null fraction rises from about `0.84-0.88%` to `1.09%`.
That is a mild regression, not the cause of the plateau.

The important V90 anomaly is the spectral coordinate warp. Its final warp RMS
grows from `0.84` at epoch 1 to `2.17` at epoch 2, with frequency spacing
`0.79..1.98`. V89 is `0.37 -> 0.27`, ending at `0.91..1.12`. V90's controller
is aggressively deforming the progressive DCT schedule to compensate for the
new source. This is consistent with the lower aperture masks and the late-flow
plateau; the DCT transform is correct, but the source and progressive reader
are not cooperating cleanly.

## Controller and dwell

The controller common-mode problem remains and is worse in V89/V90. The
reported quantity is direction participation, not matrix rank:

| Run | state participation E1 -> E2 | private participation E1 -> E2 | recurrent change E1 -> E2 |
|---|---:|---:|---:|
| V87 | 2.82 -> 2.77 | 2.71 -> 2.42 | 0.258 -> 0.317 |
| V89 | 2.53 -> 2.04 | **1.87 -> 1.01** | **0.135 -> 0.023** |
| V90 | 2.44 -> 2.17 | 1.80 -> 1.32 | 0.085 -> 0.052 |

The private pair cosine near `-0.142` is not evidence of eight specialized
tokens. Eight explicitly centered slots naturally approach `-1/(8-1)`. The
participation and recurrent-change measurements show that V89 private state is
nearly one-directional and close to a recurrent fixed point.

The input reader is not load-collapsed: source ownership remains about
`7.3/8` effective slots and load about `7.99/8`. Collapse happens after
retrieval, in recurrent state formation/readout.

V89's operation-value reader is nevertheless learning useful ordering:

- correlation `0.617 -> 0.628`;
- pairwise/decision accuracy `0.847 -> 0.900`;
- block usage at epoch 2: `0.730/0.074/0.196`;
- block gains: `0.380/-0.013/0.195`.

It correctly avoids the negative-gain middle block. The weakness is
calibration: predicted spread `0.113` versus target `0.235`, common-mode ratio
`0.996`, and candidate coverage only `0.619`. Learned dwell is not empty, but
it is mostly a ranking mechanism with a large useless common offset.

Nested capacity has saturated: V89 noisy/stage/low depth and keep values are
all approximately one by epoch 2-3. It chooses block dwell but does not reduce
effective depth. Candidate probing also raises training time from about
`3.58 s/batch` in V87 to `4.14 s/batch` in V89. V90 fixed execution runs at
about `3.32 s/batch`.

## Refinement, workspace, rollout, and gradients

Typed execution improves branch balance. V87 ends at noisy/stage/low write
fractions `0.57/0.08/0.08`; V89 is `0.35/0.21/0.12`; V90 is
`0.28/0.18/0.17`. The old noisy-branch domination is reduced.

V90's total action update norm is about `4.49` (`ratio=0.20`), compared with
V89 `2.57` (`0.114`) and V87 `2.11` (`0.094`) at epoch 2. Yet V90's normalized
refine gain is much lower. It is moving more but explaining less. Cancellation
is below the orthogonal baseline and branch cosine is positive, so this is
redundant/coarse direction, not destructive branch fighting.

V89/V90 also differ substantially at the workspace interface: V89 epoch-2
global/private attention is roughly `0.60/0.40`, while V90 is `0.01/0.99`;
promotion mean/std changes from `0.063/0.023` to `0.168/0.226`. This is another
reason the current V89/V90 pair cannot isolate arm source effects.

Rollout is almost invariant to all three variants: epoch-2 dynamics loss is
`0.1832/0.1833/0.1815`, std ratio about `0.962-0.964`, and milestone norm ratio
about `0.424-0.426`. The source/controller changes have not addressed the
rollout plateau near `0.15`.

V89 host-gate gradient norm is large (`~1.15` at epoch 2 versus V87 `~0.016`),
but block gradient remains healthy (`1.54` versus `1.35`). Gradient ownership is
strongly gate-heavy, but this log does not support saying that content blocks
are dead.

## Decision and next experiment

1. Finish only V90 epoch 3 to obtain its already-nearby validation record; do
   not spend several more epochs on it without a better result.
2. Treat V89 as the current balanced reference, not V90.
3. Run the missing strict source control: V89 typed blocks with fixed dwell,
   probes off, `arm_source=ar1`, `rho=0`, against the current fixed-dwell V90.
4. Before changing covariance again, split source mean and covariance into
   independent controls. Otherwise every source experiment conflates bridge
   centering with temporal geometry.
5. Keep the DCT chart. The evidence rejects the current V90 source/controller
   interaction, not DCT orthogonality.
6. Address controller private-state collapse architecturally. Do not add a
   diversity loss first; retrieval is already distributed, while recurrent
   state formation is where dimensionality disappears.
