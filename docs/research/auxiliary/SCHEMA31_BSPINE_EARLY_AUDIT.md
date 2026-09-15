# Schema31 B-spine early audit

Status: temporary decision note, based on the first complete validation epoch
and training through approximately epoch 2 batch 1560. Replace its observations
after the complete epoch 2 and epoch 4 validations; do not treat it as an
architecture contract.

## Run identity

- Remote workspace: `/home/sen.wang/workspace/robotics/clear/schema31-bspine-pen-20260904`
- Run: `runs/schema31_pen_bspine0_b8_formal_20260904_173412`
- Local evidence: `new_logs/current/schema31/pen/{run_context.json,metrics.jsonl,schema31_pen_bspine0_b8_formal.log}`
- Source commit: `23203634a71384c23286f2c1b77b552ce7e23881`
- Dataset: Pen, 22,768 train / 1,432 validation windows, batch 8, seed 0,
  eight requested epochs
- Action normalizer digest: `cb686201cec4...`, matching the compared Schema29
  and Schema28-recovery Pen runs
- Active implementation: `fixed_bspline_coarse_detail_v1`, cubic, 12 control
  points, horizon 24, physical width 18. The run-name fragment `bspine0` is an
  implementation name and does not mean that the path is disabled.

The source contract is in
[`../00_CURRENT_ARCHITECTURE_CONTRACT.md`](../00_CURRENT_ARCHITECTURE_CONTRACT.md).
The implementation and intervention entry points are
[`../../../clearvla/mainline/v120_core/bspine.py`](../../../clearvla/mainline/v120_core/bspine.py),
[`../../../clearvla/mainline/v120_core/time_domain_mmdit.py`](../../../clearvla/mainline/v120_core/time_domain_mmdit.py),
and [`../../../clearvla/mainline/train.py`](../../../clearvla/mainline/train.py).

## Early decision

**Continue the run.** Schema31 has no new numerical failure or action-RMSE
collapse. The B-spline branch is learned, reaches the final action, and its
matched-noise zero ablation is already directionally beneficial. Evidence is
still too early and the attribution subset too small to accept Schema31 as the
new baseline; the next decisions belong at complete epoch 2 and epoch 4.

## What is established

The branch starts without perturbing the old path and grows gradually:

| Training position | B-spine update / raw action-token RMS |
|---|---:|
| batch 20 | 0.00026 |
| batch 100 | 0.00544 |
| batch 500 | 0.0381 |
| batch 1000 | 0.0474 |
| epoch 1 tail | about 0.0426 |
| epoch 2 partial tail | about 0.0500 |

Both views are active. At the current epoch 2 tail, coarse/detail token RMS is
approximately `0.0288 / 0.0334`; their gradient L2 is approximately
`0.0242 / 0.0040`. The roughly six-fold coarse-gradient lead is an observation,
not yet evidence that detail is dead. Coarse-plus-detail physical closure stays
near `1.2e-7`.

The largest B-spine owner gradient so far is `0.2076` (epoch 1 batch 2360),
with global pre-clip norm `2.0589`; it is not a numerical spike. Six finite
spikes occurred in epoch 1 and none had recurred in the observed part of epoch
2. Their count is comparable to Schema29 epoch 1 (7) and Schema28-recovery
epoch 1 (6), so spike count is not a stop signal here.

## Epoch 1 outcome

| Metric | Schema31 | Schema29 E1 | Schema28 recovery E1 |
|---|---:|---:|---:|
| full physical RMSE | 0.09659 | 0.09883 | 0.09787 |
| arm physical RMSE | 0.07126 | 0.07638 | 0.07387 |
| gripper physical RMSE | 0.18666 | 0.18268 | 0.18524 |
| band 1--4 RMSE | 0.04485 | 0.04460 | 0.04543 |
| band 5--12 RMSE | 0.07124 | 0.07238 | 0.07188 |
| band 13--24 RMSE | 0.12086 | 0.12402 | 0.12258 |
| decoded gripper event F1 | 0.2833 | 0.3025 | 0.2940 |

Schema31 is therefore slightly better directionally on full, arm, and far
RMSE, while gripper RMSE and event F1 are slightly worse. These runs share the
data, seed, and normalizer, but not the exact source base, so this is not a
strict one-variable A/B.

## Causal attribution and its boundary

At epoch 1, matched-noise B-spline intervention coverage is only `4.47%`:

| Execution | Physical RMSE on the covered subset |
|---|---:|
| primary | 0.08507 |
| zero only in refined pass | 0.08784 |
| zero over the full lifecycle | 0.08781 |

Removing the B-spline raises MSE in both interventions by about 3.2%. All six
arm/gripper by near/mid/far slices worsen: arm by about 4.8%, 5.7%, and 5.0%;
gripper by about 4.3%, 5.1%, and 0.3%. This establishes that the branch is not
merely producing non-zero internal telemetry: it affects and helps the decoded
action on the sampled subset.

The near-equality of refined-pass and full-lifecycle interventions means the
observed gain currently comes mostly from the final bottom/refined pass. Extra
propagation through proposal -> W rebuild -> refined pass is weak at this
stage. Keep this as an observation; the current coverage cannot support a
structural-failure claim.

## Unresolved observations

- Recheck gripper RMSE and decoded event F1 at complete epochs 2 and 4.
- Recheck whether detail-gradient participation grows or remains consistently
  much weaker than coarse participation.
- Recheck full-lifecycle minus refined-pass attribution with materially higher
  coverage.
- Typed S interval variation, W semantic/transport amplitude, and P2 geometry
  and consequence interaction remain weak inherited signals. B-spline has not
  repaired them, but there is no early evidence that it worsened them.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m clearvla.tools.audit_policy_logs new_logs/current/schema31/pen
```

Only logs and JSON metadata were copied locally; no checkpoint was downloaded.
