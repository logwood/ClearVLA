# ClearVLA structural rebuild implementation handoff

Updated: 2026-09-21. This file describes this branch, not live server state.

## Identity and scope

- Branch: `codex/structural-rebuild-20260921`.
- Immutable base: `0f07160d692ec8c8302880420a3d93d74512c39b`.
- Read-only control: `a0e1f5d5b9736d37a51faec36039144f181d4b12`.
- Previous implemented unit: `329819bb12008d8ccf6211350bc64a8d8a4d7de1`, M1a real-time history.
- Current candidate: `configs/mainline/structural_rebuild_m1b_calvin.json`.
- Current source unit: M1b real-tail current states and source-owned future-label support.
- Still open: remaining M1b coordinate/rotation/unit/reset/seed review, then G1/G2/G3/S-task/W/P.

The current source commit, source-owned tests, serialized config and CI artifact
are authoritative. Do not infer a complete rebuild from the branch name. User
authorization covers normal pushes to this isolated branch. No master or old
branch update, deletion, data/cache write, checkpoint migration, formal training
or server job was performed. A CI-generated commit must be verified by its
publisher's final source tree and test artifact, not merely by a green bootstrap
job. Direct Git DNS in the local container is unavailable; GitHub API writes
and downloaded workflow artifacts are available.

## Current semantic unit

Read the architecture contract's M1b section and the repair plan's map. This is
not a dataset-only switch: the actual loader shares one source-support record
with action/future supervision; Teacher, recognizer, W/S targets, coarse/history
proposal, flow bridge, all formal target losses, execution value and validation
consume it. It never enters online conditioning. Unknown flow rows remain source
noise. Fixed W intervals require complete actual support. Validation uses real
label denominators and reports empty-band coverage. The old M1a candidate and
legacy defaults remain controls; source/semantic changes are not resume aliases.

## Verification and resource boundary

The local execution environment is Python 3.13.5 / PyTorch 2.10.0+cpu, not the
repository's required Python 3.12 / PyTorch 2.11.x. Local checks are supplemental.
M1a was first rerun locally (153 focused tests). The M1b first complete-suite
attempt was killed by the local 4 GiB cgroup memory limit (exit 137), not
accepted as a pass. Subsequent per-file runs also exposed the large policy
file exceeding local memory in both untouched baseline and candidate; local
rechecks shard that same collected node inventory, not the assertions. `run_structural_tests.py` now isolates each test
file in a fresh process without dropping tests; per-file logs and combined
JUnit retain failures, missing results and abnormal exits explicitly.

Final counts belong to the final commit's workflow artifacts: `runtime.json`,
`baseline-tests.xml`, `current-tests.xml`, per-file process inventory and
`static-delta.json`. The supported-runtime publisher reruns the untouched base
and the complete selected current inventory before publishing. The persistent
read-only audit explicitly selects Python 3.12. A static regression pass means
no NEW errors in reviewed files, not zero historical repository errors.

Tests exercise actual production model forward/backward, target-payload
invariance for every formal loss and parameter gradient, cached-loader mask
transport, validation subset denominators, continuous/binary outlet loss paths,
ordinary checkpoint save/exact reload, real deployment-checkpoint loading
with data-only support omitted, and same-noise online independence. Additional
CPU BF16 tail forward/backward checks are explicitly not CUDA BF16 checks.
Only declared image/token/language/environment I/O fixtures replace external
transport; they do not replace the neural or supervision path. CPU tests are
not CUDA BF16 tests, formal dataset training or physical closed-loop results.
No task-success improvement has been established by this source unit.

## Resume work

Finish M1b's coordinate/rotation/action-normalizer meanings, nominal physical
step metadata, compact downstream missing-evidence and reset distribution
review. Then enter G1 with its producer/consumer/loss contracts. Current source
support fixes do not close W's candidate-action/observed-future mismatch, Q5
endpoint conditioning or future entity/task-state lifecycle. Keep each next
semantic unit coherent; update downstream consumers and supervision with the
producer rather than leaving silent adapters or invented labels.
