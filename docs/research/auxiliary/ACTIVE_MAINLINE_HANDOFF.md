# ClearVLA structural rebuild implementation handoff

Updated: 2026-09-21. This file describes this branch, not live server state.

## Identity

- Branch: `codex/structural-rebuild-20260921`.
- Immutable base: `0f07160d692ec8c8302880420a3d93d74512c39b`.
- Control: `a0e1f5d5b9736d37a51faec36039144f181d4b12` (read-only).
- Candidate: `configs/mainline/structural_rebuild_m1_calvin.json`.
- Implemented: M0 source isolation and M1a physical-step history path.
- Not implemented: M1b and the G1/G2/G3/S-task/W/P structural redesign.

The source at this commit, its workflow output and this candidate's serialized
config are authoritative. Do not infer full rebuild completion from the
branch name. The user authorized normal pushes to this isolated branch; no
master/old-branch update, deletion, checkpoint conversion or formal training
has been performed. GitHub API writes work; direct Git DNS in the local
execution container does not. Fixed Git archives were obtained through the
read-only CI artifact and checked against the baseline tree. Final head-to-head
comparison: 20 files, 3009 additions, 201 deletions; common raw-reader fixes were
already present in both heads and were not re-applied.

## Implemented source unit

New clock, typed HistoryTiming and separate timestamped state/action encoder;
actual dataset/loader and online checkpoint-policy adapters; S/coarse masking;
real-time and masked proposal memory; bounded CausalHistory (9 recent
observations, 24 commands, plus reset boundary); explicit config, component
and deployment ABI identity. Factory identity resolution has a single path.
The old S encoder is a deliberate control, not a fallback for missing new
metadata. New candidates require new training; no old checkpoint migration
allow-list was added.

## Verification record

Local preliminary baseline used Python 3.13.5 / PyTorch 2.10.0+cpu, not the
repository's required Python 3.12 / PyTorch 2.11.x. The initial untouched
mainline suite reported 352 passed, 1 skipped. Exact old-mode initialization
and parameter order matched for seeds 0 and 71. The new tests exercise the
real model's forward/backward, nonzero time-encoder parameter gradients,
normalized/native output, source-owned checkpoint save/exact reload, and
same-noise sampling. Fixtures replace image/T5/environment transport only
where specifically stated by a test; they do not replace the production
history/model path. CPU BF16 encoder tests are not CUDA BF16 tests.

Full-repository Ruff has pre-existing failures (785 in the initial base).
The first scoped Pyright comparison reported 105 inherited errors in the
changed legacy files and zero newly introduced errors after fixes. Raw
inventories and final counts belong in workflow/test artifacts; a scoped
regression pass is not a clean full-repository static audit. The first parallel
four-worker Pyright attempt exceeded local memory; subsequent scoped checks
run single-process. This is recorded, not hidden as a pass.

For final checks of this commit, use its `Structural rebuild audit` workflow
artifact (`runtime.json`, baseline/current JUnit, and `static-delta.json`).
The workflow installs the repository's Python/PyTorch generation. The source
bootstrap CI runs alone prove only syntax/archive/tooling, not neural tests.
GPU/CUDA, full dataset training, real checkpoint behavior and physical
closed-loop validation remain unrun. No job was started on the user's server.

## Resume work

Read the architecture contract and current repair plan first. Finish M1b's
remaining ingress semantics before declaring M1 complete or starting G1.
Each upstream change requires downstream interface, loss/mask and lifecycle
review; consumers already adapted for M1a still need their scheduled deeper
structural reviews. Do not reopen or expand the old behavior-probe checklist
as a prerequisite to this implementation.
