# ClearVLA structural rebuild implementation handoff

Updated: 2026-09-22. This describes source work, not a live server process.

## Identity and scope

- Branch: `codex/structural-rebuild-20260921`.
- Immutable baseline: `0f07160d692ec8c8302880420a3d93d74512c39b`.
- Read-only control: `a0e1f5d5b9736d37a51faec36039144f181d4b12`.
- Verified previous remote unit: M1b commit
  `12326a33bc33dad58603ddbb3cc48c892135b087`, source tree
  `632c4c0edd41c5adc22fca3a0fe9b41e64343f13`.
- Current local source: M1c state charts plus M1d source-timed visual evidence.
- Candidate: `configs/mainline/structural_rebuild_m1d_calvin.json`.
- G1/G2/G3, S-task, W-supervision and P redesign remain open.

The continuation archive was checked against all 756 recorded SHA-256 entries.
The M1b tree was independently reconstructed from the published M1a archive and
179278-byte recovery patch, producing exactly the remote tree above. The
recovery archive lacked usable Git history; local recovery commits are not
misrepresented as the remote source commit.

## Current implementation

One source-profile-specific encoder maps native CALVIN xyz/rpy/opening into
10 model features, using rotation columns before normalization. Actual affine
arithmetic is `x * scale + offset`; it was verified, not changed to another
normalizer convention. Dataset, future supervision and online policy share the
encoder. Future unavailable features are neutralized after rotation conversion.
Native statistics and seven-dimensional action/command/state boundaries remain
unchanged. Configuration and feature metadata reject incompatible old resumes.

The scoped review also traced CALVIN and ManiSkill clipping/actual-command
feedback through repository-owned simulation paths. This does not certify an
external evaluation script or an unknown physical controller. Visual duplicate
reset frames and fixed flow time are now handled by the separate M1d source-time
unit across raw/semantic losses, temporal/G reads, future queries, S/W rates
and Teacher. The native current image remains valid when motion is missing.

## Verification

Local execution is Python 3.13.5 / PyTorch 2.10.0+cpu, not the required Python
3.12 / PyTorch 2.11.x. Results are supplemental unless a matching CI artifact
explicitly records the supported runtime. There is no CUDA/BF16 hardware run,
formal dataset training, real checkpoint success or physical closed-loop result.

M1c completed 552 collected regression cases (551 passed, one skipped) on the
local supplemental CPU runtime. M1c has 36 focused tests: legacy affine equivalence, independent rotation
formula and equivalent Euler charts, wrap-boundary continuity, malformed-input
admission, config/ABI identity, production dataset/online equivalence at reset
and tail centers, actual optimizer updates with ten-dimensional states and
seven-dimensional actions, ordinary checkpoint save/exact reload, deployment
loader and same-noise action equality. External image/token/environment I/O is
replaced only where explicitly declared; neural/state paths are production code.

Scoped Ruff reports zero diagnostics; differential Pyright reports zero new
errors and no new missing imports. Inherited diagnostics and library-stub
warnings remain recorded. Full selected regression results belong to the
artifact, not an assumed pass: local runs shard the exact collected inventory
into bounded processes because the execution container has a 4 GiB memory
limit. Kills, timeouts and missing JUnit count as failures, not skips.

M1d has 26 focused tests, including source-time/frame support, masked flow
losses, actual G/S/W/Teacher propagation, duplicate-payload and gradient
isolation, future-query missing-motion tests, actual G-block masking, CPU BF16
encoder forward/backward, real model training, checkpoint/deployment metadata
and same-noise action equality. M1d scoped Ruff has zero diagnostics and
Pyright has zero new errors or missing imports; inherited core diagnostics and
library-stub warnings remain. Broader M1d regression results must be read from
the artifact, not inferred from this focused result. Fixed-time M1c mode has
bit-identical initialized parameters/order and same-noise sampled actions
between the two local source trees at seeds 0 and 71.

GitHub publication of M1c was staged from remote M1b with an exact, hash-checked
source patch. The reviewed M1c source tree is
`9a8b27fee31f51d20908231d3c8bcc3703311003`; a retained publisher workflow is
separate CI tooling. Its run and final published commit must be checked; the
transport commit by itself is NOT source publication. No workflow is silently
removed using the Actions token. Local recovery history remains synthetic,
with exact source-tree/patch identities documented separately.

## Current M2 implementation and verification

M1c is published as `c8341571e2eec12799a90c642aee634614368b86`.
Its reviewed source tree is `9a8b27fee31f51d20908231d3c8bcc3703311003`;
with the retained publisher workflow the remote tree is
`ab7fac4cebb2f1836b01cf9268e56a3199623b85`. Workflow `35672986893`
verified 552 cases (551 passed, one skipped, no failures/errors) on Python
3.12.14 / PyTorch 2.11.0 CPU and completed its normal fast-forward publication.

M1d is a locally frozen source unit with 580 cases (579 passed, one skipped,
no failures/errors). Scoped Ruff is clean and there are zero new Pyright
errors or missing imports. Its subsequent remote publication must be verified
independently; a temporary payload commit is not the implemented source.

M2's new explicit candidate is
`configs/mainline/structural_rebuild_m2_calvin.json`. Twenty-one focused tests
passed in local Python 3.13.5 / PyTorch 2.10.0 CPU. They cover same-moment
separated distributions, actual pointwise support/G2 prior use, G3 observed
content instead of empty-center sampling, all-257-channel tiling and backward
reference equality, local P1 patches, extreme and unsupported payload, query
partition/candidate permutation, real policy gradients to G1 without Teacher,
CPU BF16, optimizer update and exact checkpoint/deployment same-noise output.
Fixtures replace only named external transport/identity; real neural paths run.
M2 scoped Ruff is clean with zero added Pyright errors/imports; inherited
diagnostics remain. The full regression and disabled-mode equivalence are
recorded separately in the runtime artifact; do not infer their completion
from this focused result. GPU/CUDA BF16, full-data training, memory/latency at
production batch sizes and physical closed-loop behavior remain unrun.

## Current M3 unit and publication separation

M1d is remotely published as `18b0e5dffdca687025cd3bdf39a96cf1cb82cecf`,
with source tree `5c9a4b8fa9cac9c30466b9eaa7b1a573ffdd35e3` and retained-CI
remote tree `2b45471d5df3ec9e421cbba2a181c38475e8ae16`. Workflow
`35674979805` completed source verification and normal publication on Python
3.12.14 / PyTorch 2.11.0 CPU: 580 cases, 579 passed, one skipped, zero failures
or errors; scoped Ruff clean and zero newly introduced Pyright errors.

M2 is a separately frozen local unit; consult its test artifacts and remote
publication before assigning a remote SHA. M3 is the next local candidate,
`configs/mainline/structural_rebuild_m3_calvin.json`. It couples local location
ownership, not global physical identity or task binding. Its exact test/static
results belong with its source archive and workflow; do not infer publication
or test completion from this handoff. Local runtime remains Python 3.13.5 /
PyTorch 2.10.0 CPU. GPU and learned closed-loop behavior remain unrun.

M3 tests use real production encoders, grounding, Teacher, action decoder,
optimizer and ordinary checkpoint/deployment paths. The external cached-image
transport remains a documented fixture. An initial test used spatially and
channel-degenerate DINO fixture values, making the semantic key zero after
LayerNorm. Connectivity now uses explicitly nondegenerate observation inputs,
without editing model parameters or relaxing positive-gradient assertions.
Failed/import-mismatched preliminary test attempts remain in external logs;
only final exact-source runs count as verification.

## Resume boundary

Read the current architecture contract and work plan. M1c/M1d are published;
verify M2/M3 exact final tests and publication from their source artifacts.
Proceed to M4 global objectness, multimodal entity support, actual G3 value
ownership and causal association only after M3's local-law unit is verified.
S-task, W matched-action supervision, shared target P1/P2 and execution feedback
remain open. They are not completed by upstream interface migration.
Do not recreate an unused copy of the G1 posterior: its actual G2/G3/P1
consumers now preserve full support. State/visual source-time and feature tests
are guards for later changes. If publishing fails, preserve the local source,
patch, checksums and verification record instead of repeating payload writes.

User authorization remains ordinary fast-forward pushes to the isolated branch.
No master/old branch modifications, deletion, old-checkpoint migration or
formal training is authorized by this handoff. GitHub permission and actual
published SHA must be verified separately. Local tests are not a remote CI pass.
