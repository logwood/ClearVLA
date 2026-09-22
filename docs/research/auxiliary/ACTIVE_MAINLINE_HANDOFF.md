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

## Resume boundary

Read the current architecture contract and work plan. Verify M1d/M2 final test and publication status from artifacts, then continue
M3 typed local ownership and M4 global/persistent entity review. M2 is a real
G2/G3/P1 consumer change, not completion of the later task/entity redesign. Do not implement
a second unused posterior container: G1 already stores its full distribution.
The source-time and state-feature tests now become guards for later changes.
Do not restart M1a/M1b or repeat a failing publication loop; preserve a verified
local commit/patch/source archive and continue the next source review.

User authorization remains ordinary fast-forward pushes to the isolated branch.
No master/old branch modifications, deletion, old-checkpoint migration or
formal training is authorized by this handoff. GitHub permission and actual
published SHA must be verified separately. Local tests are not a remote CI pass.
