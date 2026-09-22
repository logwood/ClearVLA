# ClearVLA structural rebuild handoff

Updated: 2026-09-22. Source implementation status, not user-server state.

## Identity and boundaries

- Isolated branch: `codex/structural-rebuild-20260921`.
- Immutable architectural base: `0f07160d692ec8c8302880420a3d93d74512c39b`.
- Read-only experiment comparator: `a0e1f5d5b9736d37a51faec36039144f181d4b12`.
- Last independently verified remote source: M3
  `c9d4115bde7da46a965601748aa3d33fb06569b3`, tree
  `c3c4671163aec907c724ad6f59e18e9372f65c56`.
- Current source unit: recovered M4a plus implemented M4b; candidate
  `configs/mainline/structural_rebuild_m4b_calvin.json`.

M3 workflow 35677405308 verified 614 collected cases (613 passed, one skipped)
on Python 3.12.14 / PyTorch 2.11.0 CPU and published its exact final tree. The
M4a recovery archive passed all 2174 recorded file hashes. Its M4b draft was
not archived and was NOT treated as recovered; this M4b was rebuilt from the
verified source. Do not assign the M3 SHA to the locally newer candidate.

User authorization covers ordinary fast-forward pushes to this branch. No
master or old-branch update, deletion, old-checkpoint conversion, user-server
job, formal dataset training or benchmark score is part of this source unit.
If publishing fails, preserve complete source, patch, hashes and verification
records and continue source work rather than repeating opaque payload writes.

## Implemented semantics

M1a/M1b/M1c/M1d preserve actual source times, real label support, separate
native/model state charts and visual-motion support. M2 preserves the existing
full G1 spatial posterior through actual G2/G3/P1 consumers. M3 couples typed
local location ownership while keeping attribute values distinct.

M4a sends actual completed third-G-block evidence to the global entity binder;
independent observed DINO remains the reconstruction target. M4b keeps local
assignment in query coordinates and pushes its complete candidate measure to
current camera pixels for object reverse lookup and reconstruction. Geometric
centers come from that same image measure. The actual producer log measure is
retained, not rebuilt from underflowed probabilities. Segmented log-sum-exp and
camera-local normalization avoid tiny-mass division and do not confuse small
camera allocation with absent observation. No confidence floor or hard target
selection is added. Uncovered observed pixels remain in the reconstruction
loss. Conditional writes are piecewise differentiable; this is not a physical
renderer or a guarantee of global objectness.

The new selection is explicit in configuration and deployment identity. The
old-mode numerical path is retained as a control, never as a fallback for
missing current-image provenance. Ordinary exact checkpoints reject mode drift.
No new architecture mode is inferred from a filename or an old experiment tag.

## Verification record and limits

Local runtime is Python 3.13.5 / PyTorch 2.10.0+cpu and is supplemental, not the
project's supported runtime. The rebuilt final M4b has 22 focused cases covering
real production forward/backward, optimizer update, CPU BF16, ordinary
checkpoint/deployment reload and same-noise output, full-support image writes,
independent bilinear adjoint references, tiny log mass, empty support and
camera allocation versus observational validity. All 22 passed locally.
M3 mode has identical parameter contents, order and same-noise sampled action
between published M3 and the current production source at seeds 0 and 71.

The initial full selected regression exposed one old camera-permutation test
that blindly indexed optional dataclass fields. The typed permutation is now
explicit and the test covers BOTH query and current-image modes; all 104 cases
in that contract file pass. The failed preliminary result is retained, not
silently renamed a success. Complete final-suite results must be read from
its exact-source artifact. Scoped Ruff was clean and differential Pyright had
zero new errors/missing imports before the test-adapter repair; run the final
source-owned check rather than extrapolating from that result. Inherited type
errors and library-stub warnings are not a clean-repository static pass.

For accepted project-runtime verification and publication, use the retained
`Verify and publish structural source unit` workflow and its artifact. The
transport commit is not an implemented-source publication. Compare its verified
source tree, published-source archive, final branch SHA and JUnit inventory.
No CUDA/BF16 hardware run, production-batch memory/latency test, formal training
or physical closed-loop validation has occurred.

## Next structural review

Finish M4 global entity consistency and causal association. Distinguish an
uncertain local hypothesis, a learned K slot, a current physical-entity belief
and a cross-observation association; shared indices do not establish identity.
Any entity memory must have matching causal training/burn-in and deployment
reset semantics; never introduce deployment-only recurrence or ODE-updated
physical state. Source-owned support remains separate from inferred visibility.

S task organization, target-aware progress, W matched-action supervision,
shared target P1/P2 reads, P3 execution feedback and the complete bottom/outlet
review remain OPEN. Adapting their upstream interface is not their deep audit.
Current image support fixes one coordinate contract, not physical segmentation,
multiview identity, learned target control or persistent tracking.
