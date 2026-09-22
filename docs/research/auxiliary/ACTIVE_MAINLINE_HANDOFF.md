# ClearVLA structural rebuild handoff

Updated: 2026-09-22. Source implementation, not user-server job state.

## Immutable reference and current boundary

- Branch: `codex/structural-rebuild-20260921`; ordinary fast-forward only.
- Architectural base: `0f07160d692ec8c8302880420a3d93d74512c39b`.
- Read-only experiment: `a0e1f5d5b9736d37a51faec36039144f181d4b12`.
- Earlier verified remote starting point: M4a/M4b
  `57974b37a2639efa4b5e882b95799ca26f2b9a85`, complete remote tree
  `e713fbe252e1a087128de5f52aadb9a0e8f0ea0a`.
- That published artifact was independently extracted and reconstructed to the
  SAME Git tree. Its CI runtime is Python 3.12.14 / PyTorch 2.11.0 CPU:
  646 collected cases, 645 passed, one skipped, zero failures/errors;
  changed-file Ruff clean and no new Pyright errors.
- This continuation starts from that published archive, NOT an assumed previous
  unarchived workspace. Local recovery Git ancestry is synthetic; exact remote
  SHA and source tree are tracked separately, not misrepresented as one commit.

## Current source unit

Initial M4c1 is now published as `6f74b432a66aacc7f3ff9cd05eaa7591839c32bf`,
full tree `88742456555550c50de1de8f3fed39602d804fcb`, publisher run
`35688670562`. Its artifact must not be mistaken for the corrected status chart.
The current source is the M4d candidate:
`configs/mainline/structural_rebuild_m4d_calvin.json`. It includes both the
post-freeze confidence-chart correction and current-entity motion reads.


M4c1 adds flow-pulled causal observed temporal evidence to the real entity
binder. Candidate config: `configs/mainline/structural_rebuild_m4c_calvin.json`.
Read the architecture contract's M4c1 section and the producer/consumer work map.

The existing inverse flows compose older observation locations in the correct
source frame. Current full candidate support is retained. No hidden recurrent
state is added. Source availability, inferred correspondence coverage and
predicted confidence/occlusion remain different concepts. Current observed
content and loss masks are not replaced by historical guesses. Temporal keys
are consumed in K competition and its iterative update, and all S/W/P paths
receive the resulting ordinary entity facts. Dense history does not enter the
compact W/ODE belief. This is finite-window grouping, not persistent object IDs.

## Post-freeze source-chart correction

The frozen M4c1 tree `88742456555550c50de1de8f3fed39602d804fcb` passed
675 local cases (one skip), but a further producer-to-consumer review found a
confidence/occlusion chart mismatch. Its publisher is separately tracked; do
not infer the corrected property from that artifact. `_RawPyramidFlow` emits
forward confidence on earlier-source cells; only inverse flow is later-source
indexed. The correction reads status after taking the inverse step, adds an
explicit status chart and ABI fields, and uses nonuniform spatial fixtures and
an independent flow-gradient oracle. The corrected focused inventory has
33 passing cases, including production optimizer/CPU-BF16/checkpoint paths.
The correction-only local full inventory completed: 679 collected, 678 passed,
one skipped, zero failures/errors. Scoped Ruff is clean, Pyright has no added
errors/imports, and two-seed old-mode parity passed. Its complete remote result
must still be read from the subsequent corrected source artifact.
A complete frozen-source backup records this known defect, so it cannot be
mistaken for the final corrected acceptance. No existing source mask is changed.

## Verification and preservation

Local supplemental runtime: Python 3.13.5 / PyTorch 2.10.0+cpu. This is NOT the
project's supported runtime. Matching remote CI must be checked independently.
The initial focused inventory passed 23 cases. A camera-count mismatch in a
standalone fixture was fixed (the real local fixture has one camera, not two);
the source shape assertion was NOT relaxed. An initial process timeout before
JUnit completion is NOT counted as a pass. Later tests add action-only gradient
checks through the real decoder, exact flow-unit/sign comparison with the
existing adapter and isolated-past intervention of actual K assignment. The
final focused file has 30 passing cases. Extra provenance tests reject stale
current-content, current-mask and flow-gap records. Old-mode controls at seeds
0 and 71 have bit-identical state_dict contents, named parameter order/flags,
configuration and same-noise sampled actions against the published M4b source.
A first external hash script mishandled scalar buffers; its logs are retained
and its corrected full two-tree comparisons passed. No production assertion
was weakened.

The final full regression was restarted after current/history provenance guards
were tightened. The interrupted previous inventory is explicitly INVALIDATED,
not counted as a successful run. This candidate adds 47,552 trainable parameters
at its resolved production dimensions. Shape accounting is not a measured GPU
memory/latency profile. There are no extra W builds or ODE evaluations.

Scoped Ruff is clean and differential Pyright has zero new errors/imports as
of the final source review. Inherited type diagnostics remain visible. Full
regression results, final source hashes and supported-runtime CI status belong
to the verification artifacts, not to a guessed completion statement here.
External HDF5/image/token transport is replaced only in declared fixtures;
real neural modules, optimizer, checkpoint/deployment and ODE paths execute.
There is no CUDA run, production-batch benchmark, formal dataset training or
physical closed-loop result.

## Next work (still open)

Finish remaining M4c2 global physical-entity and cross-window association design.
Do not lock K indices as physical IDs or add deployment-only recurrence. Any
persistent state needs explicit matched training unroll/burn-in and reset rules.
Then continue M5 S target/operation/progress, M6 action-matched W supervision,
M7 shared-target P1/P2, M8 executed-prefix feedback, M9 bottom/outlets, M10 full
training/deployment agreement, M11 final audit. None is completed by this unit.

Do not change master, historical branches, checkpoints, user-server jobs or
formal training. If Git publication fails, save complete source, patch, hashes
and logs locally and proceed with source work. Never repeatedly regenerate an
opaque upload or confuse its transport commit with published verified source.

## Current M4d acceptance boundary

Production G3 exports motion from its current image measure and latest inverse
flow, retaining K/camera/source-time axes. Real S/W input hooks and an old-anchor
intervention verify consumption, not only field existence. No extra parameters
are added. Preliminary focused motion cases passed; acceptance is the final
exact-source inventory and independent supported-runtime artifact. Initial
import-fixture failures and superseded runs remain in external logs, not renamed
passes. Dense correspondence/history stays out of numerical solver caches.
Source masks and the independent observed reconstruction target are unchanged.
After acceptance, review S's target/operation/progress and downstream view-aware
geometry. Cross-window physical identity is still open; K numbers are not IDs.
