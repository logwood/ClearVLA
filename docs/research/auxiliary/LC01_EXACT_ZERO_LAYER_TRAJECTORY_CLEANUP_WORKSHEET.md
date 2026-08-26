# LC-01 exact-zero layer-contract trajectory cleanup worksheet

Status: `IMPLEMENTED AND STATICALLY CLOSED`

LC-01 is a deletion-only cleanup between completed R1g and R1h. It does not
add a replay capability and does not reinterpret P1, P2 or the retained V120
bottom. It removes two source-proven exact-zero trajectory aliases from the
terminal policy layer contracts while retaining every live rollout, state,
event and downstream decoder path.

No training, dataset, CUDA or checkpoint command is authorized in this unit.

## 1. Pre-edit source boundary

The audit is against R1g commit `d4d4da0`.

| File | Git blob |
|---|---|
| `clearvla/mainline/model/policy.py` | `f0e5ae1b227d1b8a829ebcb38c27ab5b16179235` |
| `clearvla/mainline/model/restored_bottom.py` | `bd0cf3dfe9f6ae1b4df914ec5efa0eec5e3ea58a` |
| `clearvla/mainline/v120_core/layer_contracts.py` | `0d7becde5fae649cb427959a47084ffed94f0350` |
| `clearvla/mainline/v120_core/time_domain_mmdit.py` | `079e2351b5854532f98a1a399554fc6f8aa6403f` |
| `clearvla/mainline/training/optimizer.py` | `21f5317451e7b3861952b8437304d3e38681e18a` |
| `clearvla/mainline/training/losses.py` | `af03984ba0bbcf68e09ec4af8117f13660f12580` |
| `clearvla/mainline/runtime/checkpoints.py` | `0598a096133ea0775de64e8b3a29b7ac4c8bc7fa` |
| `clearvla/mainline/manifest.py` | `9d13c572ad331ba16c770f70740da97a2a912be3` |
| `tests/test_mainline_policy.py` | `e8e6cfa8a19d0e0a4e9eb6b25e9f6cf2e78e8b09` |
| `tests/test_mainline_manifest.py` | `19a463a94fa74bed7ba0f25853c213809a5ee63a` |

## 2. Complete active dataflow map

### 2.1 Producers and transformations

`RestoredV120EvidenceBottom._layer_contracts()` constructs two trajectory
tensors at every dynamic call:

```text
level 1 = action_query + factual_base
level 2 = action_query + protected_consequence
```

Each `[B,24,4,H]` value is flattened to 96 rows and concatenated with state,
state history, executed history and the 512-row completed transition selector.
The two independent depth-5/depth-6 `LayerContractAdapterHeads` apply:

```text
adapted_row = row + residual_scale * MLP(LayerNorm(row))
```

There is no attention, convolution, pooling across rows, position-dependent
normalization or other token mixing in this adapter. Consequently a trajectory
row cannot change any rollout/state row.

`MidcutContractHeads` then returns a large V120-compatible dictionary. The
trajectory-only computation is:

```text
trajectory rows -> basis mean -> frozen action head, called twice
                                -> frozen motion head
```

The trajectory also supplies shapes for two fixed-zero compatibility tensors.
These values are not selected by any current consumer.

### 2.2 Exact consumer classification

`EvidenceViewAdapter._LAYER_FIELDS` reads exactly:

```text
rollout_tokens
state_tokens
state_history_tokens
```

Its separate event input is the last contract's `event_logits`. Those logits
are produced from the adapted rollout rows through `rollout_delta_head` and
`event_head`; they do not depend on a trajectory row. The final bottom action,
event and motion outputs come from the retained Evidence-MMDiT decoder heads,
not from the layer-contract action or motion probes.

| Contract value | Active status | Reason |
|---|---|---|
| `rollout_tokens` | LIVE | enters ordered layer evidence |
| `state_tokens` | LIVE | enters ordered layer evidence |
| `state_history_tokens` | LIVE | enters ordered layer evidence |
| `event_logits` | LIVE | enters the explicit event evidence bank |
| `rollout_delta_pred` internal value | LIVE internally | produces event context; returned alias is unread |
| `trajectory_tokens` / `trajectory_pooled` | DEAD | no current consumer |
| `pred_physical_velocity` / `direct_physical_velocity` | DEAD | final velocity comes from decoder heads |
| layer-contract `motion_logits` | DEAD | final motion comes from decoder heads |
| `rollout_residual_velocity` / `rollout_alpha` | DEAD | fixed compatibility outputs with no consumer |

Other unread compatibility fields are outside LC-01 unless removing the
trajectory makes their construction impossible. In particular LC-01 does not
delete rollout/state/event modules merely because some returned aliases of
their internal values are unread.

### 2.3 Loss and backward map

There is no mainline layer-contract auxiliary loss. Physical flow, decoded
action, event, motion and execution losses reach the two adapters through the
live evidence/event paths above. The optimizer role
`v120_layer_contracts/decay` owns exactly the two adapters' 12 trainable tensors
and remains unchanged.

Every parameter under `head.readout` is frozen. The trajectory-only
`action_head` and `motion_head` therefore own no optimizer state and receive no
loss gradient. Removing them cannot delete a trainable owner.

The pre-edit executable probe measured:

```text
gradient nonzeros to level inputs
  action_query                0
  factual_base                0
  protected_consequence       0

adapter gradient nonzeros by tensor
  32,32,4096,128,4096,32 per head (all 12 tensors nonzero)

1000x independent trajectory perturbation
  rollout/state/history/event live outputs: bit-identical
```

Thus only the trajectory branch has an exact-zero VJP. The terminal depth
adapters remain functional and must not be removed or frozen.

### 2.4 Runtime and compute

The layer contracts run at the five action-update calls and once at the endpoint
head call. Production shapes contain 96 trajectory rows in a 619-row contract
canvas, so the dead rows consume 15.51% of this adapter's token work. Counting
only linear multiply-accumulates, the two dead rows/readouts cost approximately
26,075,136 MACs per bottom call.

The two frozen trajectory-only readout families contain exactly 23,590
parameters and 16 state keys:

```text
two action heads   20,516 parameters / 8 tensors
two motion heads    3,074 parameters / 8 tensors
```

The formulas themselves own no parameters.

## 3. Authorized least-change implementation

LC-01 may make only these semantic changes:

1. `_layer_contracts()` takes only the completed rollout and shared seed. Both
   depth heads see the same live canvas and retain their independent adapters.
2. The contract canvas contains no trajectory rows. It may retain an empty
   trajectory slice only as non-computing structural metadata.
3. `MidcutContractHeads` stops materializing trajectory/action/motion and their
   trajectory-shaped fixed-zero outputs. Its rollout/state/event computations
   remain algebraically identical.
4. Bottom no longer accepts the duplicate `p1_fact` argument. The main policy
   caller and structural audit helper remove that argument.
5. The exact changed diagnostic is renamed from
   `bottom_event_from_p2_layer_contract` to
   `bottom_event_from_terminal_layer_contract`. The generic zero-trajectory
   decoder ingress is a different path and remains unchanged.
6. The bottom component manifest names terminal layer contracts rather than
   P1/P2 trajectory contracts, so exact resume and bottom-only migration reject
   the removed state ABI.

The removed action/motion modules must still consume their historical default
and explicit initialization draws as temporary unregistered objects in the
constructor. This preserves every retained layer-contract tensor, every
downstream decoder tensor and the seed-0 post-construction RNG stream without
retaining dead runtime/state capacity.

Pre-edit seed-0 sentinels are:

```text
retained layer-contract keys     46
retained layer-contract digest   801ce2c38e4b552b97500c20bee291cf3c548096ecab5d90356d066c1406a7fc
bottom decoder keys              268
bottom decoder digest            1d85ddad8d3e5c04413f94bb01b9e09532472d16986c02189ac5ff92416be586
post-construction CPU RNG        d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21
```

The digest algorithm streams name, shape, dtype and tensor bytes in state-dict
order. Post-edit shared-key and decoder digests must match exactly.

## 4. Over-hardening decisions

Resolved:

- no replacement trajectory, zero token, projection, gain, gate, loss or
  diagnostic proxy is added;
- the two live depth adapters are retained rather than collapsed or shared;
- rollout/state/event computation is not simplified in this unit;
- generic decoder trajectory/workspace ingress remains its separately audited
  exact-zero source and is not conflated with the two layer-contract formulas;
- unread non-trajectory compatibility outputs are recorded but not swept into
  this deletion;
- initialization compatibility is preserved without registering discarded
  parameters or adding runtime branches;
- exact resume is rejected instead of providing a partial-load shim.

No unresolved assumption remains that can change this boundary. Discovery of
any trajectory read outside the classified consumers, token mixing inside the
adapter, direct layer-contract auxiliary loss, or optimizer ownership of a
removed readout invalidates the worksheet and stops editing.

## 5. Test-first acceptance matrix

Tests must be observed red before production editing and establish:

- bottom forward and its audit helper expose no `p1_fact`/factual trajectory
  argument;
- `_layer_contracts()` exposes no action, fact or consequence operand;
- the canvas carries zero trajectory rows while preserving rollout/state rows;
- terminal contracts publish no trajectory/action/motion compatibility keys;
- inserting arbitrary trajectory rows into a reference canvas cannot change
  any retained rollout/state/event result;
- all 12 adapter tensors retain finite nonzero reverse paths;
- the event evidence is still exactly the last terminal contract's event
  logits;
- downstream velocity/event/motion losses and all retained bottom consumers
  remain reachable;
- optimizer groups and trainable/optimizer tensor counts do not change;
- total parameters fall by exactly 23,590, parameter/state keys by 16, and
  trainable parameters by zero;
- retained layer-contract tensors, decoder tensors and seed-0 RNG match the
  pre-edit sentinels exactly;
- the manifest rejects the old bottom ABI;
- retained tests, compileall, Ruff, Pyright changed-line gate and
  `git diff --check` pass;
- no training, dataset, CUDA or checkpoint command runs.

Authorized edits are limited to `model/policy.py`, `model/restored_bottom.py`,
`v120_core/layer_contracts.py`, the component manifest, direct tests and compact
architecture/replay documents. `time_domain_mmdit.py`, optimizer, losses,
checkpoint loader and runtime lifecycle are audited consumers and are not
authorized for source changes in LC-01.

## 6. Implementation closure

The test-first red state showed that bottom still accepted the duplicate
factual operand and the manifest still named P1/P2 trajectory contracts. The
production edit then removed only the boundary authorized in Section 3.

Forward re-review confirms that both independent depth-5/depth-6 adapters read
the same completed rollout/state canvas, and that `EvidenceViewAdapter` still
receives their rollout, state and state-history rows plus the last contract's
rollout-derived event logits. The final decoder action/event/motion heads and
the separate generic neutral trajectory ingress are unchanged. Injecting 96
arbitrary trajectory rows into either adapter canvas leaves every retained
contract result bit-identical.

Reverse re-review confirms finite nonzero gradients for all 12 trainable
adapter tensors through the retained rollout/state/history/event consumers.
There is no trajectory operand, trajectory VJP, replacement carrier, new
scale, optimizer owner, loss or diagnostic proxy.

Inventory closure:

| Field | R1g | LC-01 | Delta |
|---|---:|---:|---:|
| Total parameters | 168,436,164 | 168,412,574 | -23,590 frozen |
| Trainable parameters | 152,041,843 | 152,041,843 | 0 |
| Parameter tensors | 1,402 | 1,386 | -16 frozen |
| Trainable/optimizer tensors | 1,064 | 1,064 | 0 |
| Optimizer groups | 23 | 23 | 0 |
| Layer-contract optimizer tensors | 12 | 12 | 0 |
| State-key names | 1,408 | 1,392 | -16 |

Post-edit seed-0 sentinels:

```text
retained layer-contract keys     46
retained layer-contract digest   801ce2c38e4b552b97500c20bee291cf3c548096ecab5d90356d066c1406a7fc
bottom decoder keys              268
bottom decoder digest            1d85ddad8d3e5c04413f94bb01b9e09532472d16986c02189ac5ff92416be586
post-construction CPU RNG        d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21
ordered state-key-name digest    be7b4b58a8e2ec25c1e3b5c455f303a0954d20a984201173b5de12d2b1f14a20
manifest digest                  1691f3fc2c6f5be916637ea04388d69bbb023ba4dc7bdd085b45c85f70d45981
```

Verification:

| Check | Result |
|---|---|
| Focused manifest and layer-contract tests | PASS after recorded red states |
| Complete retained ten-file mainline suite | PASS: 145/145 |
| Retained weight/decoder/RNG equality | PASS: exact digest matches |
| Optimizer ownership and inventory | PASS: trainable partition unchanged |
| Python compileall over `clearvla` and `tests` | PASS |
| Ruff over touched source/test files | PASS |
| Pyright over all touched source/test files | PASS changed-line gate: 0 changed-line errors; 14 existing errors and 114 warnings outside changed lines |
| `git diff --check` | PASS; repository line-ending notices only |
| Training, dataset, CUDA and checkpoint commands | NOT RUN |

Final pre-commit source fingerprints:

| File | Git blob | SHA-256 |
|---|---|---|
| `clearvla/mainline/manifest.py` | `2fb2c6118aea1a13e3153514f31dcc98ddd24275` | `99373D85C8EC0EDC9EE8F38EC6FE4C25AA2674826C4A562738431ABBA1F4EE12` |
| `clearvla/mainline/model/policy.py` | `8aa389b1d6689b7cd643310a7ae11b5ee47bf6a2` | `7BE1F08C562761E354CE427A36C2D45FF7D4A175935AEB2127690A28D8612CEB` |
| `clearvla/mainline/model/restored_bottom.py` | `bc6abb849374e01ce3ed9376e4675e5e7ba61232` | `FA2FF2257B6DA43A5316DDF757550E89D10FDBD7E3AA9298B47E73047DDF9C29` |
| `clearvla/mainline/v120_core/layer_contracts.py` | `af615a753a19047054153ea8e331edf9eac59f14` | `3765DBBC1C0A13D94DE3BBEBB1C6D89580EF0054845495D9294EC4AC921EA539` |
| `tests/test_mainline_manifest.py` | `68118bd714bc3c639b75d799611ae2c5aa60e2f1` | `EF962DF17AA5397F286E2DB77B288BB15A83915A638DD65FDE1D418D87E55576` |
| `tests/test_mainline_policy.py` | `06dba299632868205a00311d207513ab4daa9d6b` | `70CF2135F1DCFA79C3669572D26D8A97DBE569F280FB6CB61993EBD1521614F6` |

No unresolved assumption remains in LC-01, and this cleanup does not authorize
an experiment.
