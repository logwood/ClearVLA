# Schema39 design checkpoint

Status: implemented design checkpoint retained for review ancestry. The active
source contract is `../00_CURRENT_ARCHITECTURE_CONTRACT.md`; this note does not
claim experimental success.

## Locked scope

Keep Schema38 G/S/W producers, four Teacher intervals, K/type/camera spatial
ownership, static V120 P1, existing losses and the retained bottom body. Do not
rebuild these subsystems. Schema39 changes only:

1. the P2-to-P3 temporal terminal;
2. dynamic-P1 protected policy precision ownership;
3. the observation/address numerical Jacobian boundary;
4. false diagnostics/tests found during the Schema38 audit.

## A. W/S temporal consumption

P2 legally terminates K and camera through action-conditioned spatial reads but
must retain interval and type:

```text
SelectedIntervalEvidence
  key                [B,T,Q,I,Z,H]
  value              [B,T,Q,I,Z,H]
  selected_s_context [B,T,Q,I,Z,H]
  support            [B,I,Z]

T=24, Q=4, I=4, Z=semantic/geometry
```

The same W-owned K or KxC posterior selects matching typed S metadata. S must
not affect the spatial posterior, value, validity or mask. Appearance retains
its existing terminal as W's semantic conditioner.

P3 becomes the single temporal terminal. It uses a bounded W-S relation, not
Schema38's multiply-then-renormalize formula:

```text
a   = bounded_norm(action_query)
k_i = bounded_norm(W_key_i)
s_i = tanh(bounded_norm(S_public_i + selected_typed_S_i))
logit_i = tau_action * <a,k_i> + tau_intent * <a*s_i,k_i>
```

W=0 therefore gives zero score relation and exact-zero value; S cannot create
future effect. A safe masked softmax spans only the four physical intervals,
with no learned null. All-invalid physical support returns exact zero before
softmax. Teacher reliability, dustbin, covariance and predicted magnitudes may
not become masks.

With `V_i=C+R_i`, the read is `C + sum(alpha_i R_i)`: common passes exactly
once and no null shrinks it. Uniform alpha may still legitimately cancel a
zero-sum residual; there is no entropy/quota or retention target. The repair
removes premature cancellation, not the valid common-only solution.

Time selection must occur before consequence construction:

```text
P2 interval evidence -> P3 temporal compile -> typed effect
-> one shared semantic+geometry .35 contract -> consequence
-> optional P3 refinements
```

Protected consequence is the sole mandatory W value. Optional effect/temporal
lanes may only be bias-free refinements; they must not directly duplicate the
same raw effect. Existing W `hidden.mean(interval)` common conditioning remains
unchanged in this version; reassess only after the consumer is healthy.

## B. Dynamic P1 policy precision

Static factual base remains bit-exact. Dynamic P1 remains a P2 query source and
also creates a separate `ProtectedPolicyPrecision`, never a fact/consequence:

```text
interaction = bias_free_P3(action, static_fact, raw_dynamic)
protected_policy_precision = raw_dynamic + interaction
```

Dynamic zero gives exact-zero carrier. Static zero does not delete legitimate
dynamic policy content; static/action only condition the interaction.

The carrier is removed from the optional precision lane. Bottom reads it once
with the existing shared 4-basis/no-null reader, combines that read with the
six optional lanes, and applies the single shared `.35 * .25` ingress budget.
Other optional lanes remain 4+null and protected consequence remains no-null.
Transition independently consumes one `.35` bounded view exactly once in its
action operand:

```text
action_query + protected_base + protected_policy_precision
```

It must not enter transition context/memory, layer contracts, factual base or
consequence. Do not give protected precision a second independent bottom
budget, write it into transition context, or pre-contract it and then contract
it again at the same consumer.

## C. Observation/address spikes

Use one coordinate-aware bounded-Jacobian standard-deviation map at every live
address variance boundary:

```text
safe_std(v, eps) = sqrt(v + eps^2) - eps
                 = v / (sqrt(v + eps^2) + eps)
```

It is zero-preserving and monotonic, with derivative bounded by `1/(2*eps)`.
It intentionally differs from `sqrt(v)` only in the sub-resolution regime.
`eps` is derived from native chart resolution and converted into each active
coordinate system:

```text
chart_cell_norm = 2 / (grid_side - 1)
eps_norm = chart_cell_norm / (4 * raw_reader_radius)
eps_dino = eps_norm * (dino_side - 1) / 2
```

The existing fine-radius base of `0.5 * cell_stride` remains, so zero variance
does not collapse the search lattice. For G2 centre correction, use
`sqrt(v+eps^2) = eps + safe_std(v,eps)` rather than a zero scale, because the
correction remains meaningful at deterministic address variance.

Replace every active conditional read based on a sub-probability divided by a
tiny mass with an FP32 log-domain posterior and separate observable
availability:

```text
a = log_owner + log_prior
conditional = softmax(a + log(observable_validity))
availability = exp(logsumexp(a + log(validity)) - logsumexp(a))
read = availability * conditional_value
```

All-invalid rows branch to exact-zero posterior, availability and value before
any all-`-inf` reduction. Learned null/K allocation cannot define observable
availability. Apply this consistently to iterative reads, final/typed reads,
camera aggregation and reverse chart reads; leaving a second `sum_mass` divide
would merely move the spike.

The semantic/geometry geometric mean starts from upstream FP32 `log_softmax`:

```text
typed_prior = softmax(0.5 * (log_p_semantic + log_p_geometry))
```

Do not take logs of already-underflowed BF16 probabilities. P2 must consume the
exported log availability/evidence plus an explicit support mask; remove its
live `clamp_min(1e-6).log()` path.

Keep the existing flow uncertainty SmoothL1 objective, global clipping and
spike attribution. Do not add an observation-local clip. Extend spike replay
with the four live variance minima/gains, every support denominator, and final
flow-head channel gradients split as `0:2` flow and `2:6` uncertainty. Per-loss
VJPs run offline from a deterministic replay state, not inside every spike.

Do not alter Teacher/no-grad covariance square roots, pure diagnostic RMS or
unrelated Hellinger calculations; they cannot cause the observed online
training spikes.

## D. Known diagnostic corrections

- Compute complete-field identity in the same FP32 addition order on both
  sides; the current ~1e-3 BF16 discrepancy is not a forward identity failure.
- Replace tests that allow residual retention to be exactly zero with causal
  boundary tests. Do not impose a retention magnitude target.
- Action-only tests use final physical velocity loss with future/auxiliary
  losses disabled; group-level nonzero gradients are insufficient.

## Release process

Only one complete schema is released. Before editing, write a producer-to-
consumer table for the three modified boundaries. After editing perform:

1. producer -> consumer review of axes, zero/null semantics and duplicate paths;
2. consumer -> producer review using ordinary action-only gradients and bounded
   Jacobians;
3. runtime/checkpoint review of static/dynamic calls, Teacher isolation,
   optimizer ownership, fresh/resume identity, memory and throughput.

Mandatory counterexamples:

- distinct zero-sum W interval residuals reach P3 consequence and final action;
- W=0 prevents S from creating future effect (S may still act through its
  separately legal factual path);
- optional null interventions cannot delete protected policy precision or
  protected consequence;
- observation/address Jacobians have a construction-level finite bound;
- neutral fields, empty K/camera, BF16 forward/backward and five-step deploy are
  finite.
