"""Recovered V120 stateless intent, plan recognition and coarse action."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .routing import register_gradient_rms_metric, smooth_rms_contract
from .types import (
    INTERVAL_BOUNDS,
    ActionIntentDock,
    CoarseActionIntentState,
    FutureObjectDynamics,
    FuturePlanRecognition,
    ObjectFactSet,
    ObjectIntentState,
    normalized_entropy,
)

TYPED_INTENT_NAMES = ("semantic", "appearance", "geometry")


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
    )


def _interval_slices(length: int) -> tuple[slice, ...]:
    if length < 1:
        raise ValueError("future sequence cannot be empty")
    rows: list[slice] = []
    for lower, upper in INTERVAL_BOUNDS:
        start = min(max(int(lower) - 1, 0), length - 1)
        stop = min(max(int(upper), start + 1), length)
        rows.append(slice(start, stop))
    return tuple(rows)


def _interval_common_residual(value: Tensor) -> tuple[Tensor, Tensor]:
    """Express four interval rows as one common value plus exact residuals."""

    if value.ndim < 2 or int(value.shape[1]) != 4:
        raise ValueError("S decomposition requires four interval rows")
    common = value.mean(dim=1)
    residual = value - common[:, None]
    return common, residual


def _safe_masked_softmax(logit: Tensor, support: Tensor, *, dim: int = -1) -> Tensor:
    """Finite masked softmax with an exact-zero all-invalid fallback.

    TargetFact owns a physical support mask, not a learned confidence.  In
    particular, a row with no legal object must not turn into a uniform target
    merely because the ordinary softmax kernel has no finite entry.
    """

    if support.dtype != torch.bool:
        support = support.to(dtype=torch.bool)
    if tuple(logit.shape) != tuple(support.shape):
        raise ValueError("target masked softmax support must align with logits")
    legal = support.any(dim=dim, keepdim=True)
    masked = torch.where(support, logit.float(), torch.full_like(logit.float(), -torch.inf))
    # The all-invalid row receives finite zero logits only for the kernel call;
    # its output is discarded immediately below.
    safe = torch.where(legal, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=dim)
    return torch.where(legal, probability, torch.zeros_like(probability))


class _CrossRead(nn.Module):
    def __init__(self, hidden: int, heads: int, maximum_rms: float = 0.35) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )
        self.maximum_rms = float(maximum_rms)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        *,
        padding_mask: Tensor | None = None,
        diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # ``MultiheadAttention`` returns NaNs when every key in one batch row
        # is masked.  More importantly, normalizing a padded/invalid key before
        # masking still lets a malformed cache leak through the reduction.  A
        # dummy zero key keeps the kernel finite for the all-invalid row; its
        # update is discarded below so that the row is an exact identity read.
        all_invalid: Tensor | None = None
        effective_padding_mask = padding_mask
        if padding_mask is not None:
            if padding_mask.ndim != 2 or tuple(padding_mask.shape) != tuple(
                memory.shape[:2]
            ):
                raise ValueError("cross-read padding mask must align with memory")
            if int(memory.shape[1]) < 1:
                raise ValueError("cross-read memory cannot be empty")
            effective_padding_mask = padding_mask.to(
                device=memory.device,
                dtype=torch.bool,
            )
            valid = ~effective_padding_mask
            all_invalid = ~valid.any(dim=-1)
            safe_memory = torch.where(
                valid[..., None],
                memory,
                torch.zeros_like(memory),
            )
            normalized_memory = self.memory_norm(safe_memory)
            # Unmask one zero key only for rows with no legal key.  The
            # resulting attention weight is removed after the kernel call.
            effective_padding_mask = effective_padding_mask.clone()
            effective_padding_mask[all_invalid, 0] = False
        else:
            normalized_memory = self.memory_norm(memory)
        update, weights = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=effective_padding_mask,
            need_weights=diagnostics,
            average_attn_weights=True,
        )
        update, _ = smooth_rms_contract(update, self.maximum_rms)
        value = query + update
        ffn, _ = smooth_rms_contract(self.ffn(value), self.maximum_rms)
        innovation = update + ffn
        value = value + ffn
        if all_invalid is not None:
            invalid = all_invalid[:, None, None]
            # No valid object/evidence is an exact no-op, including its FFN
            # residual.  This is intentionally a value-level mask rather than
            # a detached loss-side convention.
            value = torch.where(invalid, query, value)
            innovation = torch.where(invalid, torch.zeros_like(innovation), innovation)
            if weights is not None:
                weights = torch.where(
                    all_invalid[:, None, None],
                    torch.zeros_like(weights),
                    weights,
                )
        if weights is None:
            weights = query.new_zeros(query.shape[0], query.shape[1], memory.shape[1])
        return value, innovation, weights


class _SelfBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    def forward(self, value: Tensor, *, causal: bool = False) -> Tensor:
        normalized = self.norm(value)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=_causal_mask(int(value.shape[1]), value.device) if causal else None,
            need_weights=False,
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = value + update
        ffn, _ = smooth_rms_contract(self.ffn(value), 0.35)
        return value + ffn


class _ZeroStartTargetAddress(nn.Module):
    """Bias-free shared type-to-K address head without constructor RNG use."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if int(width) <= 0:
            raise ValueError("target-address width must be positive")
        self.weight = nn.Parameter(torch.zeros(1, int(width), dtype=torch.float32))

    def forward(self, value: Tensor) -> Tensor:
        if int(value.shape[-1]) != int(self.weight.shape[-1]):
            raise ValueError("target-address input lost typed evidence width")
        # Elementwise FP32 arithmetic is deliberately used instead of an
        # autocast-eligible GEMM.  Equal typed evidence must remain an exact
        # K-common value before the centering contract below.
        return (value.float() * self.weight.float()).sum(dim=-1, keepdim=True)


class StatelessObjectIntentOrganizer(nn.Module):
    """V120 observable intent without scalar progress or synthetic phases."""

    def __init__(
        self,
        *,
        hidden: int,
        goal_dim: int,
        state_dim: int,
        action_dim: int,
        content_dim: int,
        route_dim: int,
        horizon: int,
        heads: int,
        camera_names: tuple[str, ...] = ("top", "wrist"),
        interval_object_query_mode: str = "goal_history",
        target_object_address_mode: str = "post_pool_only",
    ) -> None:
        super().__init__()
        if interval_object_query_mode not in {"goal_history", "history_only"}:
            raise ValueError("unknown interval object query mode")
        if target_object_address_mode not in {
            "post_pool_only",
            "shared_target_prior_v1",
            "target_action_bottleneck_v1",
        }:
            raise ValueError("unknown target object address mode")
        self.interval_object_query_mode = interval_object_query_mode
        self.target_object_address_mode = target_object_address_mode
        self.target_fact_mode = target_object_address_mode == (
            "target_action_bottleneck_v1"
        )
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.camera_names = tuple(str(name) for name in camera_names)
        if any(not name or name != name.strip() for name in self.camera_names):
            raise ValueError("intent camera names must be canonical and non-empty")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("intent camera names must be unique")
        # TargetFact exports a per-camera physical-readability mass whose C
        # axis is canonicalized below.  ObjectFactSet/W/P2 retain the
        # serialized data-camera axis, so accepting a non-canonical order here
        # would silently pair (for example) ``wrist`` mass with the ``top``
        # dynamics chart.  The migration/checkpoint boundary already rejects
        # such trained artifacts; reject fresh construction as well until all
        # shared C-axis owners carry an explicit role permutation.
        if target_object_address_mode == "target_action_bottleneck_v1" and (
            self.camera_names != tuple(sorted(self.camera_names))
        ):
            raise ValueError(
                "TargetFact requires canonical sorted camera_names; "
                "shared ObjectFactSet/W/P2 camera axes are still in declared order"
            )
        self.canonical_camera_names = tuple(sorted(self.camera_names))
        self.camera_canonical_permutation = tuple(
            self.camera_names.index(name) for name in self.canonical_camera_names
        )
        self.cameras = len(self.camera_names)
        if self.cameras <= 0:
            raise ValueError("intent requires at least one physical camera")
        self.goal_input = nn.Linear(goal_dim, hidden, bias=False)
        self.goal_queries = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.goal_read = _CrossRead(hidden, heads)
        self.goal_self = _SelfBlock(hidden, heads)
        history_width = state_dim + action_dim + state_dim + 1
        self.history_input = nn.Sequential(
            nn.LayerNorm(history_width, elementwise_affine=False),
            nn.Linear(history_width, hidden, bias=False),
        )
        self.history_blocks = nn.ModuleList(_SelfBlock(hidden, heads) for _ in range(2))
        self.object_content = nn.Linear(content_dim, hidden, bias=False)
        self.object_semantic = nn.Linear(route_dim, hidden, bias=False)
        self.object_appearance = nn.Linear(route_dim, hidden, bias=False)
        self.object_geometry = nn.Linear(route_dim, hidden, bias=False)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.interval_goal = _CrossRead(hidden, heads)
        self.interval_history = _CrossRead(hidden, heads)
        # The legacy interval K reader remains available only for the two
        # compatibility modes.  TargetFact-v1 deliberately does not register
        # or call it: otherwise it would remain a second language-conditioned
        # target selector upstream of P1/coarse/P2.
        self.interval_object: _CrossRead | None
        if self.target_fact_mode:
            self.interval_object = None
        else:
            self.interval_object = _CrossRead(hidden, heads)
        self.scene_read: _CrossRead | None
        self.scene_pool: nn.Linear | None
        self.target_query: nn.Linear | None
        self.target_key: nn.Linear | None
        self.target_score: nn.Linear | None
        self.target_physical_value: nn.Linear | None
        if self.target_fact_mode:
            # Full-scene object values must not be rejoined with the complete
            # instruction after target selection.  Downstream scene physics is
            # supplied by goal-free W/P/CT under the supervised physical
            # proposal, so S exports no independent high-capacity scene token.
            self.scene_read = None
            self.scene_pool = None
            self.target_query = nn.Linear(hidden, hidden, bias=False)
            self.target_key = nn.Linear(hidden, hidden, bias=False)
            self.target_score = nn.Linear(hidden, 1, bias=False)
            nn.init.zeros_(self.target_score.weight)
            # Per camera: coordinate first moment (2), coordinate second
            # moment (xx, xy, yy), motion-prior first moment (2), and readable
            # mass (1).  Three global scalars retain object readable mass,
            # existence mass and unresolved mass.  These are declared
            # physical statistics; raw content/appearance/RGB is not exported
            # to the full-goal action proposer.
            self.target_physical_value = nn.Linear(
                8 * self.cameras + 3,
                hidden,
                bias=False,
            )
        else:
            self.scene_read = None
            self.scene_pool = None
            self.target_query = None
            self.target_key = None
            self.target_score = None
            self.target_physical_value = None
        self.typed_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.typed_relevance_queries: nn.ModuleList | None
        self.typed_relevance_queries = (
            None
            if self.target_fact_mode
            else nn.ModuleList(
                nn.Linear(hidden, route_dim, bias=False)
                for _ in TYPED_INTENT_NAMES
            )
        )
        # One shared object identity must bind semantic and geometry reads.
        # The head learns which typed evidence identifies the operated object
        # (including appearance for color instructions) without creating one
        # target per downstream reader.  Exact zero initialization preserves
        # the legacy spatial reader at initialization and lets ordinary action
        # loss open the route without a hand-set gain.
        self.target_object_address: _ZeroStartTargetAddress | None
        if target_object_address_mode == "shared_target_prior_v1":
            self.target_object_address = _ZeroStartTargetAddress(
                len(TYPED_INTENT_NAMES)
            )
        else:
            # The accepted reader must retain the exact old parameter set and
            # constructor RNG stream.  Runtime still exposes a typed zero field
            # so the downstream dock has one stable schema.
            self.target_object_address = None
        # Initial temperature is exactly one.  It remains bounded in [0.25, 4]
        # and therefore cannot turn the fixed-zero null comparison into an
        # unbounded selector gain.
        if self.target_fact_mode:
            self.register_parameter("typed_temperature_logit", None)
        else:
            self.typed_temperature_logit = nn.Parameter(
                torch.full((len(TYPED_INTENT_NAMES),), -1.3862943611198906)
            )
        self.interval_self = _SelfBlock(hidden, heads)
        self.temporal_identity = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        self.temporal_read = _CrossRead(hidden, heads)
        self.state_change_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.state_change_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.state_change_key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.state_change_read = nn.MultiheadAttention(
            hidden, heads, bias=False, dropout=0.0, batch_first=True
        )
        self.state_change_input = nn.Linear(state_dim, hidden, bias=False)
        self.state_change_transport = nn.Linear(2, hidden, bias=False)
        self.state_change_fuse = nn.Linear(2 * hidden, hidden, bias=False)

    @staticmethod
    def _paired_history(
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if state_history.ndim != 3 or state.ndim != 2 or executed_history.ndim != 3:
            raise ValueError("intent history requires state/action sequences")
        state_sequence = torch.cat((state_history, state[:, None]), dim=1)
        length = max(int(state_sequence.shape[1]), int(executed_history.shape[1]))

        def left_pad(value: Tensor, target: int) -> Tensor:
            missing = target - int(value.shape[1])
            if missing <= 0:
                return value[:, -target:]
            return torch.cat((value[:, :1].expand(-1, missing, -1), value), dim=1)

        states = left_pad(state_sequence, length)
        actions = left_pad(executed_history, length)
        previous = torch.cat((states[:, :1], states[:, :-1]), dim=1)
        delta = states - previous
        offset = torch.linspace(
            -1.0, 0.0, length, device=states.device, dtype=states.dtype
        )[None, :, None].expand(states.shape[0], -1, -1)
        return torch.cat((states, actions, delta, offset), dim=-1), delta

    def _object_tokens(self, facts: ObjectFactSet) -> Tensor:
        facts.validate()
        validity = torch.nan_to_num(
            facts.validity.to(device=facts.content.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        support = validity[..., 0] > 0.0
        content = torch.where(
            support[..., None],
            facts.content,
            torch.zeros_like(facts.content),
        )
        return self.object_content(content)

    @staticmethod
    def _bounded_unit(value: Tensor, *, floor: float = 0.25) -> Tensor:
        value_f = value.float()
        return value_f / (
            value_f.square().sum(dim=-1, keepdim=True) + float(floor) ** 2
        ).sqrt()

    def _target_fact(
        self,
        *,
        protected_goal: Tensor,
        history: Tensor,
        objects: Tensor,
        typed_route: Tensor,
        validity_values: Tensor,
        camera_coordinates: Tensor,
        camera_transport_prior: Tensor,
        camera_validity: Tensor,
        existence: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Resolve one observation-level language-to-object fact.

        The final target-score row is the only zero-start selector part.
        Query/key projections are ordinary non-zero projections; because the
        final score owner starts at zero, their target-binding VJP opens after
        the first optimizer step.  Validity is split
        into a boolean support mask and a fractional value weight; it never
        becomes a second learned selector.
        """

        if not self.target_fact_mode:
            raise RuntimeError("TargetFact requested in a compatibility mode")
        assert (
            self.target_query is not None
            and self.target_key is not None
            and self.target_score is not None
            and self.target_physical_value is not None
        )
        object_support = validity_values[..., 0] > 0.0
        query_source = protected_goal.mean(dim=1) + history[:, -1]
        target_query = self.target_query(query_source)
        target_key = self.target_key(objects)
        pair = target_query[:, None, :] * target_key
        target_logits = self.target_score(pair).squeeze(-1).float()
        target_posterior = _safe_masked_softmax(target_logits, object_support)

        has_target = object_support.any(dim=-1, keepdim=True)
        valid_count = object_support.float().sum(dim=-1, keepdim=True)
        uniform = torch.where(
            has_target,
            object_support.float() / valid_count.clamp_min(1.0),
            torch.zeros_like(target_posterior),
        )
        # Fractional validity is physical evidence mass, not a probability
        # normalizer.  Keeping it outside a denominator means a barely readable
        # object stays barely readable instead of being promoted to a complete
        # target.  The action proposer receives only declared physical moments
        # under this weight; it never receives the mixed high-capacity object
        # content that could let the full instruction reconstruct a second
        # object selector after a uniform p.
        # Keep identity and physical readability as two typed quantities.
        # ``target_posterior`` is normalized over legal identity support;
        # ``target_physical_mass`` is deliberately *not* renormalized and is
        # the only mass allowed to scale action-facing physical evidence.
        target_physical_mass = target_posterior * validity_values[..., 0].float()
        camera_index = torch.as_tensor(
            self.camera_canonical_permutation,
            device=camera_coordinates.device,
            dtype=torch.long,
        )
        camera_coordinates = camera_coordinates.index_select(2, camera_index)
        camera_transport_prior = camera_transport_prior.index_select(
            2, camera_index
        )
        camera_validity = camera_validity.index_select(2, camera_index)
        if bool((~torch.isfinite(camera_validity)).any().item()):
            raise ValueError("TargetPhysicalFact camera validity must be finite")
        camera_validity_f = torch.nan_to_num(
            camera_validity.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        camera_support = camera_validity_f[..., 0] > 0.0
        supported_coordinate_nonfinite = camera_support & ~torch.isfinite(
            camera_coordinates
        ).all(dim=-1)
        supported_transport_nonfinite = camera_support & ~torch.isfinite(
            camera_transport_prior
        ).all(dim=-1)
        if bool(supported_coordinate_nonfinite.any().item()):
            raise ValueError(
                "TargetPhysicalFact camera coordinates are non-finite on support"
            )
        if bool(supported_transport_nonfinite.any().item()):
            raise ValueError(
                "TargetPhysicalFact transport prior is non-finite on support"
            )
        coordinates = torch.where(
            camera_support[..., None],
            torch.nan_to_num(
                camera_coordinates.float(), nan=0.0, posinf=0.0, neginf=0.0
            ),
            torch.zeros_like(camera_coordinates.float()),
        )
        transport = torch.where(
            camera_support[..., None],
            torch.nan_to_num(
                camera_transport_prior.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            torch.zeros_like(camera_transport_prior.float()),
        )
        camera_weight = (
            target_posterior[:, :, None]
            * camera_validity_f[..., 0]
        )
        coordinate_first = torch.einsum(
            "bkc,bkcv->bcv", camera_weight, coordinates
        )
        coordinate_second_source = torch.stack(
            (
                coordinates[..., 0].square(),
                coordinates[..., 0] * coordinates[..., 1],
                coordinates[..., 1].square(),
            ),
            dim=-1,
        )
        coordinate_second = torch.einsum(
            "bkc,bkcv->bcv", camera_weight, coordinate_second_source
        )
        transport_first = torch.einsum(
            "bkc,bkcv->bcv", camera_weight, transport
        )
        camera_mass = camera_weight.sum(dim=1, keepdim=False)[..., None]
        camera_moments = torch.cat(
            (coordinate_first, coordinate_second, transport_first, camera_mass),
            dim=-1,
        ).flatten(start_dim=1)
        supported_existence_nonfinite = object_support & ~torch.isfinite(
            existence[..., 0]
        )
        if bool(supported_existence_nonfinite.any().item()):
            raise ValueError("TargetPhysicalFact existence is non-finite on support")
        existence_f = torch.nan_to_num(
            existence[..., 0].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        readable_mass = target_physical_mass.sum(dim=-1, keepdim=True)
        existence_mass = (
            target_posterior * existence_f
        ).sum(dim=-1, keepdim=True)
        unresolved_mass = has_target.float() - readable_mass
        physical_moments = torch.cat(
            (
                camera_moments,
                readable_mass,
                existence_mass,
                unresolved_mass.clamp_min(0.0),
            ),
            dim=-1,
        )
        target_summary = self.target_physical_value(
            physical_moments.to(dtype=objects.dtype)
        )
        target_summary = torch.where(
            has_target,
            target_summary,
            torch.zeros_like(target_summary),
        ).to(dtype=objects.dtype)
        target_summary, _ = smooth_rms_contract(target_summary, 0.35)

        typed_summary = torch.einsum(
            "bk,bktr->btr", target_physical_mass, typed_route.float()
        )
        typed_summary = torch.where(
            has_target[..., None],
            typed_summary,
            torch.zeros_like(typed_summary),
        )

        # A bounded, centered log-ratio is neutral for a uniform posterior and
        # finite for all-invalid rows.  It is a prior over K only; P2 still
        # owns support and the camera C selection.
        objects_count = int(objects.shape[1])
        bound = math.log(float(max(objects_count, 2)))
        ratio = torch.log(
            (target_posterior.float() + 1.0e-6)
            / (uniform.float() + 1.0e-6)
        )
        ratio_mean = torch.where(
            object_support,
            ratio,
            torch.zeros_like(ratio),
        ).sum(dim=-1, keepdim=True) / valid_count.clamp_min(1.0)
        centered_ratio = ratio - ratio_mean
        target_log_prior = bound * torch.tanh(centered_ratio / bound)
        target_log_prior = torch.where(
            object_support & has_target,
            target_log_prior,
            torch.zeros_like(target_log_prior),
        ).float()
        return (
            target_posterior.float(),
            target_log_prior,
            target_summary,
            typed_summary.to(dtype=objects.dtype),
            object_support,
            target_logits,
            target_physical_mass.float(),
            (
                target_posterior[:, :, None]
                * camera_validity_f[..., 0]
            ).float(),
        )

    def _typed_relevance(
        self,
        *,
        public_interval_carrier: Tensor,
        facts: ObjectFactSet,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build fixed-zero, per-type relevance without collapsing K."""

        if self.target_fact_mode:
            raise RuntimeError("legacy typed relevance is unavailable in TargetFact mode")
        if self.typed_relevance_queries is None or self.typed_temperature_logit is None:
            raise RuntimeError("legacy typed relevance parameters are missing")

        query_source = self.typed_query_norm(public_interval_carrier)
        typed_query = torch.stack(
            tuple(projection(query_source) for projection in self.typed_relevance_queries),
            dim=2,
        )  # [B,I,type,R]
        validity_values = torch.nan_to_num(
            facts.validity.to(device=public_interval_carrier.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        object_support = validity_values[..., 0] > 0.0
        typed_route = torch.stack(
            (facts.semantic, facts.appearance, facts.geometry), dim=2
        )  # [B,K,type,R]
        # Quarantine invalid producer rows before the bounded cosine and the
        # subsequent type/object reductions.  Multiplying after a projection
        # would still allow NaN * 0 to poison the upstream VJP.
        typed_route = torch.where(
            object_support[..., None, None],
            typed_route,
            torch.zeros_like(typed_route),
        )
        score = torch.einsum(
            "bitr,bktr->bikt",
            self._bounded_unit(typed_query),
            self._bounded_unit(typed_route),
        ).clamp(-1.0, 1.0)
        temperature = 0.25 + 3.75 * torch.sigmoid(
            self.typed_temperature_logit.float()
        )
        signal_probability = torch.sigmoid(
            score * temperature.to(device=score.device)[None, None, None]
        )
        # Target identity is one shared K-indexed fact, not one independently
        # chosen object per downstream P2 type.  The shared zero-start head can
        # learn whether semantic, appearance or geometry identifies the target.
        # Removing the supported K-common component makes uniform evidence an
        # exact neutral prior and leaves producer support authority unchanged.
        typed_address_score = score * temperature.to(device=score.device)[
            None, None, None
        ]
        if self.target_object_address is None:
            target_address_raw = torch.zeros_like(typed_address_score[..., 0])
        else:
            target_address_raw = self.target_object_address(typed_address_score)[..., 0]
        target_support = object_support[:, None].expand_as(target_address_raw)
        safe_target_address = torch.where(
            target_support,
            target_address_raw.float(),
            torch.zeros_like(target_address_raw, dtype=torch.float32),
        )
        first_legal_k = target_support.to(dtype=torch.int64).argmax(
            dim=-1, keepdim=True
        )
        target_address_reference = safe_target_address.gather(-1, first_legal_k)
        target_address_relative = safe_target_address - target_address_reference
        target_count = target_support.float().sum(dim=-1, keepdim=True)
        target_address_mean = torch.where(
            target_support,
            target_address_relative,
            torch.zeros_like(target_address_relative),
        ).sum(dim=-1, keepdim=True) / target_count.clamp_min(1.0)
        target_object_address_logit = torch.where(
            target_support,
            target_address_relative - target_address_mean,
            torch.zeros_like(target_address_relative),
        )
        validity = validity_values[:, None, :, None, :]
        relevance_mass = signal_probability[..., None] * validity
        relevance_mass = relevance_mass.to(dtype=typed_route.dtype)
        relevance_value = (
            relevance_mass * typed_route[:, None].to(dtype=relevance_mass.dtype)
        )

        components: list[Tensor] = []
        for type_index, projection in enumerate(
            (self.object_semantic, self.object_appearance, self.object_geometry)
        ):
            # K is a fixed identity axis.  A fixed mean preserves zero and
            # cannot cancel the optionality by renormalizing selected mass.
            selected_route = relevance_value[..., type_index, :].mean(dim=2)
            component, _ = smooth_rms_contract(projection(selected_route), 0.35)
            components.append(component)
        typed_components = torch.stack(components, dim=2)
        raw_context = typed_components.sum(dim=2) / (3.0**0.5)
        _, context_scale = smooth_rms_contract(raw_context, 0.35)
        typed_components = typed_components * context_scale[:, :, None].to(
            dtype=typed_components.dtype
        )
        return (
            relevance_mass,
            relevance_value,
            typed_components,
            target_object_address_logit,
            score,
            temperature,
        )

    def forward(
        self,
        *,
        goal_tokens: Tensor,
        goal_mask: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        facts: ObjectFactSet,
        collect_diagnostics: bool,
    ) -> tuple[ObjectIntentState, dict[str, Tensor]]:
        facts.validate()
        if goal_tokens.ndim != 3 or goal_mask.ndim != 2:
            raise ValueError("intent organizer requires full T5 tokens and mask")
        batch = int(goal_tokens.shape[0])
        goal_memory = self.goal_input(goal_tokens)
        goal_query = self.goal_queries.to(
            device=goal_memory.device, dtype=goal_memory.dtype
        ).expand(batch, -1, -1)
        protected_goal, _, goal_attention = self.goal_read(
            goal_query,
            goal_memory,
            padding_mask=~goal_mask.to(device=goal_memory.device, dtype=torch.bool),
            diagnostics=collect_diagnostics,
        )
        protected_goal = self.goal_self(protected_goal)
        paired_history, observed_state_delta = self._paired_history(
            state_history, state, executed_history
        )
        history = self.history_input(paired_history)
        for block in self.history_blocks:
            history = block(history, causal=True)
        # Materialize the producer-owned object support before any learned
        # projection.  This keeps invalid/unknown rows out of both the S
        # language read and the backward graph.
        validity_values = torch.nan_to_num(
            facts.validity.to(device=facts.content.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        object_validity = validity_values[..., 0] > 0.0
        object_mask = object_validity[..., None]
        content_input = torch.where(
            object_mask,
            facts.content,
            torch.zeros_like(facts.content),
        )
        semantic_input = torch.where(
            object_mask,
            facts.semantic,
            torch.zeros_like(facts.semantic),
        )
        appearance_input = torch.where(
            object_mask,
            facts.appearance,
            torch.zeros_like(facts.appearance),
        )
        geometry_input = torch.where(
            object_mask,
            facts.geometry,
            torch.zeros_like(facts.geometry),
        )
        content_objects = self.object_content(content_input)
        # Fuse the three observable object routes before the language read.
        # The sum is shared over K and symmetric over semantic/appearance/
        # geometry, so it supplies affordance/appearance evidence without
        # creating a color-specific slot or a learned role sidecar.  Keep the
        # update bounded and quarantine producer-invalid rows before any
        # attention reduction.
        typed_object_context = (
            self.object_semantic(semantic_input)
            + self.object_appearance(appearance_input)
            + self.object_geometry(geometry_input)
        ) / (3.0**0.5)
        typed_object_context, _ = smooth_rms_contract(typed_object_context, 0.35)
        # Keep the original content-only memory as the interval evidence
        # source so typed S relevance remains owned by its named routes.  The
        # symmetric route-enriched view is exported as the public K memory for
        # the language-conditioned coarse compiler below; this prevents an
        # appearance/geometry helper from becoming a second public interval
        # owner while still making those facts available to action selection.
        objects = content_objects + typed_object_context
        interval_objects = content_objects
        # Keep the public K memory physically safe before it reaches either
        # S's language-conditioned read or the coarse-action compiler.  The
        # producer-owned validity mask is not a learned confidence and is not
        # renormalized into a task-dependent selector.
        objects = torch.where(
            object_validity[..., None],
            objects,
            torch.zeros_like(objects),
        )
        interval_objects = torch.where(
            object_validity[..., None],
            interval_objects,
            torch.zeros_like(interval_objects),
        )
        interval_base = self.interval_identity.to(
            device=objects.device, dtype=objects.dtype
        ).expand(batch, -1, -1)
        _, goal_innovation, interval_goal_attention = self.interval_goal(
            interval_base, protected_goal, diagnostics=collect_diagnostics
        )
        _, history_innovation, interval_history_attention = self.interval_history(
            interval_base, history, diagnostics=collect_diagnostics
        )
        target_posterior: Tensor | None = None
        target_physical_mass: Tensor | None = None
        target_camera_mass: Tensor | None = None
        target_log_prior: Tensor | None = None
        target_summary: Tensor | None = None
        target_typed_summary: Tensor | None = None
        scene_context: Tensor | None = None
        p1_local_modifier_context: Tensor | None = None
        p1_scene_query_context: Tensor | None = None
        target_support: Tensor | None = None
        language_object_query = interval_base + goal_innovation + history_innovation
        object_innovation = torch.zeros_like(interval_base)
        if self.target_fact_mode:
            # TargetFact-v1 is the only language-conditioned K selector.  The
            # goal-free set summary and target summary are pooled before the
            # interval carrier.  P1 preserves K/C only under this same p, while
            # coarse may retain a separate K read only from its goal-free base
            # queries for scene facts, never for a second target decision.
            scene_context = objects.new_zeros(batch, self.hidden)
            typed_route = torch.stack(
                (facts.semantic, facts.appearance, facts.geometry), dim=2
            )
            typed_route = torch.where(
                object_validity[..., None, None],
                typed_route,
                torch.zeros_like(typed_route),
            )
            (
                target_posterior,
                target_log_prior,
                target_summary,
                target_typed_summary,
                target_support,
                target_logits,
                target_physical_mass,
                target_camera_mass,
            ) = self._target_fact(
                protected_goal=protected_goal,
                history=history,
                objects=objects,
                typed_route=typed_route,
                validity_values=validity_values,
                camera_coordinates=facts.camera_coordinates,
                camera_transport_prior=facts.camera_transport_prior,
                camera_validity=facts.camera_validity,
                existence=facts.existence,
            )
            # Observe the view consumed by public S, independently of the
            # separate coarse-action read of target_summary.
            object_innovation = target_summary[:, None, :]
            public_seed = (
                interval_base
                + goal_innovation
                + history_innovation
                + object_innovation
            )
            language_object_query = public_seed
            public_intervals = self.interval_self(public_seed)
            # P2 retains the full K/type carrier, but language no longer owns a
            # second per-type K gate.  These are producer-valid factual values;
            # the single TargetFact posterior is consumed as the P2 prior.
            route_value = typed_route.to(dtype=objects.dtype) * validity_values[
                ..., None, :
            ].to(dtype=objects.dtype)
            typed_common_value = route_value
            typed_interval_residual_value = torch.zeros(
                batch,
                4,
                int(objects.shape[1]),
                len(TYPED_INTENT_NAMES),
                int(route_value.shape[-1]),
                device=route_value.device,
                dtype=route_value.dtype,
            )
            typed_common_mass = validity_values[:, :, None, :].expand(
                -1, -1, len(TYPED_INTENT_NAMES), -1
            )
            typed_interval_residual_mass = torch.zeros(
                batch,
                4,
                int(objects.shape[1]),
                len(TYPED_INTENT_NAMES),
                1,
                device=route_value.device,
                dtype=route_value.dtype,
            )
            typed_components_list: list[Tensor] = []
            for type_index, projection in enumerate(
                (self.object_semantic, self.object_appearance, self.object_geometry)
            ):
                component, _ = smooth_rms_contract(
                    projection(target_typed_summary[:, type_index]), 0.35
                )
                typed_components_list.append(component)
            typed_policy_components = torch.stack(typed_components_list, dim=1)[
                :, None
            ].expand(-1, 4, -1, -1)
            typed_relevance_mass = (
                target_posterior[:, None, :, None, None]
                * typed_common_mass[:, None]
            ).to(dtype=route_value.dtype)
            typed_relevance_value = typed_relevance_mass * typed_route[:, None].to(
                dtype=route_value.dtype
            )
            target_object_address_logit = target_log_prior[:, None, :].expand(
                -1, 4, -1
            )
            interval_object_attention = target_posterior[:, None, :].expand(
                -1, 4, -1
            ).to(dtype=objects.dtype)
            typed_relevance_score = target_logits[:, None, :, None].expand(
                -1, 4, -1, len(TYPED_INTENT_NAMES)
            )
            typed_temperature = target_logits.new_ones(len(TYPED_INTENT_NAMES))
            # These are the only P1 language/history contexts in TargetFact
            # mode.  The target modifier is consumed only after p×G3 has fixed
            # the K/coarse address.  Coverage receives the observation-history
            # innovation and goal-free scene summary, never the protected goal
            # or the goal-conditioned public interval carrier.
            # Static P1 receives the supervised physical A0 explicitly at the
            # policy boundary.  Exporting a goal/history hidden here would
            # recreate a per-K language selector before p contraction.
            p1_local_modifier_context = torch.zeros_like(interval_base)
            p1_scene_query_context = torch.zeros_like(interval_base)
        else:
            # Compatibility graph: retain the pre-TargetFact behavior exactly
            # for post_pool_only/shared_target_prior_v1 checkpoints.
            language_object_query = interval_base + goal_innovation + history_innovation
            if self.interval_object_query_mode == "history_only":
                language_object_query = interval_base + history_innovation
            if self.interval_object is None:
                raise RuntimeError("legacy intent graph has no interval object reader")
            _, object_innovation, interval_object_attention = self.interval_object(
                language_object_query,
                interval_objects,
                padding_mask=~object_validity,
                diagnostics=collect_diagnostics,
            )
            public_intervals = self.interval_self(
                interval_base
                + goal_innovation
                + history_innovation
                + object_innovation
            )
            (
                typed_relevance_mass,
                typed_relevance_value,
                typed_policy_components,
                target_object_address_logit,
                typed_relevance_score,
                typed_temperature,
            ) = self._typed_relevance(
                public_interval_carrier=public_intervals,
                facts=facts,
            )
            typed_common_mass, typed_interval_residual_mass = (
                _interval_common_residual(typed_relevance_mass)
            )
            typed_common_value, typed_interval_residual_value = (
                _interval_common_residual(typed_relevance_value)
            )
        typed_policy_context = typed_policy_components.sum(dim=2) / (3.0**0.5)
        policy_intervals = public_intervals + typed_policy_context
        temporal_base = self.temporal_identity.to(
            device=public_intervals.device, dtype=public_intervals.dtype
        ).expand(batch, -1, -1)
        temporal, _, _ = self.temporal_read(
            temporal_base, public_intervals, diagnostics=False
        )
        state_change_values = self.state_change_input(observed_state_delta)
        state_change_history, state_change_attention = self.state_change_read(
            self.state_change_query_norm(
                self.state_change_query.to(
                    device=history.device, dtype=history.dtype
                ).expand(batch, -1, -1)
            ),
            self.state_change_key_norm(history),
            state_change_values,
            need_weights=collect_diagnostics,
            average_attn_weights=True,
        )
        if state_change_attention is None:
            state_change_attention = history.new_zeros(batch, 1, history.shape[1])
        state_change_history = state_change_history[:, 0]
        transport_prior = facts.transport_prior.to(
            device=objects.device,
            dtype=objects.dtype,
        )
        transport_prior = torch.where(
            object_mask.to(device=transport_prior.device),
            transport_prior,
            torch.zeros_like(transport_prior),
        )
        transport_tokens = self.state_change_transport(transport_prior)
        transport_validity = validity_values.to(
            device=transport_tokens.device, dtype=transport_tokens.dtype
        )
        state_change_transport = (
            transport_tokens * transport_validity
        ).sum(dim=1) / transport_validity.sum(dim=1).clamp_min(1.0)
        state_change_evidence, _ = smooth_rms_contract(
            self.state_change_fuse(
                torch.cat((state_change_history, state_change_transport), dim=-1)
            ),
            0.20,
        )
        state_out = ObjectIntentState(
            protected_goal_set=protected_goal,
            history_tokens=history,
            object_tokens=objects,
            public_interval_carrier=public_intervals,
            policy_interval_context=policy_intervals,
            temporal_queries=temporal,
            state_change_evidence=state_change_evidence,
            target_object_address_logit=target_object_address_logit,
            typed_common_mass=typed_common_mass,
            typed_common_value=typed_common_value,
            typed_interval_residual_mass=typed_interval_residual_mass,
            typed_interval_residual_value=typed_interval_residual_value,
            typed_policy_components=typed_policy_components,
            goal_attention=goal_attention,
            interval_goal_attention=interval_goal_attention,
            interval_history_attention=interval_history_attention,
            interval_object_attention=interval_object_attention,
            object_validity=validity_values.detach().to(
                device=objects.device,
                dtype=torch.float32,
            ),
            target_posterior=target_posterior,
            target_physical_mass=target_physical_mass,
            target_camera_mass=target_camera_mass,
            target_log_prior=target_log_prior,
            target_summary=target_summary,
            target_typed_summary=target_typed_summary,
            scene_context=scene_context,
            p1_local_modifier_context=p1_local_modifier_context,
            p1_scene_query_context=p1_scene_query_context,
            target_support=target_support,
        )
        state_out.validate(horizon=self.horizon, hidden=self.hidden)
        if not collect_diagnostics:
            return state_out, {}
        metrics: dict[str, Tensor] = {
            "object_intent_interval_query_goal_enabled": objects.new_tensor(
                float(self.interval_object_query_mode == "goal_history")
            ),
            "object_intent_target_fact_mode_enabled": objects.new_tensor(
                float(self.target_fact_mode)
            ),
            "object_intent_goal_attention_entropy": normalized_entropy(
                goal_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_goal_entropy": normalized_entropy(
                interval_goal_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_history_entropy": normalized_entropy(
                interval_history_attention, dim=-1
            ).detach().mean(),
            "object_intent_interval_object_entropy": normalized_entropy(
                interval_object_attention, dim=-1
            ).detach().mean(),
            "object_intent_language_object_attention_entropy": normalized_entropy(
                interval_object_attention, dim=-1
            ).detach().mean(),
            "object_intent_language_object_attention_valid_fraction": (
                object_validity.detach().float().mean()
            ),
            "object_intent_public_interval_variation": public_intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_policy_interval_variation": policy_intervals.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_temporal_variation": temporal.detach().float().std(
                dim=1, unbiased=False
            ).mean(),
            "object_intent_goal_innovation_rms": goal_innovation.detach().float().square().mean().sqrt(),
            "object_intent_history_innovation_rms": history_innovation.detach().float().square().mean().sqrt(),
            "object_intent_object_innovation_rms": object_innovation.detach().float().square().mean().sqrt(),
            "object_intent_typed_action_context_rms": typed_policy_context.detach().float().square().mean().sqrt(),
            "object_intent_target_address_logit_rms": target_object_address_logit.detach().square().mean().sqrt(),
            "object_intent_target_address_logit_k_variation": target_object_address_logit.detach().std(
                dim=2, unbiased=False
            ).mean(),
            "object_intent_target_address_logit_k_center_error": (
                (
                    target_object_address_logit.detach()
                    * object_validity[:, None].detach().float()
                ).sum(dim=2)
                / object_validity.detach().float().sum(dim=1)[:, None].clamp_min(1.0)
            ).abs().amax(),
            "object_intent_target_posterior_entropy": (
                normalized_entropy(target_posterior, dim=-1).detach().mean()
                if target_posterior is not None
                else objects.new_zeros(())
            ),
            "object_intent_target_posterior_max": (
                target_posterior.detach().amax(dim=-1).mean()
                if target_posterior is not None
                else objects.new_zeros(())
            ),
            "object_intent_target_log_prior_rms": (
                target_log_prior.detach().square().mean().sqrt()
                if target_log_prior is not None
                else objects.new_zeros(())
            ),
            "object_intent_target_summary_rms": (
                target_summary.detach().float().square().mean().sqrt()
                if target_summary is not None
                else objects.new_zeros(())
            ),
            "object_intent_scene_context_rms": (
                scene_context.detach().float().square().mean().sqrt()
                if scene_context is not None
                else objects.new_zeros(())
            ),
            "object_intent_observed_state_delta_rms": observed_state_delta.detach().float().square().mean().sqrt(),
            "object_intent_observed_transport_rms": transport_prior.detach().float().square().mean().sqrt(),
            "object_intent_state_change_history_rms": state_change_history.detach().float().square().mean().sqrt(),
            "object_intent_state_change_transport_rms": state_change_transport.detach().float().square().mean().sqrt(),
            "object_intent_state_change_evidence_rms": state_change_evidence.detach().float().square().mean().sqrt(),
            "object_intent_state_change_attention_entropy": normalized_entropy(
                state_change_attention, dim=-1
            ).detach().mean(),
        }
        # The returned attention probability is an auxiliary observation; it
        # is not consumed by the action path and therefore has no ordinary
        # owner VJP.  Observe the two value tensors that actually carry the
        # repaired language/object edge instead: the conditioned K query and
        # its bounded object innovation.  Keep the historical attention name
        # as a compatibility alias, but bind it to the consumed innovation
        # rather than reporting a guaranteed zero from a sibling output.
        register_gradient_rms_metric(
            language_object_query,
            metrics,
            "gradient_tensor_s_language_object_query_rms",
        )
        register_gradient_rms_metric(
            object_innovation,
            metrics,
            "gradient_tensor_s_language_object_update_rms",
        )
        register_gradient_rms_metric(
            object_innovation,
            metrics,
            "gradient_tensor_s_language_object_attention_rms",
        )
        if target_posterior is not None:
            register_gradient_rms_metric(
                target_posterior,
                metrics,
                "gradient_tensor_s_target_posterior_rms",
            )
        if self.target_object_address is not None:
            register_gradient_rms_metric(
                target_object_address_logit,
                metrics,
                "gradient_tensor_s_target_object_address_logit_rms",
            )
        raw_routes = (semantic_input, appearance_input, geometry_input)
        for type_index, name in enumerate(TYPED_INTENT_NAMES):
            mass = typed_relevance_mass[..., type_index, 0].detach().float()
            selected = typed_relevance_value[..., type_index, :].detach().float()
            component = typed_policy_components[..., type_index, :].detach().float()
            metrics.update(
                {
                    f"object_intent_{name}_route_raw_rms": raw_routes[type_index]
                    .detach()
                    .float()
                    .square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_relevance_mass": mass.mean(),
                    f"object_intent_{name}_null_mass": (1.0 - mass).mean(),
                    f"object_intent_{name}_selected_value_rms": selected.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_object_variation": selected.std(
                        dim=2, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_interval_variation": selected.std(
                        dim=1, unbiased=False
                    ).mean(),
                    f"object_intent_{name}_action_context_rms": component.square()
                    .mean()
                    .sqrt(),
                    f"object_intent_{name}_score_abs": typed_relevance_score[
                        ..., type_index
                    ]
                    .detach()
                    .abs()
                    .mean(),
                    f"object_intent_{name}_temperature": typed_temperature[
                        type_index
                    ].detach(),
                }
            )
        return state_out, metrics


class FuturePlanRecognizer(nn.Module):
    """Training-only V120 whole-segment posterior."""

    def __init__(
        self,
        *,
        hidden: int,
        action_dim: int,
        state_dim: int,
        content_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.content_dim = int(content_dim)
        self.action_input = nn.Linear(action_dim, hidden, bias=False)
        self.state_input = nn.Linear(state_dim, hidden, bias=False)
        self.effect_input = nn.Linear(content_dim, hidden, bias=False)
        self.interval_identity = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        self.block = _SelfBlock(hidden, heads)
        self.action_reconstruction = nn.Linear(hidden, action_dim, bias=False)
        self.state_reconstruction = nn.Linear(hidden, state_dim, bias=False)
        self.effect_reconstruction = nn.Linear(hidden, content_dim, bias=False)

    def forward(
        self,
        *,
        future_action: Tensor,
        future_state: Tensor,
        teacher: FutureObjectDynamics | None,
        current_loss_support: Tensor,
    ) -> FuturePlanRecognition:
        if future_action.ndim != 3 or future_state.ndim != 3:
            raise ValueError("plan recognizer requires full future action/state sequences")
        length = min(int(future_action.shape[1]), int(future_state.shape[1]))
        slices = _interval_slices(length)
        action_summary = torch.stack(
            [future_action[:, row].mean(dim=1) for row in slices], dim=1
        )
        state_summary = torch.stack(
            [future_state[:, row].mean(dim=1) for row in slices], dim=1
        )
        if current_loss_support.ndim != 4 or int(current_loss_support.shape[-1]) != 1:
            raise ValueError("recognizer current loss support must be [B,K,C,1]")
        if int(current_loss_support.shape[0]) != int(future_action.shape[0]):
            raise ValueError("recognizer current loss support batch does not align")
        # This is the same detached current-fact support used by the object
        # losses.  Future reliability and selector validity are deliberately
        # absent: neither may shrink a supervised target or create a routing
        # shortcut.  A camera reduction is performed exactly once because W
        # exports object-level future geometry/content.
        object_support = current_loss_support.detach().float().amax(dim=2)
        if teacher is None:
            effect_summary = future_action.new_zeros(
                future_action.shape[0], 4, self.content_dim
            )
            teacher_valid = future_action.new_zeros(future_action.shape[0], 4, 1)
        else:
            teacher.validate()
            expected_support = (
                teacher.semantic_delta.shape[0],
                teacher.semantic_delta.shape[2],
                1,
            )
            if tuple(object_support.shape) != expected_support:
                raise ValueError("recognizer object support does not align with teacher")
            support = object_support[:, None]
            denominator = support.sum(dim=2).clamp_min(1.0)
            effect_summary = (
                teacher.semantic_delta.detach().float() * support
            ).sum(dim=2) / denominator
            effect_summary = effect_summary.to(dtype=teacher.semantic_delta.dtype)
            teacher_valid = (support.sum(dim=2) > 0).to(
                dtype=teacher.semantic_delta.dtype
            ).expand(-1, teacher.semantic_delta.shape[1], -1)
        token = (
            self.action_input(action_summary)
            + self.state_input(state_summary)
            + self.effect_input(effect_summary)
            + self.interval_identity.to(
                device=future_action.device, dtype=future_action.dtype
            )
        )
        token = self.block(token)
        action_pred = self.action_reconstruction(token)
        state_pred = self.state_reconstruction(token)
        effect_pred = self.effect_reconstruction(token)
        effect_error = (
            (effect_pred.float() - effect_summary.detach().float()).square()
            * teacher_valid.detach().float()
        ).sum() / teacher_valid.detach().float().sum().clamp_min(1.0) / float(
            self.content_dim
        )
        reconstruction = (
            (action_pred.float() - action_summary.detach().float()).square().mean()
            + (state_pred.float() - state_summary.detach().float()).square().mean()
            + 0.25 * effect_error
        )
        result = FuturePlanRecognition(
            interval_targets=token.detach(),
            action_summary=action_summary.detach(),
            state_summary=state_summary.detach(),
            effect_summary=effect_summary.detach(),
            reconstruction_loss=reconstruction,
        )
        result.validate(hidden=self.hidden)
        return result


class CoarseActionIntent(nn.Module):
    """V120 online clean action intent used exactly once by W."""

    def __init__(
        self,
        *,
        hidden: int,
        action_dim: int,
        heads: int,
        horizon: int = 24,
        action_condition_mode: str = "interval_mean_v1",
        target_fact_mode: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        if self.horizon != 24:
            raise ValueError("the active coarse action contract requires 24 rows")
        self.action_condition_mode = str(action_condition_mode)
        self.target_fact_mode = bool(target_fact_mode)
        if self.action_condition_mode not in {
            "interval_mean_v1",
            "sequence_prefix_v1",
        }:
            raise ValueError("unknown coarse action condition mode")
        self.query = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        # Keep the trained four-query basis as the shared temporal scaffold.
        # The new zero-initialized row offsets add row-specific capacity
        # without replacing or resetting the existing action head/query.
        self.sequence_row_offset: nn.Parameter | None = None
        if self.action_condition_mode == "sequence_prefix_v1":
            self.sequence_row_offset = nn.Parameter(
                torch.zeros(1, self.horizon, hidden)
            )
        self.intent_read = _CrossRead(hidden, heads)
        # A target-action bottleneck cannot expose a second K-resolved scene
        # reader to the full-goal action block.  Legacy modes retain the reader
        # unchanged; the bottleneck mode obtains scene physics only after A0
        # through W/P/CT.
        self.object_read: _CrossRead | None = (
            None if self.target_fact_mode else _CrossRead(hidden, heads)
        )
        self.history_read = _CrossRead(hidden, heads)
        self.block = _SelfBlock(hidden, heads)
        self.action_head = nn.Linear(hidden, action_dim, bias=False)

    def forward(
        self,
        intent: ActionIntentDock,
        *,
        future_action: Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> CoarseActionIntentState:
        intent.validate(hidden=int(self.query.shape[-1]))
        batch = int(intent.public_interval_carrier.shape[0])
        query_seed = self.query
        if self.action_condition_mode == "sequence_prefix_v1":
            if self.sequence_row_offset is None:
                raise RuntimeError("sequence coarse action has no row offsets")
            query_seed = F.interpolate(
                self.query.transpose(1, 2),
                size=self.horizon,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2) + self.sequence_row_offset
        query = query_seed.to(
            device=intent.public_interval_carrier.device,
            dtype=intent.public_interval_carrier.dtype,
        ).expand(batch, -1, -1)
        _, intent_delta, _ = self.intent_read(
            query,
            intent.public_interval_carrier,
            diagnostics=collect_diagnostics,
        )
        # Carry the language-conditioned interval read into both auxiliary
        # memories.  This keeps the object and history reads in the same S
        # query frame instead of letting a learned query silently wash out the
        # instruction before the physical proposal is formed.
        conditioned_query = query + intent_delta
        _, history_delta, _ = self.history_read(
            conditioned_query,
            intent.history_memory,
            diagnostics=collect_diagnostics,
        )
        object_padding_mask = None
        if intent.public_object_validity is not None:
            validity = intent.public_object_validity.to(
                device=intent.public_object_memory.device,
                dtype=torch.float32,
            ).squeeze(-1).clamp(0.0, 1.0)
            object_padding_mask = ~(validity > 0.0)
        if self.target_fact_mode:
            if intent.target_summary is None or intent.scene_context is None:
                raise ValueError(
                    "TargetFact coarse path requires target_summary and scene_context"
                )
            target_context = intent.target_summary[:, None, :].expand(
                -1, int(conditioned_query.shape[1]), -1
            )
            token = self.block(
                conditioned_query + target_context + history_delta
            )
        else:
            if self.object_read is None:
                raise RuntimeError("legacy coarse action lost its object reader")
            _, object_delta, _ = self.object_read(
                conditioned_query,
                intent.public_object_memory,
                padding_mask=object_padding_mask,
                diagnostics=collect_diagnostics,
            )
            token = self.block(conditioned_query + object_delta + history_delta)
        action_prediction = self.action_head(token)
        if future_action is None:
            target = None
            loss = action_prediction.new_zeros(())
        else:
            if self.action_condition_mode == "sequence_prefix_v1":
                if int(future_action.shape[1]) < self.horizon:
                    raise ValueError(
                        "sequence coarse supervision requires 24 future rows"
                    )
                target = future_action[:, : self.horizon].detach()
            else:
                slices = _interval_slices(int(future_action.shape[1]))
                target = torch.stack(
                    [future_action[:, row].mean(dim=1) for row in slices], dim=1
                ).detach()
            loss = (action_prediction.float() - target.float()).square().mean()
        return CoarseActionIntentState(
            tokens=token,
            action_prediction=action_prediction,
            target=target,
            loss=loss,
        )


__all__ = [
    "CoarseActionIntent",
    "FuturePlanRecognizer",
    "StatelessObjectIntentOrganizer",
]
