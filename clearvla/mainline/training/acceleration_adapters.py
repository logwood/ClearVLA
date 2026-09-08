"""Adapters for version-specific capture preparation.

The CUDA-Graph backend imports only the generic contract.  Mainline-specific
knowledge lives here, so another architecture branch can provide a different
adapter without editing the backend itself.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from .acceleration_contract import (
    TrainingCompilePlan,
    TrainingCompileRegion,
    TrainingEagerBoundary,
    module_structure_signature,
)

_NUMERICS_OPTIONS: tuple[tuple[str, str | int | bool], ...] = (
    ("deterministic", True),
    ("force_same_precision", True),
    ("emulate_precision_casts", True),
    ("fallback_random", True),
    ("use_fast_math", False),
    ("cuda.use_fast_math", False),
    ("split_reductions", False),
    ("triton.cooperative_reductions", False),
    ("triton.force_cooperative_reductions", False),
    ("triton.mix_order_reduction", False),
    ("triton.persistent_reductions", False),
    ("triton.tile_reductions", False),
    ("epilogue_fusion", False),
    ("prologue_fusion", False),
)

_PROFILE_SCOPES = {"mainline", "mmdit", "visual", "combined"}
_PROFILE_STRATEGIES = {"fast", "precision", "partitioned"}


def _first_existing(names: set[str], *candidates: str) -> str | None:
    return next((candidate for candidate in candidates if candidate in names), None)


def _direct_children(names: set[str], parent: str) -> tuple[str, ...]:
    prefix = parent + "."
    return tuple(
        sorted(
            name
            for name in names
            if name.startswith(prefix) and "." not in name[len(prefix) :]
        )
    )


def _numerical_boundaries(
    model: Any,
    compile_paths: tuple[str, ...],
) -> tuple[TrainingEagerBoundary, ...]:
    """Keep known reductions/random kernels outside compiled fragments."""

    modules = dict(model.named_modules())
    boundaries: dict[tuple[str, str], TrainingEagerBoundary] = {}
    sensitive_type_tokens = (
        "attention",
        "attn",
        "dropout",
        "norm",
        "pool",
        "softmax",
    )
    for parent in compile_paths:
        prefix = parent + "."
        for path, module in modules.items():
            if not path.startswith(prefix):
                continue
            type_name = type(module).__qualname__.lower()
            token = next(
                (value for value in sensitive_type_tokens if value in type_name),
                None,
            )
            if token is None:
                continue
            boundary = TrainingEagerBoundary(
                module_path=path,
                reason=f"preserve-eager-{token}",
            )
            boundaries[(path, "forward")] = boundary

        parent_module = modules[parent]
        # These helpers contain functional reductions/softmax operations that
        # are not represented by a child nn.Module and therefore cannot be
        # discovered by the type scan above.
        for method_name, reason in (
            ("_attention", "preserve-eager-softmax-reduction"),
            ("_unit_rms", "preserve-eager-rms-reduction"),
            ("prepare_static_context", "preserve-eager-context-reductions"),
            ("_contract_residual", "preserve-eager-rms-contract"),
            ("_coeff", "preserve-eager-attention-and-mean"),
        ):
            if callable(getattr(parent_module, method_name, None)):
                boundary = TrainingEagerBoundary(
                    module_path=parent,
                    method_name=method_name,
                    reason=reason,
                )
                boundaries[(parent, method_name)] = boundary
    return tuple(boundaries[key] for key in sorted(boundaries))


@dataclass(frozen=True)
class MainlineTrainingAccelerationAdapter:
    """Capture preparation and topology identity for the active mainline.

    These flags are implementation-local rewrites whose independent
    equivalence gates already exist.  Keeping them in an adapter prevents the
    generic runner from reaching into ``execution_bottom`` or knowing the
    decoder's private fields.
    """

    name: str = "clearvla-mainline-cuda-graph-v1"

    @staticmethod
    def _decoder(engine: Any) -> Any | None:
        execution_bottom = getattr(engine.model, "execution_bottom", None)
        if execution_bottom is not None:
            return getattr(execution_bottom, "decoder", None)
        bottom = getattr(engine.model, "bottom", None)
        return getattr(bottom, "decoder", None)

    def prepare(self, engine: Any) -> None:
        decoder = self._decoder(engine)
        if decoder is None:
            return
        decoder._static_neutral_owner = True
        decoder._training_candidate_prefix_reuse = True

    def topology_signature(self, engine: Any) -> Hashable:
        decoder = self._decoder(engine)
        progress = getattr(decoder, "_execution_progress_value", None)
        if progress is None:
            return ("static",)
        return (
            "execution-progress",
            "identity" if float(progress) <= 0.0 else "active",
        )

    def structure_signature(self, engine: Any) -> Hashable:
        return module_structure_signature(engine.model)

    def compile_plan(self, engine: Any, profile: str) -> TrainingCompilePlan:
        """Build one layout-aware candidate plan from semantic module groups.

        The profile grammar is ``{fast|precision|partitioned}-`` followed by
        ``{visual|mainline|mmdit|combined}``.  Every plan remains a candidate
        until the Pen and RDT gates promote that exact plan fingerprint.
        """

        try:
            strategy, scope = profile.split("-", 1)
        except ValueError as error:
            raise ValueError(f"invalid selective compile profile: {profile!r}") from error
        if strategy not in _PROFILE_STRATEGIES or scope not in _PROFILE_SCOPES:
            supported = tuple(
                f"{left}-{right}"
                for left in sorted(_PROFILE_STRATEGIES)
                for right in sorted(_PROFILE_SCOPES)
            )
            raise ValueError(
                f"unknown selective compile profile {profile!r}; supported={supported!r}"
            )

        model = engine.model
        modules = dict(model.named_modules())
        names = set(modules)
        decoder_path = _first_existing(names, "execution_bottom.decoder", "bottom.decoder")
        if decoder_path is None:
            raise ValueError("mainline compile plan cannot locate the execution decoder")
        observation_path = _first_existing(
            names,
            "observation.compiler.encoder",
            "observation.encoder",
        )
        grounding_path = _first_existing(
            names,
            "grounding.blocks",
            "top.grounding_blocks",
        )
        p1_path = _first_existing(
            names,
            "p1.dynamic_policy_block",
            "bottom.p1_policy_block",
        )
        world_prefix = _first_existing(names, "world.dynamics", "top.dynamics")
        layer_heads_path = _first_existing(
            names,
            "execution_bottom.layer_contract_heads",
            "bottom.layer_contract_heads",
        )

        mainline_paths: list[str] = []
        if grounding_path is not None:
            mainline_paths.extend(_direct_children(names, grounding_path))
        if p1_path is not None:
            mainline_paths.append(p1_path)
        if world_prefix is not None:
            mainline_paths.extend(
                path
                for path in (f"{world_prefix}.w1", f"{world_prefix}.w2")
                if path in names
            )
        if "transition.v120_transition" in names:
            mainline_paths.append("transition.v120_transition")
        if layer_heads_path is not None:
            mainline_paths.extend(_direct_children(names, layer_heads_path))

        mmdit_paths = _direct_children(names, f"{decoder_path}.blocks")
        visual_paths: tuple[str, ...] = ()
        if observation_path is not None:
            if strategy == "partitioned":
                visual_candidates = (
                    f"{observation_path}.flow.encoder",
                    f"{observation_path}.flow.context",
                    f"{observation_path}.flow.update",
                    f"{observation_path}.flow.delta_head",
                    f"{observation_path}.flow.initial_residual",
                    f"{observation_path}.raw_flow.pyramid",
                    f"{observation_path}.raw_flow.mid.feature",
                    f"{observation_path}.raw_flow.mid.update",
                    f"{observation_path}.raw_flow.high.feature",
                    f"{observation_path}.raw_flow.high.update",
                )
            else:
                visual_candidates = (
                    f"{observation_path}.flow",
                    f"{observation_path}.raw_flow",
                )
            visual_paths = tuple(path for path in visual_candidates if path in names)

        family_paths = {
            "visual": visual_paths,
            "mainline": tuple(mainline_paths),
            "mmdit": mmdit_paths,
        }
        selected_families = (
            ("visual", "mainline", "mmdit")
            if scope == "combined"
            else (scope,)
        )
        options = _NUMERICS_OPTIONS if strategy != "fast" else ()
        mode = "default" if strategy == "fast" else None
        regions = tuple(
            TrainingCompileRegion(
                name=family,
                module_paths=family_paths[family],
                mode=mode,
                options=options,
            )
            for family in selected_families
            if family_paths[family]
        )
        missing_families = tuple(
            family for family in selected_families if not family_paths[family]
        )
        if missing_families:
            raise ValueError(
                f"compile profile {profile!r} cannot resolve families "
                f"{missing_families!r} on this model layout"
            )
        compile_paths = tuple(path for region in regions for path in region.module_paths)
        boundaries = (
            _numerical_boundaries(model, compile_paths)
            if strategy == "partitioned"
            else ()
        )
        layout = "modular" if decoder_path.startswith("execution_bottom") else "legacy"
        return TrainingCompilePlan(
            name=f"clearvla-{layout}-{profile}-v1",
            regions=regions,
            eager_boundaries=boundaries,
            acceptance="candidate",
            rng_policy=(
                "eager-boundary-or-explicit-input"
                if strategy == "partitioned"
                else "inductor-fallback-random"
            ),
            numerical_policy=(
                "partition-known-reductions-and-force-precision"
                if strategy == "partitioned"
                else (
                    "force-precision-with-inductor-reduction-order"
                    if strategy == "precision"
                    else "unrestricted-inductor-candidate"
                )
            ),
        )


__all__ = ["MainlineTrainingAccelerationAdapter"]
