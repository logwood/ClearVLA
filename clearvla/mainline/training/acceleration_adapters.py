"""Adapters for version-specific capture preparation.

The CUDA-Graph backend imports only the generic contract.  Mainline-specific
knowledge lives here, so another architecture branch can provide a different
adapter without editing the backend itself.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

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

_AUTOTUNED_NUMERICS_OPTIONS: tuple[tuple[str, str | int | bool], ...] = (
    *_NUMERICS_OPTIONS,
    # Match ``max-autotune-no-cudagraphs`` kernel search without surrendering
    # any of the precision and reduction constraints above.  Outer training
    # CUDA Graph capture remains owned by the shared runner.
    ("max_autotune", True),
    ("coordinate_descent_tuning", True),
)

_PROFILE_FAMILIES = {
    "conditioning",
    "bridge",
    "intent",
    "factual",
    "factual-microgrid",
    "factual-microgrid-values",
    "factual-refiners",
    "factual-shell-eager-values",
    "factual-shell-eager-microgrid",
    "factual-shell-safe",
    "factual-atomic",
    "policy",
    "execution",
    "mainline",
    "mainline-grounding",
    "mainline-grounding-safe",
    "mainline-p1",
    "mainline-world",
    "mainline-transition",
    "mainline-layer-heads",
    "mmdit",
    "flow",
    "raw",
    "visual",
}
_PROFILE_ALIASES: dict[str, tuple[str, ...]] = {
    "core": ("mainline", "mmdit"),
    # Preserve the pre-existing meaning of ``combined``.  The new semantic
    # families are opt-in until their own Pen/RDT gates have passed.
    "combined": ("visual", "mainline", "mmdit"),
    "expanded": (
        "mainline",
        "conditioning",
        "bridge",
        "intent",
        "factual",
        "policy",
        "execution",
    ),
}
_PROFILE_STRATEGIES = {
    "fast",
    "precision",
    "autotune",
    "partitioned",
    "hybrid",
}
_HYBRID_SCOPES = {"mainline-flow", "core-flow", "mainline-mmdit"}
_CUDNN_BENCHMARK_ORIGINAL_FORWARD = (
    "_clearvla_training_cudnn_benchmark_original_forward"
)
_CUDNN_DEFAULT_ORIGINAL_FORWARD = (
    "_clearvla_training_cudnn_default_original_forward"
)
_ATOMIC_FACTUAL_CONTRACTION_METHOD = (
    "_configured_typed_microgrid_rgb_detail_contraction"
)
_ATOMIC_FACTUAL_CONTRACTION_STATE = (
    "_clearvla_training_atomic_factual_contraction_state"
)
_CUDNN_CONVOLUTION_TYPES = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
)


def _set_module_cudnn_benchmark(module: Any, *, enabled: bool) -> bool:
    """Toggle cuDNN search around one module without changing its state ABI."""

    original = getattr(module, _CUDNN_BENCHMARK_ORIGINAL_FORWARD, None)
    if enabled:
        if original is not None:
            return False
        original = module.forward

        @wraps(original)
        def benchmark_forward(*args: Any, **kwargs: Any) -> Any:
            with torch.backends.cudnn.flags(
                enabled=bool(torch.backends.cudnn.enabled),
                benchmark=True,
                deterministic=bool(torch.backends.cudnn.deterministic),
                allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            ):
                return original(*args, **kwargs)

        setattr(module, _CUDNN_BENCHMARK_ORIGINAL_FORWARD, original)
        module.forward = benchmark_forward
        return True
    if original is None:
        return False
    module.forward = original
    delattr(module, _CUDNN_BENCHMARK_ORIGINAL_FORWARD)
    return True


def _set_module_cudnn_default_guard(module: Any, *, enabled: bool) -> bool:
    """Keep one convolution forward on a benchmark-disabled plan.

    Backward is deliberately owned by the engine-level phase policy.  A
    convolution can be called repeatedly in one graph, while parameter hooks
    may run only once for the accumulated gradient; coupling flag restoration
    to those hooks therefore leaks process-global state.  This forward-only
    context is balanced for every call and adds no state-dict entries.
    """

    original = getattr(module, _CUDNN_DEFAULT_ORIGINAL_FORWARD, None)
    if enabled:
        if original is not None:
            return False
        original = module.forward

        @wraps(original)
        def default_forward(*args: Any, **kwargs: Any) -> Any:
            with torch.backends.cudnn.flags(
                enabled=bool(torch.backends.cudnn.enabled),
                benchmark=False,
                deterministic=bool(torch.backends.cudnn.deterministic),
                allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            ):
                return original(*args, **kwargs)

        setattr(module, _CUDNN_DEFAULT_ORIGINAL_FORWARD, original)
        module.forward = default_forward
        return True
    if original is None:
        return False
    module.forward = original
    delattr(module, _CUDNN_DEFAULT_ORIGINAL_FORWARD)
    return True


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


def _resolve_profile_families(scope: str) -> tuple[str, ...]:
    """Expand a composable semantic-family expression deterministically.

    A profile such as ``precision-mainline+intent+factual`` can therefore be
    swept one family at a time without adding a bespoke branch for every
    combination.  Aliases remain conveniences only; the expanded regions are
    serialized in the compile-plan fingerprint.
    """

    tokens = tuple(scope.split("+"))
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"invalid selective compile scope: {scope!r}")
    families: list[str] = []
    for token in tokens:
        if token in _PROFILE_FAMILIES:
            expanded = (token,)
        else:
            expanded = _PROFILE_ALIASES.get(token, ())
        if not expanded:
            supported = tuple(sorted((*_PROFILE_FAMILIES, *_PROFILE_ALIASES)))
            raise ValueError(
                f"unknown selective compile scope token {token!r}; "
                f"supported={supported!r}"
            )
        for family in expanded:
            if family not in families:
                families.append(family)
    return tuple(families)


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
            (
                "_variance_floored_centered_norm",
                "preserve-eager-centered-normalization-reduction",
            ),
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

    @staticmethod
    def _factual_reader(engine: Any) -> Any | None:
        factual_reader = getattr(engine.model, "factual_reader", None)
        if factual_reader is not None:
            return factual_reader
        p1 = getattr(engine.model, "p1", None)
        factual_reader = getattr(p1, "factual_reader", None)
        if factual_reader is not None:
            return factual_reader
        execution_bottom = getattr(engine.model, "execution_bottom", None)
        return getattr(execution_bottom, "factual_reader", None)

    def prepare(self, engine: Any) -> None:
        decoder = self._decoder(engine)
        if decoder is None:
            return
        decoder._static_neutral_owner = True
        decoder._training_candidate_prefix_reuse = True

    def set_context_reuse(self, engine: Any, *, enabled: bool) -> int:
        """Toggle attached common-subexpression reuse owned by the decoder."""

        decoder = self._decoder(engine)
        if decoder is None:
            return 0
        for name in (
            "_reuse_prepared_block_contexts",
            "_reuse_prepared_controller_context",
            "_reuse_terminal_candidate_velocity",
        ):
            setattr(decoder, name, bool(enabled))
        return 3

    def set_batched_raw_flow_sampling(self, engine: Any, *, enabled: bool) -> int:
        """Toggle the opt-in batched local raw-flow sampler.

        The sampler groups independent local-offset ``grid_sample`` calls
        without changing registered parameters, buffers, checkpoint keys or
        tensor shapes.  This version-owned hook lets shared acceleration tools
        probe the launch organization without importing a raw-flow class.
        """

        observation = getattr(engine.model, "observation", None)
        compiler = getattr(observation, "compiler", None)
        encoder = getattr(compiler, "encoder", None)
        if encoder is None:
            encoder = getattr(observation, "encoder", None)
        raw_flow = getattr(encoder, "raw_flow", None)
        if raw_flow is None:
            return 0
        changed = 0
        for name in ("mid", "high"):
            refiner = getattr(raw_flow, name, None)
            if refiner is None or not hasattr(refiner, "_batched_samples"):
                continue
            setattr(refiner, "_batched_offset_sampling", bool(enabled))
            changed += 1
        return changed

    def set_atomic_factual_contraction(
        self,
        engine: Any,
        *,
        enabled: bool,
    ) -> int:
        """Toggle the compiler-visible exact-order factual contraction.

        This hook changes one instance method only.  It adds no module,
        parameter, buffer or checkpoint key and restores the exact prior
        instance/class lookup when disabled.  Callers must install it before
        selective compilation or CUDA Graph capture.
        """

        reader = self._factual_reader(engine)
        if reader is None:
            return 0
        state = getattr(reader, _ATOMIC_FACTUAL_CONTRACTION_STATE, None)
        if enabled:
            if state is not None:
                return 0
            original = getattr(
                reader,
                _ATOMIC_FACTUAL_CONTRACTION_METHOD,
                None,
            )
            if not callable(original):
                return 0
            from .factual_contraction import (
                atomic_factual_rgb_detail_contraction,
            )

            had_instance_method = (
                _ATOMIC_FACTUAL_CONTRACTION_METHOD in reader.__dict__
            )
            original_instance_method = reader.__dict__.get(
                _ATOMIC_FACTUAL_CONTRACTION_METHOD
            )
            setattr(
                reader,
                _ATOMIC_FACTUAL_CONTRACTION_STATE,
                (
                    had_instance_method,
                    original_instance_method,
                    original,
                ),
            )
            setattr(
                reader,
                _ATOMIC_FACTUAL_CONTRACTION_METHOD,
                atomic_factual_rgb_detail_contraction,
            )
            return 1
        if state is None:
            return 0
        if not isinstance(state, tuple) or len(state) != 3:
            raise TypeError("reserved atomic factual-contraction state is invalid")
        had_instance_method, original_instance_method, _original = state
        if had_instance_method:
            setattr(
                reader,
                _ATOMIC_FACTUAL_CONTRACTION_METHOD,
                original_instance_method,
            )
        elif _ATOMIC_FACTUAL_CONTRACTION_METHOD in reader.__dict__:
            delattr(reader, _ATOMIC_FACTUAL_CONTRACTION_METHOD)
        delattr(reader, _ATOMIC_FACTUAL_CONTRACTION_STATE)
        return 1

    def set_cudnn_benchmark_scopes(
        self,
        engine: Any,
        *,
        enabled: bool,
        scopes: tuple[str, ...],
    ) -> int:
        """Select fixed-shape cuDNN algorithms only inside named visual owners.

        cuDNN benchmark may choose a different convolution reduction order, so
        each exact scope set remains a numerical candidate until Pen and RDT
        cross-process traces pass.  The wrapper adds no parameters, buffers or
        checkpoint keys and is resolved here rather than in the shared CUDA
        Graph backend, allowing another model layout to own a different path
        map behind the same hook.
        """

        requested = tuple(dict.fromkeys(str(scope) for scope in scopes))
        known = {
            "flow",
            "flow_encoder",
            "flow_context",
            "flow_update",
            "flow_delta",
            "flow_initial",
            "raw_pyramid",
            "raw_mid",
            "raw_high",
            "raw_context",
        }
        unknown = set(requested).difference(known)
        if not requested or unknown:
            raise ValueError(
                "cuDNN benchmark scopes must be a non-empty subset of "
                f"{sorted(known)!r}; unknown={sorted(unknown)!r}"
            )
        observation = getattr(engine.model, "observation", None)
        compiler = getattr(observation, "compiler", None)
        encoder = getattr(compiler, "encoder", None)
        if encoder is None:
            encoder = getattr(observation, "encoder", None)
        raw_flow = getattr(encoder, "raw_flow", None)
        flow = getattr(encoder, "flow", None)
        modules = {
            "flow": flow,
            "flow_encoder": getattr(flow, "encoder", None),
            "flow_context": getattr(flow, "context", None),
            "flow_update": getattr(flow, "update", None),
            "flow_delta": getattr(flow, "delta_head", None),
            "flow_initial": getattr(flow, "initial_residual", None),
            "raw_pyramid": getattr(raw_flow, "pyramid", None),
            "raw_mid": getattr(raw_flow, "mid", None),
            "raw_high": getattr(raw_flow, "high", None),
            "raw_context": getattr(encoder, "early_masked_raw_context", None),
        }
        missing = tuple(scope for scope in requested if modules[scope] is None)
        if missing:
            raise ValueError(
                f"cuDNN benchmark scopes are absent from this model: {missing!r}"
            )
        return sum(
            int(_set_module_cudnn_benchmark(modules[scope], enabled=enabled))
            for scope in requested
        )

    def set_cudnn_default_scopes(
        self,
        engine: Any,
        *,
        enabled: bool,
        scopes: tuple[str, ...],
    ) -> int:
        """Keep named visual owners on default forward algorithms."""

        requested = tuple(dict.fromkeys(str(scope) for scope in scopes))
        known = {
            "flow",
            "flow_encoder",
            "flow_context",
            "flow_update",
            "flow_delta",
            "flow_initial",
            "raw_pyramid",
            "raw_mid",
            "raw_high",
            "raw_context",
        }
        unknown = set(requested).difference(known)
        if not requested or unknown:
            raise ValueError(
                "cuDNN default scopes must be a non-empty subset of "
                f"{sorted(known)!r}; unknown={sorted(unknown)!r}"
            )
        observation = getattr(engine.model, "observation", None)
        compiler = getattr(observation, "compiler", None)
        encoder = getattr(compiler, "encoder", None)
        if encoder is None:
            encoder = getattr(observation, "encoder", None)
        raw_flow = getattr(encoder, "raw_flow", None)
        flow = getattr(encoder, "flow", None)
        modules = {
            "flow": flow,
            "flow_encoder": getattr(flow, "encoder", None),
            "flow_context": getattr(flow, "context", None),
            "flow_update": getattr(flow, "update", None),
            "flow_delta": getattr(flow, "delta_head", None),
            "flow_initial": getattr(flow, "initial_residual", None),
            "raw_pyramid": getattr(raw_flow, "pyramid", None),
            "raw_mid": getattr(raw_flow, "mid", None),
            "raw_high": getattr(raw_flow, "high", None),
            "raw_context": getattr(encoder, "early_masked_raw_context", None),
        }
        missing = tuple(scope for scope in requested if modules[scope] is None)
        if missing:
            raise ValueError(
                f"cuDNN default scopes are absent from this model: {missing!r}"
            )
        configured = 0
        for scope in requested:
            convolutions = tuple(
                module
                for module in modules[scope].modules()
                if isinstance(module, _CUDNN_CONVOLUTION_TYPES)
            )
            if not convolutions:
                raise ValueError(
                    f"cuDNN default scope {scope!r} contains no convolutions"
                )
            changed: list[Any] = []
            try:
                for convolution in convolutions:
                    if _set_module_cudnn_default_guard(
                        convolution,
                        enabled=enabled,
                    ):
                        changed.append(convolution)
            except Exception:
                for convolution in reversed(changed):
                    _set_module_cudnn_default_guard(
                        convolution,
                        enabled=not enabled,
                    )
                raise
            if changed and len(changed) != len(convolutions):
                for convolution in reversed(changed):
                    _set_module_cudnn_default_guard(
                        convolution,
                        enabled=not enabled,
                    )
                raise RuntimeError(
                    f"cuDNN default scope {scope!r} was only partially configured"
                )
            configured += int(bool(changed))
        return configured

    def prewarm_cudnn_default_scopes(
        self,
        engine: Any,
        batch: Any,
        *,
        scopes: tuple[str, ...],
    ) -> int:
        """Seed selected shape plans before enabling global cuDNN search.

        The warm-up performs no optimizer step and restores every owned RNG
        byte.  Autocast caching is disabled so temporary weight casts cannot
        cross into the formal forward.  This currently supports the raw RGB
        pyramid because its concrete input is a typed online-batch boundary;
        other implementations may expose their own adapter-owned scopes.
        """

        requested = tuple(dict.fromkeys(str(scope) for scope in scopes))
        known = {"raw_pyramid"}
        unknown = set(requested).difference(known)
        if not requested or unknown:
            raise ValueError(
                "default cuDNN prewarm scopes must be a non-empty subset of "
                f"{sorted(known)!r}; unknown={sorted(unknown)!r}"
            )
        if engine.device.type != "cuda":
            raise ValueError("default cuDNN prewarm requires a CUDA engine")
        if torch.backends.cudnn.benchmark:
            raise RuntimeError(
                "default cuDNN plans must be prewarmed before global benchmark is enabled"
            )
        observation = getattr(engine.model, "observation", None)
        compiler = getattr(observation, "compiler", None)
        encoder = getattr(compiler, "encoder", None)
        if encoder is None:
            encoder = getattr(observation, "encoder", None)
        pyramid = getattr(getattr(encoder, "raw_flow", None), "pyramid", None)
        if pyramid is None:
            raise ValueError("raw_pyramid is absent from this model")
        if any(parameter.grad is not None for parameter in engine.model.parameters()):
            raise RuntimeError("default cuDNN prewarm requires an empty gradient boundary")

        raw = batch.online.observation.raw_rgb
        if raw.device.type != "cuda" or (
            engine.device.index is not None
            and raw.device.index != engine.device.index
        ):
            raise ValueError("default cuDNN prewarm batch is on the wrong device")
        flat = raw.reshape(-1, *raw.shape[-3:])
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(engine.device).clone()
        owned_generators = tuple(
            generator
            for generator in (
                getattr(engine, "train_flow_generator", None),
                getattr(engine, "train_condition_generator", None),
            )
            if generator is not None
        )
        owned_rng = tuple(generator.get_state().clone() for generator in owned_generators)
        outputs: tuple[torch.Tensor, ...] | None = None
        engine.optimizer.zero_grad(set_to_none=True)
        try:
            with torch.backends.cudnn.flags(
                enabled=bool(torch.backends.cudnn.enabled),
                benchmark=False,
                deterministic=bool(torch.backends.cudnn.deterministic),
                allow_tf32=bool(torch.backends.cudnn.allow_tf32),
            ), torch.autocast(
                device_type=engine.device.type,
                dtype=engine.dtype,
                enabled=engine.dtype in {torch.bfloat16, torch.float16},
                cache_enabled=False,
            ):
                pyramid_outputs = pyramid(flat)
                outputs = (
                    (pyramid_outputs,)
                    if isinstance(pyramid_outputs, torch.Tensor)
                    else tuple(pyramid_outputs)
                )
                if not outputs or not all(
                    isinstance(value, torch.Tensor) for value in outputs
                ):
                    raise TypeError(
                        "default cuDNN prewarm scope must return one or more tensors"
                    )
            torch.autograd.backward(
                outputs,
                grad_tensors=tuple(torch.ones_like(value) for value in outputs),
            )
            torch.cuda.synchronize(engine.device)
        finally:
            engine.optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, engine.device)
            for generator, state in zip(
                owned_generators,
                owned_rng,
                strict=True,
            ):
                generator.set_state(state)
        del outputs
        torch.cuda.empty_cache()
        changed = self.set_cudnn_default_scopes(
            engine,
            enabled=True,
            scopes=requested,
        )
        if changed != len(requested):
            raise RuntimeError("default cuDNN guard was already installed")
        return len(requested)

    def set_activation_checkpointing(
        self,
        engine: Any,
        *,
        enabled: bool,
        scopes: tuple[str, ...] = ("raw_flow", "p1"),
    ) -> int:
        """Toggle only deterministic activation-recompute wrappers.

        The V120 raw-flow pyramid/refiners and late P1 reader expose their
        checkpoint policy as plain execution attributes.  Turning those
        wrappers off retains the exact operators and tensor arithmetic while
        trading memory for the recomputation work in backward.  The hook is
        intentionally version-owned: branches with different checkpoint
        surfaces can implement the same optional contract without changing
        the shared graph backend or checkpoint ABI.
        """

        requested_scopes = frozenset(str(scope) for scope in scopes)
        raw_scope_modules = {
            "raw_pyramid": "pyramid",
            "raw_mid": "mid",
            "raw_high": "high",
            "raw_context": "early_masked_raw_context",
        }
        known_scopes = {"raw_flow", "p1", *raw_scope_modules}
        unknown_scopes = requested_scopes.difference(known_scopes)
        if unknown_scopes:
            raise ValueError(f"unknown activation-checkpoint scopes: {sorted(unknown_scopes)!r}")
        changed = 0
        observation = getattr(engine.model, "observation", None)
        compiler = getattr(observation, "compiler", None)
        encoder = getattr(compiler, "encoder", None)
        if encoder is None:
            encoder = getattr(observation, "encoder", None)
        raw_flow = getattr(encoder, "raw_flow", None)
        selected_raw_scopes = (
            frozenset(raw_scope_modules)
            if "raw_flow" in requested_scopes
            else requested_scopes.intersection(raw_scope_modules)
        )
        if selected_raw_scopes and raw_flow is not None:
            for scope in sorted(selected_raw_scopes):
                module = getattr(raw_flow, raw_scope_modules[scope], None)
                if module is None or not hasattr(module, "activation_checkpoint"):
                    continue
                module.activation_checkpoint = bool(enabled)
                changed += 1
        factual_reader = getattr(engine.model, "factual_reader", None)
        if factual_reader is None:
            p1 = getattr(engine.model, "p1", None)
            factual_reader = getattr(p1, "factual_reader", None)
        if factual_reader is None:
            bottom = getattr(engine.model, "execution_bottom", None)
            factual_reader = getattr(bottom, "factual_reader", None)
        if "p1" in requested_scopes and factual_reader is not None and hasattr(
            factual_reader, "raw_activation_checkpoint"
        ):
            factual_reader.raw_activation_checkpoint = bool(enabled)
            changed += 1
        return changed

    def set_selective_checkpoint_save_operations(
        self,
        engine: Any,
        *,
        scope: str,
        operations: tuple[str, ...],
    ) -> int:
        """Cache selected expensive ops inside one raw-flow checkpoint.

        This retains the checkpoint boundary and all original operators.  It
        only changes which forward activations PyTorch saves for backward,
        giving each implementation an adapter-owned speed/memory policy.
        """

        module_names = {
            "raw_mid": "mid",
            "raw_high": "high",
        }
        if scope not in module_names:
            raise ValueError(f"unsupported selective checkpoint scope: {scope!r}")
        requested = tuple(dict.fromkeys(str(name) for name in operations))
        supported = {"convolution", "linear", "grid_sample"}
        unknown = set(requested).difference(supported)
        if not requested or unknown:
            raise ValueError(
                "selective checkpoint operations must be a non-empty subset of "
                f"{sorted(supported)!r}; unknown={sorted(unknown)!r}"
            )
        observation = getattr(engine.model, "observation", None)
        compiler = getattr(observation, "compiler", None)
        encoder = getattr(compiler, "encoder", None)
        if encoder is None:
            encoder = getattr(observation, "encoder", None)
        raw_flow = getattr(encoder, "raw_flow", None)
        module = getattr(raw_flow, module_names[scope], None)
        if module is None or not hasattr(module, "checkpoint_save_operations"):
            return 0
        if not bool(getattr(module, "activation_checkpoint", False)):
            raise ValueError(
                f"selective checkpoint scope {scope!r} is disabled, so it cannot "
                "own saved operations"
            )
        module.checkpoint_save_operations = requested
        return 1

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

        The ordinary profile grammar is
        ``{fast|precision|autotune|partitioned}-`` followed by one
        or more
        ``+``-joined semantic families.  Stable aliases include ``visual``,
        ``core``, ``combined`` and ``expanded``.  For example,
        ``precision-mainline+intent`` adds S/coarse-action to the established
        mainline plan without changing any other family.  ``autotune``
        retains every precision option from ``precision`` while searching
        equivalent kernel schedules.  ``flow`` and
        ``raw`` isolate the two visual compiler owners; ``visual`` selects
        both.  ``core`` combines only the mainline and MMDiT families so
        visual-shape recompilation remains independently attributable.  Every
        plan remains a candidate until the Pen and RDT gates promote that
        exact fingerprint.  The explicitly named ``hybrid-mainline-flow`` and
        ``hybrid-core-flow`` profiles use partitioned visual-flow leaves plus
        precision mainline blocks, with the latter also adding precision
        MMDiT blocks.  ``hybrid-mainline-mmdit`` instead keeps known MMDiT
        reductions eager while compiling the precision mainline family.
        ``factual-microgrid`` targets only the factual reader's configured
        high-resolution contraction method; its parent route/fine softmax and
        all other factual behavior remain eager.
        ``factual-microgrid-values`` is narrower still: micro-basis
        normalization and the FP32 coordinate read remain eager, and only the
        already-normalized BF16 RGB/detail contraction is compiled.
        ``factual-shell-eager-values`` and
        ``factual-shell-eager-microgrid`` invert that experiment: the complete
        factual reader stays one compiled shell while a method-level eager
        island preserves either the BF16 value contraction or the complete
        micro-grid expectation.  This tests whether cross-operation fusion can
        retain the full-reader speed without changing the sensitive reduction
        order.
        ``factual-shell-safe`` is the portable spelling: it prefers the narrow
        value-contraction boundary when that version exposes it and otherwise
        falls back to the complete micro-grid expectation boundary.
        ``factual-atomic`` compiles the same complete factual shell after an
        adapter-installed opaque value contraction has fixed the eager einsum's
        BF16 matrix and backward order.  It fails closed unless that opt-in
        hook is already installed on the model instance.
        The ``mainline-*`` subfamilies expose the existing mainline targets as
        portable diagnostic/selection units.  Their ordered union is exactly
        ``mainline``; selecting them never changes the established mainline
        fingerprint or any model implementation.
        ``mainline-grounding-safe`` compiles the same grounding blocks but
        retains their version-owned centered-normalization reduction method
        in eager form.  Versions that do not expose that narrow method fail
        closed instead of silently compiling a different numerical contract.
        """

        try:
            strategy, scope = profile.split("-", 1)
        except ValueError as error:
            raise ValueError(f"invalid selective compile profile: {profile!r}") from error
        if strategy not in _PROFILE_STRATEGIES:
            raise ValueError(
                f"unknown selective compile strategy {strategy!r}; "
                f"supported={tuple(sorted(_PROFILE_STRATEGIES))!r}"
            )
        composable_hybrid = scope.startswith("mainline+")
        if strategy == "hybrid" and not (
            scope in _HYBRID_SCOPES or composable_hybrid
        ):
            raise ValueError(
                "hybrid selective compile profiles must be a named legacy "
                "hybrid or start with 'hybrid-mainline+'; "
                f"received={profile!r}"
            )
        if strategy != "hybrid" and scope in _HYBRID_SCOPES:
            raise ValueError(
                f"scope {scope!r} is reserved for the hybrid strategy"
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

        conditioning_paths = tuple(
            path
            for path in (
                _first_existing(
                    names,
                    "conditioning.history_proposal",
                    "history_proposal",
                ),
            )
            if path is not None
        )
        bridge_paths = tuple(
            path
            for path in (
                _first_existing(
                    names,
                    "bridge.query_encoder",
                    "bottom.query_encoder",
                ),
            )
            if path is not None
        )
        intent_paths = tuple(
            path
            for path in (
                _first_existing(names, "intent.organizer", "top.intent"),
                _first_existing(names, "intent.coarse_action", "top.coarse_action"),
            )
            if path is not None
        )
        factual_paths = tuple(
            path
            for path in (
                _first_existing(
                    names,
                    "p1.factual_reader",
                    "factual_reader",
                    "bottom.factual_reader",
                ),
            )
            if path is not None
        )
        factual_microgrid_paths: tuple[str, ...] = ()
        factual_microgrid_value_paths: tuple[str, ...] = ()
        factual_shell_safe_paths: tuple[str, ...] = ()
        factual_atomic_paths: tuple[str, ...] = ()
        factual_shell_safe_method: str | None = None
        factual_shell_safe_reason: str | None = None
        factual_refiner_paths: tuple[str, ...] = ()
        if factual_paths:
            if callable(
                getattr(
                    modules[factual_paths[0]],
                    "_configured_typed_microgrid_expectation",
                    None,
                )
            ):
                factual_microgrid_paths = factual_paths
            if callable(
                getattr(
                    modules[factual_paths[0]],
                    "_configured_typed_microgrid_rgb_detail_contraction",
                    None,
                )
            ):
                factual_microgrid_value_paths = factual_paths
            if factual_microgrid_value_paths:
                factual_shell_safe_paths = factual_paths
                factual_shell_safe_method = (
                    "_configured_typed_microgrid_rgb_detail_contraction"
                )
                factual_shell_safe_reason = (
                    "preserve-eager-normalized-rgb-detail-contraction"
                )
            elif factual_microgrid_paths:
                factual_shell_safe_paths = factual_paths
                factual_shell_safe_method = (
                    "_configured_typed_microgrid_expectation"
                )
                factual_shell_safe_reason = (
                    "preserve-eager-complete-microgrid-expectation"
                )
            if getattr(
                modules[factual_paths[0]],
                _ATOMIC_FACTUAL_CONTRACTION_STATE,
                None,
            ) is not None:
                factual_atomic_paths = factual_paths
            refiner_root = f"{factual_paths[0]}.typed_local_refiners"
            if refiner_root in names:
                factual_refiner_paths = _direct_children(names, refiner_root)
        policy_paths = tuple(
            path
            for path in (
                _first_existing(
                    names,
                    "policy_compiler.effect_reader",
                    "top.effect_reader",
                ),
                _first_existing(
                    names,
                    "policy_compiler.consequence",
                    "top.consequence",
                ),
                _first_existing(
                    names,
                    "policy_compiler.plan_compiler",
                    "top.plan_compiler",
                ),
            )
            if path is not None
        )

        grounding_paths = (
            ()
            if grounding_path is None
            else _direct_children(names, grounding_path)
        )
        grounding_safe_paths = (
            grounding_paths
            if grounding_paths
            and all(
                callable(
                    getattr(
                        modules[path],
                        "_variance_floored_centered_norm",
                        None,
                    )
                )
                for path in grounding_paths
            )
            else ()
        )
        p1_paths = () if p1_path is None else (p1_path,)
        world_paths = (
            ()
            if world_prefix is None
            else tuple(
                path
                for path in (f"{world_prefix}.w1", f"{world_prefix}.w2")
                if path in names
            )
        )
        transition_paths = (
            ("transition.v120_transition",)
            if "transition.v120_transition" in names
            else ()
        )
        layer_head_paths = (
            ()
            if layer_heads_path is None
            else _direct_children(names, layer_heads_path)
        )
        mainline_paths = (
            *grounding_paths,
            *p1_paths,
            *world_paths,
            *transition_paths,
            *layer_head_paths,
        )

        mmdit_paths = _direct_children(names, f"{decoder_path}.blocks")
        execution_candidates = (
            f"{decoder_path}.evidence_adapter",
            f"{decoder_path}.organizer",
            *_direct_children(names, f"{decoder_path}.operator_contractions"),
            f"{decoder_path}.execution_controller",
        )
        execution_paths = tuple(
            path for path in execution_candidates if path in names
        )
        flow_paths: tuple[str, ...] = ()
        raw_paths: tuple[str, ...] = ()
        if observation_path is not None:
            if strategy in {"partitioned", "hybrid"}:
                flow_candidates = (
                    f"{observation_path}.flow.encoder",
                    f"{observation_path}.flow.context",
                    f"{observation_path}.flow.update",
                    f"{observation_path}.flow.delta_head",
                    f"{observation_path}.flow.initial_residual",
                )
                raw_candidates = (
                    f"{observation_path}.raw_flow.pyramid",
                    f"{observation_path}.raw_flow.mid.feature",
                    f"{observation_path}.raw_flow.mid.update",
                    f"{observation_path}.raw_flow.high.feature",
                    f"{observation_path}.raw_flow.high.update",
                )
            else:
                flow_candidates = (f"{observation_path}.flow",)
                raw_candidates = (f"{observation_path}.raw_flow",)
            flow_paths = tuple(path for path in flow_candidates if path in names)
            raw_paths = tuple(path for path in raw_candidates if path in names)
        visual_paths = (*flow_paths, *raw_paths)

        family_paths = {
            "conditioning": conditioning_paths,
            "bridge": bridge_paths,
            "intent": intent_paths,
            "factual": factual_paths,
            # Compile one version-owned method rather than the complete
            # factual reader. Parent posterior softmax/reductions retain their
            # eager call order while Inductor sees the N=49 x 3x3 contraction.
            "factual-microgrid": factual_microgrid_paths,
            "factual-microgrid-values": factual_microgrid_value_paths,
            # Compile the complete factual reader but install one explicit
            # eager method boundary below it.  These two candidates share the
            # same module path while owning distinct plan fingerprints and
            # numerical policies.
            "factual-shell-eager-values": factual_paths,
            "factual-shell-eager-microgrid": factual_paths,
            "factual-shell-safe": factual_shell_safe_paths,
            "factual-atomic": factual_atomic_paths,
            # Keep the parent P1 posterior and its functional softmax/
            # reductions eager.  These local readers are the largest stable
            # child boundary shared by the Pen and RDT layouts, so a
            # partitioned profile can fuse their tensor work without letting
            # Inductor reassociate the global factual posterior.
            "factual-refiners": factual_refiner_paths,
            "policy": policy_paths,
            "execution": execution_paths,
            "flow": flow_paths,
            "raw": raw_paths,
            "visual": visual_paths,
            "mainline": mainline_paths,
            "mainline-grounding": grounding_paths,
            "mainline-grounding-safe": grounding_safe_paths,
            "mainline-p1": p1_paths,
            "mainline-world": world_paths,
            "mainline-transition": transition_paths,
            "mainline-layer-heads": layer_head_paths,
            "mmdit": mmdit_paths,
        }
        family_methods = {
            "factual-microgrid": "_configured_typed_microgrid_expectation",
            "factual-microgrid-values": (
                "_configured_typed_microgrid_rgb_detail_contraction"
            ),
        }
        family_eager_methods = {
            "mainline-grounding-safe": (
                "_variance_floored_centered_norm",
                "preserve-eager-centered-normalization-reduction",
            ),
            "factual-shell-eager-values": (
                "_configured_typed_microgrid_rgb_detail_contraction",
                "preserve-eager-normalized-rgb-detail-contraction",
            ),
            "factual-shell-eager-microgrid": (
                "_configured_typed_microgrid_expectation",
                "preserve-eager-complete-microgrid-expectation",
            ),
        }
        if (
            factual_shell_safe_method is not None
            and factual_shell_safe_reason is not None
        ):
            family_eager_methods["factual-shell-safe"] = (
                factual_shell_safe_method,
                factual_shell_safe_reason,
            )
        if strategy == "hybrid":
            if scope == "mainline-flow":
                selected_families = ("flow", "mainline")
                partitioned_families = ("flow",)
            elif scope == "core-flow":
                selected_families = ("flow", "mainline", "mmdit")
                partitioned_families = ("flow",)
            elif scope == "mainline-mmdit":
                selected_families = ("mainline", "mmdit")
                partitioned_families = ("mmdit",)
            else:
                selected_families = _resolve_profile_families(scope)
                if selected_families[0] != "mainline" or len(selected_families) < 2:
                    raise ValueError(
                        "composable hybrid profiles require mainline followed "
                        "by at least one partitioned family"
                    )
                partitioned_families = tuple(
                    family for family in selected_families if family != "mainline"
                )
        else:
            selected_families = _resolve_profile_families(scope)
        if strategy == "fast":
            options = ()
        elif strategy == "autotune":
            options = _AUTOTUNED_NUMERICS_OPTIONS
        else:
            options = _NUMERICS_OPTIONS
        mode = "default" if strategy == "fast" else None
        regions = tuple(
            TrainingCompileRegion(
                name=family,
                module_paths=family_paths[family],
                method_name=family_methods.get(family, "forward"),
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
        if strategy == "partitioned":
            generic_boundaries = _numerical_boundaries(
                model,
                tuple(
                    path
                    for family in selected_families
                    if family not in family_eager_methods
                    if family_methods.get(family, "forward") == "forward"
                    for path in family_paths[family]
                ),
            )
        elif strategy == "hybrid":
            generic_boundaries = _numerical_boundaries(
                model,
                tuple(
                    path
                    for family in partitioned_families
                    if family not in family_eager_methods
                    if family_methods.get(family, "forward") == "forward"
                    for path in family_paths[family]
                ),
            )
        else:
            generic_boundaries = ()
        explicit_boundaries = tuple(
            TrainingEagerBoundary(
                module_path=path,
                method_name=family_eager_methods[family][0],
                reason=family_eager_methods[family][1],
            )
            for family in selected_families
            if family in family_eager_methods
            for path in family_paths[family]
        )
        boundaries = (*generic_boundaries, *explicit_boundaries)
        layout = "modular" if decoder_path.startswith("execution_bottom") else "legacy"
        return TrainingCompilePlan(
            name=f"clearvla-{layout}-{profile}-v1",
            regions=regions,
            eager_boundaries=boundaries,
            acceptance="candidate",
            rng_policy=(
                "eager-boundary-or-explicit-input"
                if boundaries
                else "inductor-fallback-random"
            ),
            numerical_policy=(
                (
                    "partition-"
                    + "+".join(partitioned_families)
                    + "-and-force-precision-mainline"
                )
                if strategy == "hybrid"
                else (
                    "partition-known-reductions-and-force-precision"
                    if strategy == "partitioned"
                    else (
                        "force-precision-with-autotuned-kernel-schedules"
                        if strategy == "autotune"
                        else (
                            "force-precision-with-inductor-reduction-order"
                            if strategy == "precision"
                            else "unrestricted-inductor-candidate"
                        )
                    )
                )
            ),
            disable_aot_autograd_buffer_donation=(
                scope in {"core", "core-flow", "mainline-mmdit"}
                or ("+" in scope and "mmdit" in selected_families)
            ),
        )


__all__ = ["MainlineTrainingAccelerationAdapter"]
