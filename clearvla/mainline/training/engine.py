"""Single-stage training engine for the capability-named mainline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import torch
from torch import Tensor

from ..config import ExperimentConfig
from ..interfaces import TrainingBatch
from ..model.action_codec import anchor_horizon_weights
from ..model.policy import (
    ClearVLAMainlinePolicy,
    OnlinePolicyCache,
    OnlineTrainingState,
)
from ..model.types import PhysicalActionCondition
from ..runtime.logging import tensor_scalars
from ..runtime.numerics import resolve_compute_dtype
from .gradient_audit import (
    DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD,
    FiniteGradientSpikeReport,
    build_finite_gradient_spike_report,
)
from .losses import LossLedger, compose_losses, sample_flow_matching
from .optimizer import WarmupCosineSchedule, gradient_diagnostics, parameter_role

_R2_PARAMETER_GRADIENT_METRICS: tuple[tuple[str, str], ...] = (
    (
        "top.dynamics.transport_head.weight",
        "gradient_parameter_w_transport_head_weight_rms",
    ),
    (
        "top.effect_reader.source_query.0.weight",
        "gradient_parameter_p2_semantic_spatial_query_weight_rms",
    ),
    (
        "top.effect_reader.source_query.1.weight",
        "gradient_parameter_p2_geometry_spatial_query_weight_rms",
    ),
    (
        "top.effect_reader.terminal_query.0.weight",
        "gradient_parameter_p2_semantic_terminal_query_weight_rms",
    ),
    (
        "top.effect_reader.terminal_query.1.weight",
        "gradient_parameter_p2_geometry_terminal_query_weight_rms",
    ),
    (
        "top.effect_reader.semantic_value.weight",
        "gradient_parameter_p2_semantic_value_weight_rms",
    ),
    (
        "top.effect_reader.transport_value.weight",
        "gradient_parameter_p2_geometry_value_weight_rms",
    ),
    (
        "top.consequence.semantic_interaction.weight",
        "gradient_parameter_consequence_semantic_interaction_weight_rms",
    ),
    (
        "top.consequence.geometry_interaction.weight",
        "gradient_parameter_consequence_geometry_interaction_weight_rms",
    ),
    (
        "bottom.decoder.velocity_head.gripper_gate.weight",
        "gradient_parameter_gripper_private_gate_weight_rms",
    ),
)


def _autocast(
    device: torch.device,
    dtype: torch.dtype,
    *,
    cache_enabled: bool = True,
):
    enabled = device.type in {"cuda", "cpu"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=bool(cache_enabled),
    )


def validate_finite_training_batch(batch: TrainingBatch) -> None:
    """Expensive value audit for preflight only, never the hot path."""

    values = {
        "online.dino_history": batch.online.observation.dino_history,
        "online.raw": batch.online.observation.raw_rgb,
        "online.state": batch.online.history.state,
        "online.state_history": batch.online.history.state_history,
        "online.executed_history": batch.online.history.executed_action_history,
        "online.goal": batch.online.goal.tokens,
        "target.action": batch.action_target.normalized,
        "target.action_raw_units": batch.action_target.raw_units,
        "target.current_raw_units": batch.action_target.current_raw_units,
        "target.gripper_transition_boundary": (
            batch.action_target.gripper_transition_boundary
        ),
        "target.gripper_transition_boundary_raw_units": (
            batch.action_target.gripper_transition_boundary_raw_units
        ),
        "future.dino": batch.future.dino_supports,
        "future.action": batch.future.action_sequence,
        "future.state": batch.future.state_sequence,
    }
    invalid = [name for name, value in values.items() if not bool(torch.isfinite(value).all())]
    if invalid:
        raise ValueError(f"training batch contains non-finite tensors: {', '.join(invalid)}")
    expected_offsets = torch.arange(
        4,
        49,
        4,
        device=batch.future.offsets.device,
        dtype=torch.long,
    )[None].expand(batch.future.batch, -1)
    if not torch.equal(batch.future.offsets, expected_offsets):
        raise ValueError("future teacher offsets must be exactly 4,8,...,48")


@dataclass(frozen=True)
class TrainStepResult:
    loss: Tensor
    gradient_norm: Tensor
    learning_rate: float
    metrics: dict[str, Tensor]
    gradient_norm_scalar: float | None = None

    def materialize(self) -> dict[str, float]:
        """Synchronize scalar metrics only when the logger actually emits."""

        values = {
            "loss_total": self.loss,
            "gradient_global_preclip_l2": self.gradient_norm,
            **self.metrics,
        }
        result = tensor_scalars(values)
        result["learning_rate"] = float(self.learning_rate)
        return result


@dataclass(frozen=True)
class NonFiniteGradientReport:
    """JSON-safe identity of the first non-finite parameter gradient."""

    parameter_name: str
    parameter_role: str
    optimizer_group: str
    shape: tuple[int, ...]
    dtype: str
    finite_fraction: float
    finite_max_abs: float
    nan_count: int
    positive_inf_count: int
    negative_inf_count: int
    global_norm: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class NonFiniteGradientError(FloatingPointError):
    """Fail before clipping/optimizer mutation with a named owner report."""

    def __init__(self, report: NonFiniteGradientReport) -> None:
        self.report = report
        super().__init__(
            "non-finite gradient before clipping: "
            f"parameter={report.parameter_name} role={report.parameter_role} "
            f"optimizer_group={report.optimizer_group} "
            f"finite_fraction={report.finite_fraction:.6f} "
            f"finite_max_abs={report.finite_max_abs:.6g} "
            f"global_norm={report.global_norm}"
        )


@dataclass(frozen=True)
class EncodedTrainingBatch:
    """One current-only static graph shared by validation loss and sampling."""

    cache: OnlinePolicyCache
    training_state: OnlineTrainingState
    metrics: dict[str, Tensor]


class MainlineTrainingEngine:
    """Own one forward/backward/update without legacy runtime side effects."""

    def __init__(
        self,
        *,
        model: ClearVLAMainlinePolicy,
        config: ExperimentConfig,
        optimizer: torch.optim.Optimizer,
        schedule: WarmupCosineSchedule,
        device: torch.device,
        dtype: torch.dtype | None = None,
        train_flow_generator: torch.Generator | None = None,
        train_condition_generator: torch.Generator | None = None,
        gradient_spike_audit_threshold: float | None = (
            DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD
        ),
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.schedule = schedule
        self.device = device
        self.dtype = resolve_compute_dtype(config, dtype)
        self.train_flow_generator = train_flow_generator
        self.train_condition_generator = train_condition_generator
        if gradient_spike_audit_threshold is not None and (
            not math.isfinite(float(gradient_spike_audit_threshold))
            or float(gradient_spike_audit_threshold) <= 0.0
        ):
            raise ValueError("gradient spike audit threshold must be finite and positive")
        self.gradient_spike_audit_threshold = (
            None
            if gradient_spike_audit_threshold is None
            else float(gradient_spike_audit_threshold)
        )
        self.global_step = 0

    def _optimizer_group_name(self, parameter_name: str) -> str:
        for group in self.optimizer.param_groups:
            raw_names = group.get("parameter_names", ())
            if parameter_name in raw_names:
                return str(group.get("name", "unnamed"))
        return "unowned"

    @staticmethod
    def _nonfinite_scalar_label(value: Tensor) -> str:
        scalar = value.detach().float().reshape(())
        if bool(torch.isnan(scalar)):
            return "nan"
        if bool(torch.isposinf(scalar)):
            return "+inf"
        if bool(torch.isneginf(scalar)):
            return "-inf"
        return f"{float(scalar):.9g}"

    def _first_nonfinite_gradient_report(
        self,
        *,
        global_norm: Tensor,
    ) -> NonFiniteGradientReport:
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            detached = gradient.detach()
            finite = torch.isfinite(detached)
            parameter_norm = torch.nn.utils.get_total_norm(
                [detached],
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=True,
            )
            # A parameter can contain only finite FP32 elements and still make
            # its L2 reduction overflow.  That tensor is the first actionable
            # owner and must not fall through to an unstructured RuntimeError.
            if bool(finite.all()) and bool(torch.isfinite(parameter_norm)):
                continue
            finite_count = int(finite.sum().item())
            finite_values = detached[finite]
            finite_max_abs = (
                float(finite_values.float().abs().amax().item())
                if finite_count
                else 0.0
            )
            return NonFiniteGradientReport(
                parameter_name=name,
                parameter_role=parameter_role(name),
                optimizer_group=self._optimizer_group_name(name),
                shape=tuple(int(value) for value in detached.shape),
                dtype=str(detached.dtype).removeprefix("torch."),
                finite_fraction=float(finite_count) / float(max(detached.numel(), 1)),
                finite_max_abs=finite_max_abs,
                nan_count=int(torch.isnan(detached).sum().item()),
                positive_inf_count=int(torch.isposinf(detached).sum().item()),
                negative_inf_count=int(torch.isneginf(detached).sum().item()),
                global_norm=self._nonfinite_scalar_label(global_norm),
            )
        raise RuntimeError("global gradient norm was non-finite without a named owner")

    def _r2_parameter_gradient_metrics(self) -> dict[str, Tensor]:
        """Read raw parameter gradients without installing persistent hooks.

        Intermediate activation hooks die with their forward graph. Parameter
        hooks do not: registering them on each diagnostic batch would retain
        every old scalar slot and make a long run progressively slower. These
        named R2 owners are therefore sampled once, after backward and
        before either clipping stage.
        """

        parameters = dict(self.model.named_parameters())
        metrics: dict[str, Tensor] = {}
        for parameter_name, metric_name in _R2_PARAMETER_GRADIENT_METRICS:
            try:
                parameter = parameters[parameter_name]
            except KeyError as error:
                raise RuntimeError(
                    f"R2 gradient diagnostic lost parameter {parameter_name!r}"
                ) from error
            gradient = parameter.grad
            metrics[metric_name] = (
                parameter.new_zeros((), dtype=torch.float32)
                if gradient is None
                else gradient.detach().float().square().mean().sqrt()
            )
        return metrics

    def _gradient_lifecycle(
        self,
        *,
        collect_diagnostics: bool,
        gradient_spike_handler: Callable[[FiniteGradientSpikeReport], None]
        | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], float]:
        """V120 finite-check -> decoder-local -> global clip lifecycle."""

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        gradients = [parameter.grad for parameter in parameters]
        try:
            total_norm = torch.nn.utils.get_total_norm(
                gradients,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as error:
            # Recompute only on the failure path so the report owns the
            # observed global value while normal batches pay for one norm.
            nonfinite_norm = torch.nn.utils.get_total_norm(
                gradients,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=True,
            )
            raise NonFiniteGradientError(
                self._first_nonfinite_gradient_report(global_norm=nonfinite_norm)
            ) from error
        # One host scalar owns both the spike boundary and the training-window
        # mean/max/current record.  The expensive parameter scan remains
        # conditional and happens before any clipping or optimizer mutation.
        gradient_norm_scalar = float(total_norm.detach().float().cpu().item())
        if (
            gradient_spike_handler is not None
            and self.gradient_spike_audit_threshold is not None
            and gradient_norm_scalar > self.gradient_spike_audit_threshold
        ):
            named_parameters = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            ]
            gradient_spike_handler(
                build_finite_gradient_spike_report(
                    named_parameters,
                    global_norm=gradient_norm_scalar,
                    audit_threshold=self.gradient_spike_audit_threshold,
                    optimizer_group_name=self._optimizer_group_name,
                )
            )
        metrics = (
            gradient_diagnostics(self.model, stage="raw")
            if collect_diagnostics
            else {}
        )
        if collect_diagnostics:
            metrics.update(self._r2_parameter_gradient_metrics())
        decoder_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name.startswith("bottom.decoder.") and parameter.grad is not None
        ]
        decoder_norm: Tensor | None = None
        if decoder_parameters:
            decoder_norm = torch.nn.utils.get_total_norm(
                [parameter.grad for parameter in decoder_parameters],
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=True,
            )
            torch.nn.utils.clip_grads_with_norm_(
                decoder_parameters,
                1.0,
                decoder_norm,
                foreach=True,
            )
        if collect_diagnostics:
            metrics.update(gradient_diagnostics(self.model, stage="postlocal"))
            metrics["gradient_raw_bottom_decoder_l2"] = (
                decoder_norm.detach().float()
                if decoder_norm is not None
                else total_norm.new_zeros((), dtype=torch.float32)
            )
            metrics["gradient_postlocal_bottom_decoder_l2"] = (
                torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in decoder_parameters],
                    norm_type=2.0,
                    error_if_nonfinite=True,
                    foreach=True,
                ).detach().float()
                if decoder_parameters
                else total_norm.new_zeros((), dtype=torch.float32)
            )
        postlocal_norm = torch.nn.utils.get_total_norm(
            [parameter.grad for parameter in parameters],
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=True,
        )
        torch.nn.utils.clip_grads_with_norm_(
            parameters,
            self.config.optimizer.grad_clip,
            postlocal_norm,
            foreach=True,
        )
        postglobal_norm = torch.nn.utils.get_total_norm(
            [parameter.grad for parameter in parameters],
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=True,
        )
        if collect_diagnostics:
            metrics.update(gradient_diagnostics(self.model, stage="postglobal"))
            metrics["gradient_raw_global_l2"] = total_norm.detach().float()
            metrics["gradient_postlocal_global_l2"] = postlocal_norm.detach().float()
            metrics["gradient_postglobal_global_l2"] = postglobal_norm.detach().float()
            metrics["gradient_postglobal_bottom_decoder_l2"] = (
                torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in decoder_parameters],
                    norm_type=2.0,
                    error_if_nonfinite=True,
                    foreach=True,
                ).detach().float()
                if decoder_parameters
                else total_norm.new_zeros((), dtype=torch.float32)
            )
        return total_norm, metrics, gradient_norm_scalar

    def _clip_gradients_with_first_offender(self) -> Tensor:
        """Compatibility wrapper used by the non-finite regression test."""

        total_norm, _, _ = self._gradient_lifecycle(collect_diagnostics=False)
        return total_norm

    @staticmethod
    def _tensor_metrics(
        ledger: LossLedger,
        values: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        tensors = {
            "loss_total": ledger.total,
            **{f"loss_group_{name}": value for name, value in ledger.groups.items()},
            **{
                f"loss_contrib_{name}": value
                for name, value in ledger.contributions.items()
            },
            **{f"loss_{name}": value for name, value in ledger.terms.items()},
            **values,
        }
        result: dict[str, Tensor] = {}
        for name, value in tensors.items():
            if value.ndim != 0:
                continue
            result[name] = value.detach().float()
        result["loss_ledger_gap"] = ledger.total.detach().float() - sum(
            value.detach().float() for value in ledger.groups.values()
        )
        result["loss_contribution_gap"] = ledger.total.detach().float() - sum(
            value.detach().float() for value in ledger.contributions.values()
        )
        return result

    @staticmethod
    def _detached_pearson(left: Tensor, right: Tensor) -> Tensor:
        """Batch-local audit correlation with a legal zero-variance result."""

        left_f = left.detach().float().flatten()
        right_f = right.detach().float().flatten()
        if left_f.numel() != right_f.numel():
            raise ValueError("audit correlation rows must align")
        if left_f.numel() < 2:
            return left_f.new_zeros(())
        left_centered = left_f - left_f.mean()
        right_centered = right_f - right_f.mean()
        denominator = (left_centered.square().sum() * right_centered.square().sum()).sqrt()
        correlation = (left_centered * right_centered).sum() / denominator.clamp_min(1e-8)
        return torch.where(denominator > 1e-8, correlation, correlation.new_zeros(()))

    @classmethod
    def _audit_progress_metrics(
        cls,
        batch: TrainingBatch,
        encoded: EncodedTrainingBatch,
        *,
        formal_cache: OnlinePolicyCache,
    ) -> dict[str, Tensor]:
        """Compare real frame position with S/W behaviour without training on it.

        The derived interval centroid is explicitly an energy audit, not a
        phase estimate and not a forward input.  It answers whether S keeps
        emitting one fixed interval pattern as trajectories advance.  Every
        value is detached and this function is called only on logging rows.
        """

        if batch.audit.frame_progress is None:
            return {}
        intent = encoded.training_state.top.intent
        # S remains the single static online/training owner.  W diagnostics
        # must instead read the cache that owns the formal pass-one action and
        # future losses; reading OnlineTrainingState here would silently audit
        # the retired coarse-conditioned W.
        dynamics = formal_cache.top.predicted_dynamics
        progress = batch.audit.frame_progress.to(
            device=intent.public_interval_carrier.device,
            dtype=torch.float32,
        )

        def sample_rms(value: Tensor) -> Tensor:
            return value.detach().float().flatten(1).square().mean(dim=1).sqrt()

        # Audit the supervised public carrier.  Optional typed values keep
        # their own interval/K/type metrics and must not silently redefine the
        # public progress diagnostic.
        interval_energy = (
            intent.public_interval_carrier.detach().float().square().mean(dim=-1).sqrt()
        )
        centers = interval_energy.new_tensor((6.0, 12.0, 24.0, 40.0)) / 48.0
        energy_total = interval_energy.sum(dim=1)
        centroid = (interval_energy * centers[None]).sum(dim=1) / energy_total.clamp_min(1e-8)
        centroid = torch.where(energy_total > 1e-8, centroid, centroid.new_zeros(centroid.shape))

        interval_variation = sample_rms(
            intent.public_interval_carrier.detach().float()
            - intent.public_interval_carrier.detach().float().mean(dim=1, keepdim=True)
        )
        state_change = sample_rms(intent.state_change_evidence)
        successor_innovation = sample_rms(
            dynamics.successor_content.detach().float()
            - dynamics.current_reference.detach().float()[:, None]
        )
        w_interval_variation = sample_rms(
            dynamics.semantic_delta.detach().float()
            - dynamics.semantic_delta.detach().float().mean(dim=1, keepdim=True)
        )
        return {
            "object_intent_audit_frame_progress_mean": progress.mean(),
            "object_intent_audit_frame_progress_std": progress.std(unbiased=False),
            "object_intent_audit_interval_energy_centroid": centroid.mean(),
            "object_intent_audit_interval_energy_centroid_std": centroid.std(unbiased=False),
            "object_intent_audit_interval_centroid_frame_absolute_gap": (centroid - progress)
            .abs()
            .mean(),
            "object_intent_audit_frame_progress_centroid_correlation": cls._detached_pearson(
                progress, centroid
            ),
            "object_intent_audit_frame_progress_interval_variation_correlation": (
                cls._detached_pearson(progress, interval_variation)
            ),
            "object_intent_audit_frame_progress_state_change_correlation": (
                cls._detached_pearson(progress, state_change)
            ),
            "object_w_audit_frame_progress_successor_correlation": cls._detached_pearson(
                progress, successor_innovation
            ),
            "object_w_audit_frame_progress_interval_variation_correlation": (
                cls._detached_pearson(progress, w_interval_variation)
            ),
        }

    def _forward(
        self,
        batch: TrainingBatch,
        *,
        training: bool,
        collect_diagnostics: bool,
        generator: torch.Generator | None,
        condition_generator: torch.Generator | None = None,
    ) -> tuple[LossLedger, dict[str, Tensor]]:
        batch.validate(self.config)
        cache, training_state, static_metrics = self.model.encode_online(
            batch.online,
            training_mask=training,
            collect_diagnostics=collect_diagnostics,
            condition_generator=condition_generator,
        )
        return self._forward_encoded(
            batch,
            encoded=EncodedTrainingBatch(cache, training_state, static_metrics),
            collect_diagnostics=collect_diagnostics,
            generator=generator,
        )

    def _forward_encoded(
        self,
        batch: TrainingBatch,
        *,
        encoded: EncodedTrainingBatch,
        collect_diagnostics: bool,
        generator: torch.Generator | None,
    ) -> tuple[LossLedger, dict[str, Tensor]]:
        encoded.cache.validate(self.config)
        encoded.training_state.validate(self.config)
        top_targets, teacher_metrics = self.model.build_training_targets(
            encoded.training_state,
            batch.future,
            collect_diagnostics=collect_diagnostics,
        )
        flow_state = sample_flow_matching(
            batch.action_target.normalized,
            action_state=batch.online.history.action_state,
            codec=self.model.action_codec,
            distribution=self.config.bottom.flow_time_distribution,
            generator=generator,
        )

        # Training and deployment must assign the final action objective to
        # the same action-conditioned W distribution.  Pass zero is only a
        # detached endpoint estimator.  Forking the global CPU/current-CUDA
        # streams lets it use the exact dropout stream that the formal pass
        # will see, then restores that stream before pass one.  The net global
        # RNG advance is therefore one dynamic pass, exactly as before.
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device()
                if self.device.index is None
                else int(self.device.index)
            )
        with torch.random.fork_rng(devices=cuda_devices):
            with torch.no_grad():
                # CUDA autocast caches lower-precision parameter copies for
                # the lifetime of the outer training context.  If a module's
                # first dynamic call happens under no-grad, a cached BF16
                # copy can lose its parameter edge and then be reused by the
                # formal pass.  Keep pass0 numerically identical but forbid
                # this detached estimator from publishing weight copies into
                # the surrounding pass1 cache.
                with _autocast(
                    self.device,
                    self.dtype,
                    cache_enabled=False,
                ):
                    pass0_output = self.model.velocity(
                        encoded.cache,
                        noisy_action_field=flow_state.noisy_physical,
                        time=flow_state.time,
                        require_execution_supervision=False,
                        collect_diagnostics=False,
                    )
                    remaining = (1.0 - flow_state.time.to(
                        dtype=flow_state.noisy_physical.dtype
                    ))[:, None, None]
                    pass0_clean_physical = flow_state.noisy_physical + remaining * (
                        pass0_output.bottom.physical_velocity.to(
                            dtype=flow_state.noisy_physical.dtype
                        )
                    )
                    pass0_clean_action = self.model.action_codec.decode(
                        pass0_clean_physical,
                        encoded.cache.history.action_state,
                    ).detach()
                    pass0_condition = PhysicalActionCondition.from_horizon_action(
                        pass0_clean_action,
                        encoded.cache.history.action_state.detach(),
                    )
                    pass0_action_flow = (
                        self._detached_v120_action_flow(
                            pass0_output.bottom.physical_velocity,
                            flow_state.target_physical_velocity,
                        )
                        if collect_diagnostics
                        else None
                    )
                    del pass0_output, pass0_clean_physical

        refined_top, refinement_metrics = self.model.top.refine_deployment_world(
            encoded.cache.top,
            action_condition=pass0_condition,
            collect_diagnostics=collect_diagnostics,
        )
        formal_cache = replace(encoded.cache, top=refined_top)
        formal_cache.validate(self.config)
        output = self.model.velocity(
            formal_cache,
            noisy_action_field=flow_state.noisy_physical,
            time=flow_state.time,
            # Execution-value regression is part of the loss on every train
            # and validation batch.  It owns non-scalar candidate tensors and
            # therefore cannot be coupled to the optional diagnostic-batch
            # budget used only for logging and interventions.
            require_execution_supervision=True,
            collect_diagnostics=collect_diagnostics,
        )
        ledger = compose_losses(
            self.config,
            policy_output=output,
            action_target=batch.action_target,
            history=batch.online.history,
            flow_state=flow_state,
            observation=encoded.training_state.observation,
            top_targets=top_targets,
            predicted_dynamics=formal_cache.top.predicted_dynamics,
            action_codec=self.model.action_codec,
            collect_diagnostics=collect_diagnostics,
        )
        self_conditioning_metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            if pass0_action_flow is None:
                raise RuntimeError("self-conditioning action-flow audit was not materialized")
            with torch.no_grad():
                remaining = (1.0 - flow_state.time.to(
                    dtype=flow_state.noisy_physical.dtype
                ))[:, None, None]
                pass1_clean_physical = flow_state.noisy_physical + remaining * (
                    output.bottom.physical_velocity.detach().to(
                        dtype=flow_state.noisy_physical.dtype
                    )
                )
                pass1_clean_action = self.model.action_codec.decode(
                    pass1_clean_physical,
                    formal_cache.history.action_state,
                )
                pass1_condition = PhysicalActionCondition.from_horizon_action(
                    pass1_clean_action,
                    formal_cache.history.action_state,
                )
                coarse_fingerprint = encoded.cache.top.action_condition.fingerprint.float()
                pass0_fingerprint = pass0_condition.fingerprint.float()
                self_conditioning_metrics = {
                    "training_self_conditioning_pass0_clean_action_rms": (
                        pass0_clean_action.float().square().mean().sqrt()
                    ),
                    "training_self_conditioning_coarse_to_pass0_condition_rms": (
                        pass0_fingerprint - coarse_fingerprint
                    )
                    .square()
                    .mean()
                    .sqrt(),
                    "training_self_conditioning_pass0_to_pass1_clean_action_delta_rms": (
                        pass1_clean_action.float() - pass0_clean_action.float()
                    )
                    .square()
                    .mean()
                    .sqrt(),
                    "training_self_conditioning_pass1_world_interval_mismatch_rms": (
                        pass1_condition.interval_action.float()
                        - pass0_condition.interval_action.float()
                    )
                    .square()
                    .mean()
                    .sqrt(),
                    "training_self_conditioning_pass1_world_delta_mismatch_rms": (
                        pass1_condition.interval_delta.float()
                        - pass0_condition.interval_delta.float()
                    )
                    .square()
                    .mean()
                    .sqrt(),
                    "training_self_conditioning_pass0_action_flow_audit": pass0_action_flow,
                    "training_self_conditioning_pass1_action_flow_minus_pass0": (
                        ledger.terms["action_flow"].detach().float()
                        - pass0_action_flow
                    ),
                }
        metrics = {
            **encoded.metrics,
            **teacher_metrics,
            **refinement_metrics,
            **output.metrics,
            **self_conditioning_metrics,
        }
        if collect_diagnostics:
            metrics.update(
                self._audit_progress_metrics(
                    batch,
                    encoded,
                    formal_cache=formal_cache,
                )
            )
        return ledger, metrics

    def _detached_v120_action_flow(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Reproduce the formal V120 physical flow scalar for pass-zero audit.

        This value is detached, diagnostic-only and deliberately excludes the
        event-balanced counterfactual.  Keeping the exact arm absolute/delta
        and gripper-field geometry makes its delta to the formal pass-one
        ``loss_action_flow`` interpretable without composing a second loss.
        """

        residual = prediction.detach().float() - target.detach().float()
        parts = self.model.action_codec.split(residual)
        arm_error = 0.5 * (
            parts.arm_absolute.square() + parts.arm_delta.square()
        )
        gripper_error = parts.gripper_field.square().mean(dim=-1)
        physical_error = (
            arm_error.sum(dim=-1) + gripper_error
        ) / float(self.model.action_codec.arm_dim + 1)
        horizon_weight = anchor_horizon_weights(
            horizon=self.config.dimensions.action_horizon,
            tail_emphasis=self.config.objectives.horizon_tail_emphasis,
            first_step_protection=self.config.objectives.horizon_first_step_protection,
            device=prediction.device,
        )
        return (physical_error * horizon_weight[None]).mean()

    @torch.no_grad()
    def encode_eval(
        self,
        batch: TrainingBatch,
        *,
        collect_diagnostics: bool,
    ) -> EncodedTrainingBatch:
        """Build the static validation graph once for loss and deployment."""

        batch.validate(self.config)
        self.model.eval()
        with _autocast(self.device, self.dtype):
            cache, training_state, metrics = self.model.encode_online(
                batch.online,
                training_mask=False,
                collect_diagnostics=collect_diagnostics,
            )
        return EncodedTrainingBatch(cache, training_state, metrics)

    def train_step(
        self,
        batch: TrainingBatch,
        *,
        collect_diagnostics: bool = False,
        gradient_spike_handler: Callable[[FiniteGradientSpikeReport], None]
        | None = None,
    ) -> TrainStepResult:
        self.model.train()
        # V120 intentionally keeps the execution controller and ordered
        # capacity bank out of the task graph for 200 steps, then opens them
        # continuously over 1000 steps.  Omitting this call left the recovered
        # mainline permanently at the warm-up identity boundary.
        self.model.set_training_step(self.global_step)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.dtype):
            ledger, metrics = self._forward(
                batch,
                training=True,
                collect_diagnostics=collect_diagnostics,
                generator=self.train_flow_generator,
                condition_generator=self.train_condition_generator,
            )
        ledger.total.backward()
        gradient_norm, gradient_metrics, gradient_norm_scalar = (
            self._gradient_lifecycle(
                collect_diagnostics=collect_diagnostics,
                gradient_spike_handler=gradient_spike_handler,
            )
        )
        metrics.update(gradient_metrics)
        # Record the LR that owns this update.  Advancing the scheduler first
        # and then logging the optimizer group reported the *next* batch's LR
        # beside the current batch loss, an off-by-one semantic error during
        # the entire warmup.
        # Optimizer groups intentionally have different V120-resolved scales.
        # The public ``learning_rate`` metric therefore owns the base schedule,
        # not whichever alphabetically sorted role happens to be group zero.
        learning_rate = float(
            self.config.optimizer.learning_rate
            * self.schedule.ratio(self.schedule.step_index)
        )
        self.optimizer.step()
        self.schedule.step()
        self.global_step += 1
        return TrainStepResult(
            loss=ledger.total.detach().float(),
            gradient_norm=gradient_norm.detach().float(),
            learning_rate=learning_rate,
            metrics=self._tensor_metrics(ledger, metrics),
            gradient_norm_scalar=gradient_norm_scalar,
        )

    @torch.no_grad()
    def eval_step(
        self,
        batch: TrainingBatch,
        *,
        collect_diagnostics: bool = True,
        generator: torch.Generator | None = None,
        encoded: EncodedTrainingBatch | None = None,
    ) -> TrainStepResult:
        self.model.eval()
        with _autocast(self.device, self.dtype):
            if encoded is None:
                ledger, metrics = self._forward(
                    batch,
                    training=False,
                    collect_diagnostics=collect_diagnostics,
                    generator=generator,
                    condition_generator=None,
                )
            else:
                ledger, metrics = self._forward_encoded(
                    batch,
                    encoded=encoded,
                    collect_diagnostics=collect_diagnostics,
                    generator=generator,
                )
        return TrainStepResult(
            loss=ledger.total.detach().float(),
            gradient_norm=ledger.total.new_zeros((), dtype=torch.float32),
            # Optimizer group zero is alphabetic, not the public/base role.
            # Report the same base schedule semantic as train_step instead of
            # whichever role-specific LR happens to sort first.
            learning_rate=float(
                self.config.optimizer.learning_rate
                * self.schedule.ratio(self.schedule.step_index)
            ),
            metrics=self._tensor_metrics(ledger, metrics),
            gradient_norm_scalar=0.0,
        )


__all__ = [
    "EncodedTrainingBatch",
    "MainlineTrainingEngine",
    "NonFiniteGradientError",
    "NonFiniteGradientReport",
    "TrainStepResult",
    "validate_finite_training_batch",
]
