"""Small, nested configuration for the active ClearVLA graph.

Fixed architectural facts such as the 3-2-3 topology, four intervals and one
protected bottom ingress live in the architecture manifest and module graph.
They are deliberately not represented as combinable boolean switches here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, TypeVar, cast

from .manifest import ARCHITECTURE_MANIFEST


@dataclass(frozen=True)
class DataConfig:
    raw_hdf5_root: str = "/data/liang.zhang/dataset/grab_pen_single/grab_pen_single"
    hdf5_glob: str = "*.hdf5"
    decoded_cache: str = "/data/senwang/data/cache_336"
    dino_cache: str = "/data/senwang/data/dinov2_cache_336"
    t5_condition: str = "/data/senwang/checkpoint/grasp_pen_embed.pt"
    output_dir: str = "runs/clearvla_mainline"
    camera_names: tuple[str, ...] = ("top", "wrist")
    action_key: str = "action"
    state_key: str = "qpos"
    top_camera_key: str = "observations/images/cam_high"
    wrist_camera_key: str = "observations/images/cam_right_wrist"
    cache_side: int = 336
    dinov2_model: str = "facebook/dinov2-base"
    split_mode: str = "ordered-counts"
    train_episodes: int = 63
    val_episodes: int = 5
    test_episodes: int = 5
    normalizer: str = "zscore"
    stride: int = 1
    num_workers: int = 4
    seed: int = 0
    information_uniform_fraction: float = 0.50
    information_event_fraction: float = 0.125
    information_motion_quantile: float = 0.70

    def validate(self) -> None:
        string_fields = (
            "raw_hdf5_root",
            "hdf5_glob",
            "decoded_cache",
            "dino_cache",
            "t5_condition",
            "output_dir",
            "action_key",
            "state_key",
            "top_camera_key",
            "wrist_camera_key",
            "dinov2_model",
            "split_mode",
            "normalizer",
        )
        for name in string_fields:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{name} must be a non-empty string")
        if self.hdf5_glob != "*.hdf5":
            raise ValueError("the formal dataset contract uses the flat *.hdf5 glob")
        if tuple(self.camera_names) != ("top", "wrist"):
            raise ValueError("the current observation contract uses top and wrist cameras")
        if self.cache_side != 336:
            raise ValueError("the established decoded and DINO caches use 336x336 preprocessing")
        if self.split_mode != "ordered-counts":
            raise ValueError("formal comparison runs require an ordered-count split")
        if (self.train_episodes, self.val_episodes, self.test_episodes) != (63, 5, 5):
            raise ValueError("formal comparison runs use the established 63/5/5 episode split")
        if self.normalizer != "zscore":
            raise ValueError("the active action/state chart uses z-score normalization")
        if min(self.stride, self.num_workers) < 0 or self.stride == 0:
            raise ValueError("data stride must be positive and worker count non-negative")
        if self.seed < 0:
            raise ValueError("data seed must be non-negative")
        if (
            self.information_uniform_fraction != 0.50
            or self.information_event_fraction != 0.125
            or self.information_motion_quantile != 0.70
        ):
            raise ValueError(
                "formal information-balanced sampling is fixed at "
                "uniform=0.50, event=0.125, motion_quantile=0.70"
            )


@dataclass(frozen=True)
class ModelDimensions:
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 24
    action_basis_tokens: int = 4
    visual_history_length: int = 3
    state_history_length: int = 3
    executed_history_length: int = 8
    hidden_size: int = 512
    num_heads: int = 8
    visual_token_dim: int = 768
    # The formal precomputed condition is T5-XXL hidden state [L,4096].
    # V122 resolves this width from the .pt file before model construction;
    # the independent graph records the same boundary explicitly.
    goal_token_dim: int = 4096
    goal_max_tokens: int = 32
    num_cameras: int = 2
    # ``dinov2_cache_336`` stores the backbone's 16x16 patch chart.  The
    # 336-pixel decoded image size is a preprocessing boundary, not evidence
    # for a 24x24 cached-token chart.
    patches_per_camera: int = 256
    future_supports: int = 12

    def validate(self) -> None:
        values = asdict(self)
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError("all model dimensions must be positive")
        if self.action_dim != self.state_dim:
            raise ValueError("the active action/state chart requires equal dimensions")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_horizon != 24:
            raise ValueError("the active bottom and interval contract requires horizon 24")
        if self.num_cameras != 2:
            raise ValueError("the current observation contract owns two cameras")
        if self.visual_history_length != 3:
            raise ValueError("the causal visual history requires offsets -8/-4/0")
        if self.state_history_length != 3:
            raise ValueError("the active state history owns three causal rows")
        if self.executed_history_length != 8:
            raise ValueError("the active executed-action history owns eight causal rows")
        if self.future_supports != 12:
            raise ValueError("the teacher contract requires supports at offsets 4..48")


@dataclass(frozen=True)
class ObservationConfig:
    grid_size: int = 8
    local_hypotheses: int = 4
    feature_dim: int = 96
    address_route_dim: int = 32
    flow_iterations: int = 3
    correlation_radius: int = 2
    flow_reference_frames: int = 4
    raw_base_channels: int = 32
    mask_ratio: float = 0.375
    mask_block_size: int = 2
    motion_mask_fraction: float = 0.60
    uncertainty_floor: float = 0.03
    microgrid_side: int = 3

    def validate(self) -> None:
        integer_fields = (
            self.grid_size,
            self.local_hypotheses,
            self.feature_dim,
            self.address_route_dim,
            self.flow_iterations,
            self.correlation_radius,
            self.flow_reference_frames,
            self.raw_base_channels,
            self.mask_block_size,
            self.microgrid_side,
        )
        if any(int(value) <= 0 for value in integer_fields):
            raise ValueError("observation dimensions and radii must be positive")
        if self.grid_size != 8 or self.local_hypotheses != 4:
            raise ValueError("the active dense chart is [C,8,8,M=4]")
        if self.flow_reference_frames != 4:
            raise ValueError("the learned flow contract uses a four-frame raw pair")
        if self.feature_dim % 8:
            raise ValueError("observation feature_dim must be divisible by eight")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0,1)")
        if not 0.0 <= self.motion_mask_fraction <= 1.0:
            raise ValueError("motion_mask_fraction must be in [0,1]")
        if self.uncertainty_floor <= 0.0:
            raise ValueError("uncertainty_floor must be positive")
        if self.microgrid_side != 3:
            raise ValueError("the active P1 contract owns four 3x3 factual glimpses")


@dataclass(frozen=True)
class TopConfig:
    object_slots: int = 4
    grounder_iterations: int = 3
    teacher_key_dim: int = 64
    role_host_depth: int = 3
    role_host_ffn_expansion: float = 4.0
    role_host_dropout: float = 0.05
    proposal_depth: int = 2
    proposal_recent_tokens: int = 4
    proposal_summary_tokens: int = 3
    goal_condition_dropout: float = 0.05
    action_history_condition_dropout: float = 0.10
    proposal_condition_dropout: float = 0.25

    def validate(self) -> None:
        if self.object_slots != ARCHITECTURE_MANIFEST.object_slots:
            raise ValueError("top object count must match the manifest")
        if (
            self.grounder_iterations <= 0
            or self.teacher_key_dim <= 0
            or self.role_host_depth != 3
            or self.proposal_depth != 2
            or self.proposal_recent_tokens != 4
            or self.proposal_summary_tokens != 3
            or self.role_host_ffn_expansion <= 0.0
        ):
            raise ValueError("top iteration/key dimensions must be positive")
        for name, value in (
            ("role_host_dropout", self.role_host_dropout),
            ("goal_condition_dropout", self.goal_condition_dropout),
            (
                "action_history_condition_dropout",
                self.action_history_condition_dropout,
            ),
            ("proposal_condition_dropout", self.proposal_condition_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"top {name} must be in [0,1)")


@dataclass(frozen=True)
class BottomConfig:
    flow_time_distribution: str = "beta_1_5_1"
    evidence_depth: int = 3
    latent_dim: int = 64
    ffn_expansion: float = 2.0
    dropout: float = 0.05
    residual_scale_max: float = 0.25
    residual_scale_init: float = 0.05
    normalization_floor: float = 0.25
    operator_rank: int = 32
    operator_groups: int = 32
    operator_depth_logit_init: float = 2.268683541
    controller_tokens: int = 8
    controller_depth: int = 2
    controller_heads: int = 8
    initial_exit_probability: float = 0.10
    controlled_delta_rank: int = 8
    controlled_action_tokens: int = 8
    controlled_delta_dropout: float = 0.0
    gripper_field_dim: int = 6
    physical_decode_delta_blend: float = 0.25

    def validate(self) -> None:
        if self.flow_time_distribution != "beta_1_5_1":
            raise ValueError("formal training uses beta_1_5_1 flow time")
        integer_fields = (
            self.evidence_depth,
            self.latent_dim,
            self.operator_rank,
            self.operator_groups,
            self.controller_tokens,
            self.controller_depth,
            self.controller_heads,
            self.controlled_delta_rank,
            self.controlled_action_tokens,
            self.gripper_field_dim,
        )
        if any(int(value) <= 0 for value in integer_fields):
            raise ValueError("bottom dimensions must be positive")
        if self.operator_rank % self.operator_groups:
            raise ValueError("operator rank must be divisible by groups")
        if (
            min(
                self.residual_scale_max,
                self.residual_scale_init,
                self.normalization_floor,
                self.ffn_expansion,
                self.operator_depth_logit_init,
            )
            <= 0.0
        ):
            raise ValueError("bottom numerical scales must be positive")
        if self.residual_scale_init > self.residual_scale_max:
            raise ValueError("bottom initial residual scale cannot exceed its bound")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("bottom.dropout must be in [0,1)")
        if not 0.0 <= self.controlled_delta_dropout < 1.0:
            raise ValueError("controlled transition dropout must be in [0,1)")
        if not 0.0 < self.initial_exit_probability < 1.0:
            raise ValueError("bottom initial exit probability must be in (0,1)")
        if self.gripper_field_dim != 6:
            raise ValueError("the resolved legacy physical action field has six gripper channels")
        if self.physical_decode_delta_blend != 0.25:
            raise ValueError("the resolved physical action decode blend is exactly 0.25")


@dataclass(frozen=True)
class ObjectiveConfig:
    future_dynamics: float = 0.10
    intent_structure: float = 0.02
    flow_warp: float = 0.03
    flow_identity_advantage: float = 0.02
    flow_static_identity: float = 0.01
    flow_cycle: float = 0.01
    flow_smoothness: float = 0.002
    flow_uncertainty: float = 0.005
    flow_refinement_sequence: float = 0.02
    proposal: float = 0.05
    event: float = 0.03
    motion: float = 0.03
    decoded_action: float = 0.08
    smooth_delta: float = 0.02
    physical_delta_consistency: float = 0.03
    event_positive_weight: float = 4.0
    event_focal_gamma: float = 1.0
    gripper_event_threshold: float = 0.10
    arm_motion_threshold: float = 0.02
    horizon_tail_emphasis: float = 0.20
    horizon_first_step_protection: float = 0.05

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if float(value) < 0.0:
                raise ValueError(f"objective.{name} must be non-negative")
        if self.future_dynamics <= 0.0 or self.intent_structure <= 0.0:
            raise ValueError("W and G/S require active future/structure budgets")
        if self.event_positive_weight != 4.0 or self.event_focal_gamma != 1.0:
            raise ValueError("the resolved focal event contract is positive=4 and gamma=1")
        if self.gripper_event_threshold != 0.10 or self.arm_motion_threshold != 0.02:
            raise ValueError("the resolved event/motion thresholds are 0.10 raw and 0.02 normalized")
        if self.horizon_tail_emphasis != 0.20 or self.horizon_first_step_protection != 0.05:
            raise ValueError("the resolved anchor-band emphasis is tail=0.20 and first=0.05")


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    # Preserve the resolved V120 optimization geometry.  The history proposal
    # was trained at 5e-5 while the public role trunk used 8e-5.  The active
    # Evidence-MMDiT decoder used 0.7x of the public LR, with its nested
    # contraction basis at 2x that decoder LR and without weight decay.
    history_proposal_lr_scale: float = 0.625
    bottom_decoder_lr_scale: float = 0.70
    bottom_capacity_relative_lr_scale: float = 2.0

    def validate(self) -> None:
        if min(self.epochs, self.batch_size, self.warmup_steps) <= 0:
            raise ValueError("epochs, batch_size and warmup_steps must be positive")
        if min(self.learning_rate, self.epsilon, self.grad_clip) <= 0.0:
            raise ValueError("optimizer lr/epsilon/grad_clip must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer betas must be in [0,1)")
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0,1]")
        if min(
            self.history_proposal_lr_scale,
            self.bottom_decoder_lr_scale,
            self.bottom_capacity_relative_lr_scale,
        ) <= 0.0:
            raise ValueError("optimizer role LR scales must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    compute_dtype: str = "bf16"
    inference_steps: int = 5
    log_every: int = 20
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_diagnostic_batches: int = 4

    def validate(self) -> None:
        if self.compute_dtype not in {"bf16", "fp32"}:
            raise ValueError("runtime.compute_dtype must be bf16 or fp32")
        if self.inference_steps != 5:
            raise ValueError("the active deployment contract uses five ODE steps")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        limits = (
            self.max_train_batches,
            self.max_val_batches,
            self.eval_diagnostic_batches,
        )
        if any(int(value) < 0 for value in limits):
            raise ValueError("runtime batch limits and diagnostics must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    dimensions: ModelDimensions = ModelDimensions()
    observation: ObservationConfig = ObservationConfig()
    top: TopConfig = TopConfig()
    bottom: BottomConfig = BottomConfig()
    objectives: ObjectiveConfig = ObjectiveConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def validate(self) -> None:
        ARCHITECTURE_MANIFEST.validate()
        for section in (
            self.data,
            self.dimensions,
            self.observation,
            self.top,
            self.bottom,
            self.objectives,
            self.optimizer,
            self.runtime,
        ):
            section.validate()
        native_side = round(self.dimensions.patches_per_camera**0.5)
        if native_side * native_side != self.dimensions.patches_per_camera:
            raise ValueError("patches_per_camera must form one square native chart")
        if self.observation.grid_size > native_side:
            raise ValueError("the coarse evidence grid cannot exceed the native chart")
        if self.bottom.controller_heads != self.dimensions.num_heads:
            raise ValueError("top and execution controller head counts must align")

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    def digest(self, *, include_paths: bool = False) -> str:
        payload = self.as_dict()
        if not include_paths:
            payload = dict(payload)
            # Files may be relocated without changing an experiment, but the
            # rest of the data section is executable semantics.  Dropping the
            # entire section used to make stride, camera keys, split policy,
            # normalizer and seed invisible to exact-resume identity.
            data = dict(cast(dict[str, object], payload["data"]))
            for name in (
                "raw_hdf5_root",
                "decoded_cache",
                "dino_cache",
                "t5_condition",
                "output_dir",
            ):
                data.pop(name, None)
            payload["data"] = data
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


SectionT = TypeVar("SectionT")


def _section(cls: type[SectionT], value: object, *, name: str) -> SectionT:
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name} must be a mapping")
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")
    return cls(**dict(value))


def config_from_mapping(value: Mapping[str, object]) -> ExperimentConfig:
    expected = {
        "data",
        "dimensions",
        "observation",
        "top",
        "bottom",
        "objectives",
        "optimizer",
        "runtime",
    }
    unknown = sorted(set(value) - expected)
    if unknown:
        raise ValueError(f"unknown config sections: {', '.join(unknown)}")
    raw_data = value.get("data", {})
    if isinstance(raw_data, Mapping) and "camera_names" in raw_data:
        raw_data = dict(raw_data)
        raw_data["camera_names"] = tuple(str(item) for item in raw_data["camera_names"])  # type: ignore[index]
    config = ExperimentConfig(
        data=_section(DataConfig, raw_data, name="data"),
        dimensions=_section(ModelDimensions, value.get("dimensions", {}), name="dimensions"),
        observation=_section(ObservationConfig, value.get("observation", {}), name="observation"),
        top=_section(TopConfig, value.get("top", {}), name="top"),
        bottom=_section(BottomConfig, value.get("bottom", {}), name="bottom"),
        objectives=_section(ObjectiveConfig, value.get("objectives", {}), name="objectives"),
        optimizer=_section(OptimizerConfig, value.get("optimizer", {}), name="optimizer"),
        runtime=_section(RuntimeConfig, value.get("runtime", {}), name="runtime"),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("mainline config root must be a mapping")
    return config_from_mapping(payload)
