"""Small, nested configuration for the active ClearVLA graph.

Fixed architectural facts such as the 3-2-3 topology, four intervals and one
protected bottom ingress live in the architecture manifest and module graph.
They are deliberately not represented as combinable boolean switches here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, TypeVar, cast

from clearvla.data.action_chart import resolve_action_state_profile

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
    data_profile: str = "identity_7d_pen"
    action_key: str = "action"
    action_state_key: str = ""
    state_key: str = "qpos"
    top_camera_key: str = "observations/images/cam_high"
    wrist_camera_key: str = "observations/images/cam_right_wrist"
    camera_key_overrides: tuple[tuple[str, str], ...] = ()
    image_store_mode: str = "decoded-cache"
    image_frame_lru_capacity: int = 512
    image_open_file_capacity: int = 8
    cache_side: int = 336
    dinov2_model: str = "facebook/dinov2-base"
    # CUDA attention kernels can choose a different numerical reduction path
    # for different batch shapes (especially in bf16).  The existing DINO
    # cache was built with 32 samples per encoder call; deployment must use
    # the same reference shape to remain token-equivalent to training.
    dinov2_reference_batch_size: int = 32
    split_mode: str = "ordered-counts"
    split_manifest: str = ""
    task_selection_manifest: str = ""
    normalizer_artifact: str = ""
    # Optional source-side task filter.  It is applied before split resolution
    # so a single CALVIN task can reuse the full /data cache namespace without
    # copying or relinking HDF5 files.
    task_filter: str = ""
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
    # ``None`` inherits the established Pen threshold only for the Pen
    # identity profile.  Other source charts must opt into a threshold rather
    # than silently borrowing raw units from another dataset.
    sampling_gripper_event_threshold: float | None = None

    def camera_key_map(self) -> dict[str, str]:
        if self.camera_key_overrides:
            return {str(name): str(key) for name, key in self.camera_key_overrides}
        legacy = {
            "top": self.top_camera_key,
            "wrist": self.wrist_camera_key,
        }
        return {name: legacy[name] for name in self.camera_names if name in legacy}

    def validate(self) -> None:
        string_fields = (
            "raw_hdf5_root",
            "hdf5_glob",
            "decoded_cache",
            "dino_cache",
            "t5_condition",
            "output_dir",
            "data_profile",
            "action_key",
            "state_key",
            "top_camera_key",
            "wrist_camera_key",
            "image_store_mode",
            "dinov2_model",
            "split_mode",
            "normalizer",
        )
        for name in string_fields:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{name} must be a non-empty string")
        if not isinstance(self.action_state_key, str):
            raise ValueError("data.action_state_key must be a string")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("data.camera_names must be a non-empty ordered unique tuple")
        if any(
            not name or any(character in name for character in "/\\") for name in self.camera_names
        ):
            raise ValueError("camera names must be non-empty cache-safe identifiers")
        override_rows = tuple((str(name), str(key)) for name, key in self.camera_key_overrides)
        if len({name for name, _ in override_rows}) != len(override_rows):
            raise ValueError("camera key overrides cannot contain duplicate camera names")
        if override_rows:
            if tuple(name for name, _ in override_rows) != tuple(self.camera_names):
                raise ValueError(
                    "explicit camera key overrides must follow and cover camera_names exactly"
                )
            if any(not key.strip() for _, key in override_rows):
                raise ValueError("camera key override paths must be non-empty")
        if self.cache_side != 336:
            raise ValueError("the established decoded and DINO caches use 336x336 preprocessing")
        if self.dinov2_reference_batch_size <= 0:
            raise ValueError("data.dinov2_reference_batch_size must be positive")
        if self.image_store_mode not in {"decoded-cache", "hdf5-direct"}:
            raise ValueError("data.image_store_mode must be decoded-cache or hdf5-direct")
        if self.image_frame_lru_capacity < 0 or self.image_open_file_capacity <= 0:
            raise ValueError("image-store LRU capacity must be non-negative and files positive")
        if not isinstance(self.task_filter, str):
            raise ValueError("data.task_filter must be a string")
        if self.split_mode == "ordered-counts":
            if self.split_manifest or self.task_selection_manifest or self.normalizer_artifact:
                raise ValueError(
                    "ordered-counts split cannot name RDT split/selection/normalizer artifacts"
                )
            if (self.train_episodes, self.val_episodes, self.test_episodes) != (63, 5, 5):
                raise ValueError("formal Pen comparison runs use the established 63/5/5 split")
        elif self.split_mode == "manifest":
            if not isinstance(self.split_manifest, str) or not self.split_manifest.strip():
                raise ValueError("manifest split requires data.split_manifest")
            if (self.train_episodes, self.val_episodes, self.test_episodes) != (0, 0, 0):
                raise ValueError("manifest membership cannot also use ordered episode counts")
            if bool(self.task_selection_manifest) != bool(self.normalizer_artifact):
                raise ValueError(
                    "a bounded task selection requires one shared normalizer artifact, "
                    "and the artifact cannot be configured without the selection"
                )
        elif self.split_mode == "episode-manifest":
            if not isinstance(self.split_manifest, str) or not self.split_manifest.strip():
                raise ValueError("episode-manifest split requires data.split_manifest")
            if (self.train_episodes, self.val_episodes, self.test_episodes) != (0, 0, 0):
                raise ValueError(
                    "episode-manifest membership cannot also use ordered episode counts"
                )
            if self.task_selection_manifest or self.normalizer_artifact:
                raise ValueError(
                    "episode-manifest split cannot use RDT task-selection or normalizer artifacts"
                )
        else:
            raise ValueError(
                "data.split_mode must be ordered-counts, manifest, or episode-manifest"
            )
        if self.normalizer != "zscore":
            raise ValueError("the active action/state chart uses z-score normalization")
        if min(self.stride, self.num_workers) < 0 or self.stride == 0:
            raise ValueError("data stride must be positive and worker count non-negative")
        if self.seed < 0:
            raise ValueError("data seed must be non-negative")
        resolve_action_state_profile(self.data_profile).validate()
        if self.sampling_gripper_event_threshold is not None:
            threshold = float(self.sampling_gripper_event_threshold)
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("sampling gripper event threshold must be finite and non-negative")
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
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"top {name} must be in [0,1)")


@dataclass(frozen=True)
class BottomConfig:
    flow_time_distribution: str = "v120_mirrored_beta_1_5_1"
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
    max_dwell: int = 2
    execution_warmup_steps: int = 200
    execution_transition_steps: int = 1000
    execution_eval_policy: str = "soft"
    initial_exit_probability: float = 0.10
    controlled_delta_rank: int = 8
    controlled_action_tokens: int = 8
    controlled_delta_dropout: float = 0.0
    gripper_field_dim: int = 6
    physical_decode_delta_blend: float = 0.25

    def validate(self) -> None:
        if self.flow_time_distribution != "v120_mirrored_beta_1_5_1":
            raise ValueError("formal training uses the mirrored V120 beta_1_5_1 flow time")
        integer_fields = (
            self.evidence_depth,
            self.latent_dim,
            self.operator_rank,
            self.operator_groups,
            self.controller_tokens,
            self.controller_depth,
            self.controller_heads,
            self.max_dwell,
            self.execution_transition_steps,
            self.controlled_delta_rank,
            self.controlled_action_tokens,
            self.gripper_field_dim,
        )
        if any(int(value) <= 0 for value in integer_fields):
            raise ValueError("bottom dimensions must be positive")
        if self.execution_warmup_steps < 0:
            raise ValueError("bottom execution warmup cannot be negative")
        if self.max_dwell != 2 or self.execution_eval_policy != "soft":
            raise ValueError("the recovered V120 execution contract is dwell=2/eval=soft")
        if self.execution_warmup_steps != 200 or self.execution_transition_steps != 1000:
            raise ValueError("the recovered V120 execution schedule is warmup=200/transition=1000")
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
    gripper_trajectory: float = 0.03
    motion: float = 0.03
    decoded_action: float = 0.08
    smooth_delta: float = 0.02
    physical_delta_consistency: float = 0.03
    # V120 trains the candidate value reader; execution cost itself is audit
    # only and therefore has no weight here.
    execution_value: float = 0.05
    execution_value_huber_delta: float = 0.10
    # The raw-unit event threshold selects where continuous gripper trajectory
    # closure begins. It never binarizes the action target or enters runtime.
    gripper_event_threshold: float = 0.10
    arm_motion_threshold: float = 0.02
    horizon_tail_emphasis: float = 0.20
    horizon_first_step_protection: float = 0.05

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"objective.{name} must be finite and non-negative")
        if self.future_dynamics <= 0.0 or self.intent_structure <= 0.0:
            raise ValueError("W and G/S require active future/structure budgets")
        if self.arm_motion_threshold != 0.02:
            raise ValueError("the resolved normalized arm-motion threshold is 0.02")
        if self.horizon_tail_emphasis != 0.20 or self.horizon_first_step_protection != 0.05:
            raise ValueError("the resolved anchor-band emphasis is tail=0.20 and first=0.05")
        if self.execution_value != 0.05 or self.execution_value_huber_delta != 0.10:
            raise ValueError("the recovered V120 execution-value contract is weight=.05 beta=.10")


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
        if (
            min(
                self.history_proposal_lr_scale,
                self.bottom_decoder_lr_scale,
                self.bottom_capacity_relative_lr_scale,
            )
            <= 0.0
        ):
            raise ValueError("optimizer role LR scales must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    compute_dtype: str = "bf16"
    inference_steps: int = 5
    log_every: int = 20
    max_train_batches: int = 0
    max_val_batches: int = 0
    eval_sampling_diagnostic_batches: int = 16
    eval_proposal_ablation_batches: int = 16
    eval_execution_ablation_batches: int = 8

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
            self.eval_sampling_diagnostic_batches,
            self.eval_proposal_ablation_batches,
            self.eval_execution_ablation_batches,
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
        if len(self.data.camera_names) != self.dimensions.num_cameras:
            raise ValueError("data camera order must align with model num_cameras")
        profile = resolve_action_state_profile(self.data.data_profile)
        if profile.output_dim != self.dimensions.action_dim:
            raise ValueError("data profile width must align with dimensions.action_dim")
        if profile.output_dim != self.dimensions.state_dim:
            raise ValueError("data profile width must align with dimensions.state_dim")
        sampling_threshold = self.data.sampling_gripper_event_threshold
        if profile.name == "identity_7d_pen":
            if self.objectives.gripper_event_threshold != 0.10:
                raise ValueError("the Pen gripper trajectory threshold remains exactly 0.10 raw")
            if sampling_threshold is not None and float(sampling_threshold) != 0.10:
                raise ValueError("the Pen sampler threshold cannot differ from 0.10 raw")
        elif sampling_threshold is not None:
            if float(sampling_threshold) <= 0.0:
                raise ValueError("non-Pen gripper event threshold must be positive")
            if float(sampling_threshold) != float(self.objectives.gripper_event_threshold):
                raise ValueError(
                    "non-Pen sampling, gripper trajectory and validation thresholds "
                    "must be identical"
                )
        elif self.data.split_mode == "episode-manifest":
            raise ValueError(
                "episode-manifest training requires an explicit source-chart "
                "gripper-event threshold"
            )

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
                "split_manifest",
                "task_selection_manifest",
                "normalizer_artifact",
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
    if isinstance(raw_data, Mapping):
        raw_data = dict(raw_data)
        if "camera_names" in raw_data:
            raw_data["camera_names"] = tuple(  # type: ignore[index]
                str(item)
                for item in raw_data["camera_names"]  # type: ignore[index]
            )
        if "camera_key_overrides" in raw_data:
            camera_keys = raw_data["camera_key_overrides"]
            if isinstance(camera_keys, Mapping):
                names = tuple(raw_data.get("camera_names", ()))
                unknown = sorted(set(str(name) for name in camera_keys) - set(names))
                if unknown:
                    raise ValueError(f"camera key overrides name unknown cameras: {unknown}")
                raw_data["camera_key_overrides"] = tuple(
                    (str(name), str(camera_keys[name])) for name in names
                )
            elif isinstance(camera_keys, (tuple, list)):
                raw_data["camera_key_overrides"] = tuple(
                    (str(row[0]), str(row[1])) for row in camera_keys
                )
            else:
                raise TypeError("data.camera_key_overrides must be a mapping or pair sequence")
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
