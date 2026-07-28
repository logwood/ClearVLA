from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.dataset import (
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
    PolicyWindowDataset,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
    evaluate_model_path_intervention,
    evaluate_v39_policy,
    evaluate_v98_flow_address_intervention,
)
from clearvla.experiments.observed_state_lab.sampling_path_probe import (
    evaluate_sampling_path_probe,
)
from clearvla.experiments.observed_state_lab.world_runtime import jsonable
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.system import V39PolicySystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V39 staged mid-cut temporal policy.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--eval-inference-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--flow-address-intervention-batches",
        type=int,
        default=0,
        help=(
            "Run a paired V98+ baseline/zero/spatial-shuffle raw-address probe "
            "on N evenly spaced batches with episode-cluster confidence intervals."
        ),
    )
    parser.add_argument("--flow-address-bootstrap-reps", type=int, default=2000)
    parser.add_argument("--flow-address-bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--action-path-intervention-batches",
        type=int,
        default=0,
        help=(
            "Run the paired V101 policy/world-residual/raw-address action-path "
            "probe on N validation-wide plus gripper-event batches."
        ),
    )
    parser.add_argument(
        "--model-path-intervention-batches",
        type=int,
        default=0,
        help=(
            "Run the paired V103-V111 goal/history/phase/interval/"
            "flow/address/G-W-P causal model-path probe on N selected "
            "validation batches. The newest complete serialized contract is "
            "selected unless explicitly required."
        ),
    )
    parser.add_argument(
        "--model-path-required-contract",
        choices=(
            "auto",
            "v103",
            "v104",
            "v105",
            "v106",
            "v107",
            "v108",
            "v109",
            "v110",
            "v111",
        ),
        default="auto",
        help=(
            "Formal contract required before a model-path probe. 'auto' uses "
            "the newest complete serialized V111-to-V103 contract."
        ),
    )
    parser.add_argument(
        "--sampling-path-probe-batches",
        type=int,
        default=0,
        help=(
            "Run a frozen paired probe on N selected validation batches that "
            "separates training-call, recursive off-path, and 5/10/20-step "
            "solver error."
        ),
    )
    parser.add_argument(
        "--sampling-path-probe-steps",
        type=int,
        nargs="+",
        default=(5, 10, 20),
        help="Positive solver step counts; the deployed five-step baseline is required.",
    )
    parser.add_argument(
        "--sampling-path-require-v104-contract",
        action="store_true",
        help="Require the complete serialized V104 parent contract before probing.",
    )
    parser.add_argument("--action-path-bootstrap-reps", type=int, default=2000)
    parser.add_argument("--action-path-bootstrap-seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)
    selected_probe_count = sum(
        int(value) > 0
        for value in (
            args.flow_address_intervention_batches,
            args.action_path_intervention_batches,
            args.model_path_intervention_batches,
            args.sampling_path_probe_batches,
        )
    )
    if selected_probe_count > 1:
        raise ValueError(
            "choose only one flow-address/action-path/model-path/sampling-path probe"
        )
    checkpoint_probe = (
        int(args.flow_address_intervention_batches) > 0
        or int(args.action_path_intervention_batches) > 0
        or int(args.model_path_intervention_batches) > 0
        or int(args.sampling_path_probe_batches) > 0
    )
    if (
        args.sampling_path_require_v104_contract
        and int(args.sampling_path_probe_batches) <= 0
    ):
        raise ValueError(
            "--sampling-path-require-v104-contract applies only to "
            "--sampling-path-probe-batches"
        )
    checkpoint_stat_before = args.checkpoint.stat() if checkpoint_probe else None
    checkpoint_sha256_before = _sha256_file(args.checkpoint) if checkpoint_probe else None
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_schema = payload.get("schema")
    if checkpoint_schema not in POLICY_CHECKPOINT_SCHEMAS:
        raise ValueError("--checkpoint must be a V39/V40 policy checkpoint")
    context = payload["context"]
    dataset_config = ObservedStateDatasetConfig(**context["dataset"])
    visual_geometry = context.get("visual_geometry")
    if visual_geometry is None and "source_world_model" in context:
        # Backward compatibility with the first V38 package.
        source_world = context["source_world_model"]
        visual_geometry = {
            "history_length": int(source_world["history_length"]),
            "future_count": int(source_world["num_future"]),
            "num_cameras": int(source_world["num_cameras"]),
            "patches_per_camera": int(source_world["patches_per_camera"]),
            "latent_dim": int(source_world["latent_dim"]),
        }
    if visual_geometry is None:
        raise ValueError("checkpoint context is missing visual_geometry")

    class Geometry:
        history_length = int(visual_geometry["history_length"])
        num_future = int(visual_geometry["future_count"])
        num_cameras = int(visual_geometry["num_cameras"])
        patches_per_camera = int(visual_geometry["patches_per_camera"])
        latent_dim = int(visual_geometry["latent_dim"])

    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    policy_config = V39PolicyConfig(**payload["policy_config"])
    min_length = (
        dataset_config.world_horizon
        + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets))
        + 2
    )
    episodes, train_ids, val_ids, test_ids, _, _, image_store, skipped = load_data(
        args,
        min_length=min_length,
        normalizer_mode=action_norm.mode,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        splits=context["splits"],
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    effective = ObservedStateDatasetConfig(
        **{
            **context["dataset"],
            "return_images": (
                args.condition_mode != "dinov2-cache"
                or bool(int(policy_config.flow_jepa_raw_image_enabled))
            ),
        }
    )
    dataset = PolicyWindowDataset(
        ObservedStateWindowDataset(
            episodes,
            split_ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_norm,
            action_normalizer=action_norm,
            config=effective,
        )
    )
    loader = make_loader(
        dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device
    )
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=Geometry.latent_dim,
        debug_patches_per_camera=Geometry.patches_per_camera,
        device=device,
        dtype=dtype,
    )
    if latent_dim != Geometry.latent_dim or (
        patches is not None and patches != Geometry.patches_per_camera
    ):
        raise ValueError("conditioner geometry does not match checkpoint")
    trainer = V39PolicyTrainerConfig(**payload["trainer_config"])
    if args.eval_inference_steps is not None:
        trainer = V39PolicyTrainerConfig(
            **{**asdict(trainer), "eval_inference_steps": int(args.eval_inference_steps)}
        )
    system = V39PolicySystem(policy_config)
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32)
    flow_address_result = None
    action_path_result = None
    sampling_path_result = None
    model_path_batches = int(args.model_path_intervention_batches)
    action_path_batches = int(args.action_path_intervention_batches)
    sampling_path_batches = int(args.sampling_path_probe_batches)
    if sampling_path_batches > 0:
        sampling_path_result = evaluate_sampling_path_probe(
            system=system,
            loader=loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=cameras,
            action_normalizer=action_norm,
            trainer=trainer,
            probe_batches=sampling_path_batches,
            solver_steps=tuple(int(value) for value in args.sampling_path_probe_steps),
            max_batches=args.max_val_batches,
            require_complete_v104_contract=bool(
                args.sampling_path_require_v104_contract
            ),
        )
        metrics = {
            "sampling_path_teacher_contract_velocity_rmse": sampling_path_result[
                "summary"
            ]["teacher_contract_velocity_rmse"],
            "sampling_path_deploy_bridge_velocity_rmse": sampling_path_result[
                "summary"
            ]["deploy_bridge_velocity_rmse"],
            "sampling_path_recursive_velocity_rmse": sampling_path_result[
                "summary"
            ]["recursive_velocity_rmse"],
            "sampling_path_call_contract_excess_mse": sampling_path_result[
                "summary"
            ]["call_contract_excess_mse"],
            "sampling_path_off_path_excess_mse": sampling_path_result[
                "summary"
            ]["off_path_excess_mse"],
        }
    elif model_path_batches > 0 or action_path_batches > 0:
        requested_model_contract = str(args.model_path_required_contract)
        serialized_v104 = all(
            int(getattr(policy_config, name, 0)) == 1
            for name in (
                "flow_jepa_bounded_flow_coordinates",
                "flow_jepa_sequential_horizon_memory",
                "role_residual_amplitude_contract",
            )
        )
        serialized_v105 = bool(
            serialized_v104
            and int(
                getattr(policy_config, "flow_jepa_horizon_soft_address", 0)
            )
            == 1
            and int(
                getattr(
                    trainer,
                    "flow_jepa_future_reliable_normalization",
                    0,
                )
            )
            == 1
            and float(
                getattr(
                    trainer,
                    "flow_jepa_horizon_address_loss_weight",
                    0.0,
                )
            )
            > 0.0
        )
        serialized_v106 = bool(
            serialized_v105
            and int(
                getattr(policy_config, "flow_jepa_interval_stage_delta", 0)
            )
            == 1
            and int(
                getattr(policy_config, "flow_jepa_variance_safe_routing", 0)
            )
            == 1
            and float(
                getattr(
                    trainer,
                    "flow_jepa_interval_stage_loss_weight",
                    0.0,
                )
            )
            > 0.0
        )
        serialized_v107 = bool(
            serialized_v106
            and all(
                int(getattr(policy_config, name, 0)) == 1
                for name in (
                    "flow_jepa_policy_multi_glimpse_address",
                    "flow_jepa_horizon_cell_fine_address",
                    "flow_jepa_interval_stage_typed_value",
                    "role_residual_contract_after_gate",
                )
            )
        )
        serialized_v108 = bool(
            serialized_v107
            and int(
                getattr(policy_config, "flow_jepa_online_horizon_address", 0)
            )
            == 1
        )
        serialized_v109 = bool(
            serialized_v108
            and int(
                getattr(
                    policy_config,
                    "flow_jepa_progressive_grounding_address",
                    0,
                )
            )
            == 1
        )
        serialized_v110 = bool(
            serialized_v109
            and int(
                getattr(
                    policy_config,
                    "flow_jepa_coordinate_typed_raw_detail",
                    0,
                )
            )
            == 1
            and int(getattr(policy_config, "flow_jepa_raw_micro_grid", 0)) == 3
        )
        serialized_v111 = bool(
            serialized_v110
            and int(
                getattr(
                    policy_config,
                    "flow_jepa_structured_ownership_bottleneck",
                    0,
                )
            )
            == 1
        )
        required_model_contract = (
            (
                "v111"
                if serialized_v111
                else "v110"
                if serialized_v110
                else "v109"
                if serialized_v109
                else "v108"
                if serialized_v108
                else "v107"
                if serialized_v107
                else "v106"
                if serialized_v106
                else "v105"
                if serialized_v105
                else "v104"
                if serialized_v104
                else "v103"
            )
            if requested_model_contract == "auto"
            else requested_model_contract
        )
        complete_v110_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v110"
        )
        complete_v111_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v111"
        )
        complete_v109_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v109"
        )
        complete_v108_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v108"
        )
        complete_v107_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v107"
        )
        complete_v106_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v106"
        )
        complete_v105_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v105"
        )
        complete_v104_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v104"
        )
        complete_v103_candidate = bool(
            model_path_batches > 0 and required_model_contract == "v103"
        )
        if model_path_batches <= 0 and requested_model_contract != "auto":
            raise ValueError(
                "--model-path-required-contract applies only to "
                "--model-path-intervention-batches"
            )
        if complete_v104_candidate and not all(
                int(getattr(policy_config, name, 0)) == 1
                for name in (
                    "flow_jepa_bounded_flow_coordinates",
                    "flow_jepa_sequential_horizon_memory",
                    "role_residual_amplitude_contract",
                )
            ):
            # Keep the error at the CLI boundary explicit. The complete V104
            # validator below still checks the full V103 parent and RMS limits.
            raise ValueError(
                "the requested V104 model-path probe checkpoint does not "
                "serialize all three V104 structural flags"
            )
        if complete_v105_candidate and not serialized_v105:
            raise ValueError(
                "the requested V105 model-path probe checkpoint does not "
                "serialize the complete horizon-address/reliable-future contract"
            )
        if complete_v106_candidate and not serialized_v106:
            raise ValueError(
                "the requested V106 model-path probe checkpoint does not "
                "serialize the complete interval-stage/variance-safe contract"
            )
        if complete_v107_candidate and not serialized_v107:
            raise ValueError(
                "the requested V107 model-path probe checkpoint does not "
                "serialize the complete address/typed-stage/post-gate contract"
            )
        if complete_v108_candidate and not serialized_v108:
            raise ValueError(
                "the requested V108 model-path probe checkpoint does not "
                "serialize the online G3-to-W1 horizon-address topology"
            )
        if complete_v109_candidate and not serialized_v109:
            raise ValueError(
                "the requested V109 model-path probe checkpoint does not "
                "serialize the progressive G1/G2/G3 address topology"
            )
        if complete_v110_candidate and not serialized_v110:
            raise ValueError(
                "the requested V110 model-path probe checkpoint does not "
                "serialize typed coordinate RGB/detail ownership and the 3x3 P1 micro-grid"
            )
        if complete_v111_candidate and not serialized_v111:
            raise ValueError(
                "the requested V111 model-path probe checkpoint does not "
                "serialize functional public/owner evidence separation"
            )
        action_path_result = evaluate_model_path_intervention(
            system=system,
            loader=loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=cameras,
            action_normalizer=action_norm,
            trainer=trainer,
            intervention_batches=(
                model_path_batches if model_path_batches > 0 else action_path_batches
            ),
            max_batches=args.max_val_batches,
            bootstrap_reps=int(args.action_path_bootstrap_reps),
            bootstrap_seed=int(args.action_path_bootstrap_seed),
            require_complete_v103_contract=complete_v103_candidate,
            require_complete_v104_contract=complete_v104_candidate,
            require_complete_v105_contract=complete_v105_candidate,
            require_complete_v106_contract=complete_v106_candidate,
            require_complete_v107_contract=complete_v107_candidate,
            require_complete_v108_contract=complete_v108_candidate,
            require_complete_v109_contract=complete_v109_candidate,
            require_complete_v110_contract=complete_v110_candidate,
            require_complete_v111_contract=complete_v111_candidate,
        )
        metrics = action_path_result["modes"]["baseline"]
    elif int(args.flow_address_intervention_batches) > 0:
        flow_address_result = evaluate_v98_flow_address_intervention(
            system=system,
            loader=loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=cameras,
            action_normalizer=action_norm,
            trainer=trainer,
            intervention_batches=int(args.flow_address_intervention_batches),
            max_batches=args.max_val_batches,
            bootstrap_reps=int(args.flow_address_bootstrap_reps),
            bootstrap_seed=int(args.flow_address_bootstrap_seed),
        )
        metrics = flow_address_result["modes"]["baseline"]
    else:
        metrics = evaluate_v39_policy(
            system=system,
            loader=loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=cameras,
            action_normalizer=action_norm,
            trainer=trainer,
            max_batches=args.max_val_batches,
        )
    eval_schema = (
        "clearvla-v40-policy-eval-v1"
        if checkpoint_schema == "clearvla-v40-policy-checkpoint-v1"
        else "clearvla-v39-policy-eval-v1"
    )
    out = {
        "schema": eval_schema,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
        "skipped": skipped,
    }
    probe_result = (
        sampling_path_result
        if sampling_path_result is not None
        else action_path_result
        if action_path_result is not None
        else flow_address_result
    )
    if probe_result is not None:
        assert checkpoint_stat_before is not None
        assert checkpoint_sha256_before is not None
        checkpoint_stat_after = args.checkpoint.stat()
        checkpoint_sha256_after = _sha256_file(args.checkpoint)
        checkpoint_stable = bool(
            checkpoint_sha256_before == checkpoint_sha256_after
            and checkpoint_stat_before.st_mtime_ns == checkpoint_stat_after.st_mtime_ns
            and checkpoint_stat_before.st_size == checkpoint_stat_after.st_size
        )
        if not checkpoint_stable:
            raise RuntimeError("checkpoint changed while the intervention probe was running")
        git_commit, git_dirty = _git_identity()
        normalizer_fingerprint = hashlib.sha256(
            json.dumps(payload["action_normalizer"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        out["run_identity"] = {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_schema": checkpoint_schema,
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_global_step": payload.get("global_step"),
            "checkpoint_sha256": checkpoint_sha256_before,
            "checkpoint_mtime_ns": checkpoint_stat_before.st_mtime_ns,
            "checkpoint_size": checkpoint_stat_before.st_size,
            "checkpoint_stable": checkpoint_stable,
            "complete_v103_contract_verified": bool(
                probe_result.get("complete_v103_contract_verified", False)
            ),
            "complete_v104_contract_verified": bool(
                probe_result.get("complete_v104_contract_verified", False)
            ),
            "complete_v105_contract_verified": bool(
                probe_result.get("complete_v105_contract_verified", False)
            ),
            "complete_v106_contract_verified": bool(
                probe_result.get("complete_v106_contract_verified", False)
            ),
            "complete_v107_contract_verified": bool(
                probe_result.get("complete_v107_contract_verified", False)
            ),
            "complete_v108_contract_verified": bool(
                probe_result.get("complete_v108_contract_verified", False)
            ),
            "complete_v109_contract_verified": bool(
                probe_result.get("complete_v109_contract_verified", False)
            ),
            "complete_v110_contract_verified": bool(
                probe_result.get("complete_v110_contract_verified", False)
            ),
            "complete_v111_contract_verified": bool(
                probe_result.get("complete_v111_contract_verified", False)
            ),
            "flow_jepa_raw_image_enabled": int(policy_config.flow_jepa_raw_image_enabled),
            "flow_jepa_coordinate_typed_raw_detail": int(
                getattr(policy_config, "flow_jepa_coordinate_typed_raw_detail", 0)
            ),
            "flow_jepa_structured_ownership_bottleneck": int(
                getattr(
                    policy_config,
                    "flow_jepa_structured_ownership_bottleneck",
                    0,
                )
            ),
            "flow_jepa_raw_micro_grid": int(
                getattr(policy_config, "flow_jepa_raw_micro_grid", 0)
            ),
            "flow_jepa_zero_flow_guard": int(
                getattr(policy_config, "flow_jepa_zero_flow_guard", 0)
            ),
            "flow_jepa_world_anchor_write_only": int(
                getattr(policy_config, "flow_jepa_world_anchor_write_only", 0)
            ),
            "flow_jepa_late_policy_detail": int(
                getattr(policy_config, "flow_jepa_late_policy_detail", 0)
            ),
            "flow_jepa_late_policy_detail_scale": float(
                getattr(policy_config, "flow_jepa_late_policy_detail_scale", 0.25)
            ),
            "flow_jepa_policy_workspace_horizon_pool": int(
                getattr(
                    policy_config,
                    "flow_jepa_policy_workspace_horizon_pool",
                    0,
                )
            ),
            "flow_jepa_soft_address_lattice": int(
                getattr(policy_config, "flow_jepa_soft_address_lattice", 0)
            ),
            "flow_jepa_predictive_change_contract": int(
                getattr(
                    policy_config,
                    "flow_jepa_predictive_change_contract",
                    0,
                )
            ),
            "flow_jepa_horizon_soft_address": int(
                getattr(policy_config, "flow_jepa_horizon_soft_address", 0)
            ),
            "flow_jepa_horizon_address_update_scale": float(
                getattr(
                    policy_config,
                    "flow_jepa_horizon_address_update_scale",
                    0.0,
                )
            ),
            "flow_jepa_future_reliable_normalization": int(
                getattr(
                    trainer,
                    "flow_jepa_future_reliable_normalization",
                    0,
                )
            ),
            "flow_jepa_horizon_address_loss_weight": float(
                getattr(
                    trainer,
                    "flow_jepa_horizon_address_loss_weight",
                    0.0,
                )
            ),
            "flow_jepa_interval_stage_delta": int(
                getattr(policy_config, "flow_jepa_interval_stage_delta", 0)
            ),
            "flow_jepa_interval_boundaries": [
                int(value)
                for value in getattr(
                    policy_config,
                    "flow_jepa_effective_interval_boundaries",
                    (),
                )
            ],
            "flow_jepa_interval_support_offsets": [
                int(value)
                for value in getattr(
                    policy_config,
                    "flow_jepa_effective_interval_support_offsets",
                    (),
                )
            ],
            "flow_jepa_interval_stage_update_scale": float(
                getattr(
                    policy_config,
                    "flow_jepa_interval_stage_update_scale",
                    0.0,
                )
            ),
            "flow_jepa_interval_stage_loss_weight": float(
                getattr(
                    trainer,
                    "flow_jepa_interval_stage_loss_weight",
                    0.0,
                )
            ),
            "flow_jepa_variance_safe_routing": int(
                getattr(policy_config, "flow_jepa_variance_safe_routing", 0)
            ),
            "flow_jepa_routing_norm_floor": float(
                getattr(policy_config, "flow_jepa_routing_norm_floor", 0.0)
            ),
            "flow_jepa_horizon_value_max_rms": float(
                getattr(policy_config, "flow_jepa_horizon_value_max_rms", 0.0)
            ),
            "role_attnres_enabled": int(
                getattr(policy_config, "role_attnres_enabled", 0)
            ),
            "role_attnres_ground_to_world": int(
                getattr(policy_config, "role_attnres_ground_to_world", 0)
            ),
            "role_attnres_world_to_policy": int(
                getattr(policy_config, "role_attnres_world_to_policy", 0)
            ),
            "role_attnres_policy_to_mmdit": int(
                getattr(policy_config, "role_attnres_policy_to_mmdit", 0)
            ),
            "goal_condition_exact_null": int(
                getattr(policy_config, "goal_condition_exact_null", 0)
            ),
            "action_history_condition_exact_null": int(
                getattr(
                    policy_config,
                    "action_history_condition_exact_null",
                    0,
                )
            ),
            "stateless_phase_enabled": int(
                getattr(policy_config, "stateless_phase_enabled", 0)
            ),
            "action_normalizer_fingerprint": normalizer_fingerprint,
            "split": args.split,
            "split_episode_ids": [int(value) for value in split_ids],
            "planned_batches": probe_result["planned_batches"],
            "finished_intervention_batches": probe_result[
                "finished_intervention_batches"
            ],
            "intervention_samples": probe_result["intervention_samples"],
            "intervention_coverage": probe_result["intervention_coverage"],
            "selection_strategy": probe_result.get("selection_strategy"),
            "event_candidate_batches": probe_result.get("event_candidate_batches"),
            "selected_event_batches": probe_result.get("selected_event_batches"),
            "selected_episode_count": probe_result.get("selected_episode_count"),
            "selected_episode_ids": probe_result.get("selected_episode_ids"),
            "selected_event_episode_ids": probe_result.get(
                "selected_event_episode_ids"
            ),
        }
    if flow_address_result is not None:
        out["flow_address_intervention"] = flow_address_result
    if action_path_result is not None:
        result_key = (
            "model_path_intervention"
            if action_path_result.get("schema")
            in {
                "clearvla-v103-model-path-intervention-v1",
                "clearvla-v103-model-path-intervention-v2",
                "clearvla-v103-model-path-intervention-v3",
                "clearvla-v104-model-path-intervention-v3",
                "clearvla-v105-model-path-intervention-v4",
                "clearvla-v106-model-path-intervention-v5",
                "clearvla-v107-model-path-intervention-v6",
                "clearvla-v108-model-path-intervention-v7",
                "clearvla-v109-model-path-intervention-v8",
                "clearvla-v110-model-path-intervention-v9",
                "clearvla-v111-model-path-intervention-v10",
            }
            else "action_path_intervention"
        )
        out[result_key] = action_path_result
    if sampling_path_result is not None:
        out["sampling_path_probe"] = sampling_path_result
    print(json.dumps(jsonable(out), indent=2), flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
