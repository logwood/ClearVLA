"""Receding-horizon adapter from native simulator evidence to the mainline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

import numpy as np
import torch

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.instructions import normalize_instruction
from clearvla.mainline.gripper_contract import (
    VALID_GRIPPER_OUTPUT_MODES,
    is_binary_gripper_mode,
)
from clearvla.mainline.interfaces import (
    CurrentObservation,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.vision.preprocessing import PreprocessConfig, preprocessing_identity

from .checkpoint import DeploymentBundle, load_deployment_checkpoint
from .history import HistorySnapshot
from .vision import DinoV2OnlineEncoder


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"deployment ABI {name} must be a mapping")
    return cast(Mapping[str, object], value)


class ClearVLACheckpointPolicy:
    """Strict online policy; no teacher or future-supervision path is present."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: torch.device,
        t5_condition: str | Path | None = None,
        dinov2_model: str | Path | None = None,
        dinov2_local_files_only: bool = False,
        seed: int = 0,
    ) -> None:
        self.bundle: DeploymentBundle = load_deployment_checkpoint(
            checkpoint,
            device=device,
            t5_condition=t5_condition,
        )
        config = self.bundle.config
        dtype = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[config.runtime.compute_dtype]
        observation_abi = _mapping(
            self.bundle.deployment_abi["observation"], name="observation"
        )
        preprocessing_abi = _mapping(
            observation_abi["rgb_preprocessing"], name="observation.rgb_preprocessing"
        )
        preprocessing_config = _mapping(
            preprocessing_abi["config"], name="observation.rgb_preprocessing.config"
        )
        self.preprocessing = PreprocessConfig.from_dict(dict(preprocessing_config))
        actual_preprocessing = preprocessing_identity(self.preprocessing)
        if actual_preprocessing != dict(preprocessing_abi):
            raise ValueError(
                "runtime RGB preprocessing implementation differs from checkpoint ABI: "
                f"saved={dict(preprocessing_abi)} current={actual_preprocessing}"
            )
        dino_abi = _mapping(observation_abi["dinov2"], name="observation.dinov2")
        encoder_source = (
            str(dino_abi["model"]) if dinov2_model is None else str(dinov2_model)
        )
        self.encoder = DinoV2OnlineEncoder(
            encoder_source,
            device=device,
            compute_dtype=dtype,
            expected_patches=int(dino_abi["patches_per_camera"]),
            expected_width=int(dino_abi["token_width"]),
            reference_batch_size=int(dino_abi["reference_batch_size"]),
            local_files_only=dinov2_local_files_only,
        )
        runtime_identity = self.encoder.identity()
        if str(runtime_identity.get("compute_dtype")) != str(dino_abi["compute_dtype"]):
            raise ValueError(
                "runtime DINO compute dtype differs from checkpoint ABI: "
                f"saved={dino_abi['compute_dtype']} current={runtime_identity.get('compute_dtype')}"
            )
        if int(runtime_identity.get("reference_batch_size", 0)) != int(
            dino_abi["reference_batch_size"]
        ):
            raise ValueError("runtime DINO reference batch size differs from checkpoint ABI")
        self.device = device
        self._seed = int(seed)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(self._seed)
        self._last_input_shapes: dict[str, object] | None = None
        self._last_action_audit: dict[str, object] | None = None

    def reset(self) -> None:
        self._generator.manual_seed(self._seed)
        self._last_input_shapes = None
        self._last_action_audit = None

    def _normal(self, value: np.ndarray, *, action: bool) -> torch.Tensor:
        normalizer = (
            self.bundle.action_normalizer if action else self.bundle.state_normalizer
        )
        encoded = normalizer.encode(np.asarray(value, dtype=np.float32))
        return torch.from_numpy(np.ascontiguousarray(encoded)).to(
            device=self.device,
            dtype=torch.float32,
        )

    def _goal(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        bank = self.bundle.language
        normalized = normalize_instruction(instruction)
        if not bank.is_instruction_bank:
            return bank.tokens[:1], bank.mask[:1]
        index = {value: row for row, value in enumerate(bank.instructions)}
        try:
            row = index[normalized]
        except KeyError as error:
            raise KeyError(
                f"rollout instruction is absent from checkpoint language bank: {normalized!r}"
            ) from error
        return bank.tokens[row : row + 1], bank.mask[row : row + 1]

    def deployment_health(self) -> dict[str, object]:
        abi = self.bundle.deployment_abi
        action_abi = _mapping(abi["action"], name="action")
        observation_abi = _mapping(abi["observation"], name="observation")
        normalizer_abi = _mapping(abi["normalizers"], name="normalizers")
        language_abi = _mapping(abi["language"], name="language")
        return {
            "checkpoint": {
                "path": str(self.bundle.checkpoint_path),
                "sha256": self.bundle.checkpoint_sha256,
                "epoch": int(self.bundle.epoch),
                "global_step": int(self.bundle.global_step),
                "git_commit": self.bundle.identity.git_commit,
                "source_digest": self.bundle.identity.source.digest,
            },
            "architecture": {
                "manifest": self.bundle.identity.manifest,
                "manifest_digest": self.bundle.identity.manifest_digest,
                "graph_config_sha256": abi["graph_config_sha256"],
                "flow_schedule": abi.get("flow_schedule"),
                "flow_schedule_sha256": abi.get("flow_schedule_sha256"),
            },
            "observation": {
                **dict(observation_abi),
                "dinov2_runtime": self.encoder.identity(),
                "last_input_shapes": self._last_input_shapes,
            },
            "action": {
                **dict(action_abi),
                "normalizers": dict(normalizer_abi),
                "raw_min": self.bundle.action_normalizer.minimum.reshape(-1).tolist(),
                "raw_max": self.bundle.action_normalizer.maximum.reshape(-1).tolist(),
                "last_prediction": self._last_action_audit,
            },
            "language": {
                **dict(language_abi),
                "instruction_rows": len(self.bundle.language.instructions),
            },
            "sampling": {
                "policy_seed": int(self._seed),
                "reset_to_policy_seed_each_episode": True,
            },
            "device": str(self.device),
        }

    @torch.no_grad()
    def act(self, history: HistorySnapshot, instruction: str) -> np.ndarray:
        return self.act_with_input(history, instruction)[0]

    @torch.no_grad()
    def act_with_input(
        self, history: HistorySnapshot, instruction: str
    ) -> tuple[np.ndarray, OnlinePolicyInput]:
        """Expose the already-encoded causal input to an external frozen reader.

        The ordinary action path is unchanged. No extra DINO/online encode,
        Teacher, reward, simulator object pose or future target is introduced.
        """
        history.validate()
        if not instruction.strip():
            raise ValueError("ClearVLA rollout instruction must be non-empty")
        config = self.bundle.config
        goal_tokens, goal_mask = self._goal(instruction)
        dino, images = self.encoder.encode(history.rgb_history, self.preprocessing)
        raw_rgb = (
            torch.from_numpy(np.ascontiguousarray(images))
            .permute(0, 1, 4, 2, 3)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
            .div_(255.0)
        )
        action_state = self._normal(history.action_state[None], action=True)
        executed_action_history = self._normal(
            history.executed_action_history[None],
            action=True,
        )
        profile = resolve_action_state_profile(config.data.data_profile)
        codec_gripper_boundary = action_state[:, -1:]
        if profile.gripper_transition_boundary == "previous_command":
            codec_gripper_boundary = executed_action_history[:, -1, -1:]
        online = OnlinePolicyInput(
            observation=CurrentObservation(
                dino_history=dino.unsqueeze(0),
                raw_rgb=raw_rgb,
            ),
            history=ObservableHistory(
                state=self._normal(history.state[None], action=False),
                action_state=action_state,
                codec_gripper_boundary=codec_gripper_boundary,
                state_history=self._normal(history.state_history[None], action=False),
                executed_action_history=executed_action_history,
            ),
            goal=GoalCondition(
                tokens=goal_tokens.to(self.device),
                mask=goal_mask.to(self.device),
            ),
        )
        online.validate(config)
        sampled = sample_action(
            self.bundle.model,
            online,
            config,
            generator=self._generator,
        )
        normalized = sampled.action[0].float().cpu().numpy()
        raw = self.bundle.action_normalizer.decode(normalized).astype(np.float32)
        expected = (config.dimensions.action_horizon, config.dimensions.action_dim)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(f"ClearVLA decoded action must be finite {expected}")
        output_mode = str(config.bottom.gripper_output_mode)
        if is_binary_gripper_mode(output_mode):
            command = sampled.gripper_command
            if command is None:
                raise RuntimeError(
                    "binary gripper deployment returned no command-state sample"
                )
            command_np = command[0].detach().float().cpu().numpy()
            if command_np.shape != (config.dimensions.action_horizon,):
                raise ValueError("binary command-state sample has the wrong horizon")
            if not np.isfinite(command_np).all() or not np.isin(
                command_np, (-1.0, 1.0)
            ).all():
                raise ValueError("binary command-state sample must contain only {-1,+1}")
            # The action normalizer is still used for the arm chart, but a
            # command-state output must not be interpreted as a normalized
            # continuous value.  Write the native raw command explicitly.
            raw[:, -1] = command_np
        elif output_mode not in VALID_GRIPPER_OUTPUT_MODES:
            raise ValueError(f"unknown gripper output mode {output_mode!r}")
        self._last_input_shapes = {
            "native": {
                camera: list(np.asarray(history.rgb_history[camera]).shape[1:])
                for camera in history.rgb_history
            },
            "preprocessed": list(images.shape),
            "dino": list(dino.shape),
            "camera_order": list(history.rgb_history),
        }
        self._last_action_audit = {
            "gripper_output_mode": output_mode,
            "normalized_abs_max": np.abs(normalized).max(axis=0).tolist(),
            "raw_abs_max": np.abs(raw).max(axis=0).tolist(),
            "raw_first": raw[0].tolist(),
        }
        if is_binary_gripper_mode(output_mode):
            self._last_action_audit["gripper_command"] = raw[:, -1].tolist()
        return raw, online


__all__ = ["ClearVLACheckpointPolicy"]
