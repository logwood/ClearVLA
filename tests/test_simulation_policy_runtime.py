from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from clearvla.benchmarks.bridge import policy_observation
from clearvla.mainline.config import ExperimentConfig
from clearvla.simulation.checkpoint import _preflight_model_state, _validated_model_state
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.vision import preprocess_rgb_history
from clearvla.vision.preprocessing import PreprocessConfig


def _observation(step: int, action_state: np.ndarray):
    return policy_observation(
        top=np.full((32, 32, 3), step, dtype=np.uint8),
        wrist=np.full((32, 32, 3), step + 1, dtype=np.uint8),
        state=np.full(7, step, dtype=np.float32),
        action_state=action_state,
    )


def _libero_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(base.data, data_profile="libero_relative_7d_v1"),
        bottom=replace(base.bottom, arm_flow_mode="relative_command_direct"),
    )
    config.validate()
    return config


class _IdentityNormalizer:
    mode = "zscore"
    minimum = np.full((1, 7), -1.0, dtype=np.float32)
    maximum = np.full((1, 7), 1.0, dtype=np.float32)

    def encode(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)

    def decode(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)


class _Encoder:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def encode(self, rgb_history, preprocessing):
        del rgb_history, preprocessing
        dims = self.config.dimensions
        dino = torch.zeros(
            dims.visual_history_length,
            dims.num_cameras,
            dims.patches_per_camera,
            dims.visual_token_dim,
            dtype=torch.float16,
        )
        images = np.zeros(
            (
                dims.visual_history_length,
                dims.num_cameras,
                32,
                32,
                3,
            ),
            dtype=np.uint8,
        )
        return dino, images

    def identity(self) -> dict[str, object]:
        return {"model": "fake", "compute_dtype": "fp32", "reference_batch_size": 1}


def _policy() -> ClearVLACheckpointPolicy:
    config = _libero_config()
    policy = object.__new__(ClearVLACheckpointPolicy)
    normalizer = _IdentityNormalizer()
    language = SimpleNamespace(
        is_instruction_bank=True,
        instructions=("move",),
        tokens=torch.zeros(1, 1, config.dimensions.goal_token_dim),
        mask=torch.ones(1, 1, dtype=torch.bool),
    )
    policy.bundle = SimpleNamespace(
        config=config,
        model=object(),
        action_normalizer=normalizer,
        state_normalizer=normalizer,
        language=language,
        deployment_abi={
            "graph_config_sha256": "a" * 64,
            "observation": {"camera_names": ["top", "wrist"]},
            "action": {"arm_flow_mode": "relative_command_direct"},
            "normalizers": {
                "mode": "zscore",
                "action_sha256": "b" * 64,
                "state_sha256": "c" * 64,
            },
            "language": {"sha256": "d" * 64},
        },
        checkpoint_path="checkpoint.pt",
        checkpoint_sha256="e" * 64,
        epoch=3,
        global_step=7,
        identity=SimpleNamespace(
            git_commit="f" * 40,
            source=SimpleNamespace(digest="1" * 64),
            manifest={"schema": 30},
            manifest_digest="2" * 64,
        ),
    )
    policy.preprocessing = PreprocessConfig(resize_hw=(32, 32))
    policy.encoder = _Encoder(config)
    policy.device = torch.device("cpu")
    policy._seed = 0
    policy._generator = torch.Generator(device="cpu")
    policy._generator.manual_seed(0)
    policy._last_input_shapes = None
    policy._last_action_audit = None
    return policy


def test_online_preprocessing_preserves_camera_identity_and_one_shared_shape() -> None:
    rgb = {
        "top": np.zeros((3, 8, 10, 3), dtype=np.uint8),
        "wrist": np.full((3, 4, 6, 3), 17, dtype=np.uint8),
    }
    output = preprocess_rgb_history(rgb, PreprocessConfig(resize_hw=(32, 48)))
    assert output.shape == (3, 2, 32, 48, 3)
    assert np.all(output[:, 0] == 0)
    assert np.all(output[:, 1] == 17)
    assert rgb["top"].shape == (3, 8, 10, 3)
    assert rgb["wrist"].shape == (3, 4, 6, 3)

    with pytest.raises(ValueError, match="shared model-side shape"):
        preprocess_rgb_history(rgb, PreprocessConfig())


def test_libero_policy_uses_executed_command_for_gripper_codec_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    history = CausalHistory()
    reset = _observation(0, np.zeros(7, dtype=np.float32))
    history.reset(reset)
    executed = np.zeros(7, dtype=np.float32)
    executed[-1] = 0.75
    current_action_state = np.zeros(7, dtype=np.float32)
    current_action_state[-1] = -0.5
    history.append(executed, _observation(1, current_action_state))

    def fake_sample_action(model, online, config, *, generator):
        del model, online, generator
        return SimpleNamespace(
            action=torch.zeros(
                1,
                config.dimensions.action_horizon,
                config.dimensions.action_dim,
            ),
            gripper_command=None,
        )

    monkeypatch.setattr(
        "clearvla.simulation.clearvla_policy.sample_action",
        fake_sample_action,
    )
    action, online = policy.act_with_input(history.snapshot(), "move")
    assert action.shape == (24, 7)
    assert online.history.action_state[0, -1].item() == pytest.approx(-0.5)
    assert online.history.executed_action_history[0, -1, -1].item() == pytest.approx(
        0.75
    )
    assert online.history.codec_gripper_boundary[0, 0].item() == pytest.approx(0.75)


def test_deployment_health_nests_normalizer_identity_for_evaluators() -> None:
    policy = _policy()
    health = policy.deployment_health()
    assert health["action"]["normalizers"] == {
        "mode": "zscore",
        "action_sha256": "b" * 64,
        "state_sha256": "c" * 64,
    }
    assert health["checkpoint"]["sha256"] == "e" * 64
    assert health["architecture"]["graph_config_sha256"] == "a" * 64


def test_checkpoint_model_state_preflight_is_exact_and_finite() -> None:
    current = {
        "weight": torch.zeros(2, 3),
        "counter": torch.zeros((), dtype=torch.long),
    }
    saved = {
        "weight": torch.ones(2, 3),
        "counter": torch.ones((), dtype=torch.long),
    }
    assert _preflight_model_state(saved) is saved
    assert _validated_model_state(saved, current) is saved

    with pytest.raises(ValueError, match="non-empty"):
        _preflight_model_state({})
    stale = dict(saved)
    stale["weight"] = torch.full((2, 3), float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        _preflight_model_state(stale)
    with pytest.raises(ValueError, match="ownership differs"):
        _validated_model_state({"weight": saved["weight"]}, current)
