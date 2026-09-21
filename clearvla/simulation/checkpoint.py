"""Strict inference-only restoration through the exported deployment ABI."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import torch
from torch import Tensor

from clearvla.mainline.checkpoint import (
    CheckpointIdentity,
    checkpoint_identity_from_mapping,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.data.language import T5ConditionBank, load_t5_condition_bank
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import CHECKPOINT_SCHEMA
from clearvla.mainline.runtime.deployment import (
    canonical_sha256,
    deployment_config_from_checkpoint,
    validate_deployment_abi,
)


@dataclass(frozen=True)
class DeploymentBundle:
    checkpoint_path: Path
    checkpoint_sha256: str
    config: ExperimentConfig
    identity: CheckpointIdentity
    deployment_abi: Mapping[str, object]
    model: ClearVLAMainlinePolicy
    action_normalizer: ArrayNormalizer
    state_normalizer: ArrayNormalizer
    language: T5ConditionBank
    global_step: int
    epoch: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalizer(value: object, *, name: str, width: int) -> ArrayNormalizer:
    if not isinstance(value, Mapping):
        raise ValueError(f"deployment checkpoint has no {name} normalizer mapping")
    try:
        normalizer = ArrayNormalizer.from_dict(dict(value))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"deployment {name} normalizer is malformed") from error
    arrays = (
        normalizer.offset,
        normalizer.scale,
        normalizer.mean,
        normalizer.std,
        normalizer.minimum,
        normalizer.maximum,
    )
    if any(array.shape != (1, int(width)) for array in arrays):
        raise ValueError(f"deployment {name} normalizer must own [1,{width}] rows")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"deployment {name} normalizer contains NaN or infinity")
    if np.any(normalizer.scale == 0.0):
        raise ValueError(f"deployment {name} normalizer has a zero scale")
    if normalizer.mode != "zscore":
        raise ValueError(f"deployment {name} normalizer must use zscore")
    return normalizer


def _validated_model_state(
    saved: object,
    current: Mapping[str, Tensor],
) -> Mapping[str, Tensor]:
    if not isinstance(saved, Mapping):
        raise ValueError("deployment checkpoint has no model state mapping")
    if set(saved) != set(current):
        missing = sorted(set(current).difference(saved))
        extra = sorted(set(saved).difference(current))
        raise ValueError(
            f"deployment model ownership differs: missing={missing[:8]} extra={extra[:8]}"
        )
    typed = cast(Mapping[str, object], saved)
    for name, target in current.items():
        value = typed[name]
        if not isinstance(value, Tensor):
            raise ValueError(f"deployment model state {name!r} is not a tensor")
        if value.shape != target.shape:
            raise ValueError(
                f"deployment model state {name!r} shape {tuple(value.shape)} "
                f"!= {tuple(target.shape)}"
            )
        if value.dtype != target.dtype:
            raise ValueError(
                f"deployment model state {name!r} dtype {value.dtype} != {target.dtype}"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"deployment model state {name!r} is non-finite")
    return cast(Mapping[str, Tensor], saved)


def _preflight_model_state(saved: object) -> Mapping[str, Tensor]:
    """Reject an obviously unusable payload before allocating the live model."""

    if not isinstance(saved, Mapping) or not saved:
        raise ValueError("deployment checkpoint has no non-empty model state mapping")
    for name, value in saved.items():
        if not isinstance(name, str) or not name:
            raise ValueError("deployment model state keys must be non-empty strings")
        if not isinstance(value, Tensor):
            raise ValueError(f"deployment model state {name!r} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"deployment model state {name!r} is non-finite")
    return cast(Mapping[str, Tensor], saved)


def load_deployment_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    t5_condition: str | Path | None = None,
) -> DeploymentBundle:
    """Restore inference state after validating every exported boundary."""

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("deployment checkpoint payload must be a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"deployment requires {CHECKPOINT_SCHEMA}, got {payload.get('schema')!r}"
        )
    raw_config = payload.get("config")
    raw_identity = payload.get("identity")
    data_state = payload.get("data_state")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_identity, Mapping):
        raise ValueError("deployment checkpoint lacks typed config/identity")
    if not isinstance(data_state, Mapping):
        raise ValueError("deployment checkpoint lacks serialized data_state")
    abi = validate_deployment_abi(data_state.get("deployment_abi"))
    config = deployment_config_from_checkpoint(raw_config, abi)
    identity = checkpoint_identity_from_mapping(raw_identity)
    identity.validate()
    if str(abi.get("source_config_digest", "")) != identity.config_digest:
        raise ValueError("deployment ABI source config digest differs from checkpoint identity")
    if dict(cast(Mapping[str, object], abi["architecture_manifest"])) != identity.manifest:
        raise ValueError("deployment ABI architecture manifest differs from checkpoint identity")

    dims = config.dimensions
    action_normalizer = _normalizer(
        data_state.get("action_normalizer"),
        name="action",
        width=dims.action_dim,
    )
    state_normalizer = _normalizer(
        data_state.get("state_normalizer"),
        name="state",
        width=dims.state_dim,
    )
    normalizer_abi = cast(Mapping[str, object], abi["normalizers"])
    action_digest = canonical_sha256(action_normalizer.to_dict())
    state_digest = canonical_sha256(state_normalizer.to_dict())
    if action_digest != str(normalizer_abi.get("action_sha256", "")):
        raise ValueError("deployment action normalizer differs from its ABI")
    if state_digest != str(normalizer_abi.get("state_sha256", "")):
        raise ValueError("deployment state normalizer differs from its ABI")
    if action_digest != identity.dataset.action_normalizer_sha256:
        raise ValueError("deployment action normalizer differs from checkpoint identity")
    if state_digest != identity.dataset.state_normalizer_sha256:
        raise ValueError("deployment state normalizer differs from checkpoint identity")

    try:
        global_step = int(payload["global_step"])
        epoch = int(payload["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("deployment checkpoint training position is malformed") from error
    if global_step < 0 or epoch < 0:
        raise ValueError("deployment checkpoint training position cannot be negative")

    saved_model = _preflight_model_state(payload.get("model"))
    language_path = (
        Path(t5_condition).expanduser()
        if t5_condition is not None
        else Path(identity.language.path).expanduser()
    )
    if not language_path.is_file():
        raise FileNotFoundError(
            f"deployment T5 condition does not exist: {language_path}; "
            "pass an explicit relocated path"
        )
    language_digest = _sha256(language_path)
    if language_digest != identity.language.sha256:
        raise ValueError("deployment T5 condition hash differs from checkpoint identity")
    language_abi = cast(Mapping[str, object], abi["language"])
    if language_digest != str(language_abi.get("sha256", "")):
        raise ValueError("deployment T5 condition hash differs from its ABI")
    language = load_t5_condition_bank(
        language_path,
        max_tokens=dims.goal_max_tokens,
        expected_width=dims.goal_token_dim,
    )

    model = ClearVLAMainlinePolicy(config)
    model.configure_action_normalizer(action_normalizer)
    state = _validated_model_state(saved_model, model.state_dict())
    model.load_state_dict(state, strict=True)
    # Synchronize Python execution caches with the serialized progress buffer.
    model.set_training_step(global_step)
    model.to(device=device).eval()

    return DeploymentBundle(
        checkpoint_path=source,
        checkpoint_sha256=_sha256(source),
        config=config,
        identity=identity,
        deployment_abi=abi,
        model=model,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        language=language,
        global_step=global_step,
        epoch=epoch,
    )


__all__ = [
    "DeploymentBundle",
    "_preflight_model_state",
    "_validated_model_state",
    "load_deployment_checkpoint",
]
