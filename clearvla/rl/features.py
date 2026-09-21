"""Frozen observation sidecar; no hooks into G/S/W/P and no second encode."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from clearvla.simulation.admission import STACKCUBE_INSTRUCTION, STACKCUBE_REPAIRED_PROFILE
from clearvla.mainline.gripper_contract import (
    CONTINUOUS_GRIPPER_OUTPUT_MODE,
    MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
)

FEATURE_SCHEMA = "causal_dino_3x2x2x2_project32_proprio_history_chunk_v1"


@dataclass(frozen=True)
class Decision:
    features: np.ndarray
    base_action: np.ndarray


class FrozenBaseReader:
    """Single-task pilot: the exact instruction and base ABI own task identity.

    An external fixed projection preserves camera/history/2x2 spatial axes
    while bounding replay RAM. It is not claimed to preserve G's K objects or
    P1 precision and never feeds back into the mainline. Its identity is fixed.
    """

    def __init__(self, policy, *, instruction: str = STACKCUBE_INSTRUCTION) -> None:
        config = policy.bundle.config
        if (
            config.data.data_profile != STACKCUBE_REPAIRED_PROFILE
            or config.bottom.arm_flow_mode != "relative_command_adapter"
            or config.bottom.gripper_output_mode
            not in {
                CONTINUOUS_GRIPPER_OUTPUT_MODE,
                MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
            }
        ):
            raise ValueError(
                "RL requires a repaired v2 ManiSkill base; legacy/incompatible checkpoints are not substitutes"
            )
        if instruction != STACKCUBE_INSTRUCTION:
            raise ValueError("v1 supports only the fixed StackCube instruction")
        if (
            not policy.bundle.language.is_instruction_bank
            or instruction not in policy.bundle.language.instructions
        ):
            raise ValueError("StackCube base must carry an instruction-verified language bank")
        self.policy, self.instruction = policy, instruction
        policy.bundle.model.requires_grad_(False).eval()
        policy.encoder.requires_grad_(False).eval()
        width = int(policy.encoder.expected_width)
        generator = torch.Generator(device="cpu").manual_seed(1729)
        self.projection = torch.randn(width, 32, generator=generator) / math.sqrt(width)
        self.projection_sha256 = hashlib.sha256(self.projection.numpy().tobytes()).hexdigest()
        self.feature_dim = 3 * 2 * 2 * 2 * 32 + 7 + 7 + 3 * 7 + 8 * 7 + 24 * 7

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def decide(self, history) -> Decision:
        chunk, online = self.policy.act_with_input(history, self.instruction)
        dino = online.observation.dino_history[0].float()
        frames, cameras, patches, width = dino.shape
        side = math.isqrt(patches)
        if (frames, cameras) != (3, 2) or side * side != patches:
            raise ValueError("RL reader requires the true 3-frame/2-camera square DINO patch chart")
        projected = F.layer_norm(dino, (width,)) @ self.projection.to(dino.device)
        grid = projected.permute(0, 1, 3, 2).reshape(6, 32, side, side)
        vision = F.adaptive_avg_pool2d(grid, (2, 2)).reshape(-1)
        parts = [
            vision,
            online.history.state.flatten(),
            online.history.action_state.flatten(),
            online.history.state_history.flatten(),
            online.history.executed_action_history.flatten(),
            torch.from_numpy(np.clip(chunk, -1, 1)).to(dino.device).flatten(),
        ]
        features = torch.cat(parts).float().cpu().numpy()
        if features.shape != (self.feature_dim,) or not np.isfinite(features).all():
            raise ValueError("RL causal feature shape/finite check failed")
        # The exact stochastic base sample included in features is executed
        # next; never redraw it during the critic update or collector handoff.
        return Decision(features.copy(), np.asarray(chunk[0], np.float32).copy())
