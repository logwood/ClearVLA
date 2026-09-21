#!/usr/bin/env python3
"""Serve a frozen CALVIN checkpoint with a reset-bound terminal-bias ablation.

The checkpoint and model are loaded once.  Before each episode reset, the
requested mode is read from a small text file and the terminal physical
velocity head is either restored byte-for-byte or has only its affine bias
parameters zeroed in memory.  No checkpoint is written.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from clearvla.benchmarks.bridge import PolicyBridge, serve_policy_bridge
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy


BASELINE = "baseline"
ZERO_TERMINAL_BIAS = "zero_terminal_bias"
VALID_MODES = (BASELINE, ZERO_TERMINAL_BIAS)


class TerminalBiasProbePolicy:
    """Apply a frozen physical-head bias intervention at episode reset only."""

    def __init__(self, policy: ClearVLACheckpointPolicy, *, mode_file: Path) -> None:
        self.policy = policy
        self.mode_file = Path(mode_file)
        velocity_head = (
            policy.bundle.model.execution_bottom.decoder.terminal_controller.velocity_head
        )
        parameters: dict[str, nn.Parameter] = {}
        norm = getattr(velocity_head, "norm", None)
        norm_bias = getattr(norm, "bias", None)
        if isinstance(norm_bias, nn.Parameter):
            parameters["norm.bias"] = norm_bias
        output_layers = getattr(velocity_head, "output_layers", None)
        if not callable(output_layers):
            raise TypeError("terminal velocity head has no output_layers() contract")
        for index, layer in enumerate(output_layers()):
            bias = getattr(layer, "bias", None)
            if isinstance(bias, nn.Parameter):
                parameters[f"output_layers.{index}.bias"] = bias
        if not parameters:
            raise ValueError("terminal velocity head exposes no affine bias parameters")
        self._parameters = parameters
        self._baseline = {
            name: value.detach().clone() for name, value in parameters.items()
        }
        self._active_mode = BASELINE
        self._episode_mode = BASELINE
        self._apply(BASELINE)

    def _requested_mode(self) -> str:
        try:
            mode = self.mode_file.read_text(encoding="ascii").strip()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"CALVIN terminal-bias mode file is missing: {self.mode_file}"
            ) from error
        if mode not in VALID_MODES:
            raise ValueError(
                f"CALVIN terminal-bias mode must be one of {VALID_MODES}, got {mode!r}"
            )
        return mode

    @torch.no_grad()
    def _apply(self, mode: str) -> None:
        for name, parameter in self._parameters.items():
            if mode == ZERO_TERMINAL_BIAS:
                parameter.zero_()
            else:
                parameter.copy_(self._baseline[name])
        self._active_mode = mode

    def reset(self) -> None:
        requested = self._requested_mode()
        self._apply(requested)
        self._episode_mode = requested
        self.policy.reset()

    def act(self, history: Any, instruction: str):
        if self._active_mode != self._episode_mode:
            raise RuntimeError("terminal-bias intervention changed inside an episode")
        return self.policy.act(history, instruction)

    def deployment_health(self) -> dict[str, object]:
        health = dict(self.policy.deployment_health())
        requested: str | None
        try:
            requested = self._requested_mode()
        except (FileNotFoundError, ValueError):
            requested = None
        tensor_rows: dict[str, object] = {}
        for name, baseline in self._baseline.items():
            baseline_cpu = baseline.detach().float().cpu().contiguous()
            current = self._parameters[name].detach().float().cpu()
            tensor_rows[name] = {
                "shape": list(baseline.shape),
                "baseline_l2": float(baseline_cpu.norm()),
                "active_l2": float(current.norm()),
                "baseline_sha256": hashlib.sha256(
                    baseline_cpu.numpy().tobytes()
                ).hexdigest(),
            }
        health["diagnostic_intervention"] = {
            "contract": "calvin_terminal_velocity_affine_bias_ablation_v1",
            "active_mode": self._active_mode,
            "episode_mode": self._episode_mode,
            "requested_mode": requested,
            "activation_boundary": "policy_reset_only",
            "checkpoint_written": False,
            "parameters": tensor_rows,
        }
        return health


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode-file", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18772)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    base = ClearVLACheckpointPolicy(
        args.checkpoint,
        device=device,
        t5_condition=args.t5_condition,
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=bool(args.dinov2_local_files_only),
        seed=int(args.seed),
    )
    policy = TerminalBiasProbePolicy(base, mode_file=args.mode_file)
    serve_policy_bridge(PolicyBridge(policy), host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()

