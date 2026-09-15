"""Run an isolated LIBERO gripper sign/hold boundary probe.

This is deliberately a diagnostic wrapper around the canonical LIBERO
evaluator.  It does not change the checkpoint, the shared action codec, or
the bridge.  The policy still emits a continuous normalized seven-dimensional
action; immediately before ``env.step`` this wrapper projects only the last
coordinate to ``{-1, 0, +1}`` (open / hold / close).  The projected row is
also fed back as the next ``action_state`` so the probe measures the complete
closed-loop boundary, not an open-loop actuator substitution.

The evaluator's normal result and video artifacts are retained.  A compact
per-step trace is written beside the result so the arm path and gripper
projection can be audited without another model forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from clearvla.benchmarks import libero_eval as _libero
from clearvla.benchmarks.io import atomic_json


class SignHoldBridgePolicy(_libero.LiberoBridgePolicy):
    """LIBERO bridge adapter with a ternary gripper execution boundary."""

    def __init__(self, *args: Any, threshold: float = 0.1, **kwargs: Any) -> None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
            raise ValueError("sign/hold threshold must be finite in [0,1)")
        self.threshold = threshold
        self.episode_traces: list[list[dict[str, Any]]] = []
        self._trace: list[dict[str, Any]] = []
        self._continuous_executed: list[np.ndarray] = []
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        # ``LiberoBridgePolicy.__init__`` calls reset dynamically, so all trace
        # containers are initialized before the first super call above.
        if self._trace:
            self.episode_traces.append(self._trace)
        self._trace = []
        self._continuous_executed = []
        super().reset()

    def finish_trace(self) -> None:
        if self._trace:
            self.episode_traces.append(self._trace)
            self._trace = []

    def _project_gripper(self, value: float) -> float:
        if value > self.threshold:
            return 1.0
        if value < -self.threshold:
            return -1.0
        # PandaGripper treats zero as no incremental motion, i.e. physical
        # hold.  Keep zero in the action state as the actually executed value.
        return 0.0

    def step(self, observation: Any, instruction: str) -> np.ndarray:
        # Parent step performs the canonical observation projection, bridge
        # call, [-1,1] clipping, and raw-action audit.  Replace only the
        # gripper coordinate after that boundary and keep the arm coordinates
        # byte-for-byte identical.
        continuous = np.asarray(super().step(observation, instruction), dtype=np.float32)
        projected = continuous.copy()
        projected_gripper = self._project_gripper(float(continuous[6]))
        projected[6] = projected_gripper

        self._continuous_executed.append(continuous.copy())
        # Parent has already recorded the row and set the next action state.
        # Update those owned buffers to the command that actually crosses the
        # environment boundary.
        self._executed_actions[-1] = projected.copy()
        self._previous_action = projected.copy()
        self._trace.append(
            {
                "step": len(self._trace) + 1,
                "raw_action": self._raw_actions[-1].astype(np.float32).tolist(),
                "continuous_clipped_action": continuous.tolist(),
                "executed_action": projected.tolist(),
                "projection_changed": bool(projected_gripper != float(continuous[6])),
                "gripper_raw": float(self._raw_actions[-1][6]),
                "gripper_continuous_clipped": float(continuous[6]),
                "gripper_executed": float(projected_gripper),
            }
        )
        return projected

    def action_audit(self) -> dict[str, object]:
        audit = dict(super().action_audit())
        if self._continuous_executed:
            continuous = np.stack(self._continuous_executed).astype(np.float32)
            projected = np.stack(self._executed_actions).astype(np.float32)
            raw = np.stack(self._raw_actions).astype(np.float32)
            # Keep the canonical clipping count (raw proposal versus the
            # continuous [-1,1] boundary) separate from this diagnostic's
            # ternary projection count.
            audit["clipped_row_count"] = int(
                np.any(raw != continuous, axis=1).sum()
            )
            audit["continuous_executed_min"] = continuous.min(axis=0).tolist()
            audit["continuous_executed_max"] = continuous.max(axis=0).tolist()
            changed = np.abs(projected[:, 6] - continuous[:, 6]) > 0.0
            audit["gripper_projection_changed_row_count"] = int(changed.sum())
            audit["gripper_projection_threshold"] = float(self.threshold)
            audit["gripper_projection_mode"] = "sign_hold_feedback"
            audit["gripper_executed_value_counts"] = {
                "-1": int((projected[:, 6] < 0.0).sum()),
                "0": int((projected[:, 6] == 0.0).sum()),
                "+1": int((projected[:, 6] > 0.0).sum()),
            }
            audit["gripper_projection_transition_count"] = int(
                np.count_nonzero(np.diff(projected[:, 6]))
            )
        return audit


def _diagnostic_action_contract(
    threshold: float,
    base_contract: Any,
) -> dict[str, object]:
    contract = dict(base_contract())
    contract.update(
        {
            "diagnostic_execution_projection": "sign_hold_feedback",
            "diagnostic_gripper_threshold": float(threshold),
            "gripper_semantics": (
                "continuous policy command projected to ternary sign/hold at env boundary"
            ),
            "action_state_semantics": (
                "previous ternary projected native action; reset row is zero"
            ),
        }
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--image-side", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    threshold = float(args.threshold)
    original_policy_class = _libero.LiberoBridgePolicy
    original_contract = _libero.libero_action_contract
    holder: dict[str, SignHoldBridgePolicy] = {}

    def make_policy(*policy_args: Any, **policy_kwargs: Any) -> SignHoldBridgePolicy:
        policy = SignHoldBridgePolicy(
            *policy_args,
            threshold=threshold,
            **policy_kwargs,
        )
        holder["policy"] = policy
        return policy

    # ``evaluate_libero`` resolves these globals when it starts the rollout;
    # the substitution is process-local and is restored even on failure.
    _libero.LiberoBridgePolicy = make_policy  # type: ignore[assignment]
    _libero.libero_action_contract = (  # type: ignore[assignment]
        lambda: _diagnostic_action_contract(threshold, original_contract)
    )
    try:
        result = _libero.evaluate_libero(
            args.suite,
            args.output,
            endpoint=args.endpoint,
            timeout=args.timeout,
            task_ids=args.task_ids,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            image_side=args.image_side,
            resume=False,
            record_video=True,
            video_dir=args.video_dir,
            video_fps=10.0,
            video_episodes=args.episodes_per_task,
        )
    finally:
        _libero.LiberoBridgePolicy = original_policy_class  # type: ignore[assignment]
        _libero.libero_action_contract = original_contract  # type: ignore[assignment]

    policy = holder.get("policy")
    if policy is None:
        raise RuntimeError("LIBERO evaluator did not construct the probe policy")
    policy.finish_trace()
    trace_path = args.output.with_suffix(args.output.suffix + ".trace.json")
    trace_payload = {
        "schema": "clearvla-libero-sign-hold-trace-v1",
        "mode": "sign_hold_feedback",
        "threshold": threshold,
        "output": str(args.output),
        "video_dir": str(args.video_dir),
        "episodes": policy.episode_traces,
    }
    atomic_json(trace_path, trace_payload)
    result["diagnostic"] = {
        "mode": "sign_hold_feedback",
        "threshold": threshold,
        "trace": str(trace_path),
        "gripper_mapping": {"negative": "open", "zero": "hold", "positive": "closed"},
        "feedback": "projected action is used as next action_state",
    }
    atomic_json(args.output, result)
    print(json.dumps({"result": str(args.output), "trace": str(trace_path)}, indent=2))


if __name__ == "__main__":
    main()
