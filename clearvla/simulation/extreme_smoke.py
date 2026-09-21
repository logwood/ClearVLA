"""Exercise simulator action boundaries and optionally show policy RGB over X11."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .contracts import ACTION_DIM, PolicyObservation, SimulationEnvironment
from .dataset import EpisodeRecorder, audit_simulation_dataset


@dataclass(frozen=True)
class ExtremeCase:
    name: str
    requested_action: np.ndarray
    seed: int


def _extreme_cases(
    low: np.ndarray,
    high: np.ndarray,
    *,
    base_seed: int,
) -> tuple[ExtremeCase, ...]:
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
        raise ValueError("extreme suite requires [7] action bounds")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
        raise ValueError("extreme suite action bounds must be finite and ordered")
    center = (low + high) * 0.5
    span = high - low
    alternating_low = np.where(np.arange(ACTION_DIM) % 2 == 0, low, high)
    alternating_high = np.where(np.arange(ACTION_DIM) % 2 == 0, high, low)
    gripper_low = center.copy()
    gripper_low[-1] = low[-1]
    gripper_high = center.copy()
    gripper_high[-1] = high[-1]
    seeds = [int(base_seed) + index for index in range(6)]
    seeds.append(np.iinfo(np.int32).max)
    return (
        ExtremeCase("below_bounds_clipped", low - span, seeds[0]),
        ExtremeCase("above_bounds_clipped", high + span, seeds[1]),
        ExtremeCase("alternating_low_high", alternating_low, seeds[2]),
        ExtremeCase("alternating_high_low", alternating_high, seeds[3]),
        ExtremeCase("gripper_low_arm_center", gripper_low, seeds[4]),
        ExtremeCase("gripper_high_arm_center", gripper_high, seeds[5]),
        ExtremeCase("max_seed_center", center, seeds[6]),
    )


def _panel(observation: PolicyObservation, label: str) -> np.ndarray:
    observation.validate()
    top = Image.fromarray(observation.rgb["top"])
    wrist = Image.fromarray(observation.rgb["wrist"])
    if top.size != wrist.size:
        raise ValueError("extreme display requires equal policy camera sizes")
    header = 42
    panel = Image.new("RGB", (top.width + wrist.width, top.height + header), "black")
    panel.paste(top, (0, header))
    panel.paste(wrist, (top.width, header))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), f"{label} | top", fill="white")
    draw.text((top.width + 8, 6), "wrist", fill="white")
    return np.asarray(panel, dtype=np.uint8)


def _image_stats(observation: PolicyObservation) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for name, image in observation.rgb.items():
        value = np.asarray(image)
        stats[name] = {
            "minimum": int(value.min()),
            "maximum": int(value.max()),
            "mean": float(value.mean(dtype=np.float64)),
            "std": float(value.std(dtype=np.float64)),
        }
        if stats[name]["std"] == 0.0:
            raise ValueError(f"{name} camera collapsed to a constant image")
    return stats


class _Display:
    def __init__(self, enabled: bool, *, wait_ms: int) -> None:
        self.enabled = bool(enabled)
        self.wait_ms = int(wait_ms)
        self._cv2: Any = None
        if self.wait_ms < 1:
            raise ValueError("display wait must be positive")
        if not self.enabled:
            return
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("--display requires an SSH-forwarded DISPLAY")
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - optional GUI dependency
            raise RuntimeError("--display requires OpenCV from simulation-maniskill") from error
        self._cv2 = cv2
        cv2.namedWindow("ClearVLA extreme simulation", cv2.WINDOW_NORMAL)

    def show(self, panel: np.ndarray) -> None:
        if self._cv2 is None:
            return
        self._cv2.imshow("ClearVLA extreme simulation", panel[..., ::-1])
        self._cv2.waitKey(self.wait_ms)

    def close(self) -> None:
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()


def _environment(args: argparse.Namespace) -> SimulationEnvironment:
    if args.environment == "alicia-proxy":
        from .alicia_proxy import AliciaProxyEnv

        return AliciaProxyEnv(
            image_size=args.image_size,
            control_hz=args.control_hz,
            max_episode_steps=args.max_episode_steps,
        )
    from .maniskill_adapter import ManiSkillStackCubeEnv

    return ManiSkillStackCubeEnv(
        image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        sim_backend=args.maniskill_sim_backend,
        render_backend=args.maniskill_render_backend,
    )


def run_extreme_suite(
    environment: SimulationEnvironment,
    *,
    output_dir: Path,
    base_seed: int,
    steps_per_case: int,
    display: bool,
    display_ms: int,
) -> dict[str, Any]:
    if steps_per_case <= 0:
        raise ValueError("steps_per_case must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    low, high = environment.action_bounds()
    cases = _extreme_cases(low, high, base_seed=base_seed)
    rejected: dict[str, bool] = {}
    malformed = {
        "nan": np.full(ACTION_DIM, np.nan, dtype=np.float32),
        "positive_infinity": np.full(ACTION_DIM, np.inf, dtype=np.float32),
        "wrong_shape": np.zeros(ACTION_DIM - 1, dtype=np.float32),
    }
    for name, value in malformed.items():
        try:
            environment.clip_action(value)
        except ValueError:
            rejected[name] = True
        else:
            raise AssertionError(f"environment accepted malformed action case {name}")

    viewer = _Display(display, wait_ms=display_ms)
    rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    try:
        for case in cases:
            reset = environment.reset(seed=case.seed)
            observation = reset.observation
            command = environment.clip_action(case.requested_action)
            recorder = EpisodeRecorder(
                output_dir,
                descriptor=environment.descriptor,
                episode_id=case.name,
                seed=case.seed,
                instruction=f"diagnostic action-boundary case: {case.name}",
                collector=f"ExtremeActionSuite:{case.name}",
            )
            rewards = 0.0
            final_metrics = dict(reset.evaluation.metrics)
            executed = 0
            for _ in range(int(steps_per_case)):
                result = environment.step(command)
                if not np.allclose(result.observation.action_state, command, rtol=0.0, atol=1e-6):
                    raise AssertionError("post-step action_state differs from executed command")
                recorder.append(observation, command, result)
                observation = result.observation
                rewards += float(result.reward)
                final_metrics = dict(result.evaluation.metrics)
                executed += 1
                panel = _panel(observation, case.name)
                viewer.show(panel)
                if result.terminated or result.truncated:
                    break
            artifact = recorder.finish()
            panel = _panel(observation, case.name)
            panel_path = output_dir / f"{case.name}.png"
            Image.fromarray(panel).save(panel_path)
            panels.append(panel)
            rows.append(
                {
                    "name": case.name,
                    "seed": case.seed,
                    "requested_action": case.requested_action.tolist(),
                    "executed_action": command.tolist(),
                    "steps": executed,
                    "reward_sum": rewards,
                    "final_evaluator_metrics": final_metrics,
                    "image_stats": _image_stats(observation),
                    "episode": str(artifact.path.resolve()),
                    "panel": str(panel_path.resolve()),
                }
            )
    finally:
        viewer.close()

    cell_height = max(panel.shape[0] for panel in panels)
    cell_width = max(panel.shape[1] for panel in panels)
    sheet = Image.new("RGB", (cell_width * 2, cell_height * 4), "black")
    for index, panel in enumerate(panels):
        sheet.paste(Image.fromarray(panel), ((index % 2) * cell_width, (index // 2) * cell_height))
    contact_sheet = output_dir / "contact_sheet.png"
    sheet.save(contact_sheet)
    audit = audit_simulation_dataset(output_dir)
    payload = {
        "schema": "clearvla-simulation-extreme-smoke-v1",
        "environment": environment.descriptor.to_dict(),
        "action_bounds": {"low": low.tolist(), "high": high.tolist()},
        "malformed_actions_rejected": rejected,
        "cases": rows,
        "dataset_audit": audit,
        "contact_sheet": str(contact_sheet.resolve()),
        "x11_display": os.environ.get("DISPLAY") if display else None,
    }
    (output_dir / "extreme_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=("alicia-proxy", "maniskill-stackcube"),
        default="maniskill-stackcube",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--steps-per-case", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-ms", type=int, default=600)
    parser.add_argument(
        "--maniskill-sim-backend",
        default="physx_cpu",
        choices=("physx_cpu", "physx_cuda"),
    )
    parser.add_argument(
        "--maniskill-render-backend",
        default="gpu",
        choices=("gpu", "cpu"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = _environment(args)
    try:
        result = run_extreme_suite(
            environment,
            output_dir=args.output_dir,
            base_seed=args.base_seed,
            steps_per_case=args.steps_per_case,
            display=args.display,
            display_ms=args.display_ms,
        )
    finally:
        environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


