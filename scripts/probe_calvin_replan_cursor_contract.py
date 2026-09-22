"""Pure local contract probe for the CALVIN chunk/replan state machine.

This probe does not load a checkpoint, contact a bridge, or claim that a
remote policy consumes a global phase.  It verifies only the evaluator-side
protocol: execute_rows controls chunk retention, every plan records its
cumulative origin step, and observe-only calls fill the rows between plans.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Keep this standalone probe runnable from a clean checkout without installing
# the package or importing a model runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clearvla.benchmarks.calvin_eval as calvin_eval


@dataclass
class FakeClient:
    horizon: int = 24

    def __post_init__(self) -> None:
        self.act_calls = 0
        self.observe_calls = 0

    def act(self, observation, instruction: str, *, reset: bool) -> np.ndarray:
        del observation, instruction, reset
        self.act_calls += 1
        return np.zeros((self.horizon, 7), dtype=np.float32)

    def observe(self, observation) -> int:
        del observation
        self.observe_calls += 1
        return self.observe_calls


class FakeEnv:
    def __init__(self) -> None:
        self.steps = 0

    def observation(self) -> dict[str, int]:
        return {"step": self.steps}

    def step(self, action: np.ndarray) -> dict[str, int]:
        if tuple(action.shape) != (7,):
            raise AssertionError(f"fake environment received {action.shape}")
        self.steps += 1
        return self.observation()


def _run(*, execute_rows: int, steps: int) -> dict[str, object]:
    client = FakeClient()
    env = FakeEnv()
    model = calvin_eval.CalvinBridgeModel(client, execute_rows=execute_rows)
    # The real conversion validates simulator images/state.  The state-machine
    # contract is independent of those arrays, so keep this probe model-free.
    original_policy_observation = calvin_eval.calvin_policy_observation
    calvin_eval.calvin_policy_observation = lambda observation, previous: (  # type: ignore[assignment]
        observation,
        previous,
    )
    try:
        observation = env.observation()
        for _ in range(steps):
            action = model.step(observation, "probe")
            observation = env.step(action)
    finally:
        calvin_eval.calvin_policy_observation = original_policy_observation

    arrays = model.action_arrays()
    plan_count = (steps + execute_rows - 1) // execute_rows
    expected_origins = list(range(0, steps, execute_rows))
    expected_rows = [index % execute_rows for index in range(steps)]
    if client.act_calls != plan_count:
        raise AssertionError((execute_rows, "act_calls", client.act_calls, plan_count))
    if client.observe_calls != steps - plan_count:
        raise AssertionError(
            (execute_rows, "observe_calls", client.observe_calls, steps - plan_count)
        )
    if arrays["raw_chunks"].shape != (plan_count, 24, 7):
        raise AssertionError((execute_rows, "raw_chunks", arrays["raw_chunks"].shape))
    if arrays["executed"].shape != (steps, 7):
        raise AssertionError((execute_rows, "executed", arrays["executed"].shape))
    if arrays["plan_origin_step"].tolist() != expected_origins:
        raise AssertionError((execute_rows, "origins", arrays["plan_origin_step"].tolist()))
    if arrays["executed_chunk_row"].tolist() != expected_rows:
        raise AssertionError((execute_rows, "rows", arrays["executed_chunk_row"].tolist()))
    if arrays["executed_plan_origin_step"].tolist() != [
        origin for origin in expected_origins for _ in range(min(execute_rows, steps - origin))
    ]:
        raise AssertionError((execute_rows, "executed_origins"))
    return {
        "execute_rows": execute_rows,
        "steps": steps,
        "plan_count": plan_count,
        "act_calls": client.act_calls,
        "observe_calls": client.observe_calls,
        "plan_origin_steps": arrays["plan_origin_step"].tolist(),
        "executed_chunk_rows": arrays["executed_chunk_row"].tolist(),
        "raw_chunks_shape": list(arrays["raw_chunks"].shape),
        "executed_shape": list(arrays["executed"].shape),
    }


def main() -> None:
    reports = [_run(execute_rows=1, steps=25)]
    reports.append(_run(execute_rows=8, steps=25))
    reports.append(_run(execute_rows=24, steps=49))
    print(json.dumps({"schema": "calvin-replan-cursor-probe-v1", "reports": reports}))


if __name__ == "__main__":
    main()
