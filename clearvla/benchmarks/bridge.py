"""Loopback-only policy bridge between legacy benchmark and ClearVLA envs."""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import numpy as np

from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory


def policy_observation(
    *,
    top: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    action_state: np.ndarray,
) -> PolicyObservation:
    result = PolicyObservation(
        # Preserve both native camera arrays.  Resizing here used to turn the
        # wrist camera into a nearest-neighbour copy at the static-camera
        # resolution and then resize it a second time inside the policy.
        rgb={
            "top": np.ascontiguousarray(np.asarray(top)),
            "wrist": np.ascontiguousarray(np.asarray(wrist)),
        },
        state=np.asarray(state, dtype=np.float32),
        action_state=np.asarray(action_state, dtype=np.float32),
    )
    result.validate()
    return result


@dataclass
class RemotePolicyClient:
    endpoint: str = "http://127.0.0.1:8765"
    timeout: float = 120.0

    @staticmethod
    def _http_error(error: HTTPError) -> RuntimeError:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"message": body.strip()}
        return RuntimeError(
            f"ClearVLA bridge HTTP {error.code}: "
            f"{payload.get('error', 'error')}: {payload.get('message', '')}"
        )

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"{self.endpoint}/health", timeout=self.timeout
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise self._http_error(error) from error
        if not isinstance(payload, dict):
            raise ValueError("bridge health response must be a JSON mapping")
        return payload

    def act(
        self,
        observation: PolicyObservation,
        instruction: str,
        *,
        reset: bool,
        begin_instruction: bool = False,
    ) -> np.ndarray:
        observation.validate()
        body = io.BytesIO()
        np.savez_compressed(
            body,
            top=observation.rgb["top"],
            wrist=observation.rgb["wrist"],
            state=np.asarray(observation.state, dtype=np.float32),
            action_state=np.asarray(observation.action_state, dtype=np.float32),
            reset=np.asarray([bool(reset)], dtype=np.uint8),
            begin_instruction=np.asarray([bool(begin_instruction)], dtype=np.uint8),
        )
        query = urllib.parse.urlencode({"instruction": str(instruction)})
        request = urllib.request.Request(
            f"{self.endpoint}/act?{query}",
            data=body.getvalue(),
            headers={"Content-Type": "application/x-npz"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                action = np.load(io.BytesIO(response.read()), allow_pickle=False)
        except HTTPError as error:
            raise self._http_error(error) from error
        value = np.asarray(action, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 7 or value.shape[0] < 1:
            raise ValueError("remote policy returned an empty or invalid [T,7] action chunk")
        if not np.isfinite(value).all():
            raise ValueError("remote policy returned an invalid [T,7] action chunk")
        return value

    def observe(self, observation: PolicyObservation) -> int:
        """Append one executed transition without asking the policy to plan."""

        observation.validate()
        body = io.BytesIO()
        np.savez_compressed(
            body,
            top=observation.rgb["top"],
            wrist=observation.rgb["wrist"],
            state=np.asarray(observation.state, dtype=np.float32),
            action_state=np.asarray(observation.action_state, dtype=np.float32),
        )
        request = urllib.request.Request(
            f"{self.endpoint}/observe",
            data=body.getvalue(),
            headers={"Content-Type": "application/x-npz"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise self._http_error(error) from error
        if not isinstance(payload, dict):
            raise ValueError("bridge observe response must be a JSON mapping")
        time_index = payload.get("history_time_index")
        if not isinstance(time_index, int) or isinstance(time_index, bool) or time_index < 0:
            raise ValueError("bridge observe response has an invalid history_time_index")
        return time_index


class PolicyBridge:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.history = CausalHistory(executed_world=getattr(policy,"requires_executed_world_history",False))
        self.initialized = False

    def health(self) -> dict[str, Any]:
        describe = getattr(self.policy, "deployment_health", None)
        deployment = describe() if callable(describe) else None
        return {
            "status": "ok",
            "protocol": {
                "version": 3,
                "observe_only": True,
                "instruction_boundary": callable(getattr(self.policy, "begin_instruction", None)),
            },
            "mode": (
                "smoke-zero" if isinstance(self.policy, SmokeZeroPolicy) else "formal-checkpoint"
            ),
            "initialized": bool(self.initialized),
            "history_time_index": (
                int(self.history.time_index) if self.initialized else None
            ),
            "deployment": deployment,
        }

    def act(
        self,
        observation: PolicyObservation,
        instruction: str,
        *,
        reset: bool,
        begin_instruction: bool = False,
    ) -> np.ndarray:
        # Admit the event before mutating physical history. A new instruction
        # may have identical text, and is NOT an environment reset/RNG restart.
        observation.validate()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("policy bridge requires a non-empty instruction")
        start = getattr(self.policy, "begin_instruction", None)
        if begin_instruction and not callable(start):
            raise ValueError("policy does not support an explicit instruction boundary")
        if reset or not self.initialized:
            self.policy.reset()
            self.history.reset(observation, reset_action=observation.action_state)
            self.initialized = True
        else:
            self.history.append(observation.action_state, observation)
        if begin_instruction:
            assert callable(start)  # admitted before any history mutation
            start(instruction)
        action = np.asarray(
            self.policy.act(self.history.snapshot(), instruction), dtype=np.float32
        )
        if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7 or not np.isfinite(action).all():
            raise ValueError("policy bridge requires one finite [T,7] action chunk")
        return action

    def observe(self, observation: PolicyObservation) -> int:
        """Record an executed action/observation pair without policy inference."""

        if not self.initialized:
            raise RuntimeError("policy bridge must be initialized before observe")
        self.history.append(observation.action_state, observation)
        return int(self.history.time_index)


class SmokeZeroPolicy:
    """Explicit no-skill policy for transport and simulator smoke tests only."""

    def __init__(self, horizon: int = 24) -> None:
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("smoke policy horizon must be positive")

    def reset(self) -> None:
        return

    def begin_instruction(self, instruction: str) -> None:
        if not str(instruction).strip():
            raise ValueError("smoke bridge still requires a real instruction")

    def act(self, history, instruction: str) -> np.ndarray:
        history.validate()
        if not str(instruction).strip():
            raise ValueError("smoke bridge still requires a real instruction")
        return np.zeros((self.horizon, 7), dtype=np.float32)


def _event_flag(value: np.ndarray, *, name: str) -> bool:
    """Explicit wire events: reject ambiguous shapes, NaNs and truthy strings."""
    array = np.asarray(value)
    if array.shape != (1,) or array.dtype not in (np.dtype("uint8"), np.dtype("bool")):
        raise ValueError(f"bridge {name} must be a single bool/uint8 flag")
    if int(array[0]) not in (0, 1):
        raise ValueError(f"bridge {name} must be zero or one")
    return bool(array[0])


def serve_policy_bridge(
    bridge: PolicyBridge,
    *,
    host: str,
    port: int,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the benchmark policy bridge is intentionally loopback-only")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ClearVLABridge/3"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, code: int, value: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(404, b"not found\n", "text/plain")
                return
            payload = json.dumps(bridge.health(), sort_keys=True).encode("utf-8")
            self._send(200, payload, "application/json")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/act", "/observe"}:
                self._send(404, b"not found\n", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("bridge request length is outside its bound")
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as data:
                    observation = policy_observation(
                        top=data["top"],
                        wrist=data["wrist"],
                        state=data["state"],
                        action_state=data["action_state"],
                    )
                    if parsed.path == "/act":
                        reset = _event_flag(data["reset"], name="reset")
                        begin_instruction = (
                            _event_flag(data["begin_instruction"], name="begin_instruction")
                            if "begin_instruction" in data else False
                        )
                if parsed.path == "/observe":
                    payload = json.dumps(
                        {"history_time_index": bridge.observe(observation)},
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(200, payload, "application/json")
                else:
                    query = urllib.parse.parse_qs(parsed.query)
                    instruction = query.get("instruction", [""])[0]
                    action = bridge.act(
                        observation, instruction, reset=reset,
                        begin_instruction=begin_instruction,
                    )
                    output = io.BytesIO()
                    np.save(output, action, allow_pickle=False)
                    self._send(200, output.getvalue(), "application/x-npy")
            except Exception as error:
                payload = json.dumps(
                    {"error": type(error).__name__, "message": str(error)},
                    sort_keys=True,
                ).encode("utf-8")
                self._send(400, payload, "application/json")

    HTTPServer((host, int(port)), Handler).serve_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--checkpoint", type=Path)
    policy_group.add_argument(
        "--smoke-zero-policy",
        action="store_true",
        help="Exercise only the loopback protocol; never use for benchmark scores",
    )
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dinov2-model",
        type=Path,
        default=None,
        help="Optional relocated local DINO snapshot; model identity is still audited",
    )
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_zero_policy:
        policy = SmokeZeroPolicy()
        device_name = "none-smoke-only"
    else:
        import torch

        from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else torch.device(args.device)
        )
        assert args.checkpoint is not None
        policy = ClearVLACheckpointPolicy(
            args.checkpoint,
            device=device,
            t5_condition=args.t5_condition,
            dinov2_model=args.dinov2_model,
            dinov2_local_files_only=args.dinov2_local_files_only,
            seed=args.seed,
        )
        device_name = str(device)
    bridge = PolicyBridge(policy)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "device": device_name,
                "policy": type(policy).__name__,
                "health": bridge.health(),
            }
        ),
        flush=True,
    )
    serve_policy_bridge(bridge, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = [
    "PolicyBridge",
    "RemotePolicyClient",
    "SmokeZeroPolicy",
    "policy_observation",
    "serve_policy_bridge",
]
