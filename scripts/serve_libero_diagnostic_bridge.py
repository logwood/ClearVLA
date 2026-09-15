"""Serve a formal ClearVLA checkpoint with a diagnostic RNG-reseed endpoint.

The normal ClearVLA loopback bridge exposes ``/health``, ``/act`` and
``/observe``.  This isolated server keeps those handlers byte-for-byte in
spirit and adds ``POST /reseed`` for the expert-handoff diagnostic: it calls
the checkpoint policy's existing ``reset()`` method (which only reseeds its
owned generator and clears mutable telemetry) without touching the causal
history.  The endpoint is intentionally not part of the formal evaluator.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from clearvla.benchmarks.bridge import PolicyBridge, policy_observation


def serve_diagnostic_bridge(
    bridge: PolicyBridge,
    *,
    host: str,
    port: int,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the diagnostic bridge is intentionally loopback-only")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ClearVLABridgeDiagnostic/1"

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
            if parsed.path == "/reseed":
                try:
                    # ClearVLACheckpointPolicy.reset() deliberately owns only
                    # its generator and telemetry; PolicyBridge.history is
                    # left intact so the next /act consumes the same causal
                    # history with a matched first noise draw.
                    reset = getattr(bridge.policy, "reset", None)
                    if not callable(reset):
                        raise RuntimeError("checkpoint policy has no reset method")
                    reset()
                    payload = json.dumps(
                        {
                            "status": "ok",
                            "history_time_index": (
                                int(bridge.history.time_index)
                                if bridge.initialized
                                else None
                            ),
                            "mode": "policy_generator_only",
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(200, payload, "application/json")
                except Exception as error:
                    payload = json.dumps(
                        {"error": type(error).__name__, "message": str(error)},
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(400, payload, "application/json")
                return
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
                        reset_value = np.asarray(data["reset"]).reshape(-1)
                        if reset_value.shape != (1,):
                            raise ValueError("bridge reset flag must contain one value")
                        reset = bool(reset_value[0])
                if parsed.path == "/observe":
                    payload = json.dumps(
                        {"history_time_index": bridge.observe(observation)},
                        sort_keys=True,
                    ).encode("utf-8")
                    self._send(200, payload, "application/json")
                else:
                    query = urllib.parse.parse_qs(parsed.query)
                    instruction = query.get("instruction", [""])[0]
                    action = bridge.act(observation, instruction, reset=reset)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18780)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

    device = torch.device(args.device)
    policy = ClearVLACheckpointPolicy(
        args.checkpoint,
        device=device,
        t5_condition=args.t5_condition,
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        seed=int(args.seed),
    )
    bridge = PolicyBridge(policy)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": int(args.port),
                "device": str(device),
                "policy": type(policy).__name__,
                "diagnostic_endpoint": "POST /reseed (policy generator only)",
                "health": bridge.health(),
            }
        ),
        flush=True,
    )
    serve_diagnostic_bridge(bridge, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

