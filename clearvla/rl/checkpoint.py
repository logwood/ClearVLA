"""Atomic adapter-only snapshots; never overwrite or relabel a base checkpoint."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

SCHEMA = "clearvla-residual-sac-v1"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256()
    for directory in ("rl", "simulation", "mainline", "data", "vision", "action_solvers"):
        for path in sorted((root / directory).rglob("*.py")):
            sha.update(path.relative_to(root).as_posix().encode())
            sha.update(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return sha.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    _atomic(
        path,
        lambda stream: stream.write(
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # All public artifacts are immutable. Hard-link publication is atomic and
    # no-clobber even when another process races the initial existence check.
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_adapter(path: Path, learner, identity: dict, *, episodes: int, env_steps: int) -> None:
    payload = dict(
        schema=SCHEMA,
        identity=identity,
        identity_sha256=digest(identity),
        continuation="episode_boundary_reset_v1",
        episodes=episodes,
        env_steps=env_steps,
        learner=learner.state_dict(),
    )
    _atomic(path, lambda stream: torch.save(payload, stream))


def load_adapter(path: Path, learner, identity: dict) -> dict:
    # Load only your own trusted local artifacts (torch optimizers/replay have
    # Python/NumPy state). Never load arbitrary downloaded RL checkpoints here.
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("identity") != identity
        or payload.get("identity_sha256") != digest(identity)
        or payload.get("continuation") != "episode_boundary_reset_v1"
    ):
        raise ValueError("adapter/base/environment/source identity differs")
    for name in ("episodes", "env_steps"):
        if type(payload.get(name)) is not int or payload[name] < 0:
            raise ValueError("invalid adapter continuation counter")
    learner.load_state_dict(payload["learner"])
    return {name: payload[name] for name in ("episodes", "env_steps")}
