from __future__ import annotations

"""Deterministic differential harness for the V39 policy stack.

The harness intentionally lives outside the policy import graph.  ``capture``
adds exactly one requested source root to ``sys.path`` and imports the model
from there.  Baseline and candidate captures must be separate processes; this
prevents module aliases or shared RNG state from turning a comparison into a
tautology.
"""

import argparse
from collections import OrderedDict, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


SCHEMA = "clearvla-policy-golden-v2"
FIXTURE_SCHEMA = "clearvla-policy-golden-fixture-v2"
DEFAULT_SEED = 7642077
LAYER_KEYS = (
    "rollout_tokens",
    "trajectory_pooled",
    "rollout_effect_pred",
    "rollout_delta_pred",
    "policy_effect_tokens",
    "policy_effect_time_tokens",
    "unified_intervention_latent_pred",
    "neutral_latent_pred",
    "milestone_step_delta_pred",
)
INTENT_SOURCE_NAMES = (
    "task",
    "state",
    "state_history",
    "executed",
    "proposal",
    "visual",
)
VARIANT_ARCHITECTURES = {
    "v76": "serial_owned_v2",
    "v77": "serial_owned_rms_v3",
}
CAPTURE_STAGES = ("construction", "boundaries", "train", "sample", "checkpoint")


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _support_tree_metadata(entries: Sequence[str]) -> list[dict[str, Any]]:
    """Fingerprint source trees supplied outside a historical Git object."""

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for entry in entries:
        name, separator, raw_path = entry.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        logical_path = Path(name)
        if not separator or not name or not raw_path:
            raise ValueError(f"support tree must use NAME=PATH syntax: {entry!r}")
        if logical_path.is_absolute() or ".." in logical_path.parts:
            raise ValueError(f"support tree name must be a repository-relative path: {name!r}")
        if name in names:
            raise ValueError(f"duplicate support tree name: {name!r}")
        names.add(name)

        root = Path(raw_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"support tree is not a directory: {root}")
        files: list[dict[str, str]] = []
        candidates = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for path in candidates:
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _file_sha256(path),
                }
            )
        if not files:
            raise ValueError(f"support tree contains no auditable files: {root}")
        serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result.append(
            {
                "name": name,
                "sha256": _sha256_bytes(serialized),
                "file_count": len(files),
                "files": files,
            }
        )
    return result


def _phase_seed(base_seed: int, name: str) -> int:
    payload = f"{int(base_seed)}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except ImportError:
        pass
    torch.manual_seed(int(seed))


def _configure_determinism(seed: int) -> None:
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None or hash_seed.lower() == "random":
        raise RuntimeError(
            "capture requires PYTHONHASHSEED to be fixed before Python starts; "
            "use PYTHONHASHSEED=0"
        )
    try:
        int(hash_seed)
    except ValueError as error:
        raise RuntimeError(f"invalid PYTHONHASHSEED={hash_seed!r}") from error
    _seed_all(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def _rng_fingerprint() -> dict[str, str]:
    result = {
        "python": _sha256_bytes(repr(random.getstate()).encode("utf-8")),
        "torch": _sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
    }
    try:
        import numpy as np

        result["numpy"] = _sha256_bytes(repr(np.random.get_state()).encode("utf-8"))
    except ImportError:
        result["numpy"] = "unavailable"
    return result


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


class ArtifactBuilder:
    """Flatten heterogeneous policy results without discarding key order."""

    def __init__(self) -> None:
        self.tensors: OrderedDict[str, Tensor] = OrderedDict()
        self.values: OrderedDict[str, Any] = OrderedDict()
        self.containers: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add(self, path: str, value: Any) -> None:
        if path in self.tensors or path in self.values or path in self.containers:
            raise KeyError(f"duplicate golden artifact path: {path}")
        if isinstance(value, Tensor):
            snapshot = value.detach().cpu().contiguous().clone()
            if (snapshot.is_floating_point() or snapshot.is_complex()) and not bool(
                torch.isfinite(snapshot).all()
            ):
                nonfinite = int((~torch.isfinite(snapshot)).sum().item())
                raise FloatingPointError(
                    f"golden artifact contains {nonfinite} non-finite values at {path}"
                )
            self.tensors[path] = snapshot
            return
        if is_dataclass(value):
            self.containers[path] = {
                "kind": "dataclass",
                "class": type(value).__name__,
                "fields": list(value.__dataclass_fields__),
            }
            for name in value.__dataclass_fields__:
                self.add(f"{path}/{name}", getattr(value, name))
            return
        if isinstance(value, Mapping):
            keys = [str(key) for key in value]
            self.containers[path] = {"kind": "mapping", "keys": keys}
            for key, item in value.items():
                self.add(f"{path}/{key}", item)
            return
        if isinstance(value, (list, tuple)):
            self.containers[path] = {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "length": len(value),
            }
            for index, item in enumerate(value):
                self.add(f"{path}/{index}", item)
            return
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"golden artifact contains non-finite scalar at {path}")
        self.values[path] = _jsonable(value)

    def write(self, output_dir: Path, metadata: Mapping[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "tensors.pt"
        torch.save(self.tensors, tensor_path)
        manifest = {
            "schema": SCHEMA,
            "metadata": _jsonable(metadata),
            "coverage": {
                "tensor_count": len(self.tensors),
                "tensor_numel": sum(int(value.numel()) for value in self.tensors.values()),
                "value_count": len(self.values),
                "container_count": len(self.containers),
                "nonfinite_count": 0,
            },
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "tensor_keys": list(self.tensors),
            "values": self.values,
            "containers": self.containers,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
        )


def _randn(generator: torch.Generator, *shape: int, scale: float = 1.0) -> Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * float(scale)


def create_fixture(path: Path, *, seed: int = DEFAULT_SEED) -> None:
    """Create one implementation-independent event-rich synthetic batch."""

    spec = {
        "batch": 2,
        "action_dim": 7,
        "state_dim": 7,
        "horizon": 24,
        "executed_history": 3,
        "hidden": 32,
        "depth": 8,
        "visual_history": 2,
        "cameras": 1,
        "patches": 4,
        "visual_dim": 16,
        "future_anchors": 2,
        "target_future": 2,
        "future_grid": 2,
        "boundary_rollout_tokens": 8,
        "boundary_transition_tokens": 6,
        "boundary_state_tokens": 3,
        "boundary_intent_tokens": 2,
        "boundary_layer_tokens": 3,
    }
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    b = int(spec["batch"])
    t = int(spec["horizon"])
    a = int(spec["action_dim"])
    h = int(spec["hidden"])
    vh = int(spec["visual_history"])
    cams = int(spec["cameras"])
    patches = int(spec["patches"])
    vd = int(spec["visual_dim"])

    time = torch.linspace(0.0, 1.0, t, dtype=torch.float32)
    action_state = torch.tensor(
        [
            [0.15, -0.20, 0.35, -0.10, 0.25, -0.30, 0.05],
            [-0.25, 0.10, -0.15, 0.30, -0.20, 0.15, 0.85],
        ],
        dtype=torch.float32,
    )
    arm_rows = []
    for batch_index in range(b):
        phase = 0.35 * float(batch_index)
        channels = []
        for dim in range(a - 1):
            frequency = 1.0 + 0.25 * float(dim)
            curve = 0.18 * torch.sin((time + phase) * math.pi * frequency)
            curve = curve + 0.04 * float(dim - 2) * time
            channels.append(action_state[batch_index, dim] + curve)
        arm_rows.append(torch.stack(channels, dim=-1))
    arm = torch.stack(arm_rows, dim=0)
    gripper = torch.tensor(
        [
            [
                0.05, 0.05, 0.05, 0.05, 0.08, 0.18, 0.52, 0.75,
                0.88, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.88,
                0.72, 0.50, 0.30, 0.15, 0.08, 0.08, 0.08, 0.08,
            ],
            [
                0.88, 0.88, 0.88, 0.88, 0.84, 0.70, 0.35, 0.18,
                0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.10,
                0.25, 0.45, 0.65, 0.78, 0.84, 0.84, 0.84, 0.84,
            ],
        ],
        dtype=torch.float32,
    )
    policy_action = torch.cat([arm, gripper[..., None]], dim=-1)
    state = action_state + torch.tensor(
        [[0.02, -0.01, 0.03, 0.01, -0.02, 0.01, 0.0], [-0.01, 0.02, -0.02, 0.03, 0.01, -0.01, 0.0]]
    )
    history_alpha = torch.linspace(0.55, 1.0, vh)[None, :, None]
    state_history = action_state[:, None] + history_alpha * (state - action_state)[:, None]
    executed_alpha = torch.linspace(-0.20, 0.0, int(spec["executed_history"]))[None, :, None]
    executed = action_state[:, None] + executed_alpha
    visual = _randn(g, b, vh, cams, patches, vd, scale=0.35)
    target_visual = visual[:, None].expand(-1, int(spec["target_future"]), -1, -1, -1, -1).clone()
    future_offset = torch.linspace(0.03, 0.12, int(spec["target_future"]))[None, :, None, None, None, None]
    target_visual = target_visual + future_offset * _randn(
        g, b, int(spec["target_future"]), vh, cams, patches, vd, scale=0.8
    )
    sample_noise_native = _randn(g, b, t, a, scale=0.55)
    sample_noise_native[..., -1] = torch.tensor(
        [
            [
                -0.60, -0.50, -0.40, -0.25, -0.10, 0.05, 0.20, 0.40,
                0.60, 0.50, 0.40, 0.25, 0.10, -0.05, -0.20, -0.40,
                -0.55, -0.45, -0.30, -0.10, 0.10, 0.30, 0.20, -0.10,
            ],
            [
                0.50, 0.40, 0.30, 0.15, 0.00, -0.20, -0.40, -0.50,
                -0.35, -0.15, 0.10, 0.30, 0.50, 0.45, 0.30, 0.10,
                -0.10, -0.30, -0.50, -0.40, -0.20, 0.00, 0.25, 0.50,
            ],
        ],
        dtype=sample_noise_native.dtype,
    )

    tensors: OrderedDict[str, Tensor] = OrderedDict(
        visual=visual,
        target_visual=target_visual,
        state=state,
        state_raw=state * 0.75 + 0.1,
        action_state=action_state,
        history_state=state_history,
        executed_action_history=executed,
        executed_action_history_raw=executed * 0.75 + 0.1,
        policy_action=policy_action,
        policy_action_raw=policy_action * 0.75 + 0.1,
        sample_noise_native=sample_noise_native,
    )
    tensors["boundary/noisy_physical"] = _randn(g, b, t, 2 * (a - 1) + 6, scale=0.4)
    tensors["boundary/time"] = torch.tensor([0.22, 0.81], dtype=torch.float32)
    tensors["boundary/trajectory_tokens"] = _randn(g, b, t, h, scale=0.3)
    tensors["boundary/trajectory_workspace_tokens"] = _randn(g, b, t, h, scale=0.3)
    tensors["boundary/rollout_tokens"] = _randn(
        g, b, int(spec["boundary_rollout_tokens"]), h, scale=0.3
    )
    tensors["boundary/transition_memory"] = _randn(
        g, b, int(spec["boundary_transition_tokens"]), h, scale=0.3
    )
    tensors["boundary/event_evidence"] = _randn(g, b, t, 3, scale=0.3)
    tensors["boundary/state_memory"] = _randn(
        g, b, int(spec["boundary_state_tokens"]), h, scale=0.3
    )
    for name in INTENT_SOURCE_NAMES:
        tensors[f"boundary/intent/{name}"] = _randn(
            g, b, int(spec["boundary_intent_tokens"]), h, scale=0.3
        )
    for layer_index in range(int(spec["depth"])):
        for key in LAYER_KEYS:
            tensors[f"boundary/layer/{layer_index}/{key}"] = _randn(
                g, b, int(spec["boundary_layer_tokens"]), h, scale=0.3
            )

    payload = {
        "schema": FIXTURE_SCHEMA,
        "seed": int(seed),
        "spec": spec,
        "tensors": tensors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"wrote fixture: {path}")


def _import_target(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    policy_file = source_root / "clearvla" / "experiments" / "observed_state_lab" / "policy_v39.py"
    if not policy_file.is_file():
        raise FileNotFoundError(f"not a ClearVLA source root: {source_root}")
    conflicts = []
    for name, module in tuple(sys.modules.items()):
        if name != "clearvla" and not name.startswith("clearvla."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            Path(module_file).resolve().relative_to(source_root)
        except ValueError:
            conflicts.append(f"{name}={module_file}")
    if conflicts:
        raise RuntimeError(
            "capture target conflicts with ClearVLA modules already loaded from another tree; "
            "run capture in a fresh process: " + ", ".join(conflicts[:5])
        )
    source_text = str(source_root)
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    policy = importlib.import_module("clearvla.experiments.observed_state_lab.policy_v39")
    runtime = importlib.import_module("clearvla.experiments.observed_state_lab.policy_runtime_v39")
    imported_policy = Path(policy.__file__).resolve()
    try:
        imported_policy.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            f"target import escaped requested source root: {imported_policy}"
        ) from error
    return {"policy": policy, "runtime": runtime}


def _filtered_dataclass(cls: type[Any], values: Mapping[str, Any]) -> Any:
    fields = getattr(cls, "__dataclass_fields__", {})
    return cls(**{name: value for name, value in values.items() if name in fields})


def _build_policy_config(policy: Any, spec: Mapping[str, Any], variant: str) -> Any:
    architecture = VARIANT_ARCHITECTURES[variant]
    values = {
        "action_dim": int(spec["action_dim"]),
        "state_dim": int(spec["state_dim"]),
        "action_horizon": int(spec["horizon"]),
        "executed_history_length": int(spec["executed_history"]),
        "hidden_size": int(spec["hidden"]),
        "num_heads": 4,
        "depth": int(spec["depth"]),
        "action_decoder_depth": 2,
        "proposal_depth": 1,
        "ffn_expansion": 2.0,
        "proposal_dropout": 0.25,
        "dropout": 0.0,
        "inference_steps": 3,
        "first_execution_steps": 4,
        "mid_execution_steps": int(spec["horizon"]),
        "gripper_field_dim": 6,
        "gripper_field_mode": "parseval_temporal",
        "arm_flow_mode": "manifold_native",
        "arm_noise_temporal_rho": 0.0,
        "visual_token_dim": int(spec["visual_dim"]),
        "visual_history_length": int(spec["visual_history"]),
        "num_cameras": int(spec["cameras"]),
        "patches_per_camera": int(spec["patches"]),
        "canvas_registers": 2,
        "future_anchors": int(spec["future_anchors"]),
        "target_future_count": int(spec["target_future"]),
        "visual_memory_dropout": 0.0,
        "canvas_dropout": 0.0,
        "role_dropout": 0.0,
        "action_basis_tokens": 2,
        "future_grid_size": int(spec["future_grid"]),
        "rollout_tail_start_step": 4,
        "rollout_tail_full_step": 6,
        "controlled_delta_rank": 4,
        "base_effect_hidden": 16,
        "latent_action_tokens": 2,
        "neutral_action_tokens": 2,
        "midcut_layer": 1,
        "controlled_base_mode": "fixed_zero",
        "layer_contract_adapters": 1,
        "layer_contract_adapter_dim": 16,
        "layer_contract_grad_scale": 1.0,
        "layer_shared_fm_probe": 0,
        "layer_recurrent_consequence": 1,
        "layer_consequence_steps": int(spec["future_anchors"]),
        "layer_consequence_hidden": 16,
        "layer_state_counterfactual": 1,
        "layer_zero_base_diagnostic": 1,
        "action_consequence_self_condition": 0,
        "final_action_decoder": "hierarchical_mmdit_action",
        "hierarchical_mmdit_depth": 2,
        "hierarchical_mmdit_refine_steps": 2,
        "hierarchical_mmdit_low_slots": 5,
        "hierarchical_mmdit_stage_slots": 3,
        "hierarchical_mmdit_ffn_expansion": 2.0,
        "hierarchical_mmdit_layer_grad_scale": 0.0,
        "hierarchical_mmdit_source_grad_scale": 0.0,
        "hierarchical_mmdit_noisy_causal": 1,
        "hierarchical_mmdit_stage_promote_scale_init": 0.05,
        "hierarchical_mmdit_output_init_std": 1e-3,
        "hierarchical_mmdit_residual_scale_max": 0.20,
        "hierarchical_mmdit_architecture_version": architecture,
        "hierarchical_mmdit_output_contract": 0,
        "hierarchical_mmdit_noisy_gate_mode": 0,
    }
    config = _filtered_dataclass(policy.V39PolicyConfig, values)
    config.validate()
    actual = getattr(config, "hierarchical_mmdit_architecture_version", None)
    if actual != architecture:
        raise RuntimeError(
            f"variant {variant} requested architecture {architecture!r}, target constructed {actual!r}"
        )
    return config


def _build_trainer(runtime: Any) -> Any:
    values = {
        "training_stage": "policy",
        "contract_mode": "layer_adapter",
        "lr": 8.0e-5,
        "proposal_lr": 5.0e-5,
        "weight_decay": 0.01,
        "warmup_steps": 2,
        "min_lr_ratio": 0.1,
        "grad_clip": 1.0,
        "latent_cvae_grad_clip": 1.0,
        "upper_lr_scale": 0.20,
        "midcut_head_lr_scale": 1.0,
        "midcut_aux_loss_weight": 0.05,
        "midcut_aux_final_ratio": 0.20,
        "midcut_aux_decay_epochs": 4,
        "layer_contract_adapter_policy_lr_scale": 1.0,
        "latent_cvae_action_decoder_lr_scale": 0.70,
        "layer_contract_loss_weight": 1.0,
        "layer_contract_final_action_loss_weight": 0.0,
        "layer_latent_loss_weight": 1.0,
        "layer_fm_probe_loss_weight": 0.0,
        "layer_event_loss_weight": 0.05,
        "layer_motion_loss_weight": 0.03,
        "layer_contrast_loss_weight": 0.03,
        "layer_variance_loss_weight": 0.05,
        "layer_norm_loss_weight": 0.02,
        "layer_delta_match_loss_weight": 0.15,
    }
    return _filtered_dataclass(runtime.V39PolicyTrainerConfig, values)


def _git_metadata(source_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"source_root": str(source_root.resolve())}
    for key, command in (
        ("commit", ["git", "-C", str(source_root), "rev-parse", "HEAD"]),
        ("describe", ["git", "-C", str(source_root), "describe", "--always", "--dirty", "--tags"]),
    ):
        try:
            result[key] = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            result[key] = "unavailable"
    return result


def _direct_parameter_paths(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    try:
        modules = module.named_modules(remove_duplicate=False)
    except TypeError:
        modules = module.named_modules()
    result = []
    for module_name, child in modules:
        prefix = f"{module_name}." if module_name else ""
        for local_name, parameter in child._parameters.items():
            if parameter is not None:
                result.append((prefix + local_name, parameter))
    return result


def _named_parameter_aliases(module: torch.nn.Module) -> list[list[str]]:
    names_by_id: defaultdict[int, list[str]] = defaultdict(list)
    for name, parameter in _direct_parameter_paths(module):
        names_by_id[id(parameter)].append(name)
    return sorted(
        (sorted(names) for names in names_by_id.values() if len(names) > 1),
        key=lambda names: names[0],
    )


def _state_manifest(system: torch.nn.Module) -> list[dict[str, Any]]:
    parameter_by_name = dict(_direct_parameter_paths(system))
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "kind": "parameter" if name in parameter_by_name else "buffer",
            "requires_grad": (
                bool(parameter_by_name[name].requires_grad)
                if name in parameter_by_name else None
            ),
        }
        for name, value in system.state_dict().items()
    ]


def _module_manifest(system: torch.nn.Module) -> list[dict[str, str]]:
    return [
        {"name": name, "class": type(module).__name__}
        for name, module in system.named_modules()
    ]


def _optimizer_manifest(
    system: torch.nn.Module, groups: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    names = {id(parameter): name for name, parameter in system.named_parameters()}
    seen: set[int] = set()
    result = []
    for index, group in enumerate(groups):
        params = list(group["params"])
        duplicates = []
        for parameter in params:
            parameter_id = id(parameter)
            if parameter_id in seen:
                duplicates.append(names.get(parameter_id, "<unnamed>"))
            else:
                seen.add(parameter_id)
        result.append(
            {
                "index": index,
                "name": str(group.get("name", f"group_{index}")),
                "lr": float(group["lr"]),
                "parameter_names": [names.get(id(parameter), "<unnamed>") for parameter in params],
                "duplicate_names": duplicates,
            }
        )
    return result


def _fixture_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    tensors = payload["tensors"]
    depth = int(payload["spec"]["depth"])
    return {
        "noisy_physical": tensors["boundary/noisy_physical"],
        "time": tensors["boundary/time"],
        "trajectory_tokens": tensors["boundary/trajectory_tokens"],
        "trajectory_workspace_tokens": tensors["boundary/trajectory_workspace_tokens"],
        "rollout_tokens": tensors["boundary/rollout_tokens"],
        "transition_memory": [tensors["boundary/transition_memory"]],
        "event_evidence": tensors["boundary/event_evidence"],
        "state_memory": [tensors["boundary/state_memory"]],
        "intent_memory": {
            name: tensors[f"boundary/intent/{name}"] for name in INTENT_SOURCE_NAMES
        },
        "layer_contracts": [
            {key: tensors[f"boundary/layer/{index}/{key}"] for key in LAYER_KEYS}
            for index in range(depth)
        ],
    }


def _add_phase_rng(builder: ArtifactBuilder, phase: str, when: str) -> None:
    builder.add(f"rng/{phase}/{when}/fingerprint", _rng_fingerprint())
    builder.add(f"rng/{phase}/{when}/torch_state", torch.get_rng_state())


def _capture_structure(
    builder: ArtifactBuilder,
    system: torch.nn.Module,
    config: Any,
    trainer: Any,
    optimizer_groups: Sequence[Mapping[str, Any]],
) -> None:
    assigned_ids = {
        id(parameter)
        for group in optimizer_groups
        for parameter in group["params"]
    }
    optimizer_manifest = _optimizer_manifest(system, optimizer_groups)
    unassigned_trainable = [
        name for name, parameter in system.named_parameters()
        if parameter.requires_grad and id(parameter) not in assigned_ids
    ]
    assigned_frozen = [
        name for name, parameter in system.named_parameters()
        if not parameter.requires_grad and id(parameter) in assigned_ids
    ]
    duplicate_assignments = [
        name
        for group in optimizer_manifest
        for name in group["duplicate_names"]
    ]
    builder.add("structure/config", asdict(config))
    builder.add("structure/trainer", asdict(trainer))
    builder.add("structure/modules", _module_manifest(system))
    builder.add("structure/state_manifest", _state_manifest(system))
    builder.add("structure/parameter_aliases", _named_parameter_aliases(system))
    builder.add("structure/optimizer_groups", optimizer_manifest)
    builder.add("structure/optimizer_unassigned_trainable", unassigned_trainable)
    builder.add("structure/optimizer_assigned_frozen", assigned_frozen)
    builder.add("structure/optimizer_duplicate_assignments", duplicate_assignments)
    builder.add("structure/parameter_report", system.parameter_report())
    builder.add("state/initial", system.state_dict())
    if unassigned_trainable:
        raise RuntimeError(
            "golden optimizer leaves trainable parameters unassigned: "
            + ", ".join(unassigned_trainable[:8])
        )
    if assigned_frozen:
        raise RuntimeError(
            "golden optimizer includes frozen parameters: "
            + ", ".join(assigned_frozen[:8])
        )
    if duplicate_assignments:
        raise RuntimeError(
            "golden optimizer assigns parameters more than once: "
            + ", ".join(duplicate_assignments[:8])
        )


@torch.no_grad()
def _capture_boundaries(
    builder: ArtifactBuilder,
    system: Any,
    payload: Mapping[str, Any],
) -> None:
    tensors = payload["tensors"]
    codec_state = tensors["action_state"]
    encoded = system.codec.encode(tensors["policy_action"], codec_state)
    projected = system.codec.project_physical(encoded, codec_state)
    decoded = system.codec.decode(projected, codec_state)
    builder.add(
        "boundary/codec",
        {"encoded": encoded, "projected": projected, "decoded": decoded},
    )

    decoder = system.planner.hierarchical_mmdit_action_decoder
    if decoder is None:
        raise RuntimeError("golden policy does not own a hierarchical MMDiT decoder")
    inputs = _fixture_boundary(payload)
    organized = decoder.organizer(
        **{key: value for key, value in inputs.items() if key not in {"noisy_physical", "time"}}
    )
    contracts = decoder.intent_compiler(
        layer_scan=organized["layer_scan"],
        geom_summary=organized["geom_summary"],
        global_summary=organized["global_summary"],
        transition_summary=organized["transition_summary"],
        event_summary=organized["event_summary"],
        state_summary=organized["state_summary"],
    )
    prepared = decoder.workspace.prepare_evidence(
        organized["evidence_sources"],
        batch_size=int(inputs["noisy_physical"].shape[0]),
        device=inputs["noisy_physical"].device,
        dtype=inputs["noisy_physical"].dtype,
    )
    stage = decoder.workspace.init_stage(contracts["stage_contract"])
    step_state, _ = decoder._step_state(
        0,
        batch_size=int(inputs["noisy_physical"].shape[0]),
        device=inputs["noisy_physical"].device,
        dtype=inputs["noisy_physical"].dtype,
    )
    time_state = decoder.time_lift(decoder.time(inputs["time"]))
    workspace_condition = decoder.workspace_condition(
        torch.cat([contracts["global_intent"], time_state], dim=-1)
    )
    workspace_step = decoder.workspace.step(
        prepared_evidence=prepared,
        stage_content=stage,
        primary_cond=workspace_condition,
        step_index=0,
        read_contract=contracts["read_contract"],
        step_state_override=step_state,
    )
    decoder_output = decoder(**inputs)
    builder.add("boundary/organizer", organized)
    builder.add("boundary/intent_compiler", contracts)
    builder.add("boundary/workspace_prepared", prepared)
    builder.add("boundary/workspace_initial_stage", stage)
    builder.add("boundary/workspace_step0", workspace_step)
    builder.add("boundary/decoder", decoder_output)


def _sample(
    system: Any,
    tensors: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    result = system.sample(
        tensors["visual"],
        tensors["history_state"],
        tensors["executed_action_history"],
        tensors["state"],
        action_state=tensors["action_state"],
        steps=3,
        noise=tensors["sample_noise_native"],
        use_proposal=True,
        return_event_logits=True,
    )
    if not isinstance(result, dict):
        raise TypeError("golden sample requested diagnostics but received a bare tensor")
    return result


def _loss_path(
    runtime: Any,
    system: Any,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: Any,
) -> dict[str, Tensor]:
    losses = runtime.flow_losses(
        system, sample, output, trainer, enable_future_loss=True, global_step=0
    )
    aux = runtime.layer_contract_losses(
        system, sample, output, trainer, enable_future_loss=True
    )
    aux_scale = runtime._midcut_aux_scale(trainer, 1)
    total = losses["loss"] + float(aux_scale) * aux["layer_contract"]
    for key, value in aux.items():
        if key == "loss":
            losses["aux_layer_contract_loss"] = value.detach()
        elif key in losses:
            losses[f"aux_layer_contract_{key}"] = value.detach()
        else:
            losses[key] = value
    losses["loss"] = total
    losses["midcut_aux_scale"] = torch.as_tensor(
        aux_scale, device=total.device, dtype=total.dtype
    )
    return losses


def _parameter_gradients(system: torch.nn.Module) -> OrderedDict[str, Tensor | None]:
    return OrderedDict(
        (name, None if parameter.grad is None else parameter.grad)
        for name, parameter in system.named_parameters()
    )


def _optimizer_state_by_name(
    system: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> OrderedDict[str, Any]:
    name_by_id = {id(parameter): name for name, parameter in system.named_parameters()}
    result: OrderedDict[str, Any] = OrderedDict()
    for parameter, state in optimizer.state.items():
        result[name_by_id.get(id(parameter), "<unnamed>")] = state
    return result


def _capture_train_step(
    builder: ArtifactBuilder,
    runtime: Any,
    system: Any,
    trainer: Any,
    optimizer: torch.optim.Optimizer,
    schedule: Any,
    payload: Mapping[str, Any],
    *,
    step_index: int,
) -> None:
    prefix = f"train/step_{int(step_index)}"
    tensors = payload["tensors"]
    sample = {
        key: tensors[key]
        for key in (
            "visual",
            "target_visual",
            "state",
            "state_raw",
            "action_state",
            "history_state",
            "executed_action_history",
            "executed_action_history_raw",
            "policy_action",
            "policy_action_raw",
        )
    }
    system.train()
    system.zero_grad(set_to_none=True)
    output = system.flow_training_forward(
        sample["visual"],
        sample["history_state"],
        sample["executed_action_history"],
        sample["state"],
        sample["policy_action"],
        action_state=sample["action_state"],
        target_visual=sample["target_visual"],
        make_counterfactuals=True,
        stop_at_midcut=False,
    )
    losses = _loss_path(runtime, system, sample, output, trainer)
    if not torch.isfinite(losses["loss"].detach()).all():
        raise FloatingPointError("golden train step produced non-finite loss")
    builder.add(f"{prefix}/output", output)
    builder.add(f"{prefix}/losses_pre_backward", losses)
    losses["loss"].float().backward()
    runtime._attach_grad_diagnostics(losses, system)
    builder.add(f"{prefix}/gradients_raw", _parameter_gradients(system))
    builder.add(f"{prefix}/grad_diagnostics", losses)

    decoder = system.planner.hierarchical_mmdit_action_decoder
    local_clip = float(getattr(trainer, "latent_cvae_grad_clip", 0.0))
    if decoder is not None and local_clip > 0.0:
        local_total = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), local_clip, error_if_nonfinite=True
        )
    else:
        local_total = torch.zeros(())
    global_total = torch.nn.utils.clip_grad_norm_(
        system.parameters(), float(trainer.grad_clip), error_if_nonfinite=True
    )
    builder.add(
        f"{prefix}/clip",
        {"local_total_before": local_total, "global_total_before": global_total},
    )
    builder.add(f"{prefix}/gradients_post_clip", _parameter_gradients(system))
    optimizer.step()
    schedule.step()
    builder.add(f"state/post_step_{int(step_index)}", system.state_dict())
    builder.add(
        f"{prefix}/optimizer_state_by_name",
        _optimizer_state_by_name(system, optimizer),
    )
    builder.add(
        f"{prefix}/optimizer_lrs_post_step",
        [group["lr"] for group in optimizer.param_groups],
    )
    builder.add(f"{prefix}/scheduler_state_post_step", schedule.state_dict())


def _sentinel_tensor(name: str, reference: Tensor) -> Tensor:
    if reference.is_floating_point() or reference.is_complex():
        code = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
        value = ((code % 2001) - 1000) / 10000.0
        return torch.full_like(reference, value)
    return reference.detach().clone()


def _capture_checkpoint_protocol(
    builder: ArtifactBuilder,
    modules: Mapping[str, Any],
    config: Any,
    trainer: Any,
    trained_system: Any,
    trained_optimizer: torch.optim.Optimizer,
    trained_schedule: Any,
    *,
    seed: int,
) -> None:
    policy = modules["policy"]
    runtime = modules["runtime"]
    cli = importlib.import_module("clearvla.cli.train_v40_policy")
    target = trained_system.state_dict()
    staged: OrderedDict[str, Tensor] = OrderedDict(
        (name, _sentinel_tensor(name, value)) for name, value in target.items()
    )
    staged["planner.layer_contract_heads.__golden_dirty__"] = torch.ones(1)
    staged["planner.seed.noisy_physical_lift.grip_value.__golden_parseval__"] = torch.ones(1)
    staged["planner.latent_cvae_action_decoder.__golden_obsolete__"] = torch.ones(1)
    staged[
        "planner.latent_cvae_action_decoder.evidence_workspace.global_state_proj.__golden_v74__"
    ] = torch.ones(1)
    shape_key = next(
        name
        for name, value in target.items()
        if value.ndim > 0
        and not name.startswith(("planner.layer_contract_heads.", "planner.latent_cvae_action_decoder."))
    )
    original_shape = tuple(staged[shape_key].shape)
    mismatch_shape = (original_shape[0] + 1, *original_shape[1:])
    staged[shape_key] = torch.zeros(mismatch_shape, dtype=staged[shape_key].dtype)

    staged, skipped_dirty = cli._filter_stage1_state_dict(
        staged, reset_dirty_adapters=True
    )
    staged, skipped_parseval = cli._filter_parseval_replaced_state_dict(
        staged, enabled=True
    )
    staged, skipped_v74 = cli._filter_v74_time_controller_state_dict(
        staged, enabled=True
    )
    obsolete_prefixes = (
        "planner.latent_cvae_action_decoder.",
        "planner.hierarchical_mmdit_action_decoder.",
    )
    skipped_obsolete = [
        name for name in staged if name.startswith(obsolete_prefixes)
    ]
    staged = OrderedDict(
        (name, value)
        for name, value in staged.items()
        if not name.startswith(obsolete_prefixes)
    )
    staged, skipped_shape = cli._filter_shape_mismatched_state_dict(staged, target)

    _seed_all(_phase_seed(seed, "checkpoint-load"))
    migration_system = policy.V39PolicySystem(config)
    missing, unexpected = migration_system.load_state_dict(staged, strict=False)
    if unexpected:
        raise RuntimeError(f"migration protocol retained unexpected keys: {unexpected[:8]}")
    rejection_state = OrderedDict(staged)
    rejection_state["planner.__golden_unexpected__"] = torch.ones(1)
    _, rejection_unexpected = migration_system.load_state_dict(rejection_state, strict=False)

    _seed_all(_phase_seed(seed, "checkpoint-resume"))
    resume_system = policy.V39PolicySystem(config)
    resume_groups = runtime._optimizer_groups(resume_system, trainer)
    resume_optimizer = torch.optim.AdamW(
        resume_groups,
        weight_decay=trainer.weight_decay,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
    )
    resume_schedule = runtime.scheduler(
        resume_optimizer, 8, trainer.warmup_steps, trainer.min_lr_ratio
    )
    resume_system.load_state_dict(trained_system.state_dict(), strict=True)
    resume_optimizer.load_state_dict(trained_optimizer.state_dict())
    resume_schedule.load_state_dict(trained_schedule.state_dict())
    builder.add(
        "checkpoint/migration",
        {
            "shape_mismatch_key": shape_key,
            "shape_mismatch_original": original_shape,
            "shape_mismatch_injected": mismatch_shape,
            "skipped_dirty": skipped_dirty,
            "skipped_parseval": skipped_parseval,
            "skipped_v74": skipped_v74,
            "skipped_obsolete": skipped_obsolete,
            "skipped_shape": skipped_shape,
            "missing": list(missing),
            "unexpected": list(unexpected),
            "unexpected_probe": list(rejection_unexpected),
            "unexpected_probe_would_reject": bool(rejection_unexpected),
        },
    )
    builder.add("checkpoint/resume_model_state", resume_system.state_dict())
    builder.add(
        "checkpoint/resume_optimizer_state_by_name",
        _optimizer_state_by_name(resume_system, resume_optimizer),
    )
    builder.add("checkpoint/resume_scheduler_state", resume_schedule.state_dict())


def capture(
    *,
    source_root: Path,
    fixture_path: Path,
    output_dir: Path,
    variant: str,
    seed: int = DEFAULT_SEED,
    through: str = "checkpoint",
    support_trees: Sequence[str] = (),
) -> None:
    support_tree_metadata = _support_tree_metadata(support_trees)
    _configure_determinism(seed)
    payload = _load_torch(fixture_path)
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"unsupported fixture schema in {fixture_path}")
    if through not in CAPTURE_STAGES:
        raise ValueError(f"unknown capture stage: {through}")
    stage_index = CAPTURE_STAGES.index(through)
    modules = _import_target(source_root)
    policy = modules["policy"]
    runtime = modules["runtime"]
    config = _build_policy_config(policy, payload["spec"], variant)
    trainer = _build_trainer(runtime)

    _seed_all(_phase_seed(seed, "model-init"))
    init_rng_before = torch.get_rng_state().clone()
    init_fingerprint_before = _rng_fingerprint()
    system = policy.V39PolicySystem(config).to(device=torch.device("cpu"), dtype=torch.float32)
    init_rng_after = torch.get_rng_state().clone()
    init_fingerprint_after = _rng_fingerprint()
    groups = runtime._optimizer_groups(system, trainer)
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=trainer.weight_decay,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
    )
    schedule = runtime.scheduler(optimizer, 8, trainer.warmup_steps, trainer.min_lr_ratio)
    builder = ArtifactBuilder()
    builder.add("rng/model_init/before/fingerprint", init_fingerprint_before)
    builder.add("rng/model_init/before/torch_state", init_rng_before)
    builder.add("rng/model_init/after/fingerprint", init_fingerprint_after)
    builder.add("rng/model_init/after/torch_state", init_rng_after)
    _capture_structure(builder, system, config, trainer, groups)
    builder.add(
        "structure/optimizer_lrs_initial",
        [group["lr"] for group in optimizer.param_groups],
    )
    builder.add("structure/scheduler_state_initial", schedule.state_dict())

    if stage_index >= CAPTURE_STAGES.index("boundaries"):
        _seed_all(_phase_seed(seed, "boundaries"))
        _add_phase_rng(builder, "boundaries", "before")
        system.eval()
        _capture_boundaries(builder, system, payload)
        _add_phase_rng(builder, "boundaries", "after")

    if stage_index >= CAPTURE_STAGES.index("sample"):
        _seed_all(_phase_seed(seed, "sample-pre"))
        _add_phase_rng(builder, "sample_pre", "before")
        system.eval()
        builder.add("sample/pre_step", _sample(system, payload["tensors"]))
        _add_phase_rng(builder, "sample_pre", "after")

    if stage_index >= CAPTURE_STAGES.index("train"):
        for train_step in (1, 2):
            phase = f"train_step_{train_step}"
            _seed_all(_phase_seed(seed, phase))
            _add_phase_rng(builder, phase, "before")
            _capture_train_step(
                builder,
                runtime,
                system,
                trainer,
                optimizer,
                schedule,
                payload,
                step_index=train_step,
            )
            _add_phase_rng(builder, phase, "after")

    if stage_index >= CAPTURE_STAGES.index("sample"):
        _seed_all(_phase_seed(seed, "sample-post"))
        _add_phase_rng(builder, "sample_post", "before")
        system.eval()
        builder.add("sample/post_step", _sample(system, payload["tensors"]))
        _add_phase_rng(builder, "sample_post", "after")

    if stage_index >= CAPTURE_STAGES.index("checkpoint"):
        _seed_all(_phase_seed(seed, "checkpoint"))
        _add_phase_rng(builder, "checkpoint", "before")
        _capture_checkpoint_protocol(
            builder,
            modules,
            config,
            trainer,
            system,
            optimizer,
            schedule,
            seed=seed,
        )
        _add_phase_rng(builder, "checkpoint", "after")

    metadata = {
        "variant": variant,
        "architecture": VARIANT_ARCHITECTURES[variant],
        "seed": int(seed),
        "through": through,
        "optimizer_steps": 2 if stage_index >= CAPTURE_STAGES.index("train") else 0,
        "fixture": str(fixture_path.resolve()),
        "fixture_sha256": _file_sha256(fixture_path),
        "source": _git_metadata(source_root),
        "support_trees": support_tree_metadata,
        "python": sys.version,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    builder.write(output_dir, metadata)
    print(f"wrote capture: {output_dir}")


def _read_artifact(path: Path) -> tuple[dict[str, Any], OrderedDict[str, Tensor]]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported golden artifact schema in {manifest_path}")
    tensor_path = path / manifest["tensor_file"]
    actual_digest = _file_sha256(tensor_path)
    if actual_digest != manifest.get("tensor_file_sha256"):
        raise ValueError(f"tensor artifact checksum mismatch: {tensor_path}")
    tensors = _load_torch(tensor_path)
    if not isinstance(tensors, OrderedDict):
        tensors = OrderedDict(tensors)
    for key, value in tensors.items():
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            nonfinite = int((~torch.isfinite(value)).sum().item())
            raise ValueError(
                f"golden artifact contains {nonfinite} non-finite values at {key}: {tensor_path}"
            )
    return manifest, tensors


def _drop_ignored(value: Any, path: str, ignored: Sequence[str]) -> Any:
    if any(path == prefix or path.startswith(prefix + "/") for prefix in ignored):
        return None
    if isinstance(value, dict):
        return {
            key: child
            for key, item in value.items()
            if (child := _drop_ignored(item, f"{path}/{key}" if path else key, ignored)) is not None
        }
    if isinstance(value, list):
        return [
            child
            for index, item in enumerate(value)
            if (child := _drop_ignored(item, f"{path}/{index}" if path else str(index), ignored)) is not None
        ]
    return value


def _flatten_json(value: Any, path: str, output: OrderedDict[str, Any]) -> None:
    if isinstance(value, dict):
        output[f"{path}/@keys"] = list(value)
        for key, item in value.items():
            _flatten_json(item, f"{path}/{key}", output)
        return
    if isinstance(value, list):
        output[f"{path}/@length"] = len(value)
        for index, item in enumerate(value):
            _flatten_json(item, f"{path}/{index}", output)
        return
    output[path] = value


def _matches_artifact_prefix(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _filter_manifest_artifacts(
    manifest: Mapping[str, Any], prefixes: Sequence[str]
) -> dict[str, Any]:
    filtered = {
        key: value
        for key, value in manifest.items()
        if key not in {"tensor_keys", "tensor_file", "tensor_file_sha256"}
    }
    for section in ("values", "containers"):
        values = filtered.get(section)
        if isinstance(values, dict):
            filtered[section] = {
                path: value
                for path, value in values.items()
                if not _matches_artifact_prefix(path, prefixes)
            }
    return filtered


def compare(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    ignore_prefixes: Sequence[str] = (),
    max_differences: int = 40,
    report_path: Path | None = None,
) -> bool:
    baseline_manifest, baseline_tensors = _read_artifact(baseline_dir)
    candidate_manifest, candidate_tensors = _read_artifact(candidate_dir)
    metadata_ignored = (
        "metadata/source",
        "metadata/python",
        "metadata/torch",
        "metadata/fixture",
        "tensor_file_sha256",
    )
    artifact_ignored = tuple(ignore_prefixes)
    differences: list[dict[str, Any]] = []

    baseline_meta = _filter_manifest_artifacts(baseline_manifest, artifact_ignored)
    candidate_meta = _filter_manifest_artifacts(candidate_manifest, artifact_ignored)
    baseline_meta = _drop_ignored(baseline_meta, "", metadata_ignored)
    candidate_meta = _drop_ignored(candidate_meta, "", metadata_ignored)
    baseline_flat: OrderedDict[str, Any] = OrderedDict()
    candidate_flat: OrderedDict[str, Any] = OrderedDict()
    _flatten_json(baseline_meta, "manifest", baseline_flat)
    _flatten_json(candidate_meta, "manifest", candidate_flat)
    metadata_paths = list(baseline_flat) + [
        path for path in candidate_flat if path not in baseline_flat
    ]
    for path in metadata_paths:
        if path not in candidate_flat:
            differences.append({"path": path, "kind": "metadata_missing_in_candidate"})
        elif path not in baseline_flat:
            differences.append({"path": path, "kind": "metadata_missing_in_baseline"})
        elif baseline_flat[path] != candidate_flat[path]:
            differences.append(
                {
                    "path": path,
                    "kind": "metadata_value_mismatch",
                    "baseline": baseline_flat[path],
                    "candidate": candidate_flat[path],
                }
            )
        if len(differences) >= int(max_differences):
            break

    baseline_keys = [
        key for key in baseline_tensors
        if not _matches_artifact_prefix(key, artifact_ignored)
    ]
    candidate_keys = [
        key for key in candidate_tensors
        if not _matches_artifact_prefix(key, artifact_ignored)
    ]
    if baseline_keys != candidate_keys:
        differences.append(
            {
                "path": "tensor_keys",
                "kind": "ordered_key_mismatch",
                "baseline_only": [key for key in baseline_keys if key not in set(candidate_keys)][:20],
                "candidate_only": [key for key in candidate_keys if key not in set(baseline_keys)][:20],
            }
        )

    for key in baseline_keys:
        if len(differences) >= int(max_differences):
            break
        if key not in candidate_tensors:
            continue
        baseline = baseline_tensors[key]
        candidate = candidate_tensors[key]
        if baseline.shape != candidate.shape or baseline.dtype != candidate.dtype:
            differences.append(
                {
                    "path": key,
                    "kind": "tensor_contract_mismatch",
                    "baseline_shape": list(baseline.shape),
                    "candidate_shape": list(candidate.shape),
                    "baseline_dtype": str(baseline.dtype),
                    "candidate_dtype": str(candidate.dtype),
                }
            )
        else:
            exact = float(atol) == 0.0 and float(rtol) == 0.0
            supports_tolerance = baseline.is_floating_point() or baseline.is_complex()
            if exact or not supports_tolerance:
                close = torch.equal(baseline, candidate)
            else:
                close = torch.allclose(
                    baseline,
                    candidate,
                    atol=float(atol),
                    rtol=float(rtol),
                    equal_nan=False,
                )
            if not close:
                delta = (baseline.to(torch.float64) - candidate.to(torch.float64)).abs()
                delta = torch.nan_to_num(
                    delta,
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=float("inf"),
                )
                differences.append(
                    {
                        "path": key,
                        "kind": "tensor_value_mismatch",
                        "max_abs": float(delta.max()) if delta.numel() else 0.0,
                        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
                    }
                )
    report = {
        "schema": "clearvla-policy-golden-comparison-v2",
        "pass": not differences,
        "baseline": str(baseline_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "baseline_source": baseline_manifest.get("metadata", {}).get("source", {}),
        "candidate_source": candidate_manifest.get("metadata", {}).get("source", {}),
        "atol": float(atol),
        "rtol": float(rtol),
        "comparison_coverage": {
            "metadata_paths": len(metadata_paths),
            "baseline_tensor_count": len(baseline_keys),
            "candidate_tensor_count": len(candidate_keys),
            "shared_tensor_count": sum(key in candidate_tensors for key in baseline_keys),
            "shared_tensor_numel": sum(
                int(baseline_tensors[key].numel())
                for key in baseline_keys
                if key in candidate_tensors
            ),
            "exact_tensor_values": float(atol) == 0.0 and float(rtol) == 0.0,
            "nonfinite_allowed": False,
        },
        "ignored_artifact_prefixes": list(artifact_ignored),
        "difference_count_reported": len(differences),
        "differences": differences,
    }
    rendered = json.dumps(report, indent=2)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return not differences


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and compare deterministic ClearVLA policy behavior."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture", help="create the shared synthetic fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--seed", type=int, default=DEFAULT_SEED)

    capture_parser = subparsers.add_parser("capture", help="capture one source tree in this process")
    capture_parser.add_argument("--source-root", type=Path, required=True)
    capture_parser.add_argument("--fixture", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--variant", choices=sorted(VARIANT_ARCHITECTURES), required=True)
    capture_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    capture_parser.add_argument("--through", choices=CAPTURE_STAGES, default="checkpoint")
    capture_parser.add_argument(
        "--support-tree",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="fingerprint a source tree supplied outside the selected Git object",
    )

    compare_parser = subparsers.add_parser("compare", help="compare two independent captures")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--atol", type=float, default=0.0)
    compare_parser.add_argument("--rtol", type=float, default=0.0)
    compare_parser.add_argument(
        "--ignore-prefix",
        action="append",
        default=[],
        help="reviewed artifact path prefix to ignore across tensors, values, and containers",
    )
    compare_parser.add_argument("--max-differences", type=int, default=40)
    compare_parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fixture":
        create_fixture(args.output, seed=args.seed)
        return 0
    if args.command == "capture":
        capture(
            source_root=args.source_root,
            fixture_path=args.fixture,
            output_dir=args.output,
            variant=args.variant,
            seed=args.seed,
            through=args.through,
            support_trees=args.support_tree,
        )
        return 0
    if args.command == "compare":
        passed = compare(
            args.baseline,
            args.candidate,
            atol=args.atol,
            rtol=args.rtol,
            ignore_prefixes=args.ignore_prefix,
            max_differences=args.max_differences,
            report_path=args.report,
        )
        return 0 if passed else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
