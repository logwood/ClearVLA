"""Deterministic dual-source equivalence harness for the active mainline.

Run this file by path, not with ``python -m``.  A capture imports exactly one
requested source root so the frozen and candidate implementations execute in
independent Python processes.  The reduced profile keeps the production module
topology while shrinking tensor widths enough for an edit-time CPU gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
from unittest import mock

import torch
from torch import Tensor, nn

FIXTURE_SCHEMA = "clearvla-mainline-equivalence-fixture-v1"
CAPTURE_PROFILE = "mainline-topology-complete-reduced-v1"
DEFAULT_SEED = 91_704
TIME_POINTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _load_golden_core() -> ModuleType:
    path = Path(__file__).with_name("policy_golden.py")
    spec = importlib.util.spec_from_file_location("_clearvla_policy_golden_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load golden artifact core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLDEN = _load_golden_core()


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except ImportError:
        pass
    torch.manual_seed(int(seed))


def _phase_seed(seed: int, phase: str) -> int:
    payload = f"{int(seed)}:{phase}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFF_FFFF


def _torch_state() -> Tensor:
    return torch.get_rng_state().clone()


def _generator_state(generator: torch.Generator) -> Tensor:
    return generator.get_state().clone()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(source_root: Path) -> str:
    digest = hashlib.sha256()
    package = source_root / "clearvla"
    for path in sorted(package.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _import_target(source_root: Path) -> dict[str, ModuleType]:
    source_root = source_root.resolve()
    policy_path = source_root / "clearvla" / "mainline" / "model" / "policy.py"
    if not policy_path.is_file():
        raise FileNotFoundError(f"not a ClearVLA source root: {source_root}")
    conflicts: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if name != "clearvla" and not name.startswith("clearvla."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(module_file).resolve().relative_to(source_root)
        except ValueError:
            conflicts.append(f"{name}={module_file}")
    if conflicts:
        raise RuntimeError(
            "target source conflicts with already imported ClearVLA modules: "
            + ", ".join(conflicts[:8])
        )
    sys.path.insert(0, str(source_root))
    names = {
        "config": "clearvla.mainline.config",
        "interfaces": "clearvla.mainline.interfaces",
        "policy": "clearvla.mainline.model.policy",
        "engine": "clearvla.mainline.training.engine",
        "optimizer": "clearvla.mainline.training.optimizer",
        "sampling": "clearvla.mainline.runtime.sampling",
    }
    return {key: importlib.import_module(name) for key, name in names.items()}


def build_reduced_equivalence_config(modules: Mapping[str, ModuleType]) -> Any:
    """Return a small config with the exact production parameter-key topology."""

    base = modules["config"].ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            action_basis_tokens=4,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=16,
            patches_per_camera=64,
        ),
        observation=replace(
            base.observation,
            feature_dim=16,
            address_route_dim=8,
            flow_iterations=2,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        top=replace(
            base.top,
            grounder_iterations=2,
            teacher_key_dim=8,
        ),
        bottom=replace(
            base.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_depth=2,
            controller_heads=4,
        ),
        optimizer=replace(base.optimizer, warmup_steps=2),
        runtime=replace(base.runtime, compute_dtype="fp32"),
    )
    config.validate()
    return config


def _randn(generator: torch.Generator, *shape: int) -> Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32)


def _rand(generator: torch.Generator, *shape: int) -> Tensor:
    return torch.rand(*shape, generator=generator, dtype=torch.float32)


def create_fixture(path: Path, *, seed: int = DEFAULT_SEED) -> None:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batch = 1
    horizon = 24
    action_dim = 7
    visual_history = 3
    cameras = 2
    patches = 64
    visual_dim = 16
    goal_dim = 16
    supports = 12
    tensors: OrderedDict[str, Tensor] = OrderedDict()
    tensors["dino_history"] = _randn(
        generator, batch, visual_history, cameras, patches, visual_dim
    )
    tensors["raw_rgb"] = _rand(generator, batch, visual_history, cameras, 3, 48, 48)
    tensors["state"] = _randn(generator, batch, action_dim)
    tensors["action_state"] = _randn(generator, batch, action_dim)
    tensors["codec_gripper_boundary"] = _randn(generator, batch, 1)
    tensors["state_history"] = _randn(generator, batch, 3, action_dim)
    tensors["executed_action_history"] = _randn(generator, batch, 8, action_dim)
    tensors["goal_tokens"] = _randn(generator, batch, 6, goal_dim)
    tensors["goal_mask"] = torch.tensor(
        [[True, True, True, True, False, False]], dtype=torch.bool
    )
    normalized = _randn(generator, batch, horizon, action_dim)
    raw_units = _randn(generator, batch, horizon, action_dim)
    gripper_curve = torch.cat(
        (
            torch.full((8,), -0.8),
            torch.full((8,), 0.7),
            torch.linspace(0.7, -0.5, 8),
        )
    )
    normalized[..., -1] = gripper_curve
    raw_units[..., -1] = gripper_curve * 0.6 + 0.1
    tensors["action_normalized"] = normalized
    tensors["action_raw_units"] = raw_units
    tensors["current_raw_units"] = _randn(generator, batch, action_dim)
    tensors["gripper_transition_boundary"] = _randn(generator, batch, action_dim)
    tensors["gripper_transition_boundary_raw_units"] = _randn(
        generator, batch, action_dim
    )
    tensors["future_dino_supports"] = _randn(
        generator, batch, supports, cameras, patches, visual_dim
    )
    tensors["future_action_sequence"] = _randn(generator, batch, 48, action_dim)
    tensors["future_state_sequence"] = _randn(generator, batch, 48, action_dim)
    tensors["future_offsets"] = torch.tensor(
        [[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]], dtype=torch.long
    )
    tensors["audit_sample_index"] = torch.tensor([7], dtype=torch.long)
    tensors["audit_episode_index"] = torch.tensor([3], dtype=torch.long)
    tensors["audit_frame_progress"] = torch.tensor([0.375], dtype=torch.float32)
    tensors["dynamic_physical_field"] = _randn(generator, batch, horizon, 18)
    tensors["deployment_initial_noise"] = _randn(generator, batch, horizon, 18)
    payload = {
        "schema": FIXTURE_SCHEMA,
        "seed": int(seed),
        "profile": CAPTURE_PROFILE,
        "tensors": tensors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"wrote fixture: {path}")


def _load_fixture(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"unsupported mainline equivalence fixture: {path}")
    return payload


def _training_batch(
    modules: Mapping[str, ModuleType], config: Any, payload: Mapping[str, Any]
) -> Any:
    api = modules["interfaces"]
    tensors = payload["tensors"]
    batch = api.TrainingBatch(
        online=api.OnlinePolicyInput(
            observation=api.CurrentObservation(
                dino_history=tensors["dino_history"].clone(),
                raw_rgb=tensors["raw_rgb"].clone(),
            ),
            history=api.ObservableHistory(
                state=tensors["state"].clone(),
                action_state=tensors["action_state"].clone(),
                codec_gripper_boundary=tensors["codec_gripper_boundary"].clone(),
                state_history=tensors["state_history"].clone(),
                executed_action_history=tensors["executed_action_history"].clone(),
            ),
            goal=api.GoalCondition(
                tokens=tensors["goal_tokens"].clone(),
                mask=tensors["goal_mask"].clone(),
            ),
        ),
        action_target=api.ActionSupervision(
            normalized=tensors["action_normalized"].clone(),
            raw_units=tensors["action_raw_units"].clone(),
            current_raw_units=tensors["current_raw_units"].clone(),
            gripper_transition_boundary=tensors[
                "gripper_transition_boundary"
            ].clone(),
            gripper_transition_boundary_raw_units=tensors[
                "gripper_transition_boundary_raw_units"
            ].clone(),
        ),
        future=api.FutureSupervision(
            dino_supports=tensors["future_dino_supports"].clone(),
            action_sequence=tensors["future_action_sequence"].clone(),
            state_sequence=tensors["future_state_sequence"].clone(),
            offsets=tensors["future_offsets"].clone(),
        ),
        audit=api.AuditMetadata(
            sample_index=tensors["audit_sample_index"].clone(),
            episode_index=tensors["audit_episode_index"].clone(),
            frame_progress=tensors["audit_frame_progress"].clone(),
        ),
    )
    batch.validate(config)
    return batch


MODULAR_TO_LEGACY_PREFIXES = (
    ("execution_bottom.decoder.terminal_controller.optional_command_head.",
     "bottom.decoder.gripper_command_head."),
    ("execution_bottom.decoder.terminal_controller.velocity_head.",
     "bottom.decoder.velocity_head."),
    ("execution_bottom.decoder.terminal_controller.motion_head.",
     "bottom.decoder.motion_head."),
    ("execution_bottom.decoder.terminal_controller.action_norm.",
     "bottom.decoder.action_norm."),
    ("conditioning.history_proposal.", "history_proposal."),
    ("observation.compiler.", "observation."),
    ("outlet_adapter.codec.", "action_codec."),
    ("bridge.query_encoder.", "bottom.query_encoder."),
    ("grounding.blocks.", "top.grounding_blocks."),
    ("grounding.content_mod.", "top.grounding_content_mod."),
    ("grounding.content_mod_scale", "top.grounding_content_mod_scale"),
    ("grounding.grounder.", "top.grounder."),
    ("intent.organizer.", "top.intent."),
    ("intent.coarse_action.", "top.coarse_action."),
    ("world.dynamics.", "top.dynamics."),
    ("training_targets.teacher.", "top.teacher."),
    ("training_targets.recognizer.", "top.recognizer."),
    ("p1.factual_reader.", "factual_reader."),
    ("p1.dynamic_time.", "bottom.p1_time."),
    ("p1.dynamic_content_mod.", "bottom.p1_content_mod."),
    ("p1.dynamic_content_mod_scale", "bottom.p1_content_mod_scale"),
    ("p1.dynamic_policy_block.", "bottom.p1_policy_block."),
    ("policy_compiler.effect_reader.", "top.effect_reader."),
    ("policy_compiler.consequence.", "top.consequence."),
    ("policy_compiler.plan_compiler.", "top.plan_compiler."),
    ("execution_bottom.layer_contract_heads.", "bottom.layer_contract_heads."),
    ("execution_bottom.decoder.", "bottom.decoder."),
)


def logical_parameter_name(name: str, *, layout: str) -> str:
    if layout == "legacy":
        return name
    for modular, legacy in MODULAR_TO_LEGACY_PREFIXES:
        if name == modular.rstrip(".") or name.startswith(modular):
            suffix = name[len(modular) :] if name.startswith(modular) else ""
            return legacy + suffix
    if name.startswith("transition."):
        return name
    raise KeyError(f"modular parameter/state path has no legacy map: {name}")


def _logical_named_parameters(
    model: nn.Module, *, layout: str, remove_duplicate: bool = True
) -> list[tuple[str, nn.Parameter]]:
    rows: list[tuple[str, nn.Parameter]] = []
    explicit_ledger = getattr(model, "_legacy_parameter_order", None)
    if layout == "modular" and explicit_ledger is not None:
        by_name = dict(model.named_parameters())
        for final_name, logical_name in explicit_ledger:
            try:
                parameter = by_name[final_name]
            except KeyError as error:
                raise RuntimeError(
                    f"modular parameter ledger references missing {final_name!r}"
                ) from error
            rows.append((logical_name, parameter))
        return rows
    for name, parameter in model.named_parameters(remove_duplicate=remove_duplicate):
        rows.append((logical_parameter_name(name, layout=layout), parameter))
    names = [name for name, _ in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("logical parameter map contains a collision")
    return rows


def _logical_state(model: nn.Module, *, layout: str) -> OrderedDict[str, Tensor]:
    result: OrderedDict[str, Tensor] = OrderedDict()
    explicit_order = getattr(model, "_legacy_state_order", None)
    if layout == "modular" and explicit_order is not None:
        state = model.state_dict()
        for final_name, logical in explicit_order:
            if final_name not in state:
                raise RuntimeError(f"modular state ledger references missing {final_name!r}")
            result[logical] = state[final_name]
        return result
    for name, value in model.state_dict().items():
        logical = logical_parameter_name(name, layout=layout)
        if logical in result:
            raise RuntimeError(f"logical state map collision at {logical}")
        result[logical] = value
    return result


def _parameter_manifest(model: nn.Module, *, layout: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in _logical_named_parameters(model, layout=layout)
    ]


def _alias_manifest(model: nn.Module, *, layout: str) -> list[list[str]]:
    aliases: dict[int, list[str]] = defaultdict(list)
    for name, parameter in _logical_named_parameters(
        model, layout=layout, remove_duplicate=False
    ):
        aliases[id(parameter)].append(name)
    return [names for names in aliases.values() if len(names) > 1]


def _parameter_lookup(model: nn.Module, *, layout: str) -> dict[int, str]:
    return {
        id(parameter): name
        for name, parameter in _logical_named_parameters(model, layout=layout)
    }


def _optimizer_manifest(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, layout: str
) -> list[dict[str, Any]]:
    lookup = _parameter_lookup(model, layout=layout)
    result: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        result.append(
            {
                "name": str(group.get("name", "unnamed")),
                "parameter_names": [lookup[id(parameter)] for parameter in group["params"]],
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "eps": float(group["eps"]),
            }
        )
    return result


def _optimizer_state_by_name(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, layout: str
) -> OrderedDict[str, Any]:
    lookup = _parameter_lookup(model, layout=layout)
    result: OrderedDict[str, Any] = OrderedDict()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = lookup[id(parameter)]
            result[name] = optimizer.state.get(parameter, {})
    return result


def _gradient_tree(model: nn.Module, *, layout: str) -> OrderedDict[str, Tensor | None]:
    return OrderedDict(
        (name, None if parameter.grad is None else parameter.grad.detach().clone())
        for name, parameter in _logical_named_parameters(model, layout=layout)
    )


class IdentityLedger:
    def __init__(self) -> None:
        self._object_tokens: dict[int, int] = {}
        # Keep every observed tensor wrapper alive.  Recording only ``id`` lets
        # CPython reuse an address after a temporary state-dict view dies,
        # turning a later unrelated tensor into a false alias.
        self._object_references: list[Tensor] = []
        self._storage_tokens: dict[tuple[str, int, int], int] = {}
        self.rows: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @staticmethod
    def _children(value: Any) -> Iterable[tuple[str, Any]]:
        if is_dataclass(value):
            return ((field.name, getattr(value, field.name)) for field in fields(value))
        if isinstance(value, Mapping):
            return ((str(key), item) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return ((str(index), item) for index, item in enumerate(value))
        return ()

    def observe(self, path: str, value: Any) -> None:
        if isinstance(value, Tensor):
            object_key = id(value)
            if object_key not in self._object_tokens:
                self._object_tokens[object_key] = len(self._object_tokens)
                self._object_references.append(value)
            object_token = self._object_tokens[object_key]
            storage = value.untyped_storage()
            storage_key = (str(value.device), int(storage.data_ptr()), int(storage.nbytes()))
            storage_token = self._storage_tokens.setdefault(
                storage_key, len(self._storage_tokens)
            )
            self.rows[path] = {
                "object": object_token,
                "storage": storage_token,
                "offset": int(value.storage_offset()),
                "stride": list(value.stride()),
            }
            return
        for name, child in self._children(value):
            self.observe(f"{path}/{name}", child)


def _resolve(root: Any, dotted: str) -> Any:
    value = root
    for part in dotted.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


LEGACY_TRACE_MODULES = (
    ("conditioning.history_proposal", "history_proposal", False),
    ("grounding.g1", "top.grounding_blocks.0", False),
    ("grounding.g2", "top.grounding_blocks.1", False),
    ("grounding.g3", "top.grounding_blocks.2", False),
    ("grounding.grounder", "top.grounder", False),
    ("intent.organizer", "top.intent", False),
    ("intent.coarse_action", "top.coarse_action", False),
    ("world.w1", "top.dynamics.w1", False),
    ("world.w2", "top.dynamics.w2", False),
    ("training.teacher", "top.teacher", False),
    ("training.recognizer", "top.recognizer", False),
    ("p1.static", "factual_reader", False),
    ("p1.dynamic", "bottom.p1_policy_block", False),
    ("policy.effect", "top.effect_reader", False),
    ("policy.consequence", "top.consequence", False),
    ("policy.p3", "top.plan_compiler", False),
    ("transition", "transition", False),
    ("execution.decoder", "bottom.decoder", False),
    ("execution.block0", "bottom.decoder.blocks.0", False),
    ("execution.block1", "bottom.decoder.blocks.1", False),
    ("execution.block2", "bottom.decoder.blocks.2", False),
    ("terminal.velocity", "bottom.decoder.velocity_head", False),
    ("terminal.motion", "bottom.decoder.motion_head", False),
    ("terminal.command", "bottom.decoder.gripper_command_head", True),
)


MODULAR_TRACE_MODULES = (
    ("conditioning.history_proposal", "conditioning.history_proposal", False),
    ("grounding.g1", "grounding.blocks.0", False),
    ("grounding.g2", "grounding.blocks.1", False),
    ("grounding.g3", "grounding.blocks.2", False),
    ("grounding.grounder", "grounding.grounder", False),
    ("intent.organizer", "intent.organizer", False),
    ("intent.coarse_action", "intent.coarse_action", False),
    ("world.w1", "world.dynamics.w1", False),
    ("world.w2", "world.dynamics.w2", False),
    ("training.teacher", "training_targets.teacher", False),
    ("training.recognizer", "training_targets.recognizer", False),
    ("p1.static", "p1.factual_reader", False),
    ("p1.dynamic", "p1.dynamic_policy_block", False),
    ("policy.effect", "policy_compiler.effect_reader", False),
    ("policy.consequence", "policy_compiler.consequence", False),
    ("policy.p3", "policy_compiler.plan_compiler", False),
    ("transition", "transition", False),
    ("execution.decoder", "execution_bottom.decoder", False),
    ("execution.block0", "execution_bottom.decoder.blocks.0", False),
    ("execution.block1", "execution_bottom.decoder.blocks.1", False),
    ("execution.block2", "execution_bottom.decoder.blocks.2", False),
    (
        "terminal.velocity",
        "execution_bottom.decoder.terminal_controller.velocity_head",
        False,
    ),
    (
        "terminal.motion",
        "execution_bottom.decoder.terminal_controller.motion_head",
        False,
    ),
    (
        "terminal.command",
        "execution_bottom.decoder.terminal_controller.optional_command_head",
        True,
    ),
)


class LifecycleTrace:
    def __init__(self, model: nn.Module, *, layout: str) -> None:
        self.events: list[str] = []
        self.handles: list[Any] = []
        specs = LEGACY_TRACE_MODULES if layout == "legacy" else MODULAR_TRACE_MODULES
        for event, path, optional in specs:
            try:
                module = _resolve(model, path)
            except (AttributeError, IndexError):
                if optional:
                    continue
                raise
            if module is None:
                if optional:
                    continue
                raise RuntimeError(f"required lifecycle module is absent: {path}")
            self.handles.append(module.register_forward_pre_hook(self._hook(event)))

    def _hook(self, event: str):
        def record(_module: nn.Module, _args: tuple[Any, ...]) -> None:
            self.events.append(event)

        return record

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _capture_structure(
    builder: Any,
    identities: IdentityLedger,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ownership: Any,
    config: Any,
    *,
    layout: str,
    init_rng_before: Tensor,
    init_rng_after: Tensor,
) -> None:
    state = _logical_state(model, layout=layout)
    builder.add("structure/config", config.as_dict())
    builder.add("structure/parameter_manifest", _parameter_manifest(model, layout=layout))
    builder.add("structure/parameter_aliases", _alias_manifest(model, layout=layout))
    builder.add("structure/state", state)
    builder.add("structure/state_key_order", list(state))
    builder.add("structure/optimizer", _optimizer_manifest(model, optimizer, layout=layout))
    builder.add(
        "structure/ownership",
        {
            "trainable_count": len(ownership.trainable_names),
            "frozen_count": len(ownership.frozen_names),
            "role_counts": ownership.role_counts,
            "group_names": ownership.group_names,
        },
    )
    builder.add("rng/model_init/before", init_rng_before)
    builder.add("rng/model_init/after", init_rng_after)
    identities.observe("structure/state", state)


def _capture_eval_boundaries(
    builder: Any,
    identities: IdentityLedger,
    modules: Mapping[str, ModuleType],
    model: nn.Module,
    config: Any,
    batch: Any,
    payload: Mapping[str, Any],
    *,
    layout: str,
    seed: int,
) -> None:
    _seed_all(_phase_seed(seed, "eval-boundaries"))
    builder.add("rng/eval/before", _torch_state())
    model.eval()
    trace = LifecycleTrace(model, layout=layout)
    with torch.no_grad():
        cache, training_state, metrics = model.encode_online(
            batch.online,
            training_mask=False,
            geometry_supervision=True,
            collect_diagnostics=True,
        )
        outputs: OrderedDict[str, Any] = OrderedDict()
        for point in TIME_POINTS:
            outputs[f"{point:.1f}"] = model.velocity(
                cache,
                noisy_action_field=payload["tensors"]["dynamic_physical_field"].clone(),
                time=torch.tensor([point], dtype=torch.float32),
                collect_diagnostics=True,
            )
    trace.close()
    builder.add("behavior/eval/cache", cache)
    builder.add("behavior/eval/training_state", training_state)
    builder.add("behavior/eval/static_metrics", metrics)
    builder.add("behavior/eval/dynamic", outputs)
    builder.add("behavior/eval/events", trace.events)
    builder.add("behavior/eval/event_counts", dict(Counter(trace.events)))
    builder.add("rng/eval/after", _torch_state())
    identities.observe("behavior/eval/cache", cache)
    identities.observe("behavior/eval/training_state", training_state)
    identities.observe("behavior/eval/dynamic", outputs)


def _capture_training(
    builder: Any,
    identities: IdentityLedger,
    modules: Mapping[str, ModuleType],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    batch: Any,
    *,
    layout: str,
    seed: int,
) -> None:
    flow_generator = torch.Generator(device="cpu").manual_seed(
        _phase_seed(seed, "flow-generator")
    )
    condition_generator = torch.Generator(device="cpu").manual_seed(
        _phase_seed(seed, "condition-generator")
    )
    schedule = modules["optimizer"].WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    engine = modules["engine"].MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
        train_flow_generator=flow_generator,
        train_condition_generator=condition_generator,
    )
    _seed_all(_phase_seed(seed, "training"))
    builder.add("rng/train/global_before", _torch_state())
    builder.add("rng/train/flow_before", _generator_state(flow_generator))
    builder.add("rng/train/condition_before", _generator_state(condition_generator))
    model.train()
    model.set_training_step(engine.global_step)
    optimizer.zero_grad(set_to_none=True)
    trace = LifecycleTrace(model, layout=layout)
    ledger, metrics = engine._forward(
        batch,
        training=True,
        collect_diagnostics=True,
        generator=flow_generator,
        condition_generator=condition_generator,
    )
    builder.add("behavior/train/ledger", ledger)
    builder.add("behavior/train/metrics", metrics)
    identities.observe("behavior/train/ledger", ledger)
    ledger.total.backward()
    raw = _gradient_tree(model, layout=layout)
    clip_snapshots: list[OrderedDict[str, Tensor | None]] = []
    original_clip = torch.nn.utils.clip_grads_with_norm_

    def tracked_clip(*args: Any, **kwargs: Any) -> Any:
        result = original_clip(*args, **kwargs)
        clip_snapshots.append(_gradient_tree(model, layout=layout))
        return result

    with mock.patch.object(torch.nn.utils, "clip_grads_with_norm_", new=tracked_clip):
        total_norm, gradient_metrics, gradient_norm_scalar = engine._gradient_lifecycle(
            collect_diagnostics=True
        )
    trace.close()
    if len(clip_snapshots) != 2:
        raise RuntimeError(
            "expected decoder-local and global clipping, got "
            f"{len(clip_snapshots)} stages"
        )
    builder.add("behavior/train/gradients/raw", raw)
    builder.add("behavior/train/gradients/postlocal", clip_snapshots[0])
    builder.add("behavior/train/gradients/postglobal", clip_snapshots[1])
    builder.add("behavior/train/gradient_metrics", gradient_metrics)
    builder.add("behavior/train/gradient_total_norm", total_norm)
    builder.add("behavior/train/gradient_norm_scalar", gradient_norm_scalar)
    builder.add("behavior/train/events", trace.events)
    builder.add("behavior/train/event_counts", dict(Counter(trace.events)))
    optimizer.step()
    schedule.step()
    engine.global_step += 1
    post_state = _logical_state(model, layout=layout)
    builder.add("behavior/train/post_step_state", post_state)
    builder.add(
        "behavior/train/optimizer_state",
        _optimizer_state_by_name(model, optimizer, layout=layout),
    )
    builder.add("behavior/train/optimizer_groups", _optimizer_manifest(model, optimizer, layout=layout))
    builder.add("behavior/train/scheduler", schedule.state_dict())
    builder.add("behavior/train/global_step", engine.global_step)
    builder.add("rng/train/global_after", _torch_state())
    builder.add("rng/train/flow_after", _generator_state(flow_generator))
    builder.add("rng/train/condition_after", _generator_state(condition_generator))
    identities.observe("behavior/train/gradients/raw", raw)
    identities.observe("behavior/train/gradients/postlocal", clip_snapshots[0])
    identities.observe("behavior/train/gradients/postglobal", clip_snapshots[1])
    identities.observe("behavior/train/post_step_state", post_state)


def _capture_deployment(
    builder: Any,
    identities: IdentityLedger,
    modules: Mapping[str, ModuleType],
    model: nn.Module,
    config: Any,
    batch: Any,
    payload: Mapping[str, Any],
    *,
    layout: str,
    seed: int,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(
        _phase_seed(seed, "deployment-generator")
    )
    _seed_all(_phase_seed(seed, "deployment"))
    builder.add("rng/deployment/global_before", _torch_state())
    builder.add("rng/deployment/generator_before", _generator_state(generator))
    model.eval()
    trace = LifecycleTrace(model, layout=layout)
    result = modules["sampling"].sample_action(
        model,
        batch.online,
        config,
        generator=generator,
        initial_physical_noise=payload["tensors"]["deployment_initial_noise"].clone(),
        collect_diagnostics=True,
        dtype=torch.float32,
    )
    trace.close()
    counts = Counter(trace.events)
    expected = {
        "conditioning.history_proposal": 1,
        "grounding.g1": 1,
        "grounding.g2": 1,
        "grounding.g3": 1,
        "p1.static": 1,
        "p1.dynamic": 12,
        "transition": 12,
        "execution.decoder": 12,
        "training.teacher": 0,
    }
    mismatches = {
        name: {"expected": value, "actual": counts.get(name, 0)}
        for name, value in expected.items()
        if counts.get(name, 0) != value
    }
    if mismatches:
        raise RuntimeError(f"deployment lifecycle mismatch: {mismatches}")
    builder.add("behavior/deployment/result", result)
    builder.add("behavior/deployment/events", trace.events)
    builder.add("behavior/deployment/event_counts", dict(counts))
    builder.add("behavior/deployment/expected_counts", expected)
    builder.add("rng/deployment/global_after", _torch_state())
    builder.add("rng/deployment/generator_after", _generator_state(generator))
    identities.observe("behavior/deployment/result", result)


def _capture_sidecars(
    builder: Any, model: nn.Module, *, layout: str
) -> None:
    decoder_path = "bottom.decoder" if layout == "legacy" else "execution_bottom.decoder"
    decoder = _resolve(model, decoder_path)
    progress: OrderedDict[str, Any] = OrderedDict()
    for step in (0, 199, 200, 700, 1_200):
        returned = model.set_training_step(step)
        progress[str(step)] = {
            "returned": float(returned),
            "persistent": decoder.execution_progress.detach().clone(),
            "python_fast_path": float(decoder._execution_progress_value),
        }
    nonpersistent: OrderedDict[str, Tensor] = OrderedDict()
    state_keys = set(model.state_dict())
    explicit_buffer_order = getattr(model, "_legacy_buffer_order", None)
    if layout == "modular" and explicit_buffer_order is not None:
        buffers = dict(model.named_buffers())
        buffer_rows = (
            (final_name, logical_parameter_name(final_name, layout=layout), buffers[final_name])
            for final_name, _logical in explicit_buffer_order
        )
    else:
        buffer_rows = (
            (name, logical_parameter_name(name, layout=layout), value)
            for name, value in model.named_buffers()
        )
    for name, logical_name, value in buffer_rows:
        if name not in state_keys:
            nonpersistent[logical_name] = value
    builder.add("behavior/sidecars/progress", progress)
    builder.add("behavior/sidecars/nonpersistent_buffers", nonpersistent)


def capture(
    *,
    source_root: Path,
    fixture_path: Path,
    output_dir: Path,
    layout: str,
    seed: int = DEFAULT_SEED,
) -> None:
    if layout not in {"legacy", "modular"}:
        raise ValueError("layout must be legacy or modular")
    GOLDEN._configure_determinism(seed)
    payload = _load_fixture(fixture_path)
    modules = _import_target(source_root)
    config = build_reduced_equivalence_config(modules)
    batch = _training_batch(modules, config, payload)
    _seed_all(_phase_seed(seed, "model-init"))
    init_rng_before = _torch_state()
    model = modules["policy"].ClearVLAMainlinePolicy(config).to(
        device=torch.device("cpu"), dtype=torch.float32
    )
    init_rng_after = _torch_state()
    optimizer, ownership = modules["optimizer"].build_optimizer(model, config)
    builder = GOLDEN.ArtifactBuilder()
    identities = IdentityLedger()
    _capture_structure(
        builder,
        identities,
        model,
        optimizer,
        ownership,
        config,
        layout=layout,
        init_rng_before=init_rng_before,
        init_rng_after=init_rng_after,
    )
    _capture_eval_boundaries(
        builder,
        identities,
        modules,
        model,
        config,
        batch,
        payload,
        layout=layout,
        seed=seed,
    )
    _capture_training(
        builder,
        identities,
        modules,
        model,
        optimizer,
        config,
        batch,
        layout=layout,
        seed=seed,
    )
    _capture_deployment(
        builder,
        identities,
        modules,
        model,
        config,
        batch,
        payload,
        layout=layout,
        seed=seed,
    )
    _capture_sidecars(builder, model, layout=layout)
    builder.add("behavior/identity_relations", identities.rows)
    metadata = {
        "variant": CAPTURE_PROFILE,
        "seed": int(seed),
        "fixture_sha256": _file_sha256(fixture_path),
        "time_points": list(TIME_POINTS),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "source": {
            "root": str(source_root.resolve()),
            "layout": layout,
            "python_tree_sha256": _source_digest(source_root.resolve()),
        },
    }
    builder.write(output_dir, metadata)
    print(f"wrote mainline equivalence capture: {output_dir}")


def compare(
    baseline: Path,
    candidate: Path,
    *,
    report: Path | None = None,
    max_differences: int = 40,
) -> bool:
    return bool(
        GOLDEN.compare(
            baseline,
            candidate,
            atol=0.0,
            rtol=0.0,
            max_differences=max_differences,
            report_path=report,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and compare deterministic current-mainline behavior."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture", help="create deterministic input tensors")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--seed", type=int, default=DEFAULT_SEED)
    capture_parser = subparsers.add_parser("capture", help="capture one source root")
    capture_parser.add_argument("--source-root", type=Path, required=True)
    capture_parser.add_argument("--fixture", type=Path, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--layout", choices=("legacy", "modular"), required=True)
    capture_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    compare_parser = subparsers.add_parser("compare", help="compare two captures exactly")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, default=None)
    compare_parser.add_argument("--max-differences", type=int, default=40)
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
            output_dir=args.output_dir,
            layout=args.layout,
            seed=args.seed,
        )
        return 0
    if args.command == "compare":
        passed = compare(
            args.baseline,
            args.candidate,
            report=args.report,
            max_differences=args.max_differences,
        )
        return 0 if passed else 1
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
