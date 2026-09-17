"""Build causally aligned LIBERO prefix and terminal-suffix datasets.

Official LIBERO ``create_dataset.py`` executes ``actions[t]`` before writing
``obs[t]``.  The released observation arrays are therefore post-action rows,
whereas closed-loop deployment observes before choosing the next action.  This
converter makes that boundary explicit:

* real observation row 0 is rendered from the recorded MuJoCo ``states[0]``;
* real observation row t>0 is the released post-action row t-1;
* the released final post-action row is the genuine terminal observation;
* terminal mode appends 48 absorbing actions and repeats that terminal row.

The legacy post-action state rows are retained in one numeric-only dataset so
model-only E8 continuation can use the exact same state normalizer.  They are
never exposed as policy evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np

from clearvla.data.hdf5_episode import (
    LIBERO_E8_STATE_NORMALIZER_REFERENCE,
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
)

from .common import (
    BENCHMARK_DATASET_SCHEMA,
    LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA,
    LIBERO_CONVERTER_SCHEMA,
    LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA,
    atomic_hdf5,
    atomic_json,
    audit_benchmark_dataset,
)
from .libero import _bddl_task_records
from .libero_eval import libero_policy_observation

LIBERO_TERMINAL_SUFFIX_ROWS = 48
LIBERO_CAUSAL_OBSERVATION_ALIGNMENT = (
    "pre-action-v2: row0 rendered from raw states[0]; row t>0 is released "
    "post-action observation t-1"
)
LIBERO_LEGACY_ASSET_PREFIX = (
    "/Users/yifengz/workspace/libero-dev/chiliocosm/assets/"
)


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _text(value: object, *, name: str) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{name} must be a scalar text attribute")
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty scalar text")
    return value.strip()


def _finite_dataset(
    group: h5py.Group,
    key: str,
    *,
    ndim: int,
    name: str,
    dtype: object = np.float64,
) -> np.ndarray:
    dataset = group.get(key)
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != ndim:
        raise ValueError(f"{name} must be an HDF5 dataset with ndim={ndim}")
    try:
        value = np.asarray(dataset, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _rgb(group: h5py.Group, key: str, *, name: str) -> np.ndarray:
    dataset = group.get(key)
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.ndim != 4
        or int(dataset.shape[-1]) != 3
        or dataset.dtype != np.dtype(np.uint8)
    ):
        raise ValueError(f"{name} must be uint8 [T,H,W,3]")
    return np.asarray(dataset)


def _observation_arrays(
    observation: Mapping[str, Any],
    *,
    quat_to_axisangle: Callable[[np.ndarray], np.ndarray],
    image_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = libero_policy_observation(
        observation,
        np.zeros(7, dtype=np.float32),
        quat_to_axisangle=quat_to_axisangle,
        expected_image_side=image_side,
    )
    state = np.asarray(projected.state, dtype=np.float32)
    top = np.asarray(projected.rgb["top"])
    wrist = np.asarray(projected.rgb["wrist"])
    if state.shape != (7,) or top.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError("LIBERO simulator observation projection changed")
    return state.copy(), top.copy(), wrist.copy()


def _refresh_observation(env: Any) -> Mapping[str, Any]:
    post_process = getattr(env, "_post_process", None)
    update_observables = getattr(env, "_update_observables", None)
    inner = getattr(env, "env", env)
    getter = getattr(inner, "_get_observations", None)
    if not all(
        callable(value)
        for value in (post_process, update_observables, getter)
    ):
        raise ValueError("LIBERO environment lacks the direct observation refresh path")
    post_process()
    update_observables(force=True)
    observation = getter()
    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO direct observation refresh returned a non-mapping")
    return observation


def _reset_controller_goals(env: Any, *, perform_reset: bool = True) -> int:
    robots = getattr(env, "robots", None)
    if not isinstance(robots, (tuple, list)) or not robots:
        inner = getattr(env, "env", env)
        robots = getattr(inner, "robots", None)
    if not isinstance(robots, (tuple, list)) or not robots:
        raise ValueError("LIBERO environment does not expose robot controllers")
    count = 0
    for robot in robots:
        reset_goal = getattr(getattr(robot, "controller", None), "reset_goal", None)
        if not callable(reset_goal):
            raise ValueError("LIBERO OSC controller lacks reset_goal")
        if perform_reset:
            reset_goal()
        count += 1
    return count


def _prepare_snapshot(
    env: Any,
    state: np.ndarray,
    *,
    model_xml: str | None,
    model_xml_postprocessor: Callable[[str, Mapping[str, object]], str] | None,
) -> tuple[Mapping[str, Any], float, int]:
    reset = getattr(env, "reset", None)
    if not callable(reset):
        raise ValueError("LIBERO environment lacks reset")
    reset()
    if model_xml is not None:
        reset_xml = getattr(env, "reset_from_xml_string", None)
        if not callable(reset_xml) or model_xml_postprocessor is None:
            raise ValueError("LIBERO environment cannot restore the recorded model XML")
        reset_xml(model_xml_postprocessor(model_xml, {}))
        # ``reset_from_xml_string`` replaces ``env.sim``.  Capture the new
        # wrapper after the replacement; retaining the pre-reload object can
        # leave a detached MjSim whose model has already been freed.
        simulator = getattr(env, "sim", None)
        simulator_reset = getattr(simulator, "reset", None)
        if not callable(simulator_reset):
            raise ValueError("LIBERO environment lacks a resettable simulator")
        simulator_reset()
    simulator = getattr(env, "sim", None)
    state_value = np.array(state, dtype=np.float64, copy=True)
    simulator_set_state = getattr(simulator, "set_state_from_flattened", None)
    if callable(simulator_set_state):
        # Match the official LIBERO writer: restore the flattened state
        # directly.  The wrapper's set_init_state helper also calls
        # check_success/post-process and can perturb the hidden OSC target.
        simulator_set_state(state_value)
    else:
        setter = getattr(env, "set_init_state", None)
        if not callable(setter):
            raise ValueError("LIBERO environment lacks a simulator state setter")
        setter(state_value)
    controller_count = _reset_controller_goals(env, perform_reset=False)
    forward = getattr(simulator, "forward", None)
    if not callable(forward):
        raise ValueError("LIBERO environment lacks sim.forward")
    forward()
    observation = _refresh_observation(env)
    get_state = getattr(env, "get_sim_state", None)
    if not callable(get_state):
        raise ValueError("LIBERO environment lacks get_sim_state")
    restored = np.asarray(get_state(), dtype=np.float64)
    expected = np.asarray(state, dtype=np.float64)
    if restored.shape != expected.shape or not np.isfinite(restored).all():
        raise ValueError("restored LIBERO simulator state shape changed")
    maximum_error = float(np.max(np.abs(restored - expected)))
    if maximum_error > 1.0e-10:
        raise ValueError(
            "LIBERO simulator did not retain the requested flattened state: "
            f"max_abs={maximum_error:.6g}"
        )
    return observation, maximum_error, controller_count


def _step_result(env: Any, action: np.ndarray) -> tuple[Mapping[str, Any], float, bool, bool]:
    result = env.step(np.array(action, dtype=np.float32, copy=True))
    if not isinstance(result, (tuple, list)) or len(result) not in {4, 5}:
        raise ValueError("LIBERO step must return a four- or five-tuple")
    observation = result[0]
    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO step observation must be a mapping")
    reward = float(result[1])
    if not np.isfinite(reward):
        raise ValueError("LIBERO step reward is non-finite")
    done = bool(result[2]) if len(result) == 4 else bool(result[2]) or bool(result[3])
    checker = getattr(env, "check_success", None)
    if not callable(checker):
        inner = getattr(env, "env", env)
        checker = getattr(inner, "_check_success", None)
    success = bool(checker()) if callable(checker) else done
    return observation, reward, done, success


def _replay_full_episode(
    env: Any,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    model_xml: str,
    model_xml_postprocessor: Callable[[str, Mapping[str, object]], str],
    initial_snapshot: tuple[float, int] | None = None,
) -> tuple[
    Mapping[str, Any],
    float,
    bool,
    bool,
    float,
    int,
    float,
    int,
    int,
]:
    """Replay an episode from its reset state, retaining OSC controller state.

    A LIBERO flattened MuJoCo state does not include the OSC controller's
    internal target.  Replaying only ``states[-1]`` plus ``actions[-1]`` can
    therefore change the outcome even when the recorded data are correct.
    Starting at ``states[0]`` and executing every action preserves that hidden
    target exactly as the official dataset writer does.  The replay is audit
    only; callers continue to serialize the raw post-action terminal row.
    """

    if states.ndim != 2 or actions.ndim != 2 or rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError("LIBERO terminal replay arrays have invalid ranks")
    if states.shape[0] != actions.shape[0] or rewards.shape != (actions.shape[0],):
        raise ValueError("LIBERO terminal replay arrays are not episode-aligned")
    if dones.shape != rewards.shape or actions.shape[1] != 7:
        raise ValueError("LIBERO terminal replay action/reward widths changed")

    if initial_snapshot is None:
        _, initial_state_error, controller_count = _prepare_snapshot(
            env,
            states[0],
            model_xml=model_xml,
            model_xml_postprocessor=model_xml_postprocessor,
        )
    else:
        # The caller has just restored states[0] and only projected that
        # observation.  Projection is read-only, so continue from the same
        # simulator/controller state instead of recompiling the XML.
        initial_state_error, controller_count = initial_snapshot
    get_state = getattr(env, "get_sim_state", None)
    max_state_error = 0.0
    intermediate_reward_mismatches = 0
    intermediate_done_count = 0
    final_observation: Mapping[str, Any] | None = None
    final_reward = 0.0
    final_done = False
    final_success = False
    for index, action in enumerate(actions):
        (
            final_observation,
            final_reward,
            final_done,
            final_success,
        ) = _step_result(env, action)
        if index < actions.shape[0] - 1:
            if final_done:
                intermediate_done_count += 1
            if not np.isclose(
                final_reward, float(rewards[index]), rtol=0.0, atol=1.0e-6
            ):
                intermediate_reward_mismatches += 1
            if callable(get_state):
                replayed = np.asarray(get_state(), dtype=np.float64)
                recorded = np.asarray(states[index + 1], dtype=np.float64)
                if replayed.shape != recorded.shape:
                    raise ValueError(
                        "LIBERO terminal replay state shape changed at "
                        f"row {index + 1}"
                    )
                max_state_error = max(
                    max_state_error,
                    float(np.max(np.abs(replayed - recorded))),
                )
    if final_observation is None:
        raise ValueError("LIBERO terminal replay produced no observation")
    return (
        final_observation,
        final_reward,
        final_done,
        final_success,
        initial_state_error,
        controller_count,
        max_state_error,
        intermediate_reward_mismatches,
        intermediate_done_count,
    )


def _raw_observation_state(obs: h5py.Group) -> np.ndarray:
    ee = _finite_dataset(obs, "ee_states", ndim=2, name="raw obs.ee_states")
    gripper = _finite_dataset(
        obs, "gripper_states", ndim=2, name="raw obs.gripper_states"
    )
    if ee.shape[1:] != (6,) or gripper.shape[1:] != (2,):
        raise ValueError("raw LIBERO EEF/gripper observation widths changed")
    opening = np.abs(gripper).sum(axis=1, keepdims=True)
    return np.concatenate((ee, opening), axis=1).astype(np.float32)


def _copy_attrs(stream: h5py.File) -> dict[str, object]:
    return {str(name): value for name, value in stream.attrs.items()}


def _write_episode(
    path: Path,
    *,
    attrs: Mapping[str, object],
    actions: np.ndarray,
    legacy_states: np.ndarray,
    aligned_states: np.ndarray,
    aligned_top: np.ndarray,
    aligned_wrist: np.ndarray,
    terminal_state: np.ndarray,
    terminal_top: np.ndarray,
    terminal_wrist: np.ndarray,
    include_terminal_suffix: bool,
    replay_audit: Mapping[str, float | int | bool],
) -> None:
    source_count = int(actions.shape[0])
    if include_terminal_suffix:
        absorbing = np.zeros((LIBERO_TERMINAL_SUFFIX_ROWS, 7), dtype=np.float32)
        absorbing[:, 6] = actions[-1, 6]
        output_actions = np.concatenate((actions, absorbing), axis=0)
        output_states = np.concatenate(
            (
                aligned_states,
                np.repeat(terminal_state[None], LIBERO_TERMINAL_SUFFIX_ROWS, axis=0),
            ),
            axis=0,
        )
        output_top = np.concatenate(
            (
                aligned_top,
                np.repeat(terminal_top[None], LIBERO_TERMINAL_SUFFIX_ROWS, axis=0),
            ),
            axis=0,
        )
        output_wrist = np.concatenate(
            (
                aligned_wrist,
                np.repeat(terminal_wrist[None], LIBERO_TERMINAL_SUFFIX_ROWS, axis=0),
            ),
            axis=0,
        )
        converter_schema = LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA
    else:
        output_actions = actions
        output_states = aligned_states
        output_top = aligned_top
        output_wrist = aligned_wrist
        converter_schema = LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA
    action_state = np.concatenate(
        (np.zeros((1, 7), dtype=np.float32), output_actions[:-1]), axis=0
    )

    def writer(temporary: Path) -> None:
        with h5py.File(temporary, "w") as stream:
            for name, value in attrs.items():
                stream.attrs[name] = value
            stream.attrs["converter_schema"] = converter_schema
            stream.attrs["observation_alignment"] = LIBERO_CAUSAL_OBSERVATION_ALIGNMENT
            stream.attrs["state_normalizer_reference_key"] = (
                "normalizer_reference_state"
            )
            stream.attrs["state_normalizer_reference_semantics"] = (
                LIBERO_E8_STATE_NORMALIZER_REFERENCE
            )
            stream.attrs["valid_center_start"] = 0
            stream.attrs["strict_valid_center_start"] = 24
            stream.attrs["strict_valid_center_end"] = source_count - 49
            stream.attrs["valid_center_end"] = (
                source_count - 1 if include_terminal_suffix else source_count - 49
            )
            if include_terminal_suffix:
                stream.attrs["terminal_state_index"] = source_count
                stream.attrs["source_action_count"] = source_count
                stream.attrs["terminal_padding_mode"] = (
                    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
                )
            for name, value in replay_audit.items():
                stream.attrs[f"terminal_replay_{name}"] = value
            stream.create_dataset("action", data=output_actions)
            stream.create_dataset("action_state", data=action_state)
            stream.create_dataset("state", data=output_states)
            stream.create_dataset(
                "normalizer_reference_state", data=legacy_states
            )
            images = stream.require_group("observations/images")
            images.create_dataset(
                "cam_high",
                data=output_top,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            images.create_dataset(
                "cam_right_wrist",
                data=output_wrist,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

    atomic_hdf5(path, writer)


def _default_runtime() -> tuple[
    Callable[..., Any],
    Callable[[np.ndarray], np.ndarray],
    Callable[[str, Mapping[str, object]], str],
]:
    # The released demonstrations contain absolute paths from the original
    # ``libero-dev/chiliocosm`` checkout.  The upstream helper only rewrites
    # paths containing the literal component ``libero`` and therefore leaves
    # those paths untouched.  Resolve the package-owned asset root once and
    # make the relocation explicit and fail-closed.
    import libero.libero as libero_package
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.utils import postprocess_model_xml as libero_postprocess
    from robosuite.utils.transform_utils import quat2axisangle

    package_file = getattr(libero_package, "__file__", None)
    if not package_file:
        raise RuntimeError("LIBERO package has no filesystem location")
    asset_root = (Path(package_file).resolve().parent / "assets").resolve()
    if not asset_root.is_dir():
        raise FileNotFoundError(f"LIBERO asset root is absent: {asset_root}")

    def postprocess_model_xml(
        xml: str, cameras: Mapping[str, object]
    ) -> str:
        processed = libero_postprocess(
            xml, dict(cameras), demo_generation=True
        )
        tree = ET.fromstring(processed)
        unresolved: list[str] = []
        relocated: list[tuple[str, str]] = []
        for element in tree.iter():
            old_file = element.get("file")
            if not old_file:
                continue
            normalized = old_file.replace("\\", "/")
            if normalized.startswith(LIBERO_LEGACY_ASSET_PREFIX):
                relative = normalized[len(LIBERO_LEGACY_ASSET_PREFIX) :]
                candidate = (asset_root / relative).resolve()
                try:
                    candidate.relative_to(asset_root)
                except ValueError as error:
                    raise ValueError(
                        f"LIBERO XML asset escapes package root: {old_file}"
                    ) from error
                if not candidate.is_file():
                    raise FileNotFoundError(
                        "LIBERO relocated XML asset is absent: "
                        f"{old_file} -> {candidate}"
                    )
                element.set("file", str(candidate))
                relocated.append((old_file, str(candidate)))
                continue
            current = Path(normalized)
            if current.is_absolute() and not current.is_file():
                unresolved.append(old_file)
        if unresolved:
            preview = ", ".join(unresolved[:4])
            suffix = " ..." if len(unresolved) > 4 else ""
            raise FileNotFoundError(
                "LIBERO XML contains unresolved absolute assets: "
                f"{preview}{suffix}"
            )
        if not relocated:
            # Keep a guard against accidentally running a different converter
            # or a dataset whose XML no longer carries the expected provenance.
            raise ValueError(
                "LIBERO XML asset relocation found no legacy chiliocosm paths"
            )
        return ET.tostring(tree, encoding="utf8").decode("utf8")

    setattr(postprocess_model_xml, "asset_relocation_root", str(asset_root))
    setattr(
        postprocess_model_xml,
        "asset_relocation_source_prefix",
        LIBERO_LEGACY_ASSET_PREFIX,
    )
    return OffScreenRenderEnv, quat2axisangle, postprocess_model_xml


def convert_libero_causal_boundaries(
    input_root: str | Path,
    output_root: str | Path,
    *,
    raw_source: str | Path,
    evaluator_bddl_root: str | Path,
    include_terminal_suffix: bool,
    seed: int = 0,
    env_factory: Callable[..., Any] | None = None,
    quat_to_axisangle: Callable[[np.ndarray], np.ndarray] | None = None,
    model_xml_postprocessor: (
        Callable[[str, Mapping[str, object]], str] | None
    ) = None,
) -> dict[str, Any]:
    """Create one new audited root without mutating the legacy conversion."""

    source = Path(input_root).expanduser()
    output = Path(output_root).expanduser()
    raw_root = Path(raw_source).expanduser()
    bddl_root = Path(evaluator_bddl_root).expanduser()
    if not source.is_dir() or not raw_root.is_dir() or not bddl_root.is_dir():
        raise FileNotFoundError("LIBERO converted/raw/BDDL roots must all exist")
    if not isinstance(include_terminal_suffix, bool):
        raise ValueError("include_terminal_suffix must be boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("LIBERO causal conversion seed must be non-negative")
    if output.is_symlink():
        raise FileExistsError(f"refusing LIBERO output symlink: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing non-empty LIBERO output root: {output}")

    input_audit = audit_benchmark_dataset(source)
    manifest = _json_object(source / "dataset_manifest.json", name="dataset manifest")
    if (
        input_audit.get("benchmark") != "LIBERO"
        or manifest.get("converter_schema") != LIBERO_CONVERTER_SCHEMA
    ):
        raise ValueError("causal conversion requires the audited legacy LIBERO v1 root")
    split_payload = _json_object(source / "splits.json", name="split manifest")
    instruction_payload = _json_object(
        source / "instructions.json", name="instruction inventory"
    )
    splits = split_payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("LIBERO split manifest has no split mapping")
    episode_names = [
        str(name)
        for split in ("train", "val", "test")
        for name in splits[split]
    ]
    episode_by_task: dict[tuple[str, str], list[str]] = {}
    for name in episode_names:
        path = source / f"{name}.hdf5"
        with h5py.File(path, "r") as stream:
            suite = _text(stream.attrs.get("suite"), name=f"{path} suite")
            task_file = _text(
                stream.attrs.get("task_file"), name=f"{path} task_file"
            )
        episode_by_task.setdefault((suite, task_file), []).append(name)

    suites = tuple(str(value) for value in manifest.get("suites", ()))
    records = _bddl_task_records(bddl_root, suites=suites)
    bddl_by_task = {
        (record.suite.lower(), record.task_name.lower()): record.path
        for record in records
    }
    imported_xml_postprocessor = None
    if env_factory is None or quat_to_axisangle is None:
        default_env, default_quat, default_xml = _default_runtime()
        env_factory = default_env if env_factory is None else env_factory
        quat_to_axisangle = default_quat if quat_to_axisangle is None else quat_to_axisangle
        imported_xml_postprocessor = default_xml
    if model_xml_postprocessor is None:
        model_xml_postprocessor = (
            imported_xml_postprocessor
            if imported_xml_postprocessor is not None
            else lambda xml, _cameras: xml
        )
    asset_relocation_root = getattr(
        model_xml_postprocessor, "asset_relocation_root", None
    )
    asset_relocation_source_prefix = getattr(
        model_xml_postprocessor, "asset_relocation_source_prefix", None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    committed = False
    episode_audits: list[dict[str, float | int | bool]] = []
    env_count = 0
    try:
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("legacy LIBERO manifest has no task inventory")
        for task_index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                raise ValueError("legacy LIBERO task inventory is malformed")
            suite = str(task.get("suite", ""))
            task_file = str(task.get("task_file", ""))
            evaluator_task_name = str(task.get("evaluator_task_name", ""))
            names = episode_by_task.get((suite, task_file), [])
            if not names:
                raise ValueError(f"LIBERO task {suite}/{task_file} has no episodes")
            bddl_path = bddl_by_task.get(
                (suite.lower(), evaluator_task_name.lower())
            )
            if bddl_path is None:
                raise ValueError(
                    f"LIBERO evaluator BDDL task is absent: {suite}/{evaluator_task_name}"
                )
            raw_path = raw_root / suite / task_file
            if not raw_path.is_file():
                raise FileNotFoundError(f"raw LIBERO task file is absent: {raw_path}")

            first_episode = source / f"{names[0]}.hdf5"
            with h5py.File(first_episode, "r") as first:
                top_shape = tuple(first["observations/images/cam_high"].shape[1:])
                wrist_shape = tuple(
                    first["observations/images/cam_right_wrist"].shape[1:]
                )
            if (
                len(top_shape) != 3
                or top_shape != wrist_shape
                or top_shape[0] != top_shape[1]
                or top_shape[2] != 3
            ):
                raise ValueError("terminal replay requires equal square LIBERO cameras")
            image_side = int(top_shape[0])
            env = env_factory(
                bddl_file_name=str(bddl_path),
                camera_heights=image_side,
                camera_widths=image_side,
            )
            env_count += 1
            try:
                seeder = getattr(env, "seed", None)
                if not callable(seeder):
                    raise ValueError("LIBERO environment lacks seed")
                seeder(seed + task_index)
                with h5py.File(raw_path, "r") as raw_stream:
                    raw_data = raw_stream.get("data")
                    if not isinstance(raw_data, h5py.Group):
                        raise ValueError(f"{raw_path} has no data group")
                    for name in names:
                        input_path = source / f"{name}.hdf5"
                        with h5py.File(input_path, "r") as legacy:
                            attrs = _copy_attrs(legacy)
                            demo_name = _text(
                                legacy.attrs.get("source_demo"),
                                name=f"{input_path} source_demo",
                            )
                            raw_demo = raw_data.get(demo_name)
                            if not isinstance(raw_demo, h5py.Group):
                                raise ValueError(
                                    f"raw LIBERO demo is absent: {raw_path}:data/{demo_name}"
                                )
                            actions = np.asarray(legacy["action"], dtype=np.float32)
                            legacy_states = np.asarray(legacy["state"], dtype=np.float32)
                            legacy_top = np.asarray(
                                legacy["observations/images/cam_high"]
                            )
                            legacy_wrist = np.asarray(
                                legacy["observations/images/cam_right_wrist"]
                            )
                            raw_actions = _finite_dataset(
                                raw_demo,
                                "actions",
                                ndim=2,
                                name=f"{raw_path}:data/{demo_name}/actions",
                            ).astype(np.float32)
                            raw_states = _finite_dataset(
                                raw_demo,
                                "states",
                                ndim=2,
                                name=f"{raw_path}:data/{demo_name}/states",
                            )
                            rewards = _finite_dataset(
                                raw_demo,
                                "rewards",
                                ndim=1,
                                name=f"{raw_path}:data/{demo_name}/rewards",
                            )
                            dones = _finite_dataset(
                                raw_demo,
                                "dones",
                                ndim=1,
                                name=f"{raw_path}:data/{demo_name}/dones",
                            )
                            raw_obs = raw_demo.get("obs")
                            if not isinstance(raw_obs, h5py.Group):
                                raise ValueError("raw LIBERO demo has no obs group")
                            raw_observation_states = _raw_observation_state(raw_obs)
                            raw_top = _rgb(
                                raw_obs,
                                "agentview_rgb",
                                name="raw obs.agentview_rgb",
                            )
                            raw_wrist = _rgb(
                                raw_obs,
                                "eye_in_hand_rgb",
                                name="raw obs.eye_in_hand_rgb",
                            )
                            length = int(actions.shape[0])
                            if (
                                actions.shape != (length, 7)
                                or raw_actions.shape != actions.shape
                                or raw_states.shape[0] != length
                                or rewards.shape != (length,)
                                or dones.shape != (length,)
                                or legacy_states.shape != (length, 7)
                                or raw_observation_states.shape != legacy_states.shape
                                or raw_top.shape != legacy_top.shape
                                or raw_wrist.shape != legacy_wrist.shape
                            ):
                                raise ValueError("legacy/raw LIBERO episode alignment changed")
                            if not np.array_equal(actions, raw_actions):
                                raise ValueError("legacy converted action differs from raw demo")
                            # The legacy converter stores float32 EEF/gripper
                            # values, while the raw LIBERO observation is
                            # float64.  Recomputing the final opening scalar
                            # can therefore differ by a few float32 ulps even
                            # when the state chart is identical.
                            if not np.allclose(
                                legacy_states,
                                raw_observation_states,
                                rtol=0.0,
                                atol=1.0e-6,
                            ):
                                raise ValueError(
                                    "legacy converted state differs from raw post-action obs"
                                )
                            if not np.array_equal(legacy_top, raw_top) or not np.array_equal(
                                legacy_wrist, raw_wrist
                            ):
                                raise ValueError("legacy converted RGB differs from raw post-action obs")

                            raw_model = raw_demo.attrs.get("model_file")
                            model_xml = (
                                None
                                if raw_model is None
                                else _text(raw_model, name="raw model_file")
                            )
                            initial_observation, initial_state_error, controllers = (
                                _prepare_snapshot(
                                    env,
                                    raw_states[0],
                                    model_xml=model_xml,
                                    model_xml_postprocessor=model_xml_postprocessor,
                                )
                            )
                            initial_state, initial_top, initial_wrist = _observation_arrays(
                                initial_observation,
                                quat_to_axisangle=quat_to_axisangle,
                                image_side=image_side,
                            )
                            aligned_states = np.concatenate(
                                (initial_state[None], legacy_states[:-1]), axis=0
                            )
                            aligned_top = np.concatenate(
                                (initial_top[None], legacy_top[:-1]), axis=0
                            )
                            aligned_wrist = np.concatenate(
                                (initial_wrist[None], legacy_wrist[:-1]), axis=0
                            )

                            (
                                replay_observation,
                                replay_reward,
                                replay_done,
                                replay_success,
                                replay_initial_state_error,
                                replay_controllers,
                                replay_state_error,
                                replay_intermediate_reward_mismatches,
                                replay_intermediate_done_count,
                            ) = _replay_full_episode(
                                env,
                                states=raw_states,
                                actions=actions,
                                rewards=rewards,
                                dones=dones,
                                model_xml=model_xml,
                                model_xml_postprocessor=model_xml_postprocessor,
                                initial_snapshot=(initial_state_error, controllers),
                            )
                            expected_reward = float(rewards[-1])
                            expected_done = bool(dones[-1])
                            expected_success = bool(expected_reward > 0.0)
                            replay_reward_match = bool(
                                np.isclose(
                                    replay_reward,
                                    expected_reward,
                                    rtol=0.0,
                                    atol=1.0e-6,
                                )
                            )
                            replay_done_match = bool(replay_done == expected_done)
                            replay_success_match = bool(
                                replay_success == expected_success
                            )
                            replay_terminal_state, replay_top, replay_wrist = (
                                _observation_arrays(
                                    replay_observation,
                                    quat_to_axisangle=quat_to_axisangle,
                                    image_side=image_side,
                                )
                            )
                            terminal_state = legacy_states[-1].copy()
                            terminal_top = legacy_top[-1].copy()
                            terminal_wrist = legacy_wrist[-1].copy()
                            replay_audit: dict[str, float | int | bool] = {
                                "restored_initial_state_max_abs": initial_state_error,
                                "replay_initial_state_max_abs": replay_initial_state_error,
                                "replay_state_max_abs_to_recorded_successor": replay_state_error,
                                "controller_count": controllers,
                                "replay_controller_count": replay_controllers,
                                "replay_intermediate_reward_mismatches": (
                                    replay_intermediate_reward_mismatches
                                ),
                                "replay_intermediate_done_count": replay_intermediate_done_count,
                                "raw_reward": expected_reward,
                                "raw_done": expected_done,
                                "raw_success_label": expected_success,
                                "reward": replay_reward,
                                "done": replay_done,
                                "success": replay_success,
                                "replay_reward_match": replay_reward_match,
                                "replay_done_match": replay_done_match,
                                "replay_success_match": replay_success_match,
                                "replay_label_match": bool(
                                    replay_reward_match
                                    and replay_done_match
                                    and replay_success_match
                                ),
                                "terminal_state_max_abs_to_recorded": float(
                                    np.max(np.abs(replay_terminal_state - terminal_state))
                                ),
                                "terminal_top_mae_to_recorded": float(
                                    np.mean(
                                        np.abs(
                                            replay_top.astype(np.float32)
                                            - terminal_top.astype(np.float32)
                                        )
                                    )
                                ),
                                "terminal_wrist_mae_to_recorded": float(
                                    np.mean(
                                        np.abs(
                                            replay_wrist.astype(np.float32)
                                            - terminal_wrist.astype(np.float32)
                                        )
                                    )
                                ),
                            }
                            episode_audits.append(replay_audit)
                            _write_episode(
                                staging / f"{name}.hdf5",
                                attrs=attrs,
                                actions=actions,
                                legacy_states=legacy_states,
                                aligned_states=aligned_states,
                                aligned_top=aligned_top,
                                aligned_wrist=aligned_wrist,
                                terminal_state=terminal_state,
                                terminal_top=terminal_top,
                                terminal_wrist=terminal_wrist,
                                include_terminal_suffix=include_terminal_suffix,
                                replay_audit=replay_audit,
                            )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

        if len(episode_audits) != len(episode_names) or env_count != len(tasks):
            raise AssertionError("LIBERO causal converter lost task/episode ownership")
        converter_schema = (
            LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA
            if include_terminal_suffix
            else LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA
        )
        boundary_contract = (
            "causal_prefix_terminal_suffix_v2"
            if include_terminal_suffix
            else "causal_prefix_v1"
        )
        converted_manifest = dict(manifest)
        converted_manifest.update(
            {
                "schema": BENCHMARK_DATASET_SCHEMA,
                "converter_schema": converter_schema,
                "source_converter_schema": LIBERO_CONVERTER_SCHEMA,
                "source_converted_root": str(source.resolve()),
                "source": str(raw_root.resolve()),
                "window_boundary_contract": boundary_contract,
                "observation_alignment": LIBERO_CAUSAL_OBSERVATION_ALIGNMENT,
                "state_normalizer_reference_semantics": (
                    LIBERO_E8_STATE_NORMALIZER_REFERENCE
                ),
                "valid_center_start": 0,
                "valid_center_end": (
                    "source_action_count - 1"
                    if include_terminal_suffix
                    else "length - 49"
                ),
                "strict_valid_center_start": 24,
                "strict_valid_center_end": (
                    "source_action_count - 49"
                    if include_terminal_suffix
                    else "length - 49"
                ),
                "terminal_replay": {
                    "episodes": len(episode_audits),
                    "environments": env_count,
                    "one_environment_per_task": True,
                    "reward_done_success_exact": all(
                        bool(row["replay_label_match"]) for row in episode_audits
                    ),
                    "replay_label_mismatch_episodes": sum(
                        not bool(row["replay_label_match"])
                        for row in episode_audits
                    ),
                    "raw_terminal_label_semantics": (
                        "official LIBERO create_dataset terminal marker; "
                        "simulator replay is audit-only"
                    ),
                    "restored_state_max_abs": max(
                        max(
                            float(row["restored_initial_state_max_abs"]),
                            float(row["replay_initial_state_max_abs"]),
                        )
                        for row in episode_audits
                    ),
                    "replay_state_max_abs_to_recorded_successor": max(
                        float(row["replay_state_max_abs_to_recorded_successor"])
                        for row in episode_audits
                    ),
                    "replay_intermediate_reward_mismatches": sum(
                        int(row["replay_intermediate_reward_mismatches"])
                        for row in episode_audits
                    ),
                    "replay_intermediate_done_count": sum(
                        int(row["replay_intermediate_done_count"])
                        for row in episode_audits
                    ),
                    "replay_mode": "full_episode_from_states[0]",
                    "recorded_terminal_source": "raw post-action observation[-1]",
                    "replay_observation_use": "audit only; hidden OSC goal is not serialized",
                },
            }
        )
        if asset_relocation_root is not None:
            converted_manifest["model_xml_asset_relocation"] = {
                "source_prefix": str(asset_relocation_source_prefix),
                "target_root": str(asset_relocation_root),
                "validation": "every absolute XML asset exists before reset",
            }
        if include_terminal_suffix:
            converted_manifest.update(
                {
                    "terminal_padding_mode": (
                        LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
                    ),
                    "terminal_suffix_rows": LIBERO_TERMINAL_SUFFIX_ROWS,
                    "source_action_count_semantics": "per-episode original action length",
                }
            )
        else:
            converted_manifest.pop("terminal_padding_mode", None)
            converted_manifest.pop("terminal_suffix_rows", None)
            converted_manifest.pop("source_action_count_semantics", None)
        atomic_json(staging / "dataset_manifest.json", converted_manifest)
        atomic_json(staging / "splits.json", split_payload)
        atomic_json(staging / "instructions.json", instruction_payload)
        report = audit_benchmark_dataset(staging)
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError(f"refusing non-empty LIBERO output root: {output}")
            output.rmdir()
        os.replace(staging, output)
        committed = True
        return report
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def convert_libero_causal_prefix(
    input_root: str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return convert_libero_causal_boundaries(
        input_root,
        output_root,
        include_terminal_suffix=False,
        **kwargs,
    )


def augment_libero_terminal_suffix(
    input_root: str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return convert_libero_causal_boundaries(
        input_root,
        output_root,
        include_terminal_suffix=True,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--evaluator-bddl-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefix", "terminal"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = convert_libero_causal_boundaries(
        args.input_root,
        args.output_root,
        raw_source=args.raw_source,
        evaluator_bddl_root=args.evaluator_bddl_root,
        include_terminal_suffix=args.mode == "terminal",
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "LIBERO_CAUSAL_OBSERVATION_ALIGNMENT",
    "LIBERO_TERMINAL_SUFFIX_ROWS",
    "augment_libero_terminal_suffix",
    "convert_libero_causal_boundaries",
    "convert_libero_causal_prefix",
]
