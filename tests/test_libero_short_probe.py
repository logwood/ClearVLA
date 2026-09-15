from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from clearvla.benchmarks.libero_short_probe import (
    _resolve_flattened_qpos_offset,
    _resolve_free_joint_y,
    apply_free_joint_y_offset,
)


class _State:
    def __init__(self, flattened: np.ndarray) -> None:
        self._flattened = np.asarray(flattened, dtype=np.float64)

    def flatten(self) -> np.ndarray:
        return self._flattened.copy()


class _Model:
    # Joint 2 is the target object's free joint and starts at qpos address 9.
    jnt_type = np.asarray([3, 3, 0], dtype=np.int32)
    jnt_qposadr = np.asarray([0, 7, 9], dtype=np.int32)

    @staticmethod
    def joint_name2id(name: str) -> int:
        if name != "akita_black_bowl_1_joint0":
            raise ValueError(name)
        return 2


def _env(qpos: np.ndarray, *, qpos_prefix: np.ndarray) -> SimpleNamespace:
    qpos = np.asarray(qpos, dtype=np.float64)
    flattened = np.concatenate((np.asarray(qpos_prefix, dtype=np.float64), qpos, [8.0, 9.0]))
    sim = SimpleNamespace(
        model=_Model(),
        data=SimpleNamespace(qpos=qpos.copy()),
        get_state=lambda: _State(flattened),
    )
    return SimpleNamespace(sim=sim)


def test_short_probe_resolves_free_joint_y_and_flattened_qpos_prefix() -> None:
    qpos = np.linspace(0.1, 1.2, 12, dtype=np.float64)
    env = _env(qpos, qpos_prefix=np.asarray([0.0]))

    object_start, y_qpos_index = _resolve_free_joint_y(
        env, "akita_black_bowl_1_joint0"
    )
    qpos_flat_offset = _resolve_flattened_qpos_offset(env, state_width=15)

    assert object_start == 9
    assert y_qpos_index == 10
    assert qpos_flat_offset == 1
    assert qpos_flat_offset + y_qpos_index == 11


def test_short_probe_offset_edits_only_the_true_flattened_object_y() -> None:
    state = np.arange(20, dtype=np.float64)

    shifted, before, after = apply_free_joint_y_offset(
        state,
        qpos_size=12,
        qpos_flat_offset=1,
        y_qpos_index=10,
        offset_m=-0.03,
    )

    assert before == 11.0
    assert after == pytest.approx(10.97)
    assert shifted[11] == pytest.approx(10.97)
    # This is the neighboring coordinate that the superseded v1/v2 probe
    # accidentally edited by treating flattened state as bare qpos.
    assert shifted[10] == state[10]
    np.testing.assert_array_equal(np.delete(shifted, 11), np.delete(state, 11))


def test_short_probe_rejects_nonunique_qpos_location() -> None:
    qpos = np.zeros(12, dtype=np.float64)
    env = _env(qpos, qpos_prefix=np.asarray([0.0]))

    with pytest.raises(ValueError, match="uniquely locate qpos"):
        _resolve_flattened_qpos_offset(env, state_width=15)


def test_short_probe_rejects_non_free_target_joint() -> None:
    env = _env(np.arange(12, dtype=np.float64), qpos_prefix=np.asarray([0.0]))
    env.sim.model.jnt_type = np.asarray([3, 3, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="is not free"):
        _resolve_free_joint_y(env, "akita_black_bowl_1_joint0")
