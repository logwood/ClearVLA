from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from clearvla.vision.npy_rows import NpyRowReader


def test_npy_row_reader_preserves_requested_order_and_repetitions(tmp_path: Path) -> None:
    values = np.arange(8 * 2 * 3, dtype=np.float32).reshape(8, 2, 3)
    path = tmp_path / "values.npy"
    np.save(path, values)
    rows = np.asarray([7, 0, 2, 2, 1], dtype=np.int64)

    reader = NpyRowReader(max_open_files=1)
    actual = reader.read(path, rows, expected_shape=values.shape, expected_dtype=np.float32)
    assert np.array_equal(actual, values[rows])
    assert reader.open_file_count == 1
    assert reader.io_method in {"preadv", "pread", "locked_seek_readinto"}

    restored = pickle.loads(pickle.dumps(reader))
    assert restored.open_file_count == 0
    assert np.array_equal(restored.read(path, rows), values[rows])
    reader.close()
    restored.close()


def test_npy_row_reader_rejects_shape_dtype_and_bounds_mismatches(tmp_path: Path) -> None:
    values = np.arange(4 * 3, dtype=np.uint8).reshape(4, 3)
    path = tmp_path / "values.npy"
    np.save(path, values)
    reader = NpyRowReader()
    with pytest.raises(ValueError, match="shape changed"):
        reader.read(path, np.asarray([0]), expected_shape=(5, 3))
    with pytest.raises(ValueError, match="dtype changed"):
        reader.read(path, np.asarray([0]), expected_dtype=np.float16)
    with pytest.raises(IndexError, match="outside cache"):
        reader.read(path, np.asarray([4], dtype=np.int64))
    reader.close()
