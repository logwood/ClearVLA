from __future__ import annotations

from pathlib import Path

import pytest

from clearvla.data.index_contract import RowRequest, SourceFingerprint


def test_source_fingerprint_round_trip_and_content_change(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"clearvla")
    fingerprint = SourceFingerprint.from_path(source, mode="sha256")
    assert SourceFingerprint.from_json(fingerprint.to_json()) == fingerprint
    source.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="changed since admission"):
        fingerprint.validate(source)


def test_row_request_preserves_order_and_rejects_invalid_rows() -> None:
    request = RowRequest.of("action", [4, 1, 4])
    assert request.field == "action"
    assert request.rows == (4, 1, 4)
    with pytest.raises(ValueError, match="non-negative"):
        RowRequest.of("action", [-1])
    with pytest.raises(ValueError, match="at least one"):
        RowRequest.of("action", [])
    with pytest.raises(ValueError, match="rank-1"):
        RowRequest.of("action", [[1, 2]])
