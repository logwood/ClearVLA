"""Serialized identity for one shared train-only action/state normalizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from .normalizer import ArrayNormalizer

SHARED_NORMALIZER_SCHEMA = "clearvla-shared-train-normalizer-v1"


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_shared_normalizers(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    expected_profile_sha256: str,
    expected_train_episode_ids: Sequence[str],
    computed_action: ArrayNormalizer,
    computed_state: ArrayNormalizer,
) -> tuple[ArrayNormalizer, ArrayNormalizer, dict[str, object]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"shared normalizer artifact does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SHARED_NORMALIZER_SCHEMA:
        raise ValueError("unsupported shared normalizer artifact schema")
    recorded = str(payload.get("normalizer_sha256", ""))
    digest_payload = dict(payload)
    digest_payload.pop("normalizer_sha256", None)
    if recorded != canonical_digest(digest_payload):
        raise ValueError("shared normalizer artifact content digest is inconsistent")
    if payload.get("fit_scope") != "one_shared_normalizer_over_selected_train_split_only":
        raise ValueError("shared normalizer fit scope is invalid")
    if payload.get("per_task_normalizers") is not False:
        raise ValueError("per-task normalizers are forbidden")
    selection = payload.get("selection_manifest")
    profile = payload.get("action_profile")
    if not isinstance(selection, dict) or str(selection.get("selection_sha256", "")) != str(
        expected_selection_sha256
    ):
        raise ValueError("shared normalizer selection identity is stale")
    if not isinstance(profile, dict) or str(profile.get("sha256", "")) != str(
        expected_profile_sha256
    ):
        raise ValueError("shared normalizer action profile is stale")
    train_ids = [str(value) for value in expected_train_episode_ids]
    if (
        int(payload.get("train_episode_count", -1)) != len(train_ids)
        or str(payload.get("train_episode_inventory_sha256", ""))
        != canonical_digest(train_ids)
    ):
        raise ValueError("shared normalizer train episode identity is stale")
    action_value = payload.get("action")
    state_value = payload.get("state")
    if not isinstance(action_value, dict) or not isinstance(state_value, dict):
        raise ValueError("shared normalizer arrays are missing")
    action = ArrayNormalizer.from_dict(action_value)
    state = ArrayNormalizer.from_dict(state_value)
    if action.to_dict() != computed_action.to_dict() or state.to_dict() != computed_state.to_dict():
        raise ValueError("shared normalizer values differ from a fresh train-only fit")
    metadata: dict[str, object] = {
        "schema": SHARED_NORMALIZER_SCHEMA,
        "path": str(source.resolve()),
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "normalizer_sha256": recorded,
        "fit_scope": payload["fit_scope"],
        "train_episode_count": len(train_ids),
        "train_episode_inventory_sha256": payload[
            "train_episode_inventory_sha256"
        ],
        "per_task_normalizers": False,
    }
    return action, state, metadata


__all__ = ["SHARED_NORMALIZER_SCHEMA", "canonical_digest", "load_shared_normalizers"]
