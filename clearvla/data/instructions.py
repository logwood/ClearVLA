"""Stable instruction identities shared by external benchmark adapters."""

from __future__ import annotations

import hashlib
import re


def normalize_instruction(value: str) -> str:
    """Normalize irrelevant whitespace while preserving task wording."""

    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        raise ValueError("language instruction must be non-empty")
    return text


def instruction_key(value: str) -> str:
    """Return a collision-resistant key for one normalized instruction."""

    text = normalize_instruction(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["instruction_key", "normalize_instruction"]
