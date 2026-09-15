from __future__ import annotations

"""Explicit opt-in for retired RDT2 experiment entry points.

The source remains importable for reproducibility and checkpoint archaeology,
but user-facing CLIs must not silently compete with the consolidated mainline.
"""

import os


LEGACY_RDT2_ENV = "CLEARVLA_ENABLE_LEGACY_RDT2"


def legacy_rdt2_enabled() -> bool:
    return os.environ.get(LEGACY_RDT2_ENV, "").strip() == "1"


def require_legacy_rdt2_cli(name: str) -> None:
    if legacy_rdt2_enabled():
        return
    raise RuntimeError(
        f"{name} is a retired RDT2 experiment CLI and is shielded from normal use. "
        "Use clearvla.cli.train_rdt2_mainline / clearvla.cli.eval_rdt2_mainline for the active policy. "
        f"Set {LEGACY_RDT2_ENV}=1 only for an intentional legacy reproduction."
    )


__all__ = ["LEGACY_RDT2_ENV", "legacy_rdt2_enabled", "require_legacy_rdt2_cli"]
