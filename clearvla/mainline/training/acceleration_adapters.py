"""Adapters for version-specific capture preparation.

The CUDA-Graph backend imports only the generic contract.  Mainline-specific
knowledge lives here, so another architecture branch can provide a different
adapter without editing the backend itself.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from .acceleration_contract import module_structure_signature


@dataclass(frozen=True)
class MainlineTrainingAccelerationAdapter:
    """Capture preparation and topology identity for the active mainline.

    These flags are implementation-local rewrites whose independent
    equivalence gates already exist.  Keeping them in an adapter prevents the
    generic runner from reaching into ``execution_bottom`` or knowing the
    decoder's private fields.
    """

    name: str = "clearvla-mainline-cuda-graph-v1"

    def prepare(self, engine: Any) -> None:
        decoder = getattr(getattr(engine.model, "execution_bottom", None), "decoder", None)
        if decoder is None:
            return
        decoder._static_neutral_owner = True
        decoder._training_candidate_prefix_reuse = True

    def topology_signature(self, engine: Any) -> Hashable:
        decoder = getattr(getattr(engine.model, "execution_bottom", None), "decoder", None)
        progress = getattr(decoder, "_execution_progress_value", None)
        if progress is None:
            return ("static",)
        return (
            "execution-progress",
            "identity" if float(progress) <= 0.0 else "active",
        )

    def structure_signature(self, engine: Any) -> Hashable:
        return module_structure_signature(engine.model)


__all__ = ["MainlineTrainingAccelerationAdapter"]
