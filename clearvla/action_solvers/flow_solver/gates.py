"""Fail-closed promotion gates for a solver candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

GateName = Literal["U0", "U1", "U2", "U3"]


@dataclass(frozen=True)
class PromotionGates:
    """Evidence checklist; every gate defaults to not passed.

    U0 proves that a candidate can run through the existing instantaneous-field
    ABI.  U1 requires a relative numerical reference comparison, U2 requires
    behavior evidence (or predeclared non-inferiority paired with a material
    deployment gain), and U3 requires acceptable latency/memory, batching,
    determinism, and worst-case runtime.
    The class carries no metric thresholds because those belong to the
    experiment owner and its serialized run context.
    """

    GATE_DESCRIPTIONS: ClassVar[dict[GateName, str]] = {
        "U0": "existing checkpoint/interface can execute the candidate",
        "U1": "relative to E5/E5, candidate reduces or does not materially worsen error to a declared dense reference with per-pass and band breakdown",
        "U2": "candidate improves behavior, or demonstrates predeclared non-inferiority together with a material deployment benefit under U3 costs",
        "U3": "latency, memory, batching, determinism and worst-case runtime are acceptable",
    }

    u0_interface: bool = False
    u1_numerical: bool = False
    u2_behavior: bool = False
    u3_budget: bool = False

    @property
    def passed(self) -> tuple[GateName, ...]:
        values: list[GateName] = []
        if self.u0_interface:
            values.append("U0")
        if self.u1_numerical:
            values.append("U1")
        if self.u2_behavior:
            values.append("U2")
        if self.u3_budget:
            values.append("U3")
        return tuple(values)

    @property
    def missing(self) -> tuple[GateName, ...]:
        return tuple(name for name in ("U0", "U1", "U2", "U3") if name not in self.passed)

    @property
    def promotable(self) -> bool:
        return not self.missing

    def require_promotable(self) -> None:
        if not self.promotable:
            missing = ", ".join(self.missing)
            raise RuntimeError(f"solver candidate is not promotable; missing gates: {missing}")

    def to_dict(self) -> dict[str, object]:
        return {
            "U0_interface": self.u0_interface,
            "U1_numerical": self.u1_numerical,
            "U2_behavior": self.u2_behavior,
            "U3_budget": self.u3_budget,
            "passed": list(self.passed),
            "missing": list(self.missing),
            "promotable": self.promotable,
        }


__all__ = ["GateName", "PromotionGates"]
