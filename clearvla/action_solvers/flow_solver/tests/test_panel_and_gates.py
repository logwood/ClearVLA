from __future__ import annotations

import pytest
import torch

from clearvla.action_solvers.flow_solver import (
    U0_REQUIRED_ACCOUNTING_KEYS,
    U0_REQUIRED_IDENTITY_KEYS,
    U0_REQUIRED_SCOPE_KEYS,
    PromotionGates,
    ReplayAttachment,
    candidate_by_name,
    run_candidate_panel,
)


def test_candidate_panel_reuses_values_but_not_mutable_cache_objects() -> None:
    initial = torch.zeros(1, 1)
    cache_counter = 0
    cache_ids: list[int] = []

    def cache_factory() -> dict[str, float]:
        nonlocal cache_counter
        cache_counter += 1
        cache = {"bias": float(cache_counter)}
        cache_ids.append(id(cache))
        return cache

    def field(state: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return torch.full_like(state, float(cache["bias"]))  # type: ignore[index]

    def endpoint(state: torch.Tensor, time: torch.Tensor, cache: object) -> str:
        return "endpoint"

    def rebuild(state: torch.Tensor, endpoint_value: object, cache: object) -> dict[str, float]:
        return {"bias": float(cache["bias"]) + 1.0}  # type: ignore[index]

    candidates = (
        candidate_by_name("E5/E5"),
        candidate_by_name("E5/DJ5(.5)"),
    )
    panel = run_candidate_panel(
        initial,
        field,
        cache_factory,
        rebuild,
        endpoint_head=endpoint,
        candidates=candidates,
        attachment_factory=lambda candidate, result: ReplayAttachment.from_sections(
            identity={"candidate": candidate.name},
            accounting={"physical_nfe": result.physical_nfe},
        ),
    )
    assert [record.candidate.name for record in panel.records] == [
        "E5/E5",
        "E5/DJ5(.5)",
    ]
    assert len(cache_ids) == 2
    assert panel.by_name("E5/DJ5(.5)").row()["initial_state_reused"] is True
    assert all(row["cache_identity_changed"] for row in panel.rows())
    assert panel.by_name("E5/E5").row()["proposal_endpoint_calls"] == 1
    assert panel.by_name("E5/E5").row()["refined_endpoint_calls"] == 1
    assert panel.by_name("E5/E5").row()["replay_attachment"]["identity"]["candidate"] == "E5/E5"  # type: ignore[index]


def test_panel_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        run_candidate_panel(
            torch.zeros(1, 1),
            lambda state, time, cache: torch.zeros_like(state),
            lambda: {},
            lambda state, endpoint, cache: {},
            endpoint_head=lambda state, time, cache: None,
            candidates=(),
        )


def test_panel_rejects_reused_initial_cache_objects() -> None:
    candidate = candidate_by_name("E5/E5")
    cache: dict[str, int] = {}
    with pytest.raises(ValueError, match="fresh cache"):
        run_candidate_panel(
            torch.zeros(1, 1),
            lambda state, time, cache: torch.zeros_like(state),
            lambda: cache,
            lambda state, endpoint, cache: {},
            endpoint_head=lambda state, time, cache: None,
            candidates=(candidate, candidate),
        )


def test_promotion_gates_are_fail_closed_until_all_four_are_explicit() -> None:
    assert "determinism" in PromotionGates.GATE_DESCRIPTIONS["U3"]
    assert "worst-case" in PromotionGates.GATE_DESCRIPTIONS["U3"]
    gates = PromotionGates(u0_interface=True, u1_numerical=True)
    assert gates.passed == ("U0", "U1")
    assert gates.missing == ("U2", "U3")
    assert gates.promotable is False
    with pytest.raises(RuntimeError, match="U2, U3"):
        gates.require_promotable()
    complete = PromotionGates(True, True, True, True)
    complete.require_promotable()
    assert complete.to_dict()["promotable"] is True


def test_replay_attachment_keeps_identity_and_metrics_outside_solver_core() -> None:
    attachment = ReplayAttachment.from_sections(
        identity={"checkpoint": "ckpt-hash", "seed": 7},
        scope={"rebuild_event_id": "w-1"},
        accounting={"diagnostic_nfe": 3},
        numerical={"dense_reference_rms": 0.1},
        outer_effect={"w_delta_rms": 0.2},
        behavior={"success": True},
    )
    data = attachment.to_dict()
    assert data["identity"]["checkpoint"] == "ckpt-hash"
    assert data["scope"]["rebuild_event_id"] == "w-1"
    assert data["accounting"]["diagnostic_nfe"] == 3
    with pytest.raises(ValueError, match="unique"):
        ReplayAttachment(identity=(("seed", 1), ("seed", 2)))


def _complete_u0_attachment() -> ReplayAttachment:
    identity: dict[str, object] = {key: key for key in U0_REQUIRED_IDENTITY_KEYS}
    identity["seed"] = 7
    scope: dict[str, object] = {key: key for key in U0_REQUIRED_SCOPE_KEYS}
    scope["initial_state_equal"] = True
    accounting: dict[str, object] = {key: 1 for key in U0_REQUIRED_ACCOUNTING_KEYS}
    accounting["total_dynamic_calls"] = 4
    return ReplayAttachment.from_sections(
        identity=identity,
        scope=scope,
        accounting=accounting,
    )


def test_replay_attachment_fingerprint_and_u0_profile_are_strict() -> None:
    attachment = _complete_u0_attachment()
    fingerprint = attachment.validate_for_u0()
    assert attachment.u0_ready is True
    assert fingerprint == attachment.fingerprint
    assert attachment.as_state_dict()["schema_version"] == 1
    restored = ReplayAttachment.from_state_dict(attachment.as_state_dict())
    assert restored == attachment
    changed = ReplayAttachment.from_state_dict(
        {
            **attachment.as_state_dict(),
            "identity": {**dict(attachment.identity), "sample_id": "different"},
        }
    )
    assert changed.fingerprint != fingerprint

    reordered = ReplayAttachment.from_sections(
        identity=dict(reversed(tuple(attachment.identity))),
        scope=dict(reversed(tuple(attachment.scope))),
        accounting=dict(reversed(tuple(attachment.accounting))),
    )
    assert reordered.fingerprint == fingerprint

    incomplete = ReplayAttachment.from_sections(identity={"seed": 1})
    assert incomplete.u0_ready is False
    with pytest.raises(ValueError, match="U0-complete"):
        incomplete.validate_for_u0()
    blank = _complete_u0_attachment()
    blank_identity = dict(blank.identity)
    blank_identity["checkpoint_hash"] = "  "
    blank = ReplayAttachment.from_sections(
        identity=blank_identity,
        scope=dict(blank.scope),
        accounting=dict(blank.accounting),
    )
    with pytest.raises(ValueError, match="blank"):
        blank.validate_for_u0()
    bad_accounting = _complete_u0_attachment()
    accounting = dict(bad_accounting.accounting)
    accounting["total_dynamic_calls"] = 99
    bad_accounting = ReplayAttachment.from_sections(
        identity=dict(bad_accounting.identity),
        scope=dict(bad_accounting.scope),
        accounting=accounting,
    )
    with pytest.raises(ValueError, match="total_dynamic_calls"):
        bad_accounting.validate_for_u0()


def test_replay_attachment_fingerprint_rejects_nonfinite_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ReplayAttachment.from_sections(identity={"value": float("nan")}).fingerprint
    with pytest.raises(TypeError, match="JSON scalars"):
        ReplayAttachment.from_sections(identity={"value": object()}).fingerprint
    with pytest.raises(ValueError, match="schema"):
        ReplayAttachment.from_sections(schema_version=2)
    with pytest.raises(ValueError, match="unknown"):
        ReplayAttachment.from_state_dict({"unexpected": 1})


def test_candidate_panel_rejects_refined_cache_reuse_across_candidates() -> None:
    shared_refined: dict[str, int] = {}

    def rebuild(state: torch.Tensor, endpoint: object, cache: object) -> dict[str, int]:
        return shared_refined

    candidates = (candidate_by_name("E5/E5"), candidate_by_name("E5/DJ5(.5)"))
    with pytest.raises(ValueError, match="every candidate"):
        run_candidate_panel(
            torch.zeros(1, 1),
            lambda state, time, cache: torch.zeros_like(state),
            lambda: {},
            rebuild,
            endpoint_head=lambda state, time, cache: None,
            candidates=candidates,
        )
