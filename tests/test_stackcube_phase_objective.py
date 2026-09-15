from __future__ import annotations

import torch

from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.training.losses import stackcube_phase_masks


def test_stackcube_phase_masks_cover_approach_transport_and_hold() -> None:
    # Close is event class 1 and open is event class 2 in the continuous
    # ManiSkill target convention.
    events = torch.tensor([[0, 0, 1, 0, 0, 0, 2, 0]])
    approach, transport, pre_hold = stackcube_phase_masks(
        events, torch.tensor([1.0])
    )
    assert torch.equal(approach, torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0.]]))
    assert torch.equal(transport, torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0.]]))
    assert torch.equal(pre_hold, torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0.]]))


def test_closed_boundary_starts_transport_without_close_event() -> None:
    events = torch.zeros((1, 5), dtype=torch.long)
    approach, transport, pre_hold = stackcube_phase_masks(
        events, torch.tensor([-1.0])
    )
    assert float(approach.sum()) == 0.0
    assert torch.equal(transport, torch.ones_like(transport))
    assert torch.equal(pre_hold, torch.ones_like(pre_hold))


def test_phase_objective_defaults_are_omitted_from_legacy_identity() -> None:
    config = config_from_mapping({})
    assert "phase_control_mode" not in config.as_dict()["objectives"]
