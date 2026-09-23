"""Real G/P1 consumers at native patch counts, independent of 8x8 query canvas."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from test_mainline_checkpoint import _dataset
from test_mainline_endpoint_supervision import _config

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.qualification import synthetic_batch
from clearvla.vision.candidate_support import (
    FULL_POSTERIOR_SUPPORT,
    MOMENT_LOCAL_SUPPORT,
    candidate_support_metadata,
)


@pytest.mark.parametrize("count", [64, 144, 256])
def test_source_metadata_tracks_native_lattice_not_public_canvas(count: int) -> None:
    assert (
        candidate_support_metadata(FULL_POSTERIOR_SUPPORT, source_candidate_count=count)[
            "candidate_count"
        ]
        == count
    )
    assert (
        candidate_support_metadata(MOMENT_LOCAL_SUPPORT, source_candidate_count=count)[
            "candidate_count"
        ]
        == 49
    )


@pytest.mark.parametrize("count", [0, -1, True, 64.0])
def test_source_count_is_not_lossily_coerced(count: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        candidate_support_metadata(FULL_POSTERIOR_SUPPORT, source_candidate_count=count)


@pytest.mark.parametrize("patches,bf16", [(64, False), (144, False), (256, False), (256, True)])
def test_complete_native_support_reaches_g3_teacher_p1_and_action(patches: int, bf16: bool) -> None:
    c = _config()
    c = replace(c, dimensions=replace(c.dimensions, patches_per_camera=patches))
    c.validate()
    torch.manual_seed(841)
    m = ClearVLAMainlinePolicy(c)
    b, normalizer = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"))
    m.configure_action_normalizer(normalizer)
    m.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, training, _ = m.encode_online(b.online, collect_diagnostics=False)
        state = training.observation.progressive_state
        assert state.dynamic_fine_values is not None
        assert state.dynamic_fine_coordinates is not None
        assert state.coarse_logits is not None
        assert state.dynamic_fine_values.shape[-2] == patches
        assert state.dynamic_fine_coordinates.shape[-2] == patches
        assert state.bank.coarse_candidate_coordinates is not None
        assert state.bank.coarse_candidate_coordinates.shape[-2] == patches
        assert training.observation.grounding.context_mask.shape[-2:] == (8, 8)
        output = m.velocity(
            cache,
            noisy_action_field=torch.randn(1, 24, 18),
            time=torch.full((1,), 0.4),
            collect_diagnostics=False,
        )
        assert torch.isfinite(output.bottom.physical_velocity).all()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        targets, _ = m.build_training_targets(training, b.future)
    assert targets.teacher_dynamics is not None
    assert torch.isfinite(targets.teacher_dynamics.semantic_delta).all()
    # A conveniently truncated value axis is not accepted just because it
    # matches the historical hard-coded 64. The producer retains 256/144.
    damaged = replace(state, dynamic_fine_values=state.dynamic_fine_values[..., :49, :])
    with pytest.raises(ValueError, match="complete selected candidate support"):
        replace(training.observation, progressive_state=damaged).validate()
    wrong_source = replace(
        state,
        bank=replace(
            state.bank,
            coarse_candidate_coordinates=state.bank.coarse_candidate_coordinates[..., :49, :],
        ),
    )
    with pytest.raises(ValueError, match="source keys, coordinates"):
        replace(training.observation, progressive_state=wrong_source).validate()


@pytest.mark.parametrize("patches", [64, 256])
def test_deployment_abi_owns_actual_source_count_and_rejects_wrong_lattice(
    patches: int, tmp_path: Path
) -> None:
    c = _config()
    c = replace(c, dimensions=replace(c.dimensions, patches_per_camera=patches))
    b, n = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"))
    n = replace(n, mode="zscore")  # explicit mean 0/std 1 fixture for ABI, not fitted user data
    language = tmp_path / "language.pt"
    torch.save({"tokens": b.online.goal.tokens, "mask": b.online.goal.mask}, language)
    ds = replace(
        _dataset(),
        action_normalizer_sha256=canonical_sha256(n.to_dict()),
        state_normalizer_sha256=canonical_sha256(n.to_dict()),
    )
    ident = build_checkpoint_identity(
        c,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=ds,
        language=ArtifactIdentity.from_file("t5_goal", language),
        commit="8" * 40,
    )
    profile = resolve_action_state_profile("calvin_relative_7d_v1")
    abi = build_deployment_abi(
        c,
        ident,
        action_normalizer=n,
        state_normalizer=n,
        data_profile={
            **profile.as_dict(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        gripper_indices=(6,),
        goal_metadata={},
    )
    validate_deployment_abi(abi)
    assert candidate_metadata(abi)["candidate_count"] == patches
    for wrong in (49, 144, True):
        invalid = copy.deepcopy(abi)
        candidate_metadata(invalid)["candidate_count"] = wrong
        with pytest.raises(ValueError, match="candidate support"):
            validate_deployment_abi(invalid)
    if patches == 256:
        invalid = copy.deepcopy(abi)
        candidate_metadata(invalid)["candidate_count"] = 64
        with pytest.raises(ValueError, match="candidate support"):
            validate_deployment_abi(invalid)


def candidate_metadata(abi: dict[str, object]) -> dict[str, object]:
    observation = abi["observation"]
    assert isinstance(observation, dict)
    support = observation["candidate_support"]
    assert isinstance(support, dict)
    return support
