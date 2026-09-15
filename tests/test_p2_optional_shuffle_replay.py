from types import SimpleNamespace

import pytest
import torch

from clearvla.tools.replay_p2_optional_shuffle import OptionalResidualShuffle, cross_episode_panel


def test_panel_uses_different_episodes_and_unique_spread_windows():
    refs = [SimpleNamespace(episode_idx=ep, center=i) for ep in range(5) for i in range(80)]
    panel = cross_episode_panel(refs)
    assert len(panel) == 8
    assert len(set(i for batch in panel for i in batch)) == 64
    assert {refs[i].episode_idx for batch in panel for i in batch} == set(range(5))
    for batch in panel:
        assert len(batch) == 8
        assert all(refs[batch[i]].episode_idx != refs[batch[(i - 1) % 8]].episode_idx for i in range(8))


def test_optional_only_shuffle_respects_flattened_sample_axis_and_identity():
    carrier = torch.randn(8 * 3 * 4, 1, 6)
    value = torch.randn_like(carrier)
    probe = OptionalResidualShuffle(8)
    result = (value, {})
    assert probe.hook(None, (carrier,), result) is result
    probe.reset("identity")
    assert probe.hook(None, (carrier,), result) is result
    probe.reset("shuffle")
    shuffled, _ = probe.hook(None, (carrier,), result)
    assert torch.equal(shuffled.reshape(8, -1)[0], value.reshape(8, -1)[7])
    assert torch.equal(shuffled.reshape(8, -1)[1], value.reshape(8, -1)[0])
    assert torch.equal(probe.records[0][0], carrier)
    assert probe.summary()["shuffled_delta_rms"] > 0
    probe.reset("shuffle")
    with pytest.raises(AssertionError, match="upstream"):
        probe.hook(None, (carrier + 1,), result)
