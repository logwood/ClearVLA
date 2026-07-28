import numpy as np
import torch

from clearvla.tools.probe_flow_dino_dataset_motion import (
    configured_pairs,
    local_oracle_match,
    pool_cached_tokens,
    summarize_matches,
)


def test_pool_cached_tokens_matches_equal_area_average() -> None:
    tokens = np.arange(2 * 1 * 16 * 2, dtype=np.float32).reshape(2, 1, 16, 2)
    pooled = pool_cached_tokens(tokens, np.asarray([1], dtype=np.int64), grid_size=2)
    expected = torch.from_numpy(tokens[1:2]).reshape(1, 1, 2, 2, 2, 2, 2).mean((3, 5))
    torch.testing.assert_close(pooled, expected)


def test_static_features_prefer_identity_with_no_oracle_headroom() -> None:
    source = torch.eye(16).reshape(1, 1, 4, 4, 16).expand(2, 2, -1, -1, -1)
    result = local_oracle_match(source, source.clone(), radius=2)
    assert not result.nonzero.any()
    assert result.mutual.all()
    torch.testing.assert_close(result.identity_cosine, torch.ones_like(result.identity_cosine))
    torch.testing.assert_close(result.oracle_cosine, result.identity_cosine)
    torch.testing.assert_close(result.identity_warp_error, torch.zeros_like(result.identity_warp_error))
    torch.testing.assert_close(result.oracle_warp_error, result.identity_warp_error)
    assert torch.isfinite(result.soft_warp_error).all()
    assert float(result.soft_warp_error.max()) < 1e-4


def test_local_oracle_recovers_one_patch_translation() -> None:
    grid = 4
    dim = grid * grid + 1
    source = torch.zeros(1, 1, grid, grid, dim)
    target = torch.zeros_like(source)
    for y in range(grid):
        for x in range(grid):
            source[0, 0, y, x, y * grid + x] = 1.0
            if x + 1 < grid:
                target[0, 0, y, x + 1, y * grid + x] = 1.0
    target[..., -1] = 0.1
    result = local_oracle_match(source, target, radius=1)
    interior = result.displacement.reshape(1, 1, grid, grid)[..., :-1]
    torch.testing.assert_close(interior, torch.ones_like(interior))
    assert result.nonzero.reshape(1, 1, grid, grid)[..., :-1].all()
    oracle_error = result.oracle_warp_error.reshape(1, 1, grid, grid)[..., :-1]
    identity_error = result.identity_warp_error.reshape(1, 1, grid, grid)[..., :-1]
    assert float(oracle_error.mean()) < float(identity_error.mean())


def test_summary_reports_global_and_motion_conditioned_gain() -> None:
    arrays = {
        "identity_cosine": np.asarray([0.9, 0.8, 0.1, 0.0], dtype=np.float32),
        "oracle_cosine": np.asarray([0.9, 0.85, 0.8, 0.9], dtype=np.float32),
        "identity_warp_error": np.asarray([0.1, 0.2, 0.9, 1.0], dtype=np.float32),
        "oracle_warp_error": np.asarray([0.1, 0.15, 0.2, 0.1], dtype=np.float32),
        "soft_warp_error": np.asarray([0.1, 0.18, 0.3, 0.2], dtype=np.float32),
        "displacement": np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32),
        "soft_displacement": np.asarray([0.01, 0.2, 0.7, 1.1], dtype=np.float32),
        "soft_entropy": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "soft_margin": np.asarray([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        "nonzero": np.asarray([False, True, True, True]),
        "mutual": np.asarray([True, True, False, True]),
        "camera_id": np.zeros(4, dtype=np.int16),
        "episode_id": np.zeros(4, dtype=np.int16),
    }
    summary = summarize_matches(arrays, motion_top_fraction=0.5)
    assert summary["warp_gain_ratio"] > 0.0
    assert summary["motion_top"]["patches"] == 2
    assert summary["motion_top"]["warp_gain_ratio"] > summary["warp_gain_ratio"]


def test_configured_pairs_keep_current_history_and_sparse_future_contract() -> None:
    pairs = configured_pairs((-8, -4, 0), (4, 12, 24), 48, (1, 4))
    assert pairs == ((-8, -4), (-4, 0), (0, 1), (0, 4), (0, 12), (0, 24), (0, 48))
