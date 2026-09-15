import json

import h5py
import numpy as np
import pytest

from clearvla.simulation.video import export_episode_video


def _episode(path, *, frames=3):
    top = np.zeros((frames, 8, 8, 3), dtype=np.uint8)
    wrist = np.zeros_like(top)
    top[..., 0] = np.arange(frames, dtype=np.uint8)[:, None, None]
    wrist[..., 1] = 255
    with h5py.File(path, "w") as stream:
        stream.attrs["environment_descriptor_json"] = json.dumps({"control_hz": 12.5})
        stream.create_dataset("observations/images/cam_high", data=top)
        stream.create_dataset("observations/images/cam_right_wrist", data=wrist)


def test_export_episode_video_defaults_to_side_by_side(tmp_path):
    episode = tmp_path / "episode.hdf5"
    _episode(episode)

    result = export_episode_video(episode)

    assert result["frames"] == 3
    assert result["fps"] == pytest.approx(12.5)
    assert result["width"] == 16
    assert result["height"] == 8
    assert result["layout"] == "side_by_side"
    assert result["codec"]
    assert (tmp_path / "episode_side_by_side.mp4").is_file()

    with pytest.raises(FileExistsError):
        export_episode_video(episode)


def test_export_episode_video_rejects_misaligned_cameras(tmp_path):
    episode = tmp_path / "bad.hdf5"
    _episode(episode, frames=2)
    with h5py.File(episode, "a") as stream:
        del stream["observations/images/cam_right_wrist"]
        stream.create_dataset(
            "observations/images/cam_right_wrist", data=np.zeros((1, 8, 8, 3), dtype=np.uint8)
        )

    with pytest.raises(ValueError, match="frame counts"):
        export_episode_video(episode)
