"""Keep probe command units aligned with the actual deployment boundary."""
from types import SimpleNamespace

import numpy as np
import torch

from clearvla.mainline.data.normalizer import ArrayNormalizer
from probes.probe_target_binding_measurement_v2 import _final_nodes


def _fixture():
    normalizer = ArrayNormalizer.from_dict({
        "mode": "zscore",
        "offset": [[1., -2., 0., 0., 0., 0., 0.5]],
        "scale": [[2., 4., 5., 1., 1., 1., 2.]],
        "mean": [[0.] * 7], "std": [[1.] * 7],
        "minimum": [[-1.] * 7], "maximum": [[1.] * 7],
    })
    chart = torch.tensor([[[3., 2., 5., 0., 0., 0., 0.5],
                           [-1., -6., -5., 0., 0., 0., 2.5]]])
    return normalizer, chart


def test_probe_decodes_arm_and_preserves_binary_gripper_command():
    normalizer, chart = _fixture()
    sampled = SimpleNamespace(action=chart, physical_field=torch.zeros(1, 2, 18),
                              gripper_command=torch.tensor([[-1., 1.]]))
    nodes = _final_nodes({}, sampled, arm_dim=6, action_normalizer=normalizer)
    expected = np.array([[1., 1., 1., 0., 0., 0., -1.],
                         [-1., -1., -1., 0., 0., 0., 1.]], dtype=np.float32)
    np.testing.assert_array_equal(nodes["final_native_action"], expected)
    np.testing.assert_array_equal(nodes["final_sampled_action_chart"], chart[0].numpy())
    np.testing.assert_array_equal(nodes["final_native_action_arm_prefix2"], expected[:, :6])
    np.testing.assert_array_equal(nodes["final_native_action_gripper_switches"], [[1.]])


def test_probe_decodes_continuous_gripper_when_no_binary_command_exists():
    normalizer, chart = _fixture()
    sampled = SimpleNamespace(action=chart, physical_field=torch.zeros(1, 2, 18),
                              gripper_command=None)
    nodes = _final_nodes({}, sampled, arm_dim=6, action_normalizer=normalizer)
    np.testing.assert_array_equal(nodes["final_native_action_gripper"], [[0.], [1.]])
    assert "final_gripper_command" not in nodes
