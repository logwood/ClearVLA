"""Independent frame support must not inherit another frame's augmentation."""
from dataclasses import fields
from types import SimpleNamespace
import unittest

import torch

from clearvla.mainline.causal_identity import causal_identity_metadata
from clearvla.mainline.executed_world import executed_world_metadata
from clearvla.mainline.instruction_change import instruction_change_metadata, POSTERIOR_REFERENCE_CHANGE
from clearvla.mainline.model.source_measurement import source_consistent_measurement
from clearvla.vision.entity_chart import CanonicalImageReadSource, CurrentImageSupport, current_image_grid


def fixture(masked=True):
    descriptor = torch.eye(9).reshape(1, 1, 3, 3, 9)
    observed = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    if masked:
        observed[..., 1] = False
    # One K reads visible first-column points. Use the actual production image
    # measure; fixture only supplies facts at this narrowly tested boundary.
    support = torch.zeros(1, 1, 1, 3, 3, dtype=torch.bool)
    support[..., 0] = True
    xy = current_image_grid(3, 3, device='cpu')[None, None, :, :, None, None]
    local_support = torch.ones(1, 1, 3, 3, 1, 1, dtype=torch.bool)
    spatial = CurrentImageSupport(xy, local_support.float(), local_support, torch.zeros_like(local_support, dtype=torch.float32))
    spatial.validate()
    source = CanonicalImageReadSource(torch.zeros_like(support, dtype=torch.float32), support, spatial)
    facts = SimpleNamespace(current_image_source=source,
        dense_chart=SimpleNamespace(dino_content=descriptor, cell_observed=observed[..., None]),
        objects=1, validity=torch.ones(1, 1, 1), camera_validity=torch.ones(1, 1, 1, 1))
    return facts, descriptor


def measure(facts, target, mode, observed=None):
    module = SimpleNamespace(observation_measurement_mode=mode, _flow_horizon_scale=lambda x: x.float())
    return source_consistent_measurement(module, facts, target, torch.tensor([[4]]), observed)


class IndependentObservationSupportTests(unittest.TestCase):
    def test_visible_source_can_move_into_source_augmentation_hole(self):
        facts, current = fixture()
        target = current.roll(1, 3)[:, None]
        old = measure(facts, target, 'source_consistent_v1')
        fixed = measure(facts, target, 'source_consistent_v2')
        self.assertEqual(old.candidate_posterior[..., 1].count_nonzero().item(), 0)
        torch.testing.assert_close(fixed.transport_per_support[..., 0], torch.ones(1, 1, 1, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(fixed.transport_per_support[..., 1], torch.zeros(1, 1, 1, 1), atol=1e-6, rtol=0)
        torch.testing.assert_close(fixed.successor_per_support, fixed.current_reference, atol=1e-6, rtol=0)
        torch.testing.assert_close(fixed.null_probability, torch.zeros_like(fixed.null_probability), atol=0, rtol=0)

    def test_missing_target_frame_is_still_unknown(self):
        facts, current = fixture()
        fixed = measure(facts, torch.full_like(current[:, None], float('nan')),
                        'source_consistent_v2', torch.zeros(1, 1, dtype=torch.bool))
        torch.testing.assert_close(fixed.null_probability, torch.ones_like(fixed.null_probability), atol=1e-7, rtol=0)
        self.assertEqual(fixed.transport_per_support.count_nonzero().item(), 0)
        torch.testing.assert_close(fixed.successor_per_support, fixed.current_reference, atol=0, rtol=0)

    def test_same_image_stays_exact_at_visible_sources(self):
        facts, current = fixture()
        fixed = measure(facts, current[:, None], 'source_consistent_v2')
        self.assertEqual(fixed.transport_per_support.count_nonzero().item(), 0)
        self.assertEqual(fixed.covariance_per_support.count_nonzero().item(), 0)
        torch.testing.assert_close(fixed.successor_per_support, fixed.current_reference, atol=0, rtol=0)

    def test_no_mask_is_exactly_historical(self):
        facts, current = fixture(masked=False)
        torch.manual_seed(413)
        target = torch.randn_like(current[:, None])
        old = measure(facts, target, 'source_consistent_v1')
        fixed = measure(facts, target, 'source_consistent_v2')
        for field in fields(old):
            torch.testing.assert_close(getattr(old, field.name), getattr(fixed, field.name), atol=0, rtol=0)

    def test_hidden_source_is_never_recovered_from_future(self):
        facts, current = fixture()
        facts.dense_chart.cell_observed.zero_()
        fixed = measure(facts, current[:, None], 'source_consistent_v2')
        self.assertEqual(fixed.candidate_posterior.count_nonzero().item(), 0)
        self.assertEqual(fixed.transport_per_support.count_nonzero().item(), 0)
        self.assertTrue(torch.equal(fixed.null_probability, torch.ones_like(fixed.null_probability)))

    def test_selector_is_explicit_and_old_metadata_unchanged(self):
        old = causal_identity_metadata({'observation_measurement_mode': 'source_consistent_v1'})
        fixed = causal_identity_metadata({'observation_measurement_mode': 'source_consistent_v2'})
        self.assertNotIn('observation_target_support', old)
        self.assertIn('independent-supplied-frame', fixed['observation_target_support'])
        self.assertEqual(executed_world_metadata('source_consistent_v1')['schema'], 'source-consistent-executed-outcome-v1')
        self.assertEqual(executed_world_metadata('source_consistent_v2')['schema'], 'source-consistent-executed-outcome-v2')
        self.assertEqual(instruction_change_metadata(('top', 'wrist'), POSTERIOR_REFERENCE_CHANGE, 'source_consistent_v1'),
                         instruction_change_metadata(('top', 'wrist'), POSTERIOR_REFERENCE_CHANGE, 'source_consistent_v2'))


if __name__ == '__main__':
    unittest.main()
