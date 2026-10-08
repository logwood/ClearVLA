"""Permutation, occlusion, gradient and migration gates for region supervision."""
from dataclasses import replace
from pathlib import Path
import unittest
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.runtime.checkpoints import _validate_maniskill_spatial_repair
from clearvla.mainline.training.spatial_supervision import (
    StackCubeRegionLabels,matched_region_terms,stackcube_region_labels,
)
from clearvla.vision.entity_chart import ImageLogMeasure,current_image_grid


class SpatialSupervisionTests(unittest.TestCase):
    def fixture(self):
        masks=torch.zeros(1,2,2,64,64,dtype=torch.bool)
        masks[0,0,0,10:15,10:15]=True;masks[0,0,1,49:54,49:54]=True
        masks[0,1,0,49:54,49:54]=True;masks[0,1,1,10:15,10:15]=True
        labels=StackCubeRegionLabels(masks,masks.sum((-2,-1))>=20)
        grid=current_image_grid(64,64,device=torch.device('cpu'))
        centers=torch.tensor([[[-.62,-.62],[.62,.62]],[[.62,.62],[-.62,-.62]],[[0.,0.],[0.,0.]],[[0.,0.],[0.,0.]]])
        log=-(grid[None,None]-centers[:,:,None,None]).square().sum(-1)[None]*30
        return log.requires_grad_(),labels

    def test_slot_permutation_invariance_and_real_gradient(self):
        log,labels=self.fixture();support=torch.ones_like(log,dtype=torch.bool)
        a=matched_region_terms(ImageLogMeasure(log,support),labels)
        b=matched_region_terms(ImageLogMeasure(log[:,[2,0,3,1]],support),labels)
        torch.testing.assert_close(a['spatial_grounding'],b['spatial_grounding'])
        a['spatial_grounding'].backward()
        self.assertTrue(torch.isfinite(log.grad).all())
        self.assertGreater(log.grad.abs().max().item(),0)

    def test_one_assignment_must_explain_both_cameras(self):
        log,labels=self.fixture();support=torch.ones_like(log,dtype=torch.bool)
        correct=matched_region_terms(ImageLogMeasure(log,support),labels)['spatial_grounding']
        swapped=log.detach().clone();swapped[:,:,1]=swapped[:,[1,0,2,3],1]
        wrong=matched_region_terms(ImageLogMeasure(swapped,support),labels)['spatial_grounding']
        self.assertGreater(wrong.item(),correct.item()+1)

    def test_empty_labels_zero_loss_and_zero_gradient(self):
        log,labels=self.fixture()
        empty=StackCubeRegionLabels(torch.zeros_like(labels.masks),torch.zeros_like(labels.visible))
        loss=matched_region_terms(ImageLogMeasure(log,torch.ones_like(log,dtype=torch.bool)),empty)['spatial_grounding']
        self.assertEqual(loss.item(),0);loss.backward()
        self.assertTrue(torch.equal(log.grad,torch.zeros_like(log)))

    def test_tiny_supported_mass_has_finite_nonzero_gradient(self):
        log,labels=self.fixture()
        labels=StackCubeRegionLabels(labels.masks,labels.visible.clone())
        labels.visible[:,:,1]=False
        low=log.detach().clone();low[:,:,0]-=1000;low.requires_grad_()
        loss=matched_region_terms(ImageLogMeasure(low,torch.ones_like(low,dtype=torch.bool)),labels)['spatial_grounding']
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(low.grad).all())
        self.assertGreater(low.grad.abs().max().item(),.001)

    def test_current_color_labels_exclude_wood_and_preserve_unknown(self):
        rgb=torch.tensor([[[[[150.,240.,20.]],[[95.,20.,240.]],[[70.,15.,15.]]]]])/255
        rgb.requires_grad_();labels=stackcube_region_labels(rgb)
        self.assertEqual(labels.masks[0,0,0,0].tolist(),[False,True,False])
        self.assertEqual(labels.masks[0,1,0,0].tolist(),[False,False,True])
        self.assertFalse(labels.visible.any());self.assertFalse(labels.masks.requires_grad)

    def test_migration_only_allows_the_predeclared_objective(self):
        p=Path(__file__).resolve().parents[1]/'configs/mainline/maniskill_stackcube_dinov3_latest_20261007.json'
        cfg=load_config(p)
        self.assertNotIn('maniskill_spatial_grounding',cfg.as_dict()['objectives'])
        candidate=replace(cfg,objectives=replace(cfg.objectives,maniskill_spatial_grounding=.01))
        candidate.validate();_validate_maniskill_spatial_repair(cfg,candidate)
        _validate_maniskill_spatial_repair(cfg,cfg)
        with self.assertRaises(ValueError):
            _validate_maniskill_spatial_repair(cfg,replace(candidate,top=replace(candidate.top,action_history_condition_dropout=.5)))
        with self.assertRaises(ValueError):
            _validate_maniskill_spatial_repair(cfg,replace(candidate,objectives=replace(candidate.objectives,maniskill_spatial_grounding=.02)))


if __name__=='__main__':unittest.main()
