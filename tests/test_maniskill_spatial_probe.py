"""Checks for audit coordinate/mask correctness, not policy-quality claims."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from probe_maniskill_spatial_grounding import raster_mass,region_statistics
from probe_maniskill_rgb_calibration import color_masks


class SpatialProbeTests(unittest.TestCase):
    def test_align_corners_and_mass_conservation(self):
        coordinates=torch.tensor([[[-1.,-1.],[1.,1.],[0.,0.]]])
        density=raster_mass(coordinates,torch.tensor([[.2,.3,.5]]),side=3)
        expected=torch.tensor([[[.2,0.,0.],[0.,.5,0.],[0.,0.,.3]]])
        torch.testing.assert_close(density,expected)

    def test_bilinear_camera_separation(self):
        xy=torch.tensor([[[-.5,-.5]],[[.5,.5]]])
        density=raster_mass(xy,torch.tensor([[1.],[2.]]),side=3)
        self.assertAlmostEqual(float(density[0,0,0]),.25)
        self.assertAlmostEqual(float(density[1,2,2]),.5)
        torch.testing.assert_close(density.sum((1,2)),torch.tensor([1.,2.]))

    def test_conditional_camera_mass_does_not_change_joint_mass(self):
        density=np.zeros((1,2,336,336))
        density[0,0,100,100]=.1;density[0,1,100,100]=.9
        masks=np.zeros((2,2,336,336),bool);masks[:,0,100,100]=True
        out=region_statistics(density,masks)
        np.testing.assert_allclose(out["camera_mass"],[[.1,.9]])
        np.testing.assert_allclose(out["regions"]["red_r0"]["conditional_mass"],[[1.,1.]])
        np.testing.assert_array_equal(out["regions"]["green_r0"]["conditional_mass"],[[0.,0.]])

    def test_wood_is_not_red_cube(self):
        image=np.array([[[150,95,70],[240,20,15],[20,240,15],[120,120,120]]],np.uint8)
        masks=color_masks(image)
        np.testing.assert_array_equal(masks[0],[[False,True,False,False]])
        np.testing.assert_array_equal(masks[1],[[False,False,True,False]])


if __name__=="__main__":unittest.main()

