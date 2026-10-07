import unittest
import torch
from clearvla.vision.observed_correspondence import observed_feature_correspondence


class ObservedCorrespondenceTests(unittest.TestCase):
    def test_identity_signed_transport_and_unknown(self):
        torch.manual_seed(12)
        source=torch.randn(2,25,64)
        mask=torch.ones(2,25,dtype=torch.bool)
        law=observed_feature_correspondence(source,source.clone(),mask,mask)
        torch.testing.assert_close(law[...,:-1],torch.eye(25).expand(2,-1,-1),atol=0,rtol=0)
        assert law[...,-1].count_nonzero()==0
        permutation=torch.roll(torch.arange(25),3)
        target=source[:,permutation]
        law=observed_feature_correspondence(source,target,mask,mask)
        torch.testing.assert_close(law[...,:-1]@target,source,rtol=0,atol=0)
        grid=torch.stack(torch.meshgrid(torch.arange(5),torch.arange(5),indexing='ij'),-1).reshape(25,2).float()
        matched=law[...,:-1]@grid
        expected=grid[torch.argsort(permutation)]
        torch.testing.assert_close(matched-grid,expected[None].expand(2,-1,-1)-grid,rtol=0,atol=0)
        target_mask=torch.zeros_like(mask)
        law=observed_feature_correspondence(source,torch.full_like(target,float('nan')),mask,target_mask)
        assert law[...,:-1].count_nonzero()==0 and torch.equal(law[...,-1],torch.ones_like(mask,dtype=torch.float32))

    def test_camera_permutation_and_source_gradient(self):
        a=torch.randn(2,2,9,32,requires_grad=True);b=torch.randn_like(a)
        mask=torch.ones(a.shape[:-1],dtype=torch.bool)
        p=observed_feature_correspondence(a,b,mask,mask)
        swapped=observed_feature_correspondence(a.flip(1),b.flip(1),mask,mask)
        torch.testing.assert_close(p.flip(1),swapped)
        assert torch.isfinite(torch.autograd.grad(p[...,0].sum(),a)[0]).all()

if __name__=='__main__': unittest.main()
