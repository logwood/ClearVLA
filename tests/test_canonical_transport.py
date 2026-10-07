import unittest
from dataclasses import replace
import torch
import torch.nn.functional as F
from clearvla.vision.entity_chart import CurrentImageSupport, CanonicalImageReadSource
from clearvla.vision.canonical_transport import rasterize_source, rasterize_value


class CanonicalTransportTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(817)
        shape=(2,2,2,2,3,7)
        xy=(torch.rand(*shape,2)*1.7-.85).requires_grad_()
        logits=torch.randn(shape,requires_grad=True);logp=logits.log_softmax(-1)
        valid=torch.ones(shape,dtype=torch.bool)
        support=CurrentImageSupport(xy,logp.exp(),valid,logp)
        prior=torch.rand(shape[:-1],requires_grad=True)
        validity=torch.ones(*shape[:-1],1)
        return support,prior,validity

    def test_mass_and_adjoint_reference(self):
        s,prior,validity=self.fixture();source,mass=rasterize_source(s,prior,validity,rows=5,columns=6)
        torch.testing.assert_close(mass.sum(-1),prior.flatten(2).sum(-1))
        image=torch.randn(2,2,5,6)
        scalar=(mass.reshape_as(image)*image).sum()
        sampled=F.grid_sample(image.reshape(4,1,5,6),s.coordinates.reshape(4,1,-1,2),align_corners=True).reshape_as(s.probability)
        expected=(sampled*s.probability*prior[...,None]).sum()
        torch.testing.assert_close(scalar,expected,atol=1e-5,rtol=1e-5)
        a=torch.autograd.grad(scalar,(s.coordinates,prior),retain_graph=True)
        b=torch.autograd.grad(expected,(s.coordinates,prior),retain_graph=True)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)

    def test_duplicate_atom_and_source_value(self):
        s,prior,validity=self.fixture();a,m=rasterize_source(s,prior,validity,rows=5,columns=6)
        two=CurrentImageSupport(s.coordinates.repeat_interleave(2,-2),s.probability.repeat_interleave(2,-1)/2,s.valid.repeat_interleave(2,-1),s.log_probability.repeat_interleave(2,-1)-torch.log(torch.tensor(2.)))
        b,n=rasterize_source(two,prior,validity,rows=5,columns=6)
        torch.testing.assert_close(a,b);torch.testing.assert_close(m,n)
        value=torch.randn(*prior.shape,9,requires_grad=True)
        x=rasterize_value(a,m,value,validity);y=rasterize_value(b,n,value,validity)
        torch.testing.assert_close(x,y)
        assert torch.autograd.grad(x.square().sum(),value)[0].abs().sum()>0

    def test_canonical_mass_and_permutation(self):
        s,_,_=self.fixture();log=torch.randn(2,4,2,5,6,requires_grad=True)
        src=CanonicalImageReadSource(log,torch.ones_like(log,dtype=torch.bool),s)
        for size in (3,5,9):
            p,_=src.on_image(rows=size,columns=size).normalized((2,3,4))
            torch.testing.assert_close(p.flatten(2).sum(-1),torch.ones(2,4))
            swapped,_=src.permute(torch.tensor([2,0,3,1])).on_image(rows=size,columns=size).normalized((2,3,4))
            torch.testing.assert_close(p[:,[2,0,3,1]],swapped)
            assert torch.isfinite(torch.autograd.grad(p[...,0,0].sum(),log,retain_graph=True)[0]).all()


if __name__=='__main__':unittest.main()
