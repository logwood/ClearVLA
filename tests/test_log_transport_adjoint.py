import unittest
import torch
from clearvla.vision.entity_chart import (
    CurrentImageSupport, pushforward_to_current_image, pushforward_log_to_current_image,
)


def check_normalized_forward_and_all_source_adjoints(axes,side,knots):
    torch.manual_seed(33)
    xy = (torch.tensor([[0.,0.],[-.5,.5],[1.,1.],[-1.,-.5]]) if knots
          else torch.tensor([[.13,-.32],[-.45,.62],[.89,-.77],[-.86,-.44]]))
    xy = xy.reshape(1,1,1,1,1,4,2).repeat(1,2,1,1,1,1,1).requires_grad_()
    logits = torch.randn(1,2,1,1,1,4,requires_grad=True)
    lm = torch.randn(1,3,2,1,1,1,requires_grad=True)
    p = logits.softmax(-1)
    sp = CurrentImageSupport(xy,p,torch.ones_like(p,dtype=torch.bool),logits.log_softmax(-1))
    linear = pushforward_to_current_image(lm.exp(),sp,rows=side,columns=side)
    den=linear.sum(axes,keepdim=True)
    ref=torch.where(den>0,linear/torch.where(den>0,den,1.),0.)
    image=pushforward_log_to_current_image(lm,torch.ones_like(lm,dtype=torch.bool),sp,
        rows=side,columns=side,ordinary_coordinate_gradients=True)
    actual, logp=image.normalized(axes)
    torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-6)
    w=torch.randn_like(ref)
    gr=torch.autograd.grad((ref*w).sum(),(lm,logits,xy),retain_graph=True)
    ga=torch.autograd.grad((actual*w).sum(),(lm,logits,xy),retain_graph=True)
    for a,r in zip(ga,gr): torch.testing.assert_close(a,r,atol=5e-6,rtol=3e-5)
    # The finite log payload at unsupported cells is not log(0).
    support=linear>0
    logref=torch.where(support,ref,1.).log()
    gr=torch.autograd.grad((logref*w).sum(),(lm,logits,xy),retain_graph=True)
    ga=torch.autograd.grad((logp*w).sum(),(lm,logits,xy))
    for a,r in zip(ga,gr): torch.testing.assert_close(a,r,atol=5e-5,rtol=3e-5)


def check_underflow_empty_quarantine_and_independent_mask():
    xy=torch.tensor([0.,0.,.5,.5]).reshape(1,1,1,1,1,2,2).requires_grad_()
    lp=torch.tensor([-.6931472,-.6931472]).reshape(1,1,1,1,1,2).requires_grad_()
    sp=CurrentImageSupport(xy,lp.exp(),torch.ones_like(lp,dtype=torch.bool),lp)
    lm=torch.tensor([-1000.,-1001.]).reshape(1,2,1,1,1,1).requires_grad_()
    image=pushforward_log_to_current_image(lm,torch.ones_like(lm,dtype=torch.bool),sp,
        rows=5,columns=5,ordinary_coordinate_gradients=True)
    p,_=image.normalized((1,))
    torch.testing.assert_close(p[:,:,0,2,2],torch.softmax(lm.flatten(),0)[None],rtol=1e-4,atol=1e-5)
    mask=torch.ones_like(p,dtype=torch.bool);mask[:,:,:,3:]=False
    p,_=image.restrict(mask).normalized((-2,-1))
    grad=torch.autograd.grad((p*torch.arange(25).reshape(1,1,1,5,5)).sum(),(lm,lp,xy))
    assert all(torch.isfinite(g).all() for g in grad)
    invalid=CurrentImageSupport(torch.full_like(xy,float('nan')),torch.zeros_like(lp),
        torch.zeros_like(lp,dtype=torch.bool),torch.zeros_like(lp))
    empty=pushforward_log_to_current_image(lm,torch.zeros_like(lm,dtype=torch.bool),invalid,
        rows=5,columns=5,ordinary_coordinate_gradients=True)
    p,logp=empty.normalized((-2,-1))
    assert torch.equal(p,torch.zeros_like(p)) and torch.equal(logp,torch.zeros_like(logp))
    assert torch.equal(torch.autograd.grad(p.sum()+logp.sum(),lm)[0],torch.zeros_like(lm))


class LogTransportTests(unittest.TestCase):
    def test_reference(self):
        for axes in [(-2,-1), (2,3,4), (1,)]:
            for side in [1,5]:
                for knots in [True,False]:
                    with self.subTest(axes=axes,side=side,knots=knots):
                        check_normalized_forward_and_all_source_adjoints(axes,side,knots)

    def test_extreme_and_masks(self):
        check_underflow_empty_quarantine_and_independent_mask()

if __name__ == '__main__':
    unittest.main()
