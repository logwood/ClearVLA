"""Gradient and label contracts, not behavioral qualification of a model."""
import math
import torch
import unittest
from clearvla.mainline.training.identity_regions import region_separation,region_balanced_prediction


def edge(n):return (~torch.eye(n,dtype=torch.bool))[None]


def test_no_edges_or_support_do_not_force_slot_occupancy():
    logits=torch.randn(1,6,4,requires_grad=True);q=logits.softmax(-1)
    for ids,adj in [(torch.full((1,6),-1),edge(2)),(torch.tensor([[0,0,0,1,1,1]]),torch.zeros(1,2,2,dtype=torch.bool))]:
        result=region_separation(q,ids,adj)
        assert result['loss'].item()==0 and result['supported_pairs'].item()==0
        assert torch.count_nonzero(torch.autograd.grad(result['loss'],logits,retain_graph=True)[0])==0


def test_k_and_group_relabeling_are_invariant():
    z=torch.tensor([[[2.,1.,0.,-.1],[1.9,1.1,0.,-.1],[2.1,.9,0.,-.1]]],requires_grad=True)
    q=z.softmax(-1);ids=torch.tensor([[0,1,2]]);adj=edge(3)
    result=region_separation(q,ids,adj)['loss']
    swapped=region_separation(q[...,torch.tensor([2,0,3,1])],2-ids,adj)['loss']
    torch.testing.assert_close(result,swapped,atol=1e-7,rtol=1e-6)
    g=torch.autograd.grad(result,z,retain_graph=True)[0];h=torch.autograd.grad(swapped,z)[0]
    torch.testing.assert_close(g,h,atol=1e-7,rtol=1e-6)


def test_five_regions_need_not_have_five_exclusive_slots():
    q=torch.cat((torch.eye(4)*.96+.01,torch.full((1,4),.25)))[None]
    out=region_separation(q,torch.arange(5)[None],edge(5))
    assert out['supported_pairs'].item()==10 and out['loss'].item()==0


def test_identical_law_symmetry_is_explicit_not_called_solved():
    z=torch.tensor([[[2.,0.,-1.,-2.],[2.,0.,-1.,-2.]]],requires_grad=True)
    out=region_separation(z.softmax(-1),torch.tensor([[0,1]]),edge(2))
    torch.testing.assert_close(out['loss'],torch.tensor(1.))
    torch.testing.assert_close(torch.autograd.grad(out['loss'],z)[0],torch.zeros_like(z),atol=1e-7,rtol=0)


def test_actual_difference_gets_separating_ordinary_gradient():
    z=torch.tensor([[[2.,.1,0.,-.1],[1.9,.2,0.,-.1]]],requires_grad=True)
    ids=torch.tensor([[0,1]]);adj=edge(2)
    a=region_separation(z.softmax(-1),ids,adj)['loss'];g=torch.autograd.grad(a,z)[0]
    assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
    b=region_separation((z-.1*g).softmax(-1),ids,adj)['loss']
    assert b.item()<a.item()


def test_null_logit_cannot_pay_the_separation_loss():
    z=torch.tensor([[[2.,.1,0.,-.1,1.],[2.1,.1,0.,-.1,-1.],
                     [1.9,.2,0.,-.1,3.],[1.9,.15,0.,-.1,0.]]],requires_grad=True)
    joint=z.log_softmax(-1);real=joint[...,:4].softmax(-1)
    result=region_separation(real,torch.tensor([[0,0,1,1]]),edge(2))['loss']
    grad=torch.autograd.grad(result,z)[0]
    assert torch.count_nonzero(grad[...,:4])>0
    torch.testing.assert_close(grad[...,-1],torch.zeros_like(grad[...,-1]),atol=2e-7,rtol=0)


def test_equal_regions_preserve_all_raw_prediction_errors():
    err=torch.tensor([[1.,3.,5.,10.,17.]],requires_grad=True)
    ids=torch.tensor([[0,0,0,1,-1]]);valid=torch.ones_like(ids,dtype=torch.bool)
    extra=region_balanced_prediction(err,ids,valid)['loss']
    torch.testing.assert_close(extra,torch.tensor(6.5))
    full=err.mean();total=full+.1*extra
    grad=torch.autograd.grad(total,err)[0]
    torch.testing.assert_close(grad,torch.tensor([[.2+.1/6,.2+.1/6,.2+.1/6,.25,.2]]))


def test_absent_strata_and_invalid_edges_fail_safely():
    err=torch.ones(2,4,requires_grad=True)
    result=region_balanced_prediction(err,torch.full((2,4),-1),torch.zeros(2,4,dtype=torch.bool))
    assert result['loss'].item()==0
    q=torch.ones(1,2,4)/4
    with unittest.TestCase().assertRaises(ValueError):region_separation(q,torch.tensor([[0,1]]),torch.ones(1,2,2,dtype=torch.bool))


if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(f) for name,f in sorted(globals().items()) if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
