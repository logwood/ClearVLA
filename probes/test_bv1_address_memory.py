"""Numerical/gradient contracts for the explicit B-v1 address-only trial."""
import copy,json,unittest
from pathlib import Path
import torch
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.canonical_grounding import encode_owners
from clearvla.mainline.model.address_memory import retain_address
from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.causal_identity import causal_identity_metadata

E=Path("/data/senwang/clearvla/experiments/causal-identity-ab-20261007")

class AddressMemoryContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(918)
        kw=dict(hidden=16,content_dim=12,route_dim=8,objects=4,iterations=3,
            entity_ownership_mode="canonical_image_v1",entity_competition_scale_mode="per_observation_v1",
            camera_names=("top","wrist"))
        self.old=DenseObjectGrounder(**kw)
        self.new=copy.deepcopy(self.old)
        self.new.entity_address_memory_mode="conditional_logits_v1"
        self.new.address_memory_gain=torch.nn.Parameter(torch.zeros(3))
        self.x=torch.randn(2,18,16);self.mass=torch.rand(2,18)
        self.legal=torch.rand(2,18)>.25
        self.mass=torch.where(self.legal,self.mass,0.)
    def test_zero_forward_and_old_gradient(self):
        x1=self.x.clone().requires_grad_();x2=self.x.clone().requires_grad_()
        a=encode_owners(self.old,x1,self.mass,self.legal)
        b=encode_owners(self.new,x2,self.mass,self.legal)
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(a,b)))
        c=torch.randn_like(a[0]);d=torch.randn_like(a[1])
        loss=lambda y:(y[0]*c).sum()+(y[1].exp()*d).sum()
        loss(a).backward();loss(b).backward()
        torch.testing.assert_close(x1.grad,x2.grad,atol=2e-6,rtol=2e-6)
        for name,value in self.old.named_parameters():
            new=dict(self.new.named_parameters())[name]
            self.assertEqual(value.grad is None,new.grad is None,name)
            if value.grad is not None:torch.testing.assert_close(value.grad,new.grad,atol=2e-6,rtol=2e-6)
        g=self.new.address_memory_gain.grad
        self.assertTrue(torch.isfinite(g).all() and (g.abs()>0).all())
    def test_support_null_and_ordinary_previous_gradient(self):
        slots=self.old.slot_seed.expand(2,-1,-1)
        prior=self.mass[...,None];lp=torch.where(prior>0,prior,1.).log()
        c=self.old._competition(slots,self.x,self.legal[...,None].float(),prior,lp,self.legal)
        previous=torch.randn_like(c[4][...,:4],requires_grad=True)
        previous_with_invalid=torch.where(self.legal[...,None],previous,float("nan"))
        g=torch.tensor(.25,requires_grad=True)
        n=retain_address(c,previous_with_invalid,g,prior,lp,self.legal)
        self.assertTrue(torch.equal(c[2],n[2]))
        self.assertTrue(torch.equal(c[4][...,-1],n[4][...,-1]))
        torch.testing.assert_close(c[0][...,:4].sum(-1),n[0][...,:4].sum(-1),atol=2e-7,rtol=2e-6)
        self.assertTrue((n[0][...,:4][~self.legal]==0).all())
        ((n[3]*torch.randn_like(n[3])).sum()).backward()
        self.assertTrue(torch.isfinite(previous.grad).all())
        self.assertGreater(float(previous.grad[self.legal].norm()),0)
        self.assertEqual(float(previous.grad[~self.legal].norm()),0)
        self.assertTrue(torch.isfinite(g.grad))
    def test_empty_support(self):
        legal=torch.zeros_like(self.legal);mass=torch.zeros_like(self.mass)
        self.new.address_memory_gain.data.fill_(.3)
        x=self.x.clone().requires_grad_()
        slots,log=encode_owners(self.new,x,mass,legal)
        self.assertTrue(torch.isfinite(slots).all())
        self.assertTrue(torch.isneginf(log[...,:4]).all())
        self.assertTrue((log[...,-1]==0).all())
        (slots.square().sum()+log.exp().sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(self.new.address_memory_gain.grad).all())
    def test_k_and_candidate_permutations(self):
        self.new.address_memory_gain.data.copy_(torch.tensor([.2,-.1,.3]))
        base=encode_owners(self.new,self.x,self.mass,self.legal)
        alt=copy.deepcopy(self.new);k=torch.tensor([2,0,3,1])
        alt.slot_seed.data.copy_(self.new.slot_seed[:,k])
        moved=encode_owners(alt,self.x,self.mass,self.legal)
        torch.testing.assert_close(moved[0],base[0][:,k],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(moved[1],base[1][...,torch.cat((k,torch.tensor([4])))],atol=4e-6,rtol=2e-6)
        n=torch.randperm(18);moved=encode_owners(self.new,self.x[:,n],self.mass[:,n],self.legal[:,n])
        torch.testing.assert_close(moved[0],base[0],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(moved[1],base[1][:,n],atol=4e-6,rtol=2e-6)
    def test_constructor_rng_and_inventory(self):
        kw=dict(hidden=16,content_dim=12,route_dim=8,entity_ownership_mode="canonical_image_v1",camera_names=("top","wrist"))
        torch.manual_seed(45);a=DenseObjectGrounder(**kw);ra=torch.random.get_rng_state()
        torch.manual_seed(45);b=DenseObjectGrounder(**kw,entity_address_memory_mode="conditional_logits_v1");rb=torch.random.get_rng_state()
        self.assertTrue(torch.equal(ra,rb))
        self.assertEqual(set(b.state_dict())-set(a.state_dict()),{"address_memory_gain"})
        self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
    def test_legacy_config_and_new_abi(self):
        payload=torch.load(E/"B-short-bs8-1024-r1/checkpoints/best.pt",map_location="meta",weights_only=False)
        cfg=config_from_mapping(payload["config"])
        self.assertEqual(json.dumps(cfg.as_dict(),sort_keys=True),json.dumps(payload["config"],sort_keys=True))
        self.assertEqual(cfg.digest(),payload["identity"]["config_digest"])
        old=causal_identity_metadata(cfg.top);self.assertNotIn("address_memory",old)
        raw=cfg.as_dict();raw["top"]["entity_address_memory_mode"]="conditional_logits_v1"
        new=config_from_mapping(raw);new.validate()
        self.assertIn("address_memory",causal_identity_metadata(new.top))
        self.assertNotEqual(cfg.digest(),new.digest())
    def test_migration_rejects_extra_keys_and_nonzero_gain(self):
        from clearvla.mainline.runtime.address_memory_migration import migrate_state,KEY
        from types import SimpleNamespace
        cfg=SimpleNamespace(top=SimpleNamespace(entity_address_memory_mode="conditional_logits_v1"))
        saved={"x":torch.ones(2)};current={**saved,KEY:torch.zeros(3)}
        self.assertEqual(set(migrate_state(saved,current,cfg)),set(current))
        with self.assertRaises(ValueError):migrate_state(saved,{**current,"bad":torch.zeros(1)},cfg)
        with self.assertRaises(ValueError):migrate_state(saved,{**current,KEY:torch.ones(3)},cfg)

if __name__=="__main__":unittest.main(verbosity=2)
