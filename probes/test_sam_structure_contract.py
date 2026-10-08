"""Integrated G invariants and narrow initialization regression."""
import copy
from dataclasses import replace
import json
import unittest
from pathlib import Path
import torch
from clearvla.mainline.config import load_config, ExperimentConfig
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.canonical_grounding import encode_owners
from clearvla.mainline.model.image_feedback import SlotToImageFeedback, RegionImageFusion
from clearvla.mainline.runtime.sam_structure_migration import migrate_state, validate_selection, config_view, SOURCE_DIGEST
from clearvla.mainline.causal_identity import causal_identity_metadata
from probes.sam_structure_adapters import mechanical_contracts

class TestStructure(unittest.TestCase):
    def test_integrated_legacy_and_zero_init(self):
        torch.set_num_threads(4)
        kwargs=dict(hidden=48,content_dim=64,route_dim=16,objects=4,iterations=3,
            entity_ownership_mode="canonical_image_v1",entity_chart_mode="current_image_support_v1",
            entity_context_mode="completed_g3_v1",entity_competition_scale_mode="per_observation_v1",
            camera_names=("rgb_static","rgb_gripper"))
        torch.manual_seed(9);base=DenseObjectGrounder(**kwargs);rng=torch.get_rng_state()
        x=torch.randn(2,32,48);mass=torch.rand(2,32);legal=mass>.3
        mass=torch.where(legal,mass,0);x=torch.where(legal[...,None],x,0.)
        with torch.no_grad():reference=encode_owners(base,x,mass,legal)
        for mode in ("slot_to_image_v1","region_fusion_v1"):
            torch.manual_seed(9);model=DenseObjectGrounder(**kwargs,entity_image_feedback_mode=mode)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in base.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
            with torch.no_grad():out=encode_owners(model,x,mass,legal,image_shape=(2,4,4))
            for a,b in zip(out,reference):self.assertTrue(torch.equal(a,b),mode)
            with self.assertRaises(ValueError):encode_owners(model,x,mass,legal)
            score=encode_owners(model,x,mass,legal,image_shape=(2,4,4))[0].square().mean()
            grad=torch.autograd.grad(score,model.image_feedback.out.weight)[0]
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.norm(),0)
    def test_adapter_contracts(self):
        # Use exactly the production classes with the same invariant harness.
        import probes.sam_structure_adapters as harness
        original=harness.RegionImageFusion
        harness.RegionImageFusion=RegionImageFusion
        try:
            for cls in (SlotToImageFeedback,RegionImageFusion):
                r=mechanical_contracts(cls);self.assertTrue(r["invalid_input_grad_zero"])
        finally:harness.RegionImageFusion=original
    def test_migration_and_legacy_serialization(self):
        path=Path("/data/senwang/clearvla/experiments/causal-identity-ab-20261007/B-nullv2-short-bs8-1024-r1.config.json")
        saved=load_config(path);current=copy.deepcopy(saved)
        self.assertNotIn("entity_image_feedback_mode",saved.as_dict()["top"])
        old_abi=causal_identity_metadata(saved.top);self.assertNotIn("image_feedback",old_abi)
        current=replace(saved,top=replace(saved.top,entity_image_feedback_mode="region_fusion_v1"));current.validate()
        validate_selection(saved,current,SOURCE_DIGEST)
        self.assertEqual(config_view(saved.as_dict()),config_view(current.as_dict()))
        with self.assertRaises(ValueError):validate_selection(saved,current,"wrong-source")
        prefix="grounding.grounder.image_feedback."
        inherited={"parent":torch.randn(3)}
        state={**inherited,**{prefix+n:torch.zeros(2,2) for n in ("down.weight","out.weight","mask.weight","depthwise.weight")}}
        mapped=migrate_state(inherited,state,current)
        self.assertIs(mapped["parent"],inherited["parent"])
        with self.assertRaises(ValueError):migrate_state(inherited,{**state,"extra":torch.zeros(1)},current)
        self.assertEqual(causal_identity_metadata(current.top)["image_feedback"]["rank"],32)

if __name__=="__main__":unittest.main()
