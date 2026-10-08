"""CPU contracts for observational diagnostics and validation-failure persistence."""
import ast
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.canonical_grounding import encode_owners
from clearvla.mainline.model.address_memory import address_parameter_metrics
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.checkpoint import ArtifactIdentity, DatasetIdentity, build_checkpoint_identity
from clearvla.mainline.runtime.checkpoints import save_checkpoint, load_checkpoint_exact, load_checkpoint_for_validation
from clearvla.mainline.training.optimizer import WarmupCosineSchedule

ROOT = Path(__file__).resolve().parents[1]

class AddressEvidence(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(521)
        self.model = DenseObjectGrounder(hidden=16, content_dim=12, route_dim=8, objects=4,
            iterations=3, entity_ownership_mode="canonical_image_v1",
            entity_competition_scale_mode="per_observation_v1", camera_names=("top","wrist"),
            entity_address_memory_mode="conditional_logits_v1")
        with torch.no_grad():
            self.model.address_memory_gain.copy_(torch.tensor([.2,-.1,.3]))
        self.x = torch.randn(2,18,16)
        self.legal = torch.rand(2,18)>.25
        self.mass = torch.where(self.legal, torch.rand(2,18), 0.)

    def test_diagnostics_exact_forward_backward_rng_and_parameter_parity(self):
        other = copy.deepcopy(self.model)
        x1=self.x.clone().requires_grad_();x2=self.x.clone().requires_grad_()
        before={n:p.clone() for n,p in self.model.state_dict().items()}
        rng=torch.random.get_rng_state().clone()
        a=encode_owners(self.model,x1,self.mass,self.legal)
        diagnostics={}
        b=encode_owners(other,x2,self.mass,self.legal,diagnostics=diagnostics)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for v,w in zip(a,b):self.assertTrue(torch.equal(v,w))
        loss=lambda y:y[0].square().sum()+y[1].exp().square().sum()
        loss(a).backward();loss(b).backward()
        self.assertTrue(torch.equal(x1.grad,x2.grad))
        for name,value in self.model.named_parameters():
            compare=dict(other.named_parameters())[name]
            self.assertEqual(value.grad is None,compare.grad is None,name)
            if value.grad is not None:self.assertTrue(torch.equal(value.grad,compare.grad),name)
        self.assertTrue(all(torch.equal(before[n],p) for n,p in self.model.state_dict().items()))
        self.assertEqual(len(diagnostics),15)
        for value in diagnostics.values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))
        for i in range(1,4):
            prefix=f"object_grounding_address_step{i}_"
            self.assertGreater(float(diagnostics[prefix+"conditional_tv"]),0)
            self.assertGreater(float(diagnostics[prefix+"read_tv"]),0)
            self.assertEqual(float(diagnostics[prefix+"null_change_max"]),0)
            self.assertLess(float(diagnostics[prefix+"real_mass_change_max"]),5e-7)

    def test_zero_gain_reports_zero_immediate_change_and_empty_support_finite(self):
        with torch.no_grad():self.model.address_memory_gain.zero_()
        for legal in (self.legal,torch.zeros_like(self.legal)):
            mass=torch.where(legal,self.mass,0.)
            d={};encode_owners(self.model,self.x,mass,legal,diagnostics=d)
            self.assertTrue(all(torch.isfinite(v) for v in d.values()))
            for name,value in d.items():
                if not name.endswith("producer_mass"):self.assertEqual(float(value),0,name)

    def test_parameter_snapshot_no_alias_disconnected_distinct_from_zero(self):
        p=self.model.address_memory_gain
        d=address_parameter_metrics(p)
        self.assertEqual(float(d["gradient_parameter_address_memory_connected"]),0)
        self.assertNotIn("gradient_parameter_address_gain1_raw",d)
        p.grad=torch.tensor([0.,-.2,.3])
        d=address_parameter_metrics(p)
        self.assertEqual(float(d["gradient_parameter_address_memory_connected"]),1)
        self.assertEqual(float(d["gradient_parameter_address_gain1_raw"]),0)
        gain=d["object_grounding_address_gain1"].clone()
        grad=d["gradient_parameter_address_gain2_raw"].clone()
        with torch.no_grad():p.add_(1);p.grad.zero_()
        self.assertTrue(torch.equal(gain,d["object_grounding_address_gain1"]))
        self.assertTrue(torch.equal(grad,d["gradient_parameter_address_gain2_raw"]))

    def test_all_diagnostics_survive_archival_filter_including_zeros(self):
        from clearvla.mainline.runtime.logging import archival_metrics, tensor_scalars
        with torch.no_grad():self.model.address_memory_gain.zero_()
        self.model.address_memory_gain.grad=torch.zeros(3)
        d={};encode_owners(self.model,self.x,self.mass,self.legal,diagnostics=d)
        d.update(address_parameter_metrics(self.model.address_memory_gain))
        scalars=archival_metrics(tensor_scalars(d))
        self.assertEqual(set(scalars),set(d))
        self.assertEqual(scalars["gradient_parameter_address_gain1_raw"],0)
        self.assertEqual(scalars["object_grounding_address_step1_conditional_tv"],0)

class CheckpointEvidence(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config=ExperimentConfig()
        zero=hashlib.sha256(b"").hexdigest()
        dataset=DatasetIdentity(raw_root="/data/liang.zhang/dataset/grab_pen_single/grab_pen_single",
            hdf5_glob="*.hdf5",inventory_sha256=zero,state_normalizer_sha256=zero,
            action_normalizer_sha256=zero,decoded_cache_identity=zero,dino_cache_identity=zero)
        condition=self.root/"goal.pt";condition.write_bytes(b"fixture-language")
        self.identity=build_checkpoint_identity(self.config,repo_root=ROOT,dataset=dataset,
            language=ArtifactIdentity.from_file("t5_goal",condition),commit="1"*40)
        self.model=torch.nn.Linear(3,2)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        self.schedule=WarmupCosineSchedule(self.optimizer,warmup_steps=2,total_steps=4,minimum_ratio=.1)
        self.model(torch.ones(2,3)).sum().backward()
        self.optimizer.step();self.schedule.step()
        self.generator=torch.Generator().manual_seed(319)
        self.kw=dict(model=self.model,optimizer=self.optimizer,schedule=self.schedule,
            config=self.config,identity=self.identity,epoch=1,global_step=1,best_metric=None,
            data_state={"fixture":"unchanged"},generators={"train_flow":self.generator})

    def test_actual_train_block_saves_before_validation_failure(self):
        # Execute the actual epoch-boundary statements, injecting only the
        # offline failure and CPU logger/runtime dependencies.
        tree=ast.parse((ROOT/"clearvla/mainline/train.py").read_text())
        blocks=[]
        for node in ast.walk(tree):
            body=getattr(node,"body",None)
            if not isinstance(body,list):continue
            start=next((i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="train_task_mix" for t in n.targets)),None)
            if start is None:continue
            end=next(i for i,n in enumerate(body[start+1:],start+1) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="validation_report" for t in n.targets))
            blocks.append(body[start+1:end+1])
        self.assertEqual(len(blocks),1)
        code=compile(ast.Module(body=blocks[0],type_ignores=[]),"<actual-train-boundary>","exec")
        rows=[]
        def fail(**kwargs):raise RuntimeError("injected offline validation failure")
        env=dict(self.kw,output_dir=self.root,engine=SimpleNamespace(global_step=1),epoch=1,
            train_loader_generator=self.generator,train_flow_generator=self.generator,
            train_condition_generator=self.generator,train_values={"loss_total":.25},train_task_mix={},
            logger=SimpleNamespace(write=lambda kind,**kw:rows.append({"kind":kind,**kw})),
            _cuda_memory_metrics=lambda device:{},device="cpu",dtype=torch.float32,
            _validate=fail,val_loader=object(),bundle=object(),save_checkpoint=save_checkpoint)
        rng=torch.random.get_rng_state().clone();gen=self.generator.get_state().clone()
        with self.assertRaisesRegex(RuntimeError,"injected offline"):
            exec(code,env)
        path=self.root/"checkpoints/training_complete.pt"
        self.assertTrue(path.exists())
        self.assertFalse((path.parent/"best.pt").exists())
        self.assertFalse((path.parent/"latest.pt").exists())
        payload=torch.load(path,map_location="cpu",weights_only=False)
        self.assertIs(payload["validation_pending"],True)
        self.assertEqual(payload["global_step"],1)
        self.assertTrue(payload["optimizer"]["state"])
        self.assertEqual(payload["schedule"],self.schedule.state_dict())
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(gen,self.generator.get_state()))
        self.assertEqual(rows[0]["kind"],"training_complete")
        self.assertTrue(rows[0]["validation_pending"])
        self.assertEqual(rows[0]["train"]["loss_total"],.25)
        target=torch.nn.Linear(3,2)
        restored=load_checkpoint_for_validation(path,model=target,config=self.config,identity=self.identity)
        self.assertEqual(restored.global_step,1)
        self.assertIsNone(restored.best_metric)
        for n,v in target.state_dict().items():self.assertTrue(torch.equal(v,payload["model"][n]))

    def test_pending_exact_resume_rejected_before_model_or_rng_mutation(self):
        path=self.root/"pending.pt"
        save_checkpoint(path,validation_pending=True,**self.kw)
        with torch.no_grad():self.model.weight.add_(3)
        before=copy.deepcopy(self.model.state_dict());rng=torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError,"validation is pending"):
            load_checkpoint_exact(path,model=self.model,optimizer=self.optimizer,schedule=self.schedule,
                config=self.config,identity=self.identity,generators=self.kw["generators"])
        for n,p in self.model.state_dict().items():self.assertTrue(torch.equal(before[n],p))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_ordinary_payload_resume_unchanged_and_invalid_flag_rejected(self):
        path=self.root/"ordinary.pt"
        save_checkpoint(path,**self.kw)
        payload=torch.load(path,map_location="cpu",weights_only=False)
        self.assertNotIn("validation_pending",payload)
        restored=load_checkpoint_exact(path,model=self.model,optimizer=self.optimizer,
            schedule=self.schedule,config=self.config,identity=self.identity,generators=self.kw["generators"])
        self.assertEqual(restored.global_step,1)
        with self.assertRaisesRegex(ValueError,"must be Boolean"):
            save_checkpoint(self.root/"bad.pt",validation_pending=1,**self.kw)
        self.assertFalse((self.root/"bad.pt").exists())

if __name__=="__main__":unittest.main(verbosity=2)
