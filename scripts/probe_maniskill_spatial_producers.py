"""Locate spatial loss before/inside the G binder and audit Teacher reference.

All interventions are temporary frozen-checkpoint diagnostics. Simulator masks
never enter online features. Uniform G2 / omitted parent prior are not repairs
and must not be reported as deployed performance.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import clearvla.mainline.v120_core.flow_dino_evidence as flow_module
from probe_maniskill_failure import array, load_policy
from probe_maniskill_spatial_causality import physical_history
from probe_maniskill_spatial_grounding import SpatialCapture, raster_mass, region_statistics, simulator_labels
from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv
from record_maniskill_npz import atomic_json


class ProducerCapture(SpatialCapture):
    def capture_producers(self,history,intervention):
        encoder=self.policy.bundle.model.observation.compiler.encoder
        advance=encoder.update_progressive_grounding_address
        couple=flow_module.couple_local_observation
        self.producer_tensors={}
        def observed_couple(typed,prior,valid):
            self.producer_tensors["g2_parent_prior"]=array(prior)
            for key,value in typed.items():self.producer_tensors["g2_"+key+"_logits"]=array(value)
            return couple(typed,torch.zeros_like(prior) if intervention=="no_parent_prior" else prior,valid)
        def observed_advance(*args,**kwargs):
            if intervention=="g1_query_zero":kwargs["intervention"]="address_g1_zero"
            if intervention=="g2_uniform":kwargs["intervention"]="address_g2_zero"
            state=advance(*args,**kwargs)
            if state.stage==1:
                xy=state.bank.coarse_candidate_coordinates[0]
                for key,p in (("source",state.bank.coarse_base_logits.softmax(-1)),("g1",state.coarse_probability)):
                    weights=p[0].mean((1,2,3))
                    self.producer_tensors[key+"_density"]=array(raster_mass(xy,weights))
                self.producer_tensors["g1_logits"]=array(state.coarse_logits)
                self.producer_tensors["source_logits"]=array(state.bank.coarse_base_logits)
            if state.stage==2:
                xy=state.dynamic_fine_coordinates[0]
                valid=state.dynamic_fine_valid.float()
                uniform=valid/valid.sum(-1,keepdim=True).clamp_min(1)
                for key,p in (("g2",state.fine_probability),("available",uniform)):
                    # Average real local hypotheses, preserving each camera.
                    weights=p[0]/np.prod(p.shape[2:5])
                    self.producer_tensors[key+"_density"]=array(raster_mass(xy,weights))
                self.producer_tensors["g2_probability"]=array(state.fine_probability)
            return state
        with patch.object(encoder,"update_progressive_grounding_address",observed_advance), patch.object(flow_module,"couple_local_observation",observed_couple):
            result=self.infer_spatial(history)
        return result

    def teacher_identity(self):
        f=self.state.top.facts
        teacher=self.policy.bundle.model.training_targets.teacher
        current=f.dense_chart.dino_content
        observations=current[:,None]
        observed=torch.ones(1,1,dtype=torch.bool,device=current.device)
        offsets=torch.zeros(1,dtype=torch.long,device=current.device)
        result={}
        original_mode=teacher.current_reference_mode
        try:
            for mode in ("g_assignment_v1","raw_chart_v1"):
                teacher.current_reference_mode=mode
                m=teacher.measure_observations(facts=f,observations=observations,relative_offsets=offsets,observed=observed)
                perturbed=teacher.measure_observations(
                    facts=replace(f,content=f.content*1.01), observations=observations,
                    relative_offsets=offsets,observed=observed)
                for name in ("candidate_posterior","null_probability","current_reference",
                             "successor_per_support","transport_per_support"):
                    result[mode+"_"+name]=array(getattr(m,name))
                result[mode+"_content_perturb_target_delta"]=array(
                    (perturbed.successor_per_support-perturbed.current_reference)-
                    (m.successor_per_support-m.current_reference))
        finally:teacher.current_reference_mode=original_mode
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","checkpoint","dino","output"):p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);rows=[]
    policy=load_policy(args);capture=ProducerCapture(policy);env=ManiSkillStackCubeEnv()
    try:
        with torch.no_grad():
            for moved in ("red","green"):
                for x,y in ((-.10,-.18),(.10,-.18),(-.10,.18),(.10,.18),(0.,-.08),(0.,.08)):
                    history=physical_history(env,moved,(x,y));masks,poses=simulator_labels(env)
                    modes=("none","g1_query_zero","no_parent_prior","g2_uniform") if moved=="red" else ("none",)
                    for mode in modes:
                        label=f"{moved}_x{x:+.2f}_y{y:+.2f}_{mode}"
                        policy.reset()
                        action,online,tensors=capture.capture_producers(history,mode)
                        tensors.update(capture.producer_tensors)
                        tensors.update(capture.teacher_identity())
                        tensors.update(masks=masks,positions=poses)
                        f=capture.state.top.facts
                        tensors["g_allocation"]=array(f.candidate_assignment.sum((2,3,4,5)))
                        np.savez_compressed(args.output/(label+".npz"),**tensors)
                        row={"label":label,"mode":mode,"moved":moved,"xy":[x,y],
                             "first8_mean_xyz":action[:8,:3].mean(0).tolist()}
                        for key in ("source_density","g1_density","g2_density","available_density","g_density","p1_micro_density"):
                            row[key]=region_statistics(tensors[key],masks)
                        row["teacher"]={}
                        for teacher_mode in ("g_assignment_v1","raw_chart_v1"):
                            delta=tensors[teacher_mode+"_successor_per_support"]-tensors[teacher_mode+"_current_reference"]
                            row["teacher"][teacher_mode]={
                                "same_observation_delta_rms":float(np.sqrt(np.mean(delta**2))),
                                "content_perturb_delta_rms":float(np.sqrt(np.mean(tensors[teacher_mode+"_content_perturb_target_delta"]**2))),
                                "mean_null":float(np.mean(tensors[teacher_mode+"_null_probability"])),
                                "same_observation_transport_rms":float(np.sqrt(np.mean(tensors[teacher_mode+"_transport_per_support"]**2))),
                            }
                        rows.append(row)
                        print(json.dumps({k:row[k] for k in ("label","first8_mean_xyz","teacher")}),flush=True)
                        atomic_json(args.output/"summary.json",{"note":__doc__,"rows":rows})
    finally:env.close()
    atomic_json(args.output/"provenance.json",policy.deployment_health())

if __name__=="__main__":main()

