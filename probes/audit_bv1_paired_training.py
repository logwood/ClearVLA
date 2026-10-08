"""Read-only matched B-v1 training-log audit, including external failure logs.

Dense losses use window_batches weights. Structural/role-gradient diagnostics
are sampled on logging batches only, so use unweighted diagnostic-row summaries.
This report cannot substitute for final weights, offline validation, or behavior.
"""
import argparse
import copy
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

NAMES = ("B-v1-address-memory-short-bs8-1024-r1", "B-v1-continuation-control-short-bs8-1024-r1")
DENSE = (
    "loss_total", "loss_action_flow", "loss_action_flow_first8", "loss_action_flow_tail",
    "loss_gripper_command", "loss_gripper_command_transition",
    "loss_gripper_command_transition_rate", "loss_gripper_command_transition_target_rate",
    "loss_gripper_command_expected_flip_rate", "loss_identity_source_prediction",
    "loss_identity_correspondence", "loss_contrib_identity_correspondence",
    "loss_identity_cross_admitted_pairs", "loss_identity_temporal_admitted_pairs",
    "loss_action_gripper_flow_owned", "runtime_window_seconds_per_batch",
)
SPARSE = (
    "object_grounding_null_mass", "object_grounding_object_content_pair_cosine",
    "object_intent_public_interval_variation", "object_intent_semantic_interval_variation",
    "object_intent_geometry_interval_variation", "object_p2_target_value_interval_variation",
    "object_p2_semantic_effect_rms", "object_p2_geometry_effect_rms",
    "object_p3_temporal_rms", "p1_protected_detail_rms",
    "gradient_raw_grounder_l2", "gradient_raw_intent_l2", "gradient_raw_dynamics_l2",
    "gradient_raw_p1_factual_l2", "gradient_raw_p2_effect_reader_l2", "gradient_raw_p3_compiler_l2",
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(rows, key, dense):
    pairs=[(row["metrics"][key],row["window_batches"] if dense else 1)
        for row in rows if key in row["metrics"]]
    if not pairs:return {"rows":0}
    return dict(rows=len(pairs),weight=sum(w for _,w in pairs),
        mean=sum(v*w for v,w in pairs)/sum(w for _,w in pairs),
        min=min(v for v,_ in pairs),max=max(v for v,_ in pairs),
        median=statistics.median(v for v,_ in pairs))

def audit(root, name, console):
    run=root/name
    ctx=json.loads((run/"run_context.json").read_text())
    records=[json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    rows=[row for row in records if row.get("kind")=="train"]
    assert len(rows)==52
    assert sum(row["window_batches"] for row in rows)==1024
    assert sum(row["window_samples"] for row in rows)==8192
    assert [(row["batch"],row["step"]) for row in rows]==[
        (batch,12036+batch) for batch in list(range(20,1021,20))+[1024]]
    bad=[(row["step"],key) for row in rows for key,value in row["metrics"].items()
        if isinstance(value,(int,float)) and not math.isfinite(value)]
    phase={"first200":[x for x in rows if x["batch"]<=200],
        "last224":[x for x in rows if x["batch"]>=820],"all":rows}
    stats={key:{"scope":"all-update window mean" if key in DENSE else "sampled diagnostic batch",
        **{stage:summarize(rs,key,key in DENSE) for stage,rs in phase.items()}}
        for key in DENSE+SPARSE}
    ledger=max(abs(v) for row in rows for k,v in row["metrics"].items()
        if k in {"loss_ledger_gap","loss_contribution_gap"})
    peak=max(rows,key=lambda row:row["metrics"]["gradient_window_preclip_l2_max"])
    text=console.read_text()
    checkpoints=list((run/"checkpoints").glob("*.pt"))
    output=dict(name=name,source_commit=ctx["identity"]["git_commit"],
        source_digest=ctx["identity"]["source"]["digest"],config_digest=ctx["identity"]["config_digest"],
        updates=1024,samples=8192,global_step=rows[-1]["step"],training_rows=len(rows),
        offline_epoch_rows=sum(x["kind"]=="epoch" for x in records),
        stored_scalar_nonfinite=bad,loss_ledger_max_abs=ledger,
        preclip_peak=peak["metrics"]["gradient_window_preclip_l2_max"],
        preclip_peak_step=peak["metrics"]["gradient_window_preclip_l2_max_global_step"],
        address_metric_names=sorted({k for row in rows for k in row["metrics"] if "address_gain" in k or "address_memory" in k}),
        checkpoint_files=[str(x) for x in checkpoints],
        failure_lines=[line.strip() for line in text.splitlines() if
            "AcceleratorError" in line or "CUDA error:" in line],
        evidence_sha256={str(run/"run_context.json"):digest(run/"run_context.json"),
            str(run/"metrics.jsonl"):digest(run/"metrics.jsonl"),str(console):digest(console)},
        statistics=stats)
    return output,ctx,rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.experiment_root
    consoles=[root/"B-v1-address-memory-short-r1-job/short1024.log",
        root/"B-v1-continuation-control-short-r1-job/short1024.log"]
    data=[audit(root,n,c) for n,c in zip(NAMES,consoles)]
    candidate,control=[x[0] for x in data]; ca,cb=[x[1] for x in data]
    a,b=copy.deepcopy(ca["config"]),copy.deepcopy(cb["config"])
    for cfg in (a,b):
        cfg["data"].pop("output_dir")
        cfg["top"].pop("entity_address_memory_mode",None)
    assert a==b,"undeclared config difference"
    for key in ("dataset","language","source"):
        assert ca["identity"][key]==cb["identity"][key],key
    for key in ("path","global_step","model_contract_migration","optimizer_initialization","training_clock"):
        assert ca["initialization_checkpoint"][key]==cb["initialization_checkpoint"][key],key
    differences={}
    for key in DENSE+SPARSE:
        aligned=[(ar["metrics"][key],br["metrics"][key],ar["window_batches"] if key in DENSE else 1)
            for ar,br in zip(data[0][2],data[1][2]) if key in ar["metrics"] and key in br["metrics"]]
        weight=sum(w for _,_,w in aligned)
        meanabs=sum(abs(x-y)*w for x,y,w in aligned)/weight
        controlmean=sum(y*w for _,y,w in aligned)/weight
        differences[key]=dict(paired_rows=len(aligned),mean_abs=meanabs,
            mean_abs_over_control_mean=meanabs/max(abs(controlmean),1e-30),
            maximum_abs=max(abs(x-y) for x,y,_ in aligned))
    report=dict(schema="bv1-paired-training-log-audit-v1",utc=datetime.now(timezone.utc).isoformat(),
        scope="Training-only; both offline jobs failed before checkpoint saving; no behavioral inference.",
        declared_exposure_matched=True,exact_sample_and_noise_identity_audited=False,
        allowed_config_differences=["data.output_dir","top.entity_address_memory_mode"],
        initializer=ca["initialization_checkpoint"],runs=[candidate,control],paired=differences,
        interpretations=[
            "Sparse diagnostics cover 51 batches, not all 1024 updates; dense windows cover all updates.",
            "Gain tensors, their individual gradients, and direct boundary changes are absent from historical logs.",
            "Conditional address memory preserves joint real/null at fixed competition input; recurrent state can change later competitions.",
            "High joint null does not alone prove empty object values: canonical spatial reads are conditionally normalized.",
            "Binary-command gripper uses command loss; continuous gripper flow audits are not active arm-only flow loss.",
            "Different physical GPUs preclude attributing wall-time differences solely to architecture.",
            "No final checkpoint means final gain values or correct-object behavior cannot be reconstructed from these logs.",
        ])
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"runs":[{k:v for k,v in x.items()
        if k in ("name","updates","samples","global_step","offline_epoch_rows","loss_ledger_max_abs",
                 "preclip_peak","checkpoint_files","address_metric_names","failure_lines")} for x in report["runs"]],
        "last224":{k:[x["statistics"][k]["last224"]["mean"] for x in report["runs"]]
            for k in ("loss_total","loss_action_flow","loss_action_flow_first8","loss_gripper_command",
                "object_grounding_null_mass","gradient_raw_dynamics_l2","runtime_window_seconds_per_batch")}},indent=2))

if __name__=="__main__":main()
