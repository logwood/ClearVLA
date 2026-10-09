"""Consolidate completed, existing probes; no model execution or new labels."""
import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "B_v1":"B-matched-factual-short-r1",
    "B_v2":"B-nullv2-matched-factual-short-r1",
    "B_v3":"B-regions-v3-matched-factual-short-r1",
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def summary(values):
    return dict(count=len(values),median=statistics.median(values),maximum=max(values)) if values else dict(count=0)

def collect(root, folder):
    path=root/folder/"results.json"
    doc=json.loads(path.read_text())
    if doc.get("complete") is not True:raise ValueError("unfinished factual probe")
    records=doc["records"]
    indexed={(row["case_id"],row["step"]):row for row in records}
    if len(indexed)!=len(records):raise ValueError("duplicate physical window")
    pairs={}
    for key,row in indexed.items():
        physical={(p["camera"],p["object_index"]):p for p in row["physical_identity"]
            if p["object_source_mass"]>0 and p["object_supported_pixels"]>0}
        for (camera,i),a in physical.items():
            for (other,j),b in physical.items():
                if camera!=other or i>=j:continue
                tv=sum(abs(x-y) for x,y in zip(a["K_distribution"],b["K_distribution"]))/2
                if abs(tv-a["same_view_other_object_tv"][str(j)])>1e-6:
                    raise ValueError("stored physical read TV does not reproduce")
                pairs[(*key,camera,i,j)]=(tv,a["dominant_K"]==b["dominant_K"])
    naturals=[n for row in records for n in row["natural"]]
    repeat=[n["native_arm_difference_rms"] for n in naturals if n["repeat"]]
    colors=[n["native_arm_difference_rms"] for n in naturals if not n["repeat"]]
    result=dict(path=str(path),sha256=sha(path),identity=doc["identity"],windows=len(records),
        same_instruction_repeat=summary(repeat),color_arm_response=summary(colors),
        maximum_G_content_change=max(n["G_content_difference_max"] for n in naturals),
        same_image_max=max(row["static_contract"][k] for row in records for k in
                           ("semantic_max","image_max","covariance_max")))
    return indexed,pairs,result

def ceilings(indexed):
    output=[]
    for case,target,competitor in ((5,"block_blue","block_red"),(11,"block_red","block_pink")):
        row=indexed[(case,24)]
        t=row["object_names"].index(target);c=row["object_names"].index(competitor)
        k=len(row["coverage"][0]["k_object_read"]);objects=len(row["object_names"])
        reads=[[sum(v["k_object_read"][slot][o] for v in row["coverage"])
                for o in range(objects)] for slot in range(k)]
        ratios=[x[t]/(x[t]+x[c]) for x in reads if x[t]+x[c]>0]
        natural=row["natural"][0]
        if not natural["repeat"]:raise ValueError("missing factual instruction")
        real=[sum(v[o] for v in natural["selected_object_read"]) for o in range(objects)]
        areas=[sum(v["visible_pixels"][o] for v in row["coverage"]) for o in range(objects)]
        adjusted=[(x[t]/areas[t])/(x[t]/areas[t]+x[c]/areas[c]) for x in reads if x[t]+x[c]>0]
        output.append(dict(case=case,step=24,target=target,competitor=competitor,
            actual=real[t]/(real[t]+real[c]),nonnegative_K_ceiling=max(ratios),
            visible_area_adjusted_ceiling=max(adjusted)))
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data={name:collect(args.experiment_root,folder) for name,folder in SOURCES.items()}
    first=data["B_v1"][0]
    for indexed,_,_ in data.values():
        if set(indexed)!=set(first):raise ValueError("different physical window inventory")
        for key,row in indexed.items():
            ref=first[key]
            if row["object_names"]!=ref["object_names"]:raise ValueError("object labels differ")
            for v,w in zip(row["coverage"],ref["coverage"],strict=True):
                for field in ("visible_pixels","object_supported_pixels"):
                    if v[field]!=w[field]:raise ValueError("physical support differs")
    shared=set.intersection(*(set(value[1]) for value in data.values()))
    all_equal=all(set(value[1])==shared for value in data.values())
    versions={}
    for name,(indexed,pairs,result) in data.items():
        result["different_object_pairs"]={}
        for camera in (0,1):
            keys=[key for key in shared if key[2]==camera]
            result["different_object_pairs"][str(camera)]=dict(
                **summary([pairs[key][0] for key in keys]),
                same_dominant_K=sum(pairs[key][1] for key in keys))
        result["fixed_G_source_read_ceilings"]=ceilings(indexed)
        versions[name]=result
    report=dict(schema="bv1-v2-v3-decision-evidence-v1",utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),
        scope="Existing same-observation supported real-K physical reads; not semantic identity or full policy capacity.",
        identical_window_and_physical_support=True,all_supported_pair_sets_equal=all_equal,
        common_pair_count=len(shared),versions=versions,
        cautions=[
            "These are conditional real-K reads, distinct from joint ownership null and correspondence unknown.",
            "Natural color response magnitude does not establish correct direction or target selection.",
            "Fixed-G convex ceilings constrain only this global source READ, not semantic information or full P1/action routes.",
            "Broad-run ceilings retain their runtime identity; do not overwrite earlier focused deterministic probe values.",
            "Three final checkpoints were independently trained; aggregation is not a controlled training ablation.",
        ])
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
