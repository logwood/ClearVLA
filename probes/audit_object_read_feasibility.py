"""Convex-mixture bounds on fixed G spatial reads, using admitted physical masks.

Masks are audit truth only. These bounds apply to fixed per-K/per-view read
distributions; a later query-dependent within-K spatial read can change them.
They are not bounds on all policy actions or on information in DINO features.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
    q=argparse.ArgumentParser()
    q.add_argument("--results",type=Path,required=True)
    q.add_argument("--plan",type=Path,required=True)
    q.add_argument("--output",type=Path,required=True)
    a=q.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    plan={r["case_id"]:r for r in json.loads(a.plan.read_text())}
    output=[]
    for row in json.loads(a.results.read_text()):
        v=next(x for x in row["variants"] if x["variant"]=="baseline")
        cameras=v["G_coverage"]["cameras"];names=cameras[0]["object_names"]
        if any(c["object_names"]!=names for c in cameras):raise ValueError("object identities differ across views")
        mass=np.asarray([c["k_object_probability"] for c in cameras],float)
        if mass.shape!=(2,4,3) or np.any(mass<0):raise ValueError("unexpected conditional source measure")
        for c in cameras:
            if not np.allclose(c["k_total_probability"],1,atol=2e-6):raise ValueError("source not normalized")
        total=mass.sum(-1);shares=np.divide(mass,total[...,None],out=np.zeros_like(mass),where=total[...,None]>0)
        objects=[]
        for j,name in enumerate(names):
            c,k=np.unravel_index(np.argmax(shares[:,:,j]),shares[:,:,j].shape)
            objects.append({"object":name,"max_source_probability":float(mass[:,:,j].max()),
                "max_share_among_visible_object_pixels":float(shares[c,k,j]),
                "share_maximizer":{"camera":cameras[c]["camera"],"K":int(k+1),
                    "target_probability":float(mass[c,k,j]),"all_object_probability":float(total[c,k])},
                "dominated_in_every_K_view_by":[n for z,n in enumerate(names) if z!=j and bool(np.all(mass[:,:,j]<=mass[:,:,z]))]})
        # This is one declared source-law mixture, not the later S/P1 view reads.
        camera_mass=np.asarray(v["global_K_camera_mass"],float)
        binding=np.asarray(v["binding"],float)
        g_joint=np.einsum("kc,cko->ko",camera_mass,mass)
        mixed=binding@g_joint
        output.append({"case_id":row["case_id"],"state":row["state"],"target":plan[row["case_id"]]["target"],
            "objects":objects,"fixed_G_global_camera_then_binding_object_mass":dict(zip(names,mixed.tolist())),
            "real_binding_mass":float(binding.sum()),"source_mask":v["G_coverage"]["mask_path"]})
    report={"rows":output,"results":str(a.results),"results_sha256":hashlib.sha256(a.results.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"convex mixtures of fixed G conditional spatial reads only; no masks fed to model; later query-dependent spatial reselection can exceed these bounds",
        "warning":"high visible-object share at negligible total object mass is not healthy coverage"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    for r in output:
        print(json.dumps({k:r[k] for k in ["case_id","state","target","objects"]}))
if __name__=="__main__":main()
