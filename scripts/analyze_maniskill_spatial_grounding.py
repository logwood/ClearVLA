"""Build a complete local, linked spatial audit from frozen probe artifacts."""
from __future__ import annotations
import argparse
import html
import json
import math
from pathlib import Path
import shutil
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(root,folder):
    return json.loads((root/folder/"summary.json").read_text())["rows"]


def joint_region(row,field,color):
    s=row[field]
    return (np.array(s["camera_mass"])*np.array(s["regions"][color+"_r12"]["conditional_mass"])).sum(-1)


def contours(ax,masks):
    for obj,color in ((0,"#ffdc49"),(1,"#40e3eb")):
        if masks[obj].any():ax.contour(masks[obj],levels=[.5],colors=[color],linewidths=.7)


def overlay(ax,rgb,density,masks,title):
    ax.imshow(rgb)
    # Smoothing is display only; all numeric masses use the unsmoothed exact push.
    mass=cv2.GaussianBlur(density.astype(np.float32),(0,0),4)
    maximum=max(float(mass.max()),1e-20)
    alpha=np.clip(np.sqrt(mass/maximum)*.75,0,.75)
    ax.imshow(mass/maximum,cmap="magma",vmin=0,vmax=1,alpha=alpha)
    contours(ax,masks);ax.set_title(title,fontsize=9);ax.axis("off")


def case_figure(npz,destination,title):
    with np.load(npz) as z:
        rgb=z["rgb"];masks=z["masks"];g=z["g_density"];p1=z["p1_micro_density"]
        binding=z["binding_probability"][0,:-1]
        selected=(g*binding[:,None,None,None]).sum(0)
        action=z["action"][:8,:3].mean(0)
    fig,axes=plt.subplots(2,3,figsize=(10.8,7.3))
    for c,camera in enumerate(("Top","Wrist")):
        axes[c,0].imshow(rgb[c]);contours(axes[c,0],masks[c])
        axes[c,0].set_title(camera+" RGB: yellow=red cube, cyan=green cube",fontsize=9);axes[c,0].axis("off")
        overlay(axes[c,1],rgb[c],selected[c],masks[c],"S binding × G spatial read (audit composition)")
        overlay(axes[c,2],rgb[c],p1[c],masks[c],"Actual P1 local RGB/detail read")
    fig.suptitle(title+f" | mean first-8 command XYZ = {action.round(4)}",fontsize=11)
    fig.text(.02,.012,"Bright = greater read mass; each map uses its own display scale. Gaussian smoothing is for display only. These maps are not object detections.",fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.95));fig.savefig(destination,dpi=130);plt.close(fig)


def table(headers,rows):
    return "<table><thead><tr>"+"".join("<th>"+html.escape(str(x))+"</th>" for x in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+str(x)+"</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table>"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--rollouts",type=Path,required=True)
    args=p.parse_args();r=args.root;(r/"figures").mkdir(exist_ok=True)
    resets=read(r,"resets-r1");experts=read(r,"experts-r1");rollouts=read(r,"rollouts-r1")
    placements=read(r,"placements-r1");causal=read(r,"causality-r2");producers=read(r,"producers-r1");swaps=read(r,"identity-r1")
    replay=json.loads((r/"rollouts-r1/summary.json").read_text())["episodes"]
    readouts=json.loads((r/"readout-r1/readouts.json").read_text())
    calibration=json.loads((r/"rgb_calibration_final.json").read_text())
    summary={"coverage":{"baseline_episodes_replayed":len(replay),"steps_per_episode":400,"failed_stage_queries":len(rollouts),
             "expert_stage_queries":len(experts),"fixed_robot_placements":len(placements),"physical_identity_swaps":len(swaps),
             "cache_factorial_pairs":len(causal),"frozen_readout_episodes":len(readouts["episodes"])},
             "replay":{"max_robot_error":max(x["robot_max_error"] for x in replay),
                       "max_object_error":max(x["object_max_error"] for x in replay),
                       "max_action_error_including_reset":max(x["action_max_error"] for x in replay),
                       "queried_actions_exact":sum(x["action_max_difference_from_archive"]==0 for x in rollouts),
                       "queried_actions_total":len(rollouts)},
             "resets":{},"stages":{},"causal":[]}
    for field in ("g_density","p1_micro_density"):
        summary["resets"][field]={}
        for color in ("red","green"):
            values=np.array([joint_region(x,field,color).max() for x in resets])
            summary["resets"][field][color]={"median_joint_region_mass":float(np.median(values)),
                 "min":float(values.min()),"max":float(values.max()),"below_one_percent_count":int((values<.01).sum())}
    for group,rows in (("expert",experts),("failure",rollouts)):
        summary["stages"][group]={}
        for stage in dict.fromkeys(x["stage"] for x in rows):
            selected=[x for x in rows if x["stage"]==stage]
            summary["stages"][group][stage]={"n":len(selected)}
            for field in ("g_density","p1_micro_density"):
                summary["stages"][group][stage][field]={color:float(np.median([joint_region(x,field,color).max() for x in selected])) for color in ("red","green")}
    for row in causal:
        v={i:np.array(row["mean8"][f"factorial_{i:03b}"]) for i in range(8)}
        shapley=[]
        for bit in range(3):
            change=np.zeros(3)
            for s in range(8):
                if s&(1<<bit):continue
                n=s.bit_count()
                change+=math.factorial(n)*math.factorial(2-n)/6*(v[s|(1<<bit)]-v[s])
            shapley.append(change.tolist())
        np.testing.assert_allclose(np.sum(shapley,axis=0),v[7]-v[0],atol=1e-10)
        summary["causal"].append(dict(label=row["label"],shapley_top_p1_ct=shapley,total_change=(v[7]-v[0]).tolist(),
                                       original_recovered_error=row["identity_a_max_error"],donor_recovered_error=row["identity_b_max_error"]))
    summary["identity_swap"]={"toward_new_red_count":sum(x["response_projection"]>0 for x in swaps),
        "count":len(swaps),"median_mean8_action_change_norm":float(np.median([x["action_change_norm"] for x in swaps]))}
    base_teacher=[x["teacher"] for x in producers if x["mode"]=="none"]
    summary["teacher"]={mode:{key:float(np.mean([x[mode][key] for x in base_teacher])) for key in
        ("same_observation_delta_rms","content_perturb_delta_rms","same_observation_transport_rms")} for mode in ("g_assignment_v1","raw_chart_v1")}
    summary["rgb_calibration"]=calibration["summary"]
    summary["frozen_readouts"]={k:{kk:vv for kk,vv in v.items() if kk not in ("predictions","val_component_rmse_by_alpha")} for k,v in readouts["rows"].items()}
    (r/"spatial_diagnosis.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")

    # The single opening chart makes action direction and the readout limitation visible.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for y,col in ((-.18,"#2774ae"),(.18,"#e28732")):
        subset=sorted([x for x in placements if x["moved"]=="red" and x["xy"][1]==y],key=lambda x:x["xy"][0])
        axes[0].plot([x["xy"][0]*100 for x in subset],[x["first8_mean_xyz"][0] for x in subset],"o-",color=col,label=f"red Y={100*y:.0f} cm")
    axes[0].axhline(0,color="black",lw=.8);axes[0].axvline(.4036,color="gray",ls=":",label="fixed robot TCP X")
    axes[0].set(xlabel="Red cube world X (cm)",ylabel="Mean first-8 native X command",title="Red cube at +10 cm: command still points toward -X")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    names=["RGB color +\nplanar calibration","G all features +\nlinear ridge","G geometry +\nlinear ridge","State only +\nlinear ridge"]
    vals=[calibration["summary"]["test"]["mean_xy_error_cm"],readouts["rows"]["g_all"]["test_mean_object_xy_error_cm"],
          readouts["rows"]["g_geometry"]["test_mean_object_xy_error_cm"],readouts["rows"]["state_only"]["test_mean_object_xy_error_cm"]]
    for j,color in enumerate(("#c94747","#39875d")):axes[1].bar(np.arange(4)+(j-.5)*.32,np.array(vals)[:,j],.32,color=color,label=("red","green")[j])
    axes[1].set_xticks(np.arange(4),names,fontsize=8);axes[1].set(ylabel="Mean XY position error (cm)",title="Seven test resets: diagnostic position readouts")
    axes[1].legend(fontsize=8);axes[1].grid(axis="y",alpha=.2)
    fig.text(.02,.015,"Different readout methods: this comparison does not prove information is absent from G, and it is not a controller success score.",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(r/"figures/spatial_findings.png",dpi=160);plt.close(fig)

    # Producer localization at each stage, no probability threshold hidden.
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    control=[x for x in producers if x["mode"]=="none" and x["moved"]=="red"]
    for row in control:
        label=f"({row['xy'][0]:+.2f},{row['xy'][1]:+.2f})"
        for c in range(2):
            fields=("source_density","g1_density","g2_density")
            values=[row[k]["regions"]["red_r12"]["conditional_mass"][0][c]*100 for k in fields]
            axes[c].plot(range(3),values,"o-",label=label)
            axes[c].set_xticks(range(3),["Source prior","After G1","After G2"])
            axes[c].set(title=("Top","Wrist")[c]+" camera: read near red cube",ylabel="Conditional camera mass (%)")
            axes[c].grid(alpha=.2)
    axes[1].legend(title="Red world XY (m)",fontsize=7,loc="upper right")
    fig.tight_layout();fig.savefig(r/"figures/producer_localization.png",dpi=160);plt.close(fig)

    sections=[]
    for seed in range(5000000,5000018):
        items=[("resets-r1",next(x for x in resets if x["seed"]==seed))]
        items += [("rollouts-r1",x) for x in rollouts if x["seed"]==seed]
        original=next(args.rollouts.glob(f"episode_*_seed_{seed}"))
        dest=r/"full_trajectories"/original.name;dest.mkdir(parents=True,exist_ok=True)
        for f in original.iterdir():
            if f.is_file() and (f.suffix in (".npz",".mp4",".json",".jsonl")):
                if not (dest/f.name).exists():shutil.copy2(f,dest/f.name)
        entries=[]
        for folder,row in items:
            filename=row["label"]+".jpg"
            case_figure(r/folder/(row["label"]+".npz"),r/"figures"/filename,f"Seed {seed} / {row.get('stage','reset')} / step {row.get('step',0)}")
            entries.append(f"<p><b>{html.escape(row.get('stage','reset'))}</b> — <a href='{folder}/{row['label']}.npz'>full probe NPZ</a></p><img loading='lazy' src='figures/{filename}'>")
        video=next(dest.glob("*.mp4"))
        source_links=f"<p><a href='full_trajectories/{original.name}/trajectory.npz'>FULL 400-action trajectory NPZ</a> · <a href='full_trajectories/{original.name}/{video.name}'>FULL video</a></p>"
        sections.append(f"<details><summary>Seed {seed}: reset, approach, closing, middle and late failure</summary>{source_links}{''.join(entries)}</details>")
    for label in ("red_x+0.00_y-0.08","red_x+0.10_y+0.18"):
        case_figure(r/"placements-r1"/(label+".npz"),r/"figures"/(label+".jpg"),"Fixed robot: "+label)

    repair={
        "status":"Diagnosis complete; no repaired checkpoint is qualified",
        "preserve":["one shared K+null identity","S intent ownership","camera and source-time axes","real P1 local reads","no object poses/masks as online inputs","Q5 proposal/refinement and replan8","all failures/full NPZ and videos"],
        "priority_1":"Train current object region identity/localization on existing G support with balanced per-object, per-visible-camera labels; use permutation matching across K and consistent cross-view identity. Small-object coverage must not disappear into whole-scene reconstruction.",
        "priority_2":"Qualify native initial reach direction, grasp alignment and post-grasp red-to-green relation separately. Current good attention cases can still produce wrong X actions; localization repair alone is insufficient.",
        "teacher_factor":"Test existing raw_chart_v1 separately. It removes direct learned-content reference sensitivity but preserves biased spatial association in this probe; it is not a standalone spatial fix.",
        "data_factor":"Add recovery demonstrations from observed missed approaches/empty closes and underrepresented occlusions, with episode-level held-out layouts. The earlier joint cold-start/dropout candidate failed and is not reused.",
        "qualification":["parameter-owner VJP and no future/oracle input leakage","held-out region localization and cross-view correspondence","red/green identity swaps and signed X/Y action response","same 18 seeds, 400 steps, replan8, no inference overrides","first-close XY miss, lifts, object-to-object placement, stack success","full finite training/validation curve and source/normalizer identity"],
        "rejected_inference_shortcuts":["remove G1 query update","remove G2 parent prior","uniform G2 posterior"]
    }
    (r/"repair_experiment.json").write_text(json.dumps(repair,indent=2)+"\n")
    factrows=[
        ("All 18 archived trajectories","The 7-D policy robot state (TCP position, causal rotation vector, gripper opening) and both cube XYZ positions match exactly at all 401 observations; 72 stage queries. This does not assert equality of every simulator state field."),
        ("Replay action numerics",f"{summary['replay']['queried_actions_exact']}/72 stage actions exactly match; maximum including reset {summary['replay']['max_action_error_including_reset']:.6f} native units. Small nonzero replay differences are retained."),
        ("Color identity","18/18 physical red/green exchanges shift the action toward the new red position. Identity is not entirely ignored."),
        ("Localization","Coverage varies sharply by location and stage. Read centers alone were insufficient evidence; full distributions and renderer masks are used here."),
        ("Execution","Both controlled red X=+10 cm placements retain negative first-8 X commands, including a case with strong cube coverage."),
        ("Teacher","Raw-chart reference removes direct learned-content perturbation sensitivity, but same-image association still has nonzero semantic and geometric drift."),
        ("Qualification","Original autonomous baseline remains 0/18. These frozen probes do not establish a successful repaired policy.")]
    readoutrows=[]
    for name in ("state_only","dino_spatial","g_content","g_geometry","g_all","p1_detail"):
        v=readouts["rows"][name]
        readoutrows.append((name,*(f"{x:.2f}" for x in v["test_mean_object_xy_error_cm"])))
    readoutrows.append(("RGB color / planar calibration",*(f"{x:.3f}" for x in calibration["summary"]["test"]["mean_xy_error_cm"])))
    stage_rows=[]
    for group,stages in summary["stages"].items():
        for stage,values in stages.items():
            stage_rows.append((group,stage,values["n"],*(f"{100*values[field][color]:.2f}%" for field in ("g_density","p1_micro_density") for color in ("red","green"))))
    body=f"""<!doctype html><html><head><meta charset="utf-8"><title>ManiSkill spatial grounding audit</title>
<style>body{{max-width:1150px;margin:36px auto;padding:0 20px;font:16px/1.5 system-ui;color:#17212b;background:#fafafa}}h1,h2{{line-height:1.2}}img{{max-width:100%;height:auto;border:1px solid #ccd3dc}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border:1px solid #ccd3dc;text-align:left;vertical-align:top}}th{{background:#e9eef4}}details{{border:1px solid #c8d1dd;padding:12px;margin:12px 0;background:white}}summary{{cursor:pointer;font-weight:600}}.lead{{font-size:20px}}.note{{background:#edf3f8;padding:16px}}code{{word-break:break-all}}a{{color:#1256a0}}</style></head><body>
<h1>ManiSkill: where spatial grounding breaks</h1><p class="lead">The failure combines unreliable object localization with an inaccurate spatial-to-action mapping. Neither complete visual blindness nor a single missing camera explains it.</p>
<p>Frozen checkpoint <code>a6e4158bb995083e…</code>, epoch 8 / step 6392. Original core source closure retained. No model weights or deployment configuration changed.</p>
<img src="figures/spatial_findings.png"><h2>What is established</h2>{table(("Probe","Finding"),factrows)}
<h2>How to read the spatial maps</h2><p>Yellow outlines the physical red cube; cyan outlines green, using the renderer's actor IDs. Bright regions show where the model reads image evidence. G has four learned slots; these are not guaranteed cube identities or calibrated 3D positions. S supplies the shared target binding. The G×S panel is an attribution composition, while P1 shows its actual local RGB/detail sampling law, averaged across factual queries and nine microcells. P1 also receives a separate G-based target-value read, which is not this map.</p>
<p>Quantitative regions expand each visible cube mask by 12 pixels to account for the sparse feature grid. We retain raw masks and the full probability distributions. Camera-conditional mass and joint mass are separate; a high conditional score in a nearly unused camera must not be mistaken for high global allocation.</p>
<img src="figures/red_x+0.00_y-0.08.jpg"><img src="figures/red_x+0.10_y+0.18.jpg">
<h2>Why the problem persists</h2><ol>
<li><b>The representation has no guaranteed object calibration.</b> G exports normalized image read coordinates and learned content. Whole-scene DINO reconstruction does not explicitly assign a physical cube to a slot. Strong coverage in one layout can coexist with almost none in another.</li>
<li><b>The execution mapping is also wrong.</b> The controlled +X targets still produce -X commands. Named-cache factorials recover both original and donor actions exactly and attribute much of the movement response to G/S/W, with meaningful P1 contributions in some directions. The path is active; its learned response is inaccurate.</li>
<li><b>The reference target has a moving-value risk.</b> A 1% perturbation to learned G content changes the active Teacher delta with fixed images and fixed addresses. Raw-chart mode removes that dependency, but does not fix spatial association. Same-frame soft association can remain biased; its nonzero drift is a calibration warning, not proof of the sole training cause.</li>
<li><b>Precision and recovery need separate qualification.</b> Expert-state action accuracy is much better after approach. This audit still finds weak local detail allocation at several stages, and the original autonomous trajectories close off target. Successful expert-state inference does not qualify autonomous recovery.</li></ol>
<h2>Producer ablations</h2><img src="figures/producer_localization.png"><p>Removing G1's query update, removing G2's parent spatial prior, or making G2 uniform did not fix the negative-X response. Several Y responses worsened. These diagnostics rule out those simple inference patches; they do not prove that training with better spatial supervision will fail.</p>
<h2>Frozen-feature readouts</h2>{table(("Diagnostic readout","Red test XY error (cm)","Green test XY error (cm)"),readoutrows)}
<p>Ridge readouts fit 59 training resets; alpha is selected on seven validation resets; seven test resets are evaluated afterward. These small-sample linear readouts do not bound what a nonlinear decoder could recover. The RGB diagnostic uses a training-fitted planar homography and task-specific color masks; all 14 test objects pass visibility admission. Five red and five green training examples are unlocalized under the 20-pixel gate. It is not a general vision model or a robot controller. The seven test green errors include a 1.885 cm partially occluded outlier.</p>
<h2>Stage measurements</h2><p>Median joint read mass within visible object +12px. G reports the best of its four slots; P1 reports the averaged local detail read. Neither is a segmentation accuracy score.</p>
{table(("Source","Stage","Queries","G red","G green","P1 red","P1 green"),stage_rows)}
<h2>Concrete repair experiment</h2><p>First qualify training-only region and multiview identity supervision on the existing G support, with per-object balancing and explicit observed masks. Preserve the shared K+null binding and all online input boundaries. Separately test native reach/grasp/placement consistency and recovery data; the observed wrong-X case already shows that a localization-only fix is insufficient. Test raw-chart Teacher reference as its own factor. Use an unchanged-recipe control and promote only after the same full 18-episode closed-loop panel passes.</p>
<p><a href="repair_experiment.json">Machine-readable repair specification and gates</a> · <a href="spatial_diagnosis.json">All decision metrics</a> · <a href="readout-r1/readouts.json">Readout results</a> · <a href="rgb_calibration_final.json">RGB calibration / visibility receipt</a> · <a href="probe-tests.log">Audit primitive tests</a></p>
<h2>Every original failure, with full trajectories</h2><p>Each case links the complete 400-action NPZ and full video, plus reset and all four probed stages. No success/failure filtering is applied.</p>{''.join(sections)}
<h2>Reproducible evidence</h2><p><a href="placements-r1/summary.json">Placement probes</a> · <a href="identity-r1/summary.json">Identity swaps</a> · <a href="causality-r2/summary.json">2³ cache factorials</a> · <a href="producers-r1/summary.json">Producer and Teacher probes</a> · <a href="experts-r1/summary.json">All 42 expert stages</a> · <a href="rollouts-r1/summary.json">All 72 failed stages</a> · <a href="resets-r1/provenance.json">Frozen checkpoint identity</a></p>
<p><a href="verification.json">Artifact verification receipt</a> · <a href="diagnostic_attempts.json">Failed and superseded diagnostic attempts</a> · <a href="reproduce.txt">Reproduction commands</a> · <a href="source_manifest.json">Diagnostic source hashes</a> · <a href="all_18_full_trajectories_npz.zip">All 18 full trajectory NPZ files</a></p>
<p class="note">The original 0/18 baseline and failed cold-start/dropout candidate remain unqualified. No training job or deployment was launched by this spatial audit. Raw NPZ data, complete videos, failed diagnostic attempts, source hashes and numerical caveats are preserved with the report.</p></body></html>"""
    (r/"index.html").write_text(body,encoding="utf-8")
    print(json.dumps(summary["coverage"]));print(json.dumps(summary["replay"]))


if __name__=="__main__":main()

