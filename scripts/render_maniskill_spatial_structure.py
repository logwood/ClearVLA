"""Render two explicit saved-data examples from the offline structural audit."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def font(size):
    for name in ('C:/Windows/Fonts/arial.ttf','DejaVuSans.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()

def cross(draw,point,color,size=9,width=4):
    x,y=point;draw.line((x-size,y-size,x+size,y+size),fill=color,width=width)
    draw.line((x-size,y+size,x+size,y-size),fill=color,width=width)

def diamond(draw,point,color,size=9):
    x,y=point;draw.line([(x,y-size),(x+size,y),(x,y+size),(x-size,y),(x,y-size)],fill=color,width=3)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.audit.read_text());out=args.audit.parent
    views=data['views'];cases=data['cases']
    target=next(x for x in views if x['arm']=='candidate' and x['case']=='test_expert_000042_preclose' and x['object']=='red' and x['camera']=='global')
    with np.load(target['path']) as z:
        rgb=z['rgb'][0];density=z['g_density'][target['slot'],0].astype(float)
    canvas=Image.new('RGB',(1040,680),'#101827');d=ImageDraw.Draw(canvas)
    d.text((26,18),'A real averaging error in the current position',font=font(27),fill='white')
    d.text((26,56),'Saved test episode 42, before closing | global camera | slot 3',font=font(18),fill='#c5d5e8')
    scale=1.4;side=round(336*scale)
    canvas.paste(Image.fromarray(rgb).resize((side,side)),(28,109))
    # A local 5x5 sum makes sparse pushed-forward pixels visible, not a new measurement.
    from analyze_maniskill_spatial_collapse import box_sum
    mass=box_sum(density,2);heat=np.log1p(1000*mass/max(mass.max(),1e-30))/np.log(1001)
    color=np.stack((35+180*heat,48+170*heat,75-30*heat),axis=-1).clip(0,255).astype('uint8')
    canvas.paste(Image.fromarray(color).resize((side,side)),(540,109))
    for xoff in (28,540):
        mean=tuple(np.array(target['mean'])*scale+[xoff,109])
        actual=tuple(np.array(target['target'])*scale+[xoff,109])
        mode=tuple(np.array(target['mode'])*scale+[xoff,109])
        d.line([mean,actual],fill='#ffdd4a',width=2)
        d.ellipse([actual[0]-15,actual[1]-15,actual[0]+15,actual[1]+15],outline='white',width=3)
        cross(d,mean,'#ffdd4a')
        diamond(d,mode,'#22f1ee',7)
    d.text((28,84),'Observed RGB',font=font(18),fill='white')
    d.text((540,84),'Saved slot probability (bright = more mass)',font=font(18),fill='white')
    d.text((28,593),'White circle: actual cube     Yellow X: exported mean     Cyan diamond: strongest local peak',font=font(19),fill='white')
    d.text((28,629),'Mean error: 83.5 px. Peak error: 2.8 px. The mean falls far from the visible cube.',font=font(22),fill='#ffdd4a')
    canvas.save(out/'averaging-example.png')
    case=next(x for x in cases if x['arm']=='candidate' and x['case']=='val_expert_000036_reset')
    cv=next(s for s in case['cross_view_identity'] if s['slot']==1)
    with np.load(case['path']) as z: rgb=z['rgb']; masks=z['masks']
    canvas=Image.new('RGB',(1040,690),'#101827');d=ImageDraw.Draw(canvas)
    d.text((26,18),'The same slot associates with different cubes',font=font(27),fill='white')
    d.text((26,56),'Saved validation episode 36, reset | slot 1 in both views',font=font(19),fill='#c5d5e8')
    for ci,xoff in enumerate((28,540)):
        canvas.paste(Image.fromarray(rgb[ci]).resize((side,side)),(xoff,109))
        for oi,color in ((0,'#ff645d'),(1,'#7bf27c')):
            y,x=np.where(masks[ci,oi])
            if len(x):
                box=[x.min()*scale+xoff-3,y.min()*scale+106,x.max()*scale+xoff+3,y.max()*scale+112]
                d.rectangle(box,outline=color,width=3)
        point=tuple(np.array(case['slot_mode'][1][ci])*scale+[xoff,109])
        diamond(d,point,'white',10)
        d.text((xoff,84),('Global: strongest peak on RED','Wrist: strongest peak near GREEN')[ci],font=font(19),fill='white')
        redmass,greenmass=np.array(cv['region_mass'])[:,ci]
        d.text((xoff,592),f'Red-region mass: {redmass:.1%} | green: {greenmass:.1%}',font=font(18),fill='white')
    d.text((28,636),'One shared K index does not yet guarantee one physical object across cameras.',font=font(22),fill='#ffdd4a')
    canvas.save(out/'cross-view-identity-example.png')
    summary=data['aggregate']['candidate']
    rows=''
    for label,key in [('Held-out expert states','heldout'),('Fixed failed-policy states','failed_states'),('Placement controls','placements')]:
        v=summary[key]['matched_views']; c=summary[key]['cases']
        rows+=f"<tr><td>{label}</td><td>{v['count']}</td><td>{v['mean_error_px']['mean']:.2f}</td><td>{v['mode_error_px']['mean']:.2f}</td><td>{v['mean_far_mode_near_count']}</td><td>{c['binding_effective_real_slots']['mean']:.2f}/4</td></tr>"
    report=f"""<!doctype html><meta charset="utf-8"><title>ManiSkill spatial structure audit</title>
<style>body{{max-width:1080px;margin:40px auto;font:17px/1.55 system-ui;color:#162031;background:#f4f7fa;padding:0 20px}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{border:1px solid #c9d4df;padding:9px;text-align:left}}img{{max-width:100%;border-radius:8px}}code{{font-size:14px}}.box{{padding:18px;background:white;border-left:5px solid #007c91;margin:24px 0}}a{{color:#006c91}}</style>
<h1>Where approximate localization loses precision</h1>
<p>Checkpoint: spatial candidate, update 6648. This audit reads existing arrays and uses one short CPU check. It starts no training, simulator rollout, or GPU job.</p>
<div class="box"><strong>Finding:</strong> there is no wholesale four-slot collapse. There is measured within-slot spatial averaging, cross-view identity inconsistency, and diffuse shared target weighting. The explicit current-position coordinate branch of W/P2 keeps only a mean. These are concrete structural bottlenecks; the audit does not establish one of them as the sole cause of all failed actions.</div>
<h2>1. A good local peak can produce a bad exported coordinate</h2>
<img src="averaging-example.png" alt="One saved slot: 83.5-pixel mean error despite a 2.8-pixel local peak error">
<p>In 67/312 visible held-out object/view entries, the mean is over 24 pixels from the cube while the strongest label-blind local peak is within 24 pixels. Posterior RMS spread averages 82.16 pixels. The exported coordinates agree with recomputed moments to within 0.000140 pixels: this is the actual current coordinate summary, not a plotting offset.</p>
<p><code>ImageLogMeasure.camera_centers()</code> computes E[x,y]. Grounding exports it as <code>facts.camera_coordinates</code>; W carries it forward, and P2 uses it for current/future position scores and geometry context. A short CPU test of the trained P2 confirms that disjoint left/right and up/down distributions with equal means give identical current-position context at fixed transport covariance.</p>
<p>S has a separate path: it computes E[phi(x,y)] over the original law before a task/robot relation. This retains some shape information. The selected v1 graph pools that spatial feature before robot/task interaction. On the 9x9 audit grid, an affine fit explains 99.25% of the trained coordinate feature variance; this is a narrow feature-map diagnostic, not proof that the complete S output is linear.</p>
<h2>2. One slot can refer to different cubes in different views</h2>
<img src="cross-view-identity-example.png" alt="Slot 1 reads mainly the red cube in the global camera and the green cube in the wrist camera">
<p>Among 41 held-out queries with both cubes visibly present in both cameras, 15 contain at least one confidently conflicting slot. Of 76 eligible slot/query comparisons, 18 conflict, including 13 evaluator-matched object slots. Eligibility requires at least 20 visible pixels per cube/view and at least 20% combined cube-region mass in each view; a conflict requires red fractions above 70% in one view and below 30% in the other. Counts use 12-pixel dilated regions and are descriptive, not physical identity labels available online. Control comparisons have much lower eligible coverage, so raw conflict counts cannot establish a regression.</p>
<h2>3. Shared target mixing remains broad</h2>
<p>The candidate shared target law uses an average of 3.78 effective real slots out of four. On these held-out queries it gives an average 18.93% absolute mass to the evaluator-matched red slot, 22.03% to green, and 23.47% to null. In shared-binding P2, the semantic K marginal is copied from this law; action-dependent selection chooses a camera inside K, not a new object. The same diffuse identity therefore reaches several consumers.</p>
<p>A composed binding-weighted image map is an attribution diagnostic, not a literal final target-coordinate output. Its red-centroid error is 92.99 pixels versus 53.54 for the evaluator-matched red slot. Actual consumers pool transformed values, so this does not prove that the robot directly executes that image centroid.</p>
<table><tr><th>Saved panel</th><th>Visible object/views</th><th>Mean error px</th><th>Local-mode error px</th><th>Mean far / mode near</th><th>Effective real target slots</th></tr>{rows}</table>
<h2>Repair decision</h2>
<p>Defer the proposed extra action-loss training. First repair the representation-to-consumer contract: preserve spatial alternatives until the robot/task-conditioned geometry read, qualify consistent physical identity across views, and keep target versus reference-object evidence distinguishable through the relation computation. Retain the single shared K+null owner, separate camera charts, and label-free online path.</p>
<p>A hard peak replacement is not qualified: across the held-out entries its mean error is 55.81 pixels, worse than the current centroid's 52.01. Some dominant peaks are on the wrong object or background. The next source change needs a small equal-mean/different-layout regression and a few fixed real cases before any longer training comparison.</p>
<p>Coverage: 98 held-out queries, 72 fixed failed-policy queries, and 12 placement cases per arm; all saved cases are included. Slots are evaluator-matched with one permutation shared across cameras. Pixel distances are not metric object-position estimates. <a href="collapse-audit.json">Full audit data</a> · <a href="moment-collision.json">Trained-weight CPU check</a></p>"""
    (out/'index.html').write_text(report,encoding='utf-8')
    print(out/'index.html')

if __name__=='__main__':main()
