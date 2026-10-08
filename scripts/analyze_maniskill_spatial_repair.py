"""Compare the predeclared repair arms using renderer labels and complete episodes."""
from __future__ import annotations
import argparse
import html
from itertools import permutations
import json
import math
from pathlib import Path
import shutil
import zipfile
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):return json.loads(path.read_text())
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def grounding(path):
    with np.load(path,allow_pickle=False) as z:
        density=z['g_density'].astype(np.float64)
        masks=z['masks'].transpose(1,0,2,3)
        binding=z['binding_probability'][0,:-1].astype(np.float64)
    slots,cameras,height,width=density.shape
    visible=masks.sum((-2,-1))>=20
    grid=np.stack(np.meshgrid(np.linspace(-1,1,width),np.linspace(-1,1,height)),axis=-1)
    camera_mass=density.sum((-2,-1))
    conditional=density/np.maximum(camera_mass[:,:,None,None],1e-30)
    center=np.einsum('kcyx,yxd->kcd',conditional,grid)
    target=np.einsum('ocyx,yxd->ocd',masks.astype(float),grid)/np.maximum(masks.sum((-2,-1))[...,None],1)
    region=np.stack([[cv2.dilate(m.astype(np.uint8),np.ones((25,25),np.uint8))>0 for m in views] for views in masks])
    mass=np.einsum('kcyx,ocyx->koc',density,region.astype(float))
    nviews=np.maximum(visible.sum(-1),1)
    nll=np.where(visible[None],-np.log(np.maximum(mass*nviews[None,:,None],1e-30)),0).sum(-1)/nviews
    distance=((center[:,None]-target[None])**2).sum(-1)
    mean_distance=np.where(visible[None],distance,0).sum(-1)/nviews
    cost=nll+mean_distance
    choice=min(permutations(range(slots),2),key=lambda pair:cost[pair[0],0]+cost[pair[1],1])
    rows=[]
    for obj,name in enumerate(('red','green')):
        k=choice[obj]
        for camera in range(cameras):
            if not visible[obj,camera]:continue
            rows.append(dict(object=name,camera=camera,slot=k,
                conditional_region_mass=float(mass[k,obj,camera]/max(camera_mass[k,camera],1e-30)),
                joint_region_mass=float(mass[k,obj,camera]),
                centroid_error_px=float(np.sqrt(distance[k,obj,camera])*(height-1)/2),
                camera_mass=float(camera_mass[k,camera])))
    return dict(views=rows,
        binding_composed_red_region_mass=float(np.einsum('k,kc->',binding,mass[:,0])),
        binding_composed_green_region_mass=float(np.einsum('k,kc->',binding,mass[:,1])),
        binding_to_matched_red=float(binding[choice[0]]),
        binding_to_matched_green=float(binding[choice[1]]),
        binding_null_mass=float(1-binding.sum()))


def paired_episode_interval(control, candidate, key):
    """Bootstrap episodes, keeping the repeated stage/view measurements together."""
    names=sorted({x['episode'] for x in control})
    differences=[]
    for name in names:
        a=[x[key] for x in control if x['episode']==name]
        b=[x[key] for x in candidate if x['episode']==name]
        assert len(a)==len(b) and len(a)>0
        differences.append(float(np.mean(b)-np.mean(a)))
    values=np.asarray(differences)
    rng=np.random.default_rng(20261008)
    samples=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(-1)
    return dict(candidate_minus_control=float(values.mean()),
        episode_bootstrap_95_percentile=np.quantile(samples,[.025,.975]).tolist(),
        per_episode=dict(zip(names,differences)),
        note=f'Descriptive paired episode bootstrap; one training seed, {len(names)} paired episodes.')


def failed_state_comparison(folder):
    """All preselected failed-policy states, without inventing action targets."""
    status=load(folder/'status.json')
    if status['status']!='complete':raise RuntimeError('Failed-state diagnostic is not complete: '+str(status))
    output={'note':'Same original-policy states, not expert action labels or new closed-loop results.',
        'arms':{},'paired_intervals':{},'per_stage':{}}
    views={}
    for arm in ('control','candidate'):
        evidence=load(folder/arm/'summary.json')
        assert evidence['complete'] and len(evidence['rows'])==72 and len(evidence['episodes'])==18
        assert all(x['robot_state_max_error']<=1e-6 and x['object_position_max_error']<=1e-6 for x in evidence['episodes'])
        if arm=='candidate':assert all(x['paired_inputs_exact'] for x in evidence['rows'])
        views[arm]=[dict(episode=x['episode'],label=x['label'],stage=x['stage'],**v)
            for x in evidence['rows'] for v in x['grounding']['views']]
        output['arms'][arm]=dict(episodes=18,queries=72,visible_object_views=len(views[arm]),
            queries_without_visible_cube=sum(not x['grounding']['views'] for x in evidence['rows']),
            mean_conditional_region_mass=float(np.mean([x['conditional_region_mass'] for x in views[arm]])),
            mean_centroid_error_px=float(np.mean([x['centroid_error_px'] for x in views[arm]])))
        output['per_stage'][arm]={}
        for stage in sorted({x['stage'] for x in views[arm]}):
            chosen=[x for x in views[arm] if x['stage']==stage]
            output['per_stage'][arm][stage]=dict(visible_object_views=len(chosen),
                mean_conditional_region_mass=float(np.mean([x['conditional_region_mass'] for x in chosen])),
                mean_centroid_error_px=float(np.mean([x['centroid_error_px'] for x in chosen])))
    assert [(x['label'],x['object'],x['camera']) for x in views['control']]==[(x['label'],x['object'],x['camera']) for x in views['candidate']]
    for key in ('conditional_region_mass','centroid_error_px'):
        output['paired_intervals'][key]=paired_episode_interval(views['control'],views['candidate'],key)
    return output


def placement_slopes(rows):
    """Finite differences along each axis, holding the other coordinate fixed."""
    result=[]
    for moved in ('red','green'):
        selected={tuple(x['xy']):x for x in rows if x['moved']==moved}
        pairs=[(0,(-.10,y),(.10,y)) for y in (-.18,.18)]
        pairs += [(1,(x,-.18),(x,.18)) for x in (-.10,.10)]
        pairs += [(1,(0.,-.08),(0.,.08))]
        for axis,lo,hi in pairs:
            a,b=selected[lo],selected[hi]
            delta=np.array(b['first8_mean_xyz'])-np.array(a['first8_mean_xyz'])
            result.append(dict(moved=moved,axis='xy'[axis],low_xy=list(lo),high_xy=list(hi),
                mean_native_xyz_delta=delta.tolist(),same_axis_delta_per_metre=float(delta[axis]/(hi[axis]-lo[axis]))))
    return result


def placement_figure(root,destination):
    """All physical placement pairs, with direction-only command arrows."""
    control=load(root/'control-placements/summary.json')['rows']
    candidate=load(root/'candidate-placements/summary.json')['rows']
    assert len(control)==len(candidate)==12
    fig,axes=plt.subplots(3,4,figsize=(14,11),sharex=True,sharey=True)
    for ax,a,b in zip(axes.ravel(),control,candidate):
        assert a['label']==b['label']
        np.testing.assert_array_equal(a['positions'],b['positions'])
        np.testing.assert_array_equal(a['tcp_xyz'],b['tcp_xyz'])
        tcp=np.array(a['tcp_xyz'][:2])*100
        objects=np.array(a['positions'])[:,:2]*100
        ax.scatter(*objects[0],marker='s',s=85,color='#d94c46',label='Red cube')
        ax.scatter(*objects[1],marker='s',s=85,color='#299854',label='Green cube')
        ax.scatter(*tcp,marker='o',s=25,color='#222',label='TCP')
        ax.plot([tcp[0],objects[0,0]],[tcp[1],objects[0,1]],':',color='#888',linewidth=1)
        for row,color,name in [(a,'#366cb0','Control command'),(b,'#a33bb2','Repair command')]:
            vector=np.array(row['first8_mean_xyz'][:2])
            length=np.linalg.norm(vector)
            if length>1e-10:
                v=vector/length*6
                ax.arrow(tcp[0],tcp[1],v[0],v[1],color=color,width=.18,head_width=1.4,length_includes_head=True,alpha=.85,label=name)
        ax.set_title(a['moved']+' moved to ('+str(round(a['xy'][0]*100))+', '+str(round(a['xy'][1]*100))+') cm',fontsize=11)
        ax.set(xlim=(-22,22),ylim=(-24,24),aspect='equal',xlabel='World X (cm)',ylabel='World Y (cm)')
        ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=10)
    fig.suptitle('All 12 placement interventions: initial command direction\nArrows have equal display length; they are not predicted TCP travel.',fontsize=15)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(destination,dpi=150);plt.close(fig)


def episodes(folder,expected_count=18):
    rows=[]
    for d in sorted(folder.glob('episode_*')):
        if not (d/'result.json').exists():continue
        result=load(d/'result.json')
        with np.load(d/'trajectory.npz',allow_pickle=False) as z:
            command=z['executed'];state=z['robot_obs_trajectory'];objects=z['object_positions']
            assert command.shape==(400,7) and state.shape==(401,15)
            assert objects.shape==(401,3,3)
            assert all(np.isfinite(x).all() for x in (command,state,objects))
            assert result['video_frames']==401 and result['steps']==400
            red=objects[:,0];green=objects[:,1];tcp=state[:,:3]
            close=np.flatnonzero((command[:,-1]<0)&(state[:-1,-1]>0))
            first=int(close[0]) if len(close) else None
            height=red[:,2]-red[0,2]
            metrics=[json.loads(x)['metrics'] for x in (d/'telemetry.jsonl').read_text().splitlines()]
            assert len(metrics)==401
            grasp=np.array([x['is_cubeA_grasped'] for x in metrics],bool)
            reset_grasp=bool(grasp[0])
            raw_grasp_frames=int(grasp.sum())
            # CPU PhysX reset can retain the preceding episode's contact cache.
            # No dynamics have occurred at row zero; retain it only as raw audit data.
            grasp[0]=False
            grasp_steps=np.flatnonzero(grasp)
            lift_steps=np.flatnonzero((height>.02)&grasp)
            release=np.flatnonzero((command[:,-1]>0)&(state[:-1,-1]<0))
            release_after_grasp=release[release>=grasp_steps[0]] if len(grasp_steps) else np.array([],dtype=int)
            rows.append(dict(seed=result['seed'],folder=d.name,success=bool(result['success']),
                success_at_end=bool(result['success_at_end']),first_success_step=result['first_success_step'],
                first_close=first,first_close_xy_cm=float(np.linalg.norm(tcp[first,:2]-red[first,:2])*100) if first is not None else None,
                min_xy_cm=float(np.linalg.norm(tcp[:,:2]-red[:,:2],axis=-1).min()*100),
                max_red_lift_cm=float(height.max()*100),grasped=bool(grasp.any()),
                grasp_frames=int(grasp.sum()),
                raw_reset_grasp_flag=reset_grasp,raw_grasp_frames=raw_grasp_frames,
                first_grasp_step=int(grasp_steps[0]) if len(grasp_steps) else None,
                first_grasped_lift_step=int(lift_steps[0]) if len(lift_steps) else None,
                first_release_after_grasp_step=int(release_after_grasp[0]) if len(release_after_grasp) else None,
                min_red_green_xy_while_lifted_cm=float(np.linalg.norm(red[lift_steps,:2]-green[lift_steps,:2],axis=-1).min()*100) if len(lift_steps) else None,
                first_close_tcp_minus_red_xyz_cm=((tcp[first]-red[first])*100).tolist() if first is not None else None,
                ik_fallback_steps=sum(bool(x.get('telemetry_ik_fallback',False)) for x in metrics[1:]),
                ik_diagnostic_steps=sum('telemetry_ik_fallback' in x for x in metrics[1:]),
                grasped_lift_above_2cm=bool(((height>.02)&grasp).any()),
                initial_red_xyz=red[0].tolist(),initial_green_xyz=green[0].tolist(),
                first8_mean_xyz=command[:8,:3].mean(0).tolist(),steps=400,observations=401))
    if expected_count is not None:assert len(rows)==expected_count
    return rows


def episode_summary(rows):
    closed=[x['first_close_xy_cm'] for x in rows if x['first_close_xy_cm'] is not None]
    return dict(episodes=len(rows),successes=sum(x['success'] for x in rows),
        successes_at_end=sum(x['success_at_end'] for x in rows),
        ik_fallback_steps=sum(x['ik_fallback_steps'] for x in rows),
        ik_diagnostic_steps=sum(x['ik_diagnostic_steps'] for x in rows),
        closes=len(closed),grasp_episodes=sum(x['grasped'] for x in rows),
        reset_grasp_flags=sum(x['raw_reset_grasp_flag'] for x in rows),
        grasped_lifts=sum(x['grasped_lift_above_2cm'] for x in rows),
        median_first_close_xy_cm=float(np.median(closed)) if closed else None,
        median_min_xy_cm=float(np.median([x['min_xy_cm'] for x in rows])),
        median_max_red_lift_cm=float(np.median([x['max_red_lift_cm'] for x in rows])))


def paired_outcomes(control,candidate):
    result={}
    for key in ('grasped','grasped_lift_above_2cm','success','success_at_end'):
        improved=[b['seed'] for a,b in zip(control,candidate) if not a[key] and b[key]]
        regressed=[b['seed'] for a,b in zip(control,candidate) if a[key] and not b[key]]
        discordant=len(improved)+len(regressed)
        p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(len(improved),len(regressed))+1))/2**discordant) if discordant else 1.
        result[key]=dict(improved_seeds=improved,regressed_seeds=regressed,
            exact_paired_two_sided_p=p,note='Exploratory exact paired sign/McNemar test, unadjusted for multiple outcomes; one training seed.')
    return result


def grounding_figure(control,candidate,destination,*,rgb_source=None,title='Fixed observation'):
    """Circles versus crosses make the image-coordinate error directly visible."""
    with np.load(rgb_source or control,allow_pickle=False) as z:rgb=z['rgb']
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for col,(arm,path) in enumerate((('Control',control),('Repair candidate',candidate))):
        measured=grounding(path)
        with np.load(path,allow_pickle=False) as z:density=z['g_density'].astype(float);masks=z['masks']
        for camera in range(2):
            ax=axes[camera,col];ax.imshow(rgb[camera]);errors=[]
            for obj,(name,color) in enumerate((('red','#ff3030'),('green','#2bff52'))):
                selected=[v for v in measured['views'] if v['object']==name and v['camera']==camera]
                if not selected:continue
                value=selected[0];mass=density[value['slot'],camera];total=mass.sum()
                if total<=1e-30:raise ValueError('Cannot display an underflowed conditional centroid')
                yy,xx=np.indices(mass.shape)
                predicted=np.array([(xx*mass).sum(),(yy*mass).sum()])/total
                y,x=np.nonzero(masks[camera,obj]);actual=np.array([x.mean(),y.mean()])
                ax.plot([actual[0],predicted[0]],[actual[1],predicted[1]],color=color,linewidth=1.5)
                ax.scatter(*actual,s=135,marker='o',facecolors='none',edgecolors='black',linewidth=4)
                ax.scatter(*actual,s=135,marker='o',facecolors='none',edgecolors=color,linewidth=2)
                ax.scatter(*predicted,s=130,marker='x',color='black',linewidth=5)
                ax.scatter(*predicted,s=130,marker='x',color=color,linewidth=2.5)
                errors.append(name+' '+str(round(value['centroid_error_px']))+' px')
            ax.set_title(arm+' · '+('global camera' if camera==0 else 'wrist camera')+'\n'+', '.join(errors),fontsize=12)
            ax.axis('off')
    fig.suptitle(title+'\nCircle = cube centre; cross = G read centroid',fontsize=15)
    fig.text(.5,.015,'Same RGB in both columns. Slots are matched by the evaluator; these are image reads, not 3D pose outputs.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.94));fig.savefig(destination,dpi=140);plt.close(fig)


def trajectory_figure(folders,records,destination):
    """Show the entire 400-action execution, including the failed late stages."""
    fig,axes=plt.subplots(4,3,figsize=(15,12),sharex='row')
    xy_limits=[];action_limit=.1
    for col,(arm,folder) in enumerate(folders):
        record=records[arm]
        with np.load(folder/record['folder']/'trajectory.npz',allow_pickle=False) as z:
            states=z['robot_obs_trajectory'];objects=z['object_positions'];actions=z['executed']
        telemetry=[json.loads(x)['metrics'] for x in (folder/record['folder']/'telemetry.jsonl').read_text().splitlines()]
        action_limit=max(action_limit,float(np.abs(actions[:,:3]).max())*1.08)
        grasp=np.array([x['is_cubeA_grasped'] for x in telemetry],bool)
        grasp[0]=False
        ax=axes[0,col]
        xy=states[:,:2]*100
        ax.plot(xy[:,0],xy[:,1],color='#2774bd',label='TCP path')
        ax.scatter(*xy[0],marker='o',color='#2774bd',label='TCP reset')
        ax.scatter(*xy[-1],marker='x',color='#2774bd',label='TCP end')
        for obj,color,name in ((0,'#d94c46','Red cube'),(1,'#279752','Green cube')):
            path=objects[:,obj,:2]*100
            ax.plot(path[:,0],path[:,1],color=color,alpha=.7)
            ax.scatter(*path[0],marker='s',s=65,color=color,label=name+' reset')
        first=record['first_close']
        if first is not None:
            ax.plot([xy[first,0],objects[first,0,0]*100],[xy[first,1],objects[first,0,1]*100],'--',color='#702a9c',label='First close miss')
            ax.scatter(*xy[first],color='#702a9c',s=35)
        xy_limits.append(np.concatenate([xy,objects[:,:2,:2].reshape(-1,2)*100]))
        ax.set(title=arm+' | '+('stack success' if record['success'] else 'no stack'),xlabel='World X (cm)',ylabel='World Y (cm)',aspect='equal')
        if col==0:ax.legend(fontsize=7,loc='best')
        ax=axes[1,col]
        ax.plot(np.arange(401),(objects[:,0,2]-objects[0,0,2])*100,color='#d94c46',label='Red cube lift')
        ax.fill_between(np.arange(401),0,1,where=grasp,transform=ax.get_xaxis_transform(),color='#2daa75',alpha=.2,label='Native grasp true')
        ax.axhline(2,color='#777',linestyle=':',label='2 cm lift')
        if first is not None:ax.axvline(first,color='#702a9c',linestyle='--',label='First close')
        ax.set(xlabel='Executed action index',ylabel='Lift above reset (cm)')
        if col==0:ax.legend(fontsize=7,loc='best')
        ax=axes[2,col]
        for d,color,name in ((0,'#2879bb','X'),(1,'#dd8c24','Y'),(2,'#6a3d9a','Z')):
            ax.plot(np.arange(400),actions[:,d],color=color,label=name,linewidth=.7)
        ax.set(xlabel='Executed action index',ylabel='Executed native XYZ command',ylim=(-1.05,1.05))
        if col==0:ax.legend(fontsize=7,ncol=3)
        ax=axes[3,col]
        ax.step(np.arange(400),actions[:,-1],where='post',color='#702a9c',label='Applied gripper command')
        ax.fill_between(np.arange(401),-1.1,1.1,where=grasp,color='#2daa75',alpha=.2,label='Native grasp true')
        ax.set(xlabel='Executed action index',ylabel='Gripper: +1 open / -1 close',ylim=(-1.15,1.15))
        if col==0:ax.legend(fontsize=7,loc='best')
    all_xy=np.concatenate(xy_limits)
    lo=all_xy.min(0)-3;hi=all_xy.max(0)+3
    for ax in axes[0]:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    for ax in axes[2]:ax.set_ylim(-action_limit,action_limit)
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.suptitle('Seed '+str(records['control']['seed'])+' — complete matched executions',fontsize=16)
    fig.tight_layout();fig.savefig(destination,dpi=135);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/data/senwang/data/clearvla_sim/stackcube_bc_20260910_v2'))
    args=p.parse_args();r=args.root;out=r/'report';out.mkdir(exist_ok=True)
    normalizer_file=out/'action_normalizer.json'
    if normalizer_file.exists():normalizer=load(normalizer_file)
    else:
        import torch
        normalizers=[]
        for arm in ('control','candidate'):
            payload=torch.load(r/(arm+'-pilot/checkpoints/latest.pt'),map_location='cpu',weights_only=False,mmap=True)
            normalizers.append(payload['data_state']['action_normalizer'])
            del payload
        assert normalizers[0]==normalizers[1]
        normalizer=normalizers[0];normalizer_file.write_text(json.dumps(normalizer,indent=2)+'\n')
    scale=np.array(normalizer['scale'],np.float32);offset=np.array(normalizer['offset'],np.float32)
    target_file=out/'teacher_action_targets.npz'
    if target_file.exists():
        with np.load(target_file,allow_pickle=False) as z:teacher_targets={k:z[k] for k in z.files}
    else:
        teacher_targets={}
        for case in load(r/'control-heldout/summary.json')['rows']:
            with h5py.File(args.data/'experts'/(case['episode']+'.hdf5')) as source:
                teacher_targets[case['label']]=source['action'][case['step']:case['step']+8]
        np.savez_compressed(target_file,**teacher_targets)
    summary=dict(note='One predeclared 256-update pilot per arm. Grounding is evaluated with exact renderer masks, not training color masks. Slot assignment is an evaluator permutation, not a deployed identity guarantee. Partial validation covers 128 windows; 14 held-out episodes receive seven-stage spatial probes.',training={},grounding={},placements={},closed_loop={})
    trajectories={};ground_rows={};ground_cases={}
    source_rows={}
    for arm in ('control','candidate'):
        rows=[json.loads(x) for x in (r/(arm+'-pilot/metrics.jsonl')).read_text().splitlines()]
        epoch=[x for x in rows if x.get('kind')=='epoch'][-1]
        assert epoch['step']==6648
        train_keys=['loss_total','loss_group_action','loss_action_flow','loss_contrib_spatial_grounding','loss_spatial_grounding','loss_spatial_region_nll','loss_spatial_centroid_error_sq','loss_ledger_gap','gradient_epoch_preclip_l2_mean','runtime_seconds_per_batch']
        val_keys=['validation_action_rmse_normalized','validation_first_rmse_normalized','validation_first8_rmse_normalized','validation_tail_rmse_normalized',
            'validation_action_rmse_source_native','validation_arm_rmse_source_native','validation_gripper_rmse_source_native','validation_first_rmse_source_native',
            'validation_gripper_command_accuracy','validation_decoded_gripper_close_precision','validation_decoded_gripper_close_recall',
            'validation_decoded_gripper_close_f1','validation_decoded_gripper_events_predicted','validation_decoded_gripper_events_target',
            'validation_decoded_gripper_event_ratio','loss_spatial_grounding']
        summary['training'][arm]={'step':epoch['step'],'train':{k:epoch['train'].get(k) for k in train_keys},'validation':{k:epoch['validation'].get(k) for k in val_keys}}
        spikes=[x['gradient_global_preclip_l2'] for x in rows if x.get('kind')=='gradient_spike']
        logged=[v for x in rows if x.get('kind')=='train' for v in x['metrics'].values() if isinstance(v,(float,int))]
        summary['training'][arm]['health']=dict(gradient_spikes_above_5=len(spikes),max_spike_before_clipping=max(spikes) if spikes else None,
            nonfinite_logged_train_scalars=sum(not np.isfinite(v) for v in logged),completed_updates=256,
            logged_learning_rate_range=[min(x['metrics']['learning_rate'] for x in rows if x.get('kind')=='train'),
                max(x['metrics']['learning_rate'] for x in rows if x.get('kind')=='train')])
        rows=[];cases=[]
        source_rows[arm]=load(r/(arm+'-heldout/summary.json'))['rows']
        assert len(source_rows[arm])==98
        for case in source_rows[arm]:
            value=grounding(r/(arm+'-heldout')/(case['label']+'.npz'))
            with np.load(r/(arm+'-heldout')/(case['label']+'.npz'),allow_pickle=False) as z:
                prediction=z['action'][:8]
                coarse=(z['coarse_action'][0,:8]-offset)/scale
                visible=z['masks'].sum((-2,-1))>=20
            target=teacher_targets[case['label']]
            assert prediction.shape==target.shape==(8,7)
            error=prediction-target
            np.testing.assert_allclose(np.sqrt(np.mean(error[:,:6]**2)),case['arm_rmse_first8'],rtol=1e-5,atol=1e-7)
            value.update(translation_rmse_first8=float(np.sqrt(np.mean(error[:,:3]**2))),
                red_visibility=('global_present' if visible[0,0] else 'wrist_only' if visible[1,0] else 'unobserved'),
                translation_mse_by_axis_first8=np.mean(error[:,:3]**2,axis=0).tolist(),
                rotation_rmse_first8=float(np.sqrt(np.mean(error[:,3:6]**2))),
                coarse_translation_rmse_first8=float(np.sqrt(np.mean((coarse[:,:3]-target[:,:3])**2))),
                coarse_rotation_rmse_first8=float(np.sqrt(np.mean((coarse[:,3:6]-target[:,3:6])**2))),
                gripper_sign_accuracy_first8=float(np.mean((prediction[:,-1]>0)==(target[:,-1]>0))))
            for view in value['views']:
                rows.append(dict(label=case['label'],split=case['split'],episode=case['episode'],stage=case['stage'],**view))
            cases.append(dict(label=case['label'],split=case['split'],episode=case['episode'],
                stage=case['stage'],arm_rmse_first8=case['arm_rmse_first8'],
                **{k:v for k,v in value.items() if k!='views'}))
        ground_rows[arm]=rows
        ground_cases[arm]=cases
        summary['grounding'][arm]={}
        for split in ('val','test'):
            selected=[x for x in rows if x['split']==split]
            summary['grounding'][arm][split]=dict(views=len(selected),episodes=len(set(x['episode'] for x in selected)),
                mean_conditional_region_mass=float(np.mean([x['conditional_region_mass'] for x in selected])),
                mean_joint_region_mass=float(np.mean([x['joint_region_mass'] for x in selected])),
                mean_centroid_error_px=float(np.mean([x['centroid_error_px'] for x in selected])))
        summary['placements'][arm]=[]
        for x in load(r/(arm+'-placements/summary.json'))['rows']:
            row={k:x[k] for k in ('label','xy','moved','first8_mean_xyz','tcp_xyz')}
            toward=np.array(x['positions'][0][:2])-np.array(x['tcp_xyz'][:2])
            row['first8_mean_xy_projection_toward_red']=float(np.dot(x['first8_mean_xyz'][:2],toward)/max(np.linalg.norm(toward),1e-12))
            summary['placements'][arm].append(row)
        trajectories[arm]=episodes(r/(arm+'-eval18'))
        summary['closed_loop'][arm]=episode_summary(trajectories[arm])
    # Frozen expert observations, evaluator labels and sampling noise must agree.
    for a,b in zip(source_rows['control'],source_rows['candidate']):
        assert all(a[k]==b[k] for k in ('label','split','episode','stage','step'))
        with np.load(r/'control-heldout'/(a['label']+'.npz'),allow_pickle=False) as za, np.load(r/'candidate-heldout'/(b['label']+'.npz'),allow_pickle=False) as zb:
            for key in ('rgb','masks','object_positions','robot_state','initial_noise'):
                np.testing.assert_array_equal(za[key],zb[key],err_msg=a['label']+'/'+key)
    summary['heldout_pairing']={'cases':98,'same_rgb_state_masks_poses_sampling_noise':True}
    summary['placement_finite_differences']={arm:placement_slopes(summary['placements'][arm]) for arm in ('control','candidate')}
    summary['grounding_paired_intervals']={}
    summary['teacher_forced_by_stage']={}
    summary['teacher_forced_actions']={}
    summary['teacher_forced_visibility']={}
    for split in ('val','test'):
        a=[x for x in ground_rows['control'] if x['split']==split]
        b=[x for x in ground_rows['candidate'] if x['split']==split]
        assert [(x['label'],x['object'],x['camera']) for x in a]==[(x['label'],x['object'],x['camera']) for x in b]
        summary['grounding_paired_intervals'][split]={key:paired_episode_interval(a,b,key) for key in ('conditional_region_mass','centroid_error_px')}
        summary['teacher_forced_by_stage'][split]={}
        summary['teacher_forced_actions'][split]={}
        summary['teacher_forced_visibility'][split]={}
        for arm in ('control','candidate'):
            selected=[x for x in ground_cases[arm] if x['split']==split]
            summary['teacher_forced_actions'][split][arm]=dict(queries=len(selected),
                mean_query_translation_rmse_first8=float(np.mean([x['translation_rmse_first8'] for x in selected])),
                translation_rmse_by_axis_first8=np.sqrt(np.mean([x['translation_mse_by_axis_first8'] for x in selected],axis=0)).tolist())
            summary['teacher_forced_visibility'][split][arm]={}
            for visibility in ('global_present','wrist_only','unobserved'):
                bucket=[x for x in selected if x['red_visibility']==visibility]
                summary['teacher_forced_visibility'][split][arm][visibility]=dict(queries=len(bucket),
                    mean_query_translation_rmse_first8=float(np.mean([x['translation_rmse_first8'] for x in bucket])) if bucket else None,
                    note='Descriptive visibility subgroup; task phase and scene are confounders, not a visibility intervention.')
            summary['teacher_forced_by_stage'][split][arm]={}
            for stage in ('reset','approach','preclose','close','lift','transport','release'):
                c=[x for x in ground_cases[arm] if x['split']==split and x['stage']==stage]
                assert len(c)==7
                keys=('arm_rmse_first8','translation_rmse_first8','rotation_rmse_first8','gripper_sign_accuracy_first8',
                    'coarse_translation_rmse_first8','coarse_rotation_rmse_first8',
                    'binding_composed_red_region_mass','binding_composed_green_region_mass','binding_to_matched_red','binding_to_matched_green','binding_null_mass')
                summary['teacher_forced_by_stage'][split][arm][stage]={key:float(np.mean([x[key] for x in c])) for key in keys}
        summary['teacher_forced_actions'][split]['paired_interval']=paired_episode_interval(
            [x for x in ground_cases['control'] if x['split']==split],
            [x for x in ground_cases['candidate'] if x['split']==split],'translation_rmse_first8')
    # The paired episode layouts must be identical across arms and baseline.
    reference=r/'reference-baseline-eval18'
    if not reference.exists():
        reference.mkdir()
        for d in sorted(args.baseline.glob('episode_*')):
            dest=reference/d.name;dest.mkdir()
            for f in d.iterdir():
                if f.is_file() and f.suffix in ('.npz','.mp4','.json','.jsonl'):shutil.copy2(f,dest/f.name)
    trajectories['original']=episodes(reference)
    summary['closed_loop']['original']=episode_summary(trajectories['original'])
    for i in range(18):
        for arm in ('control','candidate'):
            assert trajectories[arm][i]['seed']==trajectories['original'][i]['seed']
            for key in ('initial_red_xyz','initial_green_xyz'):
                np.testing.assert_array_equal(trajectories[arm][i][key],trajectories['original'][i][key])
    summary['per_episode']=trajectories
    summary['paired_closed_loop_outcomes']=paired_outcomes(trajectories['control'],trajectories['candidate'])
    binding_rows=[]
    binding_path=out/'binding-counterfactual-r2/summary.json'
    if binding_path.exists():
        evidence=load(binding_path);assert len(evidence['rows'])==42
        summary['binding_counterfactual']={'note':evidence['note'],'admission':evidence['admission'],'arms':{}}
        for arm in ('control','candidate'):
            selected=[x for x in evidence['rows'] if x['arm']==arm]
            by_mode={mode:[dict(episode=x['episode'],error=x['scores'][mode]['translation_rmse']) for x in selected] for mode in ('none','red','green')}
            summary['binding_counterfactual']['arms'][arm]=dict(
                noop_action_delta_rmse_max=max(x['scores']['none']['action_delta_rmse'] for x in selected),
                translation_rmse={mode:float(np.mean([x['error'] for x in values])) for mode,values in by_mode.items()},
                red_minus_noop=paired_episode_interval(by_mode['none'],by_mode['red'],'error'),
                green_minus_noop=paired_episode_interval(by_mode['none'],by_mode['green'],'error'))
            for stage in ('reset','preclose','lift'):
                chosen=[x for x in selected if x['stage']==stage]
                binding_rows.append((arm,stage,*[f"{np.mean([x['scores'][m]['translation_rmse'] for x in chosen]):.5f}" for m in ('none','red','green')]))
    failed_rows=[]
    failed_folder=out/'failed-state-probe'
    if failed_folder.exists():
        summary['failed_state_grounding']=failed_state_comparison(failed_folder)
        for arm in ('control','candidate'):
            for stage,value in summary['failed_state_grounding']['per_stage'][arm].items():
                failed_rows.append((arm,stage,value['visible_object_views'],
                    f"{100*value['mean_conditional_region_mass']:.2f}%",f"{value['mean_centroid_error_px']:.2f}"))
    readout_rows=[]
    readout_folder=out/'frozen-readouts'
    if readout_folder.exists():
        assert load(readout_folder/'status.json')['status']=='complete','Frozen readouts still incomplete'
        a=load(readout_folder/'control/readouts.json');b=load(readout_folder/'candidate/readouts.json')
        assert a['episodes']==b['episodes'] and a['splits']==b['splits']
        with np.load(readout_folder/'control/frozen_features.npz',allow_pickle=False) as za,np.load(readout_folder/'candidate/frozen_features.npz',allow_pickle=False) as zb:
            for key in ('targets','names','splits','state_only','dino_global','dino_spatial'):
                np.testing.assert_array_equal(za[key],zb[key],err_msg='Frozen readout pairing/'+key)
        summary['frozen_position_readouts']={arm:load(readout_folder/arm/'readouts.json') for arm in ('control','candidate')}
        for feature in ('state_only','dino_global','dino_spatial','g_content','g_geometry','g_all','p1_detail','constant_train_mean'):
            ac=a['rows'][feature]['test_mean_object_xy_error_cm'];bc=b['rows'][feature]['test_mean_object_xy_error_cm']
            readout_rows.append((feature,f'{ac[0]:.2f} / {ac[1]:.2f}',f'{bc[0]:.2f} / {bc[1]:.2f}'))
    p1_rows=[]
    p1_path=out/'p1-consumer-region-readback.json'
    if p1_path.exists():
        p1=load(p1_path)
        summary['p1_image_read_audit']={arm:p1[arm]['summary'] for arm in ('control','candidate')}
        for split in ('val','test'):
            for stage in ('reset','approach','preclose','close','lift','transport','release'):
                ac=p1['control']['summary'][split][stage];bc=p1['candidate']['summary'][split][stage]
                p1_rows.append((split,stage,f"{100*ac['red_joint']:.2f}% / {100*bc['red_joint']:.2f}%",
                    f"{100*ac['green_joint']:.2f}% / {100*bc['green_joint']:.2f}%"))
    (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'grounding_per_view.json').write_text(json.dumps(ground_rows,indent=2,allow_nan=False)+'\n')
    (out/'grounding_per_query.json').write_text(json.dumps(ground_cases,indent=2,allow_nan=False)+'\n')
    archive=out/'all_54_full_trajectories_npz.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for arm,folder in [('original',reference),('control',r/'control-eval18'),('candidate',r/'candidate-eval18')]:
            for f in sorted(folder.rglob('trajectory.npz')):z.write(f,arm+'/'+f.parent.name+'/'+f.name)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    colors={'control':'#62748a','candidate':'#218d75'}
    for i,arm in enumerate(('control','candidate')):
        x=np.arange(2)+(i-.5)*.3
        ax[0].bar(x,[100*summary['grounding'][arm][s]['mean_conditional_region_mass'] for s in ('val','test')],.3,color=colors[arm],label=arm)
        ax[1].bar(x,[summary['grounding'][arm][s]['mean_centroid_error_px'] for s in ('val','test')],.3,color=colors[arm])
        ax[2].bar(np.arange(3)+(i-.5)*.3,[summary['closed_loop'][arm][k] for k in ('grasp_episodes','grasped_lifts','successes')],.3,color=colors[arm])
    ax[0].set(title='Cube-region read mass',ylabel='Mean camera-conditional mass (%)',xticks=[0,1],xticklabels=['Validation','Test'])
    ax[1].set(title='Cube centroid error',ylabel='Mean distance (pixels)',xticks=[0,1],xticklabels=['Validation','Test'])
    ax[2].set(title='Full closed-loop outcomes',ylabel='Episodes out of 18',xticks=[0,1,2],xticklabels=['Grasp','Grasp + lift','Stack'])
    for a in ax:a.grid(axis='y',alpha=.2)
    ax[0].legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    ground_table=[]
    for arm in ('control','candidate'):
        for split in ('val','test'):
            v=summary['grounding'][arm][split]
            ground_table.append((arm,split,v['episodes'],v['views'],f"{100*v['mean_conditional_region_mass']:.2f}%",f"{v['mean_centroid_error_px']:.2f}"))
    behavior=[]
    for arm in ('original','control','candidate'):
        v=summary['closed_loop'][arm]
        behavior.append((arm,str(v['successes'])+' / '+str(v['successes_at_end']),v['grasp_episodes'],v['grasped_lifts'],v['closes'],v['median_first_close_xy_cm']))
    stage_rows=[]
    for split in ('val','test'):
        for stage in ('reset','approach','preclose','close','lift','transport','release'):
            a=summary['teacher_forced_by_stage'][split]['control'][stage]
            b=summary['teacher_forced_by_stage'][split]['candidate'][stage]
            stage_rows.append((split,stage,f"{a['translation_rmse_first8']:.4f} / {b['translation_rmse_first8']:.4f}",
                f"{a['coarse_translation_rmse_first8']:.4f} / {b['coarse_translation_rmse_first8']:.4f}",
                f"{a['rotation_rmse_first8']:.4f} / {b['rotation_rmse_first8']:.4f}",
                f"{a['gripper_sign_accuracy_first8']:.3f} / {b['gripper_sign_accuracy_first8']:.3f}",
                f"{a['binding_to_matched_red']:.3f} / {b['binding_to_matched_red']:.3f}",
                f"{a['binding_composed_red_region_mass']:.3f} / {b['binding_composed_red_region_mass']:.3f}"))
    placement_rows=[]
    for a,b in zip(summary['placements']['control'],summary['placements']['candidate']):
        assert a['label']==b['label']
        placement_rows.append((a['label'],f"{a['first8_mean_xyz'][0]:+.4f}",f"{b['first8_mean_xyz'][0]:+.4f}",
            f"{a['first8_mean_xy_projection_toward_red']:+.4f}",f"{b['first8_mean_xy_projection_toward_red']:+.4f}"))
    placement_figure(r,out/'all_12_placement_directions.png')
    grounding_images=[]
    for case in source_rows['control']:
        if case['stage']!='reset':continue
        filename='grounding_'+case['label']+'.png'
        grounding_figure(r/'control-heldout'/(case['label']+'.npz'),r/'candidate-heldout'/(case['label']+'.npz'),out/filename,
            title=case['split']+' '+case['episode']+' · reset')
        grounding_images.append('<details><summary>'+case['split']+' '+case['episode']+'</summary><img loading="lazy" src="'+filename+'"></details>')
    cases=[]
    for i in range(18):
        figure='seed_'+str(trajectories['control'][i]['seed'])+'_full.png'
        trajectory_figure([('original',reference),('control',r/'control-eval18'),('candidate',r/'candidate-eval18')],
            {arm:trajectories[arm][i] for arm in ('original','control','candidate')},out/figure)
        links=[]
        for arm,folder in [('original','reference-baseline-eval18'),('control','control-eval18'),('candidate','candidate-eval18')]:
            e=trajectories[arm][i];prefix='../'+folder+'/'+e['folder']+'/'
            links.append(f"<b>{arm}</b>: success={e['success']}, success at end={e['success_at_end']}, grasp={e['grasped']}, close miss={e['first_close_xy_cm']} cm · <a href='{prefix}trajectory.npz'>full NPZ</a> · <a href='{prefix}video.mp4'>full video</a>")
        milestones=[]
        for arm in ('original','control','candidate'):
            e=trajectories[arm][i]
            milestones.append((arm,e['first_close'],e['first_grasp_step'],e['first_grasped_lift_step'],
                e['first_release_after_grasp_step'],e['min_red_green_xy_while_lifted_cm']))
        cases.append('<details><summary>Seed '+str(trajectories['control'][i]['seed'])+'</summary><p>'+'</p><p>'.join(links)+'</p>'+table(('Arm','First close','First native grasp','First grasped 2cm lift','First release after grasp','Closest red/green XY while lifted (cm)'),milestones)+'<p>None means the event never occurred. Close/release use pre-action indices; grasp/lift use observation indices.</p><img loading="lazy" src="'+figure+'"><p>Rows: complete TCP and cube XY paths; red-cube lift; all 400 applied XYZ commands; all 400 gripper commands. Green shading marks the native grasp flag.</p></details>')
    mechanism=all(summary['grounding']['candidate'][s]['mean_conditional_region_mass']>summary['grounding']['control'][s]['mean_conditional_region_mass'] and summary['grounding']['candidate'][s]['mean_centroid_error_px']<summary['grounding']['control'][s]['mean_centroid_error_px'] for s in ('val','test'))
    behavior_gain=summary['closed_loop']['candidate']['successes']>summary['closed_loop']['control']['successes']
    verdict=('Grounding improves on both held-out splits. ' if mechanism else 'The pilot does not qualify a consistent grounding improvement. ')+('The common panel also gains stack successes; fresh seeds are still required before promotion.' if behavior_gain else 'The common panel does not gain stack successes; the checkpoint is not a qualified policy repair.')
    action_change={s:100*(summary['teacher_forced_actions'][s]['candidate']['mean_query_translation_rmse_first8']/summary['teacher_forced_actions'][s]['control']['mean_query_translation_rmse_first8']-1) for s in ('val','test')}
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>ManiSkill spatial repair comparison</title><style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#17212b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #cdd5de;text-align:left}}th{{background:#edf2f7}}details{{padding:12px;border:1px solid #ccd4dc;margin:10px 0}}a{{color:#1468a0}}</style></head><body>
<h1>Does spatial supervision repair the policy?</h1><p><b>{verdict}</b></p>
<p>Both arms start from epoch 8 / step 6392, retain the same named Adam moments, use the same 256 updates and minibatch seed, and finish at step 6648. Candidate adds only a 0.01-weight current-region objective. Online graph, inputs, action chart and Q5/replan-8 execution are unchanged.</p>
<p>This is a low-rate continuation: the global clock is restored but the cosine schedule is rebuilt for the bounded run, with logged learning rate approximately 8.29e-6 to 8.00e-6 in both arms. It is not a full retraining or an exact restoration of the old scheduler.</p>
<p>Mean first-eight XYZ command RMSE changes by <b>{action_change['val']:+.2f}% on validation and {action_change['test']:+.2f}% on test</b> expert observations. Negative means improvement. Localization gains must not be confused with action generalization.</p>
<img src="comparison.png"><h2>Grounding against renderer labels</h2>{table(('Arm','Split','Episodes','Visible object/views','Region mass','Centroid error (px)'),ground_table)}
<p>All seven validation and seven test episodes are probed at seven stages. G slots are matched with a single evaluator permutation across cameras. Regions expand exact visible actor masks by 12 pixels. These measure read localization, not calibrated 3D poses or guaranteed semantic identity; camera-conditional and joint masses remain separate in the JSON.</p>
<details><summary>See the difference: all 14 held-out reset layouts</summary>{''.join(grounding_images)}</details>
<details><summary>Does the improvement survive the original policy's failed states?</summary>
{table(('Arm','Failed-policy stage','Visible object/views','Region mass','Centroid error (px)'),failed_rows)}
<p>Each checkpoint is queried at the same four predefined stages in all 18 original failed trajectories. The complete 400-command replay is checked against recorded 7-D policy state and both cube XYZ. Paired RGB, state, actor masks, object positions and sampler noise are identical. These are off-policy localization measurements; archived failed actions are not expert targets. <a href="failed-state-probe/control/summary.json">All control cases</a> · <a href="failed-state-probe/candidate/summary.json">All candidate cases</a>.</p></details>
<details><summary>Does localization reach target binding and actions? All seven expert stages</summary>
{table(('Split','Stage','Final XYZ RMSE (C / R)','Coarse-head XYZ RMSE (C / R)','Final rotation RMSE (C / R)','Gripper accuracy (C / R)','S mass on red slot (C / R)','S × G red-region mass (C / R)'),stage_rows)}
<p>RMSE uses the first eight native arm commands against expert commands, with the same observed expert history and sampling noise. The supervised S/coarse head is decoded with the checkpoint normalizer and is not a deployed alternative controller. This is not a closed-loop score. C / R means control / repair candidate. S mass uses the evaluator-matched red slot; it does not assert stable physical identity.</p></details>
<details><summary>Can a frozen representation reveal object positions?</summary>
{table(('Frozen feature','Control red / green XY error (cm)','Repair red / green XY error (cm)'),readout_rows)}
<p>Diagnostic ridge readouts fit only 59 training reset layouts; regularization is selected on seven validation layouts and errors reported on seven test layouts. The raw state and DINO baselines are checked identical across arms. World XY coordinates are labels for this offline probe only. A better linear readout does not establish that the deployed controller uses the information; a weak one does not prove information is absent.</p></details>
<details><summary>Where P1 actually reads current image details</summary>
{table(('Split','Stage','Red-region joint read mass (C / R)','Green-region joint read mass (C / R)'),p1_rows)}
<p>These values rasterize the actual nine-cell P1 microgrid reads and average their joint camera/image mass over reader queries. Regions use the same 12-pixel tolerance. This differs from matching G slots or multiplying S binding by G, and it is not a dedicated cube detector: P1 also reads other scene evidence. Geometry has a separate P2 owner, so weak position readout from P1 alone does not prove that the geometry path is absent. <a href="p1-consumer-region-readback.json">Every P1 read measurement</a>.</p></details>
<details><summary>Controlled object placements: every signed action response</summary>
<img loading="lazy" src="all_12_placement_directions.png">
{table(('Moved cube and XY metres','Control X command','Candidate X command','Control toward-red projection','Candidate toward-red projection'),placement_rows)}
<p>All values average the first eight native commands. Positive XY projection points toward the red cube; negative points away. This tests the immediate response to physical placement, not success or multi-step reachability.</p></details>
<details><summary>Evaluator-only target-binding interventions</summary>
{table(('Arm','Expert stage','Unchanged XYZ RMSE','Red-slot XYZ RMSE','Green-slot XYZ RMSE'),binding_rows)}
<p>On seven validation demonstrations, each existing shared binding is concentrated on the renderer-matched red or green slot while preserving its original real/null mass. G is numerically unchanged and every case has a no-op repeat. The slot remains a learned representation, not an oracle physical state. These fixed-observation probes are not deployment results and do not change either closed-loop policy. <a href="binding-counterfactual-r2/summary.json">Every probe and numerical admission</a>.</p></details>
<h2>Physical outcomes</h2>{table(('Arm','Stacks any / at end (of 18)','Grasps / 18','Grasp + 2cm lift / 18','Episodes with close','Median first-close XY miss (cm)'),behavior)}
<p>The close-error median includes only episodes that issue a close; compare its denominator and per-seed records. Grasp counts use post-action observations only: CPU PhysX can carry the previous episode's contact cache into reset row zero. Raw reset flags remain in the NPZ, telemetry and JSON audit; they do not count as achieved grasps. All episodes retain 400 actions and 401 observations. A better loss or heatmap alone cannot qualify the repair.</p>
<p><a href="comparison.json">Complete metrics, paired episode uncertainty and results</a> · <a href="grounding_per_view.json">Every grounding view</a> · <a href="grounding_per_query.json">Every binding/action query</a> · <a href="all_54_full_trajectories_npz.zip">All 54 full trajectory NPZ files</a> · <a href="../preflight-admission.json">Preflight and restored-output parity</a> · <a href="../label_audit.json">Training-label audit</a></p>
<h2>Every paired trajectory</h2>{''.join(cases)}
<p>Source and objective specification: <code>configs/mainline/maniskill_spatial_repair_experiment_20261008.json</code>. This is a bounded single-seed training pilot, not a statistically established deployment result.</p></body></html>'''
    (out/'index.html').write_text(body,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('training','grounding','closed_loop')},indent=2))
    print(verdict)


if __name__=='__main__':main()
