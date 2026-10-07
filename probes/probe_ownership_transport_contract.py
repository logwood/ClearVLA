"""Audit-only ownership/transport reference; no policy or optimizer is changed.

Synthetic examples test the proposed representation, not learned performance.
Factual panels compare three explicitly different laws under the SAME fixed
content-only seed discriminator and the actual G2 source measure.
"""
from pathlib import Path
import argparse, copy, hashlib, json, sys
import torch
import torch.nn.functional as F
from clearvla.vision.entity_chart import CurrentImageSupport, pushforward_to_current_image, pushforward_log_to_current_image
from clearvla.mainline.model.grounding import dense_chart_from_local_facts
from probe_reconstruction_joint_gradient import clone_tree


def rms(x): return float(x.square().mean().sqrt())
def sample(field, xy):
    # field [C,K,H,W], coordinates [C,A,2], full-RGB endpoint chart.
    return F.grid_sample(field,xy[:,:,None],align_corners=True,padding_mode='border')[:,:,:,0]


def splat(mass, xy, valid, side):
    # [C,K,A] -> [C,K,U]; retains bilinear zero-weight derivatives.
    xy=torch.where(valid[...,None],xy,0.).clamp(-1,1)
    mass=torch.where(valid[:,None],mass,0.)
    x,y=(xy[...,0]+1)*(side-1)/2,(xy[...,1]+1)*(side-1)/2
    x0,y0=x.floor(),y.floor();dx,dy=x-x0,y-y0
    output=mass.new_zeros(mass.shape[0],mass.shape[1],side*side)
    for ox,oy,w in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        index=(y0.long()+oy).clamp_max(side-1)*side+(x0.long()+ox).clamp_max(side-1)
        output=output.scatter_add(-1,index[:,None].expand_as(mass),mass*w[:,None])
    return output.reshape(*mass.shape[:2],side,side)


def dense_reference(mass,xy,side):
    # Independent dense tent-basis reference. Derivative comparisons exclude knots.
    u=torch.arange(side,dtype=xy.dtype,device=xy.device)
    x,y=(xy[...,0]+1)*(side-1)/2,(xy[...,1]+1)*(side-1)/2
    wx=(1-(x[...,None]-u).abs()).relu();wy=(1-(y[...,None]-u).abs()).relu()
    return torch.einsum('cka,cay,cax->ckyx',mass,wy,wx)


def compare_laws(mu,xy,valid,logits,values):
    side=logits.shape[-1];q=logits.softmax(1)
    base=splat(mu[:,None],xy,valid,side)
    canonical=base*q
    qa=sample(logits,xy).softmax(1)
    atom=splat(mu[:,None]*qa,xy,valid,side)
    qp=sample(q,xy)
    roundtrip=splat(mu[:,None]*qp,xy,valid,side)
    scale=base.sum().clamp_min(1e-30)
    def read(joint):
        return torch.einsum('ckyx,cdyx->ckd',joint,values)/joint.sum((-1,-2))[...,None].clamp_min(1e-30)
    canonical_value=read(canonical)
    sampled_values=sample(values,xy)
    atom_value=torch.einsum('cka,cda->ckd',mu[:,None]*qa,sampled_values)/(mu[:,None]*qa).sum(-1)[...,None].clamp_min(1e-30)
    real_difference=(canonical_value[:,:-1]-atom_value[:,:-1]).square().mean(-1)
    real_mass=canonical[:,:-1].sum((-1,-2))
    weighted_value_error=(real_difference*real_mass).sum()/real_mass.sum().clamp_min(1e-30)
    return {'canonical_vs_atom_joint_TV':float(.5*(canonical-atom).abs().sum()/scale),
        'canonical_vs_pullback_resplat_joint_TV':float(.5*(canonical-roundtrip).abs().sum()/scale),
        'interpolate_logits_vs_probabilities_mean_TV':float((.5*(qa-qp).abs().sum(1)*mu).sum()/scale),
        'canonical_vs_atom_view_value_rms':rms(canonical_value-atom_value),
        'real_K_view_value_rms':float(real_difference.mean().sqrt()),'mass_weighted_real_K_view_value_rms':float(weighted_value_error.sqrt()),
        'source_mass':float(scale),'canonical_mass_error':float((canonical.sum(1,keepdim=True)-base).abs().max()),
        'atom_mass_error':float((atom.sum(1,keepdim=True)-base).abs().max()),
        'canonical_joint':canonical,'atom_joint':atom}


def transport_gradient_panel():
    rows=[]
    for label,xy in [('interior',[[-.73,-.21],[.27,.61]]),('grid_knots',[[0.,0.],[.5,-.5]]),('edges',[[-1.,-1.],[1.,1.]])]:
        coords=torch.tensor(xy,dtype=torch.float32).reshape(1,1,1,1,1,2,2).requires_grad_()
        source_logp=torch.tensor([.2,-.3],requires_grad=True);p=source_logp.softmax(0).reshape(1,1,1,1,1,2)
        source_logmass=torch.tensor([-.2],requires_grad=True)
        support=CurrentImageSupport(coords,p,torch.ones_like(p,dtype=torch.bool),source_logp.log_softmax(0).reshape_as(p))
        mass=source_logmass.exp().reshape(1,1,1,1,1,1)
        linear=pushforward_to_current_image(mass,support,rows=5,columns=5)
        logged=pushforward_log_to_current_image(source_logmass.reshape_as(mass),torch.ones_like(mass,dtype=torch.bool),support,rows=5,columns=5)
        restored=torch.where(logged.supported,logged.log_mass.exp(),0.)
        weight=torch.arange(25,dtype=torch.float32).reshape_as(linear)/25
        gl=torch.autograd.grad((linear*weight).sum(),(coords,source_logp,source_logmass),retain_graph=True)
        gg=torch.autograd.grad((restored*weight).sum(),(coords,source_logp,source_logmass),retain_graph=True)
        # One-sided coordinate finite differences are meaningful at grid knots.
        finite=[]
        with torch.no_grad():
            base=float((restored*weight).sum())
            for i in range(coords.numel()):
                c=coords.clone();direction=-1. if float(c.flatten()[i])>=1 else 1.
                c.flatten()[i]+=direction*1e-4
                sp=CurrentImageSupport(c,p,support.valid,support.log_probability)
                changed=pushforward_log_to_current_image(source_logmass.reshape_as(mass),torch.ones_like(mass,dtype=torch.bool),sp,rows=5,columns=5)
                val=torch.where(changed.supported,changed.log_mass.exp(),0.)
                finite.append((float((val*weight).sum())-base)/(direction*1e-4))
        rows.append({'case':label,'forward_max_abs':float((linear-restored).abs().max()),
            'coordinate_linear_vjp':gl[0].flatten().tolist(),'coordinate_log_vjp':gg[0].flatten().tolist(),
            'coordinate_one_sided_finite_difference':finite,'probability_vjp_max_abs':float((gl[1]-gg[1]).abs().max()),'mass_vjp_max_abs':float((gl[2]-gg[2]).abs().max())})
    return rows


def reference_contracts():
    torch.manual_seed(8127);dtype=torch.float64
    xy=torch.tensor([[[-.73,-.21],[.27,.61],[.42,-.36]],[[-.51,.34],[.68,.13],[-.13,-.62]]],dtype=dtype,requires_grad=True)
    mu=torch.tensor([[.1,.3,.6],[.4,.25,.35]],dtype=dtype,requires_grad=True)
    field=torch.randn(2,5,5,5,dtype=dtype,requires_grad=True)
    valid=torch.ones(2,3,dtype=torch.bool)
    prod=splat(mu[:,None],xy,valid,5);ref=dense_reference(mu[:,None],xy,5)
    weight=torch.randn_like(prod)
    gp=torch.autograd.grad((prod*weight).sum(),(mu,xy),retain_graph=True)
    gr=torch.autograd.grad((ref*weight).sum(),(mu,xy))
    values=torch.randn(2,3,5,5,dtype=dtype)
    result=compare_laws(mu,xy,valid,field,values)
    owner_weight=torch.randn_like(result['canonical_joint'][:,:-1])
    source_vjps={}
    for name in ('canonical_joint','atom_joint'):
        real=result[name][:,:-1]
        owner=real/real.sum(1,keepdim=True).clamp_min(1e-30)
        grad=torch.autograd.grad((owner*owner_weight).sum(),mu,retain_graph=True)[0]
        source_vjps[name]=float(grad.norm())
    duplicate=compare_laws(mu.repeat_interleave(2,-1)/2,xy.repeat_interleave(2,1),valid.repeat_interleave(2,-1),field,values)
    perm=torch.tensor([2,0,3,1,4]);permuted=compare_laws(mu,xy,valid,field[:,perm],values)
    camera=compare_laws(mu.flip(0),xy.flip(0),valid.flip(0),field.flip(0),values.flip(0))
    evidence={k:v for k,v in result.items() if not isinstance(v,torch.Tensor)}
    evidence['nonoverlapping_assignment_source_mass_vjp_norm']=source_vjps
    overlap_mu=torch.tensor([[.3,.7]],dtype=dtype,requires_grad=True)
    overlap_xy=torch.tensor([[[-.4,-.2],[.3,.4]]],dtype=dtype)
    overlap=compare_laws(overlap_mu,overlap_xy,torch.ones(1,2,dtype=torch.bool),torch.randn(1,5,2,2,dtype=dtype),torch.randn(1,3,2,2,dtype=dtype))
    overlap_weight=torch.randn(1,4,2,2,dtype=dtype);overlap_vjp={}
    for name in ('canonical_joint','atom_joint'):
        real=overlap[name][:,:-1];owner=real/real.sum(1,keepdim=True).clamp_min(1e-30)
        grad=torch.autograd.grad((owner*overlap_weight).sum(),overlap_mu,retain_graph=True)[0]
        overlap_vjp[name]=float(grad.norm())
    evidence['overlapping_assignment_source_mass_vjp_norm']=overlap_vjp
    from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
    binder=TaskConditionedTargetBinder(16,4).double().eval()
    task=torch.randn(1,3,16,dtype=dtype);objects=torch.randn(1,4,16,dtype=dtype);history=torch.randn(1,16,dtype=dtype)
    before=binder(task,objects,torch.ones(1,4,dtype=torch.bool),history=history)
    padded=binder(torch.cat((task,torch.zeros(1,9,16,dtype=dtype)),1),objects,torch.ones(1,4,dtype=torch.bool),history=history)
    evidence['unmasked_full_token_extension_padding_demo']={'real_K_mass_max_difference':float((before.mass-padded.mass).abs().max()),'scope':'synthetic existing binder call; current protected four-token input is not padded'}
    evidence.update({'dense_reference_forward_error':float((prod-ref).abs().max()),
        'dense_reference_mass_vjp_error':float((gp[0]-gr[0]).abs().max()),'dense_reference_coordinate_vjp_error':float((gp[1]-gr[1]).abs().max()),
        'duplicate_split_atom_error':max(float((result[k]-duplicate[k]).abs().max()) for k in ('canonical_joint','atom_joint')),
        'K_permutation_error':max(float((result[k][:,perm]-permuted[k]).abs().max()) for k in ('canonical_joint','atom_joint')),
        'camera_tensor_permutation_error':max(float((result[k].flip(0)-camera[k]).abs().max()) for k in ('canonical_joint','atom_joint'))})
    masked_xy=xy.detach().clone();masked_xy[:]=float('nan')
    empty=splat(torch.full((2,5,3),float('nan'),dtype=dtype),masked_xy,torch.zeros_like(valid),5)
    evidence['empty_invalid_finite_zero']=bool(torch.isfinite(empty).all() and (empty==0).all())
    # Analytic loss of information in an image->atom marginal->image round trip.
    t=torch.tensor([.5,.5],dtype=dtype);q=torch.tensor([[.8,.1],[.1,.8],[.1,.1]],dtype=dtype)
    j=q*t;roundtrip=(j.sum(-1)[:,None])*t
    evidence['two_pixel_roundtrip_joint_TV']=float(.5*(j-roundtrip).abs().sum())
    # One local mixture, two differently colored atoms, constant 10% null.
    v=torch.tensor([0.,1.],dtype=dtype);qa=torch.tensor([[.81,.09],[.09,.81],[.1,.1]],dtype=dtype)
    new=(qa[:2]*t*v).sum(-1)/(qa[:2]*t).sum(-1)
    evidence['mixed_two_object_old_values']=[.5,.5];evidence['mixed_two_object_atom_values']=new.tolist()
    evidence['existing_log_transport_gradient_panel']=transport_gradient_panel()
    return evidence


def factual_audit(grounder,captured,native,masks,names):
    module=copy.deepcopy(grounder).float().eval()
    local=clone_tree(captured['local_facts']);chart=dense_chart_from_local_facts(local)
    spatial=local.current_image_support
    source_mass=(chart.candidate_owner_prior.float().reshape_as(chart.candidate_validity)*chart.candidate_validity.float()).squeeze(-1)
    mu=(source_mass[...,None]*spatial.probability)[0].reshape(2,-1)
    xy=spatial.coordinates[0].reshape(2,-1,2);valid=spatial.valid[0].reshape(2,-1)
    current=F.layer_norm(native.float(),(native.shape[-1],)).reshape(1,2,16,16,-1)
    # One fixed discriminator shared across all compared mathematical designs.
    keys=module.candidate_norm(module.content_key(current)).reshape(1,512,module.hidden)
    slots=module.slot_seed.expand(1,-1,-1)
    with torch.no_grad():
        out=module._competition(slots,keys,torch.ones(1,512,1,device=keys.device),torch.ones(1,512,1,device=keys.device),torch.zeros(1,512,1,device=keys.device),torch.ones(1,512,dtype=torch.bool,device=keys.device))
    logits=out[4][0].reshape(2,16,16,-1).permute(0,3,1,2)
    values=current[0].permute(0,3,1,2)
    result=compare_laws(mu,xy,valid,logits,values)
    result={k:v for k,v in result.items() if not isinstance(v,torch.Tensor)}
    prod=pushforward_to_current_image(source_mass[:,None],spatial,rows=16,columns=16)[0,0]
    reference=splat(mu[:,None],xy,valid,16)[:,0]
    result['existing_linear_transport_forward_max_abs']=float((prod-reference).abs().max())
    image_xy=(xy+1)*7.5
    near_knots=((image_xy-image_xy.round()).abs()<1e-6).any(-1)&valid
    edge=(xy.abs()>=1-1e-7).any(-1)&valid
    result['mass_at_any_coordinate_knot']=float((mu*near_knots).sum()/mu.sum())
    result['mass_at_image_edge']=float((mu*edge).sum()/mu.sum())
    result['scope']='same current full-RGB endpoint field and original G2 measure; content-only seed discriminator, not a repaired policy or identity-success test'
    result['full_rgb_chart_shape']=list(current.shape)
    return {'ownership_contract':result}




def factual_transport_gradients(source,address_call):
    c=address_call['locals'];sp=source.spatial;bank=c['bank']
    base=bank.coarse_candidate_coordinates.float()[:,:,None,None,None].expand_as(sp.coordinates)
    jac=(1-base.square()).clamp_min(0)
    rows=[]
    for name,position in [('current',sp.coordinates),('zero_correction_counterfactual',base)]:
        for side in (8,16):
            with torch.enable_grad():
                coord=position.detach().clone().float().requires_grad_()
                spatial=CurrentImageSupport(coord,sp.probability.detach(),sp.valid,sp.log_probability.detach())
                log=source.log_measure.detach();supported=source.supported
                mass=torch.where(supported,log.exp(),0.)
                linear=pushforward_to_current_image(mass,spatial,rows=side,columns=side)
                logged=pushforward_log_to_current_image(log,supported,spatial,rows=side,columns=side)
                linear=linear/linear.sum((2,3,4),keepdim=True).clamp_min(1e-30)
                probability,_=logged.normalized((2,3,4))
                axis=torch.linspace(-1,1,side,device=coord.device)
                yy,xx=torch.meshgrid(axis,axis,indexing='ij')
                weights=(.37*xx+.83*yy)[None,None,None]
                kweight=torch.tensor([1.,-.3,.7,-.2],device=coord.device)[None,:,None,None,None]
                weights=weights*kweight
                gl=torch.autograd.grad((linear*weights).sum(),coord,retain_graph=True)[0]
                gg=torch.autograd.grad((probability*weights).sum(),coord)[0]
                pl=(gl*jac).sum(-2);pg=(gg*jac).sum(-2)
                norms=(float(pl.norm()),float(pg.norm()))
                difference=float((pl-pg).norm()/pl.norm().clamp_min(1e-30))
                cosine=float(F.cosine_similarity(pl.flatten()[None],pg.flatten()[None]))
                owner=copy.deepcopy(c['self'].g2_typed_rectifier)
                if name=='zero_correction_counterfactual':
                    final=next(m for m in reversed(list(owner.modules())) if isinstance(m,torch.nn.Linear))
                    with torch.no_grad():
                        final.weight.zero_()
                        if final.bias is not None:final.bias.zero_()
                inp=c['rectifier_input'].detach().clone()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):rectifier=owner(inp).float()
                if name=='current' and rms(rectifier-c['rectifier'])>1e-5:
                    raise RuntimeError('rectifier replay is not exact enough to attribute parameter VJP')
                correction=.25*torch.tanh(rectifier[...,:2])*c['correction_scale'].detach().clone().float()
                parameters=tuple(owner.parameters())
                vl=torch.autograd.grad(correction,parameters,grad_outputs=pl,retain_graph=True)
                vg=torch.autograd.grad(correction,parameters,grad_outputs=pg)
                vl=torch.cat([x.flatten() for x in vl]);vg=torch.cat([x.flatten() for x in vg])
                parameter_panel={'owner':'g2_typed_rectifier, coordinate-transport branch only',
                    'relative_vjp_error':float((vl-vg).norm()/vl.norm().clamp_min(1e-30)),
                    'linear_vjp_norm':float(vl.norm()),'log_vjp_norm':float(vg.norm()),
                    'cosine':float(F.cosine_similarity(vl[None],vg[None])),
                    'rectifier_replay_rms':rms(rectifier-c['rectifier']) if name=='current' else None,
                    'counterfactual_rectifier_output_max':float(rectifier.abs().max()) if name!='current' else None}

            # Frozen read/mass field; only the actual G2 correction coordinate leg varies.
            direction=pl/pl.norm().clamp_min(1e-30)
            finite=[]
            with torch.no_grad():
                for sign in (-1.,1.):
                    moved=position+sign*1e-3*jac*direction[...,None,:]
                    moved_sp=CurrentImageSupport(moved,sp.probability,sp.valid,sp.log_probability)
                    v=pushforward_log_to_current_image(log,supported,moved_sp,rows=side,columns=side)
                    v,_=v.normalized((2,3,4));finite.append(float((v*weights).sum()))
            rows.append({'mode':name,'image_side':side,'coordinate_vjp_max_error':float((gl-gg).abs().max()),
                'G2_correction_vjp_relative_error':difference,'G2_correction_vjp_cosine':cosine,
                'parameter_owner_vjp':parameter_panel,
                'G2_correction_vjp_norm_linear':norms[0],'G2_correction_vjp_norm_log':norms[1],
                'finite_directional_derivative':(finite[1]-finite[0])/.002,
                'linear_directional_derivative':float((pl*direction).sum()),'log_directional_derivative':float((pg*direction).sum())})
    return rows


def main():
    if '--reference-only' in sys.argv:
        ap=argparse.ArgumentParser();ap.add_argument('--reference-only',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
        a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
        result=reference_contracts();(a.output/'reference.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2));return
    import probe_grounding_identity_provenance as driver
    captured_source={}
    def audit_and_keep(*args,**kwargs):
        captured_source['value']=clone_tree(args[1]['facts'].current_image_source)
        return factual_audit(*args,**kwargs)
    driver.audit=audit_and_keep
    original_upstream=driver.upstream_support
    def upstream_with_gradient(rows,masks,names):
        out=original_upstream(rows,masks,names)
        with torch.inference_mode(False),torch.no_grad():
            out.append({'transport_coordinate_gradient':factual_transport_gradients(captured_source['value'],rows[1])})
        return out
    driver.upstream_support=upstream_with_gradient
    driver.main()
    output=Path(sys.argv[sys.argv.index('--output')+1]);cp=output/'complete.json';complete=json.loads(cp.read_text())
    complete['dependency_driver_sha256']=complete.pop('script_sha256')
    complete['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    complete['driver_path']=driver.__file__;cp.write_text(json.dumps(complete,indent=2)+'\n')
    (output/'reference.json').write_text(json.dumps(reference_contracts(),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
