"""Read-only agreement labels from frozen RGB masks and sensor surface groups.

Same in both -> candidate positive; different in both -> candidate negative;
disagreement or absent support -> unknown. No oracle selects a point/pair.
Physical body masks score the proposed rule after construction.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np


def proposal_partition(masks, surface, *, interior_radius=0):
    if masks.shape[1:] != surface.shape:
        raise ValueError('native observation grids differ')
    legal = (masks.sum(0) == 1) & (surface >= 0)
    sam = masks.argmax(0) if len(masks) else np.zeros(surface.shape, np.int64)
    pairs = np.stack((sam[legal], surface[legal]), -1)
    values, inverse = np.unique(pairs, axis=0, return_inverse=True)
    partition = np.full(surface.shape, -1, np.int64)
    partition[legal] = inverse
    if interior_radius not in (0, 1):
        raise ValueError('undeclared native-pixel interior rule')
    if interior_radius:
        # Confidence comes from proposal boundaries, never from oracle masks.
        # Outside the image is unknown. Keep a point only when its complete
        # 3x3 neighborhood agrees on both independent proposal identities.
        padded = np.pad(partition, 1, constant_values=-1)
        keep = partition >= 0
        for dy in range(3):
            for dx in range(3):
                keep &= padded[dy:dy+surface.shape[0],dx:dx+surface.shape[1]] == partition
        partition[~keep] = -1
        alive, inverse = np.unique(partition[keep], return_inverse=True)
        values = values[alive]
        partition[keep] = inverse
    return partition, values


def surface_partition(surface, *, interior_radius=0):
    """Sensor-only control: no frozen mask, RGB class or oracle input."""
    ids=np.unique(surface[surface>=0])
    masks=np.stack([surface==i for i in ids]) if len(ids) else np.empty((0,*surface.shape),bool)
    return proposal_partition(masks,surface,interior_radius=interior_radius)


def audit(partition, groups, truth):
    counts = []
    for index in range(len(groups)):
        selected = partition == index
        body = (truth & selected[None]).sum((1, 2)).astype(np.int64)
        counts.append((body, int(selected.sum()-body.sum())))
    pos_total = pos_wrong = pos_background = 0
    neg_total = neg_wrong = unknown_total = 0
    for body, background in counts:
        n = int(body.sum())
        pos_total += n*(n-1)//2
        pos_wrong += (n*n-int(body@body))//2
        pos_background += n*background
    for i, j in itertools.combinations(range(len(groups)), 2):
        a, b = counts[i][0], counts[j][0]
        total = int(a.sum()*b.sum())
        if np.all(groups[i] != groups[j]):
            neg_total += total; neg_wrong += int(a@b)
        else:
            unknown_total += total
    visible = truth.sum((1, 2))
    covered = (truth & (partition >= 0)[None]).sum((1, 2))
    return dict(visible_block_pixels=visible.tolist(), supported_block_pixels=covered.tolist(),
        candidate_positive_block_pairs=pos_total, wrong_positive_block_pairs=pos_wrong,
        candidate_positive_block_background_pairs=pos_background,
        candidate_negative_block_pairs=neg_total, wrong_negative_same_block_pairs=neg_wrong,
        unknown_block_pairs=unknown_total,
        scope='Native pixel-pair counts are correlated. Robot/background identities, cross-view/time links and train-distribution coverage remain unqualified.')


def negative_clique_size(groups):
    # A pair is negative iff both labels differ. A maximum clique is therefore
    # a maximum matching between SAM IDs and surface IDs, without oracle input.
    neighbors = {}
    for a, b in groups:
        neighbors.setdefault(int(a), []).append(int(b))
    assigned = {}
    def augment(a, seen):
        for b in neighbors[a]:
            if b in seen:
                continue
            seen.add(b)
            if b not in assigned or augment(assigned[b], seen):
                assigned[b] = a
                return True
        return False
    return sum(augment(a, set()) for a in neighbors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sam', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--groups', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-all-bodies', action='store_true')
    p.add_argument('--interior-radius', type=int, choices=(0,1), default=0,
                   help='Optional fixed one-native-pixel confidence interior; oracle independent')
    p.add_argument('--proposal-mode', choices=('mask_surface_agreement_v1','surface_only_v3'),
                   default='mask_surface_agreement_v1')
    args = p.parse_args(); args.output.mkdir(exist_ok=False)
    sam = json.loads(args.sam.read_text()); assert sam['complete']
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != sam['identity']['plan_sha256']:
        raise ValueError('SAM plan identity changed')
    plans = {(r['case_id'], t):r for r in json.loads(args.plan.read_text()) for t in r['steps']}
    surface = {}
    for path in args.groups:
        value = json.loads(path.read_text()); assert value['complete']
        for r in value['records']:
            key = (r['case_id'], r['step'])
            if key in surface:
                raise ValueError('duplicate surface key; use separate physical audit sets')
            surface[key] = r
    result = dict(complete=False, records=[], production_changed=False, scope=__doc__,
                  interior_radius=args.interior_radius,
                  proposal_mode=args.proposal_mode,
                  training_use='unqualified negative proposals only; existing sensor positives unchanged')
    for row in sam['records']:
        key = (row['case_id'], row['step']); group = surface[key]; plan = plans[key]
        if Path(row['case']).resolve() != Path(plan['case']).resolve():
            raise ValueError('trajectory identity differs')
        for field, expected in ((row['proposal_path'],row['proposal_sha256']),
                                (group['groups_path'],group['groups_sha256'])):
            if hashlib.sha256(Path(field).read_bytes()).hexdigest() != expected:
                raise ValueError('proposal file identity changed')
        # Construct labels without accessing any physical identity mask.
        with np.load(group['groups_path'], allow_pickle=False) as b:
            if args.proposal_mode=='surface_only_v3':
                partitions=[surface_partition(b[key],interior_radius=args.interior_radius)
                            for key in ('group_static','group_gripper')]
            else:
                with np.load(row['proposal_path'], allow_pickle=False) as a:
                    partitions = [proposal_partition(a[name],b[other],interior_radius=args.interior_radius) for name,other in
                                  [('masks_top','group_static'),('masks_wrist','group_gripper')]]
        truth_path = Path(plan['masks'])/Path(plan['case']).name/('state_%03d.npz'%row['step'])
        if hashlib.sha256(truth_path.read_bytes()).hexdigest()!=row['audit_mask_sha256']:
            raise ValueError('scoring mask identity changed')
        with np.load(truth_path, allow_pickle=False) as z:
            scored = [audit(part, groups, z[name]) for (part,groups),name in
                      zip(partitions,('top_masks','wrist_masks'))]
        record = dict(case=row['case_id'], step=row['step'], cameras=scored,
                      negative_clique_sizes=[negative_clique_size(g) for _,g in partitions])
        if args.audit_all_bodies:
            # Reopen only AFTER proposal construction. These arrays are never
            # used for foreground selection, masks, grouping or matching.
            whole, rigid = [], []
            with np.load(group['groups_path'], allow_pickle=False) as z:
                for (part,groups), camera in zip(partitions,('static','gripper')):
                    body, link = z['audit_body_'+camera], z['audit_link_'+camera]
                    ids = np.unique(body[body>=0])
                    truth = np.stack([body==i for i in ids]) if len(ids) else np.empty((0,*body.shape),bool)
                    scored_body = {k.replace('block','entity'):v for k,v in audit(part,groups,truth).items()}
                    scored_body['scope'] = 'Oracle whole-body identities score every proposed entity; never used by the producer.'
                    whole.append(dict(ids=ids.tolist(), audit=scored_body))
                    keys = np.unique(np.stack((body[body>=0],link[body>=0]),-1),axis=0)
                    truth = np.stack([(body==i)&(link==j) for i,j in keys]) if len(keys) else np.empty((0,*body.shape),bool)
                    scored_link = {k.replace('block','entity'):v for k,v in audit(part,groups,truth).items()}
                    scored_link['scope'] = 'Oracle rigid-link identities distinguish articulated parts from fragments of one rigid part; scorer only.'
                    rigid.append(dict(body_link_ids=keys.tolist(), audit=scored_link))
            record.update(all_body_cameras=whole, rigid_link_cameras=rigid)
        result['records'].append(record)
    result['complete'] = True
    result['inputs'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [args.sam,args.plan,*args.groups,Path(__file__)]}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for camera in range(2):
        rows = [r['cameras'][camera] for r in result['records']]
        print(['top','wrist'][camera],json.dumps({k:sum(int(np.sum(r[k])) for r in rows)
              for k in rows[0] if k!='scope'}))


if __name__ == '__main__':
    main()
