"""Unqualified RGB-D group proposal, independent of model and audit identities.

This is a probe leaf, not a production label generator. Metric depth and the
admitted camera calibration supply world points. A dominant near-horizontal
support plane and local color/Euclidean connectivity propose compact groups.
No scene state, object count/name, instruction, body mask or model allocation
is accepted by this interface. Disconnected surfaces can still be one object;
the caller must audit that failure before treating groups as negatives.
"""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


SETTINGS = dict(plane_tolerance_m=.003, plane_up_cosine=.97,
                plane_trials=384, plane_sample_points=6000,
                minimum_plane_fraction=.10, minimum_height_m=.006,
                maximum_height_m=.20, neighbor_radius_m=.006,
                rgb_distance=.20, minimum_group_points=24,
                maximum_group_extent_m=.18, seed=0)


def world_points(depth, view, projection):
    depth = np.asarray(depth, np.float64)
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, 1.).ravel()
    p = np.asarray(projection, np.float64)
    ndc = np.stack(((2*(xx+.5)/w-1).ravel(), (1-2*(yy+.5)/h).ravel(),
                    (-p[2, 2]*safe+p[2, 3])/safe, np.ones_like(safe)))
    xyz = np.linalg.inv(p@view)@ndc
    xyz = (xyz[:3]/xyz[3:]).T
    valid = valid.ravel() & np.isfinite(xyz).all(1)
    return xyz, valid


def propose_groups(depths, rgbs, views, projections, *, support_mode='dominant_v1', workspace_point=None):
    settings = dict(SETTINGS)
    settings['support_mode'] = support_mode
    if support_mode not in ('dominant_v1', 'under_tcp_v2'):
        raise ValueError('undeclared support plane rule')
    if support_mode == 'under_tcp_v2':
        workspace_point = np.asarray(workspace_point, np.float64)
        if workspace_point.shape != (3,) or not np.isfinite(workspace_point).all():
            raise ValueError('support plane selection needs the observed TCP only')
    if not (len(depths) == len(rgbs) == len(views) == len(projections) == 2):
        raise ValueError('two observed camera charts required')
    points, colors, valid, shapes = [], [], [], []
    for depth, rgb, view, projection in zip(depths, rgbs, views, projections):
        if rgb.shape != depth.shape+(3,) or rgb.dtype != np.uint8:
            raise ValueError('native RGB and metric-depth chart mismatch')
        xyz, ok = world_points(depth, view, projection)
        points.append(xyz); colors.append(rgb.reshape(-1, 3)/255.)
        valid.append(ok); shapes.append(depth.shape)
    points = np.concatenate(points); colors = np.concatenate(colors)
    valid = np.concatenate(valid)
    ids = np.flatnonzero(valid)
    labels = np.full(len(points), -1, np.int32)
    empty = dict(settings=settings, accepted_groups=0, status='no_admitted_support_plane')
    if len(ids) < 32:
        return [labels[:np.prod(shapes[0])].reshape(shapes[0]),
                labels[np.prod(shapes[0]):].reshape(shapes[1])], empty
    rng = np.random.default_rng(settings['seed'])
    chosen = rng.choice(ids, min(len(ids), settings['plane_sample_points']), replace=False)
    sample = points[chosen]
    triples = sample[rng.integers(0, len(sample), (settings['plane_trials'], 3))]
    normals = np.cross(triples[:, 1]-triples[:, 0], triples[:, 2]-triples[:, 0])
    length = np.linalg.norm(normals, axis=1)
    legal = length > 1e-10
    normals = normals/np.maximum(length[:, None], 1e-10)
    normals *= np.where(normals[:, 2:3] < 0, -1., 1.)
    legal &= normals[:, 2] >= settings['plane_up_cosine']
    normals = normals[legal]; origins = triples[legal, 0]
    if len(normals):
        offsets = -(normals*origins).sum(1)
        distances = np.abs(sample@normals.T+offsets[None])
        counts = (distances < settings['plane_tolerance_m']).sum(0)
        best = int(counts.argmax())
    else:
        counts = np.zeros(1, int); best = 0
    if counts[best] < settings['minimum_plane_fraction']*len(sample):
        split = np.prod(shapes[0])
        return [labels[:split].reshape(shapes[0]), labels[split:].reshape(shapes[1])], empty
    candidates = []
    if support_mode == 'under_tcp_v2':
        # An image's largest horizontal plane may be the floor. Use the
        # highest measured plane below the observed tool whose measured
        # footprint contains the tool projection, never a known table height.
        seen = []
        for candidate in np.argsort(counts)[::-1]:
            if counts[candidate] < settings['minimum_plane_fraction']*len(sample):
                break
            n = normals[candidate]; off = offsets[candidate]
            z = -(n[:2]@workspace_point[:2]+off)/n[2]
            if z >= workspace_point[2]-.002 or any(abs(z-old) < .006 for old in seen):
                continue
            selected_plane = sample[distances[:, candidate] < settings['plane_tolerance_m']]
            try:
                hull = ConvexHull(selected_plane[:, :2])
            except QhullError:
                continue
            margin = float(np.max(hull.equations[:, :2]@workspace_point[:2]+hull.equations[:, 2]))
            if margin > .01:
                continue
            candidates.append(dict(index=int(candidate), height_at_tcp=float(z), footprint_margin=margin,
                                   sample_fraction=float(counts[candidate]/len(sample))))
            seen.append(z)
        if not candidates:
            split = int(np.prod(shapes[0])); empty.update(status='no_observed_plane_below_tcp')
            return [labels[:split].reshape(shapes[0]), labels[split:].reshape(shapes[1])], empty
        best = max(candidates, key=lambda x:x['height_at_tcp'])['index']
    inside = distances[:, best] < settings['plane_tolerance_m']
    center = sample[inside].mean(0)
    _, _, vh = np.linalg.svd(sample[inside]-center, full_matrices=False)
    normal = vh[-1]*(-1 if vh[-1, 2] < 0 else 1)
    offset = -float(normal@center)
    height = points@normal+offset
    foreground = valid & (height > settings['minimum_height_m']) & (height < settings['maximum_height_m'])
    selected = np.flatnonzero(foreground)
    groups = []; rejected = []
    if len(selected):
        xyz = points[selected]; rgb = colors[selected]
        pairs = cKDTree(xyz).query_pairs(settings['neighbor_radius_m'], output_type='ndarray')
        if len(pairs):
            pairs = pairs[np.linalg.norm(rgb[pairs[:, 0]]-rgb[pairs[:, 1]], axis=1) <= settings['rgb_distance']]
        edges = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
        n, component = connected_components(edges, directed=False)
        for c in range(n):
            which = selected[component == c]
            extent = np.ptp(points[which], axis=0)
            if len(which) < settings['minimum_group_points'] or extent.max() > settings['maximum_group_extent_m']:
                rejected.append(dict(points=len(which), extent_m=extent.tolist()))
                continue
            identity = len(groups)
            labels[which] = identity
            boundary = int(np.prod(shapes[0]))
            groups.append(dict(group=identity, points=len(which), extent_m=extent.tolist(),
                               centroid_m=points[which].mean(0).tolist(),
                               camera_points=[int((which < boundary).sum()), int((which >= boundary).sum())]))
    split = int(np.prod(shapes[0]))
    metadata = dict(settings=settings, status='proposed_only', accepted_groups=len(groups),
                    valid_points=int(valid.sum()), foreground_points=len(selected),
                    plane_normal=normal.tolist(), plane_offset=offset,
                    support_candidates=candidates,
                    plane_sample_fraction=float(inside.mean()), groups=groups,
                    rejected_components=len(rejected),
                    rejected_large=[x for x in rejected if x['points'] >= settings['minimum_group_points']])
    return [labels[:split].reshape(shapes[0]), labels[split:].reshape(shapes[1])], metadata


def audit_groups(group_maps, body_maps, movable_bodies):
    """Independent scorer; body identities never influence proposal selection."""
    groups = np.concatenate([x.ravel() for x in group_maps])
    bodies = np.concatenate([x.ravel() for x in body_maps])
    rows = []
    for group in sorted(set(groups[groups >= 0].tolist())):
        mask = groups == group
        ids, counts = np.unique(bodies[mask], return_counts=True)
        dominant = int(ids[counts.argmax()])
        rows.append(dict(group=group, points=int(mask.sum()),
                         body_counts={str(int(k)): int(v) for k, v in zip(ids, counts)},
                         dominant_body=dominant, purity=float(counts.max()/counts.sum()),
                         movable_name=movable_bodies.get(dominant)))
    same_pairs = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            probability = sum(v*b['body_counts'].get(k, 0) for k, v in a['body_counts'].items())/(a['points']*b['points'])
            same_pairs.append(dict(groups=[a['group'], b['group']], same_body_pair_fraction=probability))
    objects = []
    for body, name in movable_bodies.items():
        visible = bodies == body
        covered = visible & (groups >= 0)
        ids, counts = np.unique(groups[covered], return_counts=True)
        objects.append(dict(body=int(body), name=name, visible_pixels=int(visible.sum()),
                            proposed_pixels=int(covered.sum()),
                            group_pixels={str(int(k)): int(v) for k, v in zip(ids, counts)}))
    return dict(groups=rows, distinct_group_pair_audit=same_pairs, movable_coverage=objects,
                scope='Oracle body identities score groups only; same-body fractions reveal false different-group labels. No admission decisions use this scorer.')
