"""Score proposal splits/merges without using the oracle to select proposals."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np


def summarize(path):
    data = json.loads(path.read_text())
    if not data['complete']:
        raise ValueError('cannot summarize a partial proposal set as complete')
    views = []
    for camera in ('top', 'wrist'):
        objects, pairs, images, fragments = [], [], [], []
        negative_total = negative_wrong = positive_total = positive_wrong = 0
        object_background_pairs = 0
        contaminated = []
        for row in data['records']:
            c = next(c for c in row['cameras'] if c['camera'] == camera)
            images.append(c['seconds'])
            a = c['audit']
            negative_total += a['different_exclusive_masks_object_pair_endpoints']
            negative_wrong += a['same_physical_object_pair_endpoints']
            for mask in a['masks']:
                counts = np.array(mask['exclusive_object_pixels'], dtype=np.int64)
                n = int(counts.sum())
                # Only visible movable endpoints are classified. Robot and
                # other background pixels cannot be combined into one body.
                total = n*(n-1)//2
                same = int(sum(int(x)*(int(x)-1)//2 for x in counts))
                positive_total += total
                positive_wrong += total-same
                outside = mask['exclusive_pixels']-n
                if outside < 0:
                    raise ValueError('oracle object masks overlap')
                object_background_pairs += n*outside
                if n and (np.count_nonzero(counts)>1 or outside):
                    contaminated.append(dict(case=row['case_id'], step=row['step'],
                        mask=mask['mask'], exclusive_pixels=mask['exclusive_pixels'],
                        object_pixels=counts.tolist(), nonobject_pixels=outside))
            for o in a['objects']:
                if o['visible_pixels'] <= 0:
                    continue
                item = dict(case=row['case_id'], step=row['step'], **o)
                objects.append(item)
                if sum(v>0 for v in o['exclusive_mask_pixel_counts'])>1:
                    fragments.append(item)
        ious = [o['best_proposal_iou'] for o in objects]
        views.append(dict(camera=camera, images=len(images),
            visible_body_view_observations=len(objects),
            best_iou_min_median=[min(ious), float(np.median(ious))],
            visible_object_pixels=sum(o['visible_pixels'] for o in objects),
            exclusive_object_pixels=sum(o['exclusive_covered_pixels'] for o in objects),
            median_seconds_per_image=float(np.median(images)),
            different_mask_object_pair_endpoints=negative_total,
            same_body_among_different_mask_pairs=negative_wrong,
            same_mask_object_pair_endpoints=positive_total,
            different_body_among_same_mask_pairs=positive_wrong,
            same_mask_object_nonobject_pairs=object_background_pairs,
            objects_split_by_exclusive_masks=fragments,
            objects_with_best_iou_below_point9=[o for o in objects if o['best_proposal_iou']<.9],
            exclusive_masks_with_object_contamination=contaminated))
    return dict(complete=True, windows=len(data['records']),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), views=views,
        scope='Correlated same-domain pixels/windows; not independent trials. Best-IoU is a scoring ceiling only. Split/merge counts use the actual mask-exclusive partition, never oracle selection. Background identities and cross-view/time links are not certified.',
        production_promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.results:
        result = summarize(path)
        path.with_name('decision-summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(path.parent.name, json.dumps({k:v for k,v in result.items() if k!='views'}))
        for v in result['views']:
            print(json.dumps({k:len(x) if isinstance(x,list) and k not in ('best_iou_min_median',) else x for k,x in v.items()}))
