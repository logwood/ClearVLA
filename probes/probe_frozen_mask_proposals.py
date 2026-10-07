"""RGB-only frozen instance proposals, with independent masks used ONLY to score.

No proposal is a training label. Automatic image-wide grid prompts are fixed;
oracle masks, object names, physical state and policy outputs cannot prompt the
model. Best-IoU matches below are scoring ceilings, never proposal selection.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, pipeline


SETTINGS = dict(points_per_crop=32, points_per_batch=64, crops_n_layers=0,
                pred_iou_thresh=.88, stability_score_thresh=.95,
                crops_nms_thresh=.7, mask_threshold=0.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def dump(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def propose(generator, rgb):
    """Producer accepts only RGB; oracle information never enters this call."""
    output = generator(Image.fromarray(rgb), **SETTINGS)
    masks = np.stack([np.asarray(x, dtype=bool) for x in output['masks']]) if output['masks'] else np.empty((0, *rgb.shape[:2]), bool)
    scores = output['scores']
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    scores = np.asarray(scores, np.float64)
    if len(scores) != len(masks) or not np.isfinite(scores).all():
        raise ValueError('invalid mask/score result')
    return masks, scores


def score_proposals(masks, truth, names):
    # Ambiguous overlaps stay unknown. Never use an oracle IoU to select a
    # mask or pick a granularity. Score exclusive support separately from raw
    # model proposal capacity, because nested masks can make labels unusable.
    count = masks.sum(0)
    exclusive = masks & (count[None] == 1)
    rows, objects = [], []
    for index, (mask, unique) in enumerate(zip(masks, exclusive)):
        overlap = (mask[None] & truth).sum((1, 2))
        unique_overlap = (unique[None] & truth).sum((1, 2))
        rows.append(dict(mask=index, pixels=int(mask.sum()),
                         object_pixels=overlap.tolist(),
                         exclusive_pixels=int(unique.sum()),
                         exclusive_object_pixels=unique_overlap.tolist()))
    for name, body in zip(names, truth):
        inter = (masks & body[None]).sum((1, 2))
        union = (masks | body[None]).sum((1, 2))
        iou = inter/np.maximum(union, 1)
        unique = (exclusive & body[None]).sum((1, 2))
        best = int(iou.argmax()) if len(iou) else None
        objects.append(dict(name=name, visible_pixels=int(body.sum()),
                            best_proposal_iou=float(iou.max()) if len(iou) else 0.,
                            best_proposal_index=best,
                            exclusive_covered_pixels=int(unique.sum()),
                            exclusive_mask_pixel_counts=unique.tolist()))
    # This denominator only concerns endpoints belonging to the three visible
    # movable blocks. Background is unclassified, not one giant physical body.
    total, same = 0, 0
    for left, right in itertools.combinations(rows, 2):
        a = np.asarray(left['exclusive_object_pixels'], np.int64)
        b = np.asarray(right['exclusive_object_pixels'], np.int64)
        total += int(a.sum()*b.sum()); same += int(a@b)
    return dict(objects=objects, masks=rows,
                different_exclusive_masks_object_pair_endpoints=total,
                same_physical_object_pair_endpoints=same,
                scope='Only movable-block oracle endpoints scored; background/robot are not certified negatives. Proposal IoU maxima are audit ceilings, not a selector.')


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'weights', 'weight-receipt', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    receipt = json.loads(args.weight_receipt.read_text())
    if args.weights.resolve() != Path(receipt['path']).resolve():
        raise ValueError('weight identity mismatch')
    for item in receipt['files']:
        if sha(args.weights/item['name']) != item['sha256']:
            raise ValueError('weight digest mismatch')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = Sam2Model.from_pretrained(args.weights, local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    processor = Sam2Processor.from_pretrained(args.weights, local_files_only=True)
    generator = pipeline('mask-generation', model=model, image_processor=processor.image_processor, device=0)
    identity = dict(script_sha256=sha(__file__), plan_sha256=sha(args.plan),
                    weights=receipt, settings=SETTINGS, input='raw RGB only',
                    mask_role='independent scorer only', production_changed=False,
                    shared_image_features='one generator call only; no persistent RGB/feature cache')
    report = dict(identity=identity, records=[], complete=False)
    plan = json.loads(args.plan.read_text())
    checked_repeat = set()
    for row in plan:
        case = Path(row['case'])
        with np.load(case/'trajectory.npz', allow_pickle=False) as z:
            rgb = [z['rgb_static'], z['rgb_gripper']]
        for step in row['steps']:
            proposed, camera_records = [], []
            for view, images in enumerate(rgb):
                torch.cuda.synchronize(); start = time.monotonic()
                with torch.inference_mode():
                    masks, scores = propose(generator, images[step])
                torch.cuda.synchronize(); elapsed = time.monotonic()-start
                if view not in checked_repeat:
                    with torch.inference_mode():
                        again, other = propose(generator, images[step])
                    np.testing.assert_array_equal(masks, again)
                    np.testing.assert_array_equal(scores, other)
                    checked_repeat.add(view)
                proposed.append((masks, scores))
                camera_records.append(dict(camera=['top', 'wrist'][view],
                    rgb_sha256=hashlib.sha256(images[step].tobytes()).hexdigest(),
                    proposals=len(masks), seconds=elapsed,
                    scores=scores.tolist()))
            # Load the audit masks after every producer call for this window.
            path = Path(row['masks'])/case.name/('state_%03d.npz'%step)
            with np.load(path, allow_pickle=False) as z:
                truth = [z['top_masks'], z['wrist_masks']]
                names = z['object_names'].tolist()
            saved = {}
            for view, ((masks, scores), labels, record) in enumerate(zip(proposed, truth, camera_records)):
                if masks.shape[1:] != labels.shape[1:]:
                    raise ValueError('native camera size mismatch')
                record['audit'] = score_proposals(masks, labels, names)
                saved['masks_'+record['camera']] = masks
            target = args.output/('case_%02d_state_%03d.npz'%(row['case_id'], step))
            # Includes only binary proposals, never image embeddings.
            np.savez_compressed(target, **saved)
            report['records'].append(dict(case=str(case), case_id=row['case_id'], step=step,
                audit_mask_sha256=sha(path), cameras=camera_records,
                proposal_path=str(target), proposal_sha256=sha(target)))
            dump(args.output/'results.json', report)
            print('MASK_PROPOSALS', row['case_id'], step, [r['proposals'] for r in camera_records], flush=True)
    report['complete'] = True
    report['repeat_cameras'] = sorted(checked_repeat)
    dump(args.output/'results.json', report)


if __name__ == '__main__':
    main()
