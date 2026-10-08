"""Official RoMa v2.0.1 precise-settings RGB-only measurement qualification.

Both full official refinement passes and both directions are retained. Only
network downloads are redirected to pinned local official artifacts. Geometry
labels score native-pixel endpoints after inference; they never enter matching.
This external matcher is not inserted into training or the deployed policy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from probe_observed_rgb_motion_candidate import score, stats, dump


class Estimator:
    def __init__(self, weights, dino_source):
        from romav2 import RoMaV2
        original_weights = torch.hub.load_state_dict_from_url
        original_hub = torch.hub.load
        def local_weights(url, **kwargs):
            if url != 'https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt':
                raise ValueError('unexpected model download request')
            return torch.load(weights, map_location='cpu', weights_only=True)
        def local_hub(repo_or_dir, model, **kwargs):
            if (repo_or_dir != 'facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa'
                    or model != 'dinov3_vitl16' or kwargs.get('pretrained') is not False
                    or kwargs.get('weights') is not None):
                raise ValueError('unexpected backbone request or external weights')
            return original_hub(str(dino_source), model, source='local', pretrained=False, weights=None)
        torch.hub.load_state_dict_from_url = local_weights
        torch.hub.load = local_hub
        try:
            self.model = RoMaV2(RoMaV2.Cfg(setting='precise', compile=False)).eval().requires_grad_(False)
        finally:
            torch.hub.load_state_dict_from_url = original_weights
            torch.hub.load = original_hub
        if (self.model.H_lr, self.model.W_lr, self.model.H_hr, self.model.W_hr, self.model.bidirectional) != (800, 800, 1280, 1280, True):
            raise ValueError('official precise budget changed')
        if self.model.threshold is not None:
            raise ValueError('audit cannot round learned overlap to one')
        self.versions = {n: p._version for n, p in self.model.named_parameters()}

    @torch.inference_mode()
    def __call__(self, source, target):
        if source.dtype != np.uint8 or source.shape != target.shape or target.dtype != np.uint8:
            raise ValueError('native RGB contract')
        h, w = source.shape[:2]
        prediction = self.model.match(source, target)
        def native(field):
            return F.interpolate(field.permute(0, 3, 1, 2).float(), size=(h, w),
                                 mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
        scale = np.array([w, h], dtype=np.float32)
        forward = (native(prediction['warp_AB'])+1)*scale/2-.5
        backward = (native(prediction['warp_BA'])+1)*scale/2-.5
        overlap = native(prediction['overlap_AB'])[..., 0]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        returned = cv2.remap(backward, forward[..., 0], forward[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped = cv2.remap(target, forward[..., 0], forward[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cycle = np.linalg.norm(returned-np.stack((xx, yy), -1), axis=-1)
        photo = np.abs(source.astype(np.float32)-warped.astype(np.float32)).mean(-1)/255
        inside = (forward[..., 0]>=0)&(forward[..., 0]<=w-1)&(forward[..., 1]>=0)&(forward[..., 1]<=h-1)
        if not np.isfinite(forward).all() or not np.isfinite(overlap).all():
            raise ValueError('nonfinite official matcher output')
        return dict(xy=forward, accepted=inside&(cycle<.75)&(photo<.08), cycle=cycle,
                    photometric=photo, overlap=overlap)


def main():
    parser = argparse.ArgumentParser()
    for key in ('plan', 'labels', 'weights', 'dino-source', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('deterministic audit environment required')
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    weight_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weight_hash != '1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7':
        raise ValueError('weights differ from official GitHub asset SHA256')
    complete = json.loads((args.labels/'complete.json').read_text())
    if complete.get('rigid_simulation_oracle_audit_only') is not True:
        raise ValueError('independent exact-render geometry scoring required')
    index = {(r['case'], r['step']): r for r in json.loads((args.labels/'results.json').read_text())}
    estimator = Estimator(args.weights, args.dino_source)
    result = dict(complete=False, records=[], repeat_controls=[], identity=dict(scope=__doc__,
        source='https://github.com/Parskatt/RoMaV2/tree/95c9968145c8906b7b59383258e9f73b02853d89',
        paper='https://arxiv.org/abs/2511.15706', weight_sha256=weight_hash,
        parameter_load='official constructor strict_all_keys', settings='official precise 800+1280 bidirectional',
        coordinate_conversion='align_corners_false normalized endpoints -> (uv+1)*[W,H]/2-.5 native pixel centers',
        local_correlation='official native_torch fallback; fused extension unavailable',
        policy_changed=False, label_complete=complete, torch=torch.__version__, opencv=cv2.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    checked = set()
    torch.cuda.reset_peak_memory_stats()
    for row in json.loads(args.plan.read_text()):
        case = Path(row['case'])
        with np.load(case/'trajectory.npz', allow_pickle=False) as z:
            rgb = {k: z[k] for k in ('rgb_static', 'rgb_gripper')}
        for step in row['steps']:
            entry = index[(case.name, step)]
            path = Path(entry['production_labels'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['production_labels_sha256']:
                raise ValueError('scoring labels changed')
            with np.load(path, allow_pickle=False) as z:
                labels = {k: z[k] for k in z.files}
            before, after = map(int, labels['source_frames'])
            if min(before, after) < 0 or max(before, after) > step:
                raise ValueError('only observed image pairs are admitted')
            cameras = []
            for camera, key in enumerate(('rgb_static', 'rgb_gripper')):
                a, b = rgb[key][before], rgb[key][after]
                torch.cuda.synchronize()
                started = time.perf_counter()
                field = estimator(a, b)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                scored = score(field, labels, camera, a.shape[:2])
                scored['seconds'] = elapsed
                q = (labels['temporal_source'][camera]+1)*np.array([(a.shape[1]-1)/2, (a.shape[0]-1)/2])
                overlap = cv2.remap(field['overlap'], q[:, 0].astype(np.float32), q[:, 1].astype(np.float32), cv2.INTER_LINEAR).reshape(-1)
                valid = labels['audit_rigid_temporal_valid'][camera] & (labels['temporal_source_body'][camera]>=0)
                hidden = labels['audit_rigid_occluded'][camera] & labels['audit_rigid_source_interior'][camera]
                scored['learned_overlap'] = dict(visible_object=stats(overlap[valid]), occluded=stats(overlap[hidden]))
                cameras.append(scored)
                if camera not in checked:
                    repeat = estimator(a, b)
                    error = float(np.abs(field['xy']-repeat['xy']).max())
                    if error != 0 or not np.array_equal(field['accepted'], repeat['accepted']):
                        raise ValueError('official candidate is not repeatable in this runtime')
                    same = estimator(a, a)
                    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
                    result['repeat_controls'].append(dict(camera=camera, repeat_max=error,
                        identical_image_motion_pixels=stats(np.linalg.norm(same['xy']-np.stack((xx, yy), -1), axis=-1))))
                    checked.add(camera)
            result['records'].append(dict(case=case.name, step=step, source_frames=[before, after],
                methods={'romav2_precise': cameras}))
            dump(args.output/'results.json', result)
            print('ROMAV2', case.name, step, flush=True)
    if estimator.versions != {n: p._version for n, p in estimator.model.named_parameters()}:
        raise ValueError('read-only matcher changed parameters')
    result.update(complete=True, parameters_unchanged=True, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    dump(args.output/'results.json', result)


if __name__ == '__main__':
    main()
