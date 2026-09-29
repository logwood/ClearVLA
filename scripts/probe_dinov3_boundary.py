"""Explicit encoder-only probe; not a ClearVLA training or rollout entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from clearvla.vision.dinov3_online import FrozenDinoV3Encoder, DinoV3SpatialSpec, preprocess_full_fov
from clearvla.vision.preprocessing import PreprocessConfig, apply_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, nargs='+', required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--revision')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preprocess-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.preprocess_only and not args.model:
        parser.error('--model is required unless --preprocess-only')
    if args.output.exists():
        parser.error('output directory already exists; refusing to overwrite evidence')
    args.output.mkdir(parents=True)
    report: dict[str, object] = {
        'scope': 'encoder-boundary-only-not-policy-training',
        'pretrained_forward_completed': False, 'spatial_spec': DinoV3SpatialSpec().identity(),
        'torch': torch.__version__, 'device': args.device, 'images': [],
    }
    try:
        arrays = []
        for path in args.images:
            with Image.open(path) as img:
                native = np.asarray(img.convert('RGB'))
            arrays.append(apply_preprocess(native, PreprocessConfig(resize_hw=(336, 336), crop_hw=None)))
            report['images'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'native_shape': list(native.shape)})
        rgb = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.
        report['preprocessed_shape'] = list(preprocess_full_fov(rgb).shape)
        if not args.preprocess_only:
            dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
            model = FrozenDinoV3Encoder.from_pretrained(args.model, revision=args.revision,
                local_files_only=not args.allow_download, device=args.device, dtype=dtype, microbatch=args.microbatch)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            features = model(rgb)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            report['seconds_first_forward'] = time.perf_counter() - start
            again = model(rgb)
            report.update({'pretrained_forward_completed': True, 'encoder_identity': model.identity(),
                'feature_shape': list(features.shape), 'finite': bool(torch.isfinite(features).all()),
                'repeat_max_abs': float((features-again).abs().max()), 'output_requires_grad': features.requires_grad})
            head = torch.nn.Linear(features.shape[-1], 1, bias=False).to(features.device)
            head(features).square().mean().backward()
            report['downstream_head_gradient_finite'] = bool(torch.isfinite(head.weight.grad).all())
            report['frozen_encoder_has_grad'] = any(p.grad is not None for p in model.parameters())
        report['status'] = 'preprocessing_only' if args.preprocess_only else 'encoder_probe_completed'
    except Exception as exc:
        report.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
