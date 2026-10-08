"""CPU-only source-timed RGB requests for frozen online DINOv3.

No neural module, CUDA allocation or feature-cache access occurs in workers.
Synthetic tail fillers remain unsupported and are not loaded/encoded.
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset import CachedTokenPolicyWindowDataset, ObservedStateWindowDataset, _camera_stack


class OnlineRGBPolicyWindowDataset(CachedTokenPolicyWindowDataset):
    """Share sampler/boundary delegation, replace only the feature producer."""
    def __init__(self, base: ObservedStateWindowDataset, *, identity_raw_root=None,
                 identity_region_manifest='',identity_region_manifest_sha256=''):
        self.base = base
        self.identity_labels = None
        if identity_raw_root is not None:
            from .identity_correspondence import IdentityLabelProducer
            self.identity_labels = IdentityLabelProducer(identity_raw_root,
                region_manifest=identity_region_manifest,region_manifest_sha256=identity_region_manifest_sha256)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        history = sample.pop("history_keys")
        future = sample.pop("future_keys")
        reference = sample.pop("instruction_reference_key", None)
        executed = sample.pop("executed_world_keys", None)
        endpoint = sample.pop("annotation_endpoint_key", None)
        parts = [history, future]
        flags = [torch.ones(len(history), dtype=torch.bool),
                 sample.get("future_visual_observed", torch.ones(len(future),dtype=torch.bool)).bool()]
        if reference is not None:
            parts.append(reference); flags.append(torch.ones(len(reference),dtype=torch.bool))
        if executed is not None:
            parts.append(executed); flags.append(torch.full((len(executed),),bool(sample["executed_world_observed"])))
        if endpoint is not None:
            parts.append(endpoint); flags.append(torch.full((len(endpoint),),bool(sample["annotation_endpoint_declared"])))
        keys = torch.cat(parts)
        valid = torch.cat(flags)
        if valid.ndim != 1 or valid.shape != keys.shape[:1]:
            raise ValueError("online visual row support is not source-aligned")
        episode_idx = int(sample["episode_idx"])
        if not bool((keys[:,0] == episode_idx).all()):
            raise ValueError("one policy window cannot read another episode's frames")
        episode = self.base.episodes[episode_idx]
        if self.identity_labels is not None:
            sample.update(self.identity_labels(episode,history,self.base.image_store))
        source_rows = keys[valid,1].unique(sorted=True).numpy()
        frames = self.base.image_store.load_window(episode, source_rows)
        loaded = _camera_stack(frames, self.base.camera_names)
        # image_store transport is [T,C,3,H,W] uint8.
        if loaded.dtype != torch.uint8 or loaded.shape[2] != 3:
            raise ValueError("image store must retain uint8 RGB")
        loaded = loaded.contiguous()
        positions = {int(row):i for i,row in enumerate(source_rows)}
        cameras = len(self.base.camera_names)
        rgb = torch.zeros((len(keys),cameras,3,*loaded.shape[-2:]),dtype=torch.uint8)
        for row in valid.nonzero().flatten().tolist():
            rgb[row] = loaded[positions[int(keys[row,1])]]
        sample["visual_request_rgb"] = rgb
        sample["visual_request_keys"] = torch.cat((
            keys[:,None,:].expand(-1,cameras,-1),
            torch.arange(cameras)[None,:,None].expand(len(keys),-1,-1),
        ),dim=-1)
        sample["visual_request_valid"] = valid[:,None].expand(-1,cameras).clone()
        sample["target_future_offsets"] = sample.pop("future_offsets")
        return sample
