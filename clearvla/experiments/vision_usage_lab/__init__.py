"""Experimental laboratory for measuring and improving visual usage in robot policies.

This package is intentionally separated from the production-oriented curriculum
path.  It studies whether visual evidence carries causal, temporally aligned
information that improves action generation beyond an action-history shortcut.
"""

from .teacher import PatchTeacherConfig, build_patch_teacher
from .latent_cache import VisionLatentCacheStore, build_all_vision_latent_caches
from .dataset import VisionUsageLabDataset, LabVisualMode
from .model import AdaptiveSolverConfig, VisionUsageLabModel, VisionUsageLabModelConfig
from .losses import VisionUsageLabLossConfig, vision_usage_lab_loss

__all__ = [
    "PatchTeacherConfig",
    "build_patch_teacher",
    "VisionLatentCacheStore",
    "build_all_vision_latent_caches",
    "VisionUsageLabDataset",
    "LabVisualMode",
    "AdaptiveSolverConfig",
    "VisionUsageLabModel",
    "VisionUsageLabModelConfig",
    "VisionUsageLabLossConfig",
    "vision_usage_lab_loss",
]
