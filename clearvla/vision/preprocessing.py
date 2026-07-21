from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2

    cv2.setNumThreads(0)
except Exception:  # optional dependency
    cv2 = None


@dataclass(frozen=True)
class PreprocessConfig:
    """Explicit visual preprocessing.

    Defaults preserve native decoded image resolution. No resize or crop is applied
    unless configured by the caller.
    """

    resize_hw: tuple[int, int] | None = None
    crop_hw: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.resize_hw is not None:
            out["resize_hw"] = list(self.resize_hw)
        if self.crop_hw is not None:
            out["crop_hw"] = list(self.crop_hw)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessConfig":
        resize = data.get("resize_hw")
        crop = data.get("crop_hw")
        return cls(
            resize_hw=None if resize is None else (int(resize[0]), int(resize[1])),
            crop_hw=None if crop is None else (int(crop[0]), int(crop[1])),
        )


def parse_hw(values: list[int] | tuple[int, ...] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 integers for H W, got {values}")
    h, w = int(values[0]), int(values[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(h, w)}")
    return h, w


def _resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def _center_crop(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    crop_h, crop_w = hw
    h, w = img.shape[:2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop {(crop_h, crop_w)} exceeds image shape={(h, w)}")
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


def apply_preprocess(img: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape={img.shape}")
    out = img
    if config.resize_hw is not None:
        out = _resize(out, config.resize_hw)
    if config.crop_hw is not None:
        out = _center_crop(out, config.crop_hw)
    return np.ascontiguousarray(out, dtype=np.uint8)
