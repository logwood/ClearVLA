from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2

    cv2.setNumThreads(0)
except Exception:  # optional dependency
    cv2 = None


def decode_image_value(value: Any) -> np.ndarray:
    """Decode encoded bytes or raw RGB arrays into contiguous uint8 RGB HWC."""
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray)):
        if cv2 is not None:
            buf = np.frombuffer(value, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cv2.imdecode failed")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with Image.open(io.BytesIO(value)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()

    arr = np.asarray(value)
    if arr.ndim == 1 and arr.dtype == np.uint8:
        return decode_image_value(arr.tobytes())
    if arr.dtype.kind in {"S", "O"}:
        if arr.shape == ():
            return decode_image_value(arr.item())
        if arr.size == 1:
            return decode_image_value(arr.reshape(-1)[0])
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape={arr.shape}, dtype={arr.dtype}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"Unsupported image channel layout={arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)
