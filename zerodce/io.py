"""Image I/O with explicit RGB scaling and metadata-free exports."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def read_image(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    if min(array.shape[:2]) < 16:
        raise ValueError("Image dimensions must be at least 16 pixels.")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    if not np.isfinite(array).all():
        raise ValueError("Cannot save a non-finite image.")
    array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(array).save(path, format="PNG", optimize=True)

