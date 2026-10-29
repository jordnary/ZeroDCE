"""Checked loading and bounded-memory inference for the upstream network."""

import hashlib
import json
from pathlib import Path

import torch

from third_party.zero_dce.model import enhance_net_nopool


def load_model(weights=Path("weights/Epoch99.pth"), device="cpu"):
    weights = Path(weights)
    manifest = json.loads(weights.with_name("manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint checksum mismatch.")
    model = enhance_net_nopool()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


def apply_curves(image, curves, strength=1.0, iterations=8):
    if not 0 < strength <= 2 or not 1 <= iterations <= 8:
        raise ValueError("strength must be in (0, 2] and iterations in [1, 8]")
    result = image
    for curve in curves.split(3, dim=1)[:iterations]:
        r = (curve * strength).clamp(-1, 1)
        result = result + r * (result.square() - result)
    return result


@torch.inference_mode()
def infer(model, image, tile_size=512):
    """Return eight-step RGB output and 24 maps on CPU, with exact 7-pixel halos.

    The network has seven stride-one 3x3 convolutions. Each tile uses all
    seven pixels of context; only its center is copied to the output.
    Padding is therefore used only at true image boundaries.
    """
    if image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("Expected a single NCHW RGB image.")
    if tile_size < 0 or (tile_size and tile_size < 16):
        raise ValueError("tile_size must be zero or at least 16")
    device = next(model.parameters()).device
    if tile_size == 0:
        _, enhanced, curves = model(image.to(device))
        return enhanced.cpu(), curves.cpu()
    _, _, height, width = image.shape
    enhanced = torch.empty_like(image, device="cpu")
    curves = torch.empty((1, 24, height, width), dtype=image.dtype, device="cpu")
    halo = 7
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            y0, x0 = max(0, top - halo), max(0, left - halo)
            y1, x1 = min(height, bottom + halo), min(width, right + halo)
            _, tile, maps = model(image[:, :, y0:y1, x0:x1].to(device))
            ys, xs = slice(top-y0, bottom-y0), slice(left-x0, right-x0)
            enhanced[:, :, top:bottom, left:right] = tile[:, :, ys, xs].cpu()
            curves[:, :, top:bottom, left:right] = maps[:, :, ys, xs].cpu()
    return enhanced, curves

