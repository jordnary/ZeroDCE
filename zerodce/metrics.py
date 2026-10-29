"""Upstream loss conventions and descriptive, no-reference image statistics."""

import numpy as np
import torch
import torch.nn.functional as F


def spatial_loss(source, result):
    source = F.avg_pool2d(source.mean(1, keepdim=True), 4)
    result = F.avg_pool2d(result.mean(1, keepdim=True), 4)
    kernels = source.new_tensor([
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    ]).unsqueeze(1)
    difference = F.conv2d(source-result, kernels, padding=1)
    return difference.square().sum(1).mean()


def exposure_loss(result):
    return (F.avg_pool2d(result.mean(1, keepdim=True), 16)-0.6).square().mean()


def color_loss(result):
    r, g, b = result.mean((2, 3)).unbind(1)
    return ((r-g).pow(4)+(r-b).pow(4)+(g-b).pow(4)).sqrt().mean()


def smoothness_loss(curves):
    # Upstream sums over channels rather than averaging over them.
    vertical = (curves[:, :, 1:]-curves[:, :, :-1]).square().mean()
    horizontal = (curves[:, :, :, 1:]-curves[:, :, :, :-1]).square().mean()
    return 2 * curves.shape[1] * (vertical+horizontal)


def loss_values(source, result, curves):
    values = dict(spatial=spatial_loss(source, result), exposure=exposure_loss(result),
                  color=color_loss(result), smoothness=smoothness_loss(curves))
    values["total"] = (values["spatial"] + 10*values["exposure"]
                       + 5*values["color"] + 200*values["smoothness"])
    return {name: float(value) for name, value in values.items()}


def image_statistics(image):
    array = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    gray = array.mean(2)
    hist, _ = np.histogram(gray, bins=256, range=(0, 1))
    probabilities = hist[hist > 0] / hist.sum()
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "dark_percent": float(100*np.mean(gray < 0.1)),
        "clipped_channel_percent": float(100*np.mean(array.max(2) >= 250/255)),
        "entropy_bits": float(-(probabilities*np.log2(probabilities)).sum()),
    }

