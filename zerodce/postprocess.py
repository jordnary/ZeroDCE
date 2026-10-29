"""Conservative non-local means refinement of the delivered 8-bit image."""

import cv2
import numpy as np
import torch


def denoise(image):
    rgb = np.rint(image.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).clip(0, 255).astype(np.uint8)
    cv2.setNumThreads(4)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(bgr, None, 2, 3, 7, 21)
    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Retain half of the original signal to avoid flattening fine foliage.
    result = (0.5*rgb.astype(np.float32) + 0.5*filtered_rgb)/255
    return torch.from_numpy(result).permute(2, 0, 1).unsqueeze(0)
