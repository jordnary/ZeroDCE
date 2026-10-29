"""Minimal inference example; run with python -m examples.enhance."""

from pathlib import Path

import zerodce
import torch

from zerodce.io import read_image, save_image
from zerodce.model import infer, load_model


def main():
    files = sorted(Path("data/input").glob("*.png"))
    if not files:
        raise SystemExit("No prepared images found; run scripts/prepare_inputs.py first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path("weights/Epoch99.pth"), device)
    with torch.inference_mode():
        for path in files:
            image = read_image(path)
            enhanced, _ = infer(model, image, tile_size=512)
            save_image(enhanced, Path("scratch/example")/path.name)
    print("Example inference completed.")


if __name__ == "__main__":
    main()
