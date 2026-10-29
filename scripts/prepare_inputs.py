"""Normalize orientation and color, resize, and discard source metadata."""

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageCms, ImageOps


def prepare(source, target, max_side=1600):
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        original_size = list(image.size)
        profile = original.info.get("icc_profile")
        if profile:
            image = ImageCms.profileToProfile(
                image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"), outputMode="RGB",
            )
        if max_side and max(image.size) > max_side:
            width, height = image.size
            scale = max_side / max(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
        clean = Image.frombytes("RGB", image.size, image.tobytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        clean.save(target, format="PNG", optimize=True)
    return {
        "image": target.name, "original_size": original_size,
        "size": list(clean.size), "color_space": "sRGB",
        "embedded_profile_converted": bool(profile), "metadata": "removed",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/input"))
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args()
    if args.max_side < 16:
        parser.error("max-side must be at least 16")
    records = [prepare(p, args.output / f"scene_{i:02d}.png", args.max_side)
               for i, p in enumerate(args.inputs, 1)]
    (args.output / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} metadata-free images.")


if __name__ == "__main__":
    main()
