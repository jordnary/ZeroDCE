"""Enhance prepared sRGB images and export auditable measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .io import read_image, save_image
from .metrics import image_statistics, loss_values
from .model import apply_curves, infer, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--weights", type=Path, default=Path("weights/Epoch99.pth"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--compare-strengths", nargs="*", type=float, default=[])
    parser.add_argument("--denoise", action="store_true", help="Also save a mildly denoised final image")
    args = parser.parse_args()
    if not 0 < args.strength <= 2 or any(not 0 < s <= 2 for s in args.compare_strengths):
        parser.error("strength must be in (0, 2]")
    if args.tile_size < 0 or (args.tile_size and args.tile_size < 16):
        parser.error("tile-size must be zero or at least 16")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not args.input.is_dir():
        parser.error("Input directory is missing; prepare local images with scripts/prepare_inputs.py")
    files = sorted(p for p in args.input.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        parser.error("No input images found")
    if len({p.stem for p in files}) != len(files):
        parser.error("Input filenames must have unique stems")
    model = load_model(args.weights, device)
    records, comparisons = [], []
    for path in files:
        original = read_image(path)
        enhanced, curves = infer(model, original, args.tile_size)
        if args.strength != 1:
            enhanced = apply_curves(original, curves, args.strength)
        output = args.output / "enhanced" / f"{path.stem}.png"
        save_image(enhanced, output)
        saved = read_image(output)
        # Statistics describe the delivered 8-bit PNG; losses use float tensors.
        records.append({
            "image": path.stem, "size": list(original.shape[-2:][::-1]),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "input": image_statistics(original), "enhanced": image_statistics(saved),
            "losses": loss_values(original, enhanced, (curves*args.strength).clamp(-1, 1)),
        })
        if args.denoise:
            from .postprocess import denoise
            final_path = args.output / "final" / f"{path.stem}.png"
            save_image(denoise(saved), final_path)
            records[-1]["final"] = image_statistics(read_image(final_path))
            records[-1]["final_sha256"] = hashlib.sha256(final_path.read_bytes()).hexdigest()
        for strength in args.compare_strengths:
            candidate = apply_curves(original, curves, strength)
            candidate_path = args.output / "candidates" / f"{path.stem}_s{strength:.2f}.png"
            save_image(candidate, candidate_path)
            comparisons.append({"image": path.stem, "strength": strength,
                                **image_statistics(read_image(candidate_path)),
                                **loss_values(original, candidate, (curves*strength).clamp(-1, 1))})
        print(f"Enhanced {path.stem}: {original.shape[-1]} x {original.shape[-2]}")
    manifest = json.loads(args.weights.with_name("manifest.json").read_text(encoding="utf-8"))
    payload = {"method": "official Zero-DCE", "weight_sha256": manifest["sha256"],
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "iterations": 8, "strength": args.strength, "tile_size": args.tile_size,
               "precision": "float32", "loss_convention": "official PyTorch implementation",
               "postprocess": {"method": "NLM blend", "h": 2, "h_color": 3,
                               "blend": 0.5, "template_window": 7,
                               "search_window": 21} if args.denoise else None,
               "records": records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    rows = [{"image": r["image"], "version": version, **r[version]}
            for r in records for version in ("input", "enhanced", "final") if version in r]
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if comparisons:
        (args.output / "sensitivity.json").write_text(json.dumps(comparisons, indent=2)+"\n", encoding="utf-8")
    print(f"Completed {len(records)} images; metrics exported.")


if __name__ == "__main__":
    main()
