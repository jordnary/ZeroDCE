"""Generate local report figures from prepared images and experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def clean_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(path) as source:
        clean = Image.frombytes("RGB", source.size, source.convert("RGB").tobytes())
    clean.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--regions", type=Path, help="Local JSON mapping image stems to two crop rectangles")
    args = parser.parse_args()
    directory = args.results / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    records = json.loads((args.results/"metrics.json").read_text(encoding="utf-8"))["records"]
    if not records:
        parser.error("No experiment records found")
    if any("final" not in record for record in records):
        parser.error("Figures require denoised outputs; run inference with --denoise")
    crops = json.loads(args.regions.read_text(encoding="utf-8")) if args.regions else {}
    histogram, axes_hist = plt.subplots(1, len(records), figsize=(11, 3.5), squeeze=False)
    for index, record in enumerate(records):
        name = record["image"]
        paths = [args.input/f"{name}.png", args.results/"enhanced"/f"{name}.png",
                 args.results/"final"/f"{name}.png"]
        images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
        height, width = images[0].shape[:2]
        crops.setdefault(name, [
            [int(width*0.65), int(height*0.1), max(1, int(width*0.9)), max(1, int(height*0.4))],
            [int(width*0.35), int(height*0.4), max(1, int(width*0.6)), max(1, int(height*0.7))],
        ])
        if len(crops[name]) != 2 or any(
            len(region) != 4 or not all(isinstance(value, int) for value in region)
            or not (0 <= region[0] < region[2] <= width and 0 <= region[1] < region[3] <= height)
            for region in crops[name]
        ):
            parser.error("Each image needs two valid crop rectangles inside its dimensions")
        titles = ["Input", "Zero-DCE", "Zero-DCE + mild NLM"]
        figure, axes = plt.subplots(1, 3, figsize=(15, 3.2))
        for ax, array, title in zip(axes, images, titles):
            ax.imshow(array)
            ax.set_title(title)
            ax.axis("off")
        figure.tight_layout(pad=0.4)
        clean_figure(figure, directory/f"{name}_comparison.png")
        figure, axes = plt.subplots(2, 3, figsize=(12, 5.5))
        for row, (x0, y0, x1, y1) in enumerate(crops[name]):
            for col, (array, title) in enumerate(zip(images, titles)):
                axes[row, col].imshow(array[y0:y1, x0:x1], interpolation="nearest")
                axes[row, col].set_title(f"{title} / region {row+1}")
                axes[row, col].axis("off")
        figure.tight_layout(pad=0.8, h_pad=1.6)
        clean_figure(figure, directory/f"{name}_details.png")
        ax = axes_hist[0, index]
        for array, title, color in zip(images, titles, ["#64748b", "#2563eb", "#e18727"]):
            ax.hist(array.mean(2).ravel()/255, bins=64, range=(0, 1), density=True,
                    histtype="step", linewidth=1.4, label=title, color=color)
        ax.set(title=f"Scene {index+1:02d}", xlabel="Mean RGB intensity", ylabel="Density", xlim=(0, 1))
        ax.legend(frameon=False, fontsize=9)
    histogram.tight_layout()
    clean_figure(histogram, directory/"intensity_distribution.png")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.linspace(0, 1, 501)
    for alpha in [-1, -0.5, 0, 0.5, 1]:
        axes[0].plot(x, x+alpha*x*(1-x), label=f"alpha = {alpha:g}")
    for count in [1, 2, 4, 8]:
        y = x.copy()
        for _ in range(count):
            y = y+0.3*y*(1-y)
        axes[1].plot(x, y, label=f"n = {count}")
    for ax, title in zip(axes, ["Single curve", "Repeated curve (alpha = 0.3)"]):
        ax.set(xlabel="Input intensity", ylabel="Output intensity", title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    figure.tight_layout()
    clean_figure(figure, directory/"curves.png")
    (directory/"regions.json").write_text(json.dumps(crops, indent=2)+"\n", encoding="utf-8")
    print("Comparison, detail, distribution, and curve figures generated.")


if __name__ == "__main__":
    main()
