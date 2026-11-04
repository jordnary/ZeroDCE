"""Generate local report figures from prepared images and experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


def clean_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(path) as source:
        clean = Image.frombytes("RGB", source.size, source.convert("RGB").tobytes())
    clean.save(path, optimize=True)


def curve_figure(directory):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.linspace(0, 1, 501)
    for r in [-1, -0.5, 0, 0.5, 1]:
        axes[0].plot(x, x+r*(x*x-x), label=f"r = {r:g}", linewidth=1.8)
    axes[1].plot(x, x, color="#94a3b8", linestyle="--", label="n = 0")
    for count in [1, 2, 4, 8]:
        y = x.copy()
        for _ in range(count):
            y = y-0.3*(y*y-y)
        axes[1].plot(x, y, label=f"n = {count}", linewidth=1.8)
    for ax, title in zip(axes, ["(a) Single curve", "(b) Repeated curve (r = -0.3)"]):
        ax.set(xlabel="Input intensity", ylabel="Output intensity", title=title,
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.2)
    figure.tight_layout()
    clean_figure(figure, directory/"curves.png")


def architecture_figure(directory):
    figure, ax = plt.subplots(figsize=(12, 5.6))
    ax.set(xlim=(-0.1, 13.5), ylim=(-0.1, 6.1))
    ax.axis("off")

    def box(x, y, width, height, label, color, fontsize=10):
        ax.add_patch(FancyBboxPatch(
            (x-width/2, y-height/2), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#64748b", linewidth=1.1))
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)

    def arrow(start, end, color="#475569"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=11, color=color, linewidth=1.3))

    centers = [0.75, 2.4, 4.05, 5.7, 7.35, 9.0, 10.65, 12.3]
    labels = ["Input I\nRGB", "Conv1\n3 to 32\nReLU", "Conv2\n32 to 32\nReLU",
              "Conv3\n32 to 32\nReLU", "Conv4\n32 to 32\nReLU",
              "Conv5\n64 to 32\nReLU", "Conv6\n64 to 32\nReLU",
              "Conv7\n64 to 24\nTanh"]
    for index, (center, label) in enumerate(zip(centers, labels)):
        color = "#e2e8f0" if index == 0 else ("#dcfce7" if index == 7 else "#dbeafe")
        box(center, 2.8, 1.22, 1.15, label, color)
        if index:
            arrow((centers[index-1]+0.65, 2.8), (center-0.65, 2.8))
        if 0 < index < 7:
            ax.text(center, 2.02, f"F{index}", ha="center", fontsize=10)
    for source, target, level in [(1, 7, 5.5), (2, 6, 4.8), (3, 5, 4.1)]:
        start, end = centers[source], centers[target]
        ax.plot([start, start, end], [3.43, level, level], color="#2563eb", linewidth=1.3)
        arrow((end, level), (end, 3.43), "#2563eb")
        ax.text((start+end)/2, level+0.1, f"F{source}: concatenate at Conv{target}",
                ha="center", fontsize=10, color="#1d4ed8")
    box(7.2, 0.65, 7.1, 0.85,
        r"$Y_0=I;\quad Y_n=Y_{n-1}+r_n\odot(Y_{n-1}^{\odot2}-Y_{n-1})$"+"\n"
        "8 successive curve updates", "#fef3c7", fontsize=11)
    box(12.3, 0.65, 1.25, 0.85, "Output\nY8", "#dcfce7")
    ax.plot([centers[0], centers[0]], [2.18, 0.65], color="#475569", linewidth=1.3)
    arrow((centers[0], 0.65), (3.58, 0.65))
    arrow((10.83, 0.65), (11.62, 0.65))
    ax.plot([12.3, 12.3, 7.2], [2.18, 1.5, 1.5], color="#15803d", linewidth=1.3)
    arrow((7.2, 1.5), (7.2, 1.13), "#15803d")
    ax.text(9.6, 1.65, "24 channels = 8 RGB parameter maps", ha="center",
            fontsize=10, color="#15803d")
    ax.text(6.7, 5.98, "DCE-Net: 3 x 3 convolutions, stride 1, padding 1; H x W preserved",
            ha="center", fontsize=11)
    figure.tight_layout(pad=0.5)
    clean_figure(figure, directory/"network_structure.png")


def sensitivity_figures(results, directory, records, crops):
    path = results/"sensitivity.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    series = []
    for index, record in enumerate(records):
        selected = sorted((row for row in rows if row["image"] == record["image"]),
                          key=lambda row: row["strength"])
        if selected:
            strengths = [row["strength"] for row in selected]
            if len(strengths) != len(set(strengths)):
                raise ValueError("Sensitivity strengths must be unique for each image")
            series.append((record["image"], f"Scene {index+1:02d}", selected))
    if not series:
        return

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    fields = [("mean_intensity", "(a) Mean RGB intensity", "Mean intensity"),
              ("dark_percent", "(b) Dark pixels (g < 0.1)", "Pixels (%)"),
              ("clipped_channel_percent", "(c) Near-saturated pixels", "Pixels (%)"),
              ("entropy_bits", "(d) Gray-level entropy", "Entropy (bit)")]
    styles = [("#2563eb", "o"), ("#0f766e", "s"), ("#a21caf", "^"), ("#c2410c", "D")]
    for ax, (field, title, ylabel) in zip(axes.flat, fields):
        all_strengths = set()
        for index, (_, label, selected) in enumerate(series):
            strengths = [row["strength"] for row in selected]
            all_strengths.update(strengths)
            color, marker = styles[index % len(styles)]
            ax.plot(strengths, [row[field] for row in selected], color=color,
                    marker=marker, markersize=5, linewidth=1.8, label=label)
        ax.axvline(1, color="#94a3b8", linewidth=1, linestyle="--")
        ax.set(title=title, xlabel="Curve strength s", ylabel=ylabel)
        ax.set_xticks(sorted(all_strengths))
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=9)
    figure.tight_layout(pad=1.1)
    clean_figure(figure, directory/"strength_metrics.png")

    figure, axes = plt.subplots(1, len(series), figsize=(5.3*len(series), 4.4), squeeze=False)
    terms = [("spatial", 1, r"$L_{spa}$", "#2563eb"),
             ("exposure", 10, r"$10 L_{exp}$", "#c2410c"),
             ("color", 5, r"$5 L_{col}$", "#a21caf"),
             ("smoothness", 200, r"$200 L_{tv}$", "#0f766e")]
    for ax, (_, label, selected) in zip(axes[0], series):
        strengths = [row["strength"] for row in selected]
        for field, weight, legend, color in terms:
            values = [weight*row[field] for row in selected]
            if any(value < 0 for value in values):
                raise ValueError("Loss components must be nonnegative")
            ax.plot(strengths, values, color=color, marker="o", markersize=4,
                    linewidth=1.6, label=legend)
        ax.plot(strengths, [row["total"] for row in selected], color="#0f172a",
                linestyle="--", marker="s", markersize=4, label="Total")
        ax.axvline(1, color="#94a3b8", linewidth=1, linestyle=":")
        positive = all(row[field] > 0 for row in selected
                       for field in ["spatial", "exposure", "color", "smoothness", "total"])
        scale = "log" if positive else "symlog"
        if positive:
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-5)
        ax.set(title=label, xlabel="Curve strength s",
               ylabel=f"Weighted loss ({scale} scale)")
        ax.set_xticks(strengths)
        ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=10, ncol=5,
                  loc="lower center", bbox_to_anchor=(0.5, 0))
    figure.tight_layout(pad=1.1, rect=(0, 0.1, 1, 1))
    clean_figure(figure, directory/"strength_losses.png")

    for name, _, selected in series:
        middle = min(range(len(selected)), key=lambda i: abs(selected[i]["strength"]-1))
        shown = [selected[i] for i in sorted({0, middle, len(selected)-1})]
        paths = [results/"candidates"/f"{name}_s{row['strength']:.2f}.png" for row in shown]
        if not all(path.exists() for path in paths):
            continue
        figure, axes = plt.subplots(2, len(shown), figsize=(4*len(shown), 5.8), squeeze=False)
        x0, y0, x1, y1 = crops[name][0]
        for column, (row, path) in enumerate(zip(shown, paths)):
            with Image.open(path) as source:
                array = np.asarray(source.convert("RGB"))
            axes[0, column].imshow(array)
            axes[0, column].set_title(f"s = {row['strength']:.2f}")
            axes[1, column].imshow(array[y0:y1, x0:x1], interpolation="nearest")
            axes[1, column].set_title("Region 1")
        for ax in axes.flat:
            ax.axis("off")
        figure.tight_layout(pad=0.6, h_pad=1.5)
        clean_figure(figure, directory/f"{name}_strengths.png")


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
    curve_figure(directory)
    architecture_figure(directory)
    sensitivity_figures(args.results, directory, records, crops)
    (directory/"regions.json").write_text(json.dumps(crops, indent=2)+"\n", encoding="utf-8")
    print("Comparison, detail, distribution, curve, and network figures generated.")
    if (args.results/"sensitivity.json").exists():
        print("Available strength measurements and candidate images plotted.")


if __name__ == "__main__":
    main()
