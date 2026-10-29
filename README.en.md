# Zero-DCE Low-Light Image Enhancement

[简体中文](README.md) | **English**

A low-light enhancement tool using the official Zero-DCE network and pretrained checkpoint, with tiled inference, optional denoising, measurements, and figure generation. Git contains reusable code, tests, documentation, and the official checkpoint. Assignments, reports, personal photographs, and experiment outputs stay local.

## Repository layout

| Location | Purpose | Tracked in Git |
| --- | --- | --- |
| `README.md` / `README.en.md` | Linked Chinese and English guides | Yes |
| `zerodce/` | Inference, image I/O, losses, statistics, and postprocessing | Yes |
| `scripts/` | Input preparation, checkpoint download, figures, and audits | Yes |
| `examples/` | Compact PyTorch inference example | Yes |
| `tests/` | Numerical, image I/O, and repository privacy checks | Yes |
| `third_party/` | Official network source and attribution | Yes |
| `weights/` | Official checkpoint and checksum manifest | Official files only |
| `assignment/` | Assignment requirements and attachments | No |
| `reports/` | Reports, drafts, and exported documents | No |
| `data/private/` | Original personal photographs | No |
| `data/input/` | Prepared inputs and checksum manifest | No |
| `results/` | Enhanced images, figures, metrics, and sensitivity analysis | No |
| `private/` | Other personal materials | No |
| `scratch/` | Temporary experiments and debugging files | No |
| `dist/` | Local delivery archives | No |

Use English names for all files and directories. Both READMEs remain at the root; Chinese document text is supported. Git does not store empty directories. After cloning, create local material directories in PowerShell:

```powershell
New-Item -ItemType Directory -Force assignment, reports, data/private, data/input, results, private, scratch, dist | Out-Null
```

## Install and run

Python 3.10 or newer is required. PyTorch supports CPU execution or a compatible CUDA backend. Run commands from the repository root:

```shell
python -m pip install -r requirements.txt
python scripts/prepare_inputs.py data/private/photo_a.jpg data/private/photo_b.jpg
python -m zerodce --denoise
python scripts/make_figures.py
```

Supply your own photographs. Preparation normalizes orientation and color space, removes metadata, and limits the longest side to 1600 pixels. Outputs are named sequentially, such as `data/input/scene_01.png`. Repeated runs overwrite matching names; use `--output` to separate experiments.

The small official checkpoint `weights/Epoch99.pth` is included and checked against its SHA-256 when loaded. Run `python scripts/download_weights.py` to retrieve it again.

Model-only inference and custom experiment directories:

```shell
python -m zerodce --device cpu --output scratch/model_only
python -m examples.enhance
python scripts/prepare_inputs.py data/private/photo_a.jpg --output scratch/input
python -m zerodce --input scratch/input --output scratch/output --denoise
python scripts/make_figures.py --input scratch/input --results scratch/output
```

Figure generation requires results produced with `--denoise`. Detail regions use relative positions by default; pass `--regions results/figures/regions.json` to reuse or customize crop coordinates.

## Method and outputs

- The unmodified official seven-layer DCE-Net applies eight pixelwise RGB curve iterations without retraining.
- `--tile-size 512` is the default and retains seven pixels of context; `--tile-size 0` selects whole-image inference.
- `--strength 1.0` preserves official curve parameters. Other strengths are inference variants; use `--compare-strengths 0.85 1 1.15` to compare them.
- `results/enhanced/` contains direct model outputs, `results/final/` contains optional NLM-blended outputs, and `results/figures/` contains figures.
- `results/metrics.json` contains measurements, losses, checksums, and run settings; `results/metrics.csv` contains per-version image statistics such as exposure, dark-area, near-saturation, and intensity entropy.

Statistics use normalized sRGB code values, not physical illuminance. The four losses are not a single perceptual-quality score. Without normally exposed reference images, PSNR, SSIM, and ground-truth reconstruction errors are not reported.

## Verification and local materials

```shell
python -m unittest discover -s tests -v
python scripts/verify_project.py
```

Default tests and audits do not require personal images, assignments, or reports. The audit checks publishable files, English names, privacy rules, document links, and official source and checkpoint checksums. Verify local experiment outputs separately:

```shell
python scripts/verify_project.py --local-results
python scripts/verify_project.py --local-results --input scratch/input --results scratch/output
```

`assignment/`, `reports/`, all of `data/`, `results/`, `private/`, `scratch/`, and `dist/` are ignored by Git; do not force-add them. Reports can reference figures using relative paths under `../results/figures/`. Removing metadata does not anonymize visible scenes, so inputs and outputs both remain local.

`.gitignore` does not remove files from existing Git history. Before publishing an existing repository, check whether personal or coursework materials were committed previously.

## Terms

The upstream network and checkpoint are subject to CC BY-NC 4.0 and its academic-research restriction. See the [third-party notices](third_party/NOTICE.md) for provenance and modification boundaries. Local photographs, assignments, and reports are excluded from the public code distribution.
