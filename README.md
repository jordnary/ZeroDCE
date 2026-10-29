# Zero-DCE 暗光图像增强

**简体中文** | [English](README.en.md)

基于官方 Zero-DCE 网络与预训练权重的暗光图像增强工具，提供分块推理、可选去噪、定量统计和图表生成。Git 仓库保留可复用代码、测试、使用说明及官方权重；作业要求、报告、个人照片和实验产物仅保存在本地。

## 目录结构

| 位置 | 用途 | Git 管理 |
| --- | --- | --- |
| `README.md` / `README.en.md` | 可切换的中英文使用说明 | 是 |
| `zerodce/` | 推理、图像 I/O、损失、统计与后处理 | 是 |
| `scripts/` | 输入准备、权重下载、图表与审计工具 | 是 |
| `examples/` | 简洁的 PyTorch 推理示例 | 是 |
| `tests/` | 数值、图像 I/O 与仓库隐私检查 | 是 |
| `third_party/` | 官方网络源码及来源声明 | 是 |
| `weights/` | 官方预训练权重与校验清单 | 仅官方文件 |
| `assignment/` | 作业要求及附件 | 否 |
| `reports/` | 实验报告、草稿及报告导出文件 | 否 |
| `data/private/` | 原始个人照片 | 否 |
| `data/input/` | 预处理输入及校验清单 | 否 |
| `results/` | 增强结果、图表、指标与敏感性分析 | 否 |
| `private/` | 其他个人材料 | 否 |
| `scratch/` | 临时实验与调试文件 | 否 |
| `dist/` | 本地交付包 | 否 |

所有文件与目录使用英文名称。中英文 README 均保持在根目录；中文正文可以保留。Git 不保存空目录，首次克隆后可在 PowerShell 中创建本地材料目录：

```powershell
New-Item -ItemType Directory -Force assignment, reports, data/private, data/input, results, private, scratch, dist | Out-Null
```

## 安装与运行

需要 Python 3.10 或更高版本。PyTorch 可使用 CPU 或兼容的 CUDA 后端。以下命令均从仓库根目录执行：

```shell
python -m pip install -r requirements.txt
python scripts/prepare_inputs.py data/private/photo_a.jpg data/private/photo_b.jpg
python -m zerodce --denoise
python scripts/make_figures.py
```

照片由使用者自行提供。准备脚本统一方向与色彩空间、移除元数据，并将最长边缩至不超过 1600 像素，按顺序写入 `data/input/scene_01.png` 等文件。重新执行会覆盖同名文件；不同实验可通过 `--output` 指定独立目录。

仓库包含官方小型权重 `weights/Epoch99.pth`，加载时校验 SHA-256。需要重新获取时，执行 `python scripts/download_weights.py`。

纯模型推理与自定义实验目录：

```shell
python -m zerodce --device cpu --output scratch/model_only
python -m examples.enhance
python scripts/prepare_inputs.py data/private/photo_a.jpg --output scratch/input
python -m zerodce --input scratch/input --output scratch/output --denoise
python scripts/make_figures.py --input scratch/input --results scratch/output
```

图表脚本需要带 `--denoise` 生成的结果，默认选取相对位置的局部区域；可使用 `--regions results/figures/regions.json` 复用或自定义裁剪区域。

## 方法与输出

- 原样使用官方七层 DCE-Net，对每个 RGB 像素应用八次曲线映射，不重新训练权重。
- `--tile-size 512` 为默认分块大小，保留七像素上下文；`--tile-size 0` 表示整图推理。
- `--strength 1.0` 保持官方曲线参数；其他强度属于推理变体。可使用 `--compare-strengths 0.85 1 1.15` 进行比较。
- `results/enhanced/` 保存直接模型输出，`results/final/` 保存可选 NLM 混合去噪结果，`results/figures/` 保存图表。
- `results/metrics.json` 保存指标、损失、校验值与运行参数；`results/metrics.csv` 保存各版本的曝光、暗区比例、近饱和比例和灰度熵等图像统计。

图像指标基于归一化 sRGB 编码值，不能解释为物理照度。四项损失不能单独代表感知质量；没有正常曝光参考图时，不报告 PSNR、SSIM 或真实重构误差。

## 验证与本地材料

```shell
python -m unittest discover -s tests -v
python scripts/verify_project.py
```

默认测试与审计不依赖私人照片、作业或报告。审计检查可提交文件、英文命名、隐私规则、文档链接以及官方源码和权重校验值。需要核验本地实验结果时，另行运行：

```shell
python scripts/verify_project.py --local-results
python scripts/verify_project.py --local-results --input scratch/input --results scratch/output
```

`assignment/`、`reports/`、整个 `data/`、`results/`、`private/`、`scratch/` 和 `dist/` 都在 `.gitignore` 中，不要强制添加。报告中的图片可用 `../results/figures/` 相对路径引用。图像移除元数据后仍可能显示可识别的场景，因此输入与输出均仅保留在本地。

`.gitignore` 不会清除已有 Git 历史中的文件。首次发布已有仓库前，应检查历史中是否曾提交个人或作业材料。

## 使用范围

官方网络与权重遵循上游 CC BY-NC 4.0 及学术研究用途限制。源码、权重来源和改动边界见 [第三方声明](third_party/NOTICE.md)。本地照片、作业与报告不属于公开代码分发内容。
