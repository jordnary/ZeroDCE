# Third-party attribution

`zero_dce/model.py` is the unmodified official Zero-DCE network by Chunle Guo,
Chongyi Li and collaborators. The pretrained `weights/Epoch99.pth` checkpoint
comes from the same upstream repository.

- Repository: <https://github.com/Li-Chongyi/Zero-DCE>
- Pinned revision: `e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd`
- Network source: `Zero-DCE_code/model.py`
- Network Git blob SHA-1: `3b710a5a33b38ee742c200c69b060f05246c7d1c`
- Checkpoint: `Zero-DCE_code/snapshots/Epoch99.pth`
- Checkpoint SHA-256: `a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612`

The upstream README specifies **Attribution-NonCommercial 4.0 International**
and makes the code available for academic research only. See the
[upstream terms](https://github.com/Li-Chongyi/Zero-DCE/blob/e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd/README.md#license)
and the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
Redistribution here retains these restrictions and attribution; no commercial
license is granted.

The loss calculations in `zerodce/metrics.py` reproduce the numerical conventions
in upstream `Myloss.py` and `lowlight_train.py`. They replace hard-coded CUDA
tensors with device-independent operations and omit unused auxiliary losses.
The remaining scripts are project-specific inference, evaluation, preparation,
visualization, and verification code.

OpenCV is used for optional non-local means denoising under its own license;
it is an external dependency and is not vendored. Personal photographs, assignment
materials, reports, and generated results are local files excluded from the
public code distribution.
