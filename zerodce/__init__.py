"""Reproducible inference with the official Zero-DCE model."""

import os

# Avoid loading a second OpenMP runtime through MKL-backed NumPy builds.
# This selects a supported MKL threading mode, without allowing duplicate runtimes.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

