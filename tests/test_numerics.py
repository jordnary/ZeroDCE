"""Numerical contracts for curve mapping, tiling, and upstream losses."""

import unittest

import zerodce  # Configure supported MKL threading before importing torch.
import torch

from zerodce.metrics import color_loss, exposure_loss, smoothness_loss, spatial_loss
from zerodce.model import apply_curves, infer, load_model


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        cls.model = load_model()

    def test_official_architecture_and_forward(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 79416)
        image = torch.rand(1, 3, 21, 35)
        with torch.inference_mode():
            fourth, eighth, curves = self.model(image)
        self.assertEqual(tuple(curves.shape), (1, 24, 21, 35))
        torch.testing.assert_close(apply_curves(image, curves, iterations=4), fourth)
        torch.testing.assert_close(apply_curves(image, curves), eighth)

    def test_tiling_preserves_internal_and_outer_boundaries(self):
        image = torch.rand(1, 3, 67, 89)
        full, full_curves = infer(self.model, image, 0)
        tiled, tile_curves = infer(self.model, image, 24)
        torch.testing.assert_close(tiled, full, rtol=0, atol=2e-6)
        torch.testing.assert_close(tile_curves, full_curves, rtol=0, atol=2e-6)

    def test_curve_range_monotonicity_and_endpoints(self):
        image = torch.linspace(0, 1, 1001).reshape(1, 1, 1, -1).repeat(1, 3, 1, 1)
        for r in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            curves = torch.full((1, 24, 1, 1001), r)
            result = apply_curves(image, curves)
            self.assertTrue(torch.all((result >= 0) & (result <= 1)))
            self.assertTrue(torch.all(result.diff(dim=-1) >= 0))
            self.assertEqual(float(result[0, 0, 0, 0]), 0)
            self.assertEqual(float(result[0, 0, 0, -1]), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_tiling_on_synthetic_image(self):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        generator = torch.Generator().manual_seed(7)
        image = torch.rand(1, 3, 193, 257, generator=generator) * 0.25
        model = load_model(device="cuda")
        full, full_curves = infer(model, image, 0)
        tiled, tile_curves = infer(model, image, 64)
        torch.testing.assert_close(tiled, full, rtol=0, atol=2e-6)
        torch.testing.assert_close(tile_curves, full_curves, rtol=0, atol=2e-6)

    def test_loss_scales_and_border_convention(self):
        gray = torch.full((1, 3, 16, 16), 0.6)
        self.assertAlmostEqual(float(exposure_loss(gray)), 0, places=10)
        self.assertEqual(float(color_loss(gray)), 0)
        self.assertEqual(float(spatial_loss(gray, gray)), 0)
        self.assertAlmostEqual(float(spatial_loss(torch.zeros_like(gray), torch.ones_like(gray)*0.1)), 0.01, places=7)
        red = torch.zeros_like(gray)
        red[:, 0] = 1
        self.assertAlmostEqual(float(color_loss(red)), 2**0.5, places=6)
        rows = torch.arange(3).reshape(1, 1, 3, 1)
        columns = torch.arange(4).reshape(1, 1, 1, 4)
        maps = (rows+columns).float().repeat(1, 24, 1, 1)
        self.assertEqual(float(smoothness_loss(maps)), 96)


if __name__ == "__main__":
    unittest.main()
