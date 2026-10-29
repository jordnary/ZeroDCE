"""Metadata removal, safe checkpoint loading, and exported result integrity."""

import json
from pathlib import Path
import tempfile
import unittest

import zerodce
import torch
from PIL import Image

from scripts.prepare_inputs import prepare
from zerodce.io import read_image, save_image
from zerodce.model import load_model


class IOTests(unittest.TestCase):
    def test_prepare_applies_orientation_and_removes_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"source.png", Path(directory)/"clean.png"
            image = Image.new("RGB", (32, 16), (20, 40, 60))
            exif = Image.Exif()
            exif[274], exif[271] = 6, "private-test-marker"
            image.save(source, exif=exif)
            prepare(source, target, 32)
            with Image.open(target) as clean:
                self.assertEqual(clean.size, (16, 32))
                self.assertFalse(clean.info)
                self.assertFalse(clean.getexif())

    def test_png_round_trip_with_half_quantum_bound(self):
        torch.manual_seed(1)
        tensor = torch.rand(1, 3, 19, 31)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"result.png"
            save_image(tensor, target)
            self.assertLessEqual(float((read_image(target)-tensor).abs().max()), 0.5/255+1e-7)
            with Image.open(target) as image:
                self.assertFalse(image.info)

    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"Epoch99.pth"
            target.write_bytes(b"invalid checkpoint")
            target.with_name("manifest.json").write_text(json.dumps({"sha256": "0"*64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(target)

    def test_prepared_image_manifest_matches_saved_pixels(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"source.png", Path(directory)/"scene_01.png"
            Image.new("RGB", (96, 54), (20, 40, 60)).save(source)
            record = prepare(source, target, 48)
            self.assertEqual(record["image"], "scene_01.png")
            self.assertEqual(record["original_size"], [96, 54])
            self.assertEqual(record["size"], [48, 27])
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), record["sha256"])
            with Image.open(target) as image:
                self.assertFalse(image.info)
                self.assertEqual(list(image.size), record["size"])


if __name__ == "__main__":
    unittest.main()
