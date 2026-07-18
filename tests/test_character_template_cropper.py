import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "tools" / "character_template_cropper.py"
SPEC = importlib.util.spec_from_file_location("character_template_cropper", MODULE_PATH)
cropper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = cropper
SPEC.loader.exec_module(cropper)


class RoiTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape((10, 20, 3))

    def test_crop_valid_roi(self):
        result = cropper.crop_image(self.image, cropper.ROI(3, 2, 5, 4))
        self.assertEqual(result.shape, (4, 5, 3))
        np.testing.assert_array_equal(result, self.image[2:6, 3:8])

    def test_outside_roi_is_rejected(self):
        with self.assertRaises(cropper.CropperError):
            cropper.crop_image(self.image, cropper.ROI(18, 8, 3, 3))

    def test_zero_size_is_rejected(self):
        with self.assertRaises(cropper.CropperError):
            cropper.crop_image(self.image, cropper.ROI(0, 0, 0, 2))

    def test_clamp_roi(self):
        self.assertEqual(cropper.clamp_roi(cropper.ROI(-2, 8, 30, 8), 20, 10), cropper.ROI(0, 8, 20, 2))

    def test_mirror_roi(self):
        self.assertEqual(cropper.mirror_roi(cropper.ROI(40, 30, 260, 72), 1920), cropper.ROI(1620, 30, 260, 72))

    def test_copy_roi_is_non_mutating(self):
        original = {"left": cropper.ROI(1, 2, 3, 4), "right": cropper.ROI(9, 8, 7, 6)}
        result = cropper.copy_roi(original, "left", "right")
        self.assertEqual(result["right"], original["left"])
        self.assertNotEqual(original["right"], original["left"])


class NamingAndPathsTests(unittest.TestCase):
    def test_sanitize_name(self):
        self.assertEqual(cropper.sanitize_character_name("  Devil JIN!?  "), "devil_jin")

    def test_empty_name_is_rejected(self):
        with self.assertRaises(cropper.CropperError):
            cropper.sanitize_character_name(" ! ")

    def test_infer_name(self):
        self.assertEqual(cropper.infer_character_name("reina_right_001.png"), "reina")

    def test_output_side_is_separate(self):
        root = Path("templates/characters")
        self.assertEqual(cropper.build_output_path(root, "left", "Jin"), root / "left" / "jin.png")
        self.assertEqual(cropper.build_output_path(root, "right", "Jin"), root / "right" / "jin.png")


class ConfigTests(unittest.TestCase):
    def test_missing_config_uses_defaults(self):
        config, message = cropper.load_config(Path("does-not-exist.json"))
        self.assertEqual(config["version"], 1)
        self.assertIn("defaults", message.lower())

    def test_broken_config_uses_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text("{broken", encoding="utf-8")
            config, message = cropper.load_config(path)
            self.assertEqual(config["roi"]["left"]["width"], 260)
            self.assertIn("invalid", message.lower())

    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            config = json.loads(json.dumps(cropper.DEFAULT_CONFIG))
            config["roi"]["left"]["x"] = 12
            cropper.save_config(path, config)
            loaded, _ = cropper.load_config(path)
            self.assertEqual(loaded, config)


class BatchTests(unittest.TestCase):
    def test_prepare_batch_detects_existing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "left").mkdir()
            (root / "left" / "reina.png").touch()
            items, rejected = cropper.prepare_batch([Path("reina_left_001.png")], root, "left")
            self.assertFalse(rejected)
            self.assertTrue(items[0].exists)

    def test_batch_does_not_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "left" / "jin.png"
            output.parent.mkdir()
            output.write_bytes(b"original")
            item = cropper.BatchItem(Path("jin_001.png"), output, "jin", True)
            saved, skipped, errors = cropper.run_batch(
                [item], cropper.ROI(0, 0, 1, 1),
                reader=lambda _path: np.zeros((2, 2, 3), dtype=np.uint8),
                writer=lambda path, _image: path.write_bytes(b"changed"),
            )
            self.assertEqual((saved, skipped, errors), (0, 1, []))
            self.assertEqual(output.read_bytes(), b"original")

    def test_batch_crop_uses_source_pixels(self):
        captured = []
        item = cropper.BatchItem(Path("king.png"), Path("king-output.png"), "king", False)
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[3:5, 4:7] = 123
        saved, skipped, errors = cropper.run_batch(
            [item], cropper.ROI(4, 3, 3, 2),
            reader=lambda _path: image,
            writer=lambda _path, value: captured.append(value.copy()),
        )
        self.assertEqual((saved, skipped, errors), (1, 0, []))
        self.assertEqual(captured[0].shape, (2, 3, 3))
        self.assertTrue(np.all(captured[0] == 123))


if __name__ == "__main__":
    unittest.main()
