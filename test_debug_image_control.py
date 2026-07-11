import os
import unittest
from unittest import mock

import capture_classifier


class DebugImageControlTest(unittest.TestCase):
    def call_all_save_entries(self):
        capture_classifier.save_detection_debug_image(None, "win", 1.0, 2.0, "test")
        capture_classifier.save_candidate_debug_image(None, "win", 1.0, 2.0, "test")
        capture_classifier.save_rejected_debug_image(None, "lose", 1.0, 2.0, "test")

    def test_enabled_environment_values_suppress_all_image_savers(self):
        for value in ("1", "true", "TRUE", "yes", "YeS", "on", "ON"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"TEKKEN8_DISABLE_DEBUG_IMAGES": value},
            ), mock.patch.object(capture_classifier, "save_detection_roi") as detection, mock.patch.object(
                capture_classifier, "save_candidate_roi"
            ) as candidate, mock.patch.object(capture_classifier, "save_rejected_roi") as rejected:
                self.call_all_save_entries()
                detection.assert_not_called()
                candidate.assert_not_called()
                rejected.assert_not_called()

    def test_unset_environment_preserves_existing_save_calls(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            capture_classifier, "save_detection_roi", return_value="detection.png"
        ) as detection, mock.patch.object(
            capture_classifier, "save_candidate_roi", return_value="candidate.png"
        ) as candidate, mock.patch.object(
            capture_classifier, "save_rejected_roi", return_value="rejected.png"
        ) as rejected:
            os.environ.pop("TEKKEN8_DISABLE_DEBUG_IMAGES", None)
            self.call_all_save_entries()
            detection.assert_called_once()
            candidate.assert_called_once()
            rejected.assert_called_once()

    def test_disabled_environment_values_preserve_existing_save_calls(self):
        for value in ("0", "false", "no", "off", "", "unexpected"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"TEKKEN8_DISABLE_DEBUG_IMAGES": value},
            ), mock.patch.object(
                capture_classifier, "save_detection_roi", return_value="detection.png"
            ) as detection, mock.patch.object(
                capture_classifier, "save_candidate_roi", return_value="candidate.png"
            ) as candidate, mock.patch.object(
                capture_classifier, "save_rejected_roi", return_value="rejected.png"
            ) as rejected:
                self.call_all_save_entries()
                detection.assert_called_once()
                candidate.assert_called_once()
                rejected.assert_called_once()


if __name__ == "__main__":
    unittest.main()
