import unittest
from unittest import mock

import capture_classifier


class PreviewControlTest(unittest.TestCase):
    def test_no_preview_argument_parsing(self):
        parser = capture_classifier.create_argument_parser()
        self.assertFalse(parser.parse_args([]).no_preview)
        self.assertTrue(parser.parse_args(["--no-preview"]).no_preview)

    def test_disabled_preview_skips_all_opencv_window_calls(self):
        with mock.patch.object(capture_classifier, "draw_preview") as draw, mock.patch.object(
            capture_classifier.cv2, "imshow"
        ) as imshow, mock.patch.object(capture_classifier.cv2, "waitKey") as wait_key:
            should_quit = capture_classifier.show_preview(
                frame=None,
                roi_box=None,
                result=None,
                state=None,
                preview_enabled=False,
            )
        self.assertFalse(should_quit)
        draw.assert_not_called()
        imshow.assert_not_called()
        wait_key.assert_not_called()

    def test_enabled_preview_keeps_imshow_waitkey_and_q_behavior(self):
        preview = object()
        resized = object()
        with mock.patch.object(capture_classifier, "draw_preview", return_value=preview), mock.patch.object(
            capture_classifier, "resize_for_display", return_value=resized
        ), mock.patch.object(capture_classifier.cv2, "imshow") as imshow, mock.patch.object(
            capture_classifier.cv2, "waitKey", return_value=ord("q")
        ) as wait_key:
            should_quit = capture_classifier.show_preview(
                frame=object(),
                roi_box=object(),
                result=object(),
                state=object(),
                preview_enabled=True,
            )
        self.assertTrue(should_quit)
        imshow.assert_called_once_with(capture_classifier.WINDOW_NAME, resized)
        wait_key.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
