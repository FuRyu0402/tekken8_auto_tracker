import io
import json
import threading
import unittest
from contextlib import redirect_stdout
from unittest import mock

import monitor_cli
from capture_classifier import (
    MONITOR_INDEX,
    create_argument_parser,
    listen_for_stop_commands,
    validate_monitor_index,
)


MONITORS = [
    {"left": -1920, "top": 0, "width": 3840, "height": 1080},
    {"left": -1920, "top": 0, "width": 1920, "height": 1080, "name": "Left"},
    {"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": True},
]


class _FakeMss:
    monitors = MONITORS

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class MonitorAndControlTest(unittest.TestCase):
    def test_monitor_zero_is_excluded_and_optional_fields_are_preserved(self):
        result = monitor_cli.format_monitors(MONITORS)
        self.assertEqual([item["index"] for item in result], [1, 2])
        self.assertEqual(result[0]["left"], -1920)
        self.assertEqual(result[0]["name"], "Left")
        self.assertTrue(result[1]["is_primary"])

    def test_monitor_cli_stdout_is_json_only(self):
        output = io.StringIO()
        with mock.patch("monitor_cli.mss.MSS", return_value=_FakeMss()), redirect_stdout(output):
            exit_code = monitor_cli.main()
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["data"]), 2)

    def test_invalid_monitor_indices_are_rejected(self):
        for index in (0, -1, len(MONITORS)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_monitor_index(index, MONITORS)
        self.assertEqual(validate_monitor_index(1, MONITORS), 1)

    def test_monitor_argument_defaults_and_override(self):
        parser = create_argument_parser()
        self.assertEqual(parser.parse_args([]).monitor_index, MONITOR_INDEX)
        self.assertEqual(parser.parse_args(["--monitor-index", "1"]).monitor_index, 1)

    def test_stdin_stop_and_quit_set_event_without_capture(self):
        for command in ("stop\n", "quit\n"):
            with self.subTest(command=command):
                event = threading.Event()
                listen_for_stop_commands(event, io.StringIO(command))
                self.assertTrue(event.is_set())

    def test_closed_stdin_does_not_set_event_or_raise(self):
        stream = io.StringIO("")
        stream.close()
        event = threading.Event()
        listen_for_stop_commands(event, stream)
        self.assertFalse(event.is_set())


if __name__ == "__main__":
    unittest.main()
