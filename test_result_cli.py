import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from result_cli import create_parser
from result_logger import ARCHIVE_DIR, DEFAULT_LOG_PATH

CLI_PATH = Path(__file__).resolve().parent / "result_cli.py"


class ResultCliTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "logs" / "matches.csv"
        self.archive_dir = Path(self.temp_dir.name) / "archive"

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_cli(self, command: str) -> dict:
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                command,
                "--log-path",
                str(self.log_path),
                "--archive-dir",
                str(self.archive_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        return payload["data"]

    def run_cli_with_environment(self, command: str) -> dict:
        environment = os.environ.copy()
        environment["TEKKEN8_LOG_PATH"] = str(self.log_path)
        environment["TEKKEN8_ARCHIVE_DIR"] = str(self.archive_dir)
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), command],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        return payload["data"]

    def test_commands_use_json_and_update_temporary_csv(self):
        self.assertEqual(self.run_cli("get-stats")["total_matches"], 0)
        self.assertEqual(self.run_cli("add-win")["win_count"], 1)
        self.assertEqual(self.run_cli("add-lose")["lose_count"], 1)
        self.assertEqual(self.run_cli("undo")["stats"]["total_matches"], 1)

        cleared = self.run_cli("clear")
        self.assertEqual(cleared["stats"]["total_matches"], 0)
        self.assertTrue(Path(cleared["backup_path"]).exists())

    def test_environment_log_path_is_the_only_csv_written(self):
        self.assertEqual(self.run_cli_with_environment("add-win")["win_count"], 1)
        self.assertTrue(self.log_path.exists())
        self.assertFalse((Path(self.temp_dir.name) / "match_results.csv").exists())

    def test_environment_archive_dir_is_the_only_backup_location(self):
        self.run_cli_with_environment("add-lose")
        cleared = self.run_cli_with_environment("clear")
        backup_path = Path(cleared["backup_path"])
        self.assertEqual(backup_path.parent, self.archive_dir)
        self.assertEqual(list(self.archive_dir.glob("*.csv")), [backup_path])

    def test_environment_commands_do_not_change_real_paths(self):
        real_csv_before = DEFAULT_LOG_PATH.read_bytes() if DEFAULT_LOG_PATH.exists() else None
        real_archive_before = self._directory_snapshot(ARCHIVE_DIR)

        self.run_cli_with_environment("add-win")
        self.run_cli_with_environment("clear")

        real_csv_after = DEFAULT_LOG_PATH.read_bytes() if DEFAULT_LOG_PATH.exists() else None
        self.assertEqual(real_csv_after, real_csv_before)
        self.assertEqual(self._directory_snapshot(ARCHIVE_DIR), real_archive_before)

    def test_unset_environment_falls_back_to_existing_paths(self):
        with mock.patch.dict(
            os.environ,
            {"TEKKEN8_LOG_PATH": "", "TEKKEN8_ARCHIVE_DIR": ""},
        ):
            args = create_parser().parse_args(["get-stats"])
        self.assertEqual(args.log_path, DEFAULT_LOG_PATH)
        self.assertEqual(args.archive_dir, ARCHIVE_DIR)

    @staticmethod
    def _directory_snapshot(directory: Path) -> list[tuple[str, int, int]]:
        if not directory.exists():
            return []
        return sorted(
            (str(path.relative_to(directory)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in directory.rglob("*")
            if path.is_file()
        )


if __name__ == "__main__":
    unittest.main()
