import csv
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from csv_file_lock import CsvFileLock, CsvLockTimeoutError, lock_path_for
from result_logger import FIELDNAMES, clear_all_results, save_result, undo_last_result
from stats_calculator import get_current_stats


ROOT = Path(__file__).resolve().parent
CLI_PATH = ROOT / "result_cli.py"


def _hold_csv_lock(csv_path: str, ready: multiprocessing.Queue, seconds: float) -> None:
    with CsvFileLock(Path(csv_path)):
        ready.put(True)
        time.sleep(seconds)


class CsvSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.log_path = self.root / "logs" / "matches.csv"
        self.archive_dir = self.root / "archive"

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
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        return payload["data"]

    def test_multiple_process_appends_have_no_missing_rows(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: self.run_cli("add-win"), range(24)))
        self.assertEqual(get_current_stats(self.log_path)["total_matches"], 24)

    def test_concurrent_append_and_stats_reads(self):
        commands = ["add-win"] * 16 + ["get-stats"] * 16
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(self.run_cli, commands))
        stats = get_current_stats(self.log_path)
        self.assertEqual(stats["total_matches"], 16)

    def test_concurrent_append_and_undo_preserves_expected_count(self):
        for index in range(20):
            save_result("WIN", None, None, timestamp=str(index), log_path=self.log_path)
        commands = ["add-win"] * 12 + ["undo"] * 12
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(self.run_cli, commands))
        self.assertEqual(get_current_stats(self.log_path)["total_matches"], 20)

    def test_concurrent_append_and_clear_loses_no_rows(self):
        for index in range(10):
            save_result("WIN", None, None, timestamp=f"initial-{index}", log_path=self.log_path)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.run_cli, "add-win") for _ in range(12)]
            clear_future = executor.submit(self.run_cli, "clear")
            for future in futures:
                future.result()
            cleared = clear_future.result()

        backup_stats = get_current_stats(Path(cleared["backup_path"]))
        final_stats = get_current_stats(self.log_path)
        self.assertEqual(backup_stats["total_matches"] + final_stats["total_matches"], 22)

    def test_lock_timeout_is_clear_and_uses_test_csv_lock(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        process = context.Process(target=_hold_csv_lock, args=(str(self.log_path), ready, 1.0))
        process.start()
        self.assertTrue(ready.get(timeout=5))
        try:
            with self.assertRaisesRegex(CsvLockTimeoutError, "CSVロック"):
                with CsvFileLock(self.log_path, timeout=0.1):
                    pass
        finally:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        self.assertEqual(lock_path_for(self.log_path).parent, self.log_path.parent)

    def test_atomic_replace_failure_preserves_original_and_cleans_temp(self):
        save_result("WIN", None, None, timestamp="original", log_path=self.log_path)
        original = self.log_path.read_bytes()
        with mock.patch("result_logger.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                undo_last_result(self.log_path)
        self.assertEqual(self.log_path.read_bytes(), original)
        self.assertEqual(list(self.log_path.parent.glob(f".{self.log_path.name}.*.tmp")), [])

    def test_clear_without_backup_keeps_lock_and_atomic_replace(self):
        save_result("WIN", None, None, timestamp="original", log_path=self.log_path)
        with mock.patch("result_logger.CsvFileLock", wraps=CsvFileLock) as lock_class:
            with mock.patch("result_logger.os.replace", wraps=__import__("os").replace) as replace:
                clear_all_results(self.log_path, self.archive_dir, backup=False)
        lock_class.assert_called_once_with(self.log_path)
        replace.assert_called_once()
        self.assertFalse(self.archive_dir.exists())

    def test_csv_remains_well_formed_after_concurrency(self):
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(lambda _: self.run_cli("add-lose"), range(10)))
        with self.log_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
        self.assertEqual(rows[0], FIELDNAMES)
        self.assertTrue(all(len(row) == len(FIELDNAMES) for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
