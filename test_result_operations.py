import csv
import tempfile
import unittest
from pathlib import Path

from result_logger import (
    BASE_DIR,
    DEFAULT_LOG_PATH,
    FIELDNAMES,
    add_manual_result,
    clear_all_results,
    save_result,
    undo_last_result,
)
from stats_calculator import get_current_stats, load_match_records


class ResultOperationsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "logs" / "matches.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_path_is_relative_to_python_file(self):
        self.assertEqual(DEFAULT_LOG_PATH, BASE_DIR / "logs" / "match_results.csv")

    def test_duplicates_are_each_counted_as_a_match(self):
        save_result("WIN", 1.0, 2.0, timestamp="same", log_path=self.csv_path)
        save_result("WIN", 1.0, 2.0, timestamp="same", log_path=self.csv_path)
        stats = get_current_stats(self.csv_path)
        self.assertEqual(stats["total_matches"], 2)
        self.assertEqual(stats["win_count"], 2)

    def test_manual_add_and_undo_last_valid_row(self):
        add_manual_result("WIN", timestamp="first", log_path=self.csv_path)
        with self.csv_path.open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(["bad", "DRAW"])
        add_manual_result("LOSE", timestamp="last", log_path=self.csv_path)

        removed = undo_last_result(self.csv_path)
        self.assertEqual(removed["result"], "LOSE")
        self.assertEqual([record.result for record in load_match_records(self.csv_path)[0]], ["WIN"])

    def test_clear_creates_backup_and_leaves_header(self):
        add_manual_result("WIN", log_path=self.csv_path)
        archive_dir = self.root / "archive"
        backup = clear_all_results(self.csv_path, archive_dir)

        self.assertIsNotNone(backup)
        self.assertTrue(backup.exists())
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            self.assertEqual(list(csv.reader(file)), [FIELDNAMES])

    def test_clear_without_backup_does_not_create_archive_and_leaves_header(self):
        add_manual_result("LOSE", log_path=self.csv_path)
        archive_dir = self.root / "unused-archive"

        backup = clear_all_results(self.csv_path, archive_dir, backup=False)

        self.assertIsNone(backup)
        self.assertFalse(archive_dir.exists())
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            self.assertEqual(list(csv.reader(file)), [FIELDNAMES])

    def test_empty_header_only_and_invalid_rows_are_safe(self):
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.touch()
        self.assertEqual(get_current_stats(self.csv_path)["total_matches"], 0)

        clear_all_results(self.csv_path, self.root / "archive")
        with self.csv_path.open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerows([["broken"], ["date", "DRAW"], ["date", "WIN", "bad", "bad"]])

        stats = get_current_stats(self.csv_path)
        self.assertEqual(stats["total_matches"], 1)
        self.assertIsNone(stats["recent_matches"][0]["score"])

    def test_header_only_csv_is_zero_matches(self):
        clear_all_results(self.csv_path, self.root / "archive")
        self.assertEqual(get_current_stats(self.csv_path)["total_matches"], 0)

    def test_save_result_rejects_broken_header(self):
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.write_text("broken,header\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ヘッダーが標準形式ではない"):
            save_result("WIN", 1.0, 2.0, log_path=self.csv_path)

    def test_broken_header_csv_is_not_modified(self):
        self.csv_path.parent.mkdir(parents=True)
        original = b"broken,header\r\nexisting,data\r\n"
        self.csv_path.write_bytes(original)
        with self.assertRaises(ValueError):
            save_result("LOSE", 1.0, 2.0, log_path=self.csv_path)
        self.assertEqual(self.csv_path.read_bytes(), original)

    def test_recent_matches_is_limited_to_ten(self):
        for index in range(12):
            add_manual_result("WIN" if index % 2 == 0 else "LOSE", timestamp=str(index), log_path=self.csv_path)
        stats = get_current_stats(self.csv_path)
        self.assertEqual(stats["total_matches"], 12)
        self.assertEqual(len(stats["recent_matches"]), 10)
        self.assertEqual(stats["recent_matches"][0]["timestamp"], "2")


if __name__ == "__main__":
    unittest.main()
