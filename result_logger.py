from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
ARCHIVE_DIR = BASE_DIR / "archive"
DEFAULT_LOG_PATH = LOG_DIR / "match_results.csv"

FIELDNAMES = [
    "timestamp",
    "result",
    "score",
    "margin",
    "lose_score",
    "none_score",
    "win_score",
]
VALID_RESULTS = {"WIN", "LOSE"}


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _ensure_csv(log_path: Path) -> None:
    """Create a missing/empty CSV with the standard header."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        if not _has_standard_header(_read_rows(log_path)):
            raise ValueError(
                f"CSVのヘッダーが標準形式ではないため追記できません: {log_path}"
            )
        return

    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()


def _read_rows(log_path: Path) -> list[list[str]]:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return []
    try:
        with log_path.open("r", newline="", encoding="utf-8-sig") as file:
            return list(csv.reader(file))
    except (OSError, UnicodeError, csv.Error):
        return []


def _has_standard_header(rows: list[list[str]]) -> bool:
    return bool(rows) and [value.strip() for value in rows[0]] == FIELDNAMES


def read_valid_rows(log_path: Path = DEFAULT_LOG_PATH) -> list[dict[str, str]]:
    """Return every valid WIN/LOSE row in file order, including duplicates."""
    rows = _read_rows(Path(log_path))
    if not _has_standard_header(rows):
        return []

    valid_rows: list[dict[str, str]] = []
    for values in rows[1:]:
        row = {
            name: values[index].strip() if index < len(values) else ""
            for index, name in enumerate(FIELDNAMES)
        }
        row["result"] = row["result"].upper()
        if row["result"] in VALID_RESULTS:
            valid_rows.append(row)
    return valid_rows


def save_result(
    result: str,
    score: Optional[float],
    margin: Optional[float],
    scores: Optional[dict[str, float]] = None,
    timestamp: Optional[str] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one match result to the CSV."""
    result = result.strip().upper()
    if result not in VALID_RESULTS:
        raise ValueError(f"result must be 'WIN' or 'LOSE'. got: {result}")

    scores = scores or {}
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = Path(log_path)
    _ensure_csv(log_path)

    row = {
        "timestamp": timestamp,
        "result": result,
        "score": _format_float(score),
        "margin": _format_float(margin),
        "lose_score": _format_float(scores.get("lose")),
        "none_score": _format_float(scores.get("none")),
        "win_score": _format_float(scores.get("win")),
    }
    with log_path.open("a", newline="", encoding="utf-8-sig") as file:
        csv.DictWriter(file, fieldnames=FIELDNAMES).writerow(row)


def add_manual_result(
    result: str,
    timestamp: Optional[str] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a manually entered WIN or LOSE."""
    save_result(result, None, None, timestamp=timestamp, log_path=log_path)


def undo_last_result(log_path: Path = DEFAULT_LOG_PATH) -> Optional[dict[str, str]]:
    """Remove and return the most recent valid WIN/LOSE row."""
    log_path = Path(log_path)
    rows = _read_rows(log_path)
    if not _has_standard_header(rows):
        _ensure_csv(log_path)
        return None

    remove_index: Optional[int] = None
    removed: Optional[dict[str, str]] = None
    for index in range(len(rows) - 1, 0, -1):
        values = rows[index]
        result = values[1].strip().upper() if len(values) > 1 else ""
        if result in VALID_RESULTS:
            remove_index = index
            removed = {
                name: values[column].strip() if column < len(values) else ""
                for column, name in enumerate(FIELDNAMES)
            }
            removed["result"] = result
            break

    if remove_index is None:
        return None

    del rows[remove_index]
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerows(rows)
    return removed


def clear_all_results(
    log_path: Path = DEFAULT_LOG_PATH,
    archive_dir: Path = ARCHIVE_DIR,
) -> Optional[Path]:
    """Back up the CSV, then replace it with the standard header only."""
    log_path = Path(log_path)
    archive_dir = Path(archive_dir)
    backup_path: Optional[Path] = None

    if log_path.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = archive_dir / f"{log_path.stem}_{stamp}{log_path.suffix}"
        shutil.copy2(log_path, backup_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()
    return backup_path


def main() -> None:
    test_log_path = LOG_DIR / "test_match_results.csv"
    save_result("WIN", 4.059, 4.547, {"lose": -0.49, "none": -2.10, "win": 4.059}, log_path=test_log_path)
    save_result("LOSE", 5.849, 6.798, {"lose": 5.849, "none": -2.85, "win": -0.95}, log_path=test_log_path)
    print("テスト保存が完了しました。")
    print(f"保存先: {test_log_path}")


if __name__ == "__main__":
    main()
