from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from result_logger import DEFAULT_LOG_PATH, read_valid_rows


@dataclass
class MatchRecord:
    timestamp: str
    result: str
    score: float | None = None
    margin: float | None = None


@dataclass
class MatchStats:
    total_matches: int
    win_count: int
    lose_count: int
    win_rate: float
    current_streak_result: str | None
    current_streak_count: int
    max_win_streak: int
    max_lose_streak: int
    # Kept for callers that use the old interface. Duplicates are no longer skipped.
    duplicate_rows_skipped: int = 0


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value and value.strip() else None
    except (TypeError, ValueError):
        return None


def load_match_records(log_path: Path = DEFAULT_LOG_PATH) -> tuple[list[MatchRecord], int]:
    """Load valid matches safely. The second value is retained for compatibility."""
    records = [
        MatchRecord(
            timestamp=row["timestamp"],
            result=row["result"],
            score=_to_float(row["score"]),
            margin=_to_float(row["margin"]),
        )
        for row in read_valid_rows(Path(log_path))
    ]
    return records, 0


def calculate_stats(records: list[MatchRecord], duplicate_rows_skipped: int = 0) -> MatchStats:
    total_matches = len(records)
    win_count = sum(record.result == "WIN" for record in records)
    lose_count = sum(record.result == "LOSE" for record in records)

    current_streak_result = records[-1].result if records else None
    current_streak_count = 0
    for record in reversed(records):
        if record.result != current_streak_result:
            break
        current_streak_count += 1

    max_win_streak = max_lose_streak = 0
    previous_result: str | None = None
    streak_count = 0
    for record in records:
        streak_count = streak_count + 1 if record.result == previous_result else 1
        previous_result = record.result
        if record.result == "WIN":
            max_win_streak = max(max_win_streak, streak_count)
        else:
            max_lose_streak = max(max_lose_streak, streak_count)

    return MatchStats(
        total_matches=total_matches,
        win_count=win_count,
        lose_count=lose_count,
        win_rate=(win_count / total_matches * 100) if total_matches else 0.0,
        current_streak_result=current_streak_result,
        current_streak_count=current_streak_count,
        max_win_streak=max_win_streak,
        max_lose_streak=max_lose_streak,
        duplicate_rows_skipped=0,
    )


def get_current_stats(log_path: Path = DEFAULT_LOG_PATH) -> dict[str, Any]:
    """Return JSON-serializable totals and the latest ten valid matches."""
    records, _ = load_match_records(log_path)
    stats = calculate_stats(records)
    return {
        "total_matches": stats.total_matches,
        "win_count": stats.win_count,
        "lose_count": stats.lose_count,
        "win_rate": stats.win_rate,
        "recent_matches": [asdict(record) for record in records[-10:]],
    }


def format_current_streak(stats: MatchStats) -> str:
    if stats.current_streak_result is None:
        return "なし"
    if stats.current_streak_result == "WIN":
        return f"{stats.current_streak_count}連勝"
    if stats.current_streak_result == "LOSE":
        return f"{stats.current_streak_count}連敗"
    return "なし"


def print_stats(stats: MatchStats) -> None:
    print("======================================")
    print("Match Stats")
    print("======================================")
    print(f"総試合数: {stats.total_matches}")
    print(f"WIN     : {stats.win_count}")
    print(f"LOSE    : {stats.lose_count}")
    print(f"勝率    : {stats.win_rate:.1f}%")
    print(f"現在    : {format_current_streak(stats)}")
    print(f"最大連勝: {stats.max_win_streak}")
    print(f"最大連敗: {stats.max_lose_streak}")
    print("======================================")


def main() -> None:
    log_path = DEFAULT_LOG_PATH
    print(f"[INFO] 読み込み対象: {log_path}")
    records, skipped = load_match_records(log_path)
    if not records:
        print("[WARN] 集計できるWIN/LOSE記録がありません。")
        return
    print_stats(calculate_stats(records, skipped))


if __name__ == "__main__":
    main()
