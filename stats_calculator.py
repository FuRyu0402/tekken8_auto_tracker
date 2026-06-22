from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# ==============================
# 設定
# ==============================

DEFAULT_LOG_PATH = Path("logs") / "match_results.csv"


# ==============================
# データ構造
# ==============================

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
    duplicate_rows_skipped: int


# ==============================
# CSV読み込み
# ==============================

def _to_float(value: str | None) -> float | None:
    """
    CSVの文字列をfloatに変換する。
    空欄や不正な値の場合は None を返す。
    """
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def load_match_records(log_path: Path = DEFAULT_LOG_PATH) -> tuple[list[MatchRecord], int]:
    """
    match_results.csv を読み込み、MatchRecord のリストに変換する。

    同一内容の完全重複行はスキップする。
    これは、検証中に同じ結果が連続保存された場合の集計ズレを避けるため。
    """
    if not log_path.exists():
        raise FileNotFoundError(f"ログファイルが見つかりません: {log_path}")

    records: list[MatchRecord] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    duplicate_rows_skipped = 0

    with log_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = (row.get("timestamp") or "").strip()
            result = (row.get("result") or "").strip().upper()
            score_text = (row.get("score") or "").strip()
            margin_text = (row.get("margin") or "").strip()

            if result not in {"WIN", "LOSE"}:
                continue

            row_key = (
                timestamp,
                result,
                score_text,
                margin_text,
            )

            if row_key in seen_rows:
                duplicate_rows_skipped += 1
                continue

            seen_rows.add(row_key)

            records.append(
                MatchRecord(
                    timestamp=timestamp,
                    result=result,
                    score=_to_float(score_text),
                    margin=_to_float(margin_text),
                )
            )

    return records, duplicate_rows_skipped


# ==============================
# 集計処理
# ==============================

def calculate_stats(
    records: list[MatchRecord],
    duplicate_rows_skipped: int = 0,
) -> MatchStats:
    """
    MatchRecordの一覧から戦績を集計する。
    """
    total_matches = len(records)

    win_count = sum(1 for record in records if record.result == "WIN")
    lose_count = sum(1 for record in records if record.result == "LOSE")

    if total_matches > 0:
        win_rate = win_count / total_matches * 100
    else:
        win_rate = 0.0

    current_streak_result = None
    current_streak_count = 0

    if records:
        current_streak_result = records[-1].result

        for record in reversed(records):
            if record.result == current_streak_result:
                current_streak_count += 1
            else:
                break

    max_win_streak = 0
    max_lose_streak = 0

    current_result = None
    current_count = 0

    for record in records:
        if record.result == current_result:
            current_count += 1
        else:
            current_result = record.result
            current_count = 1

        if current_result == "WIN":
            max_win_streak = max(max_win_streak, current_count)
        elif current_result == "LOSE":
            max_lose_streak = max(max_lose_streak, current_count)

    return MatchStats(
        total_matches=total_matches,
        win_count=win_count,
        lose_count=lose_count,
        win_rate=win_rate,
        current_streak_result=current_streak_result,
        current_streak_count=current_streak_count,
        max_win_streak=max_win_streak,
        max_lose_streak=max_lose_streak,
        duplicate_rows_skipped=duplicate_rows_skipped,
    )


# ==============================
# 表示
# ==============================

def format_current_streak(stats: MatchStats) -> str:
    """
    現在の連勝/連敗を表示用文字列に変換する。
    """
    if stats.current_streak_result is None:
        return "なし"

    if stats.current_streak_result == "WIN":
        return f"{stats.current_streak_count}連勝"

    if stats.current_streak_result == "LOSE":
        return f"{stats.current_streak_count}連敗"

    return "なし"


def print_stats(stats: MatchStats) -> None:
    """
    戦績をPowerShellに表示する。
    """
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

    if stats.duplicate_rows_skipped > 0:
        print("--------------------------------------")
        print(f"重複スキップ: {stats.duplicate_rows_skipped} 行")

    print("======================================")


# ==============================
# メイン処理
# ==============================

def main() -> None:
    log_path = DEFAULT_LOG_PATH

    print(f"[INFO] 読み込み対象: {log_path}")

    try:
        records, duplicate_rows_skipped = load_match_records(log_path)

        if not records:
            print("[WARN] 集計できるWIN/LOSE記録がありません。")
            return

        stats = calculate_stats(
            records=records,
            duplicate_rows_skipped=duplicate_rows_skipped,
        )

        print_stats(stats)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] 先に capture_classifier.py で試合結果を記録してください。")


if __name__ == "__main__":
    main()