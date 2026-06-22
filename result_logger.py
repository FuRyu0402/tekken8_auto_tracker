from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


# 保存先
LOG_DIR = Path("logs")
DEFAULT_LOG_PATH = LOG_DIR / "match_results.csv"


# CSVの列
FIELDNAMES = [
    "timestamp",
    "result",
    "score",
    "margin",
    "lose_score",
    "none_score",
    "win_score",
]


def _format_float(value: Optional[float]) -> str:
    """
    floatをCSV保存しやすい文字列に変換する。
    Noneの場合は空文字にする。
    """
    if value is None:
        return ""
    return f"{value:.3f}"


def save_result(
    result: str,
    score: float,
    margin: float,
    scores: Optional[dict[str, float]] = None,
    timestamp: Optional[str] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """
    WIN / LOSE の検出結果をCSVに追記する。

    Parameters
    ----------
    result:
        "WIN" または "LOSE"
    score:
        採用された判定ラベルのスコア
    margin:
        1位と2位のスコア差
    scores:
        {"lose": ..., "none": ..., "win": ...} のような各ラベルのスコア
    timestamp:
        指定しない場合は現在時刻を自動で入れる
    log_path:
        保存先CSVファイル
    """
    result = result.upper()

    if result not in {"WIN", "LOSE"}:
        raise ValueError(f"result must be 'WIN' or 'LOSE'. got: {result}")

    if scores is None:
        scores = {}

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_path.exists()

    row = {
        "timestamp": timestamp,
        "result": result,
        "score": _format_float(score),
        "margin": _format_float(margin),
        "lose_score": _format_float(scores.get("lose")),
        "none_score": _format_float(scores.get("none")),
        "win_score": _format_float(scores.get("win")),
    }

    with log_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    """
    result_logger.py 単体の動作確認用。
    logs/test_match_results.csv にテスト記録を書き込む。
    """
    test_log_path = LOG_DIR / "test_match_results.csv"

    save_result(
        result="WIN",
        score=4.059,
        margin=4.547,
        scores={
            "lose": -0.49,
            "none": -2.10,
            "win": 4.059,
        },
        log_path=test_log_path,
    )

    save_result(
        result="LOSE",
        score=5.849,
        margin=6.798,
        scores={
            "lose": 5.849,
            "none": -2.85,
            "win": -0.95,
        },
        log_path=test_log_path,
    )

    print("テスト保存が完了しました。")
    print(f"保存先: {test_log_path}")


if __name__ == "__main__":
    main()