from datetime import datetime

from result_logger import save_result
from stats_calculator import calculate_stats, load_match_records, print_stats


def add_manual_result(result: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_result(
        result=result,
        score=None,
        margin=None,
        scores={
            "lose": None,
            "none": None,
            "win": None,
        },
        timestamp=timestamp,
    )

    print(f"[MANUAL] {timestamp} -> {result} を追加しました。")


def show_stats() -> None:
    try:
        records, duplicate_rows_skipped = load_match_records()

        if not records:
            print("[STATS] まだWIN/LOSE記録がありません。")
            return

        stats = calculate_stats(
            records=records,
            duplicate_rows_skipped=duplicate_rows_skipped,
        )

        print_stats(stats)

    except FileNotFoundError:
        print("[STATS ERROR] match_results.csv が見つかりません。")
    except Exception as e:
        print(f"[STATS ERROR] 戦績集計に失敗しました: {e}")


def main() -> None:
    print("======================================")
    print("Tekken8 Manual Result Input")
    print("======================================")
    print("1: WINを追加")
    print("2: LOSEを追加")
    print("3: 戦績表示")
    print("q: 終了")
    print("======================================")

    while True:
        command = input("> ").strip().lower()

        if command == "1":
            add_manual_result("WIN")
            show_stats()

        elif command == "2":
            add_manual_result("LOSE")
            show_stats()

        elif command == "3":
            show_stats()

        elif command == "q":
            print("[INFO] 終了しました。")
            break

        else:
            print("[ERROR] 1 / 2 / 3 / q のいずれかを入力してください。")


if __name__ == "__main__":
    main()