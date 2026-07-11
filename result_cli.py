from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from result_logger import (
    ARCHIVE_DIR,
    DEFAULT_LOG_PATH,
    add_manual_result,
    clear_all_results,
    undo_last_result,
)
from stats_calculator import get_current_stats


def _path_from_environment(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else fallback


def execute(
    command: str,
    log_path: Path = DEFAULT_LOG_PATH,
    archive_dir: Path = ARCHIVE_DIR,
    backup: bool = True,
) -> dict[str, Any]:
    if command == "get-stats":
        return get_current_stats(log_path)
    if command == "add-win":
        add_manual_result("WIN", log_path=log_path)
        return get_current_stats(log_path)
    if command == "add-lose":
        add_manual_result("LOSE", log_path=log_path)
        return get_current_stats(log_path)
    if command == "undo":
        removed = undo_last_result(log_path)
        return {"removed": removed, "stats": get_current_stats(log_path)}
    if command == "clear":
        backup_path = clear_all_results(log_path, archive_dir, backup=backup)
        return {
            "backup_path": str(backup_path) if backup_path else None,
            "stats": get_current_stats(log_path),
        }
    raise ValueError(f"未対応のコマンドです: {command}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("get-stats", "add-win", "add-lose", "undo", "clear"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=_path_from_environment("TEKKEN8_LOG_PATH", DEFAULT_LOG_PATH),
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=_path_from_environment("TEKKEN8_ARCHIVE_DIR", ARCHIVE_DIR),
    )
    backup_group = parser.add_mutually_exclusive_group()
    backup_group.add_argument("--backup", dest="backup", action="store_true")
    backup_group.add_argument("--no-backup", dest="backup", action="store_false")
    parser.set_defaults(backup=True)
    return parser


def main() -> int:
    try:
        args = create_parser().parse_args()
        response = {
            "ok": True,
            "data": execute(args.command, args.log_path, args.archive_dir, args.backup),
        }
        print(json.dumps(response, ensure_ascii=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"ok": False, "error": str(error)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
