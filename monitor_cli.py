from __future__ import annotations

import json
import sys
from typing import Any

import mss


OPTIONAL_FIELDS = ("name", "is_primary", "unique_id")


def format_monitors(monitors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for index, monitor in enumerate(monitors[1:], start=1):
        item = {
            "index": index,
            "left": int(monitor["left"]),
            "top": int(monitor["top"]),
            "width": int(monitor["width"]),
            "height": int(monitor["height"]),
        }
        for field in OPTIONAL_FIELDS:
            if field in monitor:
                item[field] = monitor[field]
        formatted.append(item)
    return formatted


def main() -> int:
    try:
        with mss.MSS() as sct:
            data = format_monitors(sct.monitors)
        print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
