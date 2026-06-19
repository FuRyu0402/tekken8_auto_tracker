import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np


# ==============================
# 設定
# ==============================

# 以前の環境では sct.monitors[2] が鉄拳8の画面でした。
# 違う画面が映る場合は 1 や 3 に変更します。
MONITOR_INDEX = 2

# 何秒ごとにROI画像を保存するか
# WIN / LOSE 表示は数秒なので、最初は 1.0 秒が安全です。
SAVE_INTERVAL_SECONDS = 0.25

# 最大保存枚数
# 容量が増えすぎないように上限を付けます。
MAX_IMAGES = 500

# ROIの大きさ
# WIN / LOSE の文字周辺に寄せた設定です。
# 1920x1080の場合、おおよそ 1056x345 前後になります。
ROI_WIDTH_RATIO = 0.55
ROI_HEIGHT_RATIO = 0.32

# ROIの上下位置調整
# 0.00 が中央。正の値で下、負の値で上に移動します。
ROI_Y_OFFSET_RATIO = 0.02

# 保存先
RAW_DIR = Path("dataset") / "raw"

# プレビュー画面を表示するか
SHOW_PREVIEW = True


# ==============================
# 初期準備
# ==============================

def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (Path("dataset") / "win").mkdir(parents=True, exist_ok=True)
    (Path("dataset") / "lose").mkdir(parents=True, exist_ok=True)
    (Path("dataset") / "none").mkdir(parents=True, exist_ok=True)


def get_monitor(sct):
    monitors = sct.monitors

    print("======================================")
    print("利用可能なモニター一覧")
    print("======================================")

    for index, monitor in enumerate(monitors):
        print(f"[{index}] {monitor}")

    print("======================================")

    if MONITOR_INDEX < len(monitors):
        print(f"[INFO] MONITOR_INDEX={MONITOR_INDEX} を使用します。")
        return monitors[MONITOR_INDEX]

    print(f"[WARN] MONITOR_INDEX={MONITOR_INDEX} が見つかりません。")
    print("[WARN] 代わりに MONITOR_INDEX=1 を使用します。")
    return monitors[1]


# ==============================
# 画像処理
# ==============================

def crop_center_roi(frame):
    height, width = frame.shape[:2]

    roi_width = int(width * ROI_WIDTH_RATIO)
    roi_height = int(height * ROI_HEIGHT_RATIO)

    center_x = width // 2
    center_y = height // 2 + int(height * ROI_Y_OFFSET_RATIO)

    x1 = center_x - roi_width // 2
    y1 = center_y - roi_height // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    roi = frame[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2)


def make_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return RAW_DIR / f"roi_{timestamp}.png"


def draw_preview(frame, roi_box, saved_count):
    preview = frame.copy()

    x1, y1, x2, y2 = roi_box

    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        preview,
        f"Saved: {saved_count} / {MAX_IMAGES}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "Press Q to quit",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return preview


def save_roi(roi):
    save_path = make_filename()
    cv2.imwrite(str(save_path), roi)
    return save_path


# ==============================
# メイン処理
# ==============================

def main():
    ensure_directories()

    print("")
    print("======================================")
    print("Tekken8 Dataset Collector")
    print("======================================")
    print(f"保存先: {RAW_DIR}")
    print(f"保存間隔: {SAVE_INTERVAL_SECONDS} 秒")
    print(f"最大保存枚数: {MAX_IMAGES}")
    print(f"ROI幅比率: {ROI_WIDTH_RATIO}")
    print(f"ROI高さ比率: {ROI_HEIGHT_RATIO}")
    print("")
    print("終了方法:")
    print("  プレビュー画面を選択して Q キー")
    print("  または PowerShell で Ctrl + C")
    print("======================================")
    print("")

    saved_count = 0
    last_save_time = 0.0

    try:
        with mss.mss() as sct:
            monitor = get_monitor(sct)

            while True:
                screenshot = sct.grab(monitor)

                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                roi, roi_box = crop_center_roi(frame)

                current_time = time.time()

                if current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
                    save_path = save_roi(roi)
                    saved_count += 1
                    last_save_time = current_time

                    print(f"[SAVE] {saved_count:03d}: {save_path}")

                if SHOW_PREVIEW:
                    preview = draw_preview(frame, roi_box, saved_count)

                    preview_height, preview_width = preview.shape[:2]

                    display_width = 960
                    scale = display_width / preview_width
                    display_height = int(preview_height * scale)

                    preview_resized = cv2.resize(
                        preview,
                        (display_width, display_height),
                        interpolation=cv2.INTER_AREA,
                    )

                    cv2.imshow("dataset_collector - ROI Preview", preview_resized)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("[INFO] Qキーで終了しました。")
                        break

                if saved_count >= MAX_IMAGES:
                    print("[INFO] 最大保存枚数に達したため終了しました。")
                    break

    except KeyboardInterrupt:
        print("")
        print("[INFO] Ctrl + C で終了しました。")

    finally:
        cv2.destroyAllWindows()
        print("")
        print("======================================")
        print(f"保存枚数: {saved_count}")
        print("完了しました。")
        print("======================================")


if __name__ == "__main__":
    main()