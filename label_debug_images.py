import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


# ==============================
# 設定
# ==============================

DEFAULT_SOURCE_DIR = Path("debug_capture") / "detections"
TARGET_ROOT_DIR = Path("debug_capture") / "selected_for_training"

WIN_DIR = TARGET_ROOT_DIR / "win"
LOSE_DIR = TARGET_ROOT_DIR / "lose"
HARD_NONE_DIR = TARGET_ROOT_DIR / "hard_none"
SKIP_DIR = TARGET_ROOT_DIR / "skip"

# True なら元画像を残してコピー
# False なら元画像を移動
COPY_MODE = True

DISPLAY_WIDTH = 1000


# ==============================
# 画像読み込み
# ==============================

def read_image_unicode(path):
    """
    日本語パスでも読み込めるように np.fromfile + cv2.imdecode を使う。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return image


def resize_for_display(image, display_width=1000):
    height, width = image.shape[:2]

    if width <= display_width:
        return image

    scale = display_width / width
    display_height = int(height * scale)

    resized = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def draw_help_text(image, index, total, path):
    preview = image.copy()

    lines = [
        f"{index + 1}/{total}",
        f"file: {path.name}",
        "1 or W: win",
        "2 or L: lose",
        "3 or H or N: hard_none",
        "4 or S: skip",
        "Q: quit",
    ]

    y = 35

    for line in lines:
        cv2.putText(
            preview,
            line,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 35

    return preview


# ==============================
# ファイル処理
# ==============================

def prepare_dirs():
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    LOSE_DIR.mkdir(parents=True, exist_ok=True)
    HARD_NONE_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_DIR.mkdir(parents=True, exist_ok=True)


def collect_image_paths(source_dir):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]

    image_paths = []

    for extension in extensions:
        image_paths.extend(source_dir.glob(extension))

    image_paths.sort()

    return image_paths


def send_file(source_path, target_dir):
    target_path = target_dir / source_path.name

    if COPY_MODE:
        shutil.copy2(source_path, target_path)
    else:
        shutil.move(source_path, target_path)

    return target_path


def label_images(source_dir):
    prepare_dirs()

    image_paths = collect_image_paths(source_dir)

    if not image_paths:
        print(f"[INFO] 画像が見つかりません: {source_dir}")
        return

    print("======================================")
    print("Debug Image Labeler")
    print("======================================")
    print(f"source : {source_dir}")
    print(f"target : {TARGET_ROOT_DIR}")
    print(f"mode   : {'copy' if COPY_MODE else 'move'}")
    print("======================================")
    print("1 or W : win")
    print("2 or L : lose")
    print("3 or H or N : hard_none")
    print("4 or S : skip")
    print("Q : quit")
    print("======================================")

    counts = {
        "win": 0,
        "lose": 0,
        "hard_none": 0,
        "skip": 0,
    }

    window_name = "Label Debug Images"

    for index, image_path in enumerate(image_paths):
        image = read_image_unicode(image_path)

        if image is None:
            print(f"[WARN] 読み込み失敗: {image_path}")
            continue

        preview = resize_for_display(image, DISPLAY_WIDTH)
        preview = draw_help_text(
            image=preview,
            index=index,
            total=len(image_paths),
            path=image_path,
        )

        cv2.imshow(window_name, preview)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in [ord("1"), ord("w")]:
                target_path = send_file(image_path, WIN_DIR)
                counts["win"] += 1
                print(f"[WIN] {image_path.name} -> {target_path}")
                break

            if key in [ord("2"), ord("l")]:
                target_path = send_file(image_path, LOSE_DIR)
                counts["lose"] += 1
                print(f"[LOSE] {image_path.name} -> {target_path}")
                break

            if key in [ord("3"), ord("h"), ord("n")]:
                target_path = send_file(image_path, HARD_NONE_DIR)
                counts["hard_none"] += 1
                print(f"[HARD_NONE] {image_path.name} -> {target_path}")
                break

            if key in [ord("4"), ord("s")]:
                target_path = send_file(image_path, SKIP_DIR)
                counts["skip"] += 1
                print(f"[SKIP] {image_path.name} -> {target_path}")
                break

            if key == ord("q"):
                print("[INFO] Qキーで終了しました。")
                cv2.destroyAllWindows()
                print_summary(counts)
                return

    cv2.destroyAllWindows()
    print_summary(counts)


def print_summary(counts):
    print("")
    print("======================================")
    print("Label Summary")
    print("======================================")
    print(f"win       : {counts['win']}")
    print(f"lose      : {counts['lose']}")
    print(f"hard_none : {counts['hard_none']}")
    print(f"skip      : {counts['skip']}")
    print("======================================")


# ==============================
# メイン
# ==============================

def main():
    if len(sys.argv) >= 2:
        source_dir = Path(sys.argv[1])
    else:
        source_dir = DEFAULT_SOURCE_DIR

    if not source_dir.exists():
        print(f"[ERROR] フォルダが見つかりません: {source_dir}")
        return

    label_images(source_dir)


if __name__ == "__main__":
    main()