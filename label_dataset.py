from pathlib import Path
import shutil

import cv2


# ==============================
# 設定
# ==============================

RAW_DIR = Path("dataset") / "raw"
WIN_DIR = Path("dataset") / "win"
LOSE_DIR = Path("dataset") / "lose"
NONE_DIR = Path("dataset") / "none"

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

WINDOW_NAME = "Tekken8 Dataset Labeler"


# ==============================
# 初期準備
# ==============================

def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    LOSE_DIR.mkdir(parents=True, exist_ok=True)
    NONE_DIR.mkdir(parents=True, exist_ok=True)


def get_image_files():
    files = []

    for extension in SUPPORTED_EXTENSIONS:
        files.extend(RAW_DIR.glob(f"*{extension}"))

    return sorted(files)


def make_unique_path(target_dir, source_path):
    target_path = target_dir / source_path.name

    if not target_path.exists():
        return target_path

    stem = source_path.stem
    suffix = source_path.suffix

    index = 1

    while True:
        new_path = target_dir / f"{stem}_{index}{suffix}"

        if not new_path.exists():
            return new_path

        index += 1


def move_image(source_path, target_dir):
    target_path = make_unique_path(target_dir, source_path)
    shutil.move(str(source_path), str(target_path))
    return target_path


# ==============================
# 表示処理
# ==============================

def resize_for_display(image, max_width=1200, max_height=800):
    height, width = image.shape[:2]

    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height, 1.0)

    display_width = int(width * scale)
    display_height = int(height * scale)

    resized = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def draw_text(image, text, position, font_scale=0.8, thickness=2):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def create_display_image(image, current_index, total_count, image_path):
    display = resize_for_display(image)

    draw_text(
        display,
        f"{current_index + 1} / {total_count}",
        (20, 35),
    )

    draw_text(
        display,
        "1/W: WIN   2/L: LOSE   3/N: NONE   S: SKIP   Q: QUIT",
        (20, 75),
        font_scale=0.65,
        thickness=2,
    )

    draw_text(
        display,
        image_path.name,
        (20, 115),
        font_scale=0.55,
        thickness=1,
    )

    return display


# ==============================
# メイン処理
# ==============================

def main():
    ensure_directories()

    image_files = get_image_files()

    if not image_files:
        print("[INFO] dataset/raw に画像がありません。")
        return

    print("======================================")
    print("Tekken8 Dataset Labeler")
    print("======================================")
    print(f"未分類画像: {len(image_files)} 枚")
    print("")
    print("キー操作:")
    print("  1 または W : win に移動")
    print("  2 または L : lose に移動")
    print("  3 または N : none に移動")
    print("  S          : 保留して次へ")
    print("  Q          : 終了")
    print("======================================")
    print("")

    current_index = 0
    skipped_files = []

    while current_index < len(image_files):
        image_path = image_files[current_index]

        if not image_path.exists():
            current_index += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] 読み込めませんでした: {image_path}")
            current_index += 1
            continue

        display = create_display_image(
            image=image,
            current_index=current_index,
            total_count=len(image_files),
            image_path=image_path,
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0) & 0xFF

        if key in [ord("1"), ord("w")]:
            target_path = move_image(image_path, WIN_DIR)
            print(f"[WIN ] {image_path.name} -> {target_path}")
            current_index += 1

        elif key in [ord("2"), ord("l")]:
            target_path = move_image(image_path, LOSE_DIR)
            print(f"[LOSE] {image_path.name} -> {target_path}")
            current_index += 1

        elif key in [ord("3"), ord("n")]:
            target_path = move_image(image_path, NONE_DIR)
            print(f"[NONE] {image_path.name} -> {target_path}")
            current_index += 1

        elif key == ord("s"):
            print(f"[SKIP] {image_path.name}")
            skipped_files.append(image_path)
            current_index += 1

        elif key == ord("q"):
            print("[INFO] Qキーで終了しました。")
            break

        else:
            print("[INFO] 未対応キーです。1/W, 2/L, 3/N, S, Q のいずれかを使用します。")

    cv2.destroyAllWindows()

    print("")
    print("======================================")
    print("分類結果")
    print("======================================")
    print(f"win : {len(list(WIN_DIR.glob('*')))} 枚")
    print(f"lose: {len(list(LOSE_DIR.glob('*')))} 枚")
    print(f"none: {len(list(NONE_DIR.glob('*')))} 枚")
    print(f"skip: {len(skipped_files)} 枚")
    print("======================================")


if __name__ == "__main__":
    main()