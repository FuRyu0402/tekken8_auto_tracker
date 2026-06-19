import argparse
import shutil
from pathlib import Path

import cv2
import joblib
import numpy as np


# ==============================
# 設定
# ==============================

MODEL_PATH = Path("models") / "win_lose_none_hog_svm.pkl"

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# WIN / LOSE の誤判定を減らすための安全判定
# True の場合、スコアが弱い win / lose は none に落とします。
USE_SAFE_THRESHOLD = True

# win / lose と判定するための最低スコア
# 低すぎると薄い残像も拾います。
# 高すぎると取りこぼします。
MIN_WIN_LOSE_SCORE = 0.0

# 1位スコアと2位スコアの差
# 差が小さい場合は曖昧なので none にします。
MIN_WIN_LOSE_MARGIN = 0.5

# 誤判定画像をコピーして確認しやすくする
SAVE_MISCLASSIFIED = True
MISCLASSIFIED_DIR = Path("debug_classifier") / "misclassified"


# ==============================
# モデル読み込み
# ==============================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")

    payload = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "label_encoder",
        "image_width",
        "image_height",
        "hog_win_size",
        "hog_block_size",
        "hog_block_stride",
        "hog_cell_size",
        "hog_nbins",
    ]

    for key in required_keys:
        if key not in payload:
            raise KeyError(f"モデルファイルに必要な情報がありません: {key}")

    return payload


def create_hog(payload):
    hog = cv2.HOGDescriptor(
        tuple(payload["hog_win_size"]),
        tuple(payload["hog_block_size"]),
        tuple(payload["hog_block_stride"]),
        tuple(payload["hog_cell_size"]),
        int(payload["hog_nbins"]),
    )

    return hog


# ==============================
# 画像処理
# ==============================

def preprocess_image(image, image_width, image_height):
    resized = cv2.resize(
        image,
        (image_width, image_height),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    return gray


def extract_hog_feature(image, hog, image_width, image_height):
    processed = preprocess_image(image, image_width, image_height)
    feature = hog.compute(processed)

    if feature is None:
        raise ValueError("HOG特徴量を作成できませんでした。")

    return feature.flatten().astype(np.float32)


# ==============================
# 判定処理
# ==============================

def apply_safe_threshold(raw_label, class_names, scores):
    """
    LinearSVC の decision_function スコアを見て、
    曖昧な win / lose を none に落とします。
    """

    if not USE_SAFE_THRESHOLD:
        return raw_label, "threshold_disabled"

    if raw_label == "none":
        return "none", "raw_none"

    score_pairs = []

    for class_name, score in zip(class_names, scores):
        score_pairs.append((class_name, float(score)))

    score_pairs.sort(key=lambda x: x[1], reverse=True)

    best_label, best_score = score_pairs[0]
    second_label, second_score = score_pairs[1]

    margin = best_score - second_score

    if raw_label in ["win", "lose"]:
        if best_score < MIN_WIN_LOSE_SCORE:
            return "none", f"low_score:{best_score:.3f}"

        if margin < MIN_WIN_LOSE_MARGIN:
            return "none", f"low_margin:{margin:.3f}"

    return raw_label, f"accepted:score={best_score:.3f},margin={margin:.3f}"


def classify_image(image_path, payload, hog):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    model = payload["model"]
    label_encoder = payload["label_encoder"]

    image_width = int(payload["image_width"])
    image_height = int(payload["image_height"])

    feature = extract_hog_feature(
        image=image,
        hog=hog,
        image_width=image_width,
        image_height=image_height,
    )

    X = np.array([feature], dtype=np.float32)

    raw_pred_encoded = model.predict(X)[0]
    raw_label = label_encoder.inverse_transform([raw_pred_encoded])[0]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)[0]
    else:
        scores = np.zeros(len(label_encoder.classes_), dtype=np.float32)

    class_names = list(label_encoder.classes_)

    final_label, reason = apply_safe_threshold(
        raw_label=raw_label,
        class_names=class_names,
        scores=scores,
    )

    score_dict = {
        class_name: float(score)
        for class_name, score in zip(class_names, scores)
    }

    return {
        "image_path": image_path,
        "raw_label": raw_label,
        "final_label": final_label,
        "reason": reason,
        "scores": score_dict,
    }


# ==============================
# 対象ファイル収集
# ==============================

def is_image_file(path):
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_target_files(target_path):
    if target_path.is_file():
        if is_image_file(target_path):
            return [target_path]

        raise ValueError(f"対応していないファイル形式です: {target_path}")

    if target_path.is_dir():
        files = []

        for extension in SUPPORTED_EXTENSIONS:
            files.extend(target_path.glob(f"*{extension}"))

        return sorted(files)

    raise FileNotFoundError(f"対象が見つかりません: {target_path}")


def get_true_label_from_path(image_path):
    parent_name = image_path.parent.name.lower()

    if parent_name in ["win", "lose", "none"]:
        return parent_name

    return None


def copy_misclassified(image_path, true_label, final_label):
    if not SAVE_MISCLASSIFIED:
        return

    if true_label is None:
        return

    if true_label == final_label:
        return

    MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    target_name = f"true-{true_label}_pred-{final_label}_{image_path.name}"
    target_path = MISCLASSIFIED_DIR / target_name

    shutil.copy2(str(image_path), str(target_path))


# ==============================
# 表示
# ==============================

def print_result(result, true_label=None):
    image_path = result["image_path"]
    raw_label = result["raw_label"]
    final_label = result["final_label"]
    reason = result["reason"]
    scores = result["scores"]

    true_text = true_label if true_label is not None else "-"

    score_text = ", ".join(
        [f"{label}:{score:.3f}" for label, score in scores.items()]
    )

    print(
        f"{image_path.name} | "
        f"true={true_text} | "
        f"raw={raw_label} | "
        f"final={final_label} | "
        f"{reason} | "
        f"{score_text}"
    )


def print_summary(results):
    total = len(results)

    if total == 0:
        print("[INFO] 対象画像がありません。")
        return

    final_counts = {}
    raw_counts = {}

    correct = 0
    known_label_count = 0

    for item in results:
        true_label = item["true_label"]
        result = item["result"]

        raw_label = result["raw_label"]
        final_label = result["final_label"]

        raw_counts[raw_label] = raw_counts.get(raw_label, 0) + 1
        final_counts[final_label] = final_counts.get(final_label, 0) + 1

        if true_label is not None:
            known_label_count += 1

            if true_label == final_label:
                correct += 1

    print("")
    print("======================================")
    print("Summary")
    print("======================================")
    print(f"対象画像数: {total}")
    print("")
    print("Raw prediction counts:")
    for label, count in sorted(raw_counts.items()):
        print(f"  {label}: {count}")
    print("")
    print("Final prediction counts:")
    for label, count in sorted(final_counts.items()):
        print(f"  {label}: {count}")

    if known_label_count > 0:
        accuracy = correct / known_label_count
        print("")
        print(f"正解ラベルあり: {known_label_count}")
        print(f"正解数: {correct}")
        print(f"Accuracy: {accuracy:.3f}")

    print("======================================")


# ==============================
# メイン処理
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="Tekken8 WIN / LOSE / NONE classifier detector"
    )

    parser.add_argument(
        "target",
        help="判定する画像ファイル、または画像フォルダ",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="フォルダ判定時に処理する最大枚数",
    )

    args = parser.parse_args()

    target_path = Path(args.target)

    payload = load_model()
    hog = create_hog(payload)

    files = collect_target_files(target_path)

    if args.limit is not None:
        files = files[:args.limit]

    print("======================================")
    print("Tekken8 Classifier Detector")
    print("======================================")
    print(f"モデル: {MODEL_PATH}")
    print(f"対象: {target_path}")
    print(f"画像数: {len(files)}")
    print("")
    print(f"USE_SAFE_THRESHOLD: {USE_SAFE_THRESHOLD}")
    print(f"MIN_WIN_LOSE_SCORE: {MIN_WIN_LOSE_SCORE}")
    print(f"MIN_WIN_LOSE_MARGIN: {MIN_WIN_LOSE_MARGIN}")
    print("======================================")
    print("")

    results = []

    for image_path in files:
        try:
            result = classify_image(
                image_path=image_path,
                payload=payload,
                hog=hog,
            )

            true_label = get_true_label_from_path(image_path)

            print_result(result, true_label=true_label)

            copy_misclassified(
                image_path=image_path,
                true_label=true_label,
                final_label=result["final_label"],
            )

            results.append(
                {
                    "true_label": true_label,
                    "result": result,
                }
            )

        except Exception as error:
            print(f"[WARN] スキップ: {image_path}")
            print(f"       理由: {error}")

    print_summary(results)


if __name__ == "__main__":
    main()