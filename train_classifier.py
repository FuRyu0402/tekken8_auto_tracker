import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


# ==============================
# 設定
# ==============================

DATASET_DIR = Path("dataset")

WIN_DIR = DATASET_DIR / "win"
LOSE_DIR = DATASET_DIR / "lose"
NONE_DIR = DATASET_DIR / "none"

# モデルが間違えやすい none 画像を入れる場所
# ここにある画像は、必ず none として学習に使います。
HARD_NONE_DIR = DATASET_DIR / "hard_none"

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "win_lose_none_hog_svm.pkl"

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# 通常 none は多すぎるため、学習時だけ最大枚数を制限します。
# hard_none はこの制限とは別に、必ず全部使います。
MAX_RANDOM_NONE_IMAGES = 120

# 乱数を固定して、毎回同じ条件で学習できるようにします。
RANDOM_SEED = 42

# 学習用に画像をこのサイズへ統一します。
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 128

# テスト用に分ける割合
TEST_SIZE = 0.25


# ==============================
# HOG 設定
# ==============================

HOG_WIN_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
HOG_BLOCK_SIZE = (32, 32)
HOG_BLOCK_STRIDE = (16, 16)
HOG_CELL_SIZE = (16, 16)
HOG_NBINS = 9

hog = cv2.HOGDescriptor(
    HOG_WIN_SIZE,
    HOG_BLOCK_SIZE,
    HOG_BLOCK_STRIDE,
    HOG_CELL_SIZE,
    HOG_NBINS,
)


# ==============================
# ファイル収集
# ==============================

def get_image_files(directory):
    if not directory.exists():
        return []

    files = []

    for extension in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(f"*{extension}"))

    return sorted(files)


def collect_dataset():
    win_files = get_image_files(WIN_DIR)
    lose_files = get_image_files(LOSE_DIR)
    none_files = get_image_files(NONE_DIR)
    hard_none_files = get_image_files(HARD_NONE_DIR)

    random.seed(RANDOM_SEED)

    if len(none_files) > MAX_RANDOM_NONE_IMAGES:
        random_none_files = random.sample(none_files, MAX_RANDOM_NONE_IMAGES)
    else:
        random_none_files = none_files

    dataset = []

    for file_path in win_files:
        dataset.append((file_path, "win"))

    for file_path in lose_files:
        dataset.append((file_path, "lose"))

    for file_path in random_none_files:
        dataset.append((file_path, "none"))

    for file_path in hard_none_files:
        dataset.append((file_path, "none"))

    return dataset, {
        "win": len(win_files),
        "lose": len(lose_files),
        "none_random_used": len(random_none_files),
        "none_total": len(none_files),
        "hard_none": len(hard_none_files),
        "none_used_total": len(random_none_files) + len(hard_none_files),
    }


# ==============================
# 画像処理
# ==============================

def load_image(file_path):
    image = cv2.imread(str(file_path))

    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {file_path}")

    return image


def preprocess_image(image):
    resized = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 明るいステージ・暗いステージの差を少し吸収します。
    gray = cv2.equalizeHist(gray)

    return gray


def extract_hog_feature(image):
    processed = preprocess_image(image)
    feature = hog.compute(processed)

    if feature is None:
        raise ValueError("HOG特徴量を作成できませんでした。")

    return feature.flatten()


def build_features(dataset):
    features = []
    labels = []

    failed_files = []

    for file_path, label in dataset:
        try:
            image = load_image(file_path)
            feature = extract_hog_feature(image)

            features.append(feature)
            labels.append(label)

        except Exception as error:
            print(f"[WARN] スキップ: {file_path}")
            print(f"       理由: {error}")
            failed_files.append(file_path)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    return X, y, failed_files


# ==============================
# 学習
# ==============================

def validate_counts(counts):
    print("======================================")
    print("Dataset Counts")
    print("======================================")
    print(f"win              : {counts['win']} 枚")
    print(f"lose             : {counts['lose']} 枚")
    print(f"none random used : {counts['none_random_used']} 枚")
    print(f"hard none        : {counts['hard_none']} 枚")
    print(f"none used total  : {counts['none_used_total']} 枚")
    print(f"none total       : {counts['none_total']} 枚")
    print("======================================")
    print("")

    if counts["win"] < 10:
        raise ValueError("win が少なすぎます。最低10枚以上が必要です。")

    if counts["lose"] < 10:
        raise ValueError("lose が少なすぎます。最低10枚以上が必要です。")

    if counts["none_used_total"] < 10:
        raise ValueError("none が少なすぎます。最低10枚以上が必要です。")


def train_model(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=10000,
        ),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    class_names = label_encoder.classes_

    print("======================================")
    print("Evaluation")
    print("======================================")
    print("")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("")
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    return model, label_encoder


def save_model(model, label_encoder, counts):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "hog_win_size": HOG_WIN_SIZE,
        "hog_block_size": HOG_BLOCK_SIZE,
        "hog_block_stride": HOG_BLOCK_STRIDE,
        "hog_cell_size": HOG_CELL_SIZE,
        "hog_nbins": HOG_NBINS,
        "max_random_none_images": MAX_RANDOM_NONE_IMAGES,
        "random_seed": RANDOM_SEED,
        "counts": counts,
    }

    joblib.dump(payload, MODEL_PATH)

    print("")
    print("======================================")
    print("Model Saved")
    print("======================================")
    print(f"保存先: {MODEL_PATH}")
    print("======================================")


# ==============================
# メイン処理
# ==============================

def main():
    print("======================================")
    print("Tekken8 WIN / LOSE / NONE Trainer")
    print("======================================")
    print("")

    dataset, counts = collect_dataset()
    validate_counts(counts)

    print("[INFO] HOG特徴量を作成中...")
    X, y, failed_files = build_features(dataset)

    print("")
    print("======================================")
    print("Feature Build Result")
    print("======================================")
    print(f"使用画像数: {len(X)} 枚")
    print(f"失敗画像数: {len(failed_files)} 枚")
    print(f"特徴量次元: {X.shape[1] if len(X) > 0 else 0}")
    print("======================================")
    print("")

    if len(X) == 0:
        raise ValueError("学習に使える画像がありません。")

    print("[INFO] 学習を開始します...")
    model, label_encoder = train_model(X, y)

    save_model(model, label_encoder, counts)

    print("")
    print("[INFO] 学習完了です。")


if __name__ == "__main__":
    main()