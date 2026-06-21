import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import mss
import numpy as np


# ==============================
# 設定
# ==============================

MODEL_PATH = Path("models") / "win_lose_none_hog_svm.pkl"

# 以前の環境では sct.monitors[2] が鉄拳8の画面でした。
# 違う画面が映る場合は 1 や 3 に変更します。
MONITOR_INDEX = 2

# 判定間隔
# 0.25秒なら1秒に4回判定します。
DETECT_INTERVAL_SECONDS = 0.20

# ROI設定
# dataset_collector.py と同じ設定にします。
ROI_WIDTH_RATIO = 0.55
ROI_HEIGHT_RATIO = 0.32
ROI_Y_OFFSET_RATIO = 0.02

# WIN / LOSE の誤判定を減らすための安全判定
USE_SAFE_THRESHOLD = True
MIN_WIN_LOSE_SCORE = 1.5
MIN_WIN_LOSE_MARGIN = 1.5

# 何回連続で同じ WIN / LOSE が出たら確定扱いにするか
# 誤判定対策として、1回だけでは確定しません。
STABLE_REQUIRED_COUNT = 2

# NONE が何回続いたら、次の勝敗検出を許可するか
NONE_RESET_REQUIRED_COUNT = 3

# 同じ勝敗画面で連続記録しないための最低クールダウン秒数
EVENT_COOLDOWN_SECONDS = 8.0

# プレビューを表示するか
SHOW_PREVIEW = True

WINDOW_NAME = "Tekken8 Capture Classifier"


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
# 画面キャプチャ
# ==============================

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


# ==============================
# 画像処理・分類
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


def apply_safe_threshold(raw_label, class_names, scores):
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


def classify_roi(roi, payload, hog):
    model = payload["model"]
    label_encoder = payload["label_encoder"]

    image_width = int(payload["image_width"])
    image_height = int(payload["image_height"])

    feature = extract_hog_feature(
        image=roi,
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
        "raw_label": raw_label,
        "final_label": final_label,
        "reason": reason,
        "scores": score_dict,
    }


# ==============================
# イベント確定ロジック
# ==============================

def update_event_state(
    result,
    state,
):
    final_label = result["final_label"]
    current_time = time.time()

    detected_event = None

    if final_label in ["win", "lose"]:
        state["none_count"] = 0

        if state["stable_label"] == final_label:
            state["stable_count"] += 1
        else:
            state["stable_label"] = final_label
            state["stable_count"] = 1

        cooldown_ok = (
            current_time - state["last_event_time"]
            >= EVENT_COOLDOWN_SECONDS
        )

        if (
            state["stable_count"] >= STABLE_REQUIRED_COUNT
            and not state["event_active"]
            and cooldown_ok
        ):
            detected_event = final_label
            state["event_active"] = True
            state["last_event_time"] = current_time

    else:
        state["none_count"] += 1

        if state["none_count"] >= NONE_RESET_REQUIRED_COUNT:
            state["stable_label"] = None
            state["stable_count"] = 0
            state["event_active"] = False

    return detected_event, state


# ==============================
# 表示
# ==============================

def format_scores(scores):
    parts = []

    for label, score in sorted(scores.items()):
        parts.append(f"{label}:{score:.2f}")

    return " ".join(parts)


def draw_preview(frame, roi_box, result, state):
    preview = frame.copy()

    x1, y1, x2, y2 = roi_box

    final_label = result["final_label"]
    raw_label = result["raw_label"]

    if final_label == "win":
        color = (0, 255, 0)
    elif final_label == "lose":
        color = (0, 0, 255)
    else:
        color = (255, 255, 255)

    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

    line1 = f"raw={raw_label} final={final_label}"
    line2 = f"stable={state['stable_label']} count={state['stable_count']} active={state['event_active']}"
    line3 = result["reason"]
    line4 = format_scores(result["scores"])
    line5 = "Press Q to quit"

    lines = [line1, line2, line3, line4, line5]

    y = 35

    for line in lines:
        cv2.putText(
            preview,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 35

    return preview


def resize_for_display(image, display_width=960):
    height, width = image.shape[:2]

    scale = display_width / width
    display_height = int(height * scale)

    resized = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


# ==============================
# メイン処理
# ==============================

def main():
    print("======================================")
    print("Tekken8 Capture Classifier")
    print("======================================")
    print(f"モデル: {MODEL_PATH}")
    print(f"判定間隔: {DETECT_INTERVAL_SECONDS} 秒")
    print(f"安定判定回数: {STABLE_REQUIRED_COUNT}")
    print(f"クールダウン: {EVENT_COOLDOWN_SECONDS} 秒")
    print("終了方法: プレビュー画面で Q、または PowerShell で Ctrl + C")
    print("======================================")
    print("")

    payload = load_model()
    hog = create_hog(payload)

    state = {
        "stable_label": None,
        "stable_count": 0,
        "none_count": 0,
        "event_active": False,
        "last_event_time": 0.0,
    }

    last_detect_time = 0.0
    last_result = {
        "raw_label": "none",
        "final_label": "none",
        "reason": "not_started",
        "scores": {},
    }

    try:
        with mss.mss() as sct:
            monitor = get_monitor(sct)

            while True:
                screenshot = sct.grab(monitor)

                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                roi, roi_box = crop_center_roi(frame)

                current_time = time.time()

                if current_time - last_detect_time >= DETECT_INTERVAL_SECONDS:
                    result = classify_roi(
                        roi=roi,
                        payload=payload,
                        hog=hog,
                    )

                    detected_event, state = update_event_state(
                        result=result,
                        state=state,
                    )

                    last_result = result
                    last_detect_time = current_time

                    if detected_event is not None:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print("")
                        print("======================================")
                        print(f"[DETECTED] {timestamp} -> {detected_event.upper()}")
                        print(f"reason: {result['reason']}")
                        print(f"scores: {format_scores(result['scores'])}")
                        print("======================================")
                        print("")

                if SHOW_PREVIEW:
                    preview = draw_preview(
                        frame=frame,
                        roi_box=roi_box,
                        result=last_result,
                        state=state,
                    )

                    preview_resized = resize_for_display(preview)
                    cv2.imshow(WINDOW_NAME, preview_resized)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("[INFO] Qキーで終了しました。")
                        break

    except KeyboardInterrupt:
        print("")
        print("[INFO] Ctrl + C で終了しました。")

    finally:
        cv2.destroyAllWindows()
        print("")
        print("[INFO] 終了しました。")


if __name__ == "__main__":
    main()