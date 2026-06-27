import time
from datetime import datetime
from pathlib import Path

from debug_image_saver import (
    save_candidate_roi,
    save_detection_roi,
    save_rejected_roi,
)
from result_logger import save_result
from stats_calculator import calculate_stats, load_match_records, print_stats

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
# 0.15秒なら1秒に約6〜7回判定します。
DETECT_INTERVAL_SECONDS = 0.15

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

# candidates / rejected の保存設定
SAVE_CANDIDATE_IMAGES = True
SAVE_REJECTED_IMAGES = False

# 保存しすぎ防止のため、同じ種類の保存は最低この秒数だけ空ける
DEBUG_IMAGE_SAVE_INTERVAL_SECONDS = 5.0

# candidates / rejected として保存する最低スコア
MIN_CANDIDATE_SAVE_SCORE = 1.5
MIN_REJECTED_SAVE_SCORE = 1.5

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

    _best_label, best_score = score_pairs[0]
    _second_label, second_score = score_pairs[1]

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


def get_score_and_margin(result, label):
    scores = result["scores"]
    label = label.lower()

    score = scores.get(label)

    sorted_scores = sorted(scores.values(), reverse=True)

    if len(sorted_scores) >= 2:
        margin = sorted_scores[0] - sorted_scores[1]
    else:
        margin = 0.0

    return score, margin


# ==============================
# イベント確定ロジック
# ==============================

def update_event_state(result, state):
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
# 表示・保存
# ==============================

def format_scores(scores):
    parts = []

    for label, score in sorted(scores.items()):
        parts.append(f"{label}:{score:.2f}")

    return " ".join(parts)


def print_current_stats():
    """
    CSVに保存済みの試合結果を読み込み、現在の戦績を表示する。
    CSV保存後に呼び出す想定。
    """
    try:
        records, duplicate_rows_skipped = load_match_records()

        if not records:
            print("[STATS] 集計できるWIN/LOSE記録がありません。")
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


def print_detected_event(timestamp, detected_event, result):
    print("")
    print("======================================")
    print(f"[DETECTED] {timestamp} -> {detected_event.upper()}")
    print(f"reason: {result['reason']}")
    print(f"scores: {format_scores(result['scores'])}")
    print("======================================")
    print("")


def save_detection_debug_image(roi, detected_event, score, margin, reason):
    try:
        saved_image_path = save_detection_roi(
            image=roi,
            label=detected_event,
            score=score,
            margin=margin,
            reason=reason,
        )

        print(f"[DEBUG IMAGE] detection ROI saved: {saved_image_path}")

    except Exception as e:
        print(f"[DEBUG IMAGE ERROR] detection ROI保存に失敗しました: {e}")


def save_candidate_debug_image(roi, label, score, margin, reason):
    try:
        saved_image_path = save_candidate_roi(
            image=roi,
            label=label,
            score=score,
            margin=margin,
            reason=reason,
        )

        print(f"[DEBUG IMAGE] candidate ROI saved: {saved_image_path}")

    except Exception as e:
        print(f"[DEBUG IMAGE ERROR] candidate ROI保存に失敗しました: {e}")


def save_rejected_debug_image(roi, label, score, margin, reason):
    try:
        saved_image_path = save_rejected_roi(
            image=roi,
            label=label,
            score=score,
            margin=margin,
            reason=reason,
        )

        print(f"[DEBUG IMAGE] rejected ROI saved: {saved_image_path}")

    except Exception as e:
        print(f"[DEBUG IMAGE ERROR] rejected ROI保存に失敗しました: {e}")


def can_save_debug_image(debug_save_state, key):
    current_time = time.time()
    last_save_time = debug_save_state.get(key, 0.0)

    if current_time - last_save_time < DEBUG_IMAGE_SAVE_INTERVAL_SECONDS:
        return False

    debug_save_state[key] = current_time
    return True


def maybe_save_candidate_or_rejected(
    roi,
    result,
    detected_event,
    state,
    debug_save_state,
):
    """
    DETECTEDにはならなかったが、WIN/LOSEに近いROIを保存する。

    candidates:
        final_label が win/lose だが、detected_event にはならなかった画像。

    rejected:
        raw_label は win/lose だが、score/margin不足で final_label が none になった画像。
    """
    if detected_event is not None:
        return

    # すでに1試合分を検出済みの間は、同じリザルト画面を候補として保存しない
    if state["event_active"]:
        return

    raw_label = result["raw_label"]
    final_label = result["final_label"]
    reason = result["reason"]

    if (
        SAVE_REJECTED_IMAGES
        and raw_label in ["win", "lose"]
        and final_label == "none"
    ):
        score, margin = get_score_and_margin(
            result=result,
            label=raw_label,
        )

        if score is not None and score >= MIN_REJECTED_SAVE_SCORE:
            key = f"rejected_{raw_label}"

            if can_save_debug_image(debug_save_state, key):
                save_rejected_debug_image(
                    roi=roi,
                    label=raw_label,
                    score=score,
                    margin=margin,
                    reason=reason,
                )

        return

    if (
        SAVE_CANDIDATE_IMAGES
        and final_label in ["win", "lose"]
        and state["stable_count"] == 1
    ):
        score, margin = get_score_and_margin(
            result=result,
            label=final_label,
        )

        if score is not None and score >= MIN_CANDIDATE_SAVE_SCORE:
            key = f"candidate_{final_label}"

            if can_save_debug_image(debug_save_state, key):
                save_candidate_debug_image(
                    roi=roi,
                    label=final_label,
                    score=score,
                    margin=margin,
                    reason=reason,
                )


def save_match_result_to_csv(detected_event, score, margin, scores, timestamp):
    try:
        save_result(
            result=detected_event.upper(),
            score=score,
            margin=margin,
            scores=scores,
            timestamp=timestamp,
        )

        print_current_stats()

    except Exception as e:
        print(f"[LOGGER ERROR] CSV保存に失敗しました: {e}")


# ==============================
# プレビュー表示
# ==============================

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


def show_preview(frame, roi_box, result, state):
    if not SHOW_PREVIEW:
        return False

    preview = draw_preview(
        frame=frame,
        roi_box=roi_box,
        result=result,
        state=state,
    )

    preview_resized = resize_for_display(preview)
    cv2.imshow(WINDOW_NAME, preview_resized)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("[INFO] Qキーで終了しました。")
        return True

    return False


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

    debug_save_state = {}

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

                # 前回の検出イベントを次のループに持ち越さない
                detected_event = None
                result = last_result

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

                    maybe_save_candidate_or_rejected(
                        roi=roi,
                        result=result,
                        detected_event=detected_event,
                        state=state,
                        debug_save_state=debug_save_state,
                    )

                    last_result = result
                    last_detect_time = current_time

                if detected_event is not None:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    score, margin = get_score_and_margin(
                        result=result,
                        label=detected_event,
                    )

                    print_detected_event(
                        timestamp=timestamp,
                        detected_event=detected_event,
                        result=result,
                    )

                    save_detection_debug_image(
                        roi=roi,
                        detected_event=detected_event,
                        score=score,
                        margin=margin,
                        reason=result["reason"],
                    )

                    save_match_result_to_csv(
                        detected_event=detected_event,
                        score=score,
                        margin=margin,
                        scores=result["scores"],
                        timestamp=timestamp,
                    )

                should_quit = show_preview(
                    frame=frame,
                    roi_box=roi_box,
                    result=last_result,
                    state=state,
                )

                if should_quit:
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