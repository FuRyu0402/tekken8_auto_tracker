from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ==============================
# 設定
# ==============================

DEBUG_CAPTURE_DIR = Path("debug_capture")

CATEGORY_DETECTIONS = "detections"
CATEGORY_CANDIDATES = "candidates"
CATEGORY_REJECTED = "rejected"

VALID_CATEGORIES = {
    CATEGORY_DETECTIONS,
    CATEGORY_CANDIDATES,
    CATEGORY_REJECTED,
}


# ==============================
# 補助関数
# ==============================

def _format_float(value: Optional[float]) -> str:
    """
    ファイル名に入れやすい形にfloatを変換する。
    """
    if value is None:
        return "none"

    return f"{value:.3f}"


def _safe_text(text: str) -> str:
    """
    ファイル名に使いやすい文字列に変換する。
    """
    safe = text.strip().lower()
    safe = safe.replace(" ", "_")
    safe = safe.replace(":", "-")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")

    if safe == "":
        safe = "unknown"

    return safe


def _create_timestamp() -> str:
    """
    ミリ秒・マイクロ秒まで含めたタイムスタンプを作る。
    同じ秒に複数保存してもファイル名が被りにくい。
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _ensure_bgr_image(image: np.ndarray) -> np.ndarray:
    """
    保存しやすいBGR画像に整える。
    """
    if image is None:
        raise ValueError("image is None です。")

    if not isinstance(image, np.ndarray):
        raise TypeError("image は numpy.ndarray である必要があります。")

    if image.size == 0:
        raise ValueError("image が空です。")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(image.shape) == 3 and image.shape[2] == 3:
        return image

    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"対応していない画像形式です: shape={image.shape}")


# ==============================
# ROI保存
# ==============================

def save_roi_image(
    image: np.ndarray,
    category: str,
    label: str,
    score: Optional[float] = None,
    margin: Optional[float] = None,
    reason: str = "",
    base_dir: Path = DEBUG_CAPTURE_DIR,
) -> Path:
    """
    ROI画像を debug_capture 配下に保存する。

    Parameters
    ----------
    image:
        保存するROI画像
    category:
        "detections", "candidates", "rejected" のいずれか
    label:
        "win", "lose", "none" などの判定ラベル
    score:
        判定スコア
    margin:
        1位と2位のスコア差
    reason:
        判定理由
    base_dir:
        保存先の親フォルダ

    Returns
    -------
    Path
        保存した画像ファイルのパス
    """
    category = _safe_text(category)

    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"category は {sorted(VALID_CATEGORIES)} のいずれかにしてください。got: {category}"
        )

    label = _safe_text(label)
    reason = _safe_text(reason)

    bgr_image = _ensure_bgr_image(image)

    save_dir = base_dir / category
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _create_timestamp()
    score_text = _format_float(score)
    margin_text = _format_float(margin)

    filename = (
        f"{timestamp}"
        f"_{category}"
        f"_{label}"
        f"_score-{score_text}"
        f"_margin-{margin_text}"
    )

    if reason:
        filename += f"_{reason}"

    filename += ".png"

    save_path = save_dir / filename

    success = cv2.imwrite(str(save_path), bgr_image)

    if not success:
        raise RuntimeError(f"画像保存に失敗しました: {save_path}")

    return save_path


def save_detection_roi(
    image: np.ndarray,
    label: str,
    score: Optional[float],
    margin: Optional[float],
    reason: str = "",
) -> Path:
    """
    実際にDETECTEDされたROIを保存する。
    """
    return save_roi_image(
        image=image,
        category=CATEGORY_DETECTIONS,
        label=label,
        score=score,
        margin=margin,
        reason=reason,
    )


def save_candidate_roi(
    image: np.ndarray,
    label: str,
    score: Optional[float],
    margin: Optional[float],
    reason: str = "",
) -> Path:
    """
    WIN/LOSE候補だったが、記録確定には至らなかったROIを保存する。
    """
    return save_roi_image(
        image=image,
        category=CATEGORY_CANDIDATES,
        label=label,
        score=score,
        margin=margin,
        reason=reason,
    )


def save_rejected_roi(
    image: np.ndarray,
    label: str,
    score: Optional[float],
    margin: Optional[float],
    reason: str = "",
) -> Path:
    """
    rawではWIN/LOSEっぽかったが、安全判定でnoneに落とされたROIを保存する。
    """
    return save_roi_image(
        image=image,
        category=CATEGORY_REJECTED,
        label=label,
        score=score,
        margin=margin,
        reason=reason,
    )


# ==============================
# 単体テスト
# ==============================

def _create_test_image(text: str) -> np.ndarray:
    """
    debug_image_saver.py 単体テスト用のダミー画像を作る。
    """
    image = np.zeros((240, 640, 3), dtype=np.uint8)

    cv2.putText(
        image,
        text,
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    return image


def main() -> None:
    """
    単体動作確認。
    debug_capture 配下にテスト画像を保存する。
    """
    detection_image = _create_test_image("DETECTION")
    candidate_image = _create_test_image("CANDIDATE")
    rejected_image = _create_test_image("REJECTED")

    detection_path = save_detection_roi(
        image=detection_image,
        label="win",
        score=4.123,
        margin=5.456,
        reason="test_detection",
    )

    candidate_path = save_candidate_roi(
        image=candidate_image,
        label="lose",
        score=2.345,
        margin=2.789,
        reason="test_candidate",
    )

    rejected_path = save_rejected_roi(
        image=rejected_image,
        label="win",
        score=1.234,
        margin=0.987,
        reason="test_rejected",
    )

    print("======================================")
    print("debug_image_saver.py test")
    print("======================================")
    print(f"detection: {detection_path}")
    print(f"candidate : {candidate_path}")
    print(f"rejected  : {rejected_path}")
    print("======================================")


if __name__ == "__main__":
    main()