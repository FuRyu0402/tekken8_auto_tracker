import shutil
from pathlib import Path

import cv2
import pytesseract


def setup_tesseract():
    """
    Tesseract OCR 本体の場所を設定する。
    """

    if shutil.which("tesseract"):
        return

    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for path in possible_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            return


def normalize_text(text):
    """
    OCR結果を判定しやすい形に整える。
    """

    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("|", "I")
    text = text.replace("0", "O")

    return text


def get_roi_candidates(image):
    """
    勝敗表示が出そうな範囲を複数試す。
    鉄拳8の YOU WIN / YOU LOSE が中央付近に出る前提。
    """

    height, width = image.shape[:2]

    roi_settings = [
        (
            "center_wide",
            0.15,
            0.85,
            0.25,
            0.75,
        ),
        (
            "center_middle",
            0.20,
            0.80,
            0.30,
            0.65,
        ),
        (
            "result_upper",
            0.20,
            0.80,
            0.25,
            0.50,
        ),
        (
            "result_center",
            0.20,
            0.80,
            0.35,
            0.60,
        ),
        (
            "result_narrow",
            0.25,
            0.75,
            0.35,
            0.55,
        ),
        (
            "result_lower",
            0.20,
            0.80,
            0.45,
            0.70,
        ),
    ]

    candidates = []

    for name, x1_rate, x2_rate, y1_rate, y2_rate in roi_settings:
        x1 = int(width * x1_rate)
        x2 = int(width * x2_rate)
        y1 = int(height * y1_rate)
        y2 = int(height * y2_rate)

        roi = image[y1:y2, x1:x2]

        candidates.append(
            (
                name,
                roi,
            )
        )

    return candidates


def preprocess_for_ocr(roi):
    """
    OCRしやすいように複数パターンへ加工する。
    """

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    enlarged = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(
        enlarged,
        (3, 3),
        0
    )

    _, otsu_binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu_inverted = cv2.bitwise_not(otsu_binary)

    adaptive_binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    adaptive_inverted = cv2.bitwise_not(adaptive_binary)

    return [
        (
            "gray",
            enlarged,
        ),
        (
            "otsu_binary",
            otsu_binary,
        ),
        (
            "otsu_inverted",
            otsu_inverted,
        ),
        (
            "adaptive_binary",
            adaptive_binary,
        ),
        (
            "adaptive_inverted",
            adaptive_inverted,
        ),
    ]


def judge_text(normalized_text):
    """
    OCR文字列から勝敗を判定する。
    """

    if "YOUWIN" in normalized_text:
        return "WIN"

    if "YOULOSE" in normalized_text:
        return "LOSE"

    if "WIN" in normalized_text:
        return "WIN"

    if "LOSE" in normalized_text:
        return "LOSE"

    return "NONE"


def detect_result_by_ocr(image):
    """
    画像から YOU WIN / YOU LOSE をOCRで読み取る。
    """

    setup_tesseract()

    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    )

    roi_candidates = get_roi_candidates(image)

    for roi_name, roi in roi_candidates:
        roi_path = debug_dir / f"{roi_name}_roi.png"

        cv2.imwrite(
            str(roi_path),
            roi
        )

        processed_images = preprocess_for_ocr(roi)

        for process_name, processed_image in processed_images:
            debug_path = debug_dir / f"{roi_name}_{process_name}.png"

            cv2.imwrite(
                str(debug_path),
                processed_image
            )

            text = pytesseract.image_to_string(
                processed_image,
                lang="eng",
                config=config
            )

            normalized = normalize_text(text)

            print(
                f"[{roi_name} / {process_name}] "
                f"OCR raw: {repr(text)}"
            )

            print(
                f"[{roi_name} / {process_name}] "
                f"OCR normalized: {normalized}"
            )

            result = judge_text(normalized)

            if result != "NONE":
                return result

    return "NONE"