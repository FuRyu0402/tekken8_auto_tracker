import cv2


def detect_result(image_path):
    screenshot = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    win_template = cv2.imread(
        "templates/win.png",
        cv2.IMREAD_GRAYSCALE
    )

    lose_template = cv2.imread(
        "templates/lose.png",
        cv2.IMREAD_GRAYSCALE
    )

    print("screenshot:", screenshot.shape if screenshot is not None else None)
    print("win:", win_template.shape if win_template is not None else None)
    print("lose:", lose_template.shape if lose_template is not None else None)

    win_result = cv2.matchTemplate(
        screenshot,
        win_template,
        cv2.TM_CCOEFF_NORMED
    )

    lose_result = cv2.matchTemplate(
        screenshot,
        lose_template,
        cv2.TM_CCOEFF_NORMED
    ) 

    win_score = win_result.max()
    lose_score = lose_result.max()

    print(f"WIN score: {win_score:.3f}")
    print(f"LOSE score: {lose_score:.3f}")

    threshold = 0.7

    if win_score >= threshold:
        return "WIN"

    if lose_score >= threshold:
        return "LOSE"

    return "NONE"


if __name__ == "__main__":
    result = detect_result(
        "screenshots/test.png"
    )

    print(result)