import cv2


def detect_result(screenshot):

    if len(screenshot.shape) == 3:
        screenshot = cv2.cvtColor(
            screenshot,
            cv2.COLOR_BGR2GRAY
        )

    win_template = cv2.imread(
        "templates/win.png",
        cv2.IMREAD_GRAYSCALE
    )

    lose_template = cv2.imread(
        "templates/lose.png",
        cv2.IMREAD_GRAYSCALE
    )

    if win_template is None:
        print("templates/win.png が見つかりません")
        return "NONE"

    if lose_template is None:
        print("templates/lose.png が見つかりません")
        return "NONE"

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

    threshold = 0.7

    if win_score >= threshold:
        return "WIN"

    if lose_score >= threshold:
        return "LOSE"

    return "NONE"