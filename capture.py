import time

import cv2
import numpy as np
import mss

from edge_detector import get_edge_scores
from edge_detector import judge_result


def capture_screen():
    with mss.MSS() as sct:

        monitor = sct.monitors[2]

        screenshot = sct.grab(monitor)

        img = np.array(screenshot)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGRA2BGR
        )

        return img


if __name__ == "__main__":

    print("monitoring started")

    win_count = 0
    lose_count = 0

    last_count_time = 0
    cooldown_seconds = 8

    while True:

        image = capture_screen()

        scores = get_edge_scores(image)

        win_score = scores["win_score"]
        lose_score = scores["lose_score"]
        win_template = scores["win_template"]
        lose_template = scores["lose_template"]

        result = judge_result(
            win_score,
            lose_score
        )

        max_score = max(
            win_score,
            lose_score
        )

        if max_score >= 0.20 or result != "NONE":
            print(
                f"WIN: {win_score:.3f} ({win_template}) "
                f"LOSE: {lose_score:.3f} ({lose_template}) "
                f"=> {result}"
            )

        now = time.time()

        if result != "NONE" and now - last_count_time >= cooldown_seconds:

            if result == "WIN":
                win_count += 1

            elif result == "LOSE":
                lose_count += 1

            last_count_time = now

            print(
                f"COUNTED {result} | "
                f"WIN: {win_count} "
                f"LOSE: {lose_count}"
            )

        time.sleep(0.2)