import time

import cv2
import numpy as np
import mss

from detector import detect_result


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

    previous_result = "NONE"

    win_count = 0
    lose_count = 0

    while True:

        image = capture_screen()

        result = detect_result(image)

        if result != "NONE" and result != previous_result:

            if result == "WIN":
                win_count += 1

            elif result == "LOSE":
                lose_count += 1

            print(
                f"{result} | "
                f"WIN: {win_count} "
                f"LOSE: {lose_count}"
            )

        previous_result = result

        time.sleep(0.2)