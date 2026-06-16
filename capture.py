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

    while True:

        image = capture_screen()

        result = detect_result(image)

        print(result)

        time.sleep(1)