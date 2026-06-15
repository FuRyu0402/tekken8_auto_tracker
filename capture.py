import os
from datetime import datetime

import cv2
import numpy as np
import mss


def capture_screen():
    with mss.mss() as sct:

        print(sct.monitors)

        monitor = sct.monitors[2]

        screenshot = sct.grab(monitor)

        img = np.array(screenshot)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img


def crop_center(img):
    height, width = img.shape[:2]

    crop_width = 700
    crop_height = 250

    x = (width - crop_width) // 2
    y = (height - crop_height) // 2

    cropped = img[
        y:y + crop_height,
        x:x + crop_width
    ]

    return cropped


def save_screenshot(img):
    os.makedirs("screenshots", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"screenshots/{timestamp}.png"

    cv2.imwrite(filename, img)

    print(f"保存完了: {filename}")


if __name__ == "__main__":
    image = capture_screen()

    image = crop_center(image)

    save_screenshot(image)