import cv2

from detector import detect_result


image = cv2.imread(
    "screenshots/test2.jpg"
)

result = detect_result(image)

print("結果:", result)