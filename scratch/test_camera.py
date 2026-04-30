"""
Interactive camera test — press Enter to capture, q+Enter to quit.
Images saved as capture_001.jpg, capture_002.jpg, ...
"""
import sys

sys.path.insert(0, "src")

import cv2
from hal.camera import PiCamera

cam = PiCamera()
cam.start()
print("Camera ready. Press Enter to capture, q+Enter to quit.")

count = 0
try:
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        frame = cam.capture_bgr()
        if frame is None:
            print("No frame captured.")
            continue
        count += 1
        path = f"capture_{count:03d}.jpg"
        cv2.imwrite(path, frame)
        print(f"Saved {path}  ({frame.shape[1]}x{frame.shape[0]})")
finally:
    cam.stop()
    print("Camera stopped.")
