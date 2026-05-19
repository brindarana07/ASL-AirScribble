import time

import cv2

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, MIRROR_CAMERA, TARGET_FPS
from utils.helpers import read_key, should_quit


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check that it is connected and not already used by another app."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_count = 0
    fps = 0.0
    last_time = time.time()

    print("Webcam test started.")
    print("Press Q in the camera window to quit.")

    window_name = "Module 1 - Webcam FPS Test"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: could not read a frame from the webcam.")
                continue

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = now

            status = "OK" if fps >= TARGET_FPS else "LOW"
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} ({status})",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if status == "OK" else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press Q or ESC to quit",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            if should_quit(read_key(1), window_name):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Last measured FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
