import time

import cv2

from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_MAX_NUM_HANDS,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    MIRROR_CAMERA,
    TARGET_FPS,
)
from modules.hand_tracker import HAND_LANDMARK_NAMES, HandTracker


def draw_status(frame, tracked_hands, fps):
    color = (0, 255, 0) if fps >= TARGET_FPS else (0, 165, 255)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    if not tracked_hands:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for index, hand in enumerate(tracked_hands):
        y = 85 + index * 35
        cv2.putText(
            frame,
            f"{hand.handedness} hand | confidence: {hand.confidence:.2f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def print_landmark_reference():
    print("MediaPipe hand landmark indices:")
    for index, name in HAND_LANDMARK_NAMES.items():
        print(f"{index:>2}: {name}")
    print()


def main():
    print_landmark_reference()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Close other camera apps and check CAMERA_INDEX in config.py."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker(
        max_num_hands=MEDIAPIPE_MAX_NUM_HANDS,
        detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
    )

    frame_count = 0
    fps = 0.0
    last_time = time.time()

    print("Hand landmark test started. Press Q in the camera window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: could not read a frame from the webcam.")
                continue

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            tracked_hands = tracker.process_frame(frame)
            tracker.draw_landmarks(frame)

            frame_count += 1
            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = now

            draw_status(frame, tracked_hands, fps)
            cv2.imshow("Module 2 - MediaPipe Hand Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
