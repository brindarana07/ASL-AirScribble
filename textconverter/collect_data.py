import time
from string import ascii_uppercase

import cv2

from config import (
    CAMERA_INDEX,
    COLLECTION_RECOMMENDED_SAMPLES_PER_CLASS,
    COLLECTION_SAVE_COOLDOWN_SECONDS,
    FEATURE_INCLUDE_ANGLES,
    FEATURE_INCLUDE_DISTANCES,
    FEATURE_INCLUDE_FINGER_FLAGS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LANDMARK_CSV_PATH,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_MAX_NUM_HANDS,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    MIRROR_CAMERA,
    TARGET_FPS,
)
from modules.feature_extractor import append_sample, extract_feature_vector, feature_header, load_label_counts
from modules.hand_tracker import HandTracker
from utils.helpers import FPSCounter


SPECIAL_KEYS = {
    ord("0"): "SPACE",
    ord("1"): "DELETE",
    ord("2"): "NOTHING",
}


def draw_collection_ui(frame, counts, capture_label, saved_message, hand_count, fps):
    h, w = frame.shape[:2]
    panel_w = 500
    cv2.rectangle(frame, (0, 0), (panel_w, h), (18, 22, 28), -1)
    cv2.rectangle(frame, (panel_w, 0), (panel_w + 2, h), (65, 72, 84), -1)

    cv2.putText(frame, "Training Data Collector", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    _draw_chip(frame, 20, 55, "HAND", str(hand_count), hand_count > 0)
    _draw_chip(frame, 142, 55, "FPS", f"{fps:.0f}", fps >= TARGET_FPS)
    _draw_chip(frame, 264, 55, "TARGET", str(COLLECTION_RECOMMENDED_SAMPLES_PER_CLASS), True, (0, 180, 255))

    _draw_active_card(frame, counts, capture_label, saved_message)
    _draw_shortcuts(frame)
    _draw_counts_grid(frame, "Letters", list(ascii_uppercase), counts, 20, 270, 6)
    _draw_counts_grid(frame, "Special", ["SPACE", "DELETE", "NOTHING"], counts, 20, 470, 1)

    hint = "Press a label key once, hold the sign, press ESC to stop."
    cv2.putText(frame, hint, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)


def _draw_active_card(frame, counts, capture_label, message):
    x1, y1, x2, y2 = 20, 92, 480, 168
    active = bool(capture_label)
    border = (0, 220, 120) if active else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (28, 34, 43), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
    label = capture_label or "IDLE"
    count = counts.get(capture_label, 0) if capture_label else 0
    cv2.putText(frame, "ACTIVE LABEL", (x1 + 14, y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (195, 200, 210), 1)
    cv2.putText(frame, label, (x1 + 14, y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
    if capture_label:
        _draw_progress_bar(frame, x1 + 190, y1 + 34, x2 - 18, y1 + 52, count)
        cv2.putText(frame, f"{count}/{COLLECTION_RECOMMENDED_SAMPLES_PER_CLASS}", (x1 + 190, y1 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
    else:
        cv2.putText(frame, message, (x1 + 190, y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)


def _draw_shortcuts(frame):
    lines = [
        "Keys",
        "A-Z letters       0 SPACE",
        "1 DELETE          2 NOTHING",
        "ESC stop          Q quit",
        "Q quit",
    ]
    x1, y1, x2, y2 = 20, 178, 480, 258
    cv2.rectangle(frame, (x1, y1), (x2, y2), (28, 34, 43), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 78, 90), 1)
    for index, line in enumerate(lines):
        color = (245, 245, 245) if index == 0 else (210, 210, 210)
        cv2.putText(frame, line, (x1 + 12, y1 + 18 + index * 9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)


def _draw_counts_grid(frame, title, labels, counts, x, y, columns):
    cv2.putText(frame, title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
    cell_w = 76 if columns > 1 else 210
    cell_h = 28
    for index, label in enumerate(labels):
        col = index % columns
        row = index // columns
        x1 = x + col * cell_w
        y1 = y + row * cell_h
        count = counts.get(label, 0)
        ready = count >= COLLECTION_RECOMMENDED_SAMPLES_PER_CLASS
        color = (0, 180, 105) if ready else (70, 78, 90)
        text_color = (235, 255, 240) if ready else (230, 230, 230)
        cv2.rectangle(frame, (x1, y1), (x1 + cell_w - 6, y1 + 22), (30, 36, 45), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + cell_w - 6, y1 + 22), color, 1)
        short = label[:5]
        cv2.putText(frame, f"{short}:{count}", (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)


def _draw_progress_bar(frame, x1, y1, x2, y2, count):
    ratio = min(1.0, count / COLLECTION_RECOMMENDED_SAMPLES_PER_CLASS)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 62, 72), -1)
    fill_x = int(x1 + (x2 - x1) * ratio)
    color = (0, 220, 120) if ratio >= 1 else (0, 180, 255)
    cv2.rectangle(frame, (x1, y1), (fill_x, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (115, 122, 134), 1)


def _draw_chip(frame, x, y, label, value, active, active_color=None):
    color = active_color if active and active_color else (0, 185, 110) if active else (0, 140, 255)
    cv2.rectangle(frame, (x, y), (x + 108, y + 26), (35, 40, 48), -1)
    cv2.rectangle(frame, (x, y), (x + 108, y + 26), color, 1)
    cv2.circle(frame, (x + 12, y + 13), 5, color, -1)
    cv2.putText(frame, f"{label}:{value}", (x + 24, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Close other camera apps and try again.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker = HandTracker(
        max_num_hands=MEDIAPIPE_MAX_NUM_HANDS,
        detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
    )
    header = feature_header(FEATURE_INCLUDE_DISTANCES, FEATURE_INCLUDE_ANGLES, FEATURE_INCLUDE_FINGER_FLAGS)
    counts = load_label_counts(LANDMARK_CSV_PATH)
    last_save_time = 0.0
    capture_label = ""
    message = "Show one hand clearly."
    fps_counter = FPSCounter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            hands = tracker.process_frame(frame)
            tracker.draw_landmarks(frame)
            fps = fps_counter.update()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 27:
                capture_label = ""
                message = "Capture stopped."

            label = None
            if ord("a") <= key <= ord("z"):
                label = chr(key).upper()
            elif ord("A") <= key <= ord("Z"):
                label = chr(key).upper()
            elif key in SPECIAL_KEYS:
                label = SPECIAL_KEYS[key]

            if label:
                capture_label = label
                message = f"Hold {label}. Capturing automatically."

            now = time.time()
            if capture_label and now - last_save_time >= COLLECTION_SAVE_COOLDOWN_SECONDS:
                if hands:
                    features = extract_feature_vector(
                        hands[0].landmarks,
                        FEATURE_INCLUDE_DISTANCES,
                        FEATURE_INCLUDE_ANGLES,
                        FEATURE_INCLUDE_FINGER_FLAGS,
                    )
                    append_sample(LANDMARK_CSV_PATH, capture_label, features, header)
                    counts[capture_label] = counts.get(capture_label, 0) + 1
                    message = f"{capture_label} saved: {counts[capture_label]}"
                    last_save_time = now
                else:
                    message = "No hand detected. Sample not saved."

            draw_collection_ui(frame, counts, capture_label, message, len(hands), fps)
            cv2.imshow("ASL Data Collection", frame)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
