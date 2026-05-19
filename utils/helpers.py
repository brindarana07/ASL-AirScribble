import time

import cv2


class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()

    def update(self) -> float:
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now
        return self.fps


def read_key(delay_ms: int = 1) -> int:
    key = cv2.waitKeyEx(delay_ms)
    return key & 0xFF if key >= 0 else -1


def should_quit(key: int, window_name: str) -> bool:
    if key in (ord("q"), ord("Q"), 27):
        return True
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
