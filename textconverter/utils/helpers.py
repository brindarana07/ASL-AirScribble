import time


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
