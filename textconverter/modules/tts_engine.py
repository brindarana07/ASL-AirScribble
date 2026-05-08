import queue
import threading
from typing import Optional

import pyttsx3


class TTSEngine:
    def __init__(self, enabled: bool = True, rate: int = 165, volume: float = 1.0, voice_index: int = 0):
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        self.clear_queue()
        try:
            self._queue.put_nowait(text.strip())
        except queue.Full:
            pass

    def clear_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self.clear_queue()
            self._queue.put_nowait(None)

    def _worker(self) -> None:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if voices:
            voice_index = min(self.voice_index, len(voices) - 1)
            engine.setProperty("voice", voices[voice_index].id)
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                self._queue.task_done()
        engine.stop()
