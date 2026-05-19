from typing import Optional

import cv2


class Display:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.last_event_text = ""

    def draw(
        self,
        frame,
        current_prediction: str,
        stable_label: str,
        confidence: float,
        current_word: str,
        sentence: str,
        mode: str,
        fps: float,
        model_ready: bool,
        event_text: Optional[str] = None,
        stability_progress: float = 0.0,
        auto_speak_words: bool = False,
        hand_count: int = 0,
        sample_count: int = 0,
        suggestions=None,
    ):
        if event_text:
            self.last_event_text = event_text

        suggestions = suggestions or []
        self._draw_top_bar(frame, fps, model_ready, mode, auto_speak_words, hand_count, sample_count)
        self._draw_prediction_panel(frame, current_prediction, stable_label, confidence, stability_progress)
        self._draw_text_panel(frame, current_word, sentence, suggestions)
        self._draw_guide_panel(frame)
        return frame

    def _draw_top_bar(
        self,
        frame,
        fps: float,
        model_ready: bool,
        mode: str,
        auto_speak_words: bool,
        hand_count: int,
        sample_count: int,
    ):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 56), (18, 22, 28), -1)
        cv2.putText(frame, "Offline ASL to Text", (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2)
        x = w - 700
        self._draw_chip(frame, x, 16, "MODEL", "ON" if model_ready else "OFF", model_ready)
        self._draw_chip(frame, x + 130, 16, "HAND", str(hand_count), hand_count > 0)
        self._draw_chip(frame, x + 250, 16, "TTS", "AUTO" if auto_speak_words else "MANUAL", auto_speak_words)
        cv2.putText(frame, f"{mode.upper()}  FPS {fps:.1f}  SAMPLES {sample_count}", (x + 420, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    def _draw_prediction_panel(
        self,
        frame,
        current_prediction: str,
        stable_label: str,
        confidence: float,
        stability_progress: float,
    ):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = 20, 76, 390, 300
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 30, 38), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 78, 90), 1)
        cv2.putText(frame, "Prediction", (x1 + 18, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1)
        label = current_prediction or "IDLE"
        font_scale = 3.1 if len(label) <= 2 else 1.4
        cv2.putText(frame, label, (x1 + 22, y1 + 112), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4)
        cv2.putText(frame, f"Stable: {stable_label}", (x1 + 18, y2 - 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1)
        self._draw_bar(frame, x1 + 18, y2 - 68, x2 - 18, y2 - 52, confidence, (0, 210, 120), "Confidence")
        self._draw_bar(frame, x1 + 18, y2 - 34, x2 - 18, y2 - 18, stability_progress, (0, 180, 255), "Hold")

    def _draw_text_panel(self, frame, current_word: str, sentence: str, suggestions):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = 20, h - 190, w - 20, h - 20
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 30, 38), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 78, 90), 1)
        cv2.putText(frame, "Current word", (x1 + 18, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
        cv2.putText(frame, current_word or "_", (x1 + 18, y1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 220, 180), 2)
        if suggestions:
            cv2.putText(frame, "Suggestions: " + " | ".join(suggestions), (x1 + 360, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2)
        elif current_word:
            cv2.putText(frame, "Suggestions: no match", (x1 + 360, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2)
        cv2.putText(frame, "Final sentence", (x1 + 18, y1 + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
        self._put_wrapped_text(frame, sentence or "_", x1 + 18, y1 + 150, x2 - x1 - 36)
        if self.last_event_text:
            cv2.putText(frame, self.last_event_text, (x2 - 300, y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

    def _draw_guide_panel(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = w - 370, 76, w - 20, 340
        cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 30, 38), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 78, 90), 1)
        lines = [
            "Letter Mode",
            "Sign A-Z to spell",
            "TAB: accept suggestion",
            "SPACE: commit word",
            "BACKSPACE: delete",
            "ENTER: finalize + speak",
            "C: clear",
            "H: history   T: auto speech",
            "Q: quit",
        ]
        for index, line in enumerate(lines):
            color = (245, 245, 245) if index == 0 else (210, 210, 210)
            cv2.putText(frame, line, (x1 + 18, y1 + 32 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    @staticmethod
    def _put_wrapped_text(frame, text: str, x: int, y: int, max_width: int):
        words = text.split(" ")
        line = ""
        line_y = y
        for word in words:
            candidate = word if not line else f"{line} {word}"
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0]
            if width > max_width and line:
                cv2.putText(frame, line, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
                line = word
                line_y += 30
            else:
                line = candidate
        if line:
            cv2.putText(frame, line, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)

    @staticmethod
    def _draw_bar(frame, x1: int, y1: int, x2: int, y2: int, value: float, color, label: str):
        value = max(0.0, min(1.0, value))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 62, 72), -1)
        fill_x = int(x1 + (x2 - x1) * value)
        cv2.rectangle(frame, (x1, y1), (fill_x, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 118, 130), 1)
        cv2.putText(frame, f"{label} {value:.0%}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1)

    @staticmethod
    def _draw_chip(frame, x: int, y: int, label: str, value: str, active: bool, active_color=None):
        color = active_color if active and active_color else (0, 185, 110) if active else (0, 140, 255)
        cv2.rectangle(frame, (x, y), (x + 112, y + 26), (35, 40, 48), -1)
        cv2.rectangle(frame, (x, y), (x + 112, y + 26), color, 1)
        cv2.circle(frame, (x + 12, y + 13), 5, color, -1)
        cv2.putText(frame, f"{label}:{value}", (x + 24, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
