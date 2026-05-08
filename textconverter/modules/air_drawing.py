from datetime import datetime
from pathlib import Path
from string import ascii_uppercase
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    AIR_DRAW_BRUSH_THICKNESS,
    AIR_DRAW_CANVAS_ALPHA,
    AIR_DRAW_ERASER_THICKNESS,
    AIR_DRAW_MIN_POINT_DISTANCE,
    AIR_DRAW_OUTPUT_DIR,
)


class AirDrawingSystem:
    def __init__(self, width: int, height: int, output_dir: Path = AIR_DRAW_OUTPUT_DIR):
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_point: Optional[Tuple[int, int]] = None
        self.mode = "PAUSE"
        self.text = ""
        self.message = "Index up draws. Open palm pauses."

    def update(self, frame, hands):
        if not hands:
            self.last_point = None
            self.mode = "NO HAND"
            return

        hand = hands[0]
        landmarks = hand.landmarks
        point = self._landmark_to_pixel(landmarks[8])

        if self._is_clear_gesture(landmarks):
            self.clear()
            self.message = "Canvas cleared"
            return

        if self._is_draw_gesture(landmarks):
            self.mode = "DRAW"
            self._draw_to(point)
        elif self._is_erase_gesture(landmarks):
            self.mode = "ERASE"
            self._erase_at(point)
        else:
            self.mode = "PAUSE"
            self.last_point = None

    def render(self, frame, fps: float, suggestions=None):
        overlay = cv2.addWeighted(frame, 1.0, self.canvas, AIR_DRAW_CANVAS_ALPHA, 0)
        self._draw_ui(overlay, fps, suggestions or [])
        return overlay

    def clear(self):
        self.canvas[:] = 0
        self.last_point = None

    def backspace(self):
        if self.text:
            self.text = self.text[:-1]
            self.message = "Deleted last character"

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self.message = "Space added"

    def add_character(self, character: str):
        character = character.upper()
        if len(character) == 1 and character in ascii_uppercase:
            self.text += character
            self.clear()
            self.message = f"Added {character}"

    def accept_suggestion(self, suggestion: str):
        if not suggestion:
            return
        words = self.text.rstrip().split(" ")
        if words:
            words[-1] = suggestion.upper()
            self.text = " ".join(words)
        else:
            self.text = suggestion.upper()
        self.message = f"Accepted {suggestion.upper()}"

    def save_canvas(self) -> Optional[Path]:
        if not np.any(self.canvas):
            self.message = "Nothing to save"
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"air_drawing_{timestamp}.png"
        cv2.imwrite(str(path), self.canvas)
        self.message = f"Saved {path.name}"
        return path

    def _draw_to(self, point: Tuple[int, int]):
        if self.last_point is None:
            self.last_point = point
            return
        if self._distance(self.last_point, point) < AIR_DRAW_MIN_POINT_DISTANCE:
            return
        cv2.line(self.canvas, self.last_point, point, (0, 230, 255), AIR_DRAW_BRUSH_THICKNESS, cv2.LINE_AA)
        self.last_point = point

    def _erase_at(self, point: Tuple[int, int]):
        cv2.circle(self.canvas, point, AIR_DRAW_ERASER_THICKNESS, (0, 0, 0), -1)
        self.last_point = None

    def _draw_ui(self, frame, fps: float, suggestions):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 58), (18, 22, 28), -1)
        cv2.putText(frame, "Air Drawing Mode", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
        cv2.putText(frame, f"STATE {self.mode}  FPS {fps:.1f}", (w - 300, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1)

        cv2.rectangle(frame, (20, 78), (410, 245), (25, 30, 38), -1)
        cv2.rectangle(frame, (20, 78), (410, 245), (70, 78, 90), 1)
        lines = [
            "Gesture controls",
            "Index finger up: draw",
            "Index + middle up: erase",
            "Open palm: clear canvas",
            "Closed hand: pause",
            "Keyboard A-Z: add drawn letter",
            "SPACE: add space  BACKSPACE: delete",
            "S: save drawing  C: clear  Q: quit",
        ]
        for index, line in enumerate(lines):
            color = (245, 245, 245) if index == 0 else (210, 210, 210)
            cv2.putText(frame, line, (38, 110 + index * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1)

        cv2.rectangle(frame, (20, h - 120), (w - 20, h - 20), (25, 30, 38), -1)
        cv2.rectangle(frame, (20, h - 120), (w - 20, h - 20), (70, 78, 90), 1)
        cv2.putText(frame, "Text from air drawing", (38, h - 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
        cv2.putText(frame, self.text or "_", (38, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)
        if suggestions:
            cv2.putText(frame, "Suggestions: " + " | ".join(suggestions), (420, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2)
        elif self._current_word():
            cv2.putText(frame, "Suggestions: no match", (420, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2)
        cv2.putText(frame, self.message, (w - 360, h - 86), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 220, 120), 1)

    def _current_word(self) -> str:
        return self.text.rstrip().split(" ")[-1] if self.text.strip() else ""

    def _landmark_to_pixel(self, landmark) -> Tuple[int, int]:
        x = int(np.clip(landmark[0], 0, 1) * self.width)
        y = int(np.clip(landmark[1], 0, 1) * self.height)
        return x, y

    @staticmethod
    def _is_draw_gesture(landmarks) -> bool:
        return (
            AirDrawingSystem._finger_up(landmarks, 8, 6)
            and not AirDrawingSystem._finger_up(landmarks, 12, 10)
            and not AirDrawingSystem._finger_up(landmarks, 16, 14)
            and not AirDrawingSystem._finger_up(landmarks, 20, 18)
        )

    @staticmethod
    def _is_erase_gesture(landmarks) -> bool:
        return (
            AirDrawingSystem._finger_up(landmarks, 8, 6)
            and AirDrawingSystem._finger_up(landmarks, 12, 10)
            and not AirDrawingSystem._finger_up(landmarks, 16, 14)
        )

    @staticmethod
    def _is_clear_gesture(landmarks) -> bool:
        return all(
            AirDrawingSystem._finger_up(landmarks, tip, pip)
            for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
        )

    @staticmethod
    def _finger_up(landmarks, tip_index: int, pip_index: int) -> bool:
        return landmarks[tip_index][1] < landmarks[pip_index][1]

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))
