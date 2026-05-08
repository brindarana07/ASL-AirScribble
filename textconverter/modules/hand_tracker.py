from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


HAND_LANDMARK_NAMES = {
    0: "Wrist",
    1: "Thumb CMC",
    2: "Thumb MCP",
    3: "Thumb IP",
    4: "Thumb Tip",
    5: "Index MCP",
    6: "Index PIP",
    7: "Index DIP",
    8: "Index Tip",
    9: "Middle MCP",
    10: "Middle PIP",
    11: "Middle DIP",
    12: "Middle Tip",
    13: "Ring MCP",
    14: "Ring PIP",
    15: "Ring DIP",
    16: "Ring Tip",
    17: "Pinky MCP",
    18: "Pinky PIP",
    19: "Pinky DIP",
    20: "Pinky Tip",
}


@dataclass
class TrackedHand:
    landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    handedness: str
    confidence: float


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._last_results = None

    def process_frame(self, frame_bgr) -> List[TrackedHand]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        self._last_results = results

        if not results.multi_hand_landmarks:
            return []

        tracked_hands = []
        handedness_list = results.multi_handedness or []

        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = np.array(
                [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
                dtype=np.float32,
            )
            handedness, confidence = self._read_handedness(handedness_list, index)
            tracked_hands.append(
                TrackedHand(
                    landmarks=landmarks,
                    normalized_landmarks=self.normalize_landmarks(landmarks),
                    handedness=handedness,
                    confidence=confidence,
                )
            )

        return tracked_hands

    def draw_landmarks(self, frame_bgr) -> None:
        if not self._last_results or not self._last_results.multi_hand_landmarks:
            return

        for hand_landmarks in self._last_results.multi_hand_landmarks:
            self._mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self._hands.close()

    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        wrist = landmarks[0]
        relative_landmarks = landmarks - wrist
        max_distance = np.max(np.linalg.norm(relative_landmarks, axis=1))

        if max_distance <= 1e-6:
            return relative_landmarks

        return relative_landmarks / max_distance

    @staticmethod
    def _read_handedness(handedness_list, index: int) -> Tuple[str, float]:
        if index >= len(handedness_list):
            return "Unknown", 0.0

        classification = handedness_list[index].classification[0]
        return classification.label, float(classification.score)


def get_landmark_name(index: int) -> Optional[str]:
    return HAND_LANDMARK_NAMES.get(index)
