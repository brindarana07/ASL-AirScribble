from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import ascii_uppercase
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np

from config import (
    AIR_DRAW_IMAGE_SIZE,
    AIR_DRAW_MODEL_PATH,
    AIR_DRAW_RECOGNITION_CONFIDENCE,
    AIR_DRAW_SAMPLE_DIR,
)


@dataclass
class AirDrawPrediction:
    label: str
    confidence: float


class AirDrawRecognizer:
    def __init__(
        self,
        model_path: Path = AIR_DRAW_MODEL_PATH,
        confidence_threshold: float = AIR_DRAW_RECOGNITION_CONFIDENCE,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            print(f"Warning: could not load air drawing model from {self.model_path}: {exc}")
            self.model = None
            return
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1

    def predict(self, canvas: np.ndarray) -> Optional[AirDrawPrediction]:
        if self.model is None:
            return None

        try:
            features = canvas_to_feature(canvas).reshape(1, -1)
        except ValueError:
            return AirDrawPrediction(label="?", confidence=0.0)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))
            label = str(self.model.classes_[best_index])
            confidence = float(probabilities[best_index])
        else:
            label = str(self.model.predict(features)[0])
            confidence = 1.0

        if confidence < self.confidence_threshold:
            return AirDrawPrediction(label="?", confidence=confidence)
        return AirDrawPrediction(label=label, confidence=confidence)


def canvas_to_feature(canvas: np.ndarray, image_size: int = AIR_DRAW_IMAGE_SIZE) -> np.ndarray:
    normalized = normalize_canvas(canvas, image_size)
    return (normalized.astype(np.float32) / 255.0).flatten()


def normalize_canvas(canvas: np.ndarray, image_size: int = AIR_DRAW_IMAGE_SIZE) -> np.ndarray:
    if canvas.ndim == 3:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas.copy()

    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("Canvas is empty. Draw a letter first.")

    x, y, w, h = cv2.boundingRect(points)
    cropped = binary[y : y + h, x : x + w]
    side = max(w, h) + 12
    square = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = cropped

    return cv2.resize(square, (image_size, image_size), interpolation=cv2.INTER_AREA)


def save_air_draw_sample(canvas: np.ndarray, label: str, sample_dir: Path = AIR_DRAW_SAMPLE_DIR) -> Path:
    label = label.upper()
    if label not in ascii_uppercase:
        raise ValueError("Air drawing samples must use labels A-Z.")

    normalized = normalize_canvas(canvas)
    label_dir = Path(sample_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = label_dir / f"{label}_{timestamp}.png"
    cv2.imwrite(str(path), normalized)
    return path


def load_air_draw_samples(sample_dir: Path = AIR_DRAW_SAMPLE_DIR) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    sample_dir = Path(sample_dir)

    for label_dir in sorted(sample_dir.iterdir()) if sample_dir.exists() else []:
        if not label_dir.is_dir():
            continue
        label = label_dir.name.upper()
        if label not in ascii_uppercase:
            continue
        for image_path in sorted(label_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (AIR_DRAW_IMAGE_SIZE, AIR_DRAW_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            features.append((image.astype(np.float32) / 255.0).flatten())
            labels.append(label)

    if not features:
        raise ValueError(f"No air drawing samples found in {sample_dir}.")
    return np.array(features, dtype=np.float32), np.array(labels)
