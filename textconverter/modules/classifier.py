from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


@dataclass
class Prediction:
    label: str
    confidence: float


class SignClassifier:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.55):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels = []
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            return
        package = joblib.load(self.model_path)
        if isinstance(package, dict) and "model" in package:
            self.model = package["model"]
            self.labels = list(package.get("labels", []))
        else:
            self.model = package
            self.labels = list(getattr(self.model, "classes_", []))

    def predict(self, feature_vector: np.ndarray) -> Optional[Prediction]:
        if self.model is None:
            return None

        sample = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            best_index = int(np.argmax(probabilities))
            label = str(self.model.classes_[best_index])
            confidence = float(probabilities[best_index])
        else:
            label = str(self.model.predict(sample)[0])
            confidence = 1.0

        if confidence < self.confidence_threshold:
            return Prediction(label="IDLE", confidence=confidence)

        return Prediction(label=label, confidence=confidence)
