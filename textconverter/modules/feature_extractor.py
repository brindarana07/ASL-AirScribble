import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

FINGER_CHAINS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

DISTANCE_PAIRS = [
    (4, 8),
    (4, 12),
    (4, 16),
    (4, 20),
    (8, 12),
    (12, 16),
    (16, 20),
    (5, 17),
    (0, 8),
    (0, 12),
    (0, 20),
]


def extract_feature_vector(
    landmarks: np.ndarray,
    include_distances: bool = True,
    include_angles: bool = True,
    include_finger_flags: bool = True,
) -> np.ndarray:
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {landmarks.shape}.")

    normalized = normalize_landmarks(landmarks)
    features: List[float] = normalized.flatten().astype(float).tolist()

    if include_distances:
        features.extend(_pairwise_distances(normalized))

    if include_angles:
        features.extend(_finger_angles(normalized))

    if include_finger_flags:
        features.extend(_finger_extension_flags(normalized))

    return np.array(features, dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    relative = landmarks.astype(np.float32) - landmarks[0].astype(np.float32)
    scale = float(np.max(np.linalg.norm(relative, axis=1)))
    if scale <= 1e-6:
        return relative
    return relative / scale


def feature_header(
    include_distances: bool = True,
    include_angles: bool = True,
    include_finger_flags: bool = True,
) -> List[str]:
    header = ["label"]
    for index in range(21):
        header.extend([f"lm_{index}_x", f"lm_{index}_y", f"lm_{index}_z"])

    if include_distances:
        header.extend([f"dist_{a}_{b}" for a, b in DISTANCE_PAIRS])

    if include_angles:
        for finger in FINGER_CHAINS:
            header.extend([f"{finger}_angle_0", f"{finger}_angle_1"])

    if include_finger_flags:
        header.extend([f"{finger}_extended" for finger in FINGER_CHAINS])

    return header


def append_sample(csv_path: Path, label: str, feature_vector: Sequence[float], header: Iterable[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(list(header))
        writer.writerow([label.upper(), *[f"{value:.8f}" for value in feature_vector]])


def load_label_counts(csv_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not csv_path.exists():
        return counts

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row.get("label", "").upper()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


def _pairwise_distances(landmarks: np.ndarray) -> List[float]:
    return [float(np.linalg.norm(landmarks[a] - landmarks[b])) for a, b in DISTANCE_PAIRS]


def _finger_angles(landmarks: np.ndarray) -> List[float]:
    angles: List[float] = []
    for chain in FINGER_CHAINS.values():
        angles.append(_angle_between(landmarks[chain[0]], landmarks[chain[1]], landmarks[chain[2]]))
        angles.append(_angle_between(landmarks[chain[1]], landmarks[chain[2]], landmarks[chain[3]]))
    return angles


def _finger_extension_flags(landmarks: np.ndarray) -> List[float]:
    flags = []
    for finger, chain in FINGER_CHAINS.items():
        tip = landmarks[chain[-1]]
        pip = landmarks[chain[1]]
        mcp = landmarks[chain[0]]
        if finger == "thumb":
            flags.append(float(np.linalg.norm(tip - landmarks[0]) > np.linalg.norm(pip - landmarks[0])))
        else:
            flags.append(float(np.linalg.norm(tip - mcp) > np.linalg.norm(pip - mcp)))
    return flags


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-6:
        return 0.0
    cosine = float(np.dot(ba, bc) / denominator)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)
