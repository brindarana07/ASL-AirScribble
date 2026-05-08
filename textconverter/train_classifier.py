import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import LANDMARK_CSV_PATH, MODEL_PATH, RANDOM_FOREST_ESTIMATORS, RANDOM_STATE, TEST_SIZE

ASL_TRAINING_LABELS = {"SPACE", "DELETE", "NOTHING", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def load_dataset():
    if not LANDMARK_CSV_PATH.exists():
        raise FileNotFoundError(f"No dataset found at {LANDMARK_CSV_PATH}. Run collect_data.py first.")

    labels = []
    features = []
    with LANDMARK_CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if not header or header[0] != "label":
            raise ValueError("CSV must have a header starting with 'label'.")
        skipped = 0
        for row in reader:
            if len(row) < 2:
                continue
            label = row[0].upper()
            if label not in ASL_TRAINING_LABELS:
                skipped += 1
                continue
            labels.append(label)
            features.append([float(value) for value in row[1:]])

    if not features:
        raise ValueError("Dataset is empty. Collect samples first.")

    if skipped:
        print(f"Skipped {skipped} non-ASL-label rows. Current ASL model trains only A-Z, SPACE, DELETE, NOTHING.")

    return np.array(features, dtype=np.float32), np.array(labels)


def main():
    x, y = load_dataset()
    counts = Counter(y)
    print("Samples per class:")
    for label, count in sorted(counts.items()):
        print(f"{label:<8} {count}")

    stratify = y if min(counts.values()) >= 2 and len(counts) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(model.classes_)

    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion matrix labels:")
    print(labels)
    print(confusion_matrix(y_test, predictions, labels=labels))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": labels}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
