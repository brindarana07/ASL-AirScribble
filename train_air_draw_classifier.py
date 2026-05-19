from collections import Counter
from math import ceil

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import AIR_DRAW_MODEL_PATH, AIR_DRAW_SAMPLE_DIR, RANDOM_STATE, TEST_SIZE
from modules.air_draw_recognizer import load_air_draw_samples


def main():
    x, y = load_air_draw_samples(AIR_DRAW_SAMPLE_DIR)
    counts = Counter(y)

    print("Air drawing samples per class:")
    for label, count in sorted(counts.items()):
        print(f"{label:<4} {count}")

    if len(counts) < 2:
        raise ValueError("Collect samples for at least two letters before training.")
    if min(counts.values()) < 3:
        raise ValueError("Collect at least 3 samples per letter. Recommended: 15-30 per letter.")

    stratify = y if min(counts.values()) >= 2 else None
    test_count = max(ceil(len(y) * TEST_SIZE), len(counts))
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    labels = sorted(model.classes_)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion matrix labels:")
    print(labels)
    print(confusion_matrix(y_test, predictions, labels=labels))

    AIR_DRAW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, AIR_DRAW_MODEL_PATH)
    print(f"\nSaved air drawing model to {AIR_DRAW_MODEL_PATH}")


if __name__ == "__main__":
    main()
