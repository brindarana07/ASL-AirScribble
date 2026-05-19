import csv
from collections import Counter

from config import LANDMARK_CSV_PATH, MODEL_PATH


def load_rows():
    if not LANDMARK_CSV_PATH.exists():
        print(f"No dataset found at {LANDMARK_CSV_PATH}")
        return [], []

    with LANDMARK_CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = list(csv.reader(file))

    if not reader:
        return [], []
    return reader[0], reader[1:]


def save_rows(header, rows):
    with LANDMARK_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def print_counts(rows):
    counts = Counter(row[0].upper() for row in rows if row)
    if not counts:
        print("Dataset is empty.")
        return
    print("\nSamples per label:")
    for label, count in sorted(counts.items()):
        print(f"{label:<10} {count}")


def main():
    header, rows = load_rows()
    if not header:
        return

    while True:
        print_counts(rows)
        print("\nOptions:")
        print("1. Delete all samples for a label")
        print("2. Delete last N samples")
        print("3. Save and exit")
        print("4. Exit without saving")
        choice = input("Choose: ").strip()

        if choice == "1":
            label = input("Label to delete, example A or HELP: ").strip().upper()
            before = len(rows)
            rows = [row for row in rows if row and row[0].upper() != label]
            print(f"Removed {before - len(rows)} samples for {label}.")
        elif choice == "2":
            try:
                count = int(input("How many recent samples to delete? ").strip())
            except ValueError:
                print("Please enter a number.")
                continue
            rows = rows[:-count] if count > 0 else rows
            print(f"Removed last {count} samples.")
        elif choice == "3":
            save_rows(header, rows)
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
                print("Deleted old model.pkl. Train again after cleaning.")
            print("Dataset saved.")
            break
        elif choice == "4":
            print("No changes saved.")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
