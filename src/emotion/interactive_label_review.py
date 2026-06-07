import argparse
import subprocess
from pathlib import Path

import pandas as pd

REVIEW_CSV = Path("data/metadata/label_review_queue.csv")

VALID_LABELS = [
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


def play_audio(filepath: str) -> None:
    subprocess.run(
        ["afplay", filepath],
        check=False,
    )


def print_options() -> None:
    print("\nChoose one:")
    print("1 = angry")
    print("2 = fearful")
    print("3 = happy")
    print("4 = neutral")
    print("5 = sad")
    print("6 = surprised")
    print("c = current label is correct")
    print("p = model prediction is correct")
    print("r = replay")
    print("x = bad/unclear clip, remove it")
    print("s = skip for now")
    print("q = save and quit")


def apply_choice(row: pd.Series, choice: str) -> tuple[str | None, str, bool]:
    label_by_choice = {
        "1": "angry",
        "2": "fearful",
        "3": "happy",
        "4": "neutral",
        "5": "sad",
        "6": "surprised",
    }

    if choice in label_by_choice:
        return label_by_choice[choice], "yes", True

    if choice == "c":
        return "", "yes", True

    if choice == "p":
        return row["predicted_label"], "yes", True

    if choice == "x":
        return "", "no", True

    return None, str(row.get("keep", "yes")), False


def review_labels(review_csv: Path) -> None:
    df = pd.read_csv(review_csv)

    if "corrected_label" not in df.columns:
        df["corrected_label"] = ""

    if "keep" not in df.columns:
        df["keep"] = "yes"

    for index, row in df.iterrows():
        corrected_label = row.get("corrected_label", "")

        if pd.isna(corrected_label):
            corrected_label = ""

        already_reviewed = (
            str(corrected_label).strip()
            or str(row.get("keep", "yes")).strip().lower() == "no"
        )

        if already_reviewed:
            continue

        while True:
            print("\n" + "=" * 72)
            print(f"Clip {index + 1} of {len(df)}")
            print(f"File: {row['filepath']}")
            print(f"Current label: {row['label']}")
            print(f"Model prediction: {row['predicted_label']}")
            print(f"Confidence: {row['confidence']}")
            print("=" * 72)

            input("Press Enter to play...")
            play_audio(row["filepath"])
            print_options()

            choice = input("Your choice: ").strip().lower()

            if choice == "q":
                df.to_csv(review_csv, index=False)
                print(f"\nSaved progress to {review_csv}")
                return

            if choice == "r":
                play_audio(row["filepath"])
                continue

            if choice == "s":
                break

            corrected_label, keep, finished = apply_choice(row, choice)

            if finished:
                if corrected_label is not None:
                    df.at[index, "corrected_label"] = corrected_label
                df.at[index, "keep"] = keep
                df.to_csv(review_csv, index=False)
                print("Saved.")
                break

            print("Invalid choice. Try again.")

    df.to_csv(review_csv, index=False)
    print(f"\nReview complete. Saved to {review_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play clips and review labels interactively in the terminal."
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=REVIEW_CSV,
        help="Review queue CSV to update.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review_labels(args.review_csv)


if __name__ == "__main__":
    main()
