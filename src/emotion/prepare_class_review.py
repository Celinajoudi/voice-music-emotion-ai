import argparse
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("data/processed")
OUTPUT_REVIEW = Path("data/metadata/weak_class_review_queue.csv")

TARGET_LABELS = [
    "fearful",
    "sad",
    "happy",
]

VALID_LABELS = [
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

LABEL_NORMALIZATION = {
    "suprised": "surprised",
    "surprized": "surprised",
}

EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "neutral",
    "surprised",
    "suprised",
    "surprized",
]


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    return LABEL_NORMALIZATION.get(label, label)


def extract_label(path: Path) -> str | None:
    text = str(path).lower()

    for emotion in EMOTIONS:
        if emotion in text:
            return normalize_label(emotion)

    return None


def build_review_queue(
    input_dir: Path,
    output_review: Path,
    target_labels: list[str],
) -> None:
    target_labels = [
        normalize_label(label)
        for label in target_labels
    ]
    rows = []

    for filepath in sorted(input_dir.rglob("*.wav")):
        label = extract_label(filepath)

        if label not in target_labels:
            continue

        rows.append(
            {
                "review_priority": 1,
                "filepath": str(filepath),
                "label": label,
                "predicted_label": "",
                "confidence": "",
                "needs_review": True,
                "corrected_label": "",
                "keep": "yes",
                "review_notes": "",
                "valid_labels": ", ".join(VALID_LABELS),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No matching clips found.")
        return

    output_review.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_review, index=False)

    print("Class review queue created")
    print(f"Target labels: {', '.join(target_labels)}")
    print(f"Rows to review: {len(df)}")
    print(f"Output CSV: {output_review}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manual review queue for selected emotion classes."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing processed WAV files.",
    )
    parser.add_argument(
        "--output-review",
        type=Path,
        default=OUTPUT_REVIEW,
        help="Review queue CSV to create.",
    )
    parser.add_argument(
        "--target-labels",
        nargs="+",
        default=TARGET_LABELS,
        help="Emotion labels to include in the review queue.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_review_queue(
        input_dir=args.input_dir,
        output_review=args.output_review,
        target_labels=args.target_labels,
    )


if __name__ == "__main__":
    main()
