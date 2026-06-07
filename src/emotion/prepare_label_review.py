import argparse
from pathlib import Path

import pandas as pd

INPUT_PREDICTIONS = Path("data/metadata/mfcc_svm_processed_65_20_15_predictions.csv")
OUTPUT_REVIEW = Path("data/metadata/label_review_queue.csv")

VALID_LABELS = [
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


def build_review_queue(
    input_predictions: Path,
    output_review: Path,
    include_correct: bool,
) -> None:
    df = pd.read_csv(input_predictions)

    required_columns = {"filepath", "label", "predicted_label", "confidence"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{input_predictions} is missing required columns: {sorted(missing_columns)}"
        )

    df["needs_review"] = df["label"] != df["predicted_label"]

    if not include_correct:
        df = df[df["needs_review"]].copy()

    df["review_priority"] = df.apply(priority, axis=1)
    df["corrected_label"] = ""
    df["keep"] = "yes"
    df["review_notes"] = ""
    df["valid_labels"] = ", ".join(VALID_LABELS)

    output_columns = [
        "review_priority",
        "filepath",
        "label",
        "predicted_label",
        "confidence",
        "needs_review",
        "corrected_label",
        "keep",
        "review_notes",
        "valid_labels",
    ]

    df = df.sort_values(
        by=["review_priority", "confidence"],
        ascending=[False, True],
    )

    output_review.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(output_review, index=False)

    print("Label review queue created")
    print(f"Input predictions: {input_predictions}")
    print(f"Rows to review: {len(df)}")
    print(f"Output CSV: {output_review}")
    print("\nReview instructions:")
    print("- Listen to each filepath.")
    print("- If the current label is wrong, write the right emotion in corrected_label.")
    print("- If the clip is unclear/noisy/multiple speakers, set keep to no.")
    print("- Leave corrected_label blank when the original label is correct.")


def priority(row: pd.Series) -> int:
    if row["label"] != row["predicted_label"] and row["confidence"] >= 0.50:
        return 3

    if row["label"] != row["predicted_label"]:
        return 2

    if row["confidence"] < 0.35:
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CSV queue for manual label/audio-quality review."
    )
    parser.add_argument(
        "--input-predictions",
        type=Path,
        default=INPUT_PREDICTIONS,
        help="Predictions CSV with filepath, label, predicted_label, confidence.",
    )
    parser.add_argument(
        "--output-review",
        type=Path,
        default=OUTPUT_REVIEW,
        help="Review queue CSV to create.",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include correctly predicted clips too.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_review_queue(
        input_predictions=args.input_predictions,
        output_review=args.output_review,
        include_correct=args.include_correct,
    )


if __name__ == "__main__":
    main()
