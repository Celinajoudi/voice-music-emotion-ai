import argparse
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("data/processed")
REVIEW_CSV = Path("data/metadata/label_review_queue.csv")
OUTPUT_CSV = Path("data/metadata/cleaned_dataset.csv")

VALID_LABELS = {
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
}

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


def load_dataset(input_dir: Path) -> pd.DataFrame:
    rows = []

    for filepath in sorted(input_dir.rglob("*.wav")):
        label = extract_label(filepath)

        if label is None or label not in VALID_LABELS:
            continue

        rows.append(
            {
                "filepath": str(filepath),
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def apply_review(dataset_df: pd.DataFrame, review_csv: Path) -> pd.DataFrame:
    review_df = pd.read_csv(review_csv)

    if review_df.empty:
        return dataset_df

    review_df["filepath"] = review_df["filepath"].astype(str)
    review_df["keep"] = review_df.get("keep", "yes").fillna("yes").astype(str).str.lower()
    review_df["corrected_label"] = (
        review_df.get("corrected_label", "")
        .fillna("")
        .astype(str)
        .map(normalize_label)
    )

    drop_paths = set(review_df[review_df["keep"].isin({"no", "false", "0"})]["filepath"])
    correction_rows = review_df[
        review_df["corrected_label"].isin(VALID_LABELS)
        & ~review_df["filepath"].isin(drop_paths)
    ]
    corrections = dict(
        zip(
            correction_rows["filepath"],
            correction_rows["corrected_label"],
        )
    )

    cleaned_df = dataset_df[~dataset_df["filepath"].isin(drop_paths)].copy()
    cleaned_df["label"] = cleaned_df.apply(
        lambda row: corrections.get(row["filepath"], row["label"]),
        axis=1,
    )

    return cleaned_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply manual label-review decisions and create a cleaned dataset CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing processed WAV files.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=REVIEW_CSV,
        help="Review CSV created by prepare_label_review.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Cleaned filepath,label CSV to create.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_df = load_dataset(args.input_dir)
    cleaned_df = apply_review(dataset_df, args.review_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(args.output_csv, index=False)

    print("Cleaned dataset created")
    print(f"Original rows: {len(dataset_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Output CSV: {args.output_csv}")
    print("\nLabel distribution:")
    print(cleaned_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
