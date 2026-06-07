import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_CSV = Path("data/metadata/emotion_predictions.csv")
DEFAULT_OUTPUT_CSV = Path("data/metadata/validated_emotion_dataset.csv")

LABEL_NORMALIZATION = {
    "suprised": "surprised",
    "surprized": "surprised",
}

VALID_LABELS = {
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
}


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    return LABEL_NORMALIZATION.get(label, label)


def choose_final_label(row: pd.Series) -> str:
    human_label = str(row.get("human_label", "")).strip()

    if human_label:
        return normalize_label(human_label)

    return normalize_label(row["predicted_label"])


def build_validated_dataset(
    input_csv: Path,
    output_csv: Path,
    include_review_rows: bool,
) -> None:
    df = pd.read_csv(input_csv)

    required_columns = {"filepath", "predicted_label", "review_status"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing required columns: {sorted(missing_columns)}"
        )

    if not include_review_rows:
        accepted_mask = df["review_status"] == "accepted"
        if "human_label" in df.columns:
            reviewed_mask = df["human_label"].fillna("").astype(str).str.strip() != ""
        else:
            reviewed_mask = pd.Series(False, index=df.index)
        df = df[accepted_mask | reviewed_mask].copy()

    df["label"] = df.apply(choose_final_label, axis=1)
    df = df[df["label"].isin(VALID_LABELS)].copy()
    output_df = df[["filepath", "label"]].drop_duplicates()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print("Validated dataset created")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Output CSV: {output_csv}")
    print("\nLabel distribution:")
    print(output_df["label"].value_counts().sort_index())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final filepath,label CSV from accepted and reviewed SER predictions."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Prediction CSV from inference_ser.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Validated dataset CSV to create.",
    )
    parser.add_argument(
        "--include-review-rows",
        action="store_true",
        help="Include manual-review rows even if human_label is still empty.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_validated_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        include_review_rows=args.include_review_rows,
    )


if __name__ == "__main__":
    main()
