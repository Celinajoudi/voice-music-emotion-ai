import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = Path("data/processed")

OUTPUT_DIR = Path("data/metadata")

EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "neutral",
    "surprised",
    "suprised",
    "surprized"
]

RANDOM_SEED = 42

TEST_SIZE = 0.20
VAL_SIZE = 0.25


def extract_label(filename):

    filename = filename.lower()

    for emotion in EMOTIONS:
        if emotion in filename:
            return emotion

    return None


def load_from_csv(input_csv):
    df = pd.read_csv(input_csv)

    required_columns = {"filepath", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing required columns: {sorted(missing_columns)}"
        )

    return df[["filepath", "label"]].dropna()


def load_from_directory(input_dir, filename_pattern):
    audio_files = list(input_dir.rglob(filename_pattern))

    rows = []

    for file in audio_files:

        label = extract_label(file.name)

        if label is None:
            label = extract_label(file.parent.name)

        if label is None:
            continue

        rows.append({
            "filepath": str(file),
            "label": label
        })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/validation/test CSV splits for SER training."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing labeled WAV files.",
    )
    parser.add_argument(
        "--filename-pattern",
        default="*.wav",
        help="Filename pattern to include, for example vocals.wav for Demucs outputs.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional CSV with filepath and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where train.csv, val.csv, and test.csv will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.input_csv is not None:
        df = load_from_csv(args.input_csv)
    else:
        df = load_from_directory(args.input_dir, args.filename_pattern)

    print(f"Total labeled samples: {len(df)}")

    if df.empty:
        print("No labeled audio files found.")
        return

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        stratify=train_df["label"],
        random_state=RANDOM_SEED
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        args.output_dir / "train.csv",
        index=False
    )

    val_df.to_csv(
        args.output_dir / "val.csv",
        index=False
    )

    test_df.to_csv(
        args.output_dir / "test.csv",
        index=False
    )

    print("\nDataset split complete")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()
