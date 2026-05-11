from pathlib import Path
import random

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

TEST_SIZE = 0.15
VAL_SIZE = 0.15


def extract_label(filename):

    filename = filename.lower()

    for emotion in EMOTIONS:
        if emotion in filename:
            return emotion

    return None


def main():

    audio_files = list(INPUT_DIR.glob("*.wav"))

    rows = []

    for file in audio_files:

        label = extract_label(file.name)

        if label is None:
            continue

        rows.append({
            "filepath": str(file),
            "label": label
        })

    df = pd.DataFrame(rows)

    print(f"Total labeled samples: {len(df)}")

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("\nDataset split complete")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()

