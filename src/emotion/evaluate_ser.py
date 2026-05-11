from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

INPUT_CSV = Path("data/metadata/emotion_predictions.csv")
OUTPUT_REPORT = Path("data/metadata/evaluation_report.txt")
OUTPUT_CONFUSION = Path("data/metadata/confusion_matrix.csv")


def main():
    df = pd.read_csv(INPUT_CSV)

    # Remove rows where true label is unknown
    df = df[df["true_label"] != "unknown"]

    if df.empty:
        print("No valid labeled rows found for evaluation.")
        return

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels]
    )

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_REPORT, "w") as f:
        f.write("Speech Emotion Recognition Evaluation\n")
        f.write("====================================\n\n")
        f.write(f"Total evaluated samples: {len(df)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    cm_df.to_csv(OUTPUT_CONFUSION)

    print("Evaluation complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved report to: {OUTPUT_REPORT}")
    print(f"Saved confusion matrix to: {OUTPUT_CONFUSION}")


if __name__ == "__main__":
    main()

