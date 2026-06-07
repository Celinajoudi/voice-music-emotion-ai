import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

INPUT_CSV = Path("data/metadata/emotion_predictions.csv")
OUTPUT_REPORT = Path("data/metadata/evaluation_report.txt")
OUTPUT_CONFUSION = Path("data/metadata/confusion_matrix.csv")

LABEL_NORMALIZATION = {
    "suprised": "surprised",
    "surprized": "surprised",
}


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    return LABEL_NORMALIZATION.get(label, label)


def load_predictions(input_csv: Path, accepted_only: bool) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_columns = {"true_label", "predicted_label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{input_csv} is missing required columns: {sorted(missing_columns)}"
        )

    df = df[df["true_label"] != "unknown"].copy()
    df["true_label"] = df["true_label"].map(normalize_label)
    df["predicted_label"] = df["predicted_label"].map(normalize_label)

    if accepted_only and "review_status" in df.columns:
        df = df[df["review_status"] == "accepted"].copy()

    return df


def evaluate_predictions(
    input_csv: Path,
    output_report: Path,
    output_confusion: Path,
    accepted_only: bool,
) -> None:
    df = load_predictions(input_csv, accepted_only)

    if df.empty:
        print("No valid labeled rows found for evaluation.")
        return

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)

    with open(output_report, "w") as report_file:
        report_file.write("Speech Emotion Recognition Evaluation\n")
        report_file.write("====================================\n\n")
        report_file.write(f"Input CSV: {input_csv}\n")
        report_file.write(f"Accepted only: {accepted_only}\n")
        report_file.write(f"Total evaluated samples: {len(df)}\n")
        report_file.write(f"Accuracy: {accuracy:.4f}\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(report)

    cm_df.to_csv(output_confusion)

    print("Evaluation complete")
    print(f"Accepted only: {accepted_only}")
    print(f"Evaluated samples: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved report to: {output_report}")
    print(f"Saved confusion matrix to: {output_confusion}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained SER predictions against known labels."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV,
        help="Prediction CSV containing true_label and predicted_label columns.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=OUTPUT_REPORT,
        help="Evaluation report path.",
    )
    parser.add_argument(
        "--output-confusion",
        type=Path,
        default=OUTPUT_CONFUSION,
        help="Confusion matrix CSV path.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Evaluate only rows with review_status=accepted.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_predictions(
        input_csv=args.input_csv,
        output_report=args.output_report,
        output_confusion=args.output_confusion,
        accepted_only=args.accepted_only,
    )


if __name__ == "__main__":
    main()
