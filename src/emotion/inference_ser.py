import argparse
import csv
import json
from pathlib import Path

import librosa
from transformers import pipeline

DEFAULT_INPUT_DIR = Path("data/separated/htdemucs")
DEFAULT_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
DEFAULT_OUTPUT_JSON = Path("data/metadata/emotion_predictions.json")
DEFAULT_OUTPUT_CSV = Path("data/metadata/emotion_predictions.csv")
DEFAULT_ACCEPTED_CSV = Path("data/metadata/emotion_predictions_accepted.csv")
DEFAULT_REVIEW_CSV = Path("data/metadata/emotion_predictions_review.csv")

TARGET_SR = 16000

LABEL_NORMALIZATION = {
    "angry": "angry",
    "anger": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
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


def extract_true_label(filepath: Path) -> str:
    text = str(filepath).lower()

    for emotion in EMOTIONS:
        if emotion in text:
            return normalize_label(emotion)

    return "unknown"


def find_vocal_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("vocals.wav"))


def load_vocal_files_from_csv(input_csv: Path) -> list[Path]:
    with open(input_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        if "filepath" not in (reader.fieldnames or []):
            raise ValueError(f"{input_csv} must include a filepath column")

        return [
            Path(row["filepath"])
            for row in reader
        ]


def predict_emotion(audio_path: Path, classifier, confidence_threshold: float) -> dict:
    print(f"\nAnalyzing: {audio_path}")

    audio, _ = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    predictions = classifier(audio, top_k=None)
    predictions = sorted(
        predictions,
        key=lambda prediction: prediction["score"],
        reverse=True,
    )

    top_prediction = predictions[0]
    second_prediction = predictions[1] if len(predictions) > 1 else None

    predicted_label = normalize_label(top_prediction["label"])
    true_label = extract_true_label(audio_path)
    confidence = float(top_prediction["score"])
    second_label = normalize_label(second_prediction["label"]) if second_prediction else ""
    second_confidence = float(second_prediction["score"]) if second_prediction else 0.0
    margin = confidence - second_confidence
    review_status = (
        "accepted"
        if confidence >= confidence_threshold
        else "manual_review"
    )

    result = {
        "filepath": str(audio_path),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "second_label": second_label,
        "second_confidence": second_confidence,
        "confidence_margin": margin,
        "review_status": review_status,
        "human_label": "",
        "notes": "",
    }

    print(
        f"True: {true_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: {confidence:.4f} | "
        f"Status: {review_status}"
    )

    return result


def save_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "filepath",
        "true_label",
        "predicted_label",
        "confidence",
        "second_label",
        "second_confidence",
        "confidence_margin",
        "review_status",
        "human_label",
        "notes",
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_csv}")


def save_outputs(
    results: list[dict],
    output_json: Path,
    output_csv: Path,
    accepted_csv: Path,
    review_csv: Path,
) -> None:
    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_json, "w") as jsonfile:
        json.dump(results, jsonfile, indent=4)

    accepted_rows = [
        row for row in results
        if row["review_status"] == "accepted"
    ]
    review_rows = [
        row for row in results
        if row["review_status"] == "manual_review"
    ]

    print(f"\nSaved JSON: {output_json}")
    save_csv(results, output_csv)
    save_csv(accepted_rows, accepted_csv)
    save_csv(review_rows, review_csv)

    print("\nPrediction summary")
    print(f"Total predictions: {len(results)}")
    print(f"Accepted: {len(accepted_rows)}")
    print(f"Manual review: {len(review_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained SER inference and flag low-confidence clips for review."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Demucs outputs.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional CSV with a filepath column.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face audio-classification model.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum confidence required to accept an automatic label.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Full prediction JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Full prediction CSV output.",
    )
    parser.add_argument(
        "--accepted-csv",
        type=Path,
        default=DEFAULT_ACCEPTED_CSV,
        help="High-confidence predictions CSV.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_REVIEW_CSV,
        help="Low-confidence manual review queue CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vocal_files = (
        load_vocal_files_from_csv(args.input_csv)
        if args.input_csv is not None
        else find_vocal_files(args.input_dir)
    )

    print(f"Found {len(vocal_files)} vocal files")
    print(f"Loading emotion recognition model: {args.model_name}")

    classifier = pipeline(
        task="audio-classification",
        model=args.model_name,
    )

    print("Model loaded successfully")

    results = []

    for vocal_file in vocal_files:
        try:
            result = predict_emotion(
                vocal_file,
                classifier,
                args.confidence_threshold,
            )
            results.append(result)

        except Exception as error:
            print(f"Failed: {vocal_file}")
            print(error)

    save_outputs(
        results,
        args.output_json,
        args.output_csv,
        args.accepted_csv,
        args.review_csv,
    )


if __name__ == "__main__":
    main()
