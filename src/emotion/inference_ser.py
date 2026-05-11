import json
import csv
from pathlib import Path

import librosa
from transformers import pipeline

INPUT_DIR = Path("data/separated/htdemucs")

OUTPUT_JSON = Path("data/metadata/emotion_predictions.json")
OUTPUT_CSV = Path("data/metadata/emotion_predictions.csv")

TARGET_SR = 16000

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


print("Loading emotion recognition model...")

classifier = pipeline(
    task="audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

print("Model loaded successfully")


def extract_true_label(filepath: str):

    filename = filepath.lower()

    for emotion in EMOTIONS:
        if emotion in filename:
            return emotion

    return "unknown"


def predict_emotion(audio_path: Path):

    print(f"\nAnalyzing: {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    prediction = classifier(audio)

    top_prediction = prediction[0]

    predicted_label = top_prediction["label"].lower()

    true_label = extract_true_label(str(audio_path))

    result = {
        "file": str(audio_path),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": float(top_prediction["score"])
    }

    print(
        f"True: {true_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: {result['confidence']:.4f}"
    )

    return result


def save_csv(results):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_CSV, "w", newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "file",
                "true_label",
                "predicted_label",
                "confidence"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved CSV: {OUTPUT_CSV}")


def main():

    vocals_files = list(
        INPUT_DIR.rglob("vocals.wav")
    )

    print(f"Found {len(vocals_files)} vocals files")

    results = []

    for vocals_file in vocals_files:

        try:
            result = predict_emotion(vocals_file)
            results.append(result)

        except Exception as e:
            print(f"Failed: {vocals_file}")
            print(e)

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved JSON: {OUTPUT_JSON}")

    save_csv(results)


if __name__ == "__main__":
    main()

