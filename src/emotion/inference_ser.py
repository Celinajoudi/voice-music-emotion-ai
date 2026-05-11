import json
from pathlib import Path

import librosa
import torch
from transformers import pipeline

INPUT_DIR = Path("separated/htdemucs")

OUTPUT_JSON = Path("data/metadata/emotion_predictions.json")

TARGET_SR = 16000


print("Loading emotion recognition model...")

classifier = pipeline(
    task="audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

print("Model loaded successfully")


def predict_emotion(audio_path: Path):

    print(f"\nAnalyzing: {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    prediction = classifier(audio)

    top_prediction = prediction[0]

    result = {
        "file": str(audio_path),
        "emotion": top_prediction["label"],
        "confidence": float(top_prediction["score"])
    }

    print(
        f"Prediction: "
        f"{result['emotion']} "
        f"({result['confidence']:.4f})"
    )

    return result


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

    print("\nSaved predictions")
    print(f"Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

