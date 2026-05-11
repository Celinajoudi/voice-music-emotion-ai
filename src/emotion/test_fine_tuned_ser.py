from pathlib import Path

import librosa
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_DIR = "models/fine_tuned_ser"
TEST_CSV = "data/metadata/test.csv"

OUTPUT_REPORT = Path("data/metadata/fine_tuned_evaluation_report.txt")
OUTPUT_CONFUSION = Path("data/metadata/fine_tuned_confusion_matrix.csv")
OUTPUT_PREDICTIONS = Path("data/metadata/fine_tuned_predictions.csv")

TARGET_SR = 16000

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)

model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

print(f"Using device: {device}")


def predict(filepath):
    audio, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)

    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()

    label = model.config.id2label[pred_id]

    return label, confidence


def main():
    df = pd.read_csv(TEST_CSV)

    results = []

    for _, row in df.iterrows():
        pred_label, confidence = predict(row["filepath"])

        results.append({
            "filepath": row["filepath"],
            "true_label": row["label"],
            "predicted_label": pred_label,
            "confidence": confidence
        })

        print(
            f"True: {row['label']} | "
            f"Predicted: {pred_label} | "
            f"Confidence: {confidence:.4f}"
        )

    results_df = pd.DataFrame(results)

    y_true = results_df["true_label"]
    y_pred = results_df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels]
    )

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    cm_df.to_csv(OUTPUT_CONFUSION)

    with open(OUTPUT_REPORT, "w") as f:
        f.write("Fine-Tuned SER Model Evaluation\n")
        f.write("===============================\n\n")
        f.write(f"Total test samples: {len(results_df)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("\nEvaluation complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
