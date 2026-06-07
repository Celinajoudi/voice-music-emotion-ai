import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import resample_poly, stft
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TRAIN_CSV = Path("data/metadata/train.csv")
VAL_CSV = Path("data/metadata/val.csv")
TEST_CSV = Path("data/metadata/test.csv")
MODEL_PATH = Path("models/mfcc_svm.joblib")
OUTPUT_REPORT = Path("data/metadata/mfcc_svm_evaluation_report.txt")
OUTPUT_CONFUSION = Path("data/metadata/mfcc_svm_confusion_matrix.csv")
OUTPUT_PREDICTIONS = Path("data/metadata/mfcc_svm_predictions.csv")

TARGET_SR = 16000
N_MFCC = 40
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

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


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {"filepath", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )

    df = df[["filepath", "label"]].dropna()
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(VALID_LABELS)].copy()

    if df.empty:
        raise ValueError(f"No valid labeled rows found in {csv_path}")

    return df


def hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595 * np.log10(1 + np.asarray(frequency) / 700)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mels / 2595) - 1)


def build_mel_filterbank(sample_rate: int) -> np.ndarray:
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor((N_FFT + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((N_MELS, N_FFT // 2 + 1))

    for index in range(1, N_MELS + 1):
        left = fft_bins[index - 1]
        center = fft_bins[index]
        right = fft_bins[index + 1]

        if center > left:
            filterbank[index - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)

        if right > center:
            filterbank[index - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filterbank


def load_audio(filepath: str) -> np.ndarray:
    audio, sample_rate = sf.read(filepath, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != TARGET_SR:
        common_divisor = math.gcd(sample_rate, TARGET_SR)
        audio = resample_poly(
            audio,
            TARGET_SR // common_divisor,
            sample_rate // common_divisor,
        ).astype(np.float32)

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    _, _, spectrogram = stft(
        audio,
        fs=TARGET_SR,
        window="hann",
        nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )

    power_spectrum = np.abs(spectrogram) ** 2
    mel_filterbank = build_mel_filterbank(TARGET_SR)
    mel_spectrum = np.dot(mel_filterbank, power_spectrum)
    log_mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))

    return dct(
        log_mel_spectrum,
        type=2,
        axis=0,
        norm="ortho",
    )[:N_MFCC]


def delta(features: np.ndarray) -> np.ndarray:
    return np.gradient(features, axis=1)


def extract_features(filepath: str) -> np.ndarray:
    audio = load_audio(filepath)
    mfcc = extract_mfcc(audio)
    delta_features = delta(mfcc)
    delta2_features = delta(delta_features)

    feature_blocks = []

    for features in (mfcc, delta_features, delta2_features):
        feature_blocks.extend(
            [
                np.mean(features, axis=1),
                np.std(features, axis=1),
                np.min(features, axis=1),
                np.max(features, axis=1),
            ]
        )

    return np.concatenate(feature_blocks)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for _, row in df.iterrows():
        try:
            features.append(extract_features(row["filepath"]))
            labels.append(row["label"])
        except Exception as error:
            print(f"Skipped {row['filepath']}: {error}")

    if not features:
        raise ValueError("No audio features could be extracted.")

    return np.vstack(features), np.array(labels)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Pipeline:
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Training samples: {len(combined_df)}")
    print(combined_df["label"].value_counts().sort_index())

    x_train, y_train = build_feature_matrix(combined_df)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=10,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    return model


def evaluate_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    model_path: Path,
    output_report: Path,
    output_confusion: Path,
    output_predictions: Path,
) -> None:
    print(f"Test samples: {len(test_df)}")
    print(test_df["label"].value_counts().sort_index())

    x_test, y_true = build_feature_matrix(test_df)
    y_pred = model.predict(x_test)
    confidences = np.max(model.predict_proba(x_test), axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = y_pred
    predictions_df["confidence"] = confidences

    output_report.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_predictions, index=False)
    cm_df.to_csv(output_confusion)

    with open(output_report, "w") as report_file:
        report_file.write("MFCC + SVM Evaluation\n")
        report_file.write("=====================\n\n")
        report_file.write(f"Total test samples: {len(test_df)}\n")
        report_file.write(f"Accuracy: {accuracy:.4f}\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(report)

    print("\nEvaluation complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved report to: {output_report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an MFCC + SVM baseline for SER."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=TRAIN_CSV,
        help="Training CSV with filepath and label columns.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=VAL_CSV,
        help="Validation CSV with filepath and label columns.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        help="Test CSV with filepath and label columns.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path where the trained SVM model will be saved.",
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
        "--output-predictions",
        type=Path,
        default=OUTPUT_PREDICTIONS,
        help="Predictions CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = load_metadata(args.train_csv)
    val_df = load_metadata(args.val_csv)
    test_df = load_metadata(args.test_csv)

    model = train_model(train_df, val_df)
    evaluate_model(
        model,
        test_df,
        args.model_path,
        args.output_report,
        args.output_confusion,
        args.output_predictions,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    print(f"Saved model to: {args.model_path}")


if __name__ == "__main__":
    main()
