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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
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


def extract_spectrum(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    return np.abs(spectrogram), power_spectrum


def extract_mfcc(power_spectrum: np.ndarray) -> np.ndarray:
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


def summarize(features: np.ndarray) -> list[np.ndarray]:
    return [
        np.mean(features, axis=1),
        np.std(features, axis=1),
        np.min(features, axis=1),
        np.max(features, axis=1),
        np.median(features, axis=1),
        np.percentile(features, 25, axis=1),
        np.percentile(features, 75, axis=1),
    ]


def extract_spectral_features(audio: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    frequencies = np.linspace(0, TARGET_SR / 2, magnitude.shape[0])[:, None]
    magnitude_sum = np.sum(magnitude, axis=0, keepdims=True) + 1e-10
    normalized_magnitude = magnitude / magnitude_sum

    centroid = np.sum(frequencies * normalized_magnitude, axis=0)
    bandwidth = np.sqrt(
        np.sum(((frequencies - centroid) ** 2) * normalized_magnitude, axis=0)
    )
    cumulative_energy = np.cumsum(normalized_magnitude, axis=0)
    rolloff_indexes = np.argmax(cumulative_energy >= 0.85, axis=0)
    rolloff = frequencies[rolloff_indexes, 0]
    flatness = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0)) / (
        np.mean(magnitude + 1e-10, axis=0)
    )

    frame_count = magnitude.shape[1]
    frame_length = max(1, len(audio) // max(frame_count, 1))
    rms = []
    zcr = []

    for index in range(frame_count):
        start = index * frame_length
        end = min(start + frame_length, len(audio))
        frame = audio[start:end]

        if len(frame) == 0:
            rms.append(0.0)
            zcr.append(0.0)
            continue

        rms.append(float(np.sqrt(np.mean(frame ** 2))))
        zcr.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))

    prosodic = np.vstack(
        [
            centroid,
            bandwidth,
            rolloff,
            flatness,
            np.array(rms),
            np.array(zcr),
        ]
    )

    return np.concatenate(summarize(prosodic))


def extract_basic_features(filepath: str) -> np.ndarray:
    audio = load_audio(filepath)
    magnitude, power_spectrum = extract_spectrum(audio)
    mfcc = extract_mfcc(power_spectrum)
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


def extract_extended_features(filepath: str) -> np.ndarray:
    audio = load_audio(filepath)
    magnitude, power_spectrum = extract_spectrum(audio)
    mfcc = extract_mfcc(power_spectrum)
    delta_features = delta(mfcc)
    delta2_features = delta(delta_features)

    feature_blocks = []

    for features in (mfcc, delta_features, delta2_features):
        feature_blocks.extend(summarize(features))

    feature_blocks.append(extract_spectral_features(audio, magnitude))

    return np.concatenate(feature_blocks)


def extract_features(filepath: str, feature_mode: str) -> np.ndarray:
    if feature_mode == "basic":
        return extract_basic_features(filepath)

    return extract_extended_features(filepath)


def build_feature_matrix(df: pd.DataFrame, feature_mode: str) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for _, row in df.iterrows():
        try:
            features.append(extract_features(row["filepath"], feature_mode))
            labels.append(row["label"])
        except Exception as error:
            print(f"Skipped {row['filepath']}: {error}")

    if not features:
        raise ValueError("No audio features could be extracted.")

    return np.vstack(features), np.array(labels)


def build_svm(c: float, gamma: str | float, kernel: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=c,
                    kernel=kernel,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=True,
                ),
            ),
        ]
    )


def build_voting_model(c: float, gamma: str | float) -> VotingClassifier:
    svm = build_svm(c, gamma, "rbf")
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )

    return VotingClassifier(
        estimators=[
            ("svm", svm),
            ("forest", forest),
        ],
        voting="soft",
        weights=[2, 1],
    )


def build_forest(model_name: str):
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        )

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )


def tune_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[Pipeline | VotingClassifier | RandomForestClassifier | ExtraTreesClassifier, dict, float]:
    candidates = []

    for feature_mode in ("basic", "extended"):
        print(f"\nExtracting {feature_mode} features for tuning...")
        x_train, y_train = build_feature_matrix(train_df, feature_mode)
        x_val, y_val = build_feature_matrix(val_df, feature_mode)

        for kernel in ("rbf", "linear"):
            for c in (0.1, 1, 3, 10, 30, 100):
                gammas = ("scale",) if kernel == "linear" else ("scale", "auto", 0.001, 0.01, 0.1)

                for gamma in gammas:
                    candidates.append(
                        (
                            build_svm(c, gamma, kernel),
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            {
                                "model": "svm",
                                "feature_mode": feature_mode,
                                "kernel": kernel,
                                "C": c,
                                "gamma": gamma,
                            },
                        )
                    )

        for c in (1, 3, 10, 30):
            for gamma in ("scale", 0.001, 0.01):
                candidates.append(
                    (
                        build_voting_model(c, gamma),
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        {
                            "model": "svm_random_forest_voting",
                            "feature_mode": feature_mode,
                            "kernel": "rbf",
                            "C": c,
                            "gamma": gamma,
                        },
                    )
                )

        for model_name in ("random_forest", "extra_trees"):
            candidates.append(
                (
                    build_forest(model_name),
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    {
                        "model": model_name,
                        "feature_mode": feature_mode,
                    },
                )
            )

    best_model = None
    best_params = {}
    best_accuracy = -1.0

    for model, x_train, y_train, x_val, y_val, params in candidates:
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        accuracy = accuracy_score(y_val, predictions)

        if accuracy > best_accuracy:
            best_model = model
            best_params = params
            best_accuracy = accuracy

    if best_model is None:
        raise ValueError("No model candidates were trained.")

    print("\nBest validation setup")
    print(f"Validation accuracy: {best_accuracy:.4f}")
    print(best_params)

    return best_model, best_params, best_accuracy


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[Pipeline | VotingClassifier, dict, float]:
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Train samples: {len(train_df)}")
    print(train_df["label"].value_counts().sort_index())
    print(f"Validation samples: {len(val_df)}")
    print(val_df["label"].value_counts().sort_index())

    _, best_params, best_val_accuracy = tune_model(train_df, val_df)

    print(f"\nFinal training samples: {len(combined_df)}")
    print(combined_df["label"].value_counts().sort_index())
    x_train, y_train = build_feature_matrix(combined_df, best_params["feature_mode"])

    if best_params["model"] == "svm_random_forest_voting":
        model = build_voting_model(best_params["C"], best_params["gamma"])
    elif best_params["model"] in {"random_forest", "extra_trees"}:
        model = build_forest(best_params["model"])
    else:
        model = build_svm(
            best_params["C"],
            best_params["gamma"],
            best_params["kernel"],
        )

    model.fit(x_train, y_train)

    return model, best_params, best_val_accuracy


def evaluate_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    model_path: Path,
    output_report: Path,
    output_confusion: Path,
    output_predictions: Path,
    best_params: dict,
    best_val_accuracy: float,
) -> None:
    print(f"Test samples: {len(test_df)}")
    print(test_df["label"].value_counts().sort_index())

    x_test, y_true = build_feature_matrix(test_df, best_params["feature_mode"])
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
        report_file.write(f"Best validation accuracy: {best_val_accuracy:.4f}\n")
        report_file.write(f"Best parameters: {best_params}\n")
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

    model, best_params, best_val_accuracy = train_model(train_df, val_df)
    evaluate_model(
        model,
        test_df,
        args.model_path,
        args.output_report,
        args.output_confusion,
        args.output_predictions,
        best_params,
        best_val_accuracy,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    print(f"Saved model to: {args.model_path}")


if __name__ == "__main__":
    main()
