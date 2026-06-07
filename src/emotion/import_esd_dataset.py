import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ESD_ROOT = Path("external/Emotion Speech Dataset")
OUTPUT_DIR = Path("data/external/esd_processed")
OUTPUT_CSV = Path("data/metadata/esd_dataset.csv")
COMBINED_CSV = Path("data/metadata/combined_dataset.csv")
PROJECT_CSV = Path("data/metadata/cleaned_dataset.csv")

TARGET_SR = 16000
RANDOM_SEED = 42

LABEL_MAP = {
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
}

SUPPORTED_LABELS = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


def detect_label(path: Path) -> str | None:
    text = str(path).lower()

    for source_label, target_label in LABEL_MAP.items():
        if source_label in text:
            return target_label

    return None


def find_audio_files(esd_root: Path) -> list[tuple[Path, str]]:
    rows = []

    for audio_path in sorted(esd_root.rglob("*.wav")):
        label = detect_label(audio_path)

        if label is None:
            continue

        rows.append((audio_path, label))

    return rows


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != TARGET_SR:
        common_divisor = math.gcd(sample_rate, TARGET_SR)
        audio = resample_poly(
            audio,
            TARGET_SR // common_divisor,
            sample_rate // common_divisor,
        ).astype(np.float32)
        sample_rate = TARGET_SR

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio, sample_rate


def write_audio(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = load_audio(source_path)
    sf.write(output_path, audio, sample_rate)


def select_balanced_rows(
    rows: list[tuple[Path, str]],
    max_per_label: int,
) -> list[tuple[Path, str]]:
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []

    for label in SUPPORTED_LABELS:
        label_rows = [
            row for row in rows
            if row[1] == label
        ]
        rng.shuffle(label_rows)
        selected.extend(label_rows[:max_per_label])

    return sorted(selected, key=lambda row: (row[1], str(row[0])))


def import_esd(
    esd_root: Path,
    output_dir: Path,
    output_csv: Path,
    max_per_label: int,
) -> None:
    rows = find_audio_files(esd_root)

    if not rows:
        raise ValueError(f"No labeled WAV files found under {esd_root}")

    selected_rows = select_balanced_rows(rows, max_per_label)
    output_rows = []

    for index, (source_path, label) in enumerate(selected_rows, start=1):
        output_path = output_dir / label / f"esd_{label}_{index:04d}.wav"
        write_audio(source_path, output_path)
        output_rows.append(
            {
                "filepath": str(output_path),
                "label": label,
                "source_file": str(source_path),
                "source_dataset": "ESD",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "filepath",
                "label",
                "source_file",
                "source_dataset",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("ESD import complete")
    print(f"Imported clips: {len(output_rows)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata CSV: {output_csv}")


def combine_with_project_data(
    project_csv: Path,
    esd_csv: Path,
    output_csv: Path,
) -> None:
    project_rows = read_simple_rows(project_csv)
    esd_rows = read_simple_rows(esd_csv)
    combined_rows = project_rows + esd_rows

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["filepath", "label"],
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    print("\nCombined dataset created")
    print(f"Project rows: {len(project_rows)}")
    print(f"ESD rows: {len(esd_rows)}")
    print(f"Combined rows: {len(combined_rows)}")
    print(f"Output CSV: {output_csv}")


def read_simple_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        if "filepath" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must include filepath and label columns")

        return [
            {
                "filepath": row["filepath"],
                "label": row["label"],
            }
            for row in reader
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a balanced subset of the Kaggle ESD dataset for SER training."
    )
    parser.add_argument(
        "--esd-root",
        type=Path,
        default=ESD_ROOT,
        help="Extracted ESD dataset folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for imported/resampled ESD clips.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="CSV for imported ESD clips.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=60,
        help="Maximum clips to import per emotion label.",
    )
    parser.add_argument(
        "--combine-with-project",
        action="store_true",
        help="Create a combined project + ESD dataset CSV after import.",
    )
    parser.add_argument(
        "--project-csv",
        type=Path,
        default=PROJECT_CSV,
        help="Project dataset CSV to combine with ESD.",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=COMBINED_CSV,
        help="Combined project + ESD CSV output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import_esd(
        esd_root=args.esd_root,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        max_per_label=args.max_per_label,
    )

    if args.combine_with_project:
        combine_with_project_data(
            project_csv=args.project_csv,
            esd_csv=args.output_csv,
            output_csv=args.combined_csv,
        )


if __name__ == "__main__":
    main()
