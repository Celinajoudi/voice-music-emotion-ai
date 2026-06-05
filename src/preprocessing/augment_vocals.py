import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000

DEFAULT_INPUT_DIR = Path("data/separated/htdemucs")
DEFAULT_OUTPUT_DIR = Path("data/augmented/vocals")
DEFAULT_METADATA_CSV = Path("data/metadata/augmented_vocals.csv")

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


def extract_label(path: Path) -> str | None:
    text = str(path).lower()

    for emotion in EMOTIONS:
        if emotion in text:
            return emotion

    return None


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))

    if peak > 0:
        return audio / peak

    return audio


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) > target_length:
        return audio[:target_length]

    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))

    return audio


def add_noise(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise_level = rng.uniform(0.002, 0.01)
    noise = rng.normal(0, noise_level, size=audio.shape)

    return audio + noise


def change_pitch(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    steps = rng.choice([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])

    return librosa.effects.pitch_shift(
        y=audio,
        sr=TARGET_SR,
        n_steps=steps,
    )


def stretch_time(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rate = rng.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(
        y=audio,
        rate=rate,
    )

    return pad_or_trim(stretched, len(audio))


def change_gain(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = rng.uniform(0.75, 1.25)

    return audio * gain


def shift_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    max_shift = int(0.25 * TARGET_SR)
    shift = rng.integers(-max_shift, max_shift + 1)

    return np.roll(audio, shift)


def augment_audio(audio: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "noise": add_noise(audio, rng),
        "pitch": change_pitch(audio, rng),
        "stretch": stretch_time(audio, rng),
        "gain": change_gain(audio, rng),
        "shift": shift_audio(audio, rng),
        "noise_gain": change_gain(add_noise(audio, rng), rng),
    }


def save_audio(audio: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = normalize_audio(audio)
    sf.write(output_path, audio, TARGET_SR)


def load_vocals(input_path: Path) -> np.ndarray:
    audio, _ = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True,
    )

    return normalize_audio(audio)


def find_vocal_files(input_dir: Path) -> list[tuple[Path, str | None]]:
    return [
        (path, None)
        for path in sorted(input_dir.rglob("vocals.wav"))
    ]


def load_vocal_files_from_csv(input_csv: Path) -> list[tuple[Path, str | None]]:
    with open(input_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required_columns = {"filepath", "label"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"{input_csv} is missing required columns: {sorted(missing_columns)}"
            )

        return [
            (Path(row["filepath"]), row["label"])
            for row in reader
        ]


def write_metadata(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "filepath",
                "label",
                "source_file",
                "augmentation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def augment_dataset(
    input_dir: Path,
    input_csv: Path | None,
    output_dir: Path,
    metadata_csv: Path,
    copies_per_clip: int,
    include_original: bool,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    vocal_files = (
        load_vocal_files_from_csv(input_csv)
        if input_csv is not None
        else find_vocal_files(input_dir)
    )

    print(f"Found {len(vocal_files)} vocal clips")

    rows = []
    skipped = 0

    for source_file, csv_label in vocal_files:
        label = csv_label or extract_label(source_file)

        if label is None:
            skipped += 1
            print(f"Skipped unlabeled file: {source_file}")
            continue

        audio = load_vocals(source_file)
        clip_id = source_file.parent.name

        if include_original:
            original_path = output_dir / label / f"{clip_id}_original.wav"
            save_audio(audio, original_path)
            rows.append(
                {
                    "filepath": str(original_path),
                    "label": label,
                    "source_file": str(source_file),
                    "augmentation": "original",
                }
            )

        for copy_index in range(copies_per_clip):
            variants = augment_audio(audio, rng)

            for augmentation_name, augmented_audio in variants.items():
                output_path = (
                    output_dir
                    / label
                    / f"{clip_id}_aug{copy_index + 1}_{augmentation_name}.wav"
                )

                save_audio(augmented_audio, output_path)
                rows.append(
                    {
                        "filepath": str(output_path),
                        "label": label,
                        "source_file": str(source_file),
                        "augmentation": augmentation_name,
                    }
                )

        print(f"Augmented: {source_file}")

    write_metadata(rows, metadata_csv)

    print("\nAugmentation complete")
    print(f"Saved clips: {len(rows)}")
    print(f"Skipped unlabeled clips: {skipped}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create augmented training clips from Demucs-separated vocals."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Demucs outputs. Defaults to data/separated/htdemucs.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional CSV with filepath and label columns. Useful for augmenting train.csv only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where augmented clips will be written.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="CSV file listing augmented clips and labels.",
    )
    parser.add_argument(
        "--copies-per-clip",
        type=int,
        default=1,
        help="How many full augmentation rounds to create per source clip.",
    )
    parser.add_argument(
        "--exclude-original",
        action="store_true",
        help="Do not copy the original vocals into the augmented dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    augment_dataset(
        input_dir=args.input_dir,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        copies_per_clip=args.copies_per_clip,
        include_original=not args.exclude_original,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
