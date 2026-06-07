import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TESS_ROOT = Path(
    "/Users/celinajoudi/.cache/kagglehub/datasets/orvile/"
    "toronto-emotional-speech-set-tess/versions/1"
)
OUTPUT_DIR = Path("data/external/tess_processed")
OUTPUT_CSV = Path("data/metadata/tess_fearful_dataset.csv")

TARGET_SR = 16000
RANDOM_SEED = 42


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


def find_fear_clips(tess_root: Path) -> list[Path]:
    return sorted(tess_root.rglob("*_fear.wav"))


def select_clips(audio_paths: list[Path], max_clips: int | None) -> list[Path]:
    if max_clips is None:
        return audio_paths

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = list(audio_paths)
    rng.shuffle(shuffled)

    return sorted(shuffled[:max_clips])


def import_tess_fearful(
    tess_root: Path,
    output_dir: Path,
    output_csv: Path,
    max_clips: int | None,
) -> None:
    audio_paths = find_fear_clips(tess_root)

    if not audio_paths:
        raise ValueError(f"No TESS fear clips found under {tess_root}")

    selected_paths = select_clips(audio_paths, max_clips)
    output_rows = []

    for index, source_path in enumerate(selected_paths, start=1):
        output_path = output_dir / "fearful" / f"tess_fearful_{index:04d}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sample_rate = load_audio(source_path)
        sf.write(output_path, audio, sample_rate)

        output_rows.append(
            {
                "filepath": str(output_path),
                "label": "fearful",
                "source_file": str(source_path),
                "source_dataset": "TESS",
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

    print("TESS fearful import complete")
    print(f"Imported clips: {len(output_rows)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata CSV: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import TESS fear clips as project fearful training samples."
    )
    parser.add_argument(
        "--tess-root",
        type=Path,
        default=TESS_ROOT,
        help="Directory containing downloaded TESS WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for imported/resampled TESS clips.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="CSV for imported TESS fearful clips.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional maximum number of fear clips to import.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_tess_fearful(
        tess_root=args.tess_root,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
