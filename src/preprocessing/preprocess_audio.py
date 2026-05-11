import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
MIN_DURATION = 3
MAX_DURATION = 15

INPUT_DIR = Path("data/raw/audio_files_selected_2026-05-11")
OUTPUT_DIR = Path("data/processed")


def preprocess_audio(input_path: Path, output_path: Path) -> bool:
    try:
        audio, _ = librosa.load(input_path, sr=TARGET_SR, mono=True)
        duration = librosa.get_duration(y=audio, sr=TARGET_SR)

        if duration < MIN_DURATION or duration > MAX_DURATION:
            print(f"Skipped {input_path.name}: duration {duration:.2f}s")
            return False

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, TARGET_SR)

        print(f"Processed: {input_path.name}")
        return True

    except Exception as e:
        print(f"Failed {input_path.name}: {e}")
        return False


def main():
    audio_extensions = [".wav", ".mp3", ".m4a", ".flac"]

    files = [
        file for file in INPUT_DIR.rglob("*")
        if file.suffix.lower() in audio_extensions
    ]

    print(f"Found {len(files)} audio files")

    processed = 0
    skipped = 0

    for file in files:
        output_file = OUTPUT_DIR / f"{file.stem}_processed.wav"

        success = preprocess_audio(file, output_file)

        if success:
            processed += 1
        else:
            skipped += 1

    print("\nDone")
    print(f"Processed: {processed}")
    print(f"Skipped/failed: {skipped}")


if __name__ == "__main__":
    main()

