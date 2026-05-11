from pathlib import Path
import subprocess

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/separated")

AUDIO_EXTENSIONS = [".wav"]


def separate_audio(input_file: Path):

    print(f"\nSeparating: {input_file.name}")

    command = [
        "python",
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-o",
        str(OUTPUT_DIR),
        str(input_file)
    ]

    try:
        subprocess.run(
            command,
            check=True
        )

        print(f"Finished: {input_file.name}")

    except subprocess.CalledProcessError as e:
        print(f"Failed: {input_file.name}")
        print(e)


def main():

    files = [
        file for file in INPUT_DIR.rglob("*")
        if file.suffix.lower() in AUDIO_EXTENSIONS
    ]

    print(f"Found {len(files)} processed files")

    for file in files:
        separate_audio(file)


if __name__ == "__main__":
    main()

