import argparse
import inspect
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"
DEFAULT_TRAIN_CSV = Path("data/metadata/train_augmented.csv")
DEFAULT_VAL_CSV = Path("data/metadata/val.csv")
DEFAULT_OUTPUT_DIR = Path("checkpoints/ser_model")
DEFAULT_MODEL_DIR = Path("models/fine_tuned_ser")

TARGET_SR = 16000

LABEL_NORMALIZATION = {
    "suprised": "surprised",
    "surprized": "surprised",
}

LABELS = [
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

label2id = {
    label: index
    for index, label in enumerate(LABELS)
}

id2label = {
    index: label
    for label, index in label2id.items()
}


def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    return LABEL_NORMALIZATION.get(label, label)


def load_audio_dataset(csv_path: Path) -> Dataset:
    df = pd.read_csv(csv_path)

    required_columns = {"filepath", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )

    df = df[["filepath", "label"]].dropna()
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(LABELS)]

    if df.empty:
        raise ValueError(f"No valid labeled rows found in {csv_path}")

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df["label"].value_counts().sort_index())

    return Dataset.from_pandas(df, preserve_index=False)


def build_preprocess_fn(feature_extractor):
    def preprocess(example):
        audio, _ = librosa.load(
            example["filepath"],
            sr=TARGET_SR,
            mono=True,
        )

        inputs = feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )

        example["input_values"] = inputs.input_values[0]
        example["label"] = label2id[example["label"]]

        return example

    return preprocess


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )

    training_arg_values = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": 10,
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "remove_unused_columns": True,
    }

    return TrainingArguments(**training_arg_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Wav2Vec2 speech emotion recognition model."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Training CSV with filepath and label columns.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_VAL_CSV,
        help="Validation CSV with filepath and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Checkpoint output directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Final fine-tuned model directory.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--freeze-feature-encoder",
        action="store_true",
        help="Freeze the Wav2Vec2 feature encoder and train only later layers.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    train_dataset = load_audio_dataset(args.train_csv)
    val_dataset = load_audio_dataset(args.val_csv)

    preprocess = build_preprocess_fn(feature_extractor)

    print("Preprocessing training dataset...")
    train_dataset = train_dataset.map(preprocess)

    print("Preprocessing validation dataset...")
    val_dataset = val_dataset.map(preprocess)

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    if args.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        print("Feature encoder frozen")

    training_args = build_training_args(args)

    data_collator = DataCollatorWithPadding(feature_extractor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")
    trainer.train()

    print("\nTraining complete")

    trainer.save_model(str(args.model_dir))
    feature_extractor.save_pretrained(str(args.model_dir))

    print(f"\nModel saved to {args.model_dir}")


if __name__ == "__main__":
    main()
