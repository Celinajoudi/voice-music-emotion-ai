from datasets import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
    


import librosa
import numpy as np
import pandas as pd
import torch

MODEL_NAME = "facebook/wav2vec2-base"

TRAIN_CSV = "data/metadata/train.csv"
VAL_CSV = "data/metadata/val.csv"

TARGET_SR = 16000

BATCH_SIZE = 1

EPOCHS = 2


LABELS = [
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
    "suprised",
    "surprized"
]

label2id = {
    label: i for i, label in enumerate(LABELS)
}

id2label = {
    i: label for label, i in label2id.items()
}


feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)


def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    return Dataset.from_pandas(df)


train_dataset = load_dataset(TRAIN_CSV)
val_dataset = load_dataset(VAL_CSV)


def preprocess(example):

    audio, sr = librosa.load(
        example["filepath"],
        sr=TARGET_SR,
        mono=True
    )

    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    example["input_values"] = inputs.input_values[0]

    example["label"] = label2id[
        example["label"]
    ]

    return example


print("Preprocessing training dataset...")

train_dataset = train_dataset.map(preprocess)

print("Preprocessing validation dataset...")

val_dataset = val_dataset.map(preprocess)


model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label
)

model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir="checkpoints/ser_model",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=10
)



data_collator = DataCollatorWithPadding(
    feature_extractor
)


 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)



print("\nStarting training...\n")

trainer.train()

print("\nTraining complete")

trainer.save_model(
    "models/fine_tuned_ser"
)

feature_extractor.save_pretrained(
    "models/fine_tuned_ser"
)

print("\nModel saved to models/fine_tuned_ser")

