# 74.21% SER Accuracy Experiment

This experiment improves the speech emotion recognition baseline by adding external
emotion datasets to the manually reviewed project clips.

## Dataset

Final metadata file:

- `data/metadata/combined_esd_tess_dataset.csv`

Dataset composition:

- Project reviewed clips: 188
- ESD clips: 1000
- TESS fearful clips: 400
- Total clips: 1588

Final label distribution:

```text
angry        235
fearful      419
happy        235
neutral      235
sad          232
surprised    232
```

The TESS clips were added because ESD does not contain a fearful class and the
project dataset had only 19 fearful clips after review.

## Split

Split folder:

- `data/metadata/esd_tess_combined_split/`

Split counts:

```text
Train: 952
Validation: 318
Test: 318
```

This uses the project's default 60/20/20 split.

## Commands

Import ESD clips:

```bash
python src/emotion/import_esd_dataset.py \
  --esd-zip /Users/celinajoudi/.cache/kagglehub/datasets/nguyenthanhlim/emotional-speech-dataset-esd/1.archive \
  --max-per-label 200 \
  --output-dir data/external/esd_processed \
  --output-csv data/metadata/esd_dataset.csv
```

Import TESS fearful clips:

```bash
python src/emotion/import_tess_dataset.py
```

Create the final split:

```bash
python src/emotion/create_dataset_split.py \
  --input-csv data/metadata/combined_esd_tess_dataset.csv \
  --output-dir data/metadata/esd_tess_combined_split
```

Train and evaluate:

```bash
python src/emotion/train_mfcc_svm.py \
  --train-csv data/metadata/esd_tess_combined_split/train.csv \
  --val-csv data/metadata/esd_tess_combined_split/val.csv \
  --test-csv data/metadata/esd_tess_combined_split/test.csv \
  --model-path models/mfcc_svm_esd_tess_combined.joblib \
  --output-report data/metadata/mfcc_svm_esd_tess_combined_report.txt \
  --output-confusion data/metadata/mfcc_svm_esd_tess_combined_confusion.csv \
  --output-predictions data/metadata/mfcc_svm_esd_tess_combined_predictions.csv
```

## Result

Final report:

- `data/metadata/mfcc_svm_esd_tess_combined_report.txt`

```text
Best validation accuracy: 0.7201
Accuracy: 0.7421
```

Classification summary:

```text
              precision    recall  f1-score   support

       angry       0.56      0.77      0.65        47
     fearful       0.99      0.98      0.98        84
       happy       0.65      0.47      0.54        47
     neutral       0.69      0.74      0.71        47
         sad       0.72      0.72      0.72        47
   surprised       0.69      0.59      0.64        46

    accuracy                           0.74       318
```
