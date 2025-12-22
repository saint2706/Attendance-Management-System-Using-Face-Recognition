# Evaluation & Benchmarking

The repository ships with an evaluation harness that reuses the exact face-recognition pipeline deployed in production. It loads the encrypted dataset, computes embeddings through DeepFace, and aggregates the metrics required for referee-quality reports (accuracy, precision, recall, macro F1, FAR, FRR, confusion matrix, and a threshold sweep).

## 1. Prepare dataset splits (optional but recommended)

```bash
python manage.py prepare_splits --seed 42
```

This command writes `reports/splits.csv`, which identifies the test split consumed during evaluation. If the file is missing the evaluator falls back to scanning the entire `face_recognition_data/training_dataset/` tree.

## 2. Run the evaluation

```bash
python manage.py eval --split-csv reports/splits.csv
```

The command accepts additional knobs such as `--threshold`, `--threshold-start/stop/step`, `--max-samples`, and `--reports-dir` for ad-hoc experiments. A convenience shortcut is also available via `make evaluate`.

## 3. Inspect the reports

Artifacts are written to `reports/evaluation/`:

- `metrics_summary.json` – accuracy/precision/recall/F1/FAR/FRR plus bookkeeping stats.
- `sample_predictions.csv` – per-image ground truth, candidate match, distance, and predicted label.
- `confusion_matrix.csv` and `confusion_matrix.png` – tabular and visual confusion matrices.
- `threshold_sweep.csv` and `threshold_sweep.png` – FAR/FRR/accuracy/F1 for each distance threshold in the sweep.

Because the evaluator defers to the same dataset cache used during attendance marking, results remain reproducible and consistent with the live service.

## Face-Matching Metric

The recognition pipeline compares embeddings with cosine similarity:

- **Similarity:** `sim(A, B) = (A · B) / (||A|| ||B||)`
- **Cosine distance:** `d(A, B) = 1 − sim(A, B)`

Attendance is accepted when the cosine distance is less than or equal to `RECOGNITION_DISTANCE_THRESHOLD`, which defaults to **0.4** in this repository. Tightening the threshold reduces false accepts while loosening it mitigates false rejects. The evaluation harness (`python manage.py eval` or `make evaluate`) sweeps a configurable range via `--threshold-start/stop/step` and records FAR/FRR trade-offs in `reports/evaluation/threshold_sweep.csv`, making it easy to justify any threshold adjustment before shipping.
