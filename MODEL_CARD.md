# Model Card — Breast Cancer Classifier

## Overview
Binary classifier predicting malignant (1) vs benign (0) using the Wisconsin Diagnostic dataset (Kaggle). 
Best model: MLP (tuned via GridSearch, 5-fold CV inside train split), deployed as a scikit-learn pipeline (scaler + MLP).

## Data
- Source: Wisconsin Diagnostic Breast Cancer (569 rows, 30 numeric features after cleaning).
- Target: `diagnosis` mapped M=1, B=0.
- Splits: stratified 80/20 train/test with fixed seed (42).

## Metrics
- CV (5-fold, train split): see `models/model_cv_results.csv`.
- Test (held-out): see `models/mlp_test_classification_report.json`, `mlp_confusion_matrix.png`, `mlp_roc_curve.png`.
- Operating threshold: `models/decision_threshold.json` (chosen to favor high recall).

## Intended Use & Limitations
- Educational/portfolio project; 
- Small dataset; performance may not generalize.
- No external validation/robust clinical evaluation.

## Reproducibility
- Environment: `requirements.txt` (+ `requirements-dev.txt`), `Makefile` targets.
- Artifacts: `models/` (pipeline, threshold, reports).
- Commands in README.
