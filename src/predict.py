# src/predict.py
import argparse
import joblib
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Predict breast cancer diagnosis (0=benign,1=malignant) from CSV.")
    p.add_argument("--model", default="models/mlp_best_grid.joblib", help="Path to saved pipeline (joblib).")
    p.add_argument("--data", required=True, help="Path to CSV with the same feature columns as training (no 'diagnosis').")
    args = p.parse_args()

    pipe = joblib.load(args.model)  # this includes scaler + MLP
    X = pd.read_csv(args.data)

    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None

    out = X.copy()
    out["prediction"] = preds
    if proba is not None:
        out["malignant_proba"] = proba

    out_path = Path(args.data).with_suffix(".predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
