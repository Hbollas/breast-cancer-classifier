# in src/predict.py (replace with this main() version)
import argparse, json
import joblib, pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Predict (0=benign,1=malignant) from CSV.")
    p.add_argument("--model", default="models/mlp_best_grid.joblib")
    p.add_argument("--data", required=True)
    p.add_argument("--threshold", type=float, default=None, help="Decision threshold for class 1 (default: use saved)")
    p.add_argument("--threshold_file", default="models/decision_threshold.json", help="JSON file with {'threshold': ...}")
    args = p.parse_args()

    pipe = joblib.load(args.model)
    X = pd.read_csv(args.data)

    has_proba = hasattr(pipe, "predict_proba")
    scores = pipe.predict_proba(X)[:, 1] if has_proba else pipe.decision_function(X)

    thr = args.threshold
    if thr is None and Path(args.threshold_file).exists():
        thr = float(json.loads(Path(args.threshold_file).read_text())["threshold"])
    if thr is None:
        thr = 0.5

    preds = (scores >= thr).astype(int)

    out = X.copy()
    out["prediction"] = preds
    out["malignant_score"] = scores
    out["threshold_used"] = thr

    out_path = Path(args.data).with_suffix(".predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
