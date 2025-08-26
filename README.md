# breast-cancer-classifier
![CI](https://github.com/Hbollas/breast-cancer-classifier/actions/workflows/python-ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
**Model card:** [MODEL_CARD.md](MODEL_CARD.md)

## Results (5-fold CV on training set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Neural Net (MLP) | 0.980 | 0.977 | 0.971 | 0.973 | 0.995 |
| Logistic Regression | 0.974 | 0.977 | 0.953 | 0.964 | 0.996 |
| SVM (RBF) | 0.971 | 0.977 | 0.947 | 0.961 | 0.995 |
| Gradient Boosting | 0.965 | 0.964 | 0.941 | 0.952 | 0.993 |
| Random Forest | 0.960 | 0.958 | 0.935 | 0.946 | 0.988 |

**Held-out test set (best MLP):** see `models/mlp_test_classification_report.json`.  
![Confusion Matrix](models/mlp_confusion_matrix.png) ![ROC Curve](models/mlp_roc_curve.png)

### Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m src.train_model --model logreg --data data/Breast_cancer_dataset.csv --out models/
python -m src.evaluate --data data/Breast_cancer_dataset.csv --artifacts models/ --model_name logreg
# or use the tuned MLP already saved: models/mlp_best_grid.joblib

### Quick Demo
```bash
# create a tiny example without the target column
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/Breast_cancer_dataset.csv").drop(columns=["id","Unnamed: 32","diagnosis"], errors="ignore")
df.head(3).to_csv("examples/sample_predict.csv", index=False)
PY

# run predictions with the tuned pipeline
python -m src.predict --model models/mlp_best_grid.joblib --data examples/sample_predict.csv
# => writes examples/sample_predict.predictions.csv with predicted label and probability



# run predictions with the tuned pipeline
python -m src.predict --model models/mlp_best_grid.joblib --data examples/sample_predict.csv
# => writes examples/sample_predict.predictions.csv with predicted label and probability
### Visuals
- **Confusion Matrix (Test):** ![Confusion Matrix](models/mlp_confusion_matrix.png)
- **ROC Curve (Test):** ![ROC Curve](models/mlp_roc_curve.png)
- **Permutation Importance (Top 10):** ![Permutation Importance](models/mlp_permutation_importance_top10.png)

### Operating Threshold
The decision threshold is stored in `models/decision_threshold.json` and can be overridden via:
```bash
python -m src.predict --model models/mlp_best_grid.joblib --data examples/sample_predict.csv --threshold 0.40


