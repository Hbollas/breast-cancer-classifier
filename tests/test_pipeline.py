# tests/test_pipeline.py
import os, pytest, joblib, pandas as pd

@pytest.mark.skipif(not os.path.exists("models/mlp_best_grid.joblib"),
                    reason="Model artifact not available in CI")
def test_pipeline_predicts_shape():
    pipe = joblib.load("models/mlp_best_grid.joblib")
    X = pd.read_csv("examples/sample_predict.csv")
    preds = pipe.predict(X)
    assert len(preds) == len(X)
