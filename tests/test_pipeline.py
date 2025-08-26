import joblib, pandas as pd

def test_pipeline_predicts_shape():
    pipe = joblib.load("models/mlp_best_grid.joblib")
    X = pd.read_csv("examples/sample_predict.csv")
    preds = pipe.predict(X)
    assert preds.shape[0] == X.shape[0]
