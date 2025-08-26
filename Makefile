# ---- variables ----
PY := python
PIP := pip
VENV := .venv
ACT := source $(VENV)/bin/activate

DATA := data/Breast_cancer_dataset.csv
MODELS := models
MODEL ?= logreg        # options: logreg | rf | svm
MODEL_NAME ?= logreg   # evaluate.py expects name of saved model file

# ---- phony targets ----
.PHONY: venv install format lint test ci train evaluate sample-predict predict clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(ACT) && $(PIP) install --upgrade pip
	$(ACT) && $(PIP) install -r requirements.txt -r requirements-dev.txt

format:
	$(ACT) && black .

lint:
	$(ACT) && black --check .
	$(ACT) && flake8 .

test:
	$(ACT) && pytest -q

ci: lint test

train:
	$(ACT) && $(PY) -m src.train_model --model $(MODEL) --data $(DATA) --out $(MODELS)

evaluate:
	$(ACT) && $(PY) -m src.evaluate --data $(DATA) --artifacts $(MODELS) --model_name $(MODEL_NAME)

# make a tiny input CSV for predict.py (3 rows, no 'diagnosis')
sample-predict:
	$(ACT) && $(PY) -c "import pandas as pd; df=pd.read_csv('$(DATA)').drop(columns=['id','Unnamed: 32','diagnosis'], errors='ignore'); df.head(3).to_csv('data/sample_predict.csv', index=False); print('Wrote data/sample_predict.csv')"

# run predictions using the tuned MLP we saved from the notebook
predict:
	$(ACT) && $(PY) -m src.predict --model $(MODELS)/mlp_best_grid.joblib --data data/sample_predict.csv

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache .ipynb_checkpoints
