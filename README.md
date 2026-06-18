# Worldcup Match Predictor

A starter machine-learning pipeline for predicting international football match outcomes from team history, rankings, squad, lineup, and match-stat inputs.

## Project Layout

- `data/raw`: source CSV files.
- `data/processed`: engineered datasets.
- `src`: collection, feature engineering, training, evaluation, and prediction scripts.
- `models`: trained model artifacts and metadata.
- `reports`: generated evaluation and prediction outputs.
- `notebooks`: exploratory checks.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\run_pipeline.py
```

The pipeline expects CSV files in `data/raw`. Empty placeholder CSVs are included so scripts fail with clear messages instead of missing-file errors.

## Prediction

After training:

```powershell
python .\src\predict_match.py --home "Brazil" --away "France"
```
