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
python .\src\predict_match.py "Brazil" "France"
```

## Streamlit Dashboard

Run the website/dashboard from the project root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard expects these existing project files when available:

- `data/raw/fixtures.csv` for World Cup fixtures.
- `reports/fixture_predictions.csv` for saved match predictions.
- `data/raw/teams.csv` for FIFA codes used to show emoji flags.
- `models/model_metadata.json` and `reports/evaluation_report.txt` for model performance.
- `reports/feature_importance.csv` for feature importance.
- `reports/match_analysis/*.json` for detailed deterministic match explanations.
- Optional processed files such as `data/processed/fotmob_features.csv`, `data/processed/fotmob_rolling_features.csv`, and `data/processed/squad_features.csv`.

If optional files are missing, the app shows an unavailable message instead of crashing.
Predictions are shown for fixtures where both teams are known; future placeholder fixtures without teams are kept visible but marked as unavailable.
