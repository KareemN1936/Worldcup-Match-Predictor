import sys
from pathlib import Path

import joblib
import pandas as pd

from .data_loader import MODELS_DIR, PROCESSED_DIR, load_fixtures, load_predictions
from .notes import build_model_notes


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_model():
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def load_latest_features() -> dict[str, pd.DataFrame]:
    features = {}
    for name in ["training_dataset", "fotmob_features", "fotmob_rolling_features", "squad_features"]:
        path = PROCESSED_DIR / f"{name}.csv"
        features[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return features


def predict_single_match(team_a: str, team_b: str, neutral: bool = True, tournament: str = "FIFA World Cup") -> dict:
    from predict_match import predict

    return predict(team_a, team_b, neutral=neutral, tournament=tournament)


def predict_all_fixtures(fixtures_df: pd.DataFrame | None = None) -> pd.DataFrame:
    from predict_match import predict_fixtures

    if fixtures_df is not None and fixtures_df.empty:
        return pd.DataFrame()
    return predict_fixtures(only_upcoming=False)


def _clean_team(value) -> str:
    return str(value or "").strip().lower()


def _confirmed_upcoming_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty or not {"fixture_id", "home_team", "away_team"}.issubset(fixtures.columns):
        return pd.DataFrame()
    unknown = {"", "nan", "none", "null", "tbd", "to be decided", "to be determined"}
    confirmed = fixtures.copy()
    if "status" in confirmed.columns:
        confirmed = confirmed[
            confirmed["status"].astype(str).str.upper().isin({"SCHEDULED", "TIMED", "POSTPONED"})
        ].copy()
    confirmed = confirmed[
        ~confirmed["home_team"].map(_clean_team).isin(unknown)
        & ~confirmed["away_team"].map(_clean_team).isin(unknown)
    ].copy()
    confirmed["fixture_id"] = confirmed["fixture_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return confirmed


def _missing_confirmed_predictions(fixtures: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    confirmed = _confirmed_upcoming_fixtures(fixtures)
    if confirmed.empty:
        return confirmed
    if predictions.empty or "fixture_id" not in predictions.columns:
        return confirmed

    saved = predictions.copy()
    saved["fixture_id"] = saved["fixture_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    saved = saved[["fixture_id", "home_team", "away_team"]].drop_duplicates("fixture_id", keep="last")
    merged = confirmed[["fixture_id", "home_team", "away_team"]].merge(
        saved,
        on="fixture_id",
        how="left",
        suffixes=("", "_prediction"),
    )
    missing = merged["home_team_prediction"].isna() | merged["away_team_prediction"].isna()
    changed = (
        merged["home_team"].map(_clean_team) != merged["home_team_prediction"].map(_clean_team)
    ) | (
        merged["away_team"].map(_clean_team) != merged["away_team_prediction"].map(_clean_team)
    )
    return confirmed[confirmed["fixture_id"].isin(merged.loc[missing | changed, "fixture_id"])]


def load_or_generate_predictions() -> pd.DataFrame:
    predictions = load_predictions()
    fixtures = load_fixtures()
    if not _missing_confirmed_predictions(fixtures, predictions).empty:
        try:
            return predict_all_fixtures(fixtures)
        except Exception:
            return predictions
    if not predictions.empty:
        return predictions
    try:
        return predict_all_fixtures()
    except Exception:
        return pd.DataFrame()


__all__ = [
    "build_model_notes",
    "load_latest_features",
    "load_model",
    "load_or_generate_predictions",
    "predict_all_fixtures",
    "predict_single_match",
]
