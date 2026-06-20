import sys
from pathlib import Path

import joblib
import pandas as pd

from .data_loader import MODELS_DIR, PROCESSED_DIR, load_predictions
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


def load_or_generate_predictions() -> pd.DataFrame:
    predictions = load_predictions()
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
