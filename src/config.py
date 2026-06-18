import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

load_dotenv(ROOT_DIR / ".env")

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")
FOOTBALL_API_BASE_URL = os.getenv("FOOTBALL_API_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

RANDOM_STATE = 42
TARGET_COLUMN = "result"

FEATURE_COLUMNS = [
    "elo_diff",
    "points_last_5_diff",
    "points_last_10_diff",
    "goal_difference_last_5_diff",
    "goals_for_last_5_diff",
    "goals_against_last_5_diff",
    "neutral",
    "match_importance",
]

RESULT_LABELS = {
    0: "Team A loss",
    1: "Draw",
    2: "Team A win",
}

# Backward-compatible aliases for older local scripts.
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR
