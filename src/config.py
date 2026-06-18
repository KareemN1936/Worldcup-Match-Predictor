import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_API_JSON_DIR = RAW_DATA_DIR / "raw_api_json"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

load_dotenv(ROOT_DIR / ".env")

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4").rstrip("/")
FOOTBALL_DATA_WORLD_CUP_CODE = os.getenv("FOOTBALL_DATA_WORLD_CUP_CODE", "WC")
FOOTBALL_DATA_SEASON = int(os.getenv("FOOTBALL_DATA_SEASON", "2026"))

# Backward-compatible names kept so older imports do not break.
FOOTBALL_API_KEY = FOOTBALL_DATA_API_KEY
FOOTBALL_API_BASE_URL = FOOTBALL_DATA_BASE_URL
WORLD_CUP_LEAGUE_ID = FOOTBALL_DATA_WORLD_CUP_CODE
WORLD_CUP_SEASON = FOOTBALL_DATA_SEASON

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

UPCOMING_FIXTURE_STATUSES = {"SCHEDULED", "TIMED", "POSTPONED"}

TEAM_NAME_ALIASES = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
    "United States of America": "United States",
}


def standardize_team_name(name: str) -> str:
    cleaned = str(name).strip()
    return TEAM_NAME_ALIASES.get(cleaned, cleaned)

# Backward-compatible aliases for older local scripts.
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR
