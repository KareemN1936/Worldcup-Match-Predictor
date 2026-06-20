import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from config import RAW_DATA_DIR, standardize_team_name


FOTMOB_RAW_DIR = RAW_DATA_DIR / "fotmob"
FOTMOB_BASE_URL = os.getenv("FOTMOB_BASE_URL", "https://www.fotmob.com/api/data").rstrip("/")
FOTMOB_HEADERS = {
    "User-Agent": os.getenv(
        "FOTMOB_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.fotmob.com/",
    "Origin": "https://www.fotmob.com",
    "x-fm-req": "1",
}


def ensure_fotmob_dirs() -> None:
    for folder in ["matches", "lineups", "player_stats", "team_stats"]:
        (FOTMOB_RAW_DIR / folder).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_data(data), indent=2, ensure_ascii=False), encoding="utf-8")


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def request_fotmob(endpoint: str, params: dict[str, Any] | None = None, verbose: bool = True) -> dict[str, Any] | None:
    url = f"{FOTMOB_BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.get(url, params=params or {}, headers=FOTMOB_HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        if verbose:
            print(f"FotMob request failed for {endpoint}: {error}")
    except ValueError as error:
        if verbose:
            print(f"FotMob returned non-JSON response for {endpoint}: {error}")
    return None


def fetch_team(team_id: int) -> dict[str, Any] | None:
    for endpoint, params in [
        ("teams", {"id": team_id}),
        ("teams", {"teamId": team_id}),
    ]:
        data = request_fotmob(endpoint, params)
        if data is not None:
            return data
    return None


def fetch_matches_by_date(date_value: str) -> dict[str, Any] | None:
    date_key = date_value.replace("-", "")
    return request_fotmob("matches", {"date": date_key})


def fetch_match_details(match_id: int | str) -> dict[str, Any] | None:
    return request_fotmob("matchDetails", {"matchId": match_id})


def parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except ValueError:
            return None


def same_day(left: Any, right: Any) -> bool:
    left_dt = parse_date(left)
    right_dt = parse_date(right)
    if not left_dt or not right_dt:
        return False
    return left_dt.date() == right_dt.date()


def normalize_name(value: Any) -> str:
    text = standardize_team_name(str(value or ""))
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def names_match(left: Any, right: Any) -> bool:
    left_name = normalize_name(left)
    right_name = normalize_name(right)
    if not left_name or not right_name:
        return False
    return left_name == right_name or left_name in right_name or right_name in left_name


def find_keys(value: Any, names: set[str]) -> list[Any]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in names:
                found.append(item)
            found.extend(find_keys(item, names))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_keys(item, names))
    return found


def first_present(mapping: dict[str, Any] | None, keys: list[str], default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] not in [None, ""]:
            return mapping[key]
    return default


def to_number(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def extract_score(match: dict[str, Any]) -> tuple[Any, Any]:
    status = match.get("status") or match.get("statusStr")
    home_score = first_present(match, ["homeScore", "home_score"])
    away_score = first_present(match, ["awayScore", "away_score"])
    scores = match.get("scores")
    if isinstance(scores, dict):
        home_score = home_score if home_score is not None else first_present(scores, ["home", "homeScore"])
        away_score = away_score if away_score is not None else first_present(scores, ["away", "awayScore"])
    if isinstance(status, dict):
        home_score = home_score if home_score is not None else first_present(status, ["homeScore"])
        away_score = away_score if away_score is not None else first_present(status, ["awayScore"])
        score_str = first_present(status, ["scoreStr"])
        if (home_score is None or away_score is None) and isinstance(score_str, str) and "-" in score_str:
            parts = [part.strip() for part in score_str.split("-", 1)]
            if len(parts) == 2:
                home_score = to_number(parts[0])
                away_score = to_number(parts[1])
    return home_score, away_score


def iter_possible_matches(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not data:
        return []
    candidates = []
    for key in ["allMatches", "matches"]:
        value = data.get(key)
        if isinstance(value, list):
            candidates.extend([item for item in value if isinstance(item, dict)])
    for league in data.get("leagues", []) if isinstance(data.get("leagues"), list) else []:
        matches = league.get("matches")
        if isinstance(matches, list):
            candidates.extend([item for item in matches if isinstance(item, dict)])
    return candidates
