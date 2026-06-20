from datetime import date, datetime

import pandas as pd

from api_client import ApiClient
from config import FOOTBALL_DATA_SEASON, FOOTBALL_DATA_WORLD_CUP_CODE, RAW_DATA_DIR, standardize_team_name


SQUAD_COLUMNS = [
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "age",
    "position",
    "club",
    "league",
    "nationality",
    "caps",
    "international_goals",
    "market_value",
    "player_rating",
    "season_minutes",
    "season_appearances",
]


def _age_on_reference_date(date_of_birth: str | None, reference_date: date) -> int | None:
    if not date_of_birth:
        return None
    try:
        born = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
    except ValueError:
        return None

    age = reference_date.year - born.year
    if (reference_date.month, reference_date.day) < (born.month, born.day):
        age -= 1
    return age


def _squads_from_football_data_response(data: dict | None) -> pd.DataFrame:
    reference_date = date(FOOTBALL_DATA_SEASON, 6, 11)
    rows = []

    for team in (data or {}).get("teams", []):
        team_id = team.get("id")
        team_name = standardize_team_name(team.get("name", ""))
        for player in team.get("squad", []) or []:
            rows.append({
                "team_id": team_id,
                "team_name": team_name,
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "age": _age_on_reference_date(player.get("dateOfBirth"), reference_date),
                "position": player.get("position"),
                "club": pd.NA,
                "league": pd.NA,
                "nationality": player.get("nationality"),
                "caps": pd.NA,
                "international_goals": pd.NA,
                "market_value": pd.NA,
                "player_rating": pd.NA,
                "season_minutes": pd.NA,
                "season_appearances": pd.NA,
            })

    return pd.DataFrame(rows, columns=SQUAD_COLUMNS)


def collect_squads() -> pd.DataFrame:
    print(
        "Collecting World Cup squads from football-data.org "
        f"for competition={FOOTBALL_DATA_WORLD_CUP_CODE}, season={FOOTBALL_DATA_SEASON}..."
    )
    client = ApiClient()
    endpoint = f"competitions/{FOOTBALL_DATA_WORLD_CUP_CODE}/teams"
    data = client.get(endpoint, params={"season": FOOTBALL_DATA_SEASON})
    if data is not None:
        client.save_json(data, f"football_data_squads_{FOOTBALL_DATA_WORLD_CUP_CODE}_{FOOTBALL_DATA_SEASON}.json")

    squads = _squads_from_football_data_response(data)
    output_path = RAW_DATA_DIR / "squads.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    squads.to_csv(output_path, index=False)
    print(f"Squads saved: {output_path} ({len(squads)} rows)")
    if squads.empty:
        print("No squad players were saved. Check API key, season, and football-data.org plan coverage.")
    else:
        print("Advanced fields like caps, goals, club, market value, ratings, and minutes are unavailable and left blank.")
    return squads


def main() -> None:
    collect_squads()


if __name__ == "__main__":
    main()
