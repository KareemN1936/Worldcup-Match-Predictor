import pandas as pd

from api_client import ApiClient
from config import FOOTBALL_DATA_SEASON, FOOTBALL_DATA_WORLD_CUP_CODE, RAW_DATA_DIR, standardize_team_name


FIXTURE_COLUMNS = [
    "fixture_id",
    "date",
    "competition",
    "season",
    "round",
    "group",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "stadium",
    "city",
    "country",
    "status",
]


def _safe_get(mapping: dict | None, key: str, default=None):
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


def _fixtures_from_football_data_response(data: dict | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    if data.get("errorCode") or data.get("message"):
        print(f"football-data.org returned a message: {data.get('message', data)}")

    rows = []
    competition = data.get("competition", {})
    filters = data.get("filters", {})
    for match in data.get("matches", []):
        home_team = match.get("homeTeam", {})
        away_team = match.get("awayTeam", {})
        season = match.get("season", {})
        area = match.get("area", {})

        rows.append({
            "fixture_id": match.get("id"),
            "date": match.get("utcDate"),
            "competition": _safe_get(competition, "name") or _safe_get(match.get("competition"), "name"),
            "season": _safe_get(filters, "season") or _safe_get(season, "startDate", "")[:4] or FOOTBALL_DATA_SEASON,
            "round": match.get("stage") or match.get("matchday"),
            "group": match.get("group"),
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "home_team": standardize_team_name(home_team.get("name", "")),
            "away_team": standardize_team_name(away_team.get("name", "")),
            "stadium": pd.NA,
            "city": pd.NA,
            "country": _safe_get(area, "name"),
            "status": match.get("status"),
        })

    fixtures = pd.DataFrame(rows, columns=FIXTURE_COLUMNS)
    if not fixtures.empty:
        fixtures = fixtures.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    return fixtures


def collect_fixtures() -> pd.DataFrame:
    print(
        "Collecting World Cup fixtures from football-data.org "
        f"for competition={FOOTBALL_DATA_WORLD_CUP_CODE}, season={FOOTBALL_DATA_SEASON}..."
    )
    client = ApiClient()
    endpoint = f"competitions/{FOOTBALL_DATA_WORLD_CUP_CODE}/matches"
    data = client.get(endpoint, params={"season": FOOTBALL_DATA_SEASON})
    if data is not None:
        client.save_json(data, f"football_data_fixtures_{FOOTBALL_DATA_WORLD_CUP_CODE}_{FOOTBALL_DATA_SEASON}.json")

    fixtures = _fixtures_from_football_data_response(data)
    output_path = RAW_DATA_DIR / "fixtures.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)
    print(f"Fixtures saved: {output_path} ({len(fixtures)} rows)")
    if fixtures.empty:
        print("No fixtures were saved. Check API key, competition code, season, and football-data.org plan coverage.")
    return fixtures


def main() -> None:
    collect_fixtures()


if __name__ == "__main__":
    main()
