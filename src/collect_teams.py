import pandas as pd

from api_client import ApiClient
from config import FOOTBALL_DATA_SEASON, FOOTBALL_DATA_WORLD_CUP_CODE, RAW_DATA_DIR, standardize_team_name


TEAM_COLUMNS = ["team_id", "team_name", "country", "fifa_code", "api_provider"]


def _empty_teams() -> pd.DataFrame:
    return pd.DataFrame(columns=TEAM_COLUMNS)


def _teams_from_football_data_response(data: dict | None) -> pd.DataFrame:
    if not data:
        return _empty_teams()

    if data.get("errorCode") or data.get("message"):
        print(f"football-data.org returned a message: {data.get('message', data)}")

    rows = []
    for team in data.get("teams", []):
        area = team.get("area", {})
        rows.append({
            "team_id": team.get("id"),
            "team_name": standardize_team_name(team.get("name", "")),
            "country": area.get("name"),
            "fifa_code": team.get("tla"),
            "api_provider": "football-data.org",
        })
    return pd.DataFrame(rows, columns=TEAM_COLUMNS).drop_duplicates(subset=["team_id", "team_name"])


def _teams_from_fixtures_file() -> pd.DataFrame:
    fixtures_path = RAW_DATA_DIR / "fixtures.csv"
    if not fixtures_path.exists():
        return _empty_teams()

    fixtures = pd.read_csv(fixtures_path)
    rows = []
    for side in ["home", "away"]:
        for _, row in fixtures.iterrows():
            rows.append({
                "team_id": row.get(f"{side}_team_id"),
                "team_name": standardize_team_name(row.get(f"{side}_team", "")),
                "country": row.get("country"),
                "fifa_code": pd.NA,
                "api_provider": "football-data.org",
            })
    return pd.DataFrame(rows, columns=TEAM_COLUMNS).drop_duplicates(subset=["team_id", "team_name"])


def collect_teams() -> pd.DataFrame:
    print(
        "Collecting World Cup teams from football-data.org "
        f"for competition={FOOTBALL_DATA_WORLD_CUP_CODE}, season={FOOTBALL_DATA_SEASON}..."
    )
    client = ApiClient()
    endpoint = f"competitions/{FOOTBALL_DATA_WORLD_CUP_CODE}/teams"
    data = client.get(endpoint, params={"season": FOOTBALL_DATA_SEASON})
    if data is not None:
        client.save_json(data, f"football_data_teams_{FOOTBALL_DATA_WORLD_CUP_CODE}_{FOOTBALL_DATA_SEASON}.json")

    teams = _teams_from_football_data_response(data)
    if teams.empty:
        print("No teams returned by API. Falling back to teams found in fixtures.csv if available.")
        teams = _teams_from_fixtures_file()

    output_path = RAW_DATA_DIR / "teams.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    teams.to_csv(output_path, index=False)
    print(f"Teams saved: {output_path} ({len(teams)} rows)")
    return teams


def main() -> None:
    collect_teams()


if __name__ == "__main__":
    main()
