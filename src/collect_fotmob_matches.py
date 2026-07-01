import pandas as pd

from config import RAW_DATA_DIR
from fotmob_utils import (
    FOTMOB_RAW_DIR,
    ensure_fotmob_dirs,
    extract_score,
    fetch_match_details,
    fetch_matches_by_date,
    first_present,
    iter_possible_matches,
    names_match,
    parse_date,
    same_day,
    save_json,
    standardize_team_name,
)


COLUMNS = [
    "fixture_id",
    "fotmob_match_id",
    "date",
    "home_team",
    "away_team",
    "status",
    "home_score",
    "away_score",
    "home_penalty_score",
    "away_penalty_score",
    "decided_by_penalties",
    "penalty_loser",
    "competition",
    "round",
    "stadium",
    "attendance",
    "referee",
]


def _match_team_name(match: dict, side: str):
    team = match.get(f"{side}Team") or match.get(side) or {}
    if isinstance(team, dict):
        return first_present(team, ["name", "shortName"])
    return team


def _match_id(match: dict):
    return first_present(match, ["id", "matchId", "match_id"])


def _find_fotmob_match(fixture: pd.Series, date_cache: dict[str, dict]) -> dict | None:
    fixture_date = parse_date(fixture.get("date"))
    if fixture_date is None:
        return None

    date_key = fixture_date.strftime("%Y-%m-%d")
    if date_key not in date_cache:
        print(f"Fetching FotMob matches for {date_key}...")
        data = fetch_matches_by_date(date_key)
        date_cache[date_key] = data or {}
        save_json(date_cache[date_key], FOTMOB_RAW_DIR / "matches" / f"date_{date_key}.json")

    for match in iter_possible_matches(date_cache[date_key]):
        home_name = _match_team_name(match, "home")
        away_name = _match_team_name(match, "away")
        if (
            same_day(fixture.get("date"), first_present(match, ["utcTime", "time", "status.utcTime"]))
            or True
        ) and names_match(fixture.get("home_team"), home_name) and names_match(fixture.get("away_team"), away_name):
            return match
    return None


def _walk_nested(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_nested(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nested(child)


def _score_pair(value) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return pd.to_numeric(value[0], errors="coerce"), pd.to_numeric(value[1], errors="coerce")
    if not isinstance(value, str):
        return None, None
    match = pd.Series([value]).str.extract(r"(\d+(?:\.\d+)?)\s*[-:]\s*(\d+(?:\.\d+)?)").iloc[0]
    if match.isna().any():
        return None, None
    return float(match.iloc[0]), float(match.iloc[1])


def _penalty_details(detail: dict | None) -> tuple[float | None, float | None, bool, str | None]:
    if not isinstance(detail, dict):
        return None, None, False, None
    status = first_present(detail.get("header", {}), ["status"]) or first_present(detail.get("general", {}), ["status"])
    if not isinstance(status, dict):
        return None, None, False, None

    decided_by_penalties = False
    penalty_loser = status.get("whoLostOnPenalties")
    home_penalties = away_penalties = None

    for item in _walk_nested(status):
        reason_short = str(item.get("short") or "").strip().lower()
        reason_key = str(item.get("shortKey") or item.get("longKey") or "").strip().lower()
        reason_long = str(item.get("long") or "").strip().lower()
        if (
            reason_short == "pen"
            or "penalties" in reason_key
            or "afterpenalties" in reason_key
            or reason_long.startswith("pen ")
            or "penalties" in item
            or "penaltyScore" in item
            or "penaltyScoreStr" in item
        ):
            decided_by_penalties = True

        if isinstance(item.get("whoLostOnPenalties"), str) and item.get("whoLostOnPenalties").strip():
            penalty_loser = item.get("whoLostOnPenalties").strip()

        if home_penalties is None or away_penalties is None:
            for key in ["penalties", "penaltyScore", "penaltyScoreStr", "long"]:
                home, away = _score_pair(item.get(key))
                if home is not None and away is not None:
                    home_penalties, away_penalties = float(home), float(away)
                    break

    return home_penalties, away_penalties, decided_by_penalties, penalty_loser


def _row_from_detail(fixture: pd.Series, match: dict | None, detail: dict | None) -> dict:
    general = (detail or {}).get("general", {}) if isinstance(detail, dict) else {}
    content = (detail or {}).get("content", {}) if isinstance(detail, dict) else {}
    match_id = _match_id(match or {}) or first_present(general, ["matchId", "id"])
    home_score, away_score = extract_score(match or general)
    home_penalties, away_penalties, decided_by_penalties, penalty_loser = _penalty_details(detail)

    return {
        "fixture_id": fixture.get("fixture_id"),
        "fotmob_match_id": match_id,
        "date": fixture.get("date") or first_present(general, ["matchTimeUTCDate"]),
        "home_team": standardize_team_name(fixture.get("home_team")),
        "away_team": standardize_team_name(fixture.get("away_team")),
        "status": first_present(match or {}, ["status", "statusStr"]) or first_present(general, ["status"]),
        "home_score": home_score,
        "away_score": away_score,
        "home_penalty_score": home_penalties,
        "away_penalty_score": away_penalties,
        "decided_by_penalties": decided_by_penalties,
        "penalty_loser": penalty_loser,
        "competition": fixture.get("competition") or first_present(general.get("parentLeague") if isinstance(general, dict) else {}, ["name"]),
        "round": fixture.get("round") or first_present(general, ["matchRound", "round"]),
        "stadium": fixture.get("stadium") or first_present(general, ["venueName", "stadium"]),
        "attendance": first_present(general, ["attendance"]) or first_present(content, ["attendance"]),
        "referee": first_present(general, ["referee"]),
    }


def collect_fotmob_matches() -> pd.DataFrame:
    ensure_fotmob_dirs()
    fixtures_path = RAW_DATA_DIR / "fixtures.csv"
    if not fixtures_path.exists():
        print("fixtures.csv does not exist. Run fixture collection first.")
        return pd.DataFrame(columns=COLUMNS)

    fixtures = pd.read_csv(fixtures_path)
    fixtures = fixtures.dropna(subset=["home_team", "away_team"])
    rows = []
    date_cache: dict[str, dict] = {}

    for _, fixture in fixtures.iterrows():
        match = _find_fotmob_match(fixture, date_cache)
        match_id = _match_id(match or {})
        detail = None
        if match_id:
            print(f"Fetching FotMob match details for {fixture.get('home_team')} vs {fixture.get('away_team')} ({match_id})...")
            detail = fetch_match_details(match_id)
            if detail is not None:
                save_json(detail, FOTMOB_RAW_DIR / "matches" / f"match_{match_id}.json")
        else:
            print(f"No FotMob match found for {fixture.get('home_team')} vs {fixture.get('away_team')} on {fixture.get('date')}.")

        rows.append(_row_from_detail(fixture, match, detail))

    output = pd.DataFrame(rows, columns=COLUMNS)
    output_path = RAW_DATA_DIR / "fotmob_match_details.csv"
    output.to_csv(output_path, index=False)
    print(f"FotMob match details saved: {output_path} ({len(output)} rows)")
    print(f"Rows with FotMob match IDs: {output['fotmob_match_id'].notna().sum() if not output.empty else 0}")
    return output


def main() -> None:
    collect_fotmob_matches()


if __name__ == "__main__":
    main()
