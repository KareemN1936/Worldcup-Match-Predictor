import json

import pandas as pd

from config import RAW_DATA_DIR, standardize_team_name
from fotmob_utils import FOTMOB_RAW_DIR, first_present, save_json, to_number


PLAYER_COLUMNS = [
    "fixture_id",
    "fotmob_match_id",
    "team_name",
    "player_id",
    "player_name",
    "position",
    "minutes",
    "rating",
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "chances_created",
    "passes",
    "pass_accuracy",
    "tackles",
    "interceptions",
    "clearances",
    "duels_won",
    "duels_lost",
    "yellow_card",
    "red_card",
]

TEAM_COLUMNS = [
    "fixture_id",
    "fotmob_match_id",
    "team_name",
    "opponent_name",
    "goals_for",
    "goals_against",
    "shots",
    "shots_on_target",
    "expected_goals",
    "possession",
    "passes",
    "pass_accuracy",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
    "big_chances",
    "big_chances_missed",
    "saves",
    "tackles",
    "interceptions",
    "clearances",
]


STAT_ALIASES = {
    "goals": ["goals", "goal"],
    "assists": ["assists", "assist"],
    "shots": ["shots", "totalShots"],
    "shots_on_target": ["shotsOnTarget", "shotsonTarget", "onTarget"],
    "chances_created": ["chancesCreated", "keyPasses"],
    "passes": ["passes", "accuratePasses"],
    "pass_accuracy": ["passAccuracy", "accuratePassesPercentage"],
    "tackles": ["tackles", "totalTackles"],
    "interceptions": ["interceptions"],
    "clearances": ["clearances"],
    "duels_won": ["duelsWon", "groundDuelsWon", "aerialsWon"],
    "duels_lost": ["duelsLost"],
    "yellow_card": ["yellowCards", "yellowCard"],
    "red_card": ["redCards", "redCard"],
}


def _load_detail(match_id) -> dict | None:
    path = FOTMOB_RAW_DIR / "matches" / f"match_{match_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stat_from_player(player: dict, target: str):
    stats = player.get("stats") if isinstance(player.get("stats"), dict) else {}
    for key in STAT_ALIASES[target]:
        value = first_present(player, [key])
        if value is None:
            value = first_present(stats, [key])
        if isinstance(value, dict):
            value = first_present(value, ["value", "stat"])
        if value is not None:
            return to_number(value)
    return None


def _iter_player_stats(value, team_name, fixture_id, match_id):
    rows = []
    if isinstance(value, dict):
        if first_present(value, ["name", "fullName", "playerName"]) and first_present(value, ["id", "playerId"]):
            row = {
                "fixture_id": fixture_id,
                "fotmob_match_id": match_id,
                "team_name": standardize_team_name(team_name),
                "player_id": first_present(value, ["id", "playerId"]),
                "player_name": first_present(value, ["name", "fullName", "playerName"]),
                "position": first_present(value, ["position", "role", "pos"]),
                "minutes": first_present(value, ["minutes", "minutesPlayed", "minsPlayed"]),
                "rating": to_number(first_present(value, ["rating", "fotmobRating"])),
            }
            for target in STAT_ALIASES:
                row[target] = _stat_from_player(value, target)
            rows.append(row)
        for item in value.values():
            rows.extend(_iter_player_stats(item, team_name, fixture_id, match_id))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_iter_player_stats(item, team_name, fixture_id, match_id))
    return rows


def _extract_player_stats(detail: dict, match: pd.Series) -> list[dict]:
    rows = []
    general = detail.get("general", {})
    content = detail.get("content", {})
    for side in ["home", "away"]:
        team = general.get(f"{side}Team", {})
        team_name = first_present(team, ["name"]) or match.get(f"{side}_team") or side
        for key in ["playerStats", "lineup", "lineups"]:
            if key in content:
                rows.extend(_iter_player_stats(content[key], team_name, match.get("fixture_id"), match.get("fotmob_match_id")))
    return rows


def _extract_team_stats(detail: dict, match: pd.Series) -> list[dict]:
    general = detail.get("general", {})
    content = detail.get("content", {})
    home_name = standardize_team_name(match.get("home_team") or first_present(general.get("homeTeam", {}), ["name"]))
    away_name = standardize_team_name(match.get("away_team") or first_present(general.get("awayTeam", {}), ["name"]))
    home_score = match.get("home_score")
    away_score = match.get("away_score")
    rows = {
        home_name: {"fixture_id": match.get("fixture_id"), "fotmob_match_id": match.get("fotmob_match_id"), "team_name": home_name, "opponent_name": away_name, "goals_for": home_score, "goals_against": away_score},
        away_name: {"fixture_id": match.get("fixture_id"), "fotmob_match_id": match.get("fotmob_match_id"), "team_name": away_name, "opponent_name": home_name, "goals_for": away_score, "goals_against": home_score},
    }
    for row in rows.values():
        for column in TEAM_COLUMNS:
            row.setdefault(column, None)

    stats = content.get("stats", {})
    periods = stats.get("Periods", {}) if isinstance(stats, dict) else {}
    period_all = periods.get("All") if isinstance(periods, dict) else None
    if isinstance(period_all, dict):
        all_stats = period_all.get("stats", [])
    elif isinstance(stats, dict):
        all_stats = stats.get("stats", [])
    else:
        all_stats = []
    for section in all_stats if isinstance(all_stats, list) else []:
        for stat in section.get("stats", []) if isinstance(section, dict) else []:
            title = str(first_present(stat, ["title", "name", "key"], "")).lower().replace(" ", "_")
            values = stat.get("stats") or stat.get("values")
            if not isinstance(values, list) or len(values) < 2:
                continue
            target = {
                "shots": "shots",
                "shots_on_target": "shots_on_target",
                "expected_goals": "expected_goals",
                "xg": "expected_goals",
                "possession": "possession",
                "passes": "passes",
                "pass_accuracy": "pass_accuracy",
                "corners": "corners",
                "fouls": "fouls",
                "yellow_cards": "yellow_cards",
                "red_cards": "red_cards",
                "big_chances": "big_chances",
                "big_chances_missed": "big_chances_missed",
                "saves": "saves",
                "tackles": "tackles",
                "interceptions": "interceptions",
                "clearances": "clearances",
            }.get(title)
            if target:
                rows[home_name][target] = to_number(values[0])
                rows[away_name][target] = to_number(values[1])
    return list(rows.values())


def collect_fotmob_player_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    match_details_path = RAW_DATA_DIR / "fotmob_match_details.csv"
    if not match_details_path.exists():
        print("fotmob_match_details.csv does not exist. Run collect_fotmob_matches.py first.")
        return pd.DataFrame(columns=PLAYER_COLUMNS), pd.DataFrame(columns=TEAM_COLUMNS)

    matches = pd.read_csv(match_details_path)
    player_rows = []
    team_rows = []
    for _, match in matches.dropna(subset=["fotmob_match_id"]).iterrows():
        match_id = str(int(match["fotmob_match_id"])) if float(match["fotmob_match_id"]).is_integer() else str(match["fotmob_match_id"])
        detail = _load_detail(match_id)
        if detail is None:
            continue
        player_extracted = _extract_player_stats(detail, match)
        team_extracted = _extract_team_stats(detail, match)
        save_json(player_extracted, FOTMOB_RAW_DIR / "player_stats" / f"player_stats_{match_id}.json")
        save_json(team_extracted, FOTMOB_RAW_DIR / "team_stats" / f"team_stats_{match_id}.json")
        print(f"Stats rows for match {match_id}: players={len(player_extracted)}, teams={len(team_extracted)}")
        player_rows.extend(player_extracted)
        team_rows.extend(team_extracted)

    player_output = pd.DataFrame(player_rows, columns=PLAYER_COLUMNS).drop_duplicates()
    team_output = pd.DataFrame(team_rows, columns=TEAM_COLUMNS).drop_duplicates()
    player_output.to_csv(RAW_DATA_DIR / "fotmob_player_match_stats.csv", index=False)
    team_output.to_csv(RAW_DATA_DIR / "fotmob_team_match_stats.csv", index=False)
    print(f"FotMob player stats saved: {RAW_DATA_DIR / 'fotmob_player_match_stats.csv'} ({len(player_output)} rows)")
    print(f"FotMob team stats saved: {RAW_DATA_DIR / 'fotmob_team_match_stats.csv'} ({len(team_output)} rows)")
    return player_output, team_output


def main() -> None:
    collect_fotmob_player_stats()


if __name__ == "__main__":
    main()
