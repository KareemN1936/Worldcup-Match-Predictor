import json
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR, standardize_team_name
from fotmob_utils import FOTMOB_RAW_DIR, first_present, save_json, to_number


COLUMNS = [
    "fixture_id",
    "fotmob_match_id",
    "team_name",
    "player_id",
    "player_name",
    "is_starting",
    "is_substitute",
    "position",
    "shirt_number",
    "formation",
    "minutes_played",
    "rating",
]


def _load_detail(match_id) -> dict | None:
    path = FOTMOB_RAW_DIR / "matches" / f"match_{match_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _player_row(player: dict, team_name: str, fixture_id, match_id, formation, is_starting: bool, is_substitute: bool) -> dict:
    performance = player.get("performance") if isinstance(player.get("performance"), dict) else {}
    return {
        "fixture_id": fixture_id,
        "fotmob_match_id": match_id,
        "team_name": standardize_team_name(team_name),
        "player_id": first_present(player, ["id", "playerId"]),
        "player_name": first_present(player, ["name", "fullName", "playerName"]),
        "is_starting": is_starting,
        "is_substitute": is_substitute,
        "position": first_present(player, ["position", "role", "pos"]),
        "shirt_number": first_present(player, ["shirt", "shirtNumber", "number"]),
        "formation": formation,
        "minutes_played": first_present(player, ["minutesPlayed", "minutes", "minsPlayed"]),
        "rating": to_number(
            first_present(player, ["rating", "fotmobRating"], first_present(performance, ["rating"]))
        ),
    }


def _player_stat_value(player_stats: dict | None, stat_keys: set[str]):
    """Read FotMob's nested player-stat cards by machine key or display name."""
    if not isinstance(player_stats, dict):
        return None
    normalized_keys = {key.lower().replace(" ", "_") for key in stat_keys}
    for section in player_stats.get("stats", []) if isinstance(player_stats.get("stats"), list) else []:
        stats = section.get("stats", {}) if isinstance(section, dict) else {}
        if not isinstance(stats, dict):
            continue
        for display_name, payload in stats.items():
            if not isinstance(payload, dict):
                continue
            machine_key = str(payload.get("key", "")).lower()
            display_key = str(display_name).lower().replace(" ", "_")
            if machine_key not in normalized_keys and display_key not in normalized_keys:
                continue
            stat = payload.get("stat")
            value = stat.get("value") if isinstance(stat, dict) else stat
            return to_number(value)
    return None


def _iter_players(value, team_name: str, fixture_id, match_id, formation=None, is_starting=True, is_substitute=False):
    rows = []
    if isinstance(value, dict):
        is_player = any(key in value for key in ["shirtNumber", "shirt", "positionId", "usualPlayingPositionId", "performance"])
        if is_player and first_present(value, ["name", "fullName", "playerName"]) and first_present(value, ["id", "playerId"]):
            rows.append(_player_row(value, team_name, fixture_id, match_id, formation, is_starting, is_substitute))
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text == "coach":
                continue
            rows.extend(_iter_players(
                item,
                team_name,
                fixture_id,
                match_id,
                formation,
                is_starting=is_starting and "bench" not in key_text and "sub" not in key_text,
                is_substitute=is_substitute or "bench" in key_text or "sub" in key_text,
            ))
    elif isinstance(value, list):
        for item in value:
            rows.extend(_iter_players(item, team_name, fixture_id, match_id, formation, is_starting, is_substitute))
    return rows


def _extract_lineups(detail: dict, fixture_id, match_id) -> list[dict]:
    rows = []
    content = detail.get("content", {}) if isinstance(detail, dict) else {}
    lineup = content.get("lineup") or content.get("lineups") or detail.get("lineup") or {}
    if not lineup:
        return rows

    for side in ["home", "away"]:
        team_block = lineup.get(f"{side}Team") or lineup.get(side) or {}
        team_name = first_present(team_block, ["name", "teamName"]) or first_present(detail.get("general", {}).get(f"{side}Team", {}), ["name"])
        formation = first_present(team_block, ["formation"])
        rows.extend(_iter_players(team_block, team_name or side, fixture_id, match_id, formation=formation))

    if not rows:
        general = detail.get("general", {})
        for side in ["home", "away"]:
            team = general.get(f"{side}Team", {})
            team_name = first_present(team, ["name"]) or side
            rows.extend(_iter_players(lineup, team_name, fixture_id, match_id))

    player_stats = content.get("playerStats", {})
    if isinstance(player_stats, dict):
        for row in rows:
            stats = player_stats.get(str(row.get("player_id")))
            minutes = _player_stat_value(stats, {"minutes_played", "minutes played"})
            rating = _player_stat_value(stats, {"rating_title", "fotmob rating"})
            if minutes is not None:
                row["minutes_played"] = minutes
            if rating is not None:
                row["rating"] = rating
    return rows


def collect_fotmob_lineups() -> pd.DataFrame:
    match_details_path = RAW_DATA_DIR / "fotmob_match_details.csv"
    if not match_details_path.exists():
        print("fotmob_match_details.csv does not exist. Run collect_fotmob_matches.py first.")
        return pd.DataFrame(columns=COLUMNS)

    matches = pd.read_csv(match_details_path)
    rows = []
    for _, match in matches.dropna(subset=["fotmob_match_id"]).iterrows():
        match_id = str(int(match["fotmob_match_id"])) if float(match["fotmob_match_id"]).is_integer() else str(match["fotmob_match_id"])
        detail = _load_detail(match_id)
        if detail is None:
            print(f"No raw FotMob detail file found for match {match_id}.")
            continue
        extracted = _extract_lineups(detail, match.get("fixture_id"), match_id)
        save_json(extracted, FOTMOB_RAW_DIR / "lineups" / f"lineup_{match_id}.json")
        print(f"Lineup rows for match {match_id}: {len(extracted)}")
        rows.extend(extracted)

    output = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates()
    output_path = RAW_DATA_DIR / "fotmob_lineups.csv"
    output.to_csv(output_path, index=False)
    print(f"FotMob lineups saved: {output_path} ({len(output)} rows)")
    return output


def main() -> None:
    collect_fotmob_lineups()


if __name__ == "__main__":
    main()
