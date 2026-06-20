from collections import defaultdict

import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, standardize_team_name


ROLLING_COLUMNS = [
    "fixture_id",
    "date",
    "home_team",
    "away_team",
    "home_fotmob_matches_before",
    "away_fotmob_matches_before",
    "fotmob_points_per_match_diff",
    "fotmob_goal_difference_per_match_diff",
    "fotmob_avg_player_rating_diff",
    "fotmob_starting_xi_avg_rating_diff",
    "fotmob_shots_on_target_per_match_diff",
    "fotmob_expected_goals_per_match_diff",
    "fotmob_chances_created_per_match_diff",
    "fotmob_yellow_cards_per_match_diff",
    "fotmob_red_cards_per_match_diff",
    "fotmob_momentum_score",
]


def _points(goals_for, goals_against) -> float:
    if pd.isna(goals_for) or pd.isna(goals_against):
        return pd.NA
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def _team_summary(history: list[dict]) -> dict:
    if not history:
        return {
            "matches": 0,
            "points_per_match": pd.NA,
            "goal_difference_per_match": pd.NA,
            "avg_player_rating": pd.NA,
            "starting_xi_avg_rating": pd.NA,
            "shots_on_target_per_match": pd.NA,
            "expected_goals_per_match": pd.NA,
            "chances_created_per_match": pd.NA,
            "yellow_cards_per_match": pd.NA,
            "red_cards_per_match": pd.NA,
        }

    frame = pd.DataFrame(history)
    return {
        "matches": len(frame),
        "points_per_match": frame["points"].mean(),
        "goal_difference_per_match": frame["goal_difference"].mean(),
        "avg_player_rating": frame["avg_player_rating"].mean(),
        "starting_xi_avg_rating": frame["starting_xi_avg_rating"].mean(),
        "shots_on_target_per_match": frame["team_shots_on_target"].mean(),
        "expected_goals_per_match": frame["team_expected_goals"].mean(),
        "chances_created_per_match": frame["team_chances_created"].mean(),
        "yellow_cards_per_match": frame["yellow_cards"].mean(),
        "red_cards_per_match": frame["red_cards"].mean(),
    }


def _diff(left, right):
    if pd.isna(left) or pd.isna(right):
        return pd.NA
    return left - right


def _momentum_score(row: dict):
    pieces = [
        ("fotmob_points_per_match_diff", 0.35),
        ("fotmob_goal_difference_per_match_diff", 0.25),
        ("fotmob_avg_player_rating_diff", 0.20),
        ("fotmob_shots_on_target_per_match_diff", 0.10),
        ("fotmob_red_cards_per_match_diff", -0.10),
    ]
    total = 0.0
    used = 0
    for column, weight in pieces:
        value = row.get(column)
        if pd.notna(value):
            total += float(value) * weight
            used += 1
    return total if used else pd.NA


def build_fotmob_rolling_features() -> pd.DataFrame:
    fixtures_path = RAW_DATA_DIR / "fixtures.csv"
    fotmob_path = PROCESSED_DATA_DIR / "fotmob_features.csv"
    if not fixtures_path.exists() or not fotmob_path.exists():
        return pd.DataFrame(columns=ROLLING_COLUMNS)

    fixtures = pd.read_csv(fixtures_path)
    fotmob = pd.read_csv(fotmob_path)
    if fixtures.empty or fotmob.empty:
        return pd.DataFrame(columns=ROLLING_COLUMNS)

    fixtures["fixture_id"] = fixtures["fixture_id"].astype(str)
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce", utc=True)
    fixtures["home_team"] = fixtures["home_team"].apply(standardize_team_name)
    fixtures["away_team"] = fixtures["away_team"].apply(standardize_team_name)

    fotmob["fixture_id"] = fotmob["fixture_id"].astype(str)
    fotmob["team_name"] = fotmob["team_name"].apply(standardize_team_name)
    fotmob = fotmob.merge(fixtures[["fixture_id", "date"]], on="fixture_id", how="left", suffixes=("", "_fixture"))

    numeric_columns = [
        "team_goals",
        "goals_conceded",
        "avg_player_rating",
        "starting_xi_avg_rating",
        "team_shots_on_target",
        "team_expected_goals",
        "team_chances_created",
        "yellow_cards",
        "red_cards",
    ]
    for column in numeric_columns:
        if column in fotmob.columns:
            fotmob[column] = pd.to_numeric(fotmob[column], errors="coerce")
        else:
            fotmob[column] = pd.NA

    fotmob["points"] = fotmob.apply(lambda row: _points(row["team_goals"], row["goals_conceded"]), axis=1)
    fotmob["goal_difference"] = fotmob["team_goals"] - fotmob["goals_conceded"]
    fotmob = fotmob.dropna(subset=["date"]).sort_values(["date", "fixture_id", "team_name"])
    fixtures = fixtures.dropna(subset=["date", "home_team", "away_team"]).sort_values(["date", "fixture_id"])

    history: dict[str, list[dict]] = defaultdict(list)
    rows = []
    fotmob_by_fixture = {fixture_id: group for fixture_id, group in fotmob.groupby("fixture_id")}

    for _, fixture in fixtures.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        home = _team_summary(history[home_team])
        away = _team_summary(history[away_team])

        row = {
            "fixture_id": fixture["fixture_id"],
            "date": fixture["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_fotmob_matches_before": home["matches"],
            "away_fotmob_matches_before": away["matches"],
            "fotmob_points_per_match_diff": _diff(home["points_per_match"], away["points_per_match"]),
            "fotmob_goal_difference_per_match_diff": _diff(home["goal_difference_per_match"], away["goal_difference_per_match"]),
            "fotmob_avg_player_rating_diff": _diff(home["avg_player_rating"], away["avg_player_rating"]),
            "fotmob_starting_xi_avg_rating_diff": _diff(home["starting_xi_avg_rating"], away["starting_xi_avg_rating"]),
            "fotmob_shots_on_target_per_match_diff": _diff(home["shots_on_target_per_match"], away["shots_on_target_per_match"]),
            "fotmob_expected_goals_per_match_diff": _diff(home["expected_goals_per_match"], away["expected_goals_per_match"]),
            "fotmob_chances_created_per_match_diff": _diff(home["chances_created_per_match"], away["chances_created_per_match"]),
            "fotmob_yellow_cards_per_match_diff": _diff(home["yellow_cards_per_match"], away["yellow_cards_per_match"]),
            "fotmob_red_cards_per_match_diff": _diff(home["red_cards_per_match"], away["red_cards_per_match"]),
        }
        row["fotmob_momentum_score"] = _momentum_score(row)
        rows.append(row)

        # Update history only after creating the current fixture row.
        # This prevents using Match 2 stats to predict Match 2.
        for _, team_row in fotmob_by_fixture.get(fixture["fixture_id"], pd.DataFrame()).iterrows():
            history[team_row["team_name"]].append(team_row.to_dict())

    output = pd.DataFrame(rows, columns=ROLLING_COLUMNS)
    return output


def main() -> None:
    features = build_fotmob_rolling_features()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "fotmob_rolling_features.csv"
    features.to_csv(output_path, index=False)
    usable = features["fotmob_momentum_score"].notna().sum() if "fotmob_momentum_score" in features.columns else 0
    print(f"FotMob rolling features saved: {output_path} ({len(features)} rows, {usable} with momentum score)")


if __name__ == "__main__":
    main()
