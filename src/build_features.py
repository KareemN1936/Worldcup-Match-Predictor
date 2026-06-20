from collections import defaultdict

import pandas as pd

from config import FEATURE_COLUMNS, FOTMOB_DIFF_FEATURE_COLUMNS, PROCESSED_DATA_DIR, RAW_DATA_DIR, standardize_team_name


def get_match_importance(tournament: str) -> float:
    name = str(tournament).lower()
    if "fifa world cup" in name or name == "world cup":
        return 4.5
    if "qualif" in name and "world cup" in name:
        return 3.0
    if "nations league" in name:
        return 2.5
    if "friendly" in name:
        return 1.0
    if any(token in name for token in ["euro", "copa america", "african cup", "asian cup", "gold cup"]):
        return 3.5
    return 1.5


def result_from_scores(team_a_goals: int, team_b_goals: int) -> int:
    if team_a_goals > team_b_goals:
        return 2
    if team_a_goals == team_b_goals:
        return 1
    return 0


def _empty_feature_totals() -> dict[str, float]:
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "clean_sheets": 0,
        "failed_to_score": 0,
        "weighted_points": 0.0,
        "weighted_goal_difference": 0.0,
    }


def _recent_totals(history: list[dict], window: int) -> dict[str, float]:
    recent = history[-window:]
    totals = _empty_feature_totals()
    totals["matches"] = len(recent)
    for match in recent:
        totals["points"] += match["points"]
        totals["goals_for"] += match["goals_for"]
        totals["goals_against"] += match["goals_against"]
        totals["goal_difference"] += match["goals_for"] - match["goals_against"]
        totals["wins"] += int(match["points"] == 3)
        totals["draws"] += int(match["points"] == 1)
        totals["losses"] += int(match["points"] == 0)
        totals["clean_sheets"] += int(match["goals_against"] == 0)
        totals["failed_to_score"] += int(match["goals_for"] == 0)
        totals["weighted_points"] += match["points"] * match.get("importance", 1.0)
        totals["weighted_goal_difference"] += (match["goals_for"] - match["goals_against"]) * match.get("importance", 1.0)
    return totals


def _two_year_rates(history: list[dict], match_date: pd.Timestamp) -> dict[str, float]:
    if pd.isna(match_date):
        return {
            "matches_played": 0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "goals_for_per_match": 0.0,
            "goals_against_per_match": 0.0,
            "goal_difference_per_match": 0.0,
            "clean_sheet_rate": 0.0,
            "failed_to_score_rate": 0.0,
        }

    start_date = match_date - pd.DateOffset(years=2)
    recent = [match for match in history if start_date <= match["date"] < match_date]
    totals = _recent_totals(recent, len(recent))
    matches = max(totals["matches"], 1)
    return {
        "matches_played": totals["matches"],
        "win_rate": totals["wins"] / matches,
        "draw_rate": totals["draws"] / matches,
        "loss_rate": totals["losses"] / matches,
        "goals_for_per_match": totals["goals_for"] / matches,
        "goals_against_per_match": totals["goals_against"] / matches,
        "goal_difference_per_match": totals["goal_difference"] / matches,
        "clean_sheet_rate": totals["clean_sheets"] / matches,
        "failed_to_score_rate": totals["failed_to_score"] / matches,
    }


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _history_entry(match_date: pd.Timestamp, goals_for: int, goals_against: int, tournament: str) -> dict:
    return {
        "date": match_date,
        "points": _points(goals_for, goals_against),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "importance": get_match_importance(tournament),
    }


def merge_optional_fotmob_features(dataset: pd.DataFrame) -> pd.DataFrame:
    fotmob_path = PROCESSED_DATA_DIR / "fotmob_features.csv"
    if not fotmob_path.exists() or dataset.empty:
        return dataset

    if "fixture_id" not in dataset.columns:
        for column in FOTMOB_DIFF_FEATURE_COLUMNS:
            if column not in dataset.columns:
                dataset[column] = pd.NA
        return dataset

    fotmob = pd.read_csv(fotmob_path)
    if fotmob.empty:
        return dataset

    fotmob["team_name"] = fotmob["team_name"].apply(standardize_team_name)
    base_columns = {
        "starting_xi_avg_rating": "starting_xi_avg_rating_diff",
        "team_shots_on_target": "team_shots_on_target_diff",
        "team_expected_goals": "team_expected_goals_diff",
        "team_chances_created": "team_chances_created_diff",
        "goals_minus_xg": "goals_minus_xg_diff",
        "avg_player_rating": "avg_player_rating_diff",
        "red_cards": "red_cards_diff",
        "yellow_cards": "yellow_cards_diff",
        "substitute_goal_contributions": "substitute_goal_contributions_diff",
    }
    available = [column for column in base_columns if column in fotmob.columns]
    if not available:
        return dataset

    team_a = fotmob[["fixture_id", "team_name", *available]].rename(columns={"team_name": "team_a"})
    team_b = fotmob[["fixture_id", "team_name", *available]].rename(columns={"team_name": "team_b"})
    team_a = team_a.rename(columns={column: f"team_a_{column}" for column in available})
    team_b = team_b.rename(columns={column: f"team_b_{column}" for column in available})

    merged = dataset.copy()
    merged["team_a"] = merged["team_a"].apply(standardize_team_name)
    merged["team_b"] = merged["team_b"].apply(standardize_team_name)
    merged = merged.merge(team_a, on=["fixture_id", "team_a"], how="left")
    merged = merged.merge(team_b, on=["fixture_id", "team_b"], how="left")

    for column in available:
        merged[base_columns[column]] = merged[f"team_a_{column}"] - merged[f"team_b_{column}"]

    return merged


def build_training_dataset() -> pd.DataFrame:
    matches_path = RAW_DATA_DIR / "historical_matches.csv"
    elo_path = PROCESSED_DATA_DIR / "elo_history.csv"

    matches = pd.read_csv(matches_path)
    if matches.empty:
        columns = ["match_id", "date", "team_a", "team_b", "result", *FEATURE_COLUMNS]
        return pd.DataFrame(columns=columns)

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["match_id"] = matches["match_id"].astype(str)
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    elo = pd.read_csv(elo_path) if elo_path.exists() else pd.DataFrame()
    if not elo.empty:
        elo["match_id"] = elo["match_id"].astype(str)
        matches = matches.merge(elo[["match_id", "elo_diff"]], on="match_id", how="left")
    else:
        matches["elo_diff"] = 0.0

    team_history: dict[str, list[dict]] = defaultdict(list)
    rows = []

    for _, match in matches.iterrows():
        team_a = match["home_team"]
        team_b = match["away_team"]
        team_a_last_5 = _recent_totals(team_history[team_a], 5)
        team_b_last_5 = _recent_totals(team_history[team_b], 5)
        team_a_last_10 = _recent_totals(team_history[team_a], 10)
        team_b_last_10 = _recent_totals(team_history[team_b], 10)
        team_a_2y = _two_year_rates(team_history[team_a], match["date"])
        team_b_2y = _two_year_rates(team_history[team_b], match["date"])

        row = {
            "match_id": match["match_id"],
            "date": match["date"],
            "team_a": team_a,
            "team_b": team_b,
            "home_score": int(match["home_score"]),
            "away_score": int(match["away_score"]),
            "tournament": match.get("tournament", "Other"),
            "result": result_from_scores(int(match["home_score"]), int(match["away_score"])),
            "elo_diff": float(match.get("elo_diff", 0.0) if pd.notna(match.get("elo_diff", 0.0)) else 0.0),
            "points_last_5_diff": team_a_last_5["points"] - team_b_last_5["points"],
            "points_last_10_diff": team_a_last_10["points"] - team_b_last_10["points"],
            "goal_difference_last_5_diff": team_a_last_5["goal_difference"] - team_b_last_5["goal_difference"],
            "goals_for_last_5_diff": team_a_last_5["goals_for"] - team_b_last_5["goals_for"],
            "goals_against_last_5_diff": team_a_last_5["goals_against"] - team_b_last_5["goals_against"],
            "weighted_points_last_5_diff": team_a_last_5["weighted_points"] - team_b_last_5["weighted_points"],
            "weighted_goal_difference_last_5_diff": team_a_last_5["weighted_goal_difference"] - team_b_last_5["weighted_goal_difference"],
            "wins_last_5_diff": team_a_last_5["wins"] - team_b_last_5["wins"],
            "draws_last_5_diff": team_a_last_5["draws"] - team_b_last_5["draws"],
            "losses_last_5_diff": team_a_last_5["losses"] - team_b_last_5["losses"],
            "clean_sheets_last_5_diff": team_a_last_5["clean_sheets"] - team_b_last_5["clean_sheets"],
            "failed_to_score_last_5_diff": team_a_last_5["failed_to_score"] - team_b_last_5["failed_to_score"],
            "matches_played_2y_diff": team_a_2y["matches_played"] - team_b_2y["matches_played"],
            "win_rate_2y_diff": team_a_2y["win_rate"] - team_b_2y["win_rate"],
            "draw_rate_2y_diff": team_a_2y["draw_rate"] - team_b_2y["draw_rate"],
            "loss_rate_2y_diff": team_a_2y["loss_rate"] - team_b_2y["loss_rate"],
            "goals_for_per_match_2y_diff": team_a_2y["goals_for_per_match"] - team_b_2y["goals_for_per_match"],
            "goals_against_per_match_2y_diff": team_a_2y["goals_against_per_match"] - team_b_2y["goals_against_per_match"],
            "goal_difference_per_match_2y_diff": team_a_2y["goal_difference_per_match"] - team_b_2y["goal_difference_per_match"],
            "clean_sheet_rate_2y_diff": team_a_2y["clean_sheet_rate"] - team_b_2y["clean_sheet_rate"],
            "failed_to_score_rate_2y_diff": team_a_2y["failed_to_score_rate"] - team_b_2y["failed_to_score_rate"],
            "neutral": int(str(match.get("neutral", False)).lower() in ["true", "1", "yes"]),
            "match_importance": get_match_importance(match.get("tournament", "Other")),
        }
        rows.append(row)

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        tournament = match.get("tournament", "Other")
        team_history[team_a].append(_history_entry(match["date"], home_score, away_score, tournament))
        team_history[team_b].append(_history_entry(match["date"], away_score, home_score, tournament))

    dataset = pd.DataFrame(rows)
    dataset[FEATURE_COLUMNS] = dataset[FEATURE_COLUMNS].fillna(0)
    dataset = merge_optional_fotmob_features(dataset)
    return dataset


def main() -> None:
    dataset = build_training_dataset()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "training_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Training dataset saved: {output_path} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
