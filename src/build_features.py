from collections import defaultdict

import pandas as pd

from config import FEATURE_COLUMNS, PROCESSED_DATA_DIR, RAW_DATA_DIR


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


def _recent_totals(history: list[dict], window: int) -> dict[str, float]:
    recent = history[-window:]
    return {
        "points": sum(match["points"] for match in recent),
        "goals_for": sum(match["goals_for"] for match in recent),
        "goals_against": sum(match["goals_against"] for match in recent),
        "goal_difference": sum(match["goals_for"] - match["goals_against"] for match in recent),
    }


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


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
            "neutral": int(str(match.get("neutral", False)).lower() in ["true", "1", "yes"]),
            "match_importance": get_match_importance(match.get("tournament", "Other")),
        }
        rows.append(row)

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        team_history[team_a].append({
            "points": _points(home_score, away_score),
            "goals_for": home_score,
            "goals_against": away_score,
        })
        team_history[team_b].append({
            "points": _points(away_score, home_score),
            "goals_for": away_score,
            "goals_against": home_score,
        })

    dataset = pd.DataFrame(rows)
    dataset[FEATURE_COLUMNS] = dataset[FEATURE_COLUMNS].fillna(0)
    return dataset


def main() -> None:
    dataset = build_training_dataset()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "training_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Training dataset saved: {output_path} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
