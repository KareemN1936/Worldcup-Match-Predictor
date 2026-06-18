import argparse
import json
from collections import defaultdict

import joblib
import pandas as pd

from build_elo import expected_score, get_k_factor
from build_features import get_match_importance
from config import FEATURE_COLUMNS, MODELS_DIR, RAW_DATA_DIR


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
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


def build_latest_team_state() -> tuple[dict[str, float], dict[str, list[dict]]]:
    matches = pd.read_csv(RAW_DATA_DIR / "historical_matches.csv")
    if matches.empty:
        raise ValueError("historical_matches.csv is empty. Add historical data and run the pipeline first.")

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.sort_values(["date", "match_id"])

    ratings: dict[str, float] = {}
    history: dict[str, list[dict]] = defaultdict(list)

    for _, match in matches.iterrows():
        team_a = match["home_team"]
        team_b = match["away_team"]
        team_a_score = int(match["home_score"])
        team_b_score = int(match["away_score"])

        rating_a = ratings.get(team_a, 1500.0)
        rating_b = ratings.get(team_b, 1500.0)
        expected_a = expected_score(rating_a, rating_b)
        if team_a_score > team_b_score:
            actual_a = 1.0
        elif team_a_score == team_b_score:
            actual_a = 0.5
        else:
            actual_a = 0.0

        k = get_k_factor(match.get("tournament", "Other"))
        ratings[team_a] = rating_a + k * (actual_a - expected_a)
        ratings[team_b] = rating_b + k * ((1 - actual_a) - (1 - expected_a))

        history[team_a].append({
            "points": _points(team_a_score, team_b_score),
            "goals_for": team_a_score,
            "goals_against": team_b_score,
        })
        history[team_b].append({
            "points": _points(team_b_score, team_a_score),
            "goals_for": team_b_score,
            "goals_against": team_a_score,
        })

    return ratings, history


def build_prediction_features(team_a: str, team_b: str, neutral: bool, tournament: str) -> pd.DataFrame:
    ratings, history = build_latest_team_state()
    team_a_last_5 = _recent_totals(history[team_a], 5)
    team_b_last_5 = _recent_totals(history[team_b], 5)
    team_a_last_10 = _recent_totals(history[team_a], 10)
    team_b_last_10 = _recent_totals(history[team_b], 10)

    row = {
        "elo_diff": ratings.get(team_a, 1500.0) - ratings.get(team_b, 1500.0),
        "points_last_5_diff": team_a_last_5["points"] - team_b_last_5["points"],
        "points_last_10_diff": team_a_last_10["points"] - team_b_last_10["points"],
        "goal_difference_last_5_diff": team_a_last_5["goal_difference"] - team_b_last_5["goal_difference"],
        "goals_for_last_5_diff": team_a_last_5["goals_for"] - team_b_last_5["goals_for"],
        "goals_against_last_5_diff": team_a_last_5["goals_against"] - team_b_last_5["goals_against"],
        "neutral": int(neutral),
        "match_importance": get_match_importance(tournament),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(team_a: str, team_b: str, neutral: bool = True, tournament: str = "Friendly") -> dict:
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("No trained model found at models/best_model.pkl. Run python src/run_pipeline.py first.")

    model = joblib.load(model_path)
    features = build_prediction_features(team_a, team_b, neutral, tournament)
    raw_probabilities = model.predict_proba(features)[0]
    probability_by_class = {int(class_label): float(raw_probabilities[index]) for index, class_label in enumerate(model.classes_)}

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win": probability_by_class.get(2, 0.0),
        "draw": probability_by_class.get(1, 0.0),
        "team_b_win": probability_by_class.get(0, 0.0),
        "features": features.iloc[0].to_dict(),
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Phase 1 international football match outcome.")
    parser.add_argument("team_a", help="First listed team, for example Argentina")
    parser.add_argument("team_b", help="Second listed team, for example France")
    parser.add_argument("--neutral", action="store_true", default=True, help="Use neutral venue. Default: true")
    parser.add_argument("--not-neutral", dest="neutral", action="store_false", help="Use non-neutral venue")
    parser.add_argument("--tournament", default="Friendly", help="Tournament name used for match_importance")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = parser.parse_args()

    prediction = predict(args.team_a, args.team_b, args.neutral, args.tournament)
    if args.json:
        print(json.dumps(prediction, indent=2))
        return

    print(f"{args.team_a} vs {args.team_b}")
    print()
    print(f"{args.team_a} Win: {_percent(prediction['team_a_win'])}")
    print(f"Draw: {_percent(prediction['draw'])}")
    print(f"{args.team_b} Win: {_percent(prediction['team_b_win'])}")


if __name__ == "__main__":
    main()
