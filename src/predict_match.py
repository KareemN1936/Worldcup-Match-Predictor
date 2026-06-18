import argparse
import json
from collections import defaultdict

import joblib
import pandas as pd

from build_elo import expected_score, get_k_factor
from build_features import get_match_importance
from config import FEATURE_COLUMNS, MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR, UPCOMING_FIXTURE_STATUSES, standardize_team_name


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
    matches["home_team"] = matches["home_team"].apply(standardize_team_name)
    matches["away_team"] = matches["away_team"].apply(standardize_team_name)
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


def build_prediction_features(
    team_a: str,
    team_b: str,
    neutral: bool,
    tournament: str,
    state: tuple[dict[str, float], dict[str, list[dict]]] | None = None,
) -> pd.DataFrame:
    team_a = standardize_team_name(team_a)
    team_b = standardize_team_name(team_b)
    ratings, history = state or build_latest_team_state()
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


def _load_model():
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("No trained model found at models/best_model.pkl. Run python src/run_pipeline.py first.")
    return joblib.load(model_path)


def predict(
    team_a: str,
    team_b: str,
    neutral: bool = True,
    tournament: str = "Friendly",
    model=None,
    state: tuple[dict[str, float], dict[str, list[dict]]] | None = None,
) -> dict:
    team_a = standardize_team_name(team_a)
    team_b = standardize_team_name(team_b)
    model = model or _load_model()
    features = build_prediction_features(team_a, team_b, neutral, tournament, state=state)
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


def predict_fixtures(only_upcoming: bool = True) -> pd.DataFrame:
    fixtures_path = RAW_DATA_DIR / "fixtures.csv"
    if not fixtures_path.exists():
        raise FileNotFoundError("data/raw/fixtures.csv does not exist. Run python src/run_pipeline.py --fixtures-only first.")

    fixtures = pd.read_csv(fixtures_path)
    if fixtures.empty:
        print("fixtures.csv is empty. Collect fixtures with a valid FOOTBALL_API_KEY before predicting fixtures.")
        return pd.DataFrame()

    if only_upcoming and "status" in fixtures.columns:
        fixtures = fixtures[fixtures["status"].astype(str).isin(UPCOMING_FIXTURE_STATUSES)].copy()
        if fixtures.empty:
            print("No upcoming fixtures found in fixtures.csv.")
            return pd.DataFrame()

    fixtures = fixtures.dropna(subset=["home_team", "away_team"]).copy()
    fixtures = fixtures[
        (fixtures["home_team"].astype(str).str.strip() != "")
        & (fixtures["away_team"].astype(str).str.strip() != "")
        & (fixtures["home_team"].astype(str).str.lower() != "nan")
        & (fixtures["away_team"].astype(str).str.lower() != "nan")
    ].copy()
    if fixtures.empty:
        print("No fixtures with both teams known were found in fixtures.csv.")
        return pd.DataFrame()

    model = _load_model()
    state = build_latest_team_state()
    rows = []
    for _, fixture in fixtures.iterrows():
        home_team = standardize_team_name(fixture["home_team"])
        away_team = standardize_team_name(fixture["away_team"])
        tournament = fixture.get("competition", "FIFA World Cup")
        prediction = predict(home_team, away_team, neutral=True, tournament=tournament, model=model, state=state)
        rows.append({
            "fixture_id": fixture.get("fixture_id"),
            "date": fixture.get("date"),
            "round": fixture.get("round"),
            "group": fixture.get("group"),
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": prediction["team_a_win"],
            "draw_probability": prediction["draw"],
            "away_win_probability": prediction["team_b_win"],
        })

    predictions = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "fixture_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Fixture predictions saved: {output_path} ({len(predictions)} rows)")
    return predictions


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_manual_prediction(prediction: dict) -> None:
    print(f"{prediction['team_a']} vs {prediction['team_b']}")
    print()
    print(f"{prediction['team_a']} Win: {_percent(prediction['team_a_win'])}")
    print(f"Draw: {_percent(prediction['draw'])}")
    print(f"{prediction['team_b']} Win: {_percent(prediction['team_b_win'])}")


def _print_fixture_predictions(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        print("No fixture predictions to show.")
        return

    for _, row in predictions.iterrows():
        print()
        print(f"{row['home_team']} vs {row['away_team']}")
        print(f"Date: {row.get('date', '')}")
        print(f"{row['home_team']} Win: {_percent(row['home_win_probability'])}")
        print(f"Draw: {_percent(row['draw_probability'])}")
        print(f"{row['away_team']} Win: {_percent(row['away_win_probability'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict international football match outcomes.")
    parser.add_argument("team_a", nargs="?", help="First listed team, for example France")
    parser.add_argument("team_b", nargs="?", help="Second listed team, for example Senegal")
    parser.add_argument("--fixtures", action="store_true", help="Predict all upcoming fixtures from data/raw/fixtures.csv")
    parser.add_argument("--all-fixtures", action="store_true", help="Predict every fixture in data/raw/fixtures.csv, including completed fixtures")
    parser.add_argument("--neutral", action="store_true", default=True, help="Use neutral venue. Default: true")
    parser.add_argument("--not-neutral", dest="neutral", action="store_false", help="Use non-neutral venue")
    parser.add_argument("--tournament", default="Friendly", help="Tournament name used for match_importance")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = parser.parse_args()

    if args.fixtures or args.all_fixtures:
        predictions = predict_fixtures(only_upcoming=not args.all_fixtures)
        if args.json:
            print(predictions.to_json(orient="records", indent=2))
        else:
            _print_fixture_predictions(predictions)
        return

    if not args.team_a or not args.team_b:
        parser.error("Provide TEAM_A TEAM_B, or use --fixtures.")

    prediction = predict(args.team_a, args.team_b, args.neutral, args.tournament)
    if args.json:
        print(json.dumps(prediction, indent=2))
        return

    _print_manual_prediction(prediction)


if __name__ == "__main__":
    main()
