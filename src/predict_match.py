import argparse
import json
from collections import defaultdict

import joblib
import pandas as pd

from build_elo import expected_score, get_k_factor
from build_features import get_match_importance
from config import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, RESULT_LABELS, UPCOMING_FIXTURE_STATUSES, standardize_team_name
from fotmob_prediction_features import enrich_probabilities_with_fotmob, read_fotmob_rolling_features, write_fotmob_coverage_report
from prediction_policy import apply_prediction_policy, load_policy


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _recent_totals(history: list[dict], window: int) -> dict[str, float]:
    recent = history[-window:]
    totals = {
        "matches": len(recent),
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


def _two_year_rates(history: list[dict], as_of_date: pd.Timestamp) -> dict[str, float]:
    if pd.isna(as_of_date):
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

    start_date = as_of_date - pd.DateOffset(years=2)
    recent = [match for match in history if start_date <= match["date"] < as_of_date]
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
            "date": match["date"],
            "points": _points(team_a_score, team_b_score),
            "goals_for": team_a_score,
            "goals_against": team_b_score,
            "importance": get_match_importance(match.get("tournament", "Other")),
        })
        history[team_b].append({
            "date": match["date"],
            "points": _points(team_b_score, team_a_score),
            "goals_for": team_b_score,
            "goals_against": team_a_score,
            "importance": get_match_importance(match.get("tournament", "Other")),
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
    all_dates = [
        match["date"]
        for team_history in history.values()
        for match in team_history
        if pd.notna(match.get("date"))
    ]
    as_of_date = max(all_dates) + pd.Timedelta(days=1) if all_dates else pd.Timestamp.today()
    team_a_2y = _two_year_rates(history[team_a], as_of_date)
    team_b_2y = _two_year_rates(history[team_b], as_of_date)

    row = {
        "elo_diff": ratings.get(team_a, 1500.0) - ratings.get(team_b, 1500.0),
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
    probabilities = pd.DataFrame([{
        0: probability_by_class.get(0, 0.0),
        1: probability_by_class.get(1, 0.0),
        2: probability_by_class.get(2, 0.0),
    }])
    policy = load_policy()
    predicted_class = int(apply_prediction_policy(probabilities, policy).iloc[0])

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win": probability_by_class.get(2, 0.0),
        "draw": probability_by_class.get(1, 0.0),
        "team_b_win": probability_by_class.get(0, 0.0),
        "predicted_class": predicted_class,
        "predicted_result": RESULT_LABELS.get(predicted_class, str(predicted_class)),
        "prediction_policy": policy.get("name", "argmax"),
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
            "predicted_class": prediction["predicted_class"],
            "predicted_result": prediction["predicted_result"],
            "prediction_policy": prediction["prediction_policy"],
        })

    predictions = pd.DataFrame(rows)
    predictions = merge_fotmob_rolling_features(predictions)
    predictions = merge_squad_features(predictions)
    predictions = enrich_probabilities_with_fotmob(predictions, policy=load_policy())
    write_fotmob_coverage_report(predictions)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "fixture_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Fixture predictions saved: {output_path} ({len(predictions)} rows)")
    return predictions


def merge_squad_features(predictions: pd.DataFrame) -> pd.DataFrame:
    squad_path = PROCESSED_DATA_DIR / "squad_features.csv"
    if predictions.empty or not squad_path.exists():
        return predictions

    squads = pd.read_csv(squad_path)
    if squads.empty or "team_name" not in squads.columns:
        return predictions

    columns = [
        "team_name",
        "squad_size",
        "squad_avg_age",
        "squad_median_age",
        "num_goalkeepers",
        "num_defenders",
        "num_midfielders",
        "num_forwards",
    ]
    available = [column for column in columns if column in squads.columns]
    squads = squads[available].copy()
    squads["team_name"] = squads["team_name"].apply(standardize_team_name)

    home = squads.rename(columns={
        "team_name": "home_team",
        "squad_size": "home_squad_size",
        "squad_avg_age": "home_squad_avg_age",
        "squad_median_age": "home_squad_median_age",
        "num_goalkeepers": "home_num_goalkeepers",
        "num_defenders": "home_num_defenders",
        "num_midfielders": "home_num_midfielders",
        "num_forwards": "home_num_forwards",
    })
    away = squads.rename(columns={
        "team_name": "away_team",
        "squad_size": "away_squad_size",
        "squad_avg_age": "away_squad_avg_age",
        "squad_median_age": "away_squad_median_age",
        "num_goalkeepers": "away_num_goalkeepers",
        "num_defenders": "away_num_defenders",
        "num_midfielders": "away_num_midfielders",
        "num_forwards": "away_num_forwards",
    })

    merged = predictions.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left")
    for column in ["squad_size", "squad_avg_age", "squad_median_age", "num_goalkeepers", "num_defenders", "num_midfielders", "num_forwards"]:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in merged.columns and away_column in merged.columns:
            merged[f"{column}_diff"] = merged[home_column] - merged[away_column]
    return merged


def merge_fotmob_rolling_features(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    rolling = read_fotmob_rolling_features()
    if rolling.empty or "fixture_id" not in rolling.columns:
        return predictions

    predictions = predictions.copy()
    predictions["fixture_id"] = predictions["fixture_id"].astype(str)
    rolling["fixture_id"] = rolling["fixture_id"].astype(str)
    keep_columns = [
        "fixture_id",
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
    available = [column for column in keep_columns if column in rolling.columns]
    merged = predictions.merge(rolling[available], on="fixture_id", how="left")
    return merged


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_manual_prediction(prediction: dict) -> None:
    print(f"{prediction['team_a']} vs {prediction['team_b']}")
    print()
    print(f"{prediction['team_a']} Win: {_percent(prediction['team_a_win'])}")
    print(f"Draw: {_percent(prediction['draw'])}")
    print(f"{prediction['team_b']} Win: {_percent(prediction['team_b_win'])}")
    print()
    print(f"Predicted result: {prediction['predicted_result']}")


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
        if "predicted_result" in row:
            print(f"Predicted result: {row['predicted_result']}")
        if "fotmob_momentum_score" in row and pd.notna(row["fotmob_momentum_score"]):
            leader = row["home_team"] if row["fotmob_momentum_score"] > 0 else row["away_team"]
            print(f"FotMob momentum: {leader} (+{abs(row['fotmob_momentum_score']):.2f})")
        if "fotmob_probability_shift" in row and abs(float(row["fotmob_probability_shift"])) > 0:
            print("FotMob feature layer:")
            print(f"Features used: {int(row.get('fotmob_feature_count', 0))} | Reliability: {float(row.get('fotmob_feature_reliability', 0.0)):.2f}")
            if pd.notna(row.get("fotmob_top_signals")) and str(row.get("fotmob_top_signals")).strip():
                print(f"Top signals: {row['fotmob_top_signals']}")
            print(f"{row['home_team']} Win: {_percent(row['fotmob_enriched_home_win_probability'])}")
            print(f"Draw: {_percent(row['fotmob_enriched_draw_probability'])}")
            print(f"{row['away_team']} Win: {_percent(row['fotmob_enriched_away_win_probability'])}")
            print(f"FotMob-enriched result: {row['fotmob_enriched_predicted_result']}")
        if "home_squad_avg_age" in row and pd.notna(row.get("home_squad_avg_age")) and pd.notna(row.get("away_squad_avg_age")):
            print("Squad comparison:")
            print(f"{row['home_team']} avg age: {row['home_squad_avg_age']:.1f} | {row['away_team']} avg age: {row['away_squad_avg_age']:.1f}")
            print(f"Forwards: {row['home_team']} {int(row['home_num_forwards'])} | {row['away_team']} {int(row['away_num_forwards'])}")
            print(f"Midfielders: {row['home_team']} {int(row['home_num_midfielders'])} | {row['away_team']} {int(row['away_num_midfielders'])}")
            print(f"Defenders: {row['home_team']} {int(row['home_num_defenders'])} | {row['away_team']} {int(row['away_num_defenders'])}")


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
