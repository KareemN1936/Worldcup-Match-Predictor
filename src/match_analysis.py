import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from config import REPORTS_DIR, standardize_team_name
from predict_match import build_latest_team_state, build_prediction_features, predict, predict_fixtures


ANALYSIS_DIR = REPORTS_DIR / "match_analysis"


HISTORICAL_FACTORS = [
    ("elo_diff", "Elo advantage"),
    ("points_last_5_diff", "points last 5"),
    ("points_last_10_diff", "points last 10"),
    ("goal_difference_last_5_diff", "goal difference last 5"),
    ("goals_for_last_5_diff", "goals for last 5"),
    ("goals_against_last_5_diff", "goals against last 5"),
    ("weighted_points_last_5_diff", "weighted points last 5"),
    ("win_rate_2y_diff", "2-year win rate"),
    ("goal_difference_per_match_2y_diff", "2-year goal difference per match"),
    ("goals_against_per_match_2y_diff", "2-year goals against per match"),
    ("clean_sheet_rate_2y_diff", "2-year clean-sheet rate"),
    ("failed_to_score_rate_2y_diff", "2-year failed-to-score rate"),
]


FOTMOB_FACTORS = [
    ("fotmob_points_per_match_diff", "points per match"),
    ("fotmob_goal_difference_per_match_diff", "goal difference per match"),
    ("fotmob_avg_player_rating_diff", "average player rating"),
    ("fotmob_starting_xi_avg_rating_diff", "starting XI rating"),
    ("fotmob_shots_on_target_per_match_diff", "shots on target per match"),
    ("fotmob_expected_goals_per_match_diff", "expected goals per match"),
    ("fotmob_chances_created_per_match_diff", "chances created per match"),
    ("fotmob_yellow_cards_per_match_diff", "yellow cards per match"),
    ("fotmob_red_cards_per_match_diff", "red cards per match"),
]


SQUAD_FACTORS = [
    ("squad_avg_age_diff", "average age"),
    ("squad_size_diff", "squad size"),
    ("num_forwards_diff", "listed forwards"),
    ("num_midfielders_diff", "listed midfielders"),
    ("num_defenders_diff", "listed defenders"),
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _percent(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _number(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _signed(value: Any, home: str, away: str, lower_is_better: bool = False) -> dict:
    if pd.isna(value):
        return {"value": None, "leader": None, "direction": "missing"}

    numeric = float(value)
    if numeric == 0:
        return {"value": numeric, "leader": "Even", "direction": "even"}

    home_leads = numeric > 0
    if lower_is_better:
        home_leads = not home_leads
    return {
        "value": numeric,
        "leader": home if home_leads else away,
        "direction": "home" if home_leads else "away",
    }


def _load_predictions(refresh: bool, all_fixtures: bool) -> pd.DataFrame:
    path = REPORTS_DIR / "fixture_predictions.csv"
    if refresh or not path.exists():
        return predict_fixtures(only_upcoming=not all_fixtures)
    predictions = pd.read_csv(path)
    if all_fixtures:
        return predictions
    return predictions


@lru_cache(maxsize=1)
def _cached_team_state():
    return build_latest_team_state()


def _find_fixture(predictions: pd.DataFrame, team_a: str, team_b: str) -> pd.Series | None:
    if predictions.empty:
        return None

    team_a = standardize_team_name(team_a)
    team_b = standardize_team_name(team_b)
    home_names = predictions["home_team"].apply(standardize_team_name)
    away_names = predictions["away_team"].apply(standardize_team_name)

    direct = predictions[(home_names == team_a) & (away_names == team_b)]
    reverse = predictions[(home_names == team_b) & (away_names == team_a)]
    matches = pd.concat([direct, reverse])
    if matches.empty:
        return None
    return matches.iloc[0]


def _historical_features(row: pd.Series) -> dict:
    tournament = row.get("competition", "FIFA World Cup")
    if pd.isna(tournament):
        tournament = "FIFA World Cup"
    features = build_prediction_features(row["home_team"], row["away_team"], True, str(tournament), state=_cached_team_state())
    return {column: _safe_value(value) for column, value in features.iloc[0].to_dict().items()}


def _manual_row(team_a: str, team_b: str) -> pd.Series:
    prediction = predict(team_a, team_b, neutral=True, tournament="FIFA World Cup")
    return pd.Series({
        "fixture_id": None,
        "date": None,
        "round": None,
        "group": None,
        "home_team": prediction["team_a"],
        "away_team": prediction["team_b"],
        "home_win_probability": prediction["team_a_win"],
        "draw_probability": prediction["draw"],
        "away_win_probability": prediction["team_b_win"],
        "predicted_result": prediction["predicted_result"],
        "prediction_policy": prediction["prediction_policy"],
    })


def _factor_table(features: dict, row: pd.Series, factors: list[tuple[str, str]], home: str, away: str, lower_is_better: set[str] | None = None) -> list[dict]:
    lower_is_better = lower_is_better or set()
    output = []
    for column, label in factors:
        value = features.get(column, row.get(column))
        item = _signed(value, home, away, lower_is_better=column in lower_is_better)
        output.append({
            "feature": column,
            "label": label,
            "value": item["value"],
            "leader": item["leader"],
        })
    return output


def _strongest_factors(factors: list[dict], limit: int = 5) -> list[dict]:
    available = [factor for factor in factors if factor["value"] is not None]
    return sorted(available, key=lambda factor: abs(float(factor["value"])), reverse=True)[:limit]


def _recommendation(row: pd.Series) -> dict:
    reliability = row.get("fotmob_feature_reliability")
    base_result = row.get("predicted_result", "n/a")
    enriched_result = row.get("fotmob_enriched_predicted_result", base_result)
    if pd.notna(reliability) and float(reliability) >= 0.50:
        return {
            "result_to_show": enriched_result,
            "source": "fotmob_enriched",
            "reason": "FotMob reliability is medium/high, so use the enriched prediction while still showing the base model.",
        }
    return {
        "result_to_show": base_result,
        "source": "base_model",
        "reason": "FotMob reliability is low or unavailable, so use the base historical model as the main prediction.",
    }


def build_analysis(row: pd.Series) -> dict:
    home = row["home_team"]
    away = row["away_team"]
    historical_features = _historical_features(row)
    historical_factors = _factor_table(
        historical_features,
        row,
        HISTORICAL_FACTORS,
        home,
        away,
        lower_is_better={"goals_against_last_5_diff", "goals_against_per_match_2y_diff", "failed_to_score_rate_2y_diff"},
    )
    fotmob_factors = _factor_table(
        {},
        row,
        FOTMOB_FACTORS,
        home,
        away,
        lower_is_better={"fotmob_yellow_cards_per_match_diff", "fotmob_red_cards_per_match_diff"},
    )
    squad_factors = _factor_table({}, row, SQUAD_FACTORS, home, away, lower_is_better={"squad_avg_age_diff"})
    recommendation = _recommendation(row)

    return {
        "fixture": {
            "fixture_id": _safe_value(row.get("fixture_id")),
            "date": _safe_value(row.get("date")),
            "round": _safe_value(row.get("round")),
            "group": _safe_value(row.get("group")),
            "home_team": home,
            "away_team": away,
        },
        "base_prediction": {
            "home_win_probability": _safe_value(row.get("home_win_probability")),
            "draw_probability": _safe_value(row.get("draw_probability")),
            "away_win_probability": _safe_value(row.get("away_win_probability")),
            "predicted_result": _safe_value(row.get("predicted_result")),
            "prediction_policy": _safe_value(row.get("prediction_policy")),
        },
        "historical_model_factors": {
            "features": historical_features,
            "strongest": _strongest_factors(historical_factors),
            "all": historical_factors,
        },
        "fotmob_layer": {
            "available": int(row.get("fotmob_feature_count", 0) or 0) > 0,
            "feature_count": _safe_value(row.get("fotmob_feature_count")),
            "reliability": _safe_value(row.get("fotmob_feature_reliability")),
            "enrichment_score": _safe_value(row.get("fotmob_enrichment_score")),
            "probability_shift": _safe_value(row.get("fotmob_probability_shift")),
            "top_signals": _safe_value(row.get("fotmob_top_signals")),
            "home_win_probability": _safe_value(row.get("fotmob_enriched_home_win_probability")),
            "draw_probability": _safe_value(row.get("fotmob_enriched_draw_probability")),
            "away_win_probability": _safe_value(row.get("fotmob_enriched_away_win_probability")),
            "predicted_result": _safe_value(row.get("fotmob_enriched_predicted_result")),
            "strongest": _strongest_factors(fotmob_factors),
            "all": fotmob_factors,
        },
        "squad_context": {
            "available": pd.notna(row.get("home_squad_avg_age")) and pd.notna(row.get("away_squad_avg_age")),
            "home_avg_age": _safe_value(row.get("home_squad_avg_age")),
            "away_avg_age": _safe_value(row.get("away_squad_avg_age")),
            "home_forwards": _safe_value(row.get("home_num_forwards")),
            "away_forwards": _safe_value(row.get("away_num_forwards")),
            "home_midfielders": _safe_value(row.get("home_num_midfielders")),
            "away_midfielders": _safe_value(row.get("away_num_midfielders")),
            "home_defenders": _safe_value(row.get("home_num_defenders")),
            "away_defenders": _safe_value(row.get("away_num_defenders")),
            "strongest": _strongest_factors(squad_factors),
            "all": squad_factors,
        },
        "final_recommendation": recommendation,
    }


def render_text(analysis: dict) -> str:
    fixture = analysis["fixture"]
    home = fixture["home_team"]
    away = fixture["away_team"]
    base = analysis["base_prediction"]
    fotmob = analysis["fotmob_layer"]
    squad = analysis["squad_context"]
    recommendation = analysis["final_recommendation"]

    lines = [
        f"{home} vs {away}",
        "",
        f"Date: {fixture.get('date') or 'n/a'}",
        f"Round: {fixture.get('round') or 'n/a'}",
        f"Group: {fixture.get('group') or 'n/a'}",
        "",
        "Base Prediction:",
        f"{home} Win: {_percent(base['home_win_probability'])}",
        f"Draw: {_percent(base['draw_probability'])}",
        f"{away} Win: {_percent(base['away_win_probability'])}",
        f"Predicted result: {base.get('predicted_result') or 'n/a'}",
        f"Decision policy: {base.get('prediction_policy') or 'n/a'}",
        "",
        "Historical Model Factors:",
    ]

    for factor in analysis["historical_model_factors"]["strongest"]:
        leader = factor["leader"] or "n/a"
        lines.append(f"- {factor['label']}: {_number(factor['value'])} | edge: {leader}")

    lines.extend(["", "FotMob Layer:"])
    if fotmob["available"]:
        lines.extend([
            f"Features available: {fotmob.get('feature_count')}",
            f"Reliability: {_number(fotmob.get('reliability'))}",
            f"Enrichment score: {_number(fotmob.get('enrichment_score'))}",
            f"Probability shift: {_number(fotmob.get('probability_shift'), 4)}",
            f"Top signals: {fotmob.get('top_signals') or 'n/a'}",
            f"{home} Win: {_percent(fotmob.get('home_win_probability'))}",
            f"Draw: {_percent(fotmob.get('draw_probability'))}",
            f"{away} Win: {_percent(fotmob.get('away_win_probability'))}",
            f"FotMob-enriched result: {fotmob.get('predicted_result') or 'n/a'}",
        ])
    else:
        lines.append("No reliable pre-match FotMob rolling data is available for both teams yet.")

    lines.extend(["", "Squad Context:"])
    if squad["available"]:
        lines.extend([
            f"Average age: {home} {_number(squad.get('home_avg_age'), 1)} | {away} {_number(squad.get('away_avg_age'), 1)}",
            f"Forwards: {home} {_number(squad.get('home_forwards'), 0)} | {away} {_number(squad.get('away_forwards'), 0)}",
            f"Midfielders: {home} {_number(squad.get('home_midfielders'), 0)} | {away} {_number(squad.get('away_midfielders'), 0)}",
            f"Defenders: {home} {_number(squad.get('home_defenders'), 0)} | {away} {_number(squad.get('away_defenders'), 0)}",
        ])
    else:
        lines.append("No squad context is available for this match.")

    lines.extend([
        "",
        "Final Recommendation:",
        f"Use: {recommendation['result_to_show']}",
        f"Source: {recommendation['source']}",
        f"Reason: {recommendation['reason']}",
    ])
    return "\n".join(lines)


def save_analysis(analysis: dict) -> tuple[Path, Path]:
    fixture = analysis["fixture"]
    filename = f"{_slug(fixture['home_team'])}_vs_{_slug(fixture['away_team'])}"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    text_path = ANALYSIS_DIR / f"{filename}.txt"
    json_path = ANALYSIS_DIR / f"{filename}.json"
    text_path.write_text(render_text(analysis), encoding="utf-8")
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return text_path, json_path


def write_fixture_analyses(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in predictions.iterrows():
        analysis = build_analysis(row)
        text_path, json_path = save_analysis(analysis)
        rows.append({
            "home_team": analysis["fixture"]["home_team"],
            "away_team": analysis["fixture"]["away_team"],
            "date": analysis["fixture"]["date"],
            "base_result": analysis["base_prediction"]["predicted_result"],
            "fotmob_result": analysis["fotmob_layer"]["predicted_result"],
            "recommended_result": analysis["final_recommendation"]["result_to_show"],
            "recommendation_source": analysis["final_recommendation"]["source"],
            "text_report": str(text_path),
            "json_report": str(json_path),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(ANALYSIS_DIR / "fixture_analysis_index.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model analysis reports for World Cup predictions.")
    parser.add_argument("team_a", nargs="?", help="First team, for example France")
    parser.add_argument("team_b", nargs="?", help="Second team, for example Iraq")
    parser.add_argument("--fixtures", action="store_true", help="Create analysis reports for all fixture predictions.")
    parser.add_argument("--refresh", action="store_true", help="Regenerate fixture predictions before analysis.")
    args = parser.parse_args()

    predictions = _load_predictions(refresh=args.refresh, all_fixtures=True)

    if args.fixtures:
        if predictions.empty:
            raise ValueError("No fixture predictions found. Run python src/predict_match.py --fixtures first.")
        summary = write_fixture_analyses(predictions)
        print(f"Match analysis reports saved: {ANALYSIS_DIR} ({len(summary)} matches)")
        print(f"Index saved: {ANALYSIS_DIR / 'fixture_analysis_index.csv'}")
        return

    if not args.team_a or not args.team_b:
        parser.error("Provide TEAM_A TEAM_B, or use --fixtures.")

    row = _find_fixture(predictions, args.team_a, args.team_b)
    if row is None:
        row = _manual_row(args.team_a, args.team_b)

    analysis = build_analysis(row)
    text_path, json_path = save_analysis(analysis)
    print(render_text(analysis))
    print()
    print(f"Text report saved: {text_path}")
    print(f"JSON report saved: {json_path}")


if __name__ == "__main__":
    main()
