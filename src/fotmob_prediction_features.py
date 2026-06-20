import pandas as pd

from config import PROCESSED_DATA_DIR, REPORTS_DIR, RESULT_LABELS
from prediction_policy import apply_prediction_policy


FOTMOB_PREDICTION_FEATURES = [
    "fotmob_points_per_match_diff",
    "fotmob_goal_difference_per_match_diff",
    "fotmob_avg_player_rating_diff",
    "fotmob_starting_xi_avg_rating_diff",
    "fotmob_shots_on_target_per_match_diff",
    "fotmob_expected_goals_per_match_diff",
    "fotmob_chances_created_per_match_diff",
    "fotmob_yellow_cards_per_match_diff",
    "fotmob_red_cards_per_match_diff",
]


FEATURE_SPECS = {
    "fotmob_points_per_match_diff": {"weight": 0.28, "scale": 2.0, "label": "points per match"},
    "fotmob_goal_difference_per_match_diff": {"weight": 0.25, "scale": 3.0, "label": "goal difference"},
    "fotmob_avg_player_rating_diff": {"weight": 0.16, "scale": 1.0, "label": "player rating"},
    "fotmob_starting_xi_avg_rating_diff": {"weight": 0.16, "scale": 1.0, "label": "starting XI rating"},
    "fotmob_shots_on_target_per_match_diff": {"weight": 0.08, "scale": 6.0, "label": "shots on target"},
    "fotmob_expected_goals_per_match_diff": {"weight": 0.12, "scale": 2.0, "label": "expected goals"},
    "fotmob_chances_created_per_match_diff": {"weight": 0.07, "scale": 10.0, "label": "chances created"},
    "fotmob_yellow_cards_per_match_diff": {"weight": -0.02, "scale": 3.0, "label": "yellow cards"},
    "fotmob_red_cards_per_match_diff": {"weight": -0.08, "scale": 1.0, "label": "red cards"},
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_probabilities(home: float, draw: float, away: float) -> tuple[float, float, float]:
    home = max(0.01, home)
    draw = max(0.01, draw)
    away = max(0.01, away)
    total = home + draw + away
    return home / total, draw / total, away / total


def _feature_contributions(row: pd.Series) -> list[dict]:
    contributions = []
    for column, spec in FEATURE_SPECS.items():
        value = row.get(column)
        if pd.isna(value):
            continue

        normalized = _clamp(float(value) / spec["scale"], -1.0, 1.0)
        contribution = normalized * spec["weight"]
        contributions.append({
            "feature": column,
            "label": spec["label"],
            "value": float(value),
            "contribution": contribution,
        })
    return contributions


def _reliability(row: pd.Series, feature_count: int) -> float:
    home_matches = row.get("home_fotmob_matches_before", 0)
    away_matches = row.get("away_fotmob_matches_before", 0)
    if pd.isna(home_matches) or pd.isna(away_matches):
        return 0.0

    common_matches = min(float(home_matches), float(away_matches))
    if common_matches < 1 or feature_count < 2:
        return 0.0

    match_reliability = _clamp(common_matches / 2.0, 0.35, 1.0)
    feature_reliability = _clamp(feature_count / max(len(FOTMOB_PREDICTION_FEATURES), 1), 0.25, 1.0)
    return match_reliability * feature_reliability


def _confidence_damping(row: pd.Series) -> float:
    base_home = float(row.get("home_win_probability", 0.0))
    base_away = float(row.get("away_win_probability", 0.0))
    favorite_strength = max(base_home, base_away)
    return _clamp(1.0 - max(0.0, favorite_strength - 0.65), 0.55, 1.0)


def _top_signal(contributions: list[dict]) -> str:
    if not contributions:
        return ""
    strongest = sorted(contributions, key=lambda item: abs(item["contribution"]), reverse=True)[:3]
    labels = []
    for item in strongest:
        side = "home" if item["contribution"] > 0 else "away"
        labels.append(f"{item['label']}:{side}")
    return "; ".join(labels)


def enrich_probabilities_with_fotmob(predictions: pd.DataFrame, policy: dict | None = None) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    output = predictions.copy()
    rows = []
    for _, row in output.iterrows():
        contributions = _feature_contributions(row)
        feature_count = len(contributions)
        reliability = _reliability(row, feature_count)

        if reliability == 0.0:
            score = 0.0
            shift = 0.0
        else:
            weight_total = sum(abs(FEATURE_SPECS[item["feature"]]["weight"]) for item in contributions)
            score = sum(item["contribution"] for item in contributions) / weight_total if weight_total else 0.0
            score = _clamp(score, -1.0, 1.0)
            shift = _clamp(score * reliability * _confidence_damping(row) * 0.10, -0.10, 0.10)

        base_home = float(row["home_win_probability"])
        base_draw = float(row["draw_probability"])
        base_away = float(row["away_win_probability"])

        draw_shift = -abs(shift) * 0.20
        enriched_home, enriched_draw, enriched_away = _normalize_probabilities(
            base_home + shift,
            base_draw + draw_shift,
            base_away - shift,
        )

        probability_frame = pd.DataFrame([{0: enriched_away, 1: enriched_draw, 2: enriched_home}])
        enriched_class = int(apply_prediction_policy(probability_frame, policy).iloc[0]) if policy else int(probability_frame.iloc[0].idxmax())

        rows.append({
            "fotmob_feature_count": feature_count,
            "fotmob_feature_reliability": reliability,
            "fotmob_enrichment_score": score,
            "fotmob_probability_shift": shift,
            "fotmob_top_signals": _top_signal(contributions),
            "fotmob_enriched_home_win_probability": enriched_home,
            "fotmob_enriched_draw_probability": enriched_draw,
            "fotmob_enriched_away_win_probability": enriched_away,
            "fotmob_enriched_predicted_class": enriched_class,
            "fotmob_enriched_predicted_result": RESULT_LABELS.get(enriched_class, str(enriched_class)),
            # Backward-compatible names kept for existing reports/scripts.
            "fotmob_adjusted_home_win_probability": enriched_home,
            "fotmob_adjusted_draw_probability": enriched_draw,
            "fotmob_adjusted_away_win_probability": enriched_away,
        })

    return pd.concat([output, pd.DataFrame(rows, index=output.index)], axis=1)


def write_fotmob_coverage_report(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return

    rows = []
    for column in FOTMOB_PREDICTION_FEATURES:
        if column in predictions.columns:
            rows.append({
                "feature": column,
                "available_rows": int(predictions[column].notna().sum()),
                "total_rows": int(len(predictions)),
                "coverage": float(predictions[column].notna().mean()),
            })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "fotmob_feature_coverage.csv", index=False)


def read_fotmob_rolling_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "fotmob_rolling_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
