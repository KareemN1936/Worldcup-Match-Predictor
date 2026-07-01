import math
from typing import Any

import pandas as pd

from .data_loader import is_knockout_round


def _is_number(value: Any) -> bool:
    try:
        return value is not None and not pd.isna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def confidence_from_probabilities(home_win: float | None, draw: float | None, away_win: float | None) -> str:
    values = [value for value in [home_win, draw, away_win] if _is_number(value)]
    if len(values) != 3:
        return "Unavailable"
    ordered = sorted([float(value) for value in values], reverse=True)
    spread = ordered[0] - ordered[1]
    if spread >= 0.25:
        return "High"
    if spread >= 0.12:
        return "Medium"
    return "Low"


def most_likely_result(row: pd.Series) -> str:
    home = row.get("display_home_win_probability", row.get("home_win_probability"))
    draw = row.get("display_draw_probability", row.get("draw_probability"))
    away = row.get("display_away_win_probability", row.get("away_win_probability"))
    if not all(_is_number(value) for value in [home, draw, away]):
        return "Prediction unavailable"

    if is_knockout_round(row.get("round")):
        winner = row.get("display_predicted_winner")
        if isinstance(winner, str) and winner.strip():
            return f"{winner.strip()} win"
        values = {
            f"{row.get('home_team', 'Team A')} win": float(home),
            f"{row.get('away_team', 'Team B')} win": float(away),
        }
        return max(values, key=values.get)

    values = {
        f"{row.get('home_team', 'Team A')} win": float(home),
        "Draw": float(draw),
        f"{row.get('away_team', 'Team B')} win": float(away),
    }
    return max(values, key=values.get)


def _leader_from_diff(value: float, home_team: str, away_team: str, lower_is_better: bool = False) -> str:
    if abs(value) < 1e-9:
        return "Neither team"
    home_leads = value > 0
    if lower_is_better:
        home_leads = value < 0
    return home_team if home_leads else away_team


def build_model_notes(row: pd.Series, analysis: dict | None = None, limit: int = 5) -> list[str]:
    home_team = str(row.get("home_team", "Team A"))
    away_team = str(row.get("away_team", "Team B"))
    notes: list[str] = []

    if analysis:
        strongest = analysis.get("historical_model_factors", {}).get("strongest", [])
        for factor in strongest[:3]:
            leader = factor.get("leader")
            label = factor.get("label")
            value = factor.get("value")
            if leader and leader != "Even" and label and _is_number(value):
                notes.append(f"{leader} leads on {label} ({float(value):.2f} difference).")

    fotmob_signals = row.get("fotmob_top_signals")
    if isinstance(fotmob_signals, str) and fotmob_signals.strip():
        notes.append(f"FotMob layer signals: {fotmob_signals.strip()}.")

    feature_notes = [
        ("fotmob_goal_difference_per_match_diff", "FotMob goal difference per match", False),
        ("fotmob_shots_on_target_per_match_diff", "FotMob shots on target per match", False),
        ("fotmob_chances_created_per_match_diff", "FotMob chances created per match", False),
        ("squad_size_diff", "listed squad size", False),
        ("num_forwards_diff", "listed forwards", False),
        ("num_midfielders_diff", "listed midfielders", False),
        ("num_defenders_diff", "listed defenders", False),
        ("squad_avg_age_diff", "younger average squad age", True),
    ]
    for column, label, lower_is_better in feature_notes:
        value = row.get(column)
        if _is_number(value) and abs(float(value)) > 0:
            leader = _leader_from_diff(float(value), home_team, away_team, lower_is_better)
            notes.append(f"{leader} has the edge in {label}.")
        if len(notes) >= limit - 1:
            break

    home = row.get("display_home_win_probability", row.get("home_win_probability"))
    draw = row.get("display_draw_probability", row.get("draw_probability"))
    away = row.get("display_away_win_probability", row.get("away_win_probability"))
    confidence = confidence_from_probabilities(home, draw, away)
    if _is_number(draw) and float(draw) >= 0.30:
        notes.append("The model is cautious because the draw probability is high.")
    elif confidence == "Low":
        notes.append("This prediction has low confidence because the top outcomes are close.")
    elif confidence == "High":
        notes.append("The model shows high confidence because the top outcome is well ahead.")

    if not notes:
        notes.append("No strong explanatory features are available for this match yet.")
    return notes[:limit]
