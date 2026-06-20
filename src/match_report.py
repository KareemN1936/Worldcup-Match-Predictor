import argparse
import re
from pathlib import Path

import pandas as pd

from config import REPORTS_DIR, standardize_team_name
from predict_match import predict_fixtures


REPORT_DIR = REPORTS_DIR / "match_reports"


def _percent(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _load_predictions(refresh: bool) -> pd.DataFrame:
    path = REPORTS_DIR / "fixture_predictions.csv"
    if refresh or not path.exists():
        return predict_fixtures(only_upcoming=True)
    return pd.read_csv(path)


def _find_match(predictions: pd.DataFrame, team_a: str, team_b: str) -> pd.Series:
    team_a = standardize_team_name(team_a)
    team_b = standardize_team_name(team_b)

    direct = predictions[
        (predictions["home_team"].apply(standardize_team_name) == team_a)
        & (predictions["away_team"].apply(standardize_team_name) == team_b)
    ]
    reverse = predictions[
        (predictions["home_team"].apply(standardize_team_name) == team_b)
        & (predictions["away_team"].apply(standardize_team_name) == team_a)
    ]

    matches = pd.concat([direct, reverse], ignore_index=True)
    if matches.empty:
        available = predictions[["home_team", "away_team", "date"]].head(12).to_string(index=False)
        raise ValueError(f"No fixture prediction found for {team_a} vs {team_b}.\n\nSample available fixtures:\n{available}")
    return matches.iloc[0]


def _reason_lines(row: pd.Series) -> list[str]:
    reasons = []
    if pd.notna(row.get("fotmob_momentum_score")):
        leader = row["home_team"] if row["fotmob_momentum_score"] > 0 else row["away_team"]
        reasons.append(f"- {leader} have the better recent FotMob tournament momentum signal.")

    if pd.notna(row.get("fotmob_points_per_match_diff")):
        leader = row["home_team"] if row["fotmob_points_per_match_diff"] > 0 else row["away_team"]
        reasons.append(f"- {leader} have earned more points per match in prior FotMob-tracked World Cup games.")

    if pd.notna(row.get("fotmob_avg_player_rating_diff")):
        leader = row["home_team"] if row["fotmob_avg_player_rating_diff"] > 0 else row["away_team"]
        reasons.append(f"- {leader} have the stronger average player-rating trend so far.")

    if pd.notna(row.get("squad_avg_age_diff")):
        younger = row["home_team"] if row["squad_avg_age_diff"] < 0 else row["away_team"]
        reasons.append(f"- {younger} have the younger squad profile.")

    if pd.notna(row.get("num_midfielders_diff")) and row.get("num_midfielders_diff") != 0:
        leader = row["home_team"] if row["num_midfielders_diff"] > 0 else row["away_team"]
        reasons.append(f"- {leader} carry more listed midfielders in the squad.")

    if not reasons:
        reasons.append("- No extra FotMob or squad context was available; use the base historical model as the main signal.")
    return reasons


def build_report(row: pd.Series) -> str:
    home = row["home_team"]
    away = row["away_team"]
    lines = [
        f"{home} vs {away}",
        "",
        f"Date: {row.get('date', 'n/a')}",
        f"Round: {row.get('round', 'n/a')}",
        "",
        "Base Prediction:",
        f"{home} Win: {_percent(row['home_win_probability'])}",
        f"Draw: {_percent(row['draw_probability'])}",
        f"{away} Win: {_percent(row['away_win_probability'])}",
        "",
    ]

    if "fotmob_adjusted_home_win_probability" in row:
        lines.extend([
            "FotMob-Adjusted Prediction:",
            f"{home} Win: {_percent(row['fotmob_adjusted_home_win_probability'])}",
            f"Draw: {_percent(row['fotmob_adjusted_draw_probability'])}",
            f"{away} Win: {_percent(row['fotmob_adjusted_away_win_probability'])}",
            f"Adjustment shift: {_num(row.get('fotmob_probability_shift'), 4)}",
            "",
        ])

    lines.extend([
        "FotMob Momentum:",
        f"Matches before: {home} {_num(row.get('home_fotmob_matches_before'), 0)} | {away} {_num(row.get('away_fotmob_matches_before'), 0)}",
        f"Momentum score: {_num(row.get('fotmob_momentum_score'), 2)}",
        f"Points/match diff: {_num(row.get('fotmob_points_per_match_diff'), 2)}",
        f"Goal-difference/match diff: {_num(row.get('fotmob_goal_difference_per_match_diff'), 2)}",
        "",
    ])

    if pd.notna(row.get("home_squad_avg_age")) and pd.notna(row.get("away_squad_avg_age")):
        lines.extend([
            "Squad Comparison:",
            f"Average age: {home} {_num(row.get('home_squad_avg_age'), 1)} | {away} {_num(row.get('away_squad_avg_age'), 1)}",
            f"Forwards: {home} {_num(row.get('home_num_forwards'), 0)} | {away} {_num(row.get('away_num_forwards'), 0)}",
            f"Midfielders: {home} {_num(row.get('home_num_midfielders'), 0)} | {away} {_num(row.get('away_num_midfielders'), 0)}",
            f"Defenders: {home} {_num(row.get('home_num_defenders'), 0)} | {away} {_num(row.get('away_num_defenders'), 0)}",
            "",
        ])

    lines.extend(["Why:", *_reason_lines(row)])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a readable report for one World Cup fixture prediction.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--refresh", action="store_true", help="Regenerate fixture predictions before building the report.")
    args = parser.parse_args()

    predictions = _load_predictions(refresh=args.refresh)
    row = _find_match(predictions, args.team_a, args.team_b)
    report = build_report(row)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"{_slug(row['home_team'])}_vs_{_slug(row['away_team'])}.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print()
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
