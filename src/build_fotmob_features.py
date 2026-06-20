import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


FEATURE_COLUMNS = [
    "fixture_id",
    "fotmob_match_id",
    "team_name",
    "starting_xi_avg_rating",
    "starting_xi_total_rating",
    "starting_xi_avg_minutes",
    "starting_xi_total_minutes",
    "substitutes_avg_rating",
    "formation",
    "team_goals",
    "team_assists",
    "team_shots",
    "team_shots_on_target",
    "team_chances_created",
    "team_expected_goals",
    "shots_on_target_rate",
    "goals_per_shot",
    "goals_minus_xg",
    "team_tackles",
    "team_interceptions",
    "team_clearances",
    "goals_conceded",
    "expected_goals_against",
    "yellow_cards",
    "red_cards",
    "best_player_rating",
    "worst_starter_rating",
    "avg_player_rating",
    "total_player_minutes",
    "goal_contributions",
    "substitute_goal_contributions",
]

LINEUP_FEATURE_COLUMNS = [
    "fixture_id",
    "fotmob_match_id",
    "team_name",
    "formation",
    "starting_xi_avg_rating",
    "starting_xi_total_rating",
    "starting_xi_avg_minutes",
    "starting_xi_total_minutes",
    "worst_starter_rating",
    "substitutes_avg_rating",
]


def _read_csv(path, columns=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return pd.read_csv(path)


def _safe_divide(left, right):
    if pd.isna(left) or pd.isna(right) or right == 0:
        return pd.NA
    return left / right


def _lineup_features(lineups: pd.DataFrame) -> pd.DataFrame:
    if lineups.empty:
        return pd.DataFrame(columns=["fixture_id", "fotmob_match_id", "team_name"])
    for column in ["rating", "minutes_played"]:
        lineups[column] = pd.to_numeric(lineups[column], errors="coerce")
    starters = lineups[lineups["is_starting"].astype(str).str.lower().isin(["true", "1"])]
    substitutes = lineups[lineups["is_substitute"].astype(str).str.lower().isin(["true", "1"])]

    base = lineups.groupby(["fixture_id", "fotmob_match_id", "team_name"], dropna=False).agg(
        formation=("formation", lambda values: next((value for value in values if pd.notna(value)), pd.NA)),
    ).reset_index()

    starter_features = starters.groupby(["fixture_id", "fotmob_match_id", "team_name"], dropna=False).agg(
        starting_xi_avg_rating=("rating", "mean"),
        starting_xi_total_rating=("rating", lambda values: values.sum(min_count=1)),
        starting_xi_avg_minutes=("minutes_played", "mean"),
        starting_xi_total_minutes=("minutes_played", lambda values: values.sum(min_count=1)),
        worst_starter_rating=("rating", "min"),
    ).reset_index()

    substitute_features = substitutes.groupby(["fixture_id", "fotmob_match_id", "team_name"], dropna=False).agg(
        substitutes_avg_rating=("rating", "mean"),
    ).reset_index()

    return base.merge(starter_features, how="left").merge(substitute_features, how="left")


def _player_features(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        return pd.DataFrame(columns=["fixture_id", "fotmob_match_id", "team_name"])
    numeric_columns = [
        "minutes",
        "rating",
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "chances_created",
        "tackles",
        "interceptions",
        "clearances",
        "yellow_card",
        "red_card",
    ]
    for column in numeric_columns:
        if column in players.columns:
            players[column] = pd.to_numeric(players[column], errors="coerce")

    grouped = players.groupby(["fixture_id", "fotmob_match_id", "team_name"], dropna=False).agg(
        team_assists=("assists", "sum"),
        team_shots=("shots", "sum"),
        team_shots_on_target=("shots_on_target", "sum"),
        team_chances_created=("chances_created", "sum"),
        team_tackles=("tackles", "sum"),
        team_interceptions=("interceptions", "sum"),
        team_clearances=("clearances", "sum"),
        yellow_cards=("yellow_card", "sum"),
        red_cards=("red_card", "sum"),
        best_player_rating=("rating", "max"),
        avg_player_rating=("rating", "mean"),
        total_player_minutes=("minutes", "sum"),
        goal_contributions=("goals", "sum"),
    ).reset_index()
    grouped["goal_contributions"] = grouped["goal_contributions"].fillna(0) + grouped["team_assists"].fillna(0)
    grouped["substitute_goal_contributions"] = pd.NA
    return grouped


def _team_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    if team_stats.empty:
        return pd.DataFrame(columns=["fixture_id", "fotmob_match_id", "team_name"])
    numeric_columns = [column for column in team_stats.columns if column not in ["team_name", "opponent_name"]]
    for column in numeric_columns:
        converted = pd.to_numeric(team_stats[column], errors="coerce")
        if converted.notna().any():
            team_stats[column] = converted

    features = team_stats.rename(columns={
        "goals_for": "team_goals",
        "expected_goals": "team_expected_goals",
        "goals_against": "goals_conceded",
        "yellow_cards": "yellow_cards_team",
        "red_cards": "red_cards_team",
    }).copy()
    features["shots_on_target_rate"] = features.apply(lambda row: _safe_divide(row.get("shots_on_target"), row.get("shots")), axis=1)
    features["goals_per_shot"] = features.apply(lambda row: _safe_divide(row.get("team_goals"), row.get("shots")), axis=1)
    features["goals_minus_xg"] = features["team_goals"] - features["team_expected_goals"]
    features["expected_goals_against"] = pd.NA

    keep = [
        "fixture_id",
        "fotmob_match_id",
        "team_name",
        "team_goals",
        "team_expected_goals",
        "goals_conceded",
        "shots_on_target_rate",
        "goals_per_shot",
        "goals_minus_xg",
        "expected_goals_against",
        "shots",
        "shots_on_target",
        "tackles",
        "interceptions",
        "clearances",
        "yellow_cards_team",
        "red_cards_team",
    ]
    return features[[column for column in keep if column in features.columns]].rename(columns={
        "shots": "team_shots_from_team_stats",
        "shots_on_target": "team_shots_on_target_from_team_stats",
        "tackles": "team_tackles_from_team_stats",
        "interceptions": "team_interceptions_from_team_stats",
        "clearances": "team_clearances_from_team_stats",
    })


def build_fotmob_features() -> pd.DataFrame:
    lineups = _read_csv(RAW_DATA_DIR / "fotmob_lineups.csv")
    players = _read_csv(RAW_DATA_DIR / "fotmob_player_match_stats.csv")
    team_stats = _read_csv(RAW_DATA_DIR / "fotmob_team_match_stats.csv")

    features = _lineup_features(lineups)
    for frame in [_player_features(players), _team_features(team_stats)]:
        if features.empty:
            features = frame
        elif not frame.empty:
            features = features.merge(frame, on=["fixture_id", "fotmob_match_id", "team_name"], how="outer")

    if features.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    for source, target in [
        ("team_shots_from_team_stats", "team_shots"),
        ("team_shots_on_target_from_team_stats", "team_shots_on_target"),
        ("team_tackles_from_team_stats", "team_tackles"),
        ("team_interceptions_from_team_stats", "team_interceptions"),
        ("team_clearances_from_team_stats", "team_clearances"),
        ("yellow_cards_team", "yellow_cards"),
        ("red_cards_team", "red_cards"),
    ]:
        if source in features.columns:
            if target not in features.columns:
                features[target] = features[source]
            else:
                features[target] = features[target].fillna(features[source])

    for column in FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = pd.NA

    return features[FEATURE_COLUMNS].sort_values(["fixture_id", "team_name"]).reset_index(drop=True)


def main() -> None:
    features = build_fotmob_features()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "fotmob_features.csv"
    features.to_csv(output_path, index=False)
    print(f"FotMob features saved: {output_path} ({len(features)} rows)")

    # Keep a dedicated Data Explorer artifact in sync with the same lineup
    # source used by the combined FotMob feature builder.
    lineups = _read_csv(RAW_DATA_DIR / "fotmob_lineups.csv")
    lineup_features = _lineup_features(lineups)
    for column in LINEUP_FEATURE_COLUMNS:
        if column not in lineup_features.columns:
            lineup_features[column] = pd.NA
    lineup_features = lineup_features[LINEUP_FEATURE_COLUMNS]
    lineup_output_path = PROCESSED_DATA_DIR / "lineup_features.csv"
    lineup_features.to_csv(lineup_output_path, index=False)
    print(f"Lineup features saved: {lineup_output_path} ({len(lineup_features)} rows)")


if __name__ == "__main__":
    main()
