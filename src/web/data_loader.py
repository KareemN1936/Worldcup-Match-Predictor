import json
import re
import ast
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


RESULT_TEAM_ALIASES = {
    "bosniaandherzegovina": "bosniaherzegovina",
    "drcongo": "congodr",
    "turkiye": "turkey",
}

KNOCKOUT_ROUNDS = {
    "LAST_32",
    "LAST_16",
    "ROUND_OF_32",
    "ROUND_OF_16",
    "QUARTER_FINALS",
    "SEMI_FINALS",
    "THIRD_PLACE",
    "FINAL",
}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _result_team_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    return RESULT_TEAM_ALIASES.get(compact, compact)


def _cached_fotmob_results() -> pd.DataFrame:
    """Read final scores from cached FotMob date feeds when detail matching failed."""
    rows: list[dict[str, Any]] = []
    matches_dir = RAW_DIR / "fotmob" / "matches"
    for path in matches_dir.glob("date_*.json") if matches_dir.exists() else []:
        data = read_json_if_exists(path)
        for league in data.get("leagues", []) if isinstance(data.get("leagues"), list) else []:
            for match in league.get("matches", []) if isinstance(league.get("matches"), list) else []:
                home = match.get("home", {})
                away = match.get("away", {})
                status = match.get("status", {})
                score = status.get("scoreStr") if isinstance(status, dict) else None
                utc_time = status.get("utcTime") if isinstance(status, dict) else None
                if (
                    not isinstance(home, dict)
                    or not isinstance(away, dict)
                    or not isinstance(status, dict)
                    or not status.get("finished")
                    or not isinstance(score, str)
                    or "-" not in score
                ):
                    continue
                score_parts = [part.strip() for part in score.split("-", 1)]
                try:
                    home_score, away_score = float(score_parts[0]), float(score_parts[1])
                except ValueError:
                    continue
                date_key = pd.to_datetime(utc_time or data.get("date"), errors="coerce")
                if pd.isna(date_key):
                    continue
                rows.append(
                    {
                        "_result_key": (
                            date_key.strftime("%Y-%m-%d"),
                            _result_team_key(home.get("name")),
                            _result_team_key(away.get("name")),
                        ),
                        "home_score_cached": home_score,
                        "away_score_cached": away_score,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("_result_key", keep="last") if rows else pd.DataFrame()


def load_fixtures() -> pd.DataFrame:
    fixtures = read_csv_if_exists(RAW_DIR / "fixtures.csv")
    match_details = read_csv_if_exists(RAW_DIR / "fotmob_match_details.csv")

    if not fixtures.empty and not match_details.empty and "fixture_id" in fixtures.columns and "fixture_id" in match_details.columns:
        fixtures = fixtures.copy()
        detail_columns = ["fixture_id", "home_score", "away_score"]
        if "fotmob_match_id" in match_details.columns:
            detail_columns.append("fotmob_match_id")
        if "status" in match_details.columns:
            detail_columns.append("status")
        for column in ["home_penalty_score", "away_penalty_score", "decided_by_penalties", "penalty_loser"]:
            if column in match_details.columns:
                detail_columns.append(column)
        details = match_details[detail_columns].drop_duplicates("fixture_id", keep="last").copy()
        if "status" in details.columns:
            penalties = details["status"].apply(_extract_penalty_scores)
            if "home_penalty_score" not in details.columns:
                details["home_penalty_score"] = pd.NA
            if "away_penalty_score" not in details.columns:
                details["away_penalty_score"] = pd.NA
            if "decided_by_penalties" not in details.columns:
                details["decided_by_penalties"] = pd.NA
            if "penalty_loser" not in details.columns:
                details["penalty_loser"] = pd.NA
            details["home_penalty_score"] = details["home_penalty_score"].combine_first(penalties.map(lambda scores: scores[0]))
            details["away_penalty_score"] = details["away_penalty_score"].combine_first(penalties.map(lambda scores: scores[1]))
            details["decided_by_penalties"] = details["decided_by_penalties"].combine_first(details["status"].apply(_is_after_penalties))
            details["penalty_loser"] = details["penalty_loser"].combine_first(details["status"].apply(_extract_penalty_loser))
            if "fotmob_match_id" in details.columns:
                for index, row in details.iterrows():
                    full_status = _match_json_status(row.get("fotmob_match_id"))
                    if not full_status:
                        continue
                    full_penalties = _extract_penalty_scores(full_status)
                    if pd.isna(details.at[index, "home_penalty_score"]) and full_penalties[0] is not None:
                        details.at[index, "home_penalty_score"] = full_penalties[0]
                    if pd.isna(details.at[index, "away_penalty_score"]) and full_penalties[1] is not None:
                        details.at[index, "away_penalty_score"] = full_penalties[1]
                    if not bool(details.at[index, "decided_by_penalties"]):
                        details.at[index, "decided_by_penalties"] = _is_after_penalties(full_status)
                    if pd.isna(details.at[index, "penalty_loser"]):
                        details.at[index, "penalty_loser"] = _extract_penalty_loser(full_status)
            details = details.drop(columns=["status"])
        details = details.drop(columns=["fotmob_match_id"], errors="ignore")
        fixtures["fixture_id"] = fixtures["fixture_id"].astype(str)
        details["fixture_id"] = details["fixture_id"].astype(str)
        fixtures = fixtures.merge(details, on="fixture_id", how="left", suffixes=("", "_fotmob"))
        for score_column in ["home_score", "away_score", "home_penalty_score", "away_penalty_score", "decided_by_penalties", "penalty_loser"]:
            fotmob_column = f"{score_column}_fotmob"
            if fotmob_column in fixtures.columns:
                if score_column not in fixtures.columns:
                    fixtures[score_column] = fixtures[fotmob_column]
                else:
                    fixtures[score_column] = fixtures[score_column].combine_first(fixtures[fotmob_column])
                fixtures = fixtures.drop(columns=[fotmob_column])

    if not fixtures.empty:
        cached_results = _cached_fotmob_results()
        if not cached_results.empty:
            fixture_dates = pd.to_datetime(fixtures.get("date"), errors="coerce")
            fixtures["_result_key"] = [
                (
                    date.strftime("%Y-%m-%d") if pd.notna(date) else "",
                    _result_team_key(home),
                    _result_team_key(away),
                )
                for date, home, away in zip(fixture_dates, fixtures.get("home_team", ""), fixtures.get("away_team", ""))
            ]
            fixtures = fixtures.merge(cached_results, on="_result_key", how="left")
            has_cached_final = fixtures[["home_score_cached", "away_score_cached"]].notna().all(axis=1)
            for score_column in ["home_score", "away_score"]:
                cached_column = f"{score_column}_cached"
                if score_column not in fixtures.columns:
                    fixtures[score_column] = fixtures[cached_column]
                else:
                    fixtures[score_column] = fixtures[score_column].combine_first(fixtures[cached_column])
                fixtures = fixtures.drop(columns=[cached_column])
            if "status" not in fixtures.columns:
                fixtures["status"] = "SCHEDULED"
            fixtures.loc[has_cached_final, "status"] = "FINISHED"
            fixtures = fixtures.drop(columns=["_result_key"])

    if not fixtures.empty and "date" in fixtures.columns:
        fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    return apply_knockout_outcomes(fixtures)


def load_predictions() -> pd.DataFrame:
    predictions = read_csv_if_exists(REPORTS_DIR / "fixture_predictions.csv")
    if not predictions.empty and "date" in predictions.columns:
        predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    return predictions


def load_teams() -> pd.DataFrame:
    return read_csv_if_exists(RAW_DIR / "teams.csv")


def load_model_metadata() -> dict[str, Any]:
    return read_json_if_exists(MODELS_DIR / "model_metadata.json")


def load_update_status() -> dict[str, Any]:
    return read_json_if_exists(REPORTS_DIR / "update_status.json")


def load_feature_importance() -> pd.DataFrame:
    return read_csv_if_exists(REPORTS_DIR / "feature_importance.csv")


def load_analysis_index() -> pd.DataFrame:
    return read_csv_if_exists(REPORTS_DIR / "match_analysis" / "fixture_analysis_index.csv")


def load_lineup_features() -> pd.DataFrame:
    """Load the aggregate artifact, falling back to collected player lineups."""
    features = read_csv_if_exists(PROCESSED_DIR / "lineup_features.csv")
    if not features.empty:
        return features
    return read_csv_if_exists(RAW_DIR / "fotmob_lineups.csv")


FOTMOB_EVIDENCE_FACTORS = [
    ("fotmob_points_per_match_diff", "points per match", "points"),
    ("fotmob_goal_difference_per_match_diff", "goal difference per match", "goal_difference"),
    ("fotmob_avg_player_rating_diff", "average player rating", "avg_player_rating"),
    ("fotmob_starting_xi_avg_rating_diff", "starting XI rating", "starting_xi_avg_rating"),
    ("fotmob_shots_on_target_per_match_diff", "shots on target per match", "team_shots_on_target"),
    ("fotmob_expected_goals_per_match_diff", "expected goals per match", "team_expected_goals"),
    ("fotmob_chances_created_per_match_diff", "chances created per match", "team_chances_created"),
    ("fotmob_yellow_cards_per_match_diff", "yellow cards per match", "yellow_cards"),
    ("fotmob_red_cards_per_match_diff", "red cards per match", "red_cards"),
]


def _fixture_id_key(value: Any) -> str:
    """Normalize CSV-inferred numeric IDs without changing real string IDs."""
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") else text


def is_knockout_round(value: Any) -> bool:
    return str(value or "").strip().upper() in KNOCKOUT_ROUNDS


def _numeric_or_none(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else None


def _score_pair_from_text(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        home = _numeric_or_none(value[0])
        away = _numeric_or_none(value[1])
        if home is not None and away is not None:
            return home, away
    if not isinstance(value, str):
        return None, None
    match = re.search(r"(\d+(?:\.\d+)?)\s*[-:]\s*(\d+(?:\.\d+)?)", value)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _walk_nested(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_nested(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nested(child)


def _parse_status_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_penalty_scores(status: Any) -> tuple[float | None, float | None]:
    """Best-effort extraction across FotMob's nested shootout shapes."""
    payload = _parse_status_payload(status)
    for item in _walk_nested(payload):
        key_text = " ".join(str(key).lower() for key in item.keys())
        if "pen" not in key_text:
            continue

        home = next(
            (
                _numeric_or_none(item.get(key))
                for key in [
                    "homePenalties",
                    "homePenaltyScore",
                    "homePenScore",
                    "penaltyHome",
                    "home",
                ]
                if key in item
            ),
            None,
        )
        away = next(
            (
                _numeric_or_none(item.get(key))
                for key in [
                    "awayPenalties",
                    "awayPenaltyScore",
                    "awayPenScore",
                    "penaltyAway",
                    "away",
                ]
                if key in item
            ),
            None,
        )
        if home is not None and away is not None:
            return home, away

        score = next(
            (item.get(key) for key in ["penaltyScore", "penaltyScoreStr", "penalties"] if key in item),
            None,
        )
        home, away = _score_pair_from_text(score)
        if home is not None and away is not None:
            return home, away

    text = str(status)
    for pattern in [
        r"(?:penalties|pens|penalty shootout)[^\d]*(\d+)\s*[-:]\s*(\d+)",
        r"\((\d+)\s*[-:]\s*(\d+)\s*(?:pens?|penalties?)\)",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None


def _is_after_penalties(status: Any) -> bool:
    payload = _parse_status_payload(status)
    for item in _walk_nested(payload):
        reason_key = str(item.get("longKey") or item.get("shortKey") or "").lower()
        reason_short = str(item.get("short") or "").strip().lower()
        reason_long = str(item.get("long") or "").strip().lower()
        if (
            "afterpenalties" in reason_key
            or "penalties_short" in reason_key
            or reason_short == "pen"
            or reason_long.startswith("pen ")
            or "penalties" in item
            or "penaltyScore" in item
            or "penaltyScoreStr" in item
        ):
            return True
    return False


def _extract_penalty_loser(status: Any) -> str | None:
    payload = _parse_status_payload(status)
    for item in _walk_nested(payload):
        loser = item.get("whoLostOnPenalties")
        if isinstance(loser, str) and loser.strip():
            return loser.strip()
    return None


def _match_json_status(fotmob_match_id: Any) -> dict[str, Any]:
    match_id = _fixture_id_key(fotmob_match_id)
    if not match_id:
        return {}
    path = RAW_DIR / "fotmob" / "matches" / f"match_{match_id}.json"
    data = read_json_if_exists(path)
    for container_key in ["header", "general"]:
        container = data.get(container_key)
        if isinstance(container, dict) and isinstance(container.get("status"), dict):
            return container["status"]
    return {}


def _knockout_winner_code(row: pd.Series, use_scores: bool = False) -> str | None:
    home_score = _numeric_or_none(row.get("home_score"))
    away_score = _numeric_or_none(row.get("away_score"))
    home_penalties = _numeric_or_none(row.get("home_penalty_score"))
    away_penalties = _numeric_or_none(row.get("away_penalty_score"))
    if use_scores and home_score is not None and away_score is not None:
        if home_score > away_score:
            return "home"
        if away_score > home_score:
            return "away"
        if home_penalties is not None and away_penalties is not None:
            if home_penalties > away_penalties:
                return "home"
            if away_penalties > home_penalties:
                return "away"
        penalty_loser = row.get("penalty_loser")
        if isinstance(penalty_loser, str) and penalty_loser.strip():
            loser_key = _result_team_key(penalty_loser)
            home_key = _result_team_key(row.get("home_team"))
            away_key = _result_team_key(row.get("away_team"))
            if loser_key == home_key:
                return "away"
            if loser_key == away_key:
                return "home"

    home_probability = _numeric_or_none(row.get("display_home_win_probability", row.get("home_win_probability")))
    away_probability = _numeric_or_none(row.get("display_away_win_probability", row.get("away_win_probability")))
    if home_probability is None or away_probability is None:
        return None
    return "home" if home_probability >= away_probability else "away"


def apply_knockout_outcomes(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or "round" not in matches.columns:
        return matches
    matches = matches.copy()
    knockout = matches["round"].map(is_knockout_round)
    if not knockout.any():
        return matches

    for column in ["home_penalty_score", "away_penalty_score", "knockout_winner", "display_predicted_result", "display_predicted_winner"]:
        if column not in matches.columns:
            matches[column] = pd.NA

    for index, row in matches[knockout].iterrows():
        winner_code = _knockout_winner_code(row, use_scores=True)
        pick_code = _knockout_winner_code(row)
        home = str(row.get("home_team", "Team A"))
        away = str(row.get("away_team", "Team B"))
        if winner_code:
            matches.at[index, "knockout_winner"] = home if winner_code == "home" else away
            matches.at[index, "actual_winner"] = home if winner_code == "home" else away
        if pick_code:
            pick = home if pick_code == "home" else away
            matches.at[index, "display_predicted_winner"] = pick
            matches.at[index, "display_predicted_result"] = f"{pick} win"
    return matches


def _completed_match_fotmob_evidence(row: pd.Series) -> list[dict[str, Any]]:
    """Build display-only evidence from the completed match's FotMob data.

    This is intentionally separate from prediction features: first group matches
    have no tournament history before kickoff, but their post-match detail page
    should still show the FotMob statistics collected for that fixture.
    """
    if str(row.get("status", "")).strip().upper() not in {"FINISHED", "COMPLETED"}:
        return []

    features = read_csv_if_exists(PROCESSED_DIR / "fotmob_features.csv")
    if features.empty or "fixture_id" not in features.columns or "team_name" not in features.columns:
        return []

    fixture_id = _fixture_id_key(row.get("fixture_id"))
    fixture_rows = features[features["fixture_id"].map(_fixture_id_key) == fixture_id].copy()
    if fixture_rows.empty:
        return []

    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    team_keys = fixture_rows["team_name"].map(_result_team_key)
    home_rows = fixture_rows[team_keys == _result_team_key(home)]
    away_rows = fixture_rows[team_keys == _result_team_key(away)]
    if home_rows.empty or away_rows.empty:
        return []

    home_row = home_rows.iloc[-1].copy()
    away_row = away_rows.iloc[-1].copy()

    # Team-level FotMob stats are authoritative for totals. Player endpoints
    # sometimes emit placeholder zeroes, which can otherwise mask real values
    # when the processed feature table combines both feeds.
    raw_team_stats = read_csv_if_exists(RAW_DIR / "fotmob_team_match_stats.csv")
    if not raw_team_stats.empty and {"fixture_id", "team_name"}.issubset(raw_team_stats.columns):
        raw_fixture = raw_team_stats[
            raw_team_stats["fixture_id"].map(_fixture_id_key) == fixture_id
        ].copy()
        raw_team_keys = raw_fixture["team_name"].map(_result_team_key)
        raw_home = raw_fixture[raw_team_keys == _result_team_key(home)]
        raw_away = raw_fixture[raw_team_keys == _result_team_key(away)]
        authoritative_totals = {
            "goals_for": "team_goals",
            "goals_against": "goals_conceded",
            "shots_on_target": "team_shots_on_target",
            "expected_goals": "team_expected_goals",
            "yellow_cards": "yellow_cards",
            "red_cards": "red_cards",
        }
        for target, raw_rows in [(home_row, raw_home), (away_row, raw_away)]:
            if raw_rows.empty:
                continue
            raw_values = raw_rows.iloc[-1]
            for raw_column, target_column in authoritative_totals.items():
                value = pd.to_numeric(raw_values.get(raw_column), errors="coerce")
                if pd.notna(value):
                    target[target_column] = value

    for item in [home_row, away_row]:
        goals_for = pd.to_numeric(item.get("team_goals"), errors="coerce")
        goals_against = pd.to_numeric(item.get("goals_conceded"), errors="coerce")
        item["goal_difference"] = goals_for - goals_against
        item["points"] = 3.0 if goals_for > goals_against else (1.0 if goals_for == goals_against else 0.0)

    lower_is_better = {"yellow_cards", "red_cards"}
    output: list[dict[str, Any]] = []
    for feature, label, source in FOTMOB_EVIDENCE_FACTORS:
        home_value = pd.to_numeric(home_row.get(source), errors="coerce")
        away_value = pd.to_numeric(away_row.get(source), errors="coerce")
        if pd.isna(home_value) or pd.isna(away_value):
            continue
        difference = float(home_value - away_value)
        if difference == 0:
            leader = "Even"
        else:
            home_leads = difference > 0
            if source in lower_is_better:
                home_leads = not home_leads
            leader = home if home_leads else away
        output.append({"feature": feature, "label": label, "value": difference, "leader": leader})
    return output


def load_data_catalog() -> dict[str, pd.DataFrame]:
    return {
        "Fixtures": load_fixtures(),
        "Processed training dataset": read_csv_if_exists(PROCESSED_DIR / "training_dataset.csv"),
        "Fixture predictions": load_predictions(),
        "Tournament features": read_csv_if_exists(PROCESSED_DIR / "tournament_features.csv"),
        "Team features": read_csv_if_exists(PROCESSED_DIR / "team_features.csv"),
        "FotMob features": read_csv_if_exists(PROCESSED_DIR / "fotmob_features.csv"),
        "FotMob rolling features": read_csv_if_exists(PROCESSED_DIR / "fotmob_rolling_features.csv"),
        "Squad features": read_csv_if_exists(PROCESSED_DIR / "squad_features.csv"),
        "Lineup features": load_lineup_features(),
    }


def merge_fixtures_and_predictions(fixtures: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return predictions.copy()
    merged = fixtures.copy()

    if predictions.empty:
        return add_matchweeks(merged)

    prediction_columns = [column for column in predictions.columns if column not in {"date", "round", "group", "home_team", "away_team"}]
    if "fixture_id" in merged.columns and "fixture_id" in predictions.columns:
        merged["fixture_id"] = merged["fixture_id"].astype(str)
        predictions = predictions.copy()
        predictions["fixture_id"] = predictions["fixture_id"].astype(str)
        merged = merged.merge(predictions[prediction_columns], on="fixture_id", how="left", suffixes=("", "_prediction"))
    else:
        keys = [key for key in ["date", "home_team", "away_team"] if key in merged.columns and key in predictions.columns]
        merged = merged.merge(predictions, on=keys, how="left", suffixes=("", "_prediction")) if keys else merged

    for base, enriched in [
        ("home_win_probability", "fotmob_enriched_home_win_probability"),
        ("draw_probability", "fotmob_enriched_draw_probability"),
        ("away_win_probability", "fotmob_enriched_away_win_probability"),
    ]:
        if enriched in merged.columns:
            fallback = merged[base] if base in merged.columns else pd.Series(pd.NA, index=merged.index)
            merged[f"display_{base}"] = merged[enriched].combine_first(fallback)
        elif base in merged.columns:
            merged[f"display_{base}"] = merged[base]

    if "fotmob_enriched_predicted_result" in merged.columns:
        fallback = merged["predicted_result"] if "predicted_result" in merged.columns else pd.Series(pd.NA, index=merged.index)
        merged["display_predicted_result"] = merged["fotmob_enriched_predicted_result"].combine_first(fallback)
    elif "predicted_result" in merged.columns:
        merged["display_predicted_result"] = merged["predicted_result"]

    merged = apply_knockout_outcomes(merged)
    return add_matchweeks(merged)


def add_matchweeks(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return matches
    matches = matches.copy()
    if "matchweek" in matches.columns:
        matches["matchweek_label"] = matches["matchweek"].astype(str)
        matches["matchweek_inferred"] = False
        return matches

    matches["_date_sort"] = pd.to_datetime(matches.get("date"), errors="coerce")
    matches["matchweek_label"] = "Unassigned"
    matches["matchweek_inferred"] = True

    round_series = matches.get("round", pd.Series(index=matches.index, dtype="object")).astype(str)
    group_stage = round_series.str.upper().str.contains("GROUP", na=False)
    team_counts: dict[tuple[str, str], int] = {}

    for index, row in matches[group_stage].sort_values(["group", "_date_sort", "fixture_id"], na_position="last").iterrows():
        group = str(row.get("group", "Group Stage"))
        home_key = (group, str(row.get("home_team", "")))
        away_key = (group, str(row.get("away_team", "")))
        team_counts[home_key] = team_counts.get(home_key, 0) + 1
        team_counts[away_key] = team_counts.get(away_key, 0) + 1
        week = max(team_counts[home_key], team_counts[away_key])
        matches.at[index, "matchweek_label"] = f"Matchweek {week}"

    non_group = ~group_stage
    if non_group.any():
        matches.loc[non_group, "matchweek_label"] = round_series[non_group].replace({"nan": "Knockout / Other"})

    matches = matches.drop(columns=["_date_sort"])
    return matches


def get_match_analysis(row: pd.Series, analysis_index: pd.DataFrame) -> dict[str, Any]:
    if analysis_index.empty:
        return {}

    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))
    candidates = analysis_index[
        (analysis_index.get("home_team", "").astype(str) == home)
        & (analysis_index.get("away_team", "").astype(str) == away)
    ]
    if candidates.empty:
        return {}
    json_path = candidates.iloc[0].get("json_report")
    if not isinstance(json_path, str) or not json_path.strip():
        return {}
    analysis = read_json_if_exists(Path(json_path))
    fotmob_layer = analysis.get("fotmob_layer", {})
    # Finished fixtures should always describe what happened in that match.
    # Upcoming fixtures retain the saved pre-match rolling evidence instead.
    current_match_factors = _completed_match_fotmob_evidence(row)
    if current_match_factors:
        strongest = sorted(current_match_factors, key=lambda factor: abs(float(factor["value"])), reverse=True)[:5]
        analysis["fotmob_layer"] = {
            **fotmob_layer,
            "available": True,
            "feature_count": len(current_match_factors),
            "evidence_scope": "completed_match",
            "strongest": strongest,
            "all": current_match_factors,
        }
    return analysis


def completed_count(fixtures: pd.DataFrame) -> int:
    if fixtures.empty or "status" not in fixtures.columns:
        return 0
    return int(fixtures["status"].astype(str).str.upper().isin({"FINISHED", "COMPLETED"}).sum())
