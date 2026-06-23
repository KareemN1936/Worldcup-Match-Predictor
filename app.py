import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.data_loader import (
    REPORTS_DIR,
    ROOT_DIR,
    completed_count,
    get_match_analysis,
    load_analysis_index,
    load_data_catalog,
    load_feature_importance,
    load_fixtures,
    load_model_metadata,
    load_predictions,
    load_teams,
    load_update_status,
    merge_fixtures_and_predictions,
)
from src.web.flags import build_flag_lookup
from src.web.notes import build_model_notes, confidence_from_probabilities, most_likely_result
from src.web.ui_components import (
    percent,
    render_hero,
    render_feature_table,
    render_match_grid,
    render_match_detail_hero,
    render_notes,
    render_probability_bars,
    render_section_head,
    render_sidebar_brand,
    render_stat_grid,
    render_topline,
)


st.set_page_config(
    page_title="World Cup Match Predictor",
    page_icon=str(ROOT_DIR / "assets" / "world-cup-26.svg"),
    layout="wide",
    # Streamlit keeps this open on desktop and collapses it behind its native
    # menu trigger on narrow screens.
    initial_sidebar_state="auto",
)

NAV_ITEMS = ["Overview", "Matchweeks", "Match Detail", "Model Performance", "Data Explorer"]


def navigate_to(destination: str) -> None:
    st.session_state["navigation"] = destination


def inject_stylesheet(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


inject_stylesheet(ROOT_DIR / "tokens.css")


@st.cache_data(show_spinner=False)
def load_dashboard_data(data_version: tuple[tuple[str, int], ...]):
    # data_version is intentionally part of the cache key. The loaders read from
    # disk, so their source files must invalidate Streamlit's cached data.
    del data_version
    fixtures = load_fixtures()
    predictions = load_predictions()
    teams = load_teams()
    metadata = load_model_metadata()
    update_status = load_update_status()
    feature_importance = load_feature_importance()
    analysis_index = load_analysis_index()
    matches = merge_fixtures_and_predictions(fixtures, predictions)
    return fixtures, predictions, teams, metadata, update_status, feature_importance, analysis_index, matches


def dashboard_data_version() -> tuple[tuple[str, int], ...]:
    paths = [
        ROOT_DIR / "data" / "raw" / "fixtures.csv",
        ROOT_DIR / "data" / "raw" / "fotmob_match_details.csv",
        ROOT_DIR / "data" / "raw" / "fotmob_team_match_stats.csv",
        ROOT_DIR / "data" / "raw" / "fotmob_lineups.csv",
        ROOT_DIR / "data" / "processed" / "fotmob_features.csv",
        ROOT_DIR / "data" / "processed" / "fotmob_rolling_features.csv",
        ROOT_DIR / "data" / "processed" / "lineup_features.csv",
        ROOT_DIR / "data" / "raw" / "teams.csv",
        ROOT_DIR / "reports" / "fixture_predictions.csv",
        ROOT_DIR / "reports" / "update_status.json",
        ROOT_DIR / "reports" / "feature_importance.csv",
        ROOT_DIR / "reports" / "match_analysis" / "fixture_analysis_index.csv",
        ROOT_DIR / "models" / "model_metadata.json",
    ]
    paths.extend(sorted((ROOT_DIR / "data" / "raw" / "fotmob" / "matches").glob("date_*.json")))
    return tuple(
        (str(path.relative_to(ROOT_DIR)), path.stat().st_mtime_ns if path.exists() else -1)
        for path in paths
    )


fixtures, predictions, teams, metadata, update_status, feature_importance, analysis_index, matches = load_dashboard_data(
    dashboard_data_version()
)
flag_lookup = build_flag_lookup(teams)

requested_match = st.query_params.get("match")
if requested_match is not None:
    selected_match = next((idx for idx in matches.index if str(idx) == str(requested_match)), None)
    if selected_match is not None:
        st.session_state["_selected_match_idx"] = selected_match
        st.session_state["navigation"] = "Match Detail"
    st.query_params.clear()
    st.rerun()

render_sidebar_brand()
page = st.sidebar.radio(
    "Navigation",
    NAV_ITEMS,
    key="navigation",
    label_visibility="collapsed",
)

render_topline(right=page)

with st.sidebar.container(key="sidebar_footer"):
    st.divider()
    last_updated = str(
        update_status.get("last_finished_at")
        or update_status.get("last_attempt_at")
        or "Not available"
    )
    if len(last_updated) > 20:
        last_updated = last_updated[:19]
    st.caption(f"Last update: {last_updated}")

    if st.button("Refresh Data", type="primary", use_container_width=True):
        with st.spinner("Updating fixtures, predictions, and analysis…"):
            proc = subprocess.run(
                [sys.executable, "src/auto_update.py", "--once"],
                capture_output=True,
                text=True,
                cwd=ROOT_DIR,
            )
        if proc.returncode == 0:
            st.success("Data refreshed successfully!")
        else:
            st.error(f"Update failed (exit code {proc.returncode})")
            with st.expander("Error log"):
                st.code(proc.stdout[-3000:] if proc.stdout else proc.stderr[-3000:])
        st.cache_data.clear()
        st.rerun()


def best_model_name() -> str:
    return metadata.get("best_model", "Unavailable")


def format_number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def live_prediction_metrics(
    match_data: pd.DataFrame,
) -> tuple[int, int, float | None, int, int, float | None]:
    """Calculate overall and decisive-match winner accuracy."""
    required = {"status", "home_score", "away_score"}
    if match_data.empty or not required.issubset(match_data.columns):
        return 0, 0, None, 0, 0, None

    probability_columns = [
        "display_home_win_probability",
        "display_draw_probability",
        "display_away_win_probability",
    ]
    if not set(probability_columns).issubset(match_data.columns):
        return 0, 0, None, 0, 0, None

    played = match_data[
        match_data["status"].astype(str).str.upper().isin({"FINISHED", "COMPLETED"})
    ].copy()
    for column in ["home_score", "away_score", *probability_columns]:
        played[column] = pd.to_numeric(played[column], errors="coerce")
    played = played.dropna(subset=["home_score", "away_score", *probability_columns])
    probability_totals = played[probability_columns].sum(axis=1)
    played = played[probability_totals > 0].copy()
    if played.empty:
        return 0, 0, None, 0, 0, None

    probabilities = played[probability_columns].clip(lower=0)
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

    actual = pd.Series("draw", index=played.index)
    actual.loc[played["home_score"] > played["away_score"]] = "home"
    actual.loc[played["away_score"] > played["home_score"]] = "away"

    predicted = probabilities.idxmax(axis=1).map(
        {
            "display_home_win_probability": "home",
            "display_draw_probability": "draw",
            "display_away_win_probability": "away",
        }
    )
    correct = int((actual == predicted).sum())
    total = len(played)
    decisive = actual != "draw"
    winner_total = int(decisive.sum())
    winner_correct = int((actual[decisive] == predicted[decisive]).sum())
    winner_accuracy = winner_correct / winner_total if winner_total else None
    return correct, total, correct / total, winner_correct, winner_total, winner_accuracy


def upcoming_matches_frame() -> pd.DataFrame:
    if matches.empty or "date" not in matches.columns:
        return pd.DataFrame()
    dated = matches.copy()
    dated["_date"] = pd.to_datetime(dated["date"], errors="coerce")
    scheduled = dated[~dated.get("status", pd.Series(index=dated.index, dtype=str)).astype(str).str.upper().isin({"FINISHED", "COMPLETED"})]
    return scheduled.sort_values("_date").head(3)


best_name = best_model_name()
(
    correct_predictions,
    played_predictions,
    live_accuracy,
    correct_winners,
    games_with_winner,
    winner_accuracy,
) = live_prediction_metrics(matches)
completed_metric_rows = matches[
    matches.get("status", pd.Series(index=matches.index, dtype=str))
    .astype(str).str.upper().isin({"FINISHED", "COMPLETED"})
]
uses_retrospective_backtest = (
    "prediction_source" in completed_metric_rows.columns
    and completed_metric_rows["prediction_source"].eq("retrospective_backtest").any()
)
metric_scope = "Retrospective backtest · " if uses_retrospective_backtest else ""
accuracy_hint = (
    f"{metric_scope}{correct_predictions} out of {played_predictions} games guessed correctly"
    if played_predictions
    else "Awaiting results from frozen pre-match predictions"
)
winner_accuracy_hint = (
    f"{metric_scope}{correct_winners} out of {games_with_winner} games with a winner guessed correctly"
    if games_with_winner
    else "Awaiting results from frozen pre-match predictions"
)


if page == "Overview":
    total_matches = len(fixtures)
    completed = completed_count(fixtures)
    upcoming = max(total_matches - completed, 0)

    render_hero(total_matches, completed, upcoming, best_name)
    with st.container(key="hero_actions"):
        action_primary, action_secondary = st.columns(2, gap="small")
        action_primary.button(
            "View matchweeks",
            type="primary",
            use_container_width=True,
            on_click=navigate_to,
            args=("Matchweeks",),
        )
        action_secondary.button(
            "Model performance",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Performance",),
        )

    last_updated = update_status.get("last_finished_at") or update_status.get("last_attempt_at") or "Not available"
    if isinstance(last_updated, str) and len(last_updated) > 20:
        last_updated = last_updated[:19]
    render_section_head(
        "Tournament Control Room",
        "A cleaner read on fixtures, probabilities, confidence, and the signals behind every pick.",
    )
    render_stat_grid(
        [
            {"label": "Total matches", "value": total_matches, "hint": "Loaded from fixtures", "color": "var(--color-accent)"},
            {"label": "Completed", "value": completed, "hint": "Finished fixtures", "color": "var(--color-accent-2)"},
            {"label": "Upcoming", "value": upcoming, "hint": "Still to play", "color": "var(--color-blue)"},
            {"label": "Predicted", "value": len(predictions), "hint": "Model-scored fixtures", "color": "var(--color-accent-3)"},
        ]
    )

    render_section_head("Model Pulse", "Live performance across completed tournament predictions.")
    render_stat_grid(
        [
            {"label": "Best model", "value": best_name, "hint": "Selected by validation metric", "color": "var(--color-blue)"},
            {"label": "Accuracy", "value": percent(live_accuracy), "hint": accuracy_hint, "color": "var(--color-accent)"},
            {"label": "Winner accuracy", "value": percent(winner_accuracy), "hint": winner_accuracy_hint, "color": "var(--color-accent-3)"},
            {"label": "Last update", "value": "Ready", "hint": str(last_updated), "color": "var(--color-accent-2)"},
        ]
    )

    if matches.get("matchweek_inferred", pd.Series(dtype=bool)).any():
        st.caption("Matchweek labels are inferred from group-stage date order and each team's group appearance number.")

    featured = upcoming_matches_frame()
    if not featured.empty:
        render_section_head("Next Fixtures", "The next model-scored matches on the board.")
        render_match_grid(featured, flag_lookup)


elif page == "Matchweeks":
    render_section_head("Matchweeks", "Filter by week, group, team, and status.")

    if matches.empty:
        st.warning("No fixtures or predictions are available yet.")
    else:
        week_options = ["All matchweeks"] + sorted(
            str(v) for v in matches["matchweek_label"].dropna().unique()
        )
        selected_week = st.selectbox("Matchweek", week_options, key="mw_week")

        if selected_week == "All matchweeks":
            filtered = matches.copy()
        else:
            filtered = matches[matches["matchweek_label"].astype(str) == selected_week].copy()
        col1, col2, col3 = st.columns(3)
        group_values = ["All"] + sorted([str(value) for value in filtered.get("group", pd.Series(dtype=str)).dropna().unique()])
        team_values = sorted(
            set(filtered.get("home_team", pd.Series(dtype=str)).dropna())
            | set(filtered.get("away_team", pd.Series(dtype=str)).dropna())
        )
        status_values = ["All"] + sorted([str(value) for value in filtered.get("status", pd.Series(dtype=str)).dropna().unique()])

        selected_group = col1.selectbox("Group", group_values)
        selected_team = col2.selectbox("Team", ["All"] + team_values)
        selected_status = col3.selectbox("Status", status_values)

        if selected_group != "All" and "group" in filtered.columns:
            filtered = filtered[filtered["group"].astype(str) == selected_group]
        if selected_team != "All":
            filtered = filtered[(filtered["home_team"] == selected_team) | (filtered["away_team"] == selected_team)]
        if selected_status != "All" and "status" in filtered.columns:
            filtered = filtered[filtered["status"].astype(str) == selected_status]

        render_stat_grid(
            [
                {"label": "Shown", "value": len(filtered), "hint": selected_week, "color": "var(--color-accent)"},
                {
                    "label": "Groups",
                    "value": filtered.get("group", pd.Series(dtype=str)).nunique(),
                    "hint": "In current view",
                    "color": "var(--color-blue)",
                },
                {"label": "Teams", "value": len(set(filtered.get("home_team", [])) | set(filtered.get("away_team", []))), "hint": "In current view", "color": "var(--color-accent-3)"},
                {
                    "label": "Finished",
                    "value": int(filtered.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin({"FINISHED", "COMPLETED"}).sum()),
                    "hint": "Completed fixtures",
                    "color": "var(--color-accent-2)",
                },
            ]
        )

        if matches["matchweek_inferred"].any():
            st.caption("Matchweek labels are inferred because the fixture file does not include a matchweek column.")

        render_section_head("Fixtures", f"{len(filtered)} matches in the current view. Select any fixture to inspect it.", selected_week)
        render_match_grid(filtered.sort_values("date"), flag_lookup, detail_link=True)


elif page == "Match Detail":
    render_section_head("Match Detail", "Inspect prediction, confidence, notes, and feature evidence.")

    if matches.empty:
        st.warning("No matches are available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            week_options = ["All matchweeks"] + sorted(
                str(v) for v in matches["matchweek_label"].dropna().unique()
            )
            selected_week_filter = st.selectbox("Matchweek", week_options, key="detail_week")
        with col2:
            if selected_week_filter == "All matchweeks":
                detail_matches = matches.sort_values("date")
            else:
                detail_matches = matches[matches["matchweek_label"].astype(str) == selected_week_filter].sort_values("date")

            labels = []
            for index, row in detail_matches.iterrows():
                date = pd.to_datetime(row.get("date"), errors="coerce")
                date_label = date.strftime("%b %d · %H:%M") if pd.notna(date) else "Date unavailable"
                labels.append((f"{date_label} | {row.get('home_team')} vs {row.get('away_team')}", index))

            selected_idx = st.session_state.pop("_selected_match_idx", None)
            if selected_idx is not None:
                for label, idx in labels:
                    if idx == selected_idx:
                        st.session_state["detail_match"] = label
                        break

            selected_label = st.selectbox("Match", [label for label, _ in labels], key="detail_match")
            selected_index = dict(labels)[selected_label]
            row = matches.loc[selected_index]
            analysis = get_match_analysis(row, analysis_index)

        render_match_detail_hero(row, flag_lookup)

        home_prob = row.get("display_home_win_probability", row.get("home_win_probability"))
        draw_prob = row.get("display_draw_probability", row.get("draw_probability"))
        away_prob = row.get("display_away_win_probability", row.get("away_win_probability"))
        render_section_head("Prediction Snapshot", "The model's probability distribution for this fixture.", "Probabilities")
        render_stat_grid(
            [
                {"label": "Most likely", "value": most_likely_result(row), "hint": "Top outcome", "color": "var(--color-accent)"},
                {
                    "label": "Confidence",
                    "value": confidence_from_probabilities(home_prob, draw_prob, away_prob),
                    "hint": "Spread between top outcomes",
                    "color": "var(--color-blue)",
                },
                {"label": "Home win", "value": percent(home_prob), "hint": str(row.get("home_team", "Team A")), "color": "var(--color-accent)"},
                {"label": "Away win", "value": percent(away_prob), "hint": str(row.get("away_team", "Team B")), "color": "var(--color-accent-2)"},
            ]
        )

        render_section_head("Win Probability", "Home, draw, and away outcome chances.", "Model output")
        render_probability_bars(row)

        render_section_head("Model Notes", "Plain-English signals from the saved model and match analysis files.", "Why this pick")
        render_notes(build_model_notes(row, analysis=analysis))

        render_section_head("Feature Evidence", "Detailed features that were available for this fixture.", "Data")
        tab1, tab2, tab3 = st.tabs(["Historical / Elo", "FotMob Stats", "Squad Strength"])
        with tab1:
            render_feature_table("Historical Form / Elo", analysis.get("historical_model_factors", {}).get("all", []))
        with tab2:
            if analysis.get("fotmob_layer", {}).get("evidence_scope") == "completed_match":
                st.caption("Completed-match FotMob evidence. These statistics describe the match and were not used as pre-match inputs.")
            render_feature_table("FotMob Stats", analysis.get("fotmob_layer", {}).get("all", []))
        with tab3:
            render_feature_table("Squad Strength", analysis.get("squad_context", {}).get("all", []))
        st.info("Lineup, player stats, injury, and fatigue sections appear only when those feature files contain match-level data.")


elif page == "Model Performance":
    render_section_head("Model Performance", "Live tournament performance and saved evaluation artifacts.")
    render_stat_grid(
        [
            {"label": "Best model", "value": best_name, "hint": "Current selected model", "color": "var(--color-blue)"},
            {"label": "Accuracy", "value": percent(live_accuracy), "hint": accuracy_hint, "color": "var(--color-accent)"},
            {"label": "Winner accuracy", "value": percent(winner_accuracy), "hint": winner_accuracy_hint, "color": "var(--color-accent-3)"},
            {"label": "Features", "value": len(metadata.get("features", [])), "hint": "Training inputs", "color": "var(--color-accent-2)"},
        ]
    )
    st.caption("Predictions are probabilistic and not guaranteed outcomes.")

    confusion = metadata.get("models", {}).get(best_name, {}).get("test", {}).get("confusion_matrix")
    if confusion:
        render_section_head("Confusion Matrix", "Class-level prediction outcomes on the test split.", "Evaluation")
        st.dataframe(
            pd.DataFrame(confusion, index=["True loss", "True draw", "True win"], columns=["Pred loss", "Pred draw", "Pred win"]),
            width="stretch",
        )

    split = metadata.get("split", {})
    if split:
        render_section_head("Training Split", "How the data was divided for evaluation.", "Dataset")
        st.json(split)

    render_section_head("Feature Importance", "The strongest model inputs by saved importance score.", "Signals")
    if feature_importance.empty:
        st.info("Feature importance file is not available.")
    else:
        st.dataframe(feature_importance, width="stretch", hide_index=True)
        st.bar_chart(feature_importance.head(15), x="feature", y="importance", width="stretch")

    report_path = REPORTS_DIR / "evaluation_report.txt"
    if report_path.exists():
        with st.expander("Full evaluation report"):
            st.text(report_path.read_text(encoding="utf-8", errors="replace"))


elif page == "Data Explorer":
    render_section_head("Data Explorer", "Browse the project datasets behind the predictions.")
    catalog = load_data_catalog()
    selected = st.selectbox("Dataset", list(catalog.keys()))
    data = catalog[selected]
    if data.empty:
        st.info(f"{selected} not available yet.")
    else:
        render_stat_grid(
            [
                {"label": "Rows", "value": len(data), "hint": selected, "color": "var(--color-accent)"},
                {"label": "Columns", "value": len(data.columns), "hint": "Available fields", "color": "var(--color-blue)"},
                {"label": "Source", "value": "CSV", "hint": "Project data", "color": "var(--color-accent-3)"},
                {"label": "Status", "value": "Loaded", "hint": "Ready to inspect", "color": "var(--color-accent-2)"},
            ]
        )
        st.dataframe(data, width="stretch")
