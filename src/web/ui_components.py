from html import escape
from textwrap import dedent

import pandas as pd
import streamlit as st

from .flags import flag_for_team
from .notes import build_model_notes, confidence_from_probabilities, most_likely_result


def percent(value) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value) * 100:.1f}%"


def _score_text(value) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def status_label(row: pd.Series) -> str:
    status = str(row.get("status", "scheduled")).strip() or "scheduled"
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        return f"{status.title()} | {_score_text(row.get('home_score'))} - {_score_text(row.get('away_score'))}"
    return status.replace("_", " ").title()


def format_match_date(value) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return "Date unavailable"
    return date.strftime("%b %d, %Y | %H:%M")


def _html(markup: str) -> str:
    """Flatten generated markup so Markdown never treats indentation as code."""
    return "".join(line.strip() for line in dedent(markup).splitlines())


def flag_html(flag: str, team: str, size_class: str = "") -> str:
    safe_team = escape(str(team))
    safe_flag = escape(str(flag or ""))
    class_name = f"team-flag {size_class}".strip()
    if safe_flag.startswith("http://") or safe_flag.startswith("https://"):
        return f'<img class="{class_name}" src="{safe_flag}" alt="{safe_team} flag" loading="lazy">'
    return f'<span class="team-flag-text {escape(size_class)}" aria-label="{safe_team} flag">{safe_flag}</span>'


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        _html("""
        <div class="sidebar-brand">
            <div class="sidebar-brand__mark">26</div>
            <div>
                <div class="sidebar-brand__name">World Cup Intel</div>
                <div class="sidebar-brand__copy">
                    Fixtures | predictions | model reads
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_topline(label: str = "FIFA World Cup 2026", right: str = "Predictive match center") -> None:
    st.markdown(
        _html(f"""
        <div class="wc-topline">
            <span class="wc-pill">{escape(label)}</span>
            <span>{escape(right)}</span>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_hero(total_matches: int, completed: int, upcoming: int, best_model: str) -> None:
    st.markdown(
        _html(f"""
        <section class="wc-hero">
            <div>
                <span class="wc-pill">Live tournament model</span>
                <div class="wc-hero__title">World Cup Match Predictor</div>
                <p class="wc-hero__copy">
                    A sharp match center for the 2026 World Cup: fixtures, win probabilities,
                    model confidence, squad context, and post-match reads without the spreadsheet fog.
                </p>
            </div>
            <aside class="wc-orbit-card" aria-label="Tournament snapshot">
                <div class="wc-orbit-card__row">
                    <span>Completed</span>
                    <span class="wc-score">{completed}</span>
                    <span>{total_matches} total</span>
                </div>
                <div class="wc-orbit-card__row">
                    <span>Upcoming</span>
                    <span class="wc-score">{upcoming}</span>
                    <span>fixtures</span>
                </div>
                <div class="wc-orbit-card__row">
                    <span>Best model</span>
                    <span class="wc-score">AI</span>
                    <span>{escape(str(best_model))}</span>
                </div>
            </aside>
        </section>
        """),
        unsafe_allow_html=True,
    )


def render_section_head(title: str, body: str = "", eyebrow: str | None = None) -> None:
    eyebrow_html = f'<span class="wc-pill">{escape(eyebrow)}</span>' if eyebrow else ""
    body_html = f"<p>{escape(body)}</p>" if body else ""
    st.markdown(
        _html(f"""
        <div class="wc-section-head">
            <div>
                {eyebrow_html}
                <h2>{escape(title)}</h2>
                {body_html}
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_stat_grid(cards: list[dict[str, str]]) -> None:
    card_html = []
    for card in cards:
        color = escape(card.get("color", "var(--color-accent)"))
        card_html.append(
            _html(f"""
            <article class="wc-stat-card" style="--stat-color:{color};">
                <div class="wc-stat-card__label">{escape(str(card.get("label", "")))}</div>
                <div class="wc-stat-card__value">{escape(str(card.get("value", "")))}</div>
                <div class="wc-stat-card__hint">{escape(str(card.get("hint", "")))}</div>
            </article>
            """)
        )
    st.markdown(f'<div class="wc-grid wc-grid--stats">{"".join(card_html)}</div>', unsafe_allow_html=True)


def render_panel_open() -> None:
    st.markdown('<div class="wc-panel">', unsafe_allow_html=True)


def render_panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _pct_value(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return max(0.0, min(float(value) * 100, 100.0))


def _is_completed(row: pd.Series) -> bool:
    return str(row.get("status", "")).strip().upper() in {"FINISHED", "COMPLETED"}


def _outcome_details(row: pd.Series) -> tuple[str, str, bool | None] | None:
    """Return actual label, AI pick label, and correctness for a scored match."""
    if not _is_completed(row) or pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
        return None

    home = str(row.get("home_team", "Team A"))
    away = str(row.get("away_team", "Team B"))
    home_score = float(row.get("home_score"))
    away_score = float(row.get("away_score"))

    if home_score > away_score:
        actual_code, actual_label = "home", f"{home} won"
    elif away_score > home_score:
        actual_code, actual_label = "away", f"{away} won"
    else:
        actual_code, actual_label = "draw", "Draw"

    probabilities = {
        "home": row.get("display_home_win_probability", row.get("home_win_probability")),
        "draw": row.get("display_draw_probability", row.get("draw_probability")),
        "away": row.get("display_away_win_probability", row.get("away_win_probability")),
    }
    if not all(pd.notna(value) for value in probabilities.values()):
        return actual_label, "Unavailable", None

    predicted_code = max(probabilities, key=lambda outcome: float(probabilities[outcome]))
    predicted_label = {"home": home, "draw": "Draw", "away": away}[predicted_code]
    return actual_label, predicted_label, predicted_code == actual_code


def _result_summary_html(row: pd.Series) -> str:
    if not _is_completed(row):
        return ""

    outcome = _outcome_details(row)
    if outcome is None:
        return _html("""
        <div class="match-result match-result--unavailable">
            <span class="match-result__verdict">Result unavailable</span>
            <span>The final score has not been supplied by the match data source.</span>
        </div>
        """)

    actual_label, predicted_label, was_correct = outcome
    if was_correct is None:
        return _html(f"""
        <div class="match-result match-result--unavailable">
            <span class="match-result__verdict">AI prediction unavailable</span>
            <span><strong>Actual:</strong> {escape(actual_label)} | No saved pre-match pick was found.</span>
        </div>
        """)
    state = "correct" if was_correct else "wrong"
    verdict = "AI predicted correctly" if was_correct else "AI prediction was wrong"
    symbol = "✓" if was_correct else "✕"
    return _html(f"""
    <div class="match-result match-result--{state}">
        <span class="match-result__verdict">{symbol} {verdict}</span>
        <span><strong>Actual:</strong> {escape(actual_label)} | <strong>AI picked:</strong> {escape(predicted_label)}</span>
    </div>
    """)


def _prob_chip(label: str, value, color_var: str) -> str:
    pct = _pct_value(value)
    if pct is None:
        return ""
    return _html(f"""
    <div class="prob-chip" style="--chip-color:{color_var};">
        <div class="prob-chip__row">
            <span class="prob-chip__label">{escape(label)}</span>
            <strong class="prob-chip__value">{pct:.0f}%</strong>
        </div>
        <div class="prob-chip__track"><div class="prob-chip__fill" style="--pct:{pct:.1f}%;"></div></div>
    </div>
    """)


def render_probability_bars(row: pd.Series) -> None:
    home = str(row.get("home_team", "Team A"))
    away = str(row.get("away_team", "Team B"))
    home_prob = row.get("display_home_win_probability", row.get("home_win_probability"))
    draw_prob = row.get("display_draw_probability", row.get("draw_probability"))
    away_prob = row.get("display_away_win_probability", row.get("away_win_probability"))

    if not all(pd.notna(v) for v in [home_prob, draw_prob, away_prob]):
        st.info("Prediction unavailable for this match.")
        return

    bars = []
    for label, value, color in [
        (home, home_prob, "var(--color-accent)"),
        ("Draw", draw_prob, "var(--color-accent-3)"),
        (away, away_prob, "var(--color-accent-2)"),
    ]:
        pct = _pct_value(value) or 0
        bars.append(
            _html(f"""
            <div class="prob-bar" style="--chip-color:{color};">
                <div class="prob-bar__label">
                    <span>{escape(label)}</span>
                    <strong>{pct:.1f}%</strong>
                </div>
                <div class="prob-bar__track">
                    <div class="prob-bar__fill" style="--pct:{pct:.1f}%;"></div>
                </div>
            </div>
            """)
        )

    st.markdown(f'<div class="prob-bars">{"".join(bars)}</div>', unsafe_allow_html=True)


def _match_card_html(
    row: pd.Series,
    flag_lookup: dict[str, str],
    analysis: dict | None = None,
    detail_idx=None,
) -> str:
    home = str(row.get("home_team", "Team A"))
    away = str(row.get("away_team", "Team B"))
    home_prob = row.get("display_home_win_probability", row.get("home_win_probability"))
    draw_prob = row.get("display_draw_probability", row.get("draw_probability"))
    away_prob = row.get("display_away_win_probability", row.get("away_win_probability"))
    confidence = confidence_from_probabilities(home_prob, draw_prob, away_prob)
    notes = build_model_notes(row, analysis=analysis, limit=2)

    home_flag = flag_for_team(home, flag_lookup)
    away_flag = flag_for_team(away, flag_lookup)
    date_str = format_match_date(row.get("date"))
    group = str(row.get("group", "Group unavailable")).replace("_", " ").title()
    round_name = str(row.get("round", "Round unavailable")).replace("_", " ").title()

    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        score_html = f'<div class="match-center__score">{_score_text(row.get("home_score"))} - {_score_text(row.get("away_score"))}</div>'
    else:
        score_html = '<div class="match-center__vs">VS</div>'

    most_likely = most_likely_result(row)
    result_html = _result_summary_html(row)
    has_probs = all(pd.notna(v) for v in [home_prob, draw_prob, away_prob])
    prob_html = ""
    if has_probs:
        prob_html = _html(f"""
        <div class="prob-strip">
            {_prob_chip(home, home_prob, "var(--color-accent)")}
            {_prob_chip("Draw", draw_prob, "var(--color-accent-3)")}
            {_prob_chip(away, away_prob, "var(--color-accent-2)")}
        </div>
        """)

    note_html = f'<div class="match-note">{escape(notes[0])}</div>' if notes else ""
    venue_bits = [str(row.get(key)) for key in ["stadium", "city"] if pd.notna(row.get(key)) and str(row.get(key)).strip()]
    venue = " | ".join(venue_bits) if venue_bits else "Venue TBD"
    details_html = ""
    if detail_idx is not None:
        details_html = (
            f'<a class="match-card-link" href="?match={escape(str(detail_idx))}" target="_self" '
            f'aria-label="View {escape(home)} versus {escape(away)} match details">'
            '<span class="match-card__details">Details →</span></a>'
        )

    return _html(f"""
    <article class="match-card">
        <div class="match-card__meta">
            <span>{escape(group)} | {escape(round_name)}</span>
            <span>{escape(status_label(row))}</span>
        </div>
        <div class="match-card__teams">
            <div class="match-team match-team--home">
                <span class="match-team__name">{escape(home)}</span>
                {flag_html(home_flag, home)}
            </div>
            <div class="match-center">
                {score_html}
                <div>{escape(date_str)}</div>
            </div>
            <div class="match-team">
                {flag_html(away_flag, away)}
                <span class="match-team__name">{escape(away)}</span>
            </div>
        </div>
        {result_html}
        {prob_html}
        <div class="match-card__footer">
            <span>Most likely: <strong>{escape(most_likely)}</strong></span>
            <span>Confidence: <strong>{escape(confidence)}</strong></span>
            <span>{escape(venue)}</span>
        </div>
        {note_html}
        {details_html}
    </article>
    """)


def render_match_card(
    row: pd.Series,
    flag_lookup: dict[str, str],
    analysis: dict | None = None,
    detail_idx=None,
) -> None:
    st.markdown(
        _match_card_html(row, flag_lookup, analysis=analysis, detail_idx=detail_idx),
        unsafe_allow_html=True,
    )


def render_match_grid(rows: pd.DataFrame, flag_lookup: dict[str, str], detail_link: bool = True) -> None:
    """Render fixtures in one responsive grid without Streamlit column stacking."""
    cards = [
        _match_card_html(
            row,
            flag_lookup,
            detail_idx=row.name if detail_link else None,
        )
        for _, row in rows.iterrows()
    ]
    st.markdown(f'<div class="match-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_match_detail_hero(row: pd.Series, flag_lookup: dict[str, str]) -> None:
    home = str(row.get("home_team", "Team A"))
    away = str(row.get("away_team", "Team B"))
    home_flag = flag_for_team(home, flag_lookup)
    away_flag = flag_for_team(away, flag_lookup)
    score = "VS"
    score_label = status_label(row)
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        score = f"{_score_text(row.get('home_score'))} - {_score_text(row.get('away_score'))}"
        score_label = "Full time" if "FINISHED" in str(row.get("status", "")).upper() else score_label

    venue_bits = [str(row.get(key)) for key in ["stadium", "city"] if pd.notna(row.get(key)) and str(row.get(key)).strip()]
    venue = " | ".join(venue_bits) if venue_bits else "Venue TBD"
    meta = [
        format_match_date(row.get("date")),
        str(row.get("round", "Round unavailable")).replace("_", " ").title(),
        str(row.get("group", "Group unavailable")).replace("_", " ").title(),
        venue,
    ]
    result_html = _result_summary_html(row)

    st.markdown(
        _html(f"""
        <section class="detail-hero">
            <div class="detail-hero__meta">
                {"".join(f'<span class="wc-pill">{escape(item)}</span>' for item in meta)}
            </div>
            <div class="detail-scoreboard">
                <div class="detail-team">
                    {flag_html(home_flag, home)}
                    <div class="detail-team__name">{escape(home)}</div>
                </div>
                <div class="detail-score">
                    <div class="detail-score__number">{escape(score)}</div>
                    <div class="detail-score__label">{escape(score_label)}</div>
                </div>
                <div class="detail-team">
                    {flag_html(away_flag, away)}
                    <div class="detail-team__name">{escape(away)}</div>
                </div>
            </div>
            {result_html}
        </section>
        """),
        unsafe_allow_html=True,
    )


def render_notes(notes: list[str]) -> None:
    if not notes:
        return
    items = "".join(f'<div class="note-item"><strong>Read:</strong> {escape(note)}</div>' for note in notes)
    st.markdown(f'<div class="note-list">{items}</div>', unsafe_allow_html=True)


def render_feature_table(title: str, rows: list[dict]) -> None:
    st.subheader(title)
    if not rows:
        st.info(f"{title} not available yet.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
