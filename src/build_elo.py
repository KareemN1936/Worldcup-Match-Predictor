import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def get_k_factor(tournament: str) -> int:
    name = str(tournament).lower()
    if "fifa world cup" in name or name == "world cup":
        return 45
    if "continental" in name or "euro" in name or "copa america" in name or "african cup" in name or "asian cup" in name:
        return 35
    if "qualif" in name and "world cup" in name:
        return 30
    if "nations league" in name:
        return 25
    if "friendly" in name:
        return 15
    return 20


def build_elo_history(matches: pd.DataFrame, base_rating: int = 1500) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    rows = []

    for _, match in matches.sort_values("date").iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_rating = ratings.get(home, base_rating)
        away_rating = ratings.get(away, base_rating)
        exp_home = expected_score(home_rating, away_rating)

        if match["home_score"] > match["away_score"]:
            actual_home = 1.0
        elif match["home_score"] == match["away_score"]:
            actual_home = 0.5
        else:
            actual_home = 0.0

        rows.append({
            "match_id": match.get("match_id"),
            "date": match["date"],
            "home_team": home,
            "away_team": away,
            "home_elo_pre": home_rating,
            "away_elo_pre": away_rating,
            "elo_diff": home_rating - away_rating,
        })

        k = get_k_factor(match.get("tournament", "Other"))
        ratings[home] = home_rating + k * (actual_home - exp_home)
        ratings[away] = away_rating + k * ((1 - actual_home) - (1 - exp_home))

    return pd.DataFrame(rows)


def main() -> None:
    input_path = RAW_DATA_DIR / "historical_matches.csv"
    output_path = PROCESSED_DATA_DIR / "elo_history.csv"
    matches = pd.read_csv(input_path)
    if matches.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["match_id", "date", "home_team", "away_team", "home_elo_pre", "away_elo_pre", "elo_diff"]).to_csv(output_path, index=False)
        print("No historical matches found. Created empty Elo history.")
        return

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    elo = build_elo_history(matches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elo.to_csv(output_path, index=False)
    print(f"Elo history saved: {output_path} ({len(elo)} rows)")


if __name__ == "__main__":
    main()
