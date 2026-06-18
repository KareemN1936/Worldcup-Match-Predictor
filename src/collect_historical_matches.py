import pandas as pd

from config import RAW_DATA_DIR


REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def _clean_column_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def load_and_clean_historical_matches(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_DIR / "historical_matches.csv"
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    matches = pd.read_csv(path)
    matches = matches.rename(columns={column: _clean_column_name(column) for column in matches.columns})

    for column in REQUIRED_COLUMNS:
        if column not in matches.columns:
            matches[column] = pd.NA

    if matches.empty:
        return matches[REQUIRED_COLUMNS]

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")
    matches["neutral"] = matches["neutral"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])

    if matches["match_id"].isna().any():
        matches["match_id"] = range(1, len(matches) + 1)

    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    matches["match_id"] = matches["match_id"].astype(str)

    return matches[REQUIRED_COLUMNS]


def main() -> None:
    path = RAW_DATA_DIR / "historical_matches.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    matches = load_and_clean_historical_matches(path)
    matches.to_csv(path, index=False)
    print(f"Historical matches cleaned: {path} ({len(matches)} rows)")


if __name__ == "__main__":
    main()
