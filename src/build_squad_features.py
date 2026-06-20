import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


SQUAD_FEATURE_COLUMNS = [
    "team_id",
    "team_name",
    "squad_size",
    "squad_avg_age",
    "squad_median_age",
    "num_goalkeepers",
    "num_defenders",
    "num_midfielders",
    "num_forwards",
]


def _position_bucket(position: str | None) -> str:
    text = str(position or "").lower()
    if "goal" in text:
        return "goalkeeper"
    if "defence" in text or "defense" in text or "back" in text or "defender" in text:
        return "defender"
    if "midfield" in text:
        return "midfielder"
    if "forward" in text or "offence" in text or "offense" in text or "attack" in text or "winger" in text:
        return "forward"
    return "other"


def build_squad_features() -> pd.DataFrame:
    squads_path = RAW_DATA_DIR / "squads.csv"
    if not squads_path.exists():
        return pd.DataFrame(columns=SQUAD_FEATURE_COLUMNS)

    squads = pd.read_csv(squads_path)
    if squads.empty:
        return pd.DataFrame(columns=SQUAD_FEATURE_COLUMNS)

    squads["age"] = pd.to_numeric(squads["age"], errors="coerce")
    squads["position_bucket"] = squads["position"].apply(_position_bucket)

    base = squads.groupby(["team_id", "team_name"], dropna=False).agg(
        squad_size=("player_id", "count"),
        squad_avg_age=("age", "mean"),
        squad_median_age=("age", "median"),
    ).reset_index()

    position_counts = (
        squads.pivot_table(
            index=["team_id", "team_name"],
            columns="position_bucket",
            values="player_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={
            "goalkeeper": "num_goalkeepers",
            "defender": "num_defenders",
            "midfielder": "num_midfielders",
            "forward": "num_forwards",
        })
    )

    features = base.merge(position_counts, on=["team_id", "team_name"], how="left")
    for column in ["num_goalkeepers", "num_defenders", "num_midfielders", "num_forwards"]:
        if column not in features.columns:
            features[column] = 0

    features = features[SQUAD_FEATURE_COLUMNS]
    return features.sort_values("team_name").reset_index(drop=True)


def main() -> None:
    features = build_squad_features()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "squad_features.csv"
    features.to_csv(output_path, index=False)
    print(f"Squad features saved: {output_path} ({len(features)} rows)")


if __name__ == "__main__":
    main()
