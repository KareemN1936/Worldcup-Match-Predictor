import pandas as pd

from config import RAW_DIR


def main() -> None:
    path = RAW_DIR / "player_match_stats.csv"
    if not path.exists():
        pd.DataFrame(columns=["match_id", "team", "player", "minutes", "goals", "assists", "xg", "xa"]).to_csv(path, index=False)
    print(f"Player match stats file ready: {path}")


if __name__ == "__main__":
    main()
