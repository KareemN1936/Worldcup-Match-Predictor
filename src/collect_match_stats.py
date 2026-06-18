import pandas as pd

from config import RAW_DIR


def main() -> None:
    path = RAW_DIR / "team_match_stats.csv"
    if not path.exists():
        pd.DataFrame(columns=["match_id", "team", "shots", "shots_on_target", "possession", "xg"]).to_csv(path, index=False)
    print(f"Team match stats file ready: {path}")


if __name__ == "__main__":
    main()
