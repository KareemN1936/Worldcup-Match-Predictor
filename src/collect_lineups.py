import pandas as pd

from config import RAW_DIR


def main() -> None:
    path = RAW_DIR / "lineups.csv"
    if not path.exists():
        pd.DataFrame(columns=["match_id", "team", "player", "position", "starter", "minutes"]).to_csv(path, index=False)
    print(f"Lineups file ready: {path}")


if __name__ == "__main__":
    main()
