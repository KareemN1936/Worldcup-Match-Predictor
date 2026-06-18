import pandas as pd

from config import RAW_DIR


def main() -> None:
    path = RAW_DIR / "fixtures.csv"
    if not path.exists():
        pd.DataFrame(columns=["fixture_id", "date", "home_team", "away_team", "venue", "stage"]).to_csv(path, index=False)
    print(f"Fixtures file ready: {path}")


if __name__ == "__main__":
    main()
