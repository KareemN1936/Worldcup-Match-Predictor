import pandas as pd

from config import RAW_DIR


def main() -> None:
    path = RAW_DIR / "squads.csv"
    if not path.exists():
        pd.DataFrame(columns=["team", "player", "position", "age", "club", "caps", "goals"]).to_csv(path, index=False)
    print(f"Squads file ready: {path}")


if __name__ == "__main__":
    main()
