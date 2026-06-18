import pandas as pd

from config import RAW_DIR


def main() -> None:
    path = RAW_DIR / "injuries.csv"
    if not path.exists():
        pd.DataFrame(columns=["team", "player", "status", "injury", "reported_date", "expected_return"]).to_csv(path, index=False)
    print(f"Injuries file ready: {path}")


if __name__ == "__main__":
    main()
