from fotmob_utils import FOTMOB_RAW_DIR, fetch_team, save_json


def main() -> None:
    print("Testing direct FotMob team fetch with team id 8634...")
    data = fetch_team(8634)
    output_path = FOTMOB_RAW_DIR / "test_team.json"

    if data is None:
        print("No data returned. FotMob test failed safely.")
        save_json({"error": "no data returned"}, output_path)
        return

    save_json(data, output_path)
    if isinstance(data, dict):
        print(f"Returned top-level fields: {sorted(data.keys())}")
    else:
        print(f"Returned object type: {type(data).__name__}")
    print(f"Raw test response saved: {output_path}")


if __name__ == "__main__":
    main()
