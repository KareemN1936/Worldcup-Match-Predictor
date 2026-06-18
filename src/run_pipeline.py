import argparse
import subprocess
import sys
from pathlib import Path


PHASE_1_STEPS = [
    "collect_historical_matches.py",
    "build_elo.py",
    "build_features.py",
    "train_model.py",
    "evaluate_model.py",
]

FIXTURE_STEPS = [
    "collect_fixtures.py",
    "collect_teams.py",
]


def _run_steps(steps: list[str], label: str) -> None:
    src_dir = Path(__file__).resolve().parent
    print(label, flush=True)
    for number, step in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] Running {step}...", flush=True)
        subprocess.run([sys.executable, str(src_dir / step)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the World Cup predictor pipeline.")
    parser.add_argument("--with-fixtures", action="store_true", help="Also collect World Cup fixtures and teams from API-Football.")
    parser.add_argument("--fixtures-only", action="store_true", help="Only collect World Cup fixtures and teams.")
    args = parser.parse_args()

    if args.fixtures_only:
        _run_steps(FIXTURE_STEPS, "Running Phase 2 fixture collection...")
        print("Fixture collection complete.")
        return

    _run_steps(PHASE_1_STEPS, "Running Phase 1 historical model pipeline...")

    if args.with_fixtures:
        _run_steps(FIXTURE_STEPS, "Running Phase 2 fixture collection...")
        print("Phase 1 pipeline and Phase 2 fixture collection complete.")
    else:
        print("Phase 1 pipeline complete. Use --with-fixtures to also collect World Cup fixtures.")


if __name__ == "__main__":
    main()
