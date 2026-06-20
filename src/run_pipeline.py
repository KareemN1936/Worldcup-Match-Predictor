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

SQUAD_STEPS = [
    "collect_squads.py",
    "build_squad_features.py",
]

FOTMOB_STEPS = [
    "collect_fotmob_test.py",
    "collect_fotmob_matches.py",
    "collect_fotmob_lineups.py",
    "collect_fotmob_player_stats.py",
    "build_fotmob_features.py",
    "build_fotmob_rolling_features.py",
]

ANALYSIS_STEPS = [
    "predict_match.py --all-fixtures",
    "match_analysis.py --fixtures",
]

UPDATE_PREDICTION_STEPS = [
    *FIXTURE_STEPS,
    *FOTMOB_STEPS,
    *ANALYSIS_STEPS,
]


def _run_steps(steps: list[str], label: str) -> None:
    src_dir = Path(__file__).resolve().parent
    print(label, flush=True)
    for number, step in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] Running {step}...", flush=True)
        parts = step.split()
        subprocess.run([sys.executable, str(src_dir / parts[0]), *parts[1:]], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the World Cup predictor pipeline.")
    parser.add_argument("--with-fixtures", action="store_true", help="Also collect World Cup fixtures and teams from football-data.org.")
    parser.add_argument("--fixtures-only", action="store_true", help="Only collect World Cup fixtures and teams.")
    parser.add_argument("--with-squads", action="store_true", help="Also collect World Cup squads and build basic squad features.")
    parser.add_argument("--squads-only", action="store_true", help="Only collect World Cup squads and build basic squad features.")
    parser.add_argument("--with-fotmob", action="store_true", help="Also run optional PyFotMob/FotMob enrichment steps.")
    parser.add_argument("--fotmob-only", action="store_true", help="Only run optional PyFotMob/FotMob enrichment steps.")
    parser.add_argument("--with-analysis", action="store_true", help="Also generate fixture predictions and match analysis reports.")
    parser.add_argument("--analysis-only", action="store_true", help="Only generate fixture predictions and match analysis reports.")
    parser.add_argument(
        "--update-predictions",
        action="store_true",
        help="Refresh fixtures, FotMob data, predictions, and match analysis reports without retraining the historical model.",
    )
    args = parser.parse_args()

    if args.update_predictions:
        _run_steps(UPDATE_PREDICTION_STEPS, "Updating World Cup predictions from latest fixture and FotMob data...")
        print("Prediction update complete.")
        return

    if args.analysis_only:
        _run_steps(ANALYSIS_STEPS, "Running prediction analysis report generation...")
        print("Prediction analysis reports complete.")
        return

    if args.fotmob_only:
        _run_steps(FOTMOB_STEPS, "Running Phase 4 experimental PyFotMob enrichment...")
        print("PyFotMob enrichment complete.")
        return

    if args.squads_only:
        _run_steps(SQUAD_STEPS, "Running Phase 3 squad collection...")
        print("Squad collection complete.")
        return

    if args.fixtures_only:
        _run_steps(FIXTURE_STEPS, "Running Phase 2 fixture collection...")
        print("Fixture collection complete.")
        return

    _run_steps(PHASE_1_STEPS, "Running Phase 1 historical model pipeline...")

    if args.with_fixtures:
        _run_steps(FIXTURE_STEPS, "Running Phase 2 fixture collection...")
        if args.with_squads:
            _run_steps(SQUAD_STEPS, "Running Phase 3 squad collection...")
        if args.with_fotmob:
            _run_steps(FOTMOB_STEPS, "Running Phase 4 experimental PyFotMob enrichment...")
            _run_steps(["build_features.py", "train_model.py", "evaluate_model.py"], "Rebuilding features and model after PyFotMob enrichment...")
        if args.with_squads and args.with_fotmob:
            print("Phase 1 pipeline, fixtures, squads, and PyFotMob enrichment complete.")
        elif args.with_fotmob:
            print("Phase 1 pipeline, fixture collection, and PyFotMob enrichment complete.")
        elif args.with_squads:
            print("Phase 1 pipeline, Phase 2 fixture collection, and Phase 3 squad collection complete.")
        else:
            print("Phase 1 pipeline and Phase 2 fixture collection complete.")
    elif args.with_squads:
        _run_steps(SQUAD_STEPS, "Running Phase 3 squad collection...")
        if args.with_fotmob:
            _run_steps(FOTMOB_STEPS, "Running Phase 4 experimental PyFotMob enrichment...")
            _run_steps(["build_features.py", "train_model.py", "evaluate_model.py"], "Rebuilding features and model after PyFotMob enrichment...")
            print("Phase 1 pipeline, squad collection, and PyFotMob enrichment complete.")
        else:
            print("Phase 1 pipeline and Phase 3 squad collection complete.")
    elif args.with_fotmob:
        _run_steps(FOTMOB_STEPS, "Running Phase 4 experimental PyFotMob enrichment...")
        _run_steps(["build_features.py", "train_model.py", "evaluate_model.py"], "Rebuilding features and model after PyFotMob enrichment...")
        print("Phase 1 pipeline and PyFotMob enrichment complete.")
    else:
        print("Phase 1 pipeline complete. Use --with-fixtures or --with-squads for later phases.")

    if args.with_analysis:
        _run_steps(ANALYSIS_STEPS, "Running prediction analysis report generation...")
        print("Prediction analysis reports complete.")


if __name__ == "__main__":
    main()
