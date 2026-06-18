import subprocess
import sys
from pathlib import Path


STEPS = [
    "collect_historical_matches.py",
    "build_elo.py",
    "build_features.py",
    "train_model.py",
    "evaluate_model.py",
]


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    for number, step in enumerate(STEPS, start=1):
        print(f"[{number}/{len(STEPS)}] Running {step}...")
        subprocess.run([sys.executable, str(src_dir / step)], check=True)
    print("Phase 1 pipeline complete.")


if __name__ == "__main__":
    main()
