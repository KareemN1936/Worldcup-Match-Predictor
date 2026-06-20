import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import REPORTS_DIR, ROOT_DIR


LOG_DIR = REPORTS_DIR / "update_logs"
STATUS_PATH = REPORTS_DIR / "update_status.json"
LOCK_PATH = REPORTS_DIR / "update.lock"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(status: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _is_lock_active(max_age_minutes: int = 180) -> bool:
    if not LOCK_PATH.exists():
        return False
    age_seconds = time.time() - LOCK_PATH.stat().st_mtime
    return age_seconds < max_age_minutes * 60


def _acquire_lock() -> bool:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if _is_lock_active():
        return False
    LOCK_PATH.write_text(str(Path.cwd()), encoding="utf-8")
    return True


def _release_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()


def run_update() -> int:
    if not _acquire_lock():
        previous = _read_status()
        skipped = {
            **previous,
            "last_attempt_at": _utc_now(),
            "last_status": "skipped",
            "message": "Another update is already running.",
        }
        _write_status(skipped)
        print("Another update is already running. Skipping this run.")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"update_{timestamp}.log"
    command = [sys.executable, str(ROOT_DIR / "src" / "run_pipeline.py"), "--update-predictions"]

    status = {
        "last_attempt_at": started_at,
        "last_status": "running",
        "command": command,
        "log_path": str(log_path),
    }
    _write_status(status)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"Started at: {started_at}\n")
            log_file.write(f"Command: {' '.join(command)}\n\n")
            log_file.flush()
            process = subprocess.run(
                command,
                cwd=ROOT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        finished_at = _utc_now()
        final_status = {
            "last_attempt_at": started_at,
            "last_finished_at": finished_at,
            "last_status": "success" if process.returncode == 0 else "failed",
            "return_code": process.returncode,
            "command": command,
            "log_path": str(log_path),
            "outputs": {
                "fixture_predictions": str(REPORTS_DIR / "fixture_predictions.csv"),
                "match_analysis": str(REPORTS_DIR / "match_analysis"),
                "fotmob_feature_coverage": str(REPORTS_DIR / "fotmob_feature_coverage.csv"),
            },
        }
        _write_status(final_status)
        print(f"Update {final_status['last_status']}. Log: {log_path}")
        return process.returncode
    finally:
        _release_lock()


def watch(interval_minutes: int) -> None:
    print(f"Starting auto-update watcher. Interval: {interval_minutes} minutes")
    while True:
        run_update()
        time.sleep(max(interval_minutes, 1) * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically refresh World Cup predictions.")
    parser.add_argument("--once", action="store_true", help="Run one update and exit.")
    parser.add_argument("--watch", action="store_true", help="Keep running and update on an interval.")
    parser.add_argument("--interval-minutes", type=int, default=60, help="Watcher interval in minutes. Default: 60.")
    args = parser.parse_args()

    if args.watch:
        watch(args.interval_minutes)
        return

    raise SystemExit(run_update())


if __name__ == "__main__":
    main()
