"""Run the Ligue 1 API-Football data update pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOG_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_football_update_log.txt"


def script_path(name: str) -> Path:
    """Return an absolute path for a repository script."""
    return SCRIPTS_DIR / name


def log_line(message: str) -> None:
    """Append one line to the pipeline log and print it."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_script(step_label: str, script_name: str, args: list[str], optional: bool = False) -> bool:
    """Run one pipeline script with check=True and return whether it executed."""
    path = script_path(script_name)
    if not path.exists():
        message = "SKIP: script not found"
        log_line(f"{step_label} {message} ({script_name})")
        return False

    command = [sys.executable, str(path), *args]
    log_line(f"{step_label} {script_name}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        if optional:
            log_line(f"{step_label} FAILED but optional: {exc}")
            return False
        raise RuntimeError(f"{step_label} failed: {script_name}") from exc
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Update the Ligue 1 API-Football pipeline.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--league-id", type=int, default=61)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-fixtures", type=int, default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the update pipeline in the required order."""
    args = parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_line("=== Ligue 1 API-Football pipeline update start ===")
    log_line(f"season={args.season} league_id={args.league_id} force={args.force} skip_fetch={args.skip_fetch}")

    total_steps = 6
    step = 0

    fetch_args = ["--league-id", str(args.league_id), "--season", str(args.season)]
    if args.force:
        fetch_args.append("--force")
    if args.limit_fixtures is not None:
        fetch_args.extend(["--limit-fixtures", str(args.limit_fixtures)])

    if not args.skip_fetch:
        step += 1
        run_script(f"[{step}/{total_steps}] Fetch API-Football", "fetch_api_football_ligue1_data.py", fetch_args)
    else:
        log_line(f"[{step + 1}/{total_steps}] Fetch API-Football SKIP: requested by --skip-fetch")
        step += 1

    step += 1
    run_script(
        f"[{step}/{total_steps}] Parse context",
        "parse_api_football_ligue1_context.py",
        ["--league-id", str(args.league_id), "--season", str(args.season)],
    )

    step += 1
    run_script(
        f"[{step}/{total_steps}] Build prematch features",
        "build_api_football_ligue1_prematch_features.py",
        ["--season", str(args.season)],
    )

    step += 1
    run_script(
        f"[{step}/{total_steps}] Build formation performance features",
        "build_ligue1_formation_performance_features.py",
        ["--season", str(args.season)],
    )

    step += 1
    run_script(
        f"[{step}/{total_steps}] Build injury impact features",
        "build_api_football_ligue1_injury_impact_features.py",
        ["--season", str(args.season)],
        optional=True,
    )

    step += 1
    run_script(
        f"[{step}/{total_steps}] Build match stakes features",
        "build_api_football_ligue1_match_stakes_features.py",
        ["--season", str(args.season), "--league-id", str(args.league_id)],
        optional=True,
    )

    log_line("=== Ligue 1 API-Football pipeline update complete ===")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        log_line(f"PIPELINE FAILED: {exc}")
        raise SystemExit(1) from exc
