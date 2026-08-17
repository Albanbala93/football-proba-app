from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Nettoyage des matchs", "app/data_pipeline/clean_matches.py"),
    ("Generation des equipes actuelles", "scripts/generate_current_teams.py"),
    ("Generation des fixtures a venir", "scripts/generate_upcoming_fixtures.py"),
    ("Construction des features", "app/data_pipeline/build_features.py"),
    ("Modele baseline", "app/core/baseline_model.py"),
    ("Modeles outcome", "app/core/outcome_models.py"),
    ("Modeles outcome calibres", "app/core/calibrated_outcome_models.py"),
    ("Modeles outcome sans cotes", "app/core/outcome_models_no_odds.py"),
    (
        "Backtest probability engine",
        "app/backtesting/run_probability_engine_backtest.py",
    ),
    (
        "Backtest probability engine sans cotes",
        "app/backtesting/run_probability_engine_no_odds_backtest.py",
    ),
    ("Analyse des profils de matchs", "app/backtesting/analyze_match_profiles.py"),
]

ESSENTIAL_FILES = [
    "data/processed/matches_clean.csv",
    "data/reference/current_teams.csv",
    "data/reference/upcoming_fixtures.csv",
    "data/processed/matches_features.csv",
    "models/outcome_models/home_win_model_classic.pkl",
    "models/outcome_models/away_win_model_classic.pkl",
    "models/outcome_models/draw_model_classic.pkl",
    "models/outcome_models_no_odds/home_win_model_no_odds.pkl",
    "data/predictions/probability_engine_backtest.csv",
    "data/predictions/probability_engine_no_odds_backtest.csv",
    "data/predictions/match_profiles_analysis.csv",
]


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)

    if hours:
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:.1f}s"
    return f"{remaining_seconds:.1f}s"


def run_step(index: int, label: str, script_path: str) -> None:
    print(f"\n[{index}/{len(STEPS)}] {label}")
    print(f"> python {script_path}")

    try:
        subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"\nÉchec à l'étape {index}: {label}")
        print(f"Script en échec: {script_path}")
        print(f"Code retour: {exc.returncode}")
        raise


def verify_outputs() -> None:
    print("\nVerification des fichiers essentiels...")
    missing_files = [
        file_path
        for file_path in ESSENTIAL_FILES
        if not (PROJECT_ROOT / file_path).exists()
    ]

    if missing_files:
        print("\nFichiers essentiels manquants:")
        for file_path in missing_files:
            print(f"- {file_path}")
        raise FileNotFoundError("Des fichiers essentiels sont manquants.")

    for file_path in ESSENTIAL_FILES:
        print(f"OK - {file_path}")


def main() -> int:
    start_time = time.perf_counter()

    print("Rebuild complet de football-proba-app")
    print(f"Racine projet: {PROJECT_ROOT}")

    try:
        for index, (label, script_path) in enumerate(STEPS, start=1):
            run_step(index, label, script_path)
        verify_outputs()
    except Exception:
        total_time = time.perf_counter() - start_time
        print(f"\nRebuild interrompu après {format_duration(total_time)}.")
        return 1

    total_time = time.perf_counter() - start_time
    print(f"\nTemps total: {format_duration(total_time)}")
    print("Rebuild terminé avec succès.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
