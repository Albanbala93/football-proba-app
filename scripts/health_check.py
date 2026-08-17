from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_PATHS = [
    "data/raw/",
    "data/processed/matches_clean.csv",
    "data/processed/matches_features.csv",
    "models/outcome_models/home_win_model_classic.pkl",
    "models/outcome_models/away_win_model_classic.pkl",
    "models/outcome_models/draw_model_classic.pkl",
    "models/outcome_models_no_odds/home_win_model_no_odds.pkl",
    "data/predictions/probability_engine_backtest.csv",
    "data/predictions/probability_engine_no_odds_backtest.csv",
    "data/predictions/match_profiles_analysis.csv",
    "frontend/streamlit_app.py",
]

REQUIRED_FEATURE_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "home_form_5",
    "away_form_5",
    "home_goals_for_5",
    "away_goals_for_5",
    "home_goals_against_5",
    "away_goals_against_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_goal_diff_5",
    "away_goal_diff_5",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_elo_with_advantage",
    "elo_diff_with_home_advantage",
]


class HealthReport:
    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"OK - {message}")

    def warning(self, message: str) -> None:
        self.warnings += 1
        print(f"WARNING - {message}")

    def error(self, message: str) -> None:
        self.errors += 1
        print(f"ERROR - {message}")

    def final_status(self) -> int:
        print()
        if self.errors == 0 and self.warnings == 0:
            print("Health check OK")
            return 0
        print("Health check termine avec avertissements/erreurs")
        print(f"Warnings: {self.warnings}")
        print(f"Errors: {self.errors}")
        return 1 if self.errors else 0


def check_required_paths(report: HealthReport) -> None:
    print("\n[1] Fichiers et dossiers requis")
    for relative_path in REQUIRED_PATHS:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            report.ok(relative_path)
        else:
            report.error(f"Manquant: {relative_path}")


def load_features(report: HealthReport) -> pd.DataFrame | None:
    features_path = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
    if not features_path.exists():
        report.error("Impossible de charger matches_features.csv: fichier manquant")
        return None

    try:
        return pd.read_csv(features_path)
    except Exception as exc:
        report.error(f"Impossible de lire matches_features.csv: {exc}")
        return None


def check_feature_columns(report: HealthReport, features: pd.DataFrame | None) -> None:
    print("\n[2] Colonnes obligatoires de matches_features.csv")
    if features is None:
        return

    missing_columns = [column for column in REQUIRED_FEATURE_COLUMNS if column not in features.columns]
    if missing_columns:
        report.error(f"Colonnes manquantes: {', '.join(missing_columns)}")
        return
    report.ok("Toutes les colonnes obligatoires sont presentes")


def print_feature_summary(report: HealthReport, features: pd.DataFrame | None) -> pd.Timestamp | None:
    print("\n[3] Resume des features")
    if features is None:
        return None

    dates = pd.to_datetime(features["Date"], errors="coerce") if "Date" in features.columns else pd.Series(dtype="datetime64[ns]")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        report.error("Aucune date valide dans matches_features.csv")
        return None

    teams = pd.concat([features["HomeTeam"], features["AwayTeam"]], ignore_index=True).dropna().astype(str)
    unique_teams = sorted(teams.unique().tolist())
    print(f"Nombre de matchs: {len(features)}")
    print(f"Date min: {valid_dates.min().date()}")
    print(f"Date max: {valid_dates.max().date()}")
    print(f"Nombre d'equipes uniques: {len(unique_teams)}")
    print("Dernieres 10 equipes par ordre alphabetique:")
    for team in unique_teams[-10:]:
        print(f"- {team}")
    report.ok("Resume des features affiche")
    return valid_dates.max()


def check_import(report: HealthReport, label: str, import_func: Callable[[], object]) -> object | None:
    try:
        module = import_func()
    except Exception as exc:
        report.error(f"Import {label}: {exc}")
        return None
    report.ok(f"Import {label}")
    return module


def check_imports(report: HealthReport) -> object | None:
    print("\n[4] Imports des moteurs")
    check_import(report, "probability_engine.py", lambda: __import__("app.core.probability_engine", fromlist=["*"]))
    check_import(
        report,
        "probability_engine_no_odds.py",
        lambda: __import__("app.core.probability_engine_no_odds", fromlist=["*"]),
    )
    return check_import(
        report,
        "upcoming_match_predictor.py",
        lambda: __import__("app.core.upcoming_match_predictor", fromlist=["predict_upcoming_match"]),
    )


def quick_prediction(report: HealthReport, predictor_module: object | None, max_date: pd.Timestamp | None) -> None:
    print("\n[5] Test rapide prediction no-odds")
    if predictor_module is None:
        report.error("Test prediction ignore: upcoming_match_predictor.py non importe")
        return
    if max_date is None:
        report.error("Test prediction ignore: date max indisponible")
        return

    match_date = (max_date + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        result = predictor_module.predict_upcoming_match(
            "Arsenal",
            "Chelsea",
            match_date,
            mode="no-odds",
        )
    except Exception as exc:
        report.error(f"Prediction rapide Arsenal vs Chelsea: {exc}")
        return

    probabilities = result["prediction"]["probabilities"]["match"]
    print(f"Match test: Arsenal vs Chelsea")
    print(f"Date test: {match_date}")
    print(f"Home win: {probabilities['home_win']:.2%}")
    print(f"Draw: {probabilities['draw']:.2%}")
    print(f"Away win: {probabilities['away_win']:.2%}")
    report.ok("Prediction rapide no-odds")


def main() -> int:
    report = HealthReport()
    print("Health check football-proba-app")
    print(f"Racine projet: {PROJECT_ROOT}")

    check_required_paths(report)
    features = load_features(report)
    check_feature_columns(report, features)
    max_date = print_feature_summary(report, features)
    predictor_module = check_imports(report)
    quick_prediction(report, predictor_module, max_date)

    return report.final_status()


if __name__ == "__main__":
    raise SystemExit(main())
