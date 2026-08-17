"""Evaluate logged upcoming predictions against cleaned match results."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "predictions" / "upcoming_predictions_log.csv"
DEFAULT_MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "upcoming_predictions_evaluated.csv"

RESULT_CANDIDATES = ["FTR", "result", "target"]
RESULT_LABELS = ["H", "D", "A"]
EXPECTED_LOG_COLUMNS = [
    "prediction_created_at",
    "match_date",
    "home_team",
    "away_team",
    "mode",
    "odds_source",
    "odds_home",
    "odds_draw",
    "odds_away",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "home_no_loss_probability",
    "away_no_loss_probability",
    "favorite_team",
    "favorite_probability",
    "match_profile",
    "confidence_score",
    "is_draw_plausible",
    "is_strong_draw_signal",
    "is_very_strong_draw_signal",
    "calibrated_draw_signal",
    "model_version",
]
EVALUATION_COLUMNS = [
    "actual_result",
    "predicted_class",
    "is_correct",
    "brier_score_1N2",
    "log_loss_match",
    "evaluation_status",
]
PROBABILITY_COLUMNS = {
    "H": "home_win_probability",
    "D": "draw_probability",
    "A": "away_win_probability",
}


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column found in a dataframe."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_team(value: Any) -> str:
    """Normalize a team name for exact matching with minor whitespace tolerance."""
    return str(value).strip()


def normalize_date_series(series: pd.Series) -> pd.Series:
    """Normalize date values to yyyy-mm-dd strings."""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def load_prediction_log(log_path: Path) -> pd.DataFrame:
    """Load the upcoming predictions log."""
    if not log_path.exists():
        return pd.DataFrame(columns=EXPECTED_LOG_COLUMNS)
    log = pd.read_csv(log_path)
    required_columns = [
        "match_date",
        "home_team",
        "away_team",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
    ]
    missing = [column for column in required_columns if column not in log.columns]
    if missing:
        raise ValueError(f"Prediction log is missing required columns: {', '.join(missing)}")
    return log


def load_actual_matches(matches_path: Path) -> tuple[pd.DataFrame, str]:
    """Load cleaned matches and return them with the detected result column."""
    if not matches_path.exists():
        raise FileNotFoundError(f"Cleaned matches file not found: {matches_path}")

    matches = pd.read_csv(matches_path)
    missing = [column for column in ["Date", "HomeTeam", "AwayTeam"] if column not in matches.columns]
    if missing:
        raise ValueError(f"Cleaned matches file is missing required columns: {', '.join(missing)}")

    result_column = first_existing_column(matches, RESULT_CANDIDATES)
    if result_column is None:
        raise ValueError("No result column found in cleaned matches. Expected FTR, result, or target.")

    matches = matches.copy()
    matches["_match_date"] = normalize_date_series(matches["Date"])
    matches["_home_team"] = matches["HomeTeam"].map(normalize_team)
    matches["_away_team"] = matches["AwayTeam"].map(normalize_team)
    matches[result_column] = matches[result_column].astype(str).str.upper().str.strip()
    return matches, result_column


def build_actual_lookup(matches: pd.DataFrame, result_column: str) -> dict[tuple[str, str, str], str]:
    """Build a lookup keyed by date, home team, and away team."""
    valid_matches = matches.dropna(subset=["_match_date"])
    lookup: dict[tuple[str, str, str], str] = {}
    for _, row in valid_matches.iterrows():
        result = str(row[result_column]).upper().strip()
        if result not in RESULT_LABELS:
            continue
        key = (row["_match_date"], row["_home_team"], row["_away_team"])
        lookup[key] = result
    return lookup


def predicted_class(row: pd.Series) -> str:
    """Return H, D, or A from logged 1N2 probabilities."""
    probabilities = {
        label: float(row[column])
        for label, column in PROBABILITY_COLUMNS.items()
    }
    return max(probabilities, key=probabilities.get)


def brier_score_1n2(row: pd.Series, actual_result: str) -> float:
    """Compute mean multiclass Brier score for one match."""
    total = 0.0
    for label, column in PROBABILITY_COLUMNS.items():
        actual = 1.0 if actual_result == label else 0.0
        probability = float(row[column])
        total += (probability - actual) ** 2
    return total / len(PROBABILITY_COLUMNS)


def log_loss_match(row: pd.Series, actual_result: str) -> float:
    """Compute per-match multiclass log loss, clipping probabilities for stability."""
    probability = float(row[PROBABILITY_COLUMNS[actual_result]])
    clipped = min(max(probability, 1e-15), 1 - 1e-15)
    return -math.log(clipped)


def evaluate_row(row: pd.Series, actual_lookup: dict[tuple[str, str, str], str]) -> dict[str, Any]:
    """Evaluate one logged prediction when the actual match is available."""
    match_date = pd.to_datetime(row["match_date"], errors="coerce")
    normalized_date = match_date.strftime("%Y-%m-%d") if not pd.isna(match_date) else ""
    key = (
        normalized_date,
        normalize_team(row["home_team"]),
        normalize_team(row["away_team"]),
    )
    actual_result = actual_lookup.get(key)

    if actual_result is None:
        return {
            "actual_result": "",
            "predicted_class": "",
            "is_correct": "",
            "brier_score_1N2": "",
            "log_loss_match": "",
            "evaluation_status": "pending",
        }

    prediction = predicted_class(row)
    return {
        "actual_result": actual_result,
        "predicted_class": prediction,
        "is_correct": prediction == actual_result,
        "brier_score_1N2": brier_score_1n2(row, actual_result),
        "log_loss_match": log_loss_match(row, actual_result),
        "evaluation_status": "evaluated",
    }


def print_summary(report: pd.DataFrame) -> None:
    """Print evaluation summary metrics."""
    total_predictions = len(report)
    evaluated = report[report["evaluation_status"] == "evaluated"].copy()
    pending_count = int((report["evaluation_status"] == "pending").sum())

    print("Upcoming predictions evaluation")
    print(f"Total predictions: {total_predictions}")
    print(f"Evaluated: {len(evaluated)}")
    print(f"Pending: {pending_count}")

    if evaluated.empty:
        print("Accuracy evaluated: not_available")
        print("Mean Brier score: not_available")
        print("Mean log loss: not_available")
        return

    accuracy = evaluated["is_correct"].astype(bool).mean()
    mean_brier = pd.to_numeric(evaluated["brier_score_1N2"], errors="coerce").mean()
    mean_log_loss = pd.to_numeric(evaluated["log_loss_match"], errors="coerce").mean()
    print(f"Accuracy evaluated: {accuracy:.4f}")
    print(f"Mean Brier score: {mean_brier:.4f}")
    print(f"Mean log loss: {mean_log_loss:.4f}")


def evaluate_upcoming_predictions(
    log_path: Path = DEFAULT_LOG_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Evaluate logged upcoming predictions and export the evaluated CSV."""
    log = load_prediction_log(log_path)

    if log.empty:
        report = pd.DataFrame(columns=list(log.columns) + EVALUATION_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

        print(f"Prediction log: {log_path}")
        if not log_path.exists():
            print("Prediction log not found yet. Created an empty evaluated export.")
        else:
            print("Prediction log is empty. Created an empty evaluated export.")
        print(f"Cleaned matches: {matches_path}")
        print_summary(report)
        print(f"Saved evaluated predictions to: {output_path}")
        return report

    matches, result_column = load_actual_matches(matches_path)
    actual_lookup = build_actual_lookup(matches, result_column)

    evaluations = [evaluate_row(row, actual_lookup) for _, row in log.iterrows()]
    report = pd.concat([log.reset_index(drop=True), pd.DataFrame(evaluations)], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(f"Prediction log: {log_path}")
    print(f"Cleaned matches: {matches_path}")
    print_summary(report)
    print(f"Saved evaluated predictions to: {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate logged upcoming match predictions.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run upcoming prediction evaluation."""
    args = parse_args()
    evaluate_upcoming_predictions(args.log, args.matches, args.output)


if __name__ == "__main__":
    main()
