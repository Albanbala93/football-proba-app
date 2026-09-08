"""Backtest the no-odds probability engine and compare it with current engine output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.outcome_models_no_odds import FEATURE_CANDIDATES as NO_ODDS_FEATURE_CANDIDATES  # noqa: E402
from app.core.probability_engine_no_odds import predict_match_probabilities_no_odds  # noqa: E402


DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_no_odds_backtest.csv"
CURRENT_ENGINE_BACKTEST_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
RESULT_LABELS = ["A", "D", "H"]
PRINT_LABELS = ["H", "D", "A"]
CALIBRATION_BUCKETS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing column."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def select_target_column(df: pd.DataFrame) -> str:
    """Detect target column."""
    target = first_existing_column(df, TARGET_CANDIDATES)
    if target is None:
        raise ValueError("No target column found. Expected one of: result, FTR, target")
    return target


def load_test_data(features_path: Path, test_ratio: float = 0.2) -> tuple[pd.DataFrame, str, str | None, str | None, str | None]:
    """Load newest temporal test split."""
    data = pd.read_csv(features_path)
    target_column = select_target_column(data)
    date_column = first_existing_column(data, DATE_CANDIDATES)
    home_team_column = first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = first_existing_column(data, AWAY_TEAM_CANDIDATES)
    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(PRINT_LABELS)].reset_index(drop=True)

    # Matches added by the API-Football fallback have no match-stats columns
    # (shots, corners, fouls, cards), so their rolling-average features are
    # missing -- drop rows missing them, same as during training.
    feature_columns = [column for column in NO_ODDS_FEATURE_CANDIDATES if column in data.columns]
    before_rows = len(data)
    data = data.dropna(subset=feature_columns).reset_index(drop=True)
    dropped_rows = before_rows - len(data)
    if dropped_rows:
        print(f"Dropped {dropped_rows} row(s) missing stats features (e.g. API-Football fallback matches).")

    split_index = int(len(data) * (1 - test_ratio))
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[split_index:].copy(), target_column, date_column, home_team_column, away_team_column


def build_match_features(row: pd.Series) -> dict[str, Any]:
    """Drop labels, metadata, and odds columns."""
    ignored = {"Date", "date", "match_date", "HomeTeam", "home_team", "AwayTeam", "away_team", "FTR", "result", "target", "source_file", "season", "B365H", "B365D", "B365A", "home_odds", "draw_odds", "away_odds"}
    return {key: value for key, value in row.to_dict().items() if key not in ignored}


def prediction_row(row: pd.Series, target_column: str, date_column: str | None, home_team_column: str | None, away_team_column: str | None) -> dict[str, Any]:
    """Run no-odds engine for one row."""
    result = predict_match_probabilities_no_odds(build_match_features(row))
    match_probs = result["probabilities"]["match"]
    home_win = match_probs["home_win"]
    draw = match_probs["draw"]
    away_win = match_probs["away_win"]
    predicted_class = max({"H": home_win, "D": draw, "A": away_win}, key={"H": home_win, "D": draw, "A": away_win}.get)
    actual_result = row[target_column]
    return {
        "date": row[date_column] if date_column else pd.NA,
        "home_team": row[home_team_column] if home_team_column else pd.NA,
        "away_team": row[away_team_column] if away_team_column else pd.NA,
        "actual_result": actual_result,
        "home_win_probability": home_win,
        "draw_probability": draw,
        "away_win_probability": away_win,
        "home_no_loss_probability": result["probabilities"]["home"]["no_loss"],
        "away_no_loss_probability": result["probabilities"]["away"]["no_loss"],
        "predicted_class": predicted_class,
        "is_correct": predicted_class == actual_result,
        "model_version": result["analysis"]["model_version"],
        "favorite_team": result["analysis"]["favorite_team"],
        "favorite_probability": result["analysis"]["favorite_probability"],
        "uncertainty_score": result["analysis"]["uncertainty_score"],
        "top_two_margin": result["analysis"]["top_two_margin"],
    }


def multiclass_brier_score(report: pd.DataFrame) -> float:
    """Compute mean multiclass Brier score."""
    total = 0.0
    for label, column in [("H", "home_win_probability"), ("D", "draw_probability"), ("A", "away_win_probability")]:
        actual = (report["actual_result"] == label).astype(float)
        total += ((report[column] - actual) ** 2).mean()
    return float(total / 3)


def metrics(report: pd.DataFrame) -> dict[str, float]:
    """Return comparable 1N2 metrics."""
    probs = report[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    probs = probs.div(probs.sum(axis=1), axis=0)
    return {
        "accuracy": float(accuracy_score(report["actual_result"], report["predicted_class"])),
        "log_loss": float(log_loss(report["actual_result"], probs, labels=RESULT_LABELS)),
        "brier_score": multiclass_brier_score(report),
    }


def calibration_draw(report: pd.DataFrame) -> pd.DataFrame:
    """Bucketed draw calibration."""
    buckets = pd.cut(report["draw_probability"], bins=CALIBRATION_BUCKETS, include_lowest=True, right=False)
    rows = []
    for bucket, bucket_data in report.groupby(buckets, observed=False):
        count = len(bucket_data)
        avg_pred = bucket_data["draw_probability"].mean() if count else 0.0
        observed = (bucket_data["actual_result"] == "D").mean() if count else 0.0
        rows.append({"bucket": str(bucket), "matches": count, "avg_draw_probability": avg_pred, "actual_draw_rate": observed, "calibration_gap": avg_pred - observed})
    return pd.DataFrame(rows)


def print_report(report: pd.DataFrame) -> None:
    """Print no-odds metrics and comparison with current engine if available."""
    no_odds_metrics = metrics(report)
    print("\nNo-odds probability engine backtest")
    print(f"Test matches: {len(report)}")
    print("Predicted distribution:")
    print(report["predicted_class"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print(f"Accuracy: {no_odds_metrics['accuracy']:.4f}")
    print(f"Log loss: {no_odds_metrics['log_loss']:.4f}")
    print(f"Brier score: {no_odds_metrics['brier_score']:.4f}")
    print("\nDraw calibration:")
    print(calibration_draw(report).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if CURRENT_ENGINE_BACKTEST_PATH.exists():
        current = pd.read_csv(CURRENT_ENGINE_BACKTEST_PATH)
        current_metrics = metrics(current)
        print("\nComparison with current engine backtest:")
        print(f"Current accuracy: {current_metrics['accuracy']:.4f} | No-odds accuracy: {no_odds_metrics['accuracy']:.4f}")
        print(f"Current log loss: {current_metrics['log_loss']:.4f} | No-odds log loss: {no_odds_metrics['log_loss']:.4f}")
        print(f"Current Brier: {current_metrics['brier_score']:.4f} | No-odds Brier: {no_odds_metrics['brier_score']:.4f}")


def run_no_odds_backtest(features_path: Path = DEFAULT_FEATURES_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Run no-odds backtest and export predictions."""
    test_data, target_column, date_column, home_team_column, away_team_column = load_test_data(features_path)
    rows = [prediction_row(row, target_column, date_column, home_team_column, away_team_column) for _, row in test_data.iterrows()]
    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Feature file: {features_path}")
    print(f"Saved no-odds backtest to: {output_path}")
    print_report(report)
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Backtest no-odds probability engine.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run no-odds backtest."""
    args = parse_args()
    run_no_odds_backtest(args.features, args.output)


if __name__ == "__main__":
    main()
