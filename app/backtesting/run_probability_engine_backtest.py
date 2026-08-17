"""Backtest the central probability engine on the latest temporal test split."""

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

from app.core.probability_engine import predict_match_probabilities  # noqa: E402


DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
RESULT_LABELS = ["A", "D", "H"]
PRINT_LABELS = ["H", "D", "A"]
CALIBRATION_BUCKETS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CALIBRATION_COLUMNS = [
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "home_no_loss_probability",
    "away_no_loss_probability",
]
DRAW_ADJUSTMENT_THRESHOLD = 0.27
DRAW_ADJUSTMENT_MAX_GAP = 0.08
ADJUSTED_RECOMMENDED_LEAGUES = {"Bundesliga", "Serie A"}


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column found in a dataframe."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def select_target_column(df: pd.DataFrame) -> str:
    """Detect the match result target column."""
    target_column = first_existing_column(df, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError("No target column found. Expected one of: result, FTR, target")
    return target_column


def load_test_data(features_path: Path, test_ratio: float = 0.2) -> tuple[pd.DataFrame, str, str | None, str | None, str | None]:
    """Load feature rows and return the newest temporal test split."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    target_column = select_target_column(data)
    date_column = first_existing_column(data, DATE_CANDIDATES)
    home_team_column = first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = first_existing_column(data, AWAY_TEAM_CANDIDATES)

    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(PRINT_LABELS)].reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("At least 10 valid matches are required for the probability engine backtest.")

    split_index = int(len(data) * (1 - test_ratio))
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[split_index:].copy(), target_column, date_column, home_team_column, away_team_column


def build_match_features(row: pd.Series) -> dict[str, Any]:
    """Convert a feature row into model input while dropping labels and metadata."""
    ignored_columns = {
        "Date",
        "date",
        "match_date",
        "HomeTeam",
        "home_team",
        "AwayTeam",
        "away_team",
        "FTR",
        "result",
        "target",
        "source_file",
        "season",
    }
    return {key: value for key, value in row.to_dict().items() if key not in ignored_columns}


def adjusted_prediction_class(probabilities: dict[str, float], argmax_class: str) -> str:
    """Apply a conservative draw adjustment rule to final probabilities."""
    max_probability = max(probabilities.values())
    draw_probability = probabilities["D"]
    if draw_probability >= DRAW_ADJUSTMENT_THRESHOLD and (max_probability - draw_probability) <= DRAW_ADJUSTMENT_MAX_GAP:
        return "D"
    return argmax_class


def recommended_prediction_class(
    league: Any,
    predicted_class_argmax: str,
    predicted_class_adjusted: str,
) -> str:
    """Select the recommended prediction strategy by league."""
    if isinstance(league, str) and league in ADJUSTED_RECOMMENDED_LEAGUES:
        return predicted_class_adjusted
    return predicted_class_argmax


def prediction_row(
    row: pd.Series,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
) -> dict[str, Any]:
    """Run the probability engine for one match row and format the export row."""
    result = predict_match_probabilities(build_match_features(row))
    match_probabilities = result["probabilities"]["match"]
    analysis = result["analysis"]

    home_win_probability = match_probabilities["home_win"]
    draw_probability = match_probabilities["draw"]
    away_win_probability = match_probabilities["away_win"]
    probabilities_by_class = {"H": home_win_probability, "D": draw_probability, "A": away_win_probability}
    predicted_class_argmax = max(probabilities_by_class, key=probabilities_by_class.get)
    predicted_class_adjusted = adjusted_prediction_class(probabilities_by_class, predicted_class_argmax)
    league = row["league"] if "league" in row.index else pd.NA
    recommended_class = recommended_prediction_class(league, predicted_class_argmax, predicted_class_adjusted)
    actual_result = row[target_column]

    return {
        "date": row[date_column] if date_column else pd.NA,
        "home_team": row[home_team_column] if home_team_column else pd.NA,
        "away_team": row[away_team_column] if away_team_column else pd.NA,
        "league": league,
        "season": row["season"] if "season" in row.index else pd.NA,
        "actual_result": actual_result,
        "home_win_probability": home_win_probability,
        "draw_probability": draw_probability,
        "away_win_probability": away_win_probability,
        "home_no_loss_probability": result["probabilities"]["home"]["no_loss"],
        "away_no_loss_probability": result["probabilities"]["away"]["no_loss"],
        "predicted_class": predicted_class_argmax,
        "predicted_class_argmax": predicted_class_argmax,
        "predicted_class_adjusted": predicted_class_adjusted,
        "recommended_prediction_class": recommended_class,
        "is_correct": predicted_class_argmax == actual_result,
        "is_correct_argmax": predicted_class_argmax == actual_result,
        "is_correct_adjusted": predicted_class_adjusted == actual_result,
        "is_correct_recommended": recommended_class == actual_result,
        "draw_warning": predicted_class_adjusted == "D" and recommended_class != "D",
        "confidence_score": analysis["confidence_score"],
        "match_profile": analysis["match_profile"],
        "model_version": analysis.get("model_version"),
        "calibrated_draw_signal": analysis.get("calibrated_draw_signal"),
        "is_draw_plausible": analysis.get("is_draw_plausible"),
        "is_strong_draw_signal": analysis.get("is_strong_draw_signal"),
        "is_very_strong_draw_signal": analysis.get("is_very_strong_draw_signal"),
        "favorite_team": analysis.get("favorite_team"),
        "favorite_probability": analysis.get("favorite_probability"),
        "uncertainty_score": analysis.get("uncertainty_score"),
        "top_two_margin": analysis.get("top_two_margin"),
    }


def multiclass_brier_score(report: pd.DataFrame) -> float:
    """Compute the mean multiclass Brier score for H/D/A probabilities."""
    total = 0.0
    for label, probability_column in [
        ("H", "home_win_probability"),
        ("D", "draw_probability"),
        ("A", "away_win_probability"),
    ]:
        actual = (report["actual_result"] == label).astype(float)
        total += ((report[probability_column] - actual) ** 2).mean()
    return float(total / 3)


def calibration_table(report: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    """Create bucketed calibration rows for one probability column."""
    actual_column_by_probability = {
        "home_win_probability": report["actual_result"] == "H",
        "draw_probability": report["actual_result"] == "D",
        "away_win_probability": report["actual_result"] == "A",
        "home_no_loss_probability": report["actual_result"].isin(["H", "D"]),
        "away_no_loss_probability": report["actual_result"].isin(["A", "D"]),
    }
    bucket_series = pd.cut(
        report[probability_column],
        bins=CALIBRATION_BUCKETS,
        include_lowest=True,
        right=False,
    )
    rows = []
    for bucket, bucket_data in report.groupby(bucket_series, observed=False):
        count = len(bucket_data)
        actual_values = actual_column_by_probability[probability_column].loc[bucket_data.index]
        avg_predicted = bucket_data[probability_column].mean() if count else 0.0
        observed = actual_values.mean() if count else 0.0
        rows.append(
            {
                "probability": probability_column,
                "bucket": str(bucket),
                "matches": count,
                "avg_predicted_probability": avg_predicted,
                "observed_frequency": observed,
                "calibration_gap": avg_predicted - observed,
            }
        )
    return pd.DataFrame(rows)


def print_metrics(report: pd.DataFrame) -> None:
    """Print global metrics and calibration tables."""
    probabilities_for_log_loss = report[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    probabilities_for_log_loss = probabilities_for_log_loss.div(probabilities_for_log_loss.sum(axis=1), axis=0)

    print("\nProbability engine backtest")
    print(f"Test matches: {len(report)}")
    print("\nActual class distribution:")
    print(report["actual_result"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nPredicted class distribution:")
    print(report["predicted_class"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nPredicted class distribution argmax:")
    print(report["predicted_class_argmax"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nPredicted class distribution adjusted:")
    print(report["predicted_class_adjusted"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nRecommended prediction class distribution:")
    print(report["recommended_prediction_class"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print(f"\nAccuracy argmax: {accuracy_score(report['actual_result'], report['predicted_class_argmax']):.4f}")
    print(f"Accuracy adjusted: {accuracy_score(report['actual_result'], report['predicted_class_adjusted']):.4f}")
    print(f"Accuracy recommended: {accuracy_score(report['actual_result'], report['recommended_prediction_class']):.4f}")
    adjusted_draw_matches = report[report["predicted_class_adjusted"] == "D"]
    print(f"Adjusted D predictions: {len(adjusted_draw_matches)}")
    if adjusted_draw_matches.empty:
        print("Accuracy on adjusted D predictions: not_available")
        print("Actual draw rate among adjusted D predictions: not_available")
    else:
        adjusted_draw_accuracy = accuracy_score(
            adjusted_draw_matches["actual_result"],
            adjusted_draw_matches["predicted_class_adjusted"],
        )
        actual_draw_rate = (adjusted_draw_matches["actual_result"] == "D").mean()
        print(f"Accuracy on adjusted D predictions: {adjusted_draw_accuracy:.4f}")
        print(f"Actual draw rate among adjusted D predictions: {actual_draw_rate:.4f}")
    draw_warning_matches = report[report["draw_warning"].astype(bool)]
    print(f"Draw warnings: {len(draw_warning_matches)}")
    if draw_warning_matches.empty:
        print("Actual draw rate among draw warnings: not_available")
    else:
        print(f"Actual draw rate among draw warnings: {(draw_warning_matches['actual_result'] == 'D').mean():.4f}")
    print(f"Log loss: {log_loss(report['actual_result'], probabilities_for_log_loss, labels=RESULT_LABELS):.4f}")
    print(f"Multiclass Brier score: {multiclass_brier_score(report):.4f}")

    for probability_column in CALIBRATION_COLUMNS:
        print(f"\nCalibration: {probability_column}")
        print(calibration_table(report, probability_column).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def run_probability_engine_backtest(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Apply the probability engine to the newest 20 percent of matches."""
    test_data, target_column, date_column, home_team_column, away_team_column = load_test_data(features_path)
    rows = [
        prediction_row(
            row=row,
            target_column=target_column,
            date_column=date_column,
            home_team_column=home_team_column,
            away_team_column=away_team_column,
        )
        for _, row in test_data.iterrows()
    ]
    report = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(f"Feature file: {features_path}")
    print(f"Temporal test split: newest {len(report)} matches")
    print_metrics(report)
    print(f"\nSaved probability engine backtest to: {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backtest the central probability engine.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the probability engine backtest from the command line."""
    args = parse_args()
    run_probability_engine_backtest(features_path=args.features, output_path=args.output)


if __name__ == "__main__":
    main()
