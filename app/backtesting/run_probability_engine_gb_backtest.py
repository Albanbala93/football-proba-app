"""Backtest the gradient boosting probability engine on the latest temporal split."""

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

from app.core.probability_engine_gb import predict_match_probabilities_gb  # noqa: E402


DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_gb_backtest.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
LEAGUE_CANDIDATES = ["league", "League"]
SEASON_CANDIDATES = ["season", "Season"]
PRINT_LABELS = ["H", "D", "A"]
RESULT_LABELS = ["A", "D", "H"]
DRAW_WARNING_COUNT_LABEL = "draw_warning"


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


def load_test_data(
    features_path: Path,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, str, str | None, str | None, str | None, str | None, str | None]:
    """Load feature rows and return the newest temporal test split."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    target_column = select_target_column(data)
    date_column = first_existing_column(data, DATE_CANDIDATES)
    home_team_column = first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = first_existing_column(data, AWAY_TEAM_CANDIDATES)
    league_column = first_existing_column(data, LEAGUE_CANDIDATES)
    season_column = first_existing_column(data, SEASON_CANDIDATES)

    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(PRINT_LABELS)].reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("At least 10 valid matches are required for the GB probability engine backtest.")

    split_index = int(len(data) * (1 - test_ratio))
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[split_index:].copy(), target_column, date_column, home_team_column, away_team_column, league_column, season_column


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


def prediction_row(
    row: pd.Series,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
    league_column: str | None,
    season_column: str | None,
) -> dict[str, Any]:
    """Run the GB probability engine for one match row and format the export row."""
    match_features = build_match_features(row)
    if league_column and league_column in row.index and pd.notna(row[league_column]):
        match_features["league"] = row[league_column]
    result = predict_match_probabilities_gb(match_features)
    match_probabilities = result["probabilities"]["match"]
    analysis = result["analysis"]

    actual_result = row[target_column]
    predicted_class_argmax = analysis["predicted_class_argmax"]
    predicted_class_adjusted = analysis["predicted_class_adjusted"]
    recommended_class = analysis["recommended_prediction_class"]

    return {
        "date": row[date_column] if date_column else pd.NA,
        "league": row[league_column] if league_column else pd.NA,
        "season": row[season_column] if season_column else pd.NA,
        "home_team": row[home_team_column] if home_team_column else pd.NA,
        "away_team": row[away_team_column] if away_team_column else pd.NA,
        "actual_result": actual_result,
        "home_win_probability": match_probabilities["home_win"],
        "draw_probability": match_probabilities["draw"],
        "away_win_probability": match_probabilities["away_win"],
        "predicted_class_argmax": predicted_class_argmax,
        "predicted_class_adjusted": predicted_class_adjusted,
        "recommended_prediction_class": recommended_class,
        "is_correct_argmax": predicted_class_argmax == actual_result,
        "is_correct_adjusted": predicted_class_adjusted == actual_result,
        "is_correct_recommended": recommended_class == actual_result,
        "draw_warning": bool(analysis.get("draw_warning")),
        "confidence_score": analysis["confidence_score"],
        "match_profile": analysis["match_profile"],
        "model_version": analysis.get("model_version"),
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


def print_metrics(report: pd.DataFrame) -> None:
    """Print global metrics and key distributions."""
    probabilities_for_log_loss = report[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    probabilities_for_log_loss = probabilities_for_log_loss.div(probabilities_for_log_loss.sum(axis=1), axis=0)

    print("\nProbability engine GB backtest")
    print(f"Test matches: {len(report)}")
    print("\nActual class distribution:")
    print(report["actual_result"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nPredicted class distribution argmax:")
    print(report["predicted_class_argmax"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nPredicted class distribution adjusted:")
    print(report["predicted_class_adjusted"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nRecommended prediction class distribution:")
    print(report["recommended_prediction_class"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print(f"\nAccuracy argmax: {accuracy_score(report['actual_result'], report['predicted_class_argmax']):.4f}")
    print(f"Accuracy adjusted: {accuracy_score(report['actual_result'], report['predicted_class_adjusted']):.4f}")
    print(f"Accuracy recommended: {accuracy_score(report['actual_result'], report['recommended_prediction_class']):.4f}")
    print(f"Log loss: {log_loss(report['actual_result'], probabilities_for_log_loss, labels=RESULT_LABELS):.4f}")
    print(f"Multiclass Brier score: {multiclass_brier_score(report):.4f}")

    draw_warning_matches = report[report[DRAW_WARNING_COUNT_LABEL].astype(bool)]
    print(f"Draw warnings: {len(draw_warning_matches)}")
    if draw_warning_matches.empty:
        print("Actual draw rate among draw warnings: not_available")
    else:
        print(f"Actual draw rate among draw warnings: {(draw_warning_matches['actual_result'] == 'D').mean():.4f}")


def run_probability_engine_gb_backtest(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Apply the GB probability engine to the newest 20 percent of matches."""
    test_data, target_column, date_column, home_team_column, away_team_column, league_column, season_column = load_test_data(
        features_path
    )
    rows = [
        prediction_row(
            row=row,
            target_column=target_column,
            date_column=date_column,
            home_team_column=home_team_column,
            away_team_column=away_team_column,
            league_column=league_column,
            season_column=season_column,
        )
        for _, row in test_data.iterrows()
    ]
    report = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(f"Feature file: {features_path}")
    print(f"Temporal test split: newest {len(report)} matches")
    print_metrics(report)
    print(f"\nSaved GB probability engine backtest to: {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backtest the gradient boosting probability engine.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the GB probability engine backtest from the command line."""
    args = parse_args()
    run_probability_engine_gb_backtest(features_path=args.features, output_path=args.output)


if __name__ == "__main__":
    main()
