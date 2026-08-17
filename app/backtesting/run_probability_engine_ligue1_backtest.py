"""Backtest the Ligue 1 specialist probability engine on the latest Ligue 1 split."""

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

from app.core.probability_engine_ligue1 import load_ligue1_models  # noqa: E402
from app.core.probability_engine_ligue1 import predict_match_probabilities_ligue1  # noqa: E402


DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_ligue1_backtest.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
PRINT_LABELS = ["H", "D", "A"]
RESULT_LABELS = ["A", "D", "H"]
LEAGUE_NAME = "Ligue 1"
OUTPUT_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "actual_result",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "home_no_loss_probability",
    "away_no_loss_probability",
    "predicted_class_argmax",
    "predicted_class_adjusted",
    "recommended_prediction_class",
    "is_correct_argmax",
    "is_correct_adjusted",
    "is_correct_recommended",
    "confidence_score",
    "match_profile",
    "model_version",
    "favorite_team",
    "favorite_probability",
    "uncertainty_score",
    "top_two_margin",
]


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


def required_model_features() -> list[str]:
    """Return the union of feature columns required by the loaded specialist models."""
    required: set[str] = set()
    for model in load_ligue1_models().values():
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            required.update(str(column) for column in feature_names)
    return sorted(required)


def load_ligue1_test_data(
    features_path: Path,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, str, str | None, str | None, str | None]:
    """Load Ligue 1 rows and return the newest temporal test split."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    if "league" not in data.columns:
        raise ValueError("No league column found in the feature dataset.")

    target_column = select_target_column(data)
    date_column = first_existing_column(data, DATE_CANDIDATES)
    home_team_column = first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = first_existing_column(data, AWAY_TEAM_CANDIDATES)

    data = data[data["league"].astype(str) == LEAGUE_NAME].copy()
    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(PRINT_LABELS)].reset_index(drop=True)

    model_feature_columns = required_model_features()
    missing_model_features = [column for column in model_feature_columns if column not in data.columns]
    if missing_model_features:
        raise ValueError(f"Missing features required by Ligue 1 specialist models: {', '.join(missing_model_features)}")
    if model_feature_columns:
        before_rows = len(data)
        data = data.dropna(subset=model_feature_columns).reset_index(drop=True)
        dropped_rows = before_rows - len(data)
        if dropped_rows:
            print(f"Dropped {dropped_rows} Ligue 1 rows with missing model features.")

    if len(data) < 10:
        raise ValueError("At least 10 valid Ligue 1 matches are required for the specialist backtest.")

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


def prediction_row(
    row: pd.Series,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
) -> dict[str, Any]:
    """Run the Ligue 1 probability engine for one match row and format the export row."""
    result = predict_match_probabilities_ligue1(build_match_features(row))
    match_probabilities = result["probabilities"]["match"]
    analysis = result["analysis"]

    actual_result = row[target_column]
    predicted_class_argmax = analysis["predicted_class_argmax"]
    predicted_class_adjusted = analysis["predicted_class_adjusted"]
    recommended_class = analysis["recommended_prediction_class"]

    return {
        "date": row[date_column] if date_column else pd.NA,
        "league": LEAGUE_NAME,
        "home_team": row[home_team_column] if home_team_column else pd.NA,
        "away_team": row[away_team_column] if away_team_column else pd.NA,
        "actual_result": actual_result,
        "home_win_probability": match_probabilities["home_win"],
        "draw_probability": match_probabilities["draw"],
        "away_win_probability": match_probabilities["away_win"],
        "home_no_loss_probability": result["probabilities"]["home"]["no_loss"],
        "away_no_loss_probability": result["probabilities"]["away"]["no_loss"],
        "predicted_class_argmax": predicted_class_argmax,
        "predicted_class_adjusted": predicted_class_adjusted,
        "recommended_prediction_class": recommended_class,
        "is_correct_argmax": predicted_class_argmax == actual_result,
        "is_correct_adjusted": predicted_class_adjusted == actual_result,
        "is_correct_recommended": recommended_class == actual_result,
        "confidence_score": analysis["confidence_score"],
        "match_profile": analysis["match_profile"],
        "model_version": analysis.get("model_version"),
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


def print_distribution(title: str, values: pd.Series) -> None:
    """Print a H/D/A distribution."""
    print(title)
    print(values.value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())


def print_metrics(report: pd.DataFrame) -> None:
    """Print global Ligue 1 specialist backtest metrics and distributions."""
    probabilities_for_log_loss = report[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    probabilities_for_log_loss = probabilities_for_log_loss.div(probabilities_for_log_loss.sum(axis=1), axis=0)

    d_recommended = report[report["recommended_prediction_class"] == "D"]

    print("\nProbability engine Ligue 1 specialist backtest")
    print(f"Matches: {len(report)}")
    print(f"Accuracy argmax: {accuracy_score(report['actual_result'], report['predicted_class_argmax']):.4f}")
    print(f"Accuracy adjusted: {accuracy_score(report['actual_result'], report['predicted_class_adjusted']):.4f}")
    print(
        f"Accuracy recommended: "
        f"{accuracy_score(report['actual_result'], report['recommended_prediction_class']):.4f}"
    )
    print(f"Log loss: {log_loss(report['actual_result'], probabilities_for_log_loss, labels=RESULT_LABELS):.4f}")
    print(f"Multiclass Brier score: {multiclass_brier_score(report):.4f}")

    print()
    print_distribution("Actual class distribution:", report["actual_result"])
    print()
    print_distribution("Predicted class distribution argmax:", report["predicted_class_argmax"])
    print()
    print_distribution("Predicted class distribution adjusted:", report["predicted_class_adjusted"])
    print()
    print_distribution("Recommended prediction class distribution:", report["recommended_prediction_class"])

    print(f"\nD recommended: {len(d_recommended)}")
    if d_recommended.empty:
        print("Actual draw rate among D recommended: not_available")
    else:
        print(f"Actual draw rate among D recommended: {(d_recommended['actual_result'] == 'D').mean():.4f}")


def run_probability_engine_ligue1_backtest(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Apply the Ligue 1 specialist engine to the newest 20 percent of Ligue 1 matches."""
    test_data, target_column, date_column, home_team_column, away_team_column = load_ligue1_test_data(features_path)
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
    report = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print_metrics(report)
    print(f"\nSaved Ligue 1 specialist backtest to: {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Backtest the Ligue 1 specialist probability engine.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the Ligue 1 specialist probability engine backtest."""
    args = parse_args()
    run_probability_engine_ligue1_backtest(features_path=args.features, output_path=args.output)


if __name__ == "__main__":
    main()
