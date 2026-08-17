"""Run a temporal backtest for the baseline football result model.

The backtest trains on the oldest matches and evaluates on the newest matches.
It reports full 1N2 probabilities plus team-centric win, draw, loss, and
no-loss probabilities for home and away teams.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "backtest_baseline_report.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_backtest_model.pkl"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
RESULT_LABELS = ["A", "D", "H"]
PRINT_LABELS = ["H", "D", "A"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CALIBRATION_BUCKETS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

FEATURE_CANDIDATES = [
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
    "home_odds",
    "draw_odds",
    "away_odds",
    "B365H",
    "B365D",
    "B365A",
    "home_shots_for_5",
    "home_shots_against_5",
    "home_shots_on_target_for_5",
    "home_shots_on_target_against_5",
    "home_corners_for_5",
    "home_corners_against_5",
    "home_fouls_for_5",
    "home_fouls_against_5",
    "home_yellow_cards_5",
    "home_red_cards_5",
    "away_shots_for_5",
    "away_shots_against_5",
    "away_shots_on_target_for_5",
    "away_shots_on_target_against_5",
    "away_corners_for_5",
    "away_corners_against_5",
    "away_fouls_for_5",
    "away_fouls_against_5",
    "away_yellow_cards_5",
    "away_red_cards_5",
]

PROBABILITY_ANALYSES = {
    "home_win_probability": "is_home_win_actual",
    "home_loss_probability": "is_home_loss_actual",
    "home_no_loss_probability": "is_home_no_loss_actual",
    "away_win_probability": "is_away_win_actual",
    "away_loss_probability": "is_away_loss_actual",
    "away_no_loss_probability": "is_away_no_loss_actual",
    "draw_probability": "is_draw_actual",
}


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column available in a dataframe."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def select_target_column(df: pd.DataFrame) -> str:
    """Detect the target result column."""
    target_column = _first_existing_column(df, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError("No target column found. Expected one of: result, FTR, target")
    return target_column


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select only available baseline feature columns."""
    feature_columns = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not feature_columns:
        raise ValueError("No usable feature columns found in the feature dataset.")
    return feature_columns


def load_backtest_data(features_path: Path) -> tuple[pd.DataFrame, list[str], str, str | None, str | None, str | None]:
    """Load, clean, and sort feature data for a temporal backtest."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    target_column = select_target_column(data)
    feature_columns = select_feature_columns(data)
    date_column = _first_existing_column(data, DATE_CANDIDATES)
    home_team_column = _first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = _first_existing_column(data, AWAY_TEAM_CANDIDATES)

    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values(date_column).reset_index(drop=True)

    required_columns = feature_columns + [target_column]
    if date_column:
        required_columns.append(date_column)

    before_rows = len(data)
    data = data.dropna(subset=required_columns).copy()
    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(PRINT_LABELS)].reset_index(drop=True)
    dropped_rows = before_rows - len(data)

    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing values or invalid targets.")
    if len(data) < 10:
        raise ValueError("At least 10 valid matches are required for this backtest.")
    if data[target_column].nunique() < 2:
        raise ValueError("At least 2 result classes are required to train LogisticRegression.")

    return data, feature_columns, target_column, date_column, home_team_column, away_team_column


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by row order: oldest rows train, newest rows test."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_model() -> Pipeline:
    """Create the baseline scaler plus logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def predict_1n2_probabilities(model: Pipeline, test_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return pred_H, pred_D, and pred_A columns in football 1N2 order."""
    raw_probabilities = model.predict_proba(test_data[feature_columns])
    class_labels = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(raw_probabilities, columns=[f"pred_{label}" for label in class_labels])

    # If a class is absent from training, keep a stable report schema.
    for label in PRINT_LABELS:
        column = f"pred_{label}"
        if column not in probabilities.columns:
            probabilities[column] = 0.0

    return probabilities[["pred_H", "pred_D", "pred_A"]]


def build_match_report(
    test_data: pd.DataFrame,
    probabilities: pd.DataFrame,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
) -> pd.DataFrame:
    """Build the per-match exported backtest report."""
    report = pd.DataFrame(index=test_data.index)

    if date_column:
        report["date"] = test_data[date_column].to_numpy()
    if home_team_column:
        report["home_team"] = test_data[home_team_column].to_numpy()
    if away_team_column:
        report["away_team"] = test_data[away_team_column].to_numpy()

    report["actual_result"] = test_data[target_column].to_numpy()
    report = pd.concat([report.reset_index(drop=True), probabilities.reset_index(drop=True)], axis=1)

    report["home_win_probability"] = report["pred_H"]
    report["home_draw_probability"] = report["pred_D"]
    report["home_loss_probability"] = report["pred_A"]
    report["home_no_loss_probability"] = report["pred_H"] + report["pred_D"]

    report["away_win_probability"] = report["pred_A"]
    report["away_draw_probability"] = report["pred_D"]
    report["away_loss_probability"] = report["pred_H"]
    report["away_no_loss_probability"] = report["pred_A"] + report["pred_D"]
    report["draw_probability"] = report["pred_D"]

    report["predicted_class"] = report[["pred_H", "pred_D", "pred_A"]].idxmax(axis=1).str.replace("pred_", "", regex=False)
    report["is_correct_1N2"] = report["predicted_class"] == report["actual_result"]

    report["is_home_win_actual"] = report["actual_result"] == "H"
    report["is_home_loss_actual"] = report["actual_result"] == "A"
    report["is_home_no_loss_actual"] = report["actual_result"].isin(["H", "D"])
    report["is_away_win_actual"] = report["actual_result"] == "A"
    report["is_away_loss_actual"] = report["actual_result"] == "H"
    report["is_away_no_loss_actual"] = report["actual_result"].isin(["A", "D"])
    report["is_draw_actual"] = report["actual_result"] == "D"

    return report


def print_global_metrics(report: pd.DataFrame, train_rows: int, test_rows: int) -> None:
    """Print global 1N2 model quality metrics."""
    probabilities_for_log_loss = report[["pred_A", "pred_D", "pred_H"]]

    print("\nGlobal metrics")
    print(f"Train matches: {train_rows}")
    print(f"Test matches: {test_rows}")
    print("\nActual class distribution:")
    print(report["actual_result"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print("\nPredicted class distribution:")
    print(report["predicted_class"].value_counts().reindex(PRINT_LABELS, fill_value=0).to_string())
    print(f"\nAccuracy 1N2: {accuracy_score(report['actual_result'], report['predicted_class']):.4f}")
    print(f"Log loss: {log_loss(report['actual_result'], probabilities_for_log_loss, labels=RESULT_LABELS):.4f}")

    print("\nConfusion matrix:")
    matrix = confusion_matrix(report["actual_result"], report["predicted_class"], labels=PRINT_LABELS)
    print(pd.DataFrame(matrix, index=["actual_H", "actual_D", "actual_A"], columns=["pred_H", "pred_D", "pred_A"]))

    print("\nClassification report:")
    print(classification_report(report["actual_result"], report["predicted_class"], labels=PRINT_LABELS, zero_division=0))


def threshold_analysis(report: pd.DataFrame) -> pd.DataFrame:
    """Calculate threshold performance for each probability indicator."""
    rows = []
    test_rows = len(report)

    for probability_column, actual_column in PROBABILITY_ANALYSES.items():
        for threshold in THRESHOLDS:
            selected = report[report[probability_column] >= threshold]
            selected_count = len(selected)
            avg_predicted = selected[probability_column].mean() if selected_count else 0.0
            observed = selected[actual_column].mean() if selected_count else 0.0
            rows.append(
                {
                    "indicator": probability_column,
                    "threshold": threshold,
                    "selected_matches": selected_count,
                    "coverage_pct": selected_count / test_rows * 100 if test_rows else 0.0,
                    "avg_predicted_probability": avg_predicted,
                    "observed_frequency": observed,
                    "calibration_gap": avg_predicted - observed,
                }
            )

    return pd.DataFrame(rows)


def calibration_analysis(report: pd.DataFrame) -> pd.DataFrame:
    """Calculate bucketed calibration for each probability indicator."""
    rows = []

    for probability_column, actual_column in PROBABILITY_ANALYSES.items():
        bucket_series = pd.cut(
            report[probability_column],
            bins=CALIBRATION_BUCKETS,
            right=False,
            include_lowest=True,
        )

        for bucket, bucket_data in report.groupby(bucket_series, observed=False):
            selected_count = len(bucket_data)
            avg_predicted = bucket_data[probability_column].mean() if selected_count else 0.0
            observed = bucket_data[actual_column].mean() if selected_count else 0.0
            rows.append(
                {
                    "indicator": probability_column,
                    "bucket": str(bucket),
                    "matches": selected_count,
                    "avg_predicted_probability": avg_predicted,
                    "observed_frequency": observed,
                    "calibration_gap": avg_predicted - observed,
                }
            )

    return pd.DataFrame(rows)


def print_analysis_table(title: str, table: pd.DataFrame) -> None:
    """Print a readable analysis table with fixed numeric formatting."""
    print(f"\n{title}")
    if table.empty:
        print("No rows to display.")
        return
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def run_backtest(
    features_path: Path = DEFAULT_FEATURES_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    """Run the full temporal backtest and export per-match predictions."""
    data, feature_columns, target_column, date_column, home_team_column, away_team_column = load_backtest_data(features_path)
    train_data, test_data = temporal_train_test_split(data, train_ratio=train_ratio)

    print(f"Feature file: {features_path}")
    print(f"Target column: {target_column}")
    print(f"Date column: {date_column or 'not found - using current file order'}")
    print(f"Feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")

    model = build_model()
    model.fit(train_data[feature_columns], train_data[target_column])

    probabilities = predict_1n2_probabilities(model, test_data, feature_columns)
    report = build_match_report(
        test_data=test_data,
        probabilities=probabilities,
        target_column=target_column,
        date_column=date_column,
        home_team_column=home_team_column,
        away_team_column=away_team_column,
    )

    print_global_metrics(report, train_rows=len(train_data), test_rows=len(test_data))
    print_analysis_table("Threshold analysis", threshold_analysis(report))
    print_analysis_table("Calibration analysis", calibration_analysis(report))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(f"\nSaved per-match backtest report to: {report_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved backtest model to: {model_path}")

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the temporal backtest."""
    parser = argparse.ArgumentParser(description="Run a temporal baseline model backtest.")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Input feature CSV path. Defaults to data/processed/matches_features.csv.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output CSV report path. Defaults to data/predictions/backtest_baseline_report.csv.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output backtest model path. Defaults to models/baseline_backtest_model.pkl.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Oldest data share used for training. Defaults to 0.8.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the backtest from the command line."""
    args = parse_args()
    run_backtest(
        features_path=args.features,
        report_path=args.report,
        model_path=args.model,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
