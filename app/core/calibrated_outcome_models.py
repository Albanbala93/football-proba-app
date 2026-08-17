"""Train and evaluate calibrated binary outcome models.

This module compares calibrated probability models against the existing binary
outcome approach. It does not change the central probability engine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "calibrated_outcome_models"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "calibrated_outcome_models_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "calibrated_outcome_models_predictions.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CALIBRATION_METHODS = ["sigmoid", "isotonic"]

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


@dataclass(frozen=True)
class OutcomeSpec:
    """Definition of a binary outcome target."""

    name: str
    positive_results: tuple[str, ...]


OUTCOME_SPECS = [
    OutcomeSpec("home_win_model", ("H",)),
    OutcomeSpec("away_win_model", ("A",)),
    OutcomeSpec("draw_model", ("D",)),
    OutcomeSpec("home_loss_model", ("A",)),
    OutcomeSpec("away_loss_model", ("H",)),
    OutcomeSpec("home_no_loss_model", ("H", "D")),
    OutcomeSpec("away_no_loss_model", ("A", "D")),
]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from a candidate list."""
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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select available feature columns, including Elo features when present."""
    feature_columns = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not feature_columns:
        raise ValueError("No usable feature columns found in the feature dataset.")
    return feature_columns


def load_data(features_path: Path) -> tuple[pd.DataFrame, list[str], str, str | None, str | None, str | None]:
    """Load, clean, and sort feature rows for temporal modeling."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    target_column = select_target_column(data)
    feature_columns = select_feature_columns(data)
    date_column = first_existing_column(data, DATE_CANDIDATES)
    home_team_column = first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = first_existing_column(data, AWAY_TEAM_CANDIDATES)

    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values(date_column).reset_index(drop=True)

    required_columns = feature_columns + [target_column]
    if date_column:
        required_columns.append(date_column)

    before_rows = len(data)
    data = data.dropna(subset=required_columns).copy()
    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(["H", "D", "A"])].reset_index(drop=True)

    dropped_rows = before_rows - len(data)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing values or invalid targets.")
    if len(data) < 20:
        raise ValueError("At least 20 valid matches are required for calibrated modeling.")

    return data, feature_columns, target_column, date_column, home_team_column, away_team_column


def temporal_train_calibration_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically into train, calibration, and test segments."""
    train_end = int(len(data) * train_ratio)
    calibration_end = int(len(data) * (train_ratio + calibration_ratio))
    train_end = max(1, min(train_end, len(data) - 2))
    calibration_end = max(train_end + 1, min(calibration_end, len(data) - 1))
    return data.iloc[:train_end].copy(), data.iloc[train_end:calibration_end].copy(), data.iloc[calibration_end:].copy()


def build_base_model() -> Pipeline:
    """Create the uncalibrated baseline classifier pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_calibrated_model(
    train_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
    feature_columns: list[str],
    y_train: pd.Series,
    y_calibration: pd.Series,
    method: str,
) -> CalibratedClassifierCV:
    """Train a base model, freeze it, and calibrate on the calibration split."""
    base_model = build_base_model()
    base_model.fit(train_data[feature_columns], y_train)
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    calibrated_model.fit(calibration_data[feature_columns], y_calibration)
    return calibrated_model


def positive_probability(model: CalibratedClassifierCV, test_data: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Return positive class probabilities for a calibrated binary model."""
    classes = list(model.classes_)
    probabilities = model.predict_proba(test_data[feature_columns])
    if 1 not in classes:
        raise ValueError("Calibrated model cannot predict positive class 1.")
    return pd.Series(probabilities[:, classes.index(1)], index=test_data.index)


def safe_roc_auc(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute ROC AUC only when both classes exist in the test set."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_probability))


def safe_log_loss(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute binary log loss with stable labels."""
    try:
        return float(log_loss(y_true, y_probability, labels=[0, 1]))
    except ValueError:
        return None


def metric_row(model_name: str, method: str, y_true: pd.Series, y_probability: pd.Series) -> dict[str, float | str | int | None]:
    """Build one aggregate metrics row."""
    y_pred = (y_probability >= 0.5).astype(int)
    avg_probability = float(y_probability.mean())
    event_rate = float(y_true.mean())
    return {
        "model": model_name,
        "calibration_method": method,
        "analysis_type": "metrics",
        "threshold": 0.5,
        "selected_matches": int(y_pred.sum()),
        "coverage_pct": float(y_pred.mean() * 100),
        "avg_predicted_probability": avg_probability,
        "observed_frequency": event_rate,
        "calibration_gap": avg_probability - event_rate,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_probability),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "log_loss": safe_log_loss(y_true, y_probability),
        "event_rate_test": event_rate,
        "avg_predicted_probability_test": avg_probability,
    }


def threshold_rows(model_name: str, method: str, y_true: pd.Series, y_probability: pd.Series) -> list[dict[str, float | str | int]]:
    """Build threshold analysis rows."""
    rows = []
    test_rows = len(y_true)
    for threshold in THRESHOLDS:
        selected_mask = y_probability >= threshold
        selected_count = int(selected_mask.sum())
        selected_probability = y_probability[selected_mask]
        selected_actual = y_true[selected_mask]
        avg_predicted = float(selected_probability.mean()) if selected_count else 0.0
        observed = float(selected_actual.mean()) if selected_count else 0.0
        rows.append(
            {
                "model": model_name,
                "calibration_method": method,
                "analysis_type": "threshold",
                "threshold": threshold,
                "selected_matches": selected_count,
                "coverage_pct": selected_count / test_rows * 100 if test_rows else 0.0,
                "avg_predicted_probability": avg_predicted,
                "observed_frequency": observed,
                "calibration_gap": avg_predicted - observed,
            }
        )
    return rows


def print_model_summary(metrics: dict[str, float | str | int | None], thresholds: pd.DataFrame) -> None:
    """Print metrics and threshold rows for one calibrated model."""
    print(f"\n{metrics['model']} / {metrics['calibration_method']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    roc_auc = metrics["roc_auc"]
    print(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: not available")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"Event rate test: {metrics['event_rate_test']:.4f}")
    print(f"Average predicted probability: {metrics['avg_predicted_probability_test']:.4f}")
    print(f"Calibration gap: {metrics['calibration_gap']:.4f}")
    print("Threshold analysis:")
    print(
        thresholds[
            [
                "threshold",
                "selected_matches",
                "coverage_pct",
                "avg_predicted_probability",
                "observed_frequency",
                "calibration_gap",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def initialize_predictions(
    test_data: pd.DataFrame,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
) -> pd.DataFrame:
    """Create base prediction export dataframe."""
    predictions = pd.DataFrame()
    if date_column:
        predictions["date"] = test_data[date_column].to_numpy()
    if home_team_column:
        predictions["home_team"] = test_data[home_team_column].to_numpy()
    if away_team_column:
        predictions["away_team"] = test_data[away_team_column].to_numpy()
    predictions["actual_result"] = test_data[target_column].to_numpy()
    return predictions


def train_evaluate_calibrated_models(
    features_path: Path = DEFAULT_FEATURES_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, calibrate, evaluate, save, and report all configured models."""
    data, feature_columns, target_column, date_column, home_team_column, away_team_column = load_data(features_path)
    train_data, calibration_data, test_data = temporal_train_calibration_test_split(data)

    print(f"Feature file: {features_path}")
    print(f"Target column: {target_column}")
    print(f"Date column: {date_column or 'not found - using current file order'}")
    print(f"Feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(
        "Temporal split: "
        f"{len(train_data)} train rows, {len(calibration_data)} calibration rows, {len(test_data)} test rows"
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, float | str | int | None]] = []
    predictions = initialize_predictions(test_data, target_column, date_column, home_team_column, away_team_column)

    for outcome in OUTCOME_SPECS:
        y_train = train_data[target_column].isin(outcome.positive_results).astype(int)
        y_calibration = calibration_data[target_column].isin(outcome.positive_results).astype(int)
        y_test = test_data[target_column].isin(outcome.positive_results).astype(int)

        if y_train.nunique() < 2 or y_calibration.nunique() < 2:
            print(f"\nSkipping {outcome.name}: train or calibration split has only one class.")
            continue

        predictions[f"{outcome.name}_actual"] = y_test.to_numpy()

        for method in CALIBRATION_METHODS:
            model = train_calibrated_model(train_data, calibration_data, feature_columns, y_train, y_calibration, method)
            y_probability = positive_probability(model, test_data, feature_columns)

            model_path = models_dir / f"{outcome.name}_{method}.pkl"
            joblib.dump(model, model_path)

            probability_column = f"{outcome.name}_{method}_probability"
            predictions[probability_column] = y_probability.to_numpy()
            predictions[f"{outcome.name}_{method}_prediction"] = (y_probability >= 0.5).astype(int).to_numpy()

            metrics = metric_row(outcome.name, method, y_test, y_probability)
            thresholds = threshold_rows(outcome.name, method, y_test, y_probability)
            report_rows.append(metrics)
            report_rows.extend(thresholds)
            print_model_summary(metrics, pd.DataFrame(thresholds))
            print(f"Saved calibrated model to: {model_path}")

    report = pd.DataFrame(report_rows)
    report.to_csv(report_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(f"\nSaved calibrated outcome models report to: {report_path}")
    print(f"Saved calibrated outcome predictions to: {predictions_path}")
    return report, predictions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train calibrated binary football outcome models.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    """Run calibrated outcome model training from the command line."""
    args = parse_args()
    train_evaluate_calibrated_models(
        features_path=args.features,
        models_dir=args.models_dir,
        report_path=args.report,
        predictions_path=args.predictions,
    )


if __name__ == "__main__":
    main()
