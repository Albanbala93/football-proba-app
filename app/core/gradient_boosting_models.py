"""Train non-linear gradient boosting models for football outcome probabilities.

This script mirrors the temporal evaluation approach used by the logistic
regression outcome models, but swaps the estimator for
HistGradientBoostingClassifier to test whether the advanced and derived
features are better exploited by a non-linear model.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.outcome_models import FEATURE_CANDIDATES as BASE_FEATURE_CANDIDATES


DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "gradient_boosting_models"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "gradient_boosting_models_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "gradient_boosting_models_predictions.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
LEAGUE_CANDIDATES = ["league", "League"]
SEASON_CANDIDATES = ["season", "Season"]

DERIVED_FEATURE_CANDIDATES = [
    "home_shots_diff_5",
    "away_shots_diff_5",
    "shots_diff_between_teams",
    "home_shots_on_target_diff_5",
    "away_shots_on_target_diff_5",
    "shots_on_target_diff_between_teams",
    "home_attack_pressure_5",
    "away_attack_pressure_5",
    "home_defensive_pressure_5",
    "away_defensive_pressure_5",
    "attack_pressure_diff",
    "defensive_pressure_diff",
    "home_attack_vs_away_defense_5",
    "away_attack_vs_home_defense_5",
    "attack_matchup_diff_5",
    "home_corner_diff_5",
    "away_corner_diff_5",
    "corner_diff_between_teams",
    "home_discipline_risk_5",
    "away_discipline_risk_5",
    "discipline_risk_diff",
]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


FEATURE_CANDIDATES = _unique_preserve_order([*BASE_FEATURE_CANDIDATES, *DERIVED_FEATURE_CANDIDATES])


@dataclass(frozen=True)
class TargetSpec:
    """One binary target to train."""

    model_name: str
    positive_result: str


TARGET_SPECS = [
    TargetSpec("home_win_gb_model", "H"),
    TargetSpec("away_win_gb_model", "A"),
    TargetSpec("draw_gb_model", "D"),
]


CONFIGS: dict[str, dict[str, float | int | bool]] = {
    "default_simple": {
        "max_iter": 100,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
    },
    "regularized": {
        "max_iter": 200,
        "learning_rate": 0.03,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
    },
}


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available column from a candidate list."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def select_target_column(df: pd.DataFrame) -> str:
    """Detect the match result column."""
    target_column = first_existing_column(df, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError("No target column found. Expected one of: result, FTR, target")
    return target_column


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select only the feature columns that are present in the dataset."""
    feature_columns = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not feature_columns:
        raise ValueError("No usable feature columns found in the feature dataset.")
    return feature_columns


def load_data(features_path: Path) -> tuple[pd.DataFrame, list[str], str, str | None, str | None, str | None, str | None, str | None]:
    """Load and temporally order the dataset."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    target_column = select_target_column(data)
    feature_columns = select_feature_columns(data)
    date_column = first_existing_column(data, DATE_CANDIDATES)
    home_team_column = first_existing_column(data, HOME_TEAM_CANDIDATES)
    away_team_column = first_existing_column(data, AWAY_TEAM_CANDIDATES)
    league_column = first_existing_column(data, LEAGUE_CANDIDATES)
    season_column = first_existing_column(data, SEASON_CANDIDATES)

    if date_column:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.sort_values(date_column).reset_index(drop=True)

    required_columns = [target_column]
    if date_column:
        required_columns.append(date_column)

    before_rows = len(data)
    data = data.dropna(subset=required_columns).copy()
    data[target_column] = data[target_column].astype(str).str.upper().str.strip()
    data = data[data[target_column].isin(["H", "D", "A"])].reset_index(drop=True)

    dropped_rows = before_rows - len(data)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing target/date values or invalid targets.")
    if len(data) < 10:
        raise ValueError("At least 10 valid matches are required.")

    return (
        data,
        feature_columns,
        target_column,
        date_column,
        home_team_column,
        away_team_column,
        league_column,
        season_column,
    )


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for training and newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def prepare_feature_matrix(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Convert feature columns to numeric and keep NaNs for the gradient booster."""
    features = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_model(config_name: str) -> HistGradientBoostingClassifier:
    """Create one HistGradientBoostingClassifier configuration."""
    config = dict(CONFIGS[config_name])
    config["early_stopping"] = False
    config["random_state"] = 42
    return HistGradientBoostingClassifier(**config)


def safe_roc_auc(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute ROC AUC only when both classes are present in the test target."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_probability))


def safe_log_loss(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute binary log loss when the target is valid for evaluation."""
    try:
        return float(log_loss(y_true, y_probability, labels=[0, 1]))
    except ValueError:
        return None


def build_prediction_frame(
    test_data: pd.DataFrame,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
    league_column: str | None,
    season_column: str | None,
) -> pd.DataFrame:
    """Create the base prediction export dataframe."""
    predictions = pd.DataFrame(index=test_data.index)
    if date_column:
        predictions["date"] = test_data[date_column].to_numpy()
    if home_team_column:
        predictions["home_team"] = test_data[home_team_column].to_numpy()
    if away_team_column:
        predictions["away_team"] = test_data[away_team_column].to_numpy()
    if league_column:
        predictions["league"] = test_data[league_column].to_numpy()
    if season_column:
        predictions["season"] = test_data[season_column].to_numpy()
    predictions["actual_result"] = test_data[target_column].to_numpy()
    return predictions


def metric_row(
    model_name: str,
    config_name: str,
    config_params: dict[str, float | int | bool],
    feature_count: int,
    train_rows: int,
    test_rows: int,
    y_true: pd.Series,
    y_probability: pd.Series,
) -> dict[str, float | str | int | bool | None]:
    """Build aggregate metrics for one trained model configuration."""
    y_pred = (y_probability >= 0.5).astype(int)
    event_rate_test = float(y_true.mean())
    avg_predicted_probability = float(y_probability.mean())
    row = {
        "model_name": model_name,
        "config_name": config_name,
        "feature_count": feature_count,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "max_iter": config_params.get("max_iter"),
        "learning_rate": config_params.get("learning_rate"),
        "max_leaf_nodes": config_params.get("max_leaf_nodes"),
        "l2_regularization": config_params.get("l2_regularization"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_probability),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "log_loss": safe_log_loss(y_true, y_probability),
        "event_rate_test": event_rate_test,
        "avg_predicted_probability": avg_predicted_probability,
        "calibration_gap": avg_predicted_probability - event_rate_test,
    }
    return row


def config_sort_key(row: dict[str, float | str | int | bool | None]) -> tuple[float, float, float]:
    """Sort configs by the requested selection priority."""
    brier_score = row.get("brier_score")
    log_loss_value = row.get("log_loss")
    roc_auc_value = row.get("roc_auc")
    return (
        float(brier_score) if brier_score is not None and not pd.isna(brier_score) else math.inf,
        float(log_loss_value) if log_loss_value is not None and not pd.isna(log_loss_value) else math.inf,
        -float(roc_auc_value) if roc_auc_value is not None and not pd.isna(roc_auc_value) else math.inf,
    )


def print_metric_summary(model_name: str, config_name: str, metrics: dict[str, float | str | int | bool | None]) -> None:
    """Print a concise human-readable metric block."""
    print(f"\n{model_name} / {config_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    roc_auc = metrics["roc_auc"]
    print(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: not available")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    log_loss_value = metrics["log_loss"]
    print(f"Log loss: {log_loss_value:.4f}" if log_loss_value is not None else "Log loss: not available")
    print(f"Event rate in test: {metrics['event_rate_test']:.4f}")
    print(f"Average predicted probability: {metrics['avg_predicted_probability']:.4f}")
    print(f"Calibration gap: {metrics['calibration_gap']:.4f}")


def save_model_artifact(
    model: HistGradientBoostingClassifier,
    feature_columns: list[str],
    model_path: Path,
    *,
    model_name: str,
    config_name: str,
    config_params: dict[str, float | int | bool],
    metrics: dict[str, float | str | int | bool | None],
) -> None:
    """Persist a trained model alongside its metadata."""
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "model_name": model_name,
        "config_name": config_name,
        "config_params": config_params,
        "metrics": metrics,
    }
    joblib.dump(artifact, model_path)


def train_evaluate_gradient_boosting_models(
    features_path: Path = DEFAULT_FEATURES_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, evaluate, save, and report all gradient boosting models."""
    started_at = time.time()
    (
        data,
        feature_columns,
        target_column,
        date_column,
        home_team_column,
        away_team_column,
        league_column,
        season_column,
    ) = load_data(features_path)
    train_data, test_data = temporal_train_test_split(data, train_ratio)

    X_train = prepare_feature_matrix(train_data, feature_columns)
    X_test = prepare_feature_matrix(test_data, feature_columns)

    print(f"Feature file: {features_path}")
    print(f"Target column: {target_column}")
    print(f"Date column: {date_column or 'not found - using current file order'}")
    print(f"Feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"Temporal split: {len(train_data)} train rows, {len(test_data)} test rows")

    models_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = build_prediction_frame(
        test_data=test_data,
        target_column=target_column,
        date_column=date_column,
        home_team_column=home_team_column,
        away_team_column=away_team_column,
        league_column=league_column,
        season_column=season_column,
    )

    report_rows: list[dict[str, float | str | int | bool | None]] = []

    for spec in TARGET_SPECS:
        y_train = train_data[target_column].astype(str).str.upper().str.strip().eq(spec.positive_result).astype(int)
        y_test = test_data[target_column].astype(str).str.upper().str.strip().eq(spec.positive_result).astype(int)

        if y_train.nunique() < 2:
            raise ValueError(
                f"Training target for {spec.model_name} has only one class after temporal split. "
                "Check the dataset or the split ratio."
            )

        predictions[f"{spec.model_name}_actual"] = y_test.to_numpy()

        target_rows: list[dict[str, float | str | int | bool | None]] = []
        target_models: dict[str, HistGradientBoostingClassifier] = {}

        for config_name, config_params in CONFIGS.items():
            model = build_model(config_name)
            model.fit(X_train, y_train)
            y_probability = pd.Series(model.predict_proba(X_test)[:, 1], index=test_data.index)
            y_prediction = (y_probability >= 0.5).astype(int)

            config_prefix = f"{spec.model_name}_{config_name}"
            predictions[f"{config_prefix}_probability"] = y_probability.to_numpy()
            predictions[f"{config_prefix}_prediction"] = y_prediction.to_numpy()

            metrics = metric_row(
                model_name=spec.model_name,
                config_name=config_name,
                config_params=config_params,
                feature_count=len(feature_columns),
                train_rows=len(train_data),
                test_rows=len(test_data),
                y_true=y_test,
                y_probability=y_probability,
            )
            metrics["positive_result"] = spec.positive_result
            metrics["is_best"] = False
            target_rows.append(metrics)
            target_models[config_name] = model

            save_model_artifact(
                model=model,
                feature_columns=feature_columns,
                model_path=models_dir / f"{spec.model_name}_{config_name}.pkl",
                model_name=spec.model_name,
                config_name=config_name,
                config_params=config_params,
                metrics=metrics,
            )
            print_metric_summary(spec.model_name, config_name, metrics)
            print(f"Saved model to: {models_dir / f'{spec.model_name}_{config_name}.pkl'}")

        best_metrics = min(target_rows, key=config_sort_key)
        best_config_name = str(best_metrics["config_name"])
        best_model = target_models[best_config_name]
        best_model_path = models_dir / f"{spec.model_name}.pkl"
        best_metrics["best_config"] = best_config_name
        best_metrics["best_model_path"] = best_model_path.as_posix()
        best_metrics["is_best"] = True

        for row in target_rows:
            row["best_config"] = best_config_name
            row["is_best"] = row["config_name"] == best_config_name
            row["best_model_path"] = best_model_path.as_posix()
            row["selected_model_path"] = (models_dir / f"{spec.model_name}_{row['config_name']}.pkl").as_posix()

        save_model_artifact(
            model=best_model,
            feature_columns=feature_columns,
            model_path=best_model_path,
            model_name=spec.model_name,
            config_name=best_config_name,
            config_params=dict(CONFIGS[best_config_name]),
            metrics=best_metrics,
        )
        best_log_loss = best_metrics["log_loss"]
        if best_log_loss is not None:
            best_summary = (
                f"Best model for {spec.model_name}: {best_config_name} "
                f"(brier={best_metrics['brier_score']:.4f}, log_loss={best_log_loss:.4f})"
            )
        else:
            best_summary = (
                f"Best model for {spec.model_name}: {best_config_name} "
                f"(brier={best_metrics['brier_score']:.4f}, log_loss=not available)"
            )
        print(best_summary)
        print(f"Saved best model to: {best_model_path}")

        report_rows.extend(target_rows)

        predictions[f"{spec.model_name}_best_config"] = best_config_name
        best_probability_column = f"{spec.model_name}_{best_config_name}_probability"
        best_prediction_column = f"{spec.model_name}_{best_config_name}_prediction"
        predictions[f"{spec.model_name}_best_probability"] = predictions[best_probability_column]
        predictions[f"{spec.model_name}_best_prediction"] = predictions[best_prediction_column]

    report = pd.DataFrame(report_rows)
    report.to_csv(report_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    elapsed = time.time() - started_at
    print(f"\nSaved gradient boosting report to: {report_path}")
    print(f"Saved gradient boosting predictions to: {predictions_path}")
    print(f"Total elapsed time: {elapsed:.1f}s")
    return report, predictions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train gradient boosting football outcome models.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    """Run the full gradient boosting experiment suite."""
    args = parse_args()
    train_evaluate_gradient_boosting_models(
        features_path=args.features,
        models_dir=args.models_dir,
        report_path=args.report,
        predictions_path=args.predictions,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
