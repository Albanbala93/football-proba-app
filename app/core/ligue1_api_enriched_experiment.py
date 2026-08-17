"""Experiment with API-Football Ligue 1 prematch features.

This file trains temporary experiment models only. It does not modify or save
the main probability engine models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_prematch_features.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_enriched_experiment_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_enriched_experiment_predictions.csv"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]

BASELINE_FEATURES = [
    "home_injuries_count",
    "away_injuries_count",
    "injuries_diff",
    "home_recent_formations_count",
    "away_recent_formations_count",
]

API_ENRICHED_FEATURES = [
    "home_formation",
    "away_formation",
    "formation_matchup",
    "home_formation_stability_5",
    "away_formation_stability_5",
    "formation_stability_diff_5",
    "home_injuries_count",
    "away_injuries_count",
    "injuries_diff",
    "home_api_shots_on_goal_for_5",
    "home_api_shots_on_goal_against_5",
    "away_api_shots_on_goal_for_5",
    "away_api_shots_on_goal_against_5",
    "home_api_total_shots_for_5",
    "home_api_total_shots_against_5",
    "away_api_total_shots_for_5",
    "away_api_total_shots_against_5",
    "home_api_expected_goals_for_5",
    "home_api_expected_goals_against_5",
    "away_api_expected_goals_for_5",
    "away_api_expected_goals_against_5",
    "api_expected_goals_diff_5",
    "api_shots_on_goal_diff_5",
    "api_total_shots_diff_5",
    "api_corners_diff_5",
    "api_possession_diff_5",
    "home_api_possession_5",
    "away_api_possession_5",
    "home_api_corners_for_5",
    "away_api_corners_for_5",
    "home_api_yellow_cards_5",
    "away_api_yellow_cards_5",
    "home_api_red_cards_5",
    "away_api_red_cards_5",
]
CATEGORICAL_FEATURES = {"home_formation", "away_formation", "formation_matchup"}


@dataclass(frozen=True)
class ExperimentSpec:
    """Definition of one experiment model."""

    name: str
    features: list[str]


EXPERIMENTS = [
    ExperimentSpec("baseline_api_simple", BASELINE_FEATURES),
    ExperimentSpec("api_enriched_model", API_ENRICHED_FEATURES),
]


def derive_result(row: pd.Series) -> str | None:
    """Derive H/D/A result from goals."""
    if pd.isna(row["goals_home"]) or pd.isna(row["goals_away"]):
        return None
    home_goals = float(row["goals_home"])
    away_goals = float(row["goals_away"])
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prematch features: {', '.join(missing)}")


def load_data(input_path: Path) -> pd.DataFrame:
    """Load scored Ligue 1 API prematch rows."""
    if not input_path.exists():
        raise FileNotFoundError(f"Prematch features file not found: {input_path}")

    data = pd.read_csv(input_path)
    require_columns(data, ["date", "goals_home", "goals_away", "fixture_id", "home_team_name", "away_team_name"])
    all_features = sorted(set(BASELINE_FEATURES + API_ENRICHED_FEATURES))
    require_columns(data, all_features)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["actual_result"] = data.apply(derive_result, axis=1)
    data = data.dropna(subset=["date", "actual_result"]).copy()
    data = data.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 scored matches are required for this experiment.")
    return data


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for training and newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_pipeline(data: pd.DataFrame, features: list[str]) -> Pipeline:
    """Build preprocessing + multinomial logistic regression pipeline."""
    categorical_features = [
        column for column in features if column in CATEGORICAL_FEATURES or data[column].dtype == object
    ]
    numeric_features = [column for column in features if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def multiclass_brier_score(y_true: pd.Series, probabilities: pd.DataFrame) -> float:
    """Compute mean multiclass Brier score for H/D/A probabilities."""
    total = 0.0
    for label in RESULT_LABELS:
        actual = (y_true == label).astype(float)
        total += ((probabilities[label] - actual) ** 2).mean()
    return float(total / len(RESULT_LABELS))


def distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A distribution columns."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def confusion_rows(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Return flattened confusion matrix columns."""
    matrix = confusion_matrix(y_true, y_pred, labels=RESULT_LABELS)
    rows = {}
    for i, actual in enumerate(RESULT_LABELS):
        for j, predicted in enumerate(RESULT_LABELS):
            rows[f"confusion_actual_{actual}_pred_{predicted}"] = int(matrix[i, j])
    rows["model"] = model_name
    return rows


def evaluate_experiment(
    spec: ExperimentSpec,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train and evaluate one experiment model."""
    model = build_pipeline(train_data, spec.features)
    model.fit(train_data[spec.features], train_data["actual_result"])

    predictions = pd.Series(model.predict(test_data[spec.features]), index=test_data.index)
    probability_array = model.predict_proba(test_data[spec.features])
    probability_columns = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(probability_array, columns=probability_columns, index=test_data.index)
    probabilities = probabilities.reindex(columns=RESULT_LABELS, fill_value=0.0)
    probabilities_for_log_loss = probabilities.reindex(columns=LOG_LOSS_LABELS)

    report_row: dict[str, Any] = {
        "model": spec.name,
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(spec.features)),
        "accuracy": float(accuracy_score(test_data["actual_result"], predictions)),
        "log_loss": float(log_loss(test_data["actual_result"], probabilities_for_log_loss, labels=LOG_LOSS_LABELS)),
        "brier_score_multiclass": multiclass_brier_score(test_data["actual_result"], probabilities),
    }
    report_row.update(distribution("actual_result", test_data["actual_result"]))
    report_row.update(distribution("predicted_result", predictions))
    report_row.update(confusion_rows(spec.name, test_data["actual_result"], predictions))

    prediction_rows = test_data[
        ["fixture_id", "date", "home_team_name", "away_team_name", "goals_home", "goals_away", "actual_result"]
    ].copy()
    prediction_rows["model"] = spec.name
    prediction_rows["predicted_result"] = predictions.to_numpy()
    prediction_rows["home_win_probability"] = probabilities["H"].to_numpy()
    prediction_rows["draw_probability"] = probabilities["D"].to_numpy()
    prediction_rows["away_win_probability"] = probabilities["A"].to_numpy()
    return report_row, prediction_rows


def run_ligue1_api_enriched_experiment(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the isolated API-Football enriched experiment."""
    data = load_data(input_path)
    train_data, test_data = temporal_train_test_split(data)

    report_rows = []
    prediction_frames = []
    for spec in EXPERIMENTS:
        report_row, predictions = evaluate_experiment(spec, train_data, test_data)
        report_rows.append(report_row)
        prediction_frames.append(predictions)

    report = pd.DataFrame(report_rows)
    prediction_export = pd.concat(prediction_frames, ignore_index=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    prediction_export.to_csv(predictions_path, index=False)

    print_summary(report, input_path, report_path, predictions_path)
    return report, prediction_export


def print_summary(report: pd.DataFrame, input_path: Path, report_path: Path, predictions_path: Path) -> None:
    """Print experiment metrics."""
    print("Ligue 1 API-Football enriched experiment")
    print(f"Input: {input_path}")
    print("\nMetrics:")
    display_columns = [
        "model",
        "train_rows",
        "test_rows",
        "feature_count",
        "accuracy",
        "log_loss",
        "brier_score_multiclass",
        "actual_result_H",
        "actual_result_D",
        "actual_result_A",
        "predicted_result_H",
        "predicted_result_D",
        "predicted_result_A",
    ]
    print(report[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nConfusion matrix columns are exported in the report CSV.")
    print(f"Saved report to: {report_path}")
    print(f"Saved predictions to: {predictions_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Ligue 1 API-Football enriched model experiment.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    run_ligue1_api_enriched_experiment(
        input_path=args.input,
        report_path=args.report,
        predictions_path=args.predictions,
    )


if __name__ == "__main__":
    main()
