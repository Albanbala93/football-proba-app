"""Test whether match stakes features improve Ligue 1 API xG v2 with injury impact."""

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
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_match_stakes_features.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_v2_match_stakes_ablation_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_v2_match_stakes_ablation_predictions.csv"

RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]

BASE_CURRENT_FEATURES = [
    "home_api_expected_goals_for_5",
    "home_api_expected_goals_against_5",
    "away_api_expected_goals_for_5",
    "away_api_expected_goals_against_5",
    "api_expected_goals_diff_5",
    "home_api_shots_on_goal_for_5",
    "home_api_shots_on_goal_against_5",
    "away_api_shots_on_goal_for_5",
    "away_api_shots_on_goal_against_5",
    "home_api_total_shots_for_5",
    "home_api_total_shots_against_5",
    "away_api_total_shots_for_5",
    "away_api_total_shots_against_5",
    "api_shots_on_goal_diff_5",
    "api_total_shots_diff_5",
    "home_formation_stability_5",
    "away_formation_stability_5",
    "formation_stability_diff_5",
    "home_injury_impact_score",
    "away_injury_impact_score",
    "injury_impact_diff",
    "home_likely_starter_injuries_count",
    "away_likely_starter_injuries_count",
    "likely_starter_injuries_diff",
]

STAKES_FEATURES = [
    "home_title_pressure",
    "away_title_pressure",
    "home_europe_pressure",
    "away_europe_pressure",
    "home_relegation_pressure",
    "away_relegation_pressure",
    "home_total_motivation_score",
    "away_total_motivation_score",
    "motivation_diff",
    "home_midtable_flag",
    "away_midtable_flag",
]


@dataclass(frozen=True)
class ModelConfig:
    """Definition of one ablation configuration."""

    name: str
    features: list[str]


CONFIGS = [
    ModelConfig("base_current", BASE_CURRENT_FEATURES),
    ModelConfig("base_plus_stakes", BASE_CURRENT_FEATURES + STAKES_FEATURES),
]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def derive_result(row: pd.Series) -> str | None:
    """Derive the H/D/A result from final goals."""
    if pd.isna(row.get("goals_home")) or pd.isna(row.get("goals_away")):
        return None
    home_goals = float(row["goals_home"])
    away_goals = float(row["goals_away"])
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def load_data(input_path: Path) -> pd.DataFrame:
    """Load match stakes rows and keep scored Ligue 1 matches."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_csv(input_path)
    required = ["date", "fixture_id", "goals_home", "goals_away", *sorted(set(BASE_CURRENT_FEATURES + STAKES_FEATURES))]
    require_columns(data, required)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data["actual_result"] = data.apply(derive_result, axis=1)
    data = data.dropna(subset=["date", "actual_result"]).copy()
    data = data.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 scored matches are required for the ablation.")
    return data


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the oldest rows for training and the newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_pipeline(features: list[str]) -> Pipeline:
    """Build preprocessing plus balanced logistic regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def multiclass_brier_score(y_true: pd.Series, probabilities: pd.DataFrame) -> float:
    """Compute a multiclass Brier score over H/D/A."""
    total = 0.0
    for label in RESULT_LABELS:
        actual = (y_true == label).astype(float)
        total += ((probabilities[label] - actual) ** 2).mean()
    return float(total / len(RESULT_LABELS))


def distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A distribution columns."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def evaluate_config(config: ModelConfig, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train and evaluate one configuration."""
    model = build_pipeline(config.features)
    model.fit(train_data[config.features], train_data["actual_result"])

    predicted = pd.Series(model.predict(test_data[config.features]), index=test_data.index)
    probability_array = model.predict_proba(test_data[config.features])
    probability_columns = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(probability_array, columns=probability_columns, index=test_data.index)
    probabilities = probabilities.reindex(columns=RESULT_LABELS, fill_value=0.0)
    probabilities_for_log_loss = probabilities.reindex(columns=LOG_LOSS_LABELS)
    confidence = probabilities[RESULT_LABELS].max(axis=1)

    summary: dict[str, Any] = {
        "row_type": "summary",
        "config": config.name,
        "matches": int(len(test_data)),
        "feature_count": int(len(config.features)),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "accuracy": float(accuracy_score(test_data["actual_result"], predicted)),
        "log_loss": float(log_loss(test_data["actual_result"], probabilities_for_log_loss, labels=LOG_LOSS_LABELS)),
        "brier_score_multiclass": float(multiclass_brier_score(test_data["actual_result"], probabilities)),
        "predicted_result_H": int((predicted == "H").sum()),
        "predicted_result_D": int((predicted == "D").sum()),
        "predicted_result_A": int((predicted == "A").sum()),
    }
    summary.update(distribution("actual_result", test_data["actual_result"]))

    prediction_rows: list[dict[str, Any]] = []
    for idx, row in test_data.iterrows():
        prediction_rows.append(
            {
                "row_type": "prediction",
                "config": config.name,
                "date": row.get("date"),
                "fixture_id": row.get("fixture_id"),
                "home_team_name": row.get("home_team_name"),
                "away_team_name": row.get("away_team_name"),
                "actual_result": row.get("actual_result"),
                "predicted_result": predicted.loc[idx],
                "confidence": float(confidence.loc[idx]),
                "home_win_probability": float(probabilities.loc[idx, "H"]),
                "draw_probability": float(probabilities.loc[idx, "D"]),
                "away_win_probability": float(probabilities.loc[idx, "A"]),
                "is_correct": bool(predicted.loc[idx] == row.get("actual_result")),
            }
        )

    errors = test_data.assign(_predicted=predicted, _confidence=confidence)
    errors = errors[errors["_predicted"] != errors["actual_result"]].sort_values("_confidence", ascending=False).head(10)
    error_rows: list[dict[str, Any]] = []
    for _, row in errors.iterrows():
        error_rows.append(
            {
                "row_type": "confident_error",
                "config": config.name,
                "date": row.get("date"),
                "fixture_id": row.get("fixture_id"),
                "home_team_name": row.get("home_team_name"),
                "away_team_name": row.get("away_team_name"),
                "actual_result": row.get("actual_result"),
                "predicted_result": row.get("_predicted"),
                "confidence": float(row.get("_confidence")),
                "home_win_probability": float(probabilities.loc[row.name, "H"]),
                "draw_probability": float(probabilities.loc[row.name, "D"]),
                "away_win_probability": float(probabilities.loc[row.name, "A"]),
            }
        )

    detail_rows = pd.DataFrame(prediction_rows + error_rows)
    return summary, detail_rows


def run_ablation(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the ablation study and export report and predictions."""
    data = load_data(input_path)
    train_data, test_data = temporal_train_test_split(data)

    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []

    for config in CONFIGS:
        summary, details = evaluate_config(config, train_data, test_data)
        summary_rows.append(summary)
        if not details.empty:
            prediction_frames.append(details[details["row_type"] == "prediction"].copy())
            report_frames.append(details[details["row_type"].isin(["confident_error"])].copy())

    summary_df = pd.DataFrame(summary_rows)
    report_df = pd.concat([summary_df, *report_frames], ignore_index=True, sort=False) if report_frames else summary_df.copy()
    predictions_df = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print_summary(summary_df, report_path)
    return report_df, predictions_df


def print_table(title: str, summary: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> None:
    """Print a compact ranked table."""
    display_columns = [
        "config",
        "feature_count",
        "accuracy",
        "log_loss",
        "brier_score_multiclass",
        "predicted_result_H",
        "predicted_result_D",
        "predicted_result_A",
    ]
    table = summary.sort_values(sort_columns, ascending=ascending).head(10)
    print(f"\n{title}")
    print(table[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_summary(summary: pd.DataFrame, report_path: Path) -> None:
    """Print the requested summary and decision."""
    print("Ligue 1 API v2 match stakes ablation")
    print(f"Rows tested: {len(summary)}")
    print_table("Top 10 par log_loss", summary, ["log_loss", "brier_score_multiclass", "accuracy"], [True, True, False])
    print_table("Top 10 par brier_score_multiclass", summary, ["brier_score_multiclass", "log_loss", "accuracy"], [True, True, False])
    print_table("Top 10 par accuracy", summary, ["accuracy", "log_loss", "brier_score_multiclass"], [False, True, True])

    base_row = summary[summary["config"] == "base_current"].iloc[0]
    stakes_row = summary[summary["config"] == "base_plus_stakes"].iloc[0]
    delta_accuracy = float(stakes_row["accuracy"] - base_row["accuracy"])
    delta_log_loss = float(stakes_row["log_loss"] - base_row["log_loss"])
    delta_brier = float(stakes_row["brier_score_multiclass"] - base_row["brier_score_multiclass"])
    print("\nComparison base_current -> base_plus_stakes")
    print(f"Delta accuracy: {delta_accuracy:+.4f}")
    print(f"Delta log_loss: {delta_log_loss:+.4f}")
    print(f"Delta brier: {delta_brier:+.4f}")

    if delta_log_loss < 0 and delta_brier < 0:
        decision = "Integrate stakes features: both log_loss and Brier improved."
    else:
        decision = "Do not integrate stakes features yet: no simultaneous improvement on log_loss and Brier."
    print(f"\nDecision: {decision}")
    print(f"\nSaved ablation report to: {report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Ligue 1 API xG v2 match stakes ablation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the ablation report."""
    args = parse_args()
    run_ablation(input_path=args.input, report_path=args.report, predictions_path=args.predictions)


if __name__ == "__main__":
    main()
