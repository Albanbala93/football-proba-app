"""Compare Ligue 1 API-Football formation feature families on a temporal split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_formation_performance_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_formation_ablation_report.csv"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]
CALIBRATION_BUCKETS = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]

BASE_XG_SHOTS_FEATURES = [
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
]
FORMATION_RAW_FEATURES = ["home_formation", "away_formation", "formation_matchup"]
FORMATION_STABILITY_FEATURES = [
    "home_formation_stability_5",
    "away_formation_stability_5",
    "home_recent_formations_count",
    "away_recent_formations_count",
    "formation_stability_diff_5",
]
FORMATION_PERFORMANCE_FEATURES = [
    "home_team_formation_matches_before",
    "home_team_formation_win_rate_before",
    "home_team_formation_draw_rate_before",
    "home_team_formation_loss_rate_before",
    "home_team_formation_xg_for_avg_before",
    "home_team_formation_xg_against_avg_before",
    "home_team_formation_goal_diff_avg_before",
    "away_team_formation_matches_before",
    "away_team_formation_win_rate_before",
    "away_team_formation_draw_rate_before",
    "away_team_formation_loss_rate_before",
    "away_team_formation_xg_for_avg_before",
    "away_team_formation_xg_against_avg_before",
    "away_team_formation_goal_diff_avg_before",
    "home_formation_signal_strength",
    "away_formation_signal_strength",
]
MATCHUP_PERFORMANCE_FEATURES = [
    "formation_matchup_matches_before",
    "formation_matchup_home_win_rate_before",
    "formation_matchup_draw_rate_before",
    "formation_matchup_away_win_rate_before",
    "formation_matchup_avg_total_goals_before",
    "formation_matchup_avg_xg_total_before",
    "matchup_signal_strength",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Definition of one feature family configuration."""

    name: str
    features: list[str]


FEATURE_CONFIGS = [
    FeatureConfig("xg_plus_shots", BASE_XG_SHOTS_FEATURES),
    FeatureConfig("xg_plus_shots_plus_formation_raw", BASE_XG_SHOTS_FEATURES + FORMATION_RAW_FEATURES),
    FeatureConfig(
        "xg_plus_shots_plus_formation_stability",
        BASE_XG_SHOTS_FEATURES + FORMATION_STABILITY_FEATURES,
    ),
    FeatureConfig(
        "xg_plus_shots_plus_formation_performance",
        BASE_XG_SHOTS_FEATURES + FORMATION_PERFORMANCE_FEATURES,
    ),
    FeatureConfig(
        "xg_plus_shots_plus_matchup_performance",
        BASE_XG_SHOTS_FEATURES + MATCHUP_PERFORMANCE_FEATURES,
    ),
    FeatureConfig(
        "xg_plus_shots_plus_all_formation_features",
        BASE_XG_SHOTS_FEATURES
        + FORMATION_RAW_FEATURES
        + FORMATION_STABILITY_FEATURES
        + FORMATION_PERFORMANCE_FEATURES
        + MATCHUP_PERFORMANCE_FEATURES,
    ),
]
CLASS_WEIGHT = "balanced"
CATEGORICAL_FEATURES = {
    "home_formation",
    "away_formation",
    "formation_matchup",
    "home_formation_signal_strength",
    "away_formation_signal_strength",
    "matchup_signal_strength",
}


def derive_result(row: pd.Series) -> str | None:
    """Derive the H/D/A result from final goals."""
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
    """Raise a clear error if columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in formation features: {', '.join(missing)}")


def load_data(input_path: Path) -> pd.DataFrame:
    """Load scored rows for the ablation."""
    if not input_path.exists():
        raise FileNotFoundError(f"Formation feature file not found: {input_path}")

    data = pd.read_csv(input_path)
    required = ["date", "fixture_id", "goals_home", "goals_away", *sorted({feature for config in FEATURE_CONFIGS for feature in config.features})]
    require_columns(data, required)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["actual_result"] = data.apply(derive_result, axis=1)
    data = data.dropna(subset=["date", "actual_result"]).copy()
    data = data.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 scored matches are required for formation ablation.")
    return data


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for training and newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_pipeline(data: pd.DataFrame, features: list[str]) -> Pipeline:
    """Build preprocessing plus balanced logistic regression."""
    categorical_features = [column for column in features if column in CATEGORICAL_FEATURES or data[column].dtype == object]
    numeric_features = [column for column in features if column not in categorical_features]
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
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT)),
        ]
    )


def multiclass_brier_score(y_true: pd.Series, probabilities: pd.DataFrame) -> float:
    """Compute the mean multiclass Brier score for H/D/A probabilities."""
    total = 0.0
    for label in RESULT_LABELS:
        actual = (y_true == label).astype(float)
        total += ((probabilities[label] - actual) ** 2).mean()
    return float(total / len(RESULT_LABELS))


def distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A distribution columns."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def confidence_from_probabilities(probabilities: pd.DataFrame) -> pd.Series:
    """Return the max class probability for each row."""
    return probabilities[RESULT_LABELS].max(axis=1)


def evaluate_config(
    config: FeatureConfig,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train and evaluate one feature configuration."""
    model = build_pipeline(train_data, config.features)
    model.fit(train_data[config.features], train_data["actual_result"])

    predicted = pd.Series(model.predict(test_data[config.features]), index=test_data.index)
    probability_array = model.predict_proba(test_data[config.features])
    probability_columns = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(probability_array, columns=probability_columns, index=test_data.index)
    probabilities = probabilities.reindex(columns=RESULT_LABELS, fill_value=0.0)
    probabilities_for_log_loss = probabilities.reindex(columns=LOG_LOSS_LABELS)
    confidence = confidence_from_probabilities(probabilities)

    summary: dict[str, Any] = {
        "row_type": "summary",
        "config": config.name,
        "matches": int(len(test_data)),
        "feature_count": int(len(config.features)),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "accuracy": float(accuracy_score(test_data["actual_result"], predicted)),
        "log_loss": float(log_loss(test_data["actual_result"], probabilities_for_log_loss, labels=LOG_LOSS_LABELS)),
        "brier_score_multiclass": multiclass_brier_score(test_data["actual_result"], probabilities),
        "predicted_result_H": int((predicted == "H").sum()),
        "predicted_result_D": int((predicted == "D").sum()),
        "predicted_result_A": int((predicted == "A").sum()),
    }
    summary.update(distribution("actual_result", test_data["actual_result"]))

    calibration_rows: list[dict[str, Any]] = []
    buckets = pd.cut(confidence, bins=CALIBRATION_BUCKETS, include_lowest=True, right=False)
    for bucket, bucket_df in test_data.assign(_confidence=confidence, _predicted=predicted).groupby(buckets, observed=False):
        if bucket_df.empty:
            continue
        bucket_confidence = bucket_df["_confidence"]
        bucket_accuracy = float((bucket_df["_predicted"] == bucket_df["actual_result"]).mean())
        calibration_rows.append(
            {
                "row_type": "calibration",
                "config": config.name,
                "bucket": str(bucket),
                "matches": int(len(bucket_df)),
                "avg_confidence": float(bucket_confidence.mean()),
                "observed_accuracy": bucket_accuracy,
                "calibration_gap": float(bucket_confidence.mean() - bucket_accuracy),
            }
        )

    errors = test_data.assign(_predicted=predicted, _confidence=confidence)
    errors = errors[errors["_predicted"] != errors["actual_result"]].sort_values("_confidence", ascending=False).head(10)
    confident_error_rows: list[dict[str, Any]] = []
    for _, row in errors.iterrows():
        confident_error_rows.append(
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

    detail_rows = pd.DataFrame(calibration_rows + confident_error_rows)
    return summary, detail_rows


def run_ligue1_api_xg_formation_ablation(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run the ablation study and export the report."""
    data = load_data(input_path)
    train_data, test_data = temporal_train_test_split(data)

    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for config in FEATURE_CONFIGS:
        summary, details = evaluate_config(config, train_data, test_data)
        summary_rows.append(summary)
        if not details.empty:
            detail_frames.append(details)

    report = pd.DataFrame(summary_rows)
    if detail_frames:
        report = pd.concat([report, *detail_frames], ignore_index=True, sort=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print_summary(pd.DataFrame(summary_rows), output_path)
    return report


def print_table(title: str, report: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> None:
    """Print a top-10 summary table."""
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
    table = report.sort_values(sort_columns, ascending=ascending).head(10)
    print(f"\n{title}")
    print(table[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Print requested rankings and the best global configuration."""
    print("Ligue 1 API xG formation ablation")
    print(f"Rows tested: {len(summary)}")
    print_table("Top 10 par log_loss", summary, ["log_loss", "brier_score_multiclass", "accuracy"], [True, True, False])
    print_table("Top 10 par brier_score_multiclass", summary, ["brier_score_multiclass", "log_loss", "accuracy"], [True, True, False])
    print_table("Top 10 par accuracy", summary, ["accuracy", "log_loss", "brier_score_multiclass"], [False, True, True])

    best_global = summary.sort_values(["log_loss", "brier_score_multiclass", "accuracy"], ascending=[True, True, False]).iloc[0]
    print(
        "\nBest global configuration: "
        f"{best_global['config']} "
        f"(accuracy={best_global['accuracy']:.4f}, log_loss={best_global['log_loss']:.4f}, "
        f"brier={best_global['brier_score_multiclass']:.4f})"
    )
    print(f"\nSaved ablation report to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Ligue 1 API xG formation ablation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the ablation report."""
    args = parse_args()
    run_ligue1_api_xg_formation_ablation(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
