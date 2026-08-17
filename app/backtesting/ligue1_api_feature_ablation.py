"""Run feature-family ablations for Ligue 1 API-Football prematch features."""

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
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_prematch_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_feature_ablation_report.csv"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]

BASELINE_SIMPLE_FEATURES = [
    "home_injuries_count",
    "away_injuries_count",
    "injuries_diff",
    "home_recent_formations_count",
    "away_recent_formations_count",
]
XG_FEATURES = [
    "home_api_expected_goals_for_5",
    "home_api_expected_goals_against_5",
    "away_api_expected_goals_for_5",
    "away_api_expected_goals_against_5",
    "api_expected_goals_diff_5",
]
SHOTS_FEATURES = [
    "home_api_shots_on_goal_for_5",
    "home_api_shots_on_goal_against_5",
    "away_api_shots_on_goal_for_5",
    "away_api_shots_on_goal_against_5",
    "home_api_total_shots_for_5",
    "away_api_total_shots_for_5",
    "api_shots_on_goal_diff_5",
    "api_total_shots_diff_5",
]
FORMATION_FEATURES = [
    "home_formation",
    "away_formation",
    "formation_matchup",
    "home_formation_stability_5",
    "away_formation_stability_5",
    "formation_stability_diff_5",
]
INJURY_FEATURES = [
    "home_injuries_count",
    "away_injuries_count",
    "injuries_diff",
]
ALL_FEATURES = [
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
class FeatureConfig:
    """Definition of one ablation feature set."""

    name: str
    features: list[str]


FEATURE_CONFIGS = [
    FeatureConfig("baseline_simple", BASELINE_SIMPLE_FEATURES),
    FeatureConfig("xg_only", XG_FEATURES),
    FeatureConfig("shots_only", SHOTS_FEATURES),
    FeatureConfig("formations_only", FORMATION_FEATURES),
    FeatureConfig("injuries_only", INJURY_FEATURES),
    FeatureConfig("xg_plus_shots", XG_FEATURES + SHOTS_FEATURES),
    FeatureConfig("xg_plus_formations", XG_FEATURES + FORMATION_FEATURES),
    FeatureConfig("all_features", ALL_FEATURES),
]
CLASS_WEIGHT_VARIANTS = {
    "classic": None,
    "balanced": "balanced",
}


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
    """Raise a clear error if columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prematch features: {', '.join(missing)}")


def load_data(input_path: Path) -> pd.DataFrame:
    """Load scored prematch rows for ablation."""
    if not input_path.exists():
        raise FileNotFoundError(f"Prematch features file not found: {input_path}")

    data = pd.read_csv(input_path)
    all_features = sorted({feature for config in FEATURE_CONFIGS for feature in config.features})
    require_columns(data, ["date", "fixture_id", "goals_home", "goals_away", *all_features])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["actual_result"] = data.apply(derive_result, axis=1)
    data = data.dropna(subset=["date", "actual_result"]).copy()
    data = data.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 scored matches are required for feature ablation.")
    return data


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for training and newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_pipeline(data: pd.DataFrame, features: list[str], class_weight: str | None) -> Pipeline:
    """Build preprocessing plus logistic regression pipeline."""
    categorical_features = [
        column for column in features if column in CATEGORICAL_FEATURES or data[column].dtype == object
    ]
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
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight)),
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


def evaluate_config(
    config: FeatureConfig,
    variant_name: str,
    class_weight: str | None,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Train and evaluate one ablation configuration."""
    model = build_pipeline(train_data, config.features, class_weight=class_weight)
    model.fit(train_data[config.features], train_data["actual_result"])
    predictions = pd.Series(model.predict(test_data[config.features]), index=test_data.index)

    probability_array = model.predict_proba(test_data[config.features])
    probability_columns = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(probability_array, columns=probability_columns, index=test_data.index)
    probabilities = probabilities.reindex(columns=RESULT_LABELS, fill_value=0.0)
    probabilities_for_log_loss = probabilities.reindex(columns=LOG_LOSS_LABELS)

    row: dict[str, Any] = {
        "config": config.name,
        "variant": variant_name,
        "class_weight": class_weight or "None",
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(config.features)),
        "accuracy": float(accuracy_score(test_data["actual_result"], predictions)),
        "log_loss": float(log_loss(test_data["actual_result"], probabilities_for_log_loss, labels=LOG_LOSS_LABELS)),
        "brier_score_multiclass": multiclass_brier_score(test_data["actual_result"], probabilities),
    }
    row.update(distribution("actual_result", test_data["actual_result"]))
    row.update(distribution("predicted_result", predictions))
    return row


def run_ligue1_api_feature_ablation(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run all feature ablations and export the report."""
    data = load_data(input_path)
    train_data, test_data = temporal_train_test_split(data)
    rows = []
    for config in FEATURE_CONFIGS:
        for variant_name, class_weight in CLASS_WEIGHT_VARIANTS.items():
            rows.append(evaluate_config(config, variant_name, class_weight, train_data, test_data))

    report = pd.DataFrame(rows).sort_values(["log_loss", "brier_score_multiclass"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print_summary(report, input_path, output_path)
    return report


def print_table(title: str, report: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> None:
    """Print one top-10 table."""
    display_columns = [
        "config",
        "variant",
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


def print_summary(report: pd.DataFrame, input_path: Path, output_path: Path) -> None:
    """Print requested top-10 summaries."""
    print("Ligue 1 API feature ablation")
    print(f"Input: {input_path}")
    print(f"Rows tested: {len(report)}")
    print_table("Top 10 par log_loss", report, ["log_loss", "brier_score_multiclass"], [True, True])
    print_table(
        "Top 10 par brier_score_multiclass",
        report,
        ["brier_score_multiclass", "log_loss"],
        [True, True],
    )
    print_table("Top 10 par accuracy", report, ["accuracy", "log_loss"], [False, True])
    print(f"\nSaved ablation report to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Ligue 1 API-Football feature ablations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the ablation report."""
    args = parse_args()
    run_ligue1_api_feature_ablation(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
