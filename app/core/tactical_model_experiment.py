"""Isolated experiment comparing simple and tactical football models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tactical_features.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "tactical_model_experiment_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "tactical_model_experiment_predictions.csv"

RESULT_LABELS = ["A", "D", "H"]
PRINT_LABELS = ["H", "D", "A"]
BASELINE_CATEGORICAL_FEATURES = ["home_team_name", "away_team_name"]
TACTICAL_CATEGORICAL_FEATURES = [
    "home_team_name",
    "away_team_name",
    "home_formation",
    "away_formation",
    "formation_matchup",
]
TACTICAL_NUMERIC_FEATURES = [
    "home_defenders_count",
    "home_midfielders_count",
    "home_forwards_count",
    "away_defenders_count",
    "away_midfielders_count",
    "away_forwards_count",
    "midfield_density_diff",
    "attacking_line_diff",
    "defensive_line_diff",
    "home_back_three",
    "home_back_four",
    "home_back_five",
    "away_back_three",
    "away_back_four",
    "away_back_five",
    "home_two_strikers",
    "away_two_strikers",
    "home_formation_stability_5",
    "away_formation_stability_5",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if "result" in missing:
        raise ValueError(
            "Missing required column 'result'. Relancez scripts/parse_sportmonks_tactical_data.py "
            "apres extraction des scores."
        )
    if missing:
        raise ValueError(f"Missing required columns in tactical features: {', '.join(missing)}")


def _multiclass_brier_score(y_true: pd.Series, probabilities: pd.DataFrame) -> float:
    """Return multiclass Brier score averaged over labels."""
    y_true = y_true.reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True)
    total = 0.0
    for label in RESULT_LABELS:
        actual = y_true.eq(label).astype(float)
        total += ((probabilities[label] - actual) ** 2).mean()
    return float(total / len(RESULT_LABELS))


def _build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    """Build preprocessing plus balanced logistic regression."""
    transformers = []
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def _temporal_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by chronological order."""
    sorted_df = df.sort_values(["date", "sportmonks_fixture_id"]).reset_index(drop=True)
    split_index = int(len(sorted_df) * train_ratio)
    split_index = max(1, min(split_index, len(sorted_df) - 1))
    return sorted_df.iloc[:split_index].copy(), sorted_df.iloc[split_index:].copy()


def _predict_probabilities(model: Pipeline, test_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return probabilities with stable H/D/A columns."""
    classifier = model.named_steps["classifier"]
    raw_probabilities = model.predict_proba(test_df[features])
    probabilities = pd.DataFrame(raw_probabilities, columns=list(classifier.classes_))
    for label in RESULT_LABELS:
        if label not in probabilities.columns:
            probabilities[label] = 0.0
    return probabilities[RESULT_LABELS]


def _evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train and evaluate one model."""
    features = categorical_features + numeric_features
    model = _build_pipeline(categorical_features, numeric_features)
    model.fit(train_df[features], train_df["result"])

    probabilities = _predict_probabilities(model, test_df, features)
    predictions = probabilities.idxmax(axis=1)

    accuracy = float(accuracy_score(test_df["result"], predictions))
    model_log_loss = float(log_loss(test_df["result"], probabilities, labels=RESULT_LABELS))
    brier_score = _multiclass_brier_score(test_df["result"], probabilities)
    prediction_distribution = predictions.value_counts().reindex(PRINT_LABELS, fill_value=0).to_dict()
    matrix = confusion_matrix(test_df["result"], predictions, labels=PRINT_LABELS)

    report_row = {
        "model_name": model_name,
        "train_matches": int(len(train_df)),
        "test_matches": int(len(test_df)),
        "accuracy": accuracy,
        "log_loss": model_log_loss,
        "brier_score_multiclass": brier_score,
        "predicted_H": int(prediction_distribution["H"]),
        "predicted_D": int(prediction_distribution["D"]),
        "predicted_A": int(prediction_distribution["A"]),
    }
    for actual_index, actual_label in enumerate(PRINT_LABELS):
        for pred_index, pred_label in enumerate(PRINT_LABELS):
            report_row[f"confusion_actual_{actual_label}_pred_{pred_label}"] = int(matrix[actual_index, pred_index])

    prediction_rows = test_df[
        [
            "sportmonks_fixture_id",
            "date",
            "league_name",
            "home_team_name",
            "away_team_name",
            "home_formation",
            "away_formation",
            "formation_matchup",
            "result",
        ]
    ].copy()
    prediction_rows.insert(0, "model_name", model_name)
    prediction_rows["predicted_result"] = predictions.to_numpy()
    prediction_rows["prob_A"] = probabilities["A"].to_numpy()
    prediction_rows["prob_D"] = probabilities["D"].to_numpy()
    prediction_rows["prob_H"] = probabilities["H"].to_numpy()
    prediction_rows["is_correct"] = prediction_rows["predicted_result"].eq(prediction_rows["result"])
    return report_row, prediction_rows


def run_tactical_model_experiment(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the isolated tactical model experiment."""
    if not input_path.exists():
        raise FileNotFoundError(f"Tactical features file not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(
        df,
        ["date", "sportmonks_fixture_id", "result", *TACTICAL_CATEGORICAL_FEATURES, *TACTICAL_NUMERIC_FEATURES],
    )
    df = df.dropna(subset=["date", "result"]).copy()
    df["result"] = df["result"].astype(str).str.upper().str.strip()
    df = df[df["result"].isin(PRINT_LABELS)].copy()
    if len(df) < 10:
        raise ValueError("At least 10 tactical matches with valid results are required.")
    if df["result"].nunique() < 2:
        raise ValueError("At least 2 result classes are required for the experiment.")

    train_df, test_df = _temporal_split(df)
    experiments = [
        ("baseline_simple", BASELINE_CATEGORICAL_FEATURES, []),
        ("tactical_model", TACTICAL_CATEGORICAL_FEATURES, TACTICAL_NUMERIC_FEATURES),
    ]

    report_rows = []
    prediction_tables = []
    for model_name, categorical_features, numeric_features in experiments:
        report_row, prediction_rows = _evaluate_model(
            model_name,
            train_df,
            test_df,
            categorical_features,
            numeric_features,
        )
        report_rows.append(report_row)
        prediction_tables.append(prediction_rows)

    report = pd.DataFrame(report_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("Tactical model experiment")
    print(f"Input: {input_path}")
    print(f"Train matches: {len(train_df)}")
    print(f"Test matches: {len(test_df)}")
    print("\nModel comparison:")
    print(
        report[
            [
                "model_name",
                "accuracy",
                "log_loss",
                "brier_score_multiclass",
                "predicted_H",
                "predicted_D",
                "predicted_A",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nConfusion matrices:")
    for _, row in report.iterrows():
        matrix = pd.DataFrame(
            [
                [row["confusion_actual_H_pred_H"], row["confusion_actual_H_pred_D"], row["confusion_actual_H_pred_A"]],
                [row["confusion_actual_D_pred_H"], row["confusion_actual_D_pred_D"], row["confusion_actual_D_pred_A"]],
                [row["confusion_actual_A_pred_H"], row["confusion_actual_A_pred_D"], row["confusion_actual_A_pred_A"]],
            ],
            index=["actual_H", "actual_D", "actual_A"],
            columns=["pred_H", "pred_D", "pred_A"],
        )
        print(f"\n{row['model_name']}")
        print(matrix)
    print(f"\nSaved report to: {report_path}")
    print(f"Saved predictions to: {predictions_path}")
    return report, predictions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare simple and tactical models on tactical features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the tactical model experiment from the command line."""
    args = parse_args()
    run_tactical_model_experiment(input_path=args.input, report_path=args.report, predictions_path=args.predictions)


if __name__ == "__main__":
    main()
