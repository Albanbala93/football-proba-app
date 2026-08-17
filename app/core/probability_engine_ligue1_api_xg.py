"""Experimental Ligue 1 API-Football xG + shots probability engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_prematch_features.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "ligue1_api_xg"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "ligue1_api_xg_model.pkl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_model_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_model_predictions.csv"
MODEL_VERSION = "ligue1_api_xg_v1"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]

FEATURE_COLUMNS = [
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
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_training_data(input_path: Path) -> pd.DataFrame:
    """Load scored prematch features for model training."""
    if not input_path.exists():
        raise FileNotFoundError(f"Prematch feature file not found: {input_path}")

    data = pd.read_csv(input_path)
    require_columns(
        data,
        [
            "fixture_id",
            "date",
            "home_team_name",
            "away_team_name",
            "goals_home",
            "goals_away",
            *FEATURE_COLUMNS,
        ],
    )
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["actual_result"] = data.apply(derive_result, axis=1)
    data = data.dropna(subset=["date", "actual_result"]).copy()
    data = data.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 scored matches are required for Ligue 1 API xG training.")
    return data


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for training and newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_pipeline() -> Pipeline:
    """Build xG + shots LogisticRegression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
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


def _probabilities_dataframe(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Return probabilities with H/D/A columns."""
    probability_array = model.predict_proba(features[FEATURE_COLUMNS])
    probability_columns = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(probability_array, columns=probability_columns, index=features.index)
    return probabilities.reindex(columns=RESULT_LABELS, fill_value=0.0)


def _confidence_score(probabilities: dict[str, float]) -> float:
    """Compute a bounded confidence score from probability shape."""
    sorted_probabilities = sorted(probabilities.values(), reverse=True)
    favorite_probability = sorted_probabilities[0]
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - favorite_probability
    score = (favorite_probability * 70) + (top_two_margin * 80) - (uncertainty_score * 35)
    return round(max(0.0, min(100.0, score)), 2)


def train_ligue1_api_xg_model(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, save, and evaluate the experimental Ligue 1 API xG model."""
    data = load_training_data(input_path)
    train_data, test_data = temporal_train_test_split(data)

    model = build_pipeline()
    model.fit(train_data[FEATURE_COLUMNS], train_data["actual_result"])
    probabilities = _probabilities_dataframe(model, test_data)
    predictions = pd.Series(model.predict(test_data[FEATURE_COLUMNS]), index=test_data.index)
    probabilities_for_log_loss = probabilities.reindex(columns=LOG_LOSS_LABELS)

    report_row: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "accuracy": float(accuracy_score(test_data["actual_result"], predictions)),
        "log_loss": float(log_loss(test_data["actual_result"], probabilities_for_log_loss, labels=LOG_LOSS_LABELS)),
        "brier_score_multiclass": multiclass_brier_score(test_data["actual_result"], probabilities),
    }
    report_row.update(distribution("actual_result", test_data["actual_result"]))
    report_row.update(distribution("predicted_result", predictions))
    report = pd.DataFrame([report_row])

    prediction_export = test_data[
        ["fixture_id", "date", "home_team_name", "away_team_name", "goals_home", "goals_away", "actual_result"]
    ].copy()
    prediction_export["predicted_class_argmax"] = predictions.to_numpy()
    prediction_export["home_win_probability"] = probabilities["H"].to_numpy()
    prediction_export["draw_probability"] = probabilities["D"].to_numpy()
    prediction_export["away_win_probability"] = probabilities["A"].to_numpy()
    prediction_export["model_version"] = MODEL_VERSION

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report.to_csv(report_path, index=False)
    prediction_export.to_csv(predictions_path, index=False)

    print("Ligue 1 API xG experimental engine training")
    print(f"Input: {input_path}")
    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Accuracy: {report_row['accuracy']:.4f}")
    print(f"Log loss: {report_row['log_loss']:.4f}")
    print(f"Brier score multiclass: {report_row['brier_score_multiclass']:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved predictions to: {predictions_path}")
    return report, prediction_export


def _load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load the saved experimental model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing Ligue 1 API xG model: {model_path}. "
            "Run `python app/core/probability_engine_ligue1_api_xg.py` first."
        )
    return joblib.load(model_path)


def predict_ligue1_api_xg(match_features: dict[str, Any]) -> dict[str, Any]:
    """Predict H/D/A probabilities from xG + shots prematch features."""
    if not match_features:
        raise ValueError("match_features cannot be empty.")
    missing = [column for column in FEATURE_COLUMNS if column not in match_features]
    if missing:
        raise ValueError(f"Missing features for Ligue 1 API xG model: {', '.join(missing)}")

    model = _load_model()
    features = pd.DataFrame([match_features])
    probabilities = _probabilities_dataframe(model, features).iloc[0]
    probability_dict = {
        "H": float(probabilities["H"]),
        "D": float(probabilities["D"]),
        "A": float(probabilities["A"]),
    }
    predicted_class_argmax = max(probability_dict, key=probability_dict.get)
    return {
        "home_win_probability": round(probability_dict["H"], 6),
        "draw_probability": round(probability_dict["D"], 6),
        "away_win_probability": round(probability_dict["A"], 6),
        "predicted_class_argmax": predicted_class_argmax,
        "confidence_score": _confidence_score(probability_dict),
        "model_version": MODEL_VERSION,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train experimental Ligue 1 API xG probability engine.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    return parser.parse_args()


def main() -> None:
    """Train and evaluate the experimental engine."""
    args = parse_args()
    train_ligue1_api_xg_model(
        input_path=args.input,
        model_path=args.model,
        report_path=args.report,
        predictions_path=args.predictions,
    )


if __name__ == "__main__":
    main()
