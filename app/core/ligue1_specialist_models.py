"""Train Ligue 1 specialist binary outcome models.

These models are trained only on Ligue 1 rows so they can be compared with the
current Big 5 models without changing the production probability engine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
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
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "ligue1_specialist_models"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_specialist_models_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_specialist_models_predictions.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
LEAGUE_NAME = "Ligue 1"

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
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
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

VARIANTS = {
    "classic": None,
    "balanced": "balanced",
}


@dataclass(frozen=True)
class OutcomeSpec:
    """Definition of one Ligue 1 specialist binary model."""

    name: str
    positive_results: tuple[str, ...]


OUTCOME_SPECS = [
    OutcomeSpec("home_win_ligue1_model", ("H",)),
    OutcomeSpec("draw_ligue1_model", ("D",)),
    OutcomeSpec("away_win_ligue1_model", ("A",)),
]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available column from a candidate list."""
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
    """Select available Ligue 1 specialist feature columns."""
    feature_columns = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not feature_columns:
        raise ValueError("No usable feature columns found in the feature dataset.")
    return feature_columns


def load_ligue1_data(features_path: Path) -> tuple[pd.DataFrame, list[str], str, str | None, str | None, str | None]:
    """Load, filter, clean, and sort Ligue 1 feature rows for temporal training."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    data = pd.read_csv(features_path)
    if "league" not in data.columns:
        raise ValueError("No league column found in the feature dataset.")

    data = data[data["league"].astype(str) == LEAGUE_NAME].copy()
    if data.empty:
        raise ValueError(f"No {LEAGUE_NAME} rows found in the feature dataset.")

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
        print(f"Dropped {dropped_rows} Ligue 1 rows with missing values or invalid targets.")
    if len(data) < 50:
        raise ValueError("At least 50 valid Ligue 1 matches are required.")

    return data, feature_columns, target_column, date_column, home_team_column, away_team_column


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for training and newest rows for testing."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_model(class_weight: str | None) -> Pipeline:
    """Create a standardized logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ]
    )


def safe_log_loss(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute binary log loss when both classes are valid for evaluation."""
    try:
        return float(log_loss(y_true, y_probability, labels=[0, 1]))
    except ValueError:
        return None


def safe_roc_auc(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute ROC AUC only when both classes are present in the test target."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_probability))


def metric_row(
    model_name: str,
    variant_name: str,
    y_true: pd.Series,
    y_probability: pd.Series,
    train_rows: int,
    test_rows: int,
    feature_count: int,
) -> dict[str, float | str | int | None]:
    """Build aggregate metric row for one trained specialist model."""
    y_pred = (y_probability >= 0.5).astype(int)
    event_rate_test = float(y_true.mean())
    avg_predicted_probability = float(y_probability.mean())
    return {
        "model": model_name,
        "variant": variant_name,
        "league": LEAGUE_NAME,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": feature_count,
        "positive_test_events": int(y_true.sum()),
        "predicted_positive_events": int(y_pred.sum()),
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


def initialize_predictions(
    test_data: pd.DataFrame,
    target_column: str,
    date_column: str | None,
    home_team_column: str | None,
    away_team_column: str | None,
) -> pd.DataFrame:
    """Create the base prediction export dataframe."""
    predictions = pd.DataFrame()
    if date_column:
        predictions["date"] = test_data[date_column].to_numpy()
    predictions["league"] = LEAGUE_NAME
    if home_team_column:
        predictions["home_team"] = test_data[home_team_column].to_numpy()
    if away_team_column:
        predictions["away_team"] = test_data[away_team_column].to_numpy()
    predictions["actual_result"] = test_data[target_column].to_numpy()
    return predictions


def print_model_summary(metrics: dict[str, float | str | int | None], model_file: Path) -> None:
    """Print readable model metrics."""
    print(f"\n{metrics['model']} / {metrics['variant']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    roc_auc = metrics["roc_auc"]
    print(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: not available")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    log_loss_value = metrics["log_loss"]
    print(f"Log loss: {log_loss_value:.4f}" if log_loss_value is not None else "Log loss: not available")
    print(f"Event rate test: {metrics['event_rate_test']:.4f}")
    print(f"Average predicted probability: {metrics['avg_predicted_probability']:.4f}")
    print(f"Calibration gap: {metrics['calibration_gap']:.4f}")
    print(f"Saved model to: {model_file}")


def train_evaluate_ligue1_specialist_models(
    features_path: Path = DEFAULT_FEATURES_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, evaluate, save, and report Ligue 1 specialist binary models."""
    data, feature_columns, target_column, date_column, home_team_column, away_team_column = load_ligue1_data(
        features_path
    )
    train_data, test_data = temporal_train_test_split(data, train_ratio)

    print("Ligue 1 specialist model training")
    print(f"Feature file: {features_path}")
    print(f"Target column: {target_column}")
    print(f"Date column: {date_column or 'not found - using current file order'}")
    print(f"Feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"Temporal split: {len(train_data)} train rows, {len(test_data)} test rows")

    models_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, float | str | int | None]] = []
    predictions = initialize_predictions(
        test_data=test_data,
        target_column=target_column,
        date_column=date_column,
        home_team_column=home_team_column,
        away_team_column=away_team_column,
    )

    for outcome in OUTCOME_SPECS:
        y_train = train_data[target_column].isin(outcome.positive_results).astype(int)
        y_test = test_data[target_column].isin(outcome.positive_results).astype(int)
        predictions[f"{outcome.name}_actual"] = y_test.to_numpy()

        if y_train.nunique() < 2:
            print(f"\nSkipping {outcome.name}: training target has only one class.")
            continue

        for variant_name, class_weight in VARIANTS.items():
            model = build_model(class_weight=class_weight)
            model.fit(train_data[feature_columns], y_train)
            y_probability = pd.Series(model.predict_proba(test_data[feature_columns])[:, 1], index=test_data.index)

            model_file = models_dir / f"{outcome.name}_{variant_name}.pkl"
            joblib.dump(model, model_file)

            probability_column = f"{outcome.name}_{variant_name}_probability"
            prediction_column = f"{outcome.name}_{variant_name}_prediction"
            predictions[probability_column] = y_probability.to_numpy()
            predictions[prediction_column] = (y_probability >= 0.5).astype(int).to_numpy()

            metrics = metric_row(
                model_name=outcome.name,
                variant_name=variant_name,
                y_true=y_test,
                y_probability=y_probability,
                train_rows=len(train_data),
                test_rows=len(test_data),
                feature_count=len(feature_columns),
            )
            report_rows.append(metrics)
            print_model_summary(metrics, model_file)

    report = pd.DataFrame(report_rows)
    report.to_csv(report_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"\nSaved Ligue 1 specialist models report to: {report_path}")
    print(f"Saved Ligue 1 specialist predictions to: {predictions_path}")
    return report, predictions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Ligue 1 specialist model training."""
    parser = argparse.ArgumentParser(description="Train Ligue 1 specialist binary outcome models.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    """Run all Ligue 1 specialist model experiments from the command line."""
    args = parse_args()
    train_evaluate_ligue1_specialist_models(
        features_path=args.features,
        models_dir=args.models_dir,
        report_path=args.report,
        predictions_path=args.predictions,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
