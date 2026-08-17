"""Train binary outcome models without bookmaker odds features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "outcome_models_no_odds"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "predictions" / "outcome_models_no_odds_report.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "outcome_models_no_odds_predictions.csv"

TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]
HOME_TEAM_CANDIDATES = ["HomeTeam", "home_team"]
AWAY_TEAM_CANDIDATES = ["AwayTeam", "away_team"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

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
    """Definition of one binary no-odds outcome model."""

    name: str
    positive_results: tuple[str, ...]


OUTCOME_SPECS = [
    OutcomeSpec("home_win_model_no_odds", ("H",)),
    OutcomeSpec("away_win_model_no_odds", ("A",)),
    OutcomeSpec("draw_model_no_odds", ("D",)),
]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from candidates."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def select_target_column(df: pd.DataFrame) -> str:
    """Detect result target column."""
    target_column = first_existing_column(df, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError("No target column found. Expected one of: result, FTR, target")
    return target_column


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select no-odds feature columns available in the dataset."""
    feature_columns = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not feature_columns:
        raise ValueError("No usable no-odds feature columns found.")
    return feature_columns


def load_data(features_path: Path) -> tuple[pd.DataFrame, list[str], str, str | None, str | None, str | None]:
    """Load, clean, and sort data for temporal training."""
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
    if len(data) < 10:
        raise ValueError("At least 10 valid matches are required.")
    return data, feature_columns, target_column, date_column, home_team_column, away_team_column


def temporal_train_test_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use oldest rows for train and newest rows for test."""
    split_index = int(len(data) * train_ratio)
    split_index = max(1, min(split_index, len(data) - 1))
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_model() -> Pipeline:
    """Create no-odds logistic model pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))])


def safe_roc_auc(y_true: pd.Series, y_probability: pd.Series) -> float | None:
    """Compute ROC AUC if possible."""
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_probability))


def metric_row(model_name: str, y_true: pd.Series, y_probability: pd.Series) -> dict[str, float | str | int | None]:
    """Build aggregate metrics row."""
    y_pred = (y_probability >= 0.5).astype(int)
    return {
        "model": model_name,
        "analysis_type": "metrics",
        "threshold": 0.5,
        "selected_matches": int(y_pred.sum()),
        "coverage_pct": float(y_pred.mean() * 100),
        "avg_predicted_probability": float(y_probability.mean()),
        "observed_frequency": float(y_true.mean()),
        "calibration_gap": float(y_probability.mean() - y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_probability),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "log_loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "actual_event_rate_test": float(y_true.mean()),
        "avg_predicted_probability_test": float(y_probability.mean()),
    }


def threshold_rows(model_name: str, y_true: pd.Series, y_probability: pd.Series) -> list[dict[str, float | str | int]]:
    """Build threshold analysis rows."""
    rows = []
    test_rows = len(y_true)
    for threshold in THRESHOLDS:
        selected = y_probability >= threshold
        selected_count = int(selected.sum())
        avg_predicted = float(y_probability[selected].mean()) if selected_count else 0.0
        observed = float(y_true[selected].mean()) if selected_count else 0.0
        rows.append(
            {
                "model": model_name,
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


def train_evaluate_no_odds_models(
    features_path: Path = DEFAULT_FEATURES_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train, evaluate, and save no-odds outcome models."""
    data, feature_columns, target_column, date_column, home_team_column, away_team_column = load_data(features_path)
    train_data, test_data = temporal_train_test_split(data, train_ratio)
    print(f"Feature file: {features_path}")
    print(f"Target column: {target_column}")
    print(f"Date column: {date_column or 'not found - using current file order'}")
    print(f"No-odds feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"Temporal split: {len(train_data)} train rows, {len(test_data)} test rows")

    models_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame()
    if date_column:
        predictions["date"] = test_data[date_column].to_numpy()
    if home_team_column:
        predictions["home_team"] = test_data[home_team_column].to_numpy()
    if away_team_column:
        predictions["away_team"] = test_data[away_team_column].to_numpy()
    predictions["actual_result"] = test_data[target_column].to_numpy()

    report_rows = []
    for outcome in OUTCOME_SPECS:
        y_train = train_data[target_column].isin(outcome.positive_results).astype(int)
        y_test = test_data[target_column].isin(outcome.positive_results).astype(int)
        if y_train.nunique() < 2:
            print(f"Skipping {outcome.name}: training target has only one class.")
            continue

        model = build_model()
        model.fit(train_data[feature_columns], y_train)
        y_probability = pd.Series(model.predict_proba(test_data[feature_columns])[:, 1], index=test_data.index)
        model_path = models_dir / f"{outcome.name}.pkl"
        joblib.dump(model, model_path)

        predictions[f"{outcome.name}_actual"] = y_test.to_numpy()
        predictions[f"{outcome.name}_probability"] = y_probability.to_numpy()
        predictions[f"{outcome.name}_prediction"] = (y_probability >= 0.5).astype(int).to_numpy()
        metrics = metric_row(outcome.name, y_test, y_probability)
        thresholds = threshold_rows(outcome.name, y_test, y_probability)
        report_rows.append(metrics)
        report_rows.extend(thresholds)

        print(f"\n{outcome.name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "ROC AUC: not available")
        print(f"Brier score: {metrics['brier_score']:.4f}")
        print(f"Log loss: {metrics['log_loss']:.4f}")
        print(f"Saved model to: {model_path}")

    report = pd.DataFrame(report_rows)
    report.to_csv(report_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(f"\nSaved no-odds report to: {report_path}")
    print(f"Saved no-odds predictions to: {predictions_path}")
    return report, predictions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train no-odds binary outcome models.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    """Run no-odds model training."""
    args = parse_args()
    train_evaluate_no_odds_models(args.features, args.models_dir, args.report, args.predictions, args.train_ratio)


if __name__ == "__main__":
    main()
