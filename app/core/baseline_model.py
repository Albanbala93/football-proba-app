"""Train a baseline logistic classifier for football match results.

The model predicts one of three labels:
H = home win, D = draw, A = away win.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_logistic_model.pkl"
RESULT_LABELS = ["A", "D", "H"]

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
    "home_draw_rate_5",
    "away_draw_rate_5",
    "h2h_draw_rate",
    "h2h_matches_count",
    "abs_elo_diff",
    "abs_form_diff_5",
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
]
TARGET_CANDIDATES = ["result", "FTR", "target"]
DATE_CANDIDATES = ["Date", "date", "match_date"]

# Kept for modules importing the previous constant. Runtime selection is based
# on the columns that actually exist in the feature dataset.
FEATURE_COLUMNS = FEATURE_CANDIDATES


def select_target_column(df: pd.DataFrame) -> str:
    """Return the first supported target column found in the dataset."""
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError("No target column found. Expected one of: result, FTR, target")


def select_date_column(df: pd.DataFrame) -> str | None:
    """Return the first supported date column found in the dataset, if any."""
    for column in DATE_CANDIDATES:
        if column in df.columns:
            return column
    return None


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return usable model features from the configured candidate list."""
    selected = [column for column in FEATURE_CANDIDATES if column in df.columns]
    if not selected:
        raise ValueError("No usable feature columns found in matches_features.csv")
    return selected


def load_training_data(features_path: Path) -> tuple[pd.DataFrame, list[str], str, str | None]:
    """Load features, select columns, clean missing rows, and preserve time order."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    df = pd.read_csv(features_path)
    target_column = select_target_column(df)
    date_column = select_date_column(df)
    feature_columns = select_feature_columns(df)

    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.sort_values(date_column).reset_index(drop=True)

    required_columns = feature_columns + [target_column]
    if date_column:
        required_columns.append(date_column)

    before_rows = len(df)
    df = df.dropna(subset=required_columns).copy()
    df[target_column] = df[target_column].astype(str).str.upper().str.strip()
    df = df[df[target_column].isin(["H", "D", "A"])].reset_index(drop=True)

    dropped_rows = before_rows - len(df)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing features, dates, or invalid targets.")

    if len(df) < 5:
        raise ValueError("At least 5 valid rows are required to train and test the baseline model.")
    if df[target_column].nunique() < 2:
        raise ValueError("At least 2 result classes are required to train LogisticRegression.")

    return df, feature_columns, target_column, date_column


def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by order: oldest 80 percent train, newest 20 percent test."""
    split_index = int(len(df) * train_ratio)
    split_index = max(1, min(split_index, len(df) - 1))
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def build_model() -> Pipeline:
    """Create the standardized logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_baseline_model(features_path: Path = DEFAULT_FEATURES_PATH, model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Train on all available cleaned feature rows and persist the model."""
    features, feature_columns, target_column, _ = load_training_data(features_path)

    model = build_model()
    model.fit(features[feature_columns], features[target_column])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def predict_probabilities(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Return H/D/A probabilities for feature rows."""
    feature_columns = list(getattr(model, "feature_names_in_", select_feature_columns(features)))
    probabilities = model.predict_proba(features[feature_columns])
    classes = list(model.named_steps["classifier"].classes_)
    output = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in classes])

    for label in ["H", "D", "A"]:
        column = f"prob_{label}"
        if column not in output.columns:
            output[column] = 0.0

    return output[["prob_H", "prob_D", "prob_A"]]


def print_sample_predictions(model: Pipeline, test_data: pd.DataFrame, feature_columns: list[str]) -> None:
    """Print a few test predictions with H/D/A probabilities."""
    sample = test_data.head(10).copy()
    probabilities = predict_probabilities(model, sample)
    predictions = probabilities.idxmax(axis=1).str.replace("prob_", "", regex=False)

    display_columns = [column for column in ["Date", "date", "HomeTeam", "AwayTeam", "FTR", "result", "target"] if column in sample]
    output = sample[display_columns].reset_index(drop=True)
    output = pd.concat([output, probabilities.reset_index(drop=True)], axis=1)
    output["prediction"] = predictions.to_numpy()

    print("\nSample predictions:")
    print(output.to_string(index=False))


def train_evaluate_and_save(features_path: Path, model_path: Path, train_ratio: float = 0.8) -> Pipeline:
    """Train the model, evaluate on the newest holdout split, and save it."""
    data, feature_columns, target_column, date_column = load_training_data(features_path)
    train_data, test_data = temporal_train_test_split(data, train_ratio=train_ratio)

    print(f"Loaded {len(data)} valid rows from: {features_path}")
    print(f"Target column: {target_column}")
    print(f"Date column: {date_column or 'not found - using current file order'}")
    print(f"Feature columns ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"Temporal split: {len(train_data)} train rows, {len(test_data)} test rows")

    model = build_model()
    model.fit(train_data[feature_columns], train_data[target_column])

    probabilities = predict_probabilities(model, test_data)
    predictions = probabilities.idxmax(axis=1).str.replace("prob_", "", regex=False)

    print("\nMetrics:")
    print(f"Accuracy: {accuracy_score(test_data[target_column], predictions):.4f}")
    print(
        "Log loss: "
        f"{log_loss(test_data[target_column], probabilities[['prob_A', 'prob_D', 'prob_H']], labels=RESULT_LABELS):.4f}"
    )

    print("\nConfusion matrix:")
    matrix = confusion_matrix(test_data[target_column], predictions, labels=["H", "D", "A"])
    print(pd.DataFrame(matrix, index=["actual_H", "actual_D", "actual_A"], columns=["pred_H", "pred_D", "pred_A"]))

    print("\nClassification report:")
    print(classification_report(test_data[target_column], predictions, labels=["H", "D", "A"], zero_division=0))

    print_sample_predictions(model, test_data, feature_columns)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to: {model_path}")
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline training."""
    parser = argparse.ArgumentParser(description="Train and evaluate a baseline logistic football model.")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Input feature CSV path. Defaults to data/processed/matches_features.csv.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output model path. Defaults to models/baseline_logistic_model.pkl.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Oldest data share used for training. Defaults to 0.8.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training, evaluation, and model persistence from the command line."""
    args = parse_args()
    train_evaluate_and_save(features_path=args.features, model_path=args.model, train_ratio=args.train_ratio)


if __name__ == "__main__":
    main()
