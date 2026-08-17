"""Analyze logistic regression feature importance across trained outcome models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "feature_importance_report.csv"

MODEL_PATHS = {
    "home_win_model_classic": PROJECT_ROOT / "models" / "outcome_models" / "home_win_model_classic.pkl",
    "away_win_model_classic": PROJECT_ROOT / "models" / "outcome_models" / "away_win_model_classic.pkl",
    "draw_model_classic": PROJECT_ROOT / "models" / "outcome_models" / "draw_model_classic.pkl",
    "home_win_model_no_odds": PROJECT_ROOT / "models" / "outcome_models_no_odds" / "home_win_model_no_odds.pkl",
    "away_win_model_no_odds": PROJECT_ROOT / "models" / "outcome_models_no_odds" / "away_win_model_no_odds.pkl",
    "draw_model_no_odds": PROJECT_ROOT / "models" / "outcome_models_no_odds" / "draw_model_no_odds.pkl",
}


def load_model(model_path: Path) -> Any:
    """Load a persisted sklearn pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_features(features_path: Path) -> pd.DataFrame:
    """Load the feature dataset used for training."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    return pd.read_csv(features_path)


def select_feature_columns(df: pd.DataFrame, model: Any) -> list[str]:
    """Resolve the feature columns used by a fitted pipeline."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scaler")
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            return list(scaler.feature_names_in_)
    raise ValueError("Unable to determine feature names from the fitted model.")


def extract_coefficients(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return a flat coefficient table for a fitted logistic regression pipeline."""
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "coef_"):
        raise ValueError("Model does not contain a logistic regression classifier with coefficients.")

    coefficients = pd.Series(classifier.coef_.ravel(), index=feature_names)
    report = pd.DataFrame(
        {
            "feature": coefficients.index,
            "coefficient": coefficients.values,
        }
    )
    report["abs_coefficient"] = report["coefficient"].abs()
    report["direction"] = report["coefficient"].apply(lambda value: "positive" if value >= 0 else "negative")
    return report


def summarize_model(model_name: str, report: pd.DataFrame) -> None:
    """Print the top coefficient summaries for one model."""
    print(f"\n{model_name}")

    top_positive = report.sort_values(["coefficient", "feature"], ascending=[False, True]).head(15)
    top_negative = report.sort_values(["coefficient", "feature"], ascending=[True, True]).head(15)
    top_absolute = report.sort_values(["abs_coefficient", "feature"], ascending=[False, True]).head(15)

    print("Top 15 positive features:")
    print(top_positive[["feature", "coefficient"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("Top 15 negative features:")
    print(top_negative[["feature", "coefficient"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("Top 15 by absolute coefficient:")
    print(top_absolute[["feature", "coefficient", "abs_coefficient"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def analyze_feature_importance(
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Analyze coefficient importance across all configured models."""
    features = load_features(features_path)
    all_rows = []

    print("Feature importance analysis")
    print(f"Feature file: {features_path}")

    for model_name, model_path in MODEL_PATHS.items():
        model = load_model(model_path)
        feature_names = select_feature_columns(features, model)
        report = extract_coefficients(model, feature_names)
        report.insert(0, "model_name", model_name)
        all_rows.append(report)
        print(f"\n{model_name}: {len(feature_names)} features")
        summarize_model(model_name, report)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[["model_name", "feature", "coefficient", "abs_coefficient", "direction"]]
    combined = combined.sort_values(["model_name", "abs_coefficient", "feature"], ascending=[True, False, True]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"\nSaved feature importance report to: {output_path}")
    return combined


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze logistic regression feature importance.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the feature importance analysis from the command line."""
    args = parse_args()
    analyze_feature_importance(features_path=args.features, output_path=args.output)


if __name__ == "__main__":
    main()
