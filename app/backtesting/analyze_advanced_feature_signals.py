"""Analyze advanced Football-Data feature signals from feature importance reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "feature_importance_report.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "advanced_feature_signals.csv"
FEATURE_KEYWORDS = ["shots", "target", "corners", "fouls", "cards"]
EXTENDED_FEATURE_KEYWORDS = [
    "shots",
    "target",
    "corners",
    "fouls",
    "cards",
    "diff",
    "pressure",
    "matchup",
    "discipline",
    "attack",
    "defensive",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in feature importance report: {', '.join(missing)}")


def _model_scope(model_names: list[str]) -> str:
    """Classify whether a feature appears in odds, no-odds, or both model families."""
    has_no_odds = any("no_odds" in name for name in model_names)
    has_with_odds = any("no_odds" not in name for name in model_names)
    if has_no_odds and has_with_odds:
        return "both"
    if has_no_odds:
        return "no_odds_only"
    return "with_odds_only"


def _signal_stability(positive_count: int, negative_count: int) -> str:
    """Categorize the sign consistency of coefficients across models."""
    total = positive_count + negative_count
    if total == 0:
        return "mixed"

    positive_ratio = positive_count / total
    negative_ratio = negative_count / total
    if positive_ratio >= 0.8:
        return "stable_positive"
    if negative_ratio >= 0.8:
        return "stable_negative"
    return "mixed"


def _feature_family(feature: str) -> str:
    """Assign a feature to a higher-level family."""
    normalized = feature.lower()
    if "shots_on_target" in normalized or "sot" in normalized or ("target" in normalized and "shots" in normalized):
        return "shots_on_target"
    if "shots" in normalized:
        return "shots"
    if "corner" in normalized:
        return "corners"
    if "foul" in normalized:
        return "fouls"
    if "yellow" in normalized or "red" in normalized or "card" in normalized:
        return "cards"
    if "pressure" in normalized:
        return "pressure"
    if "matchup" in normalized:
        return "matchup"
    if "discipline" in normalized:
        return "discipline"
    if "diff" in normalized:
        return "diff"
    return "other_advanced"


def analyze_advanced_feature_signals(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Aggregate advanced feature signals from the feature importance report."""
    if not input_path.exists():
        raise FileNotFoundError(f"Feature importance report not found: {input_path}")

    report = pd.read_csv(input_path)
    _require_columns(report, ["model_name", "feature", "coefficient", "abs_coefficient", "direction"])

    mask = report["feature"].str.contains("|".join(EXTENDED_FEATURE_KEYWORDS), case=False, na=False)
    filtered = report[mask].copy()
    if filtered.empty:
        raise ValueError("No advanced features found in the feature importance report.")

    rows = []
    for feature, feature_df in filtered.groupby("feature", sort=True):
        model_names = sorted(feature_df["model_name"].astype(str).unique().tolist())
        positive_count = int((feature_df["coefficient"] > 0).sum())
        negative_count = int((feature_df["coefficient"] < 0).sum())
        rows.append(
            {
                "feature": feature,
                "models_count": int(len(model_names)),
                "mean_coefficient": float(feature_df["coefficient"].mean()),
                "median_coefficient": float(feature_df["coefficient"].median()),
                "abs_mean_coefficient": float(feature_df["abs_coefficient"].mean()),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "models": "; ".join(model_names),
                "signal_stability": _signal_stability(positive_count, negative_count),
                "model_scope": _model_scope(model_names),
                "feature_family": _feature_family(feature),
            }
        )

    analysis = pd.DataFrame(rows).sort_values(
        ["abs_mean_coefficient", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)

    print("Advanced feature signals analysis")
    print(f"Input: {input_path}")
    print(f"Total advanced feature rows: {len(filtered)}")
    print(f"Unique advanced features: {len(analysis)}")

    print("\nTop 30 advanced features by abs_mean_coefficient:")
    print(
        analysis.head(30)[
            [
                "feature",
                "feature_family",
                "models_count",
                "mean_coefficient",
                "median_coefficient",
                "abs_mean_coefficient",
                "signal_stability",
                "model_scope",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    stable_positive = analysis[analysis["signal_stability"] == "stable_positive"]
    stable_negative = analysis[analysis["signal_stability"] == "stable_negative"]
    mixed = analysis[analysis["signal_stability"] == "mixed"]

    print("\nMost stable positive features:")
    if stable_positive.empty:
        print("None")
    else:
        print(
        stable_positive.head(20)[
                ["feature", "models_count", "mean_coefficient", "abs_mean_coefficient", "models"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )

    print("\nMost stable negative features:")
    if stable_negative.empty:
        print("None")
    else:
        print(
        stable_negative.head(20)[
                ["feature", "models_count", "mean_coefficient", "abs_mean_coefficient", "models"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )

    print("\nMixed features to monitor:")
    if mixed.empty:
        print("None")
    else:
        print(
        mixed.head(20)[
                ["feature", "models_count", "mean_coefficient", "abs_mean_coefficient", "models"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )

    print("\nTop 10 by feature family:")
    for family in sorted(analysis["feature_family"].dropna().astype(str).unique().tolist()):
        family_df = analysis[analysis["feature_family"] == family].sort_values(
            ["abs_mean_coefficient", "feature"],
            ascending=[False, True],
        ).head(10)
        print(f"\n[{family}]")
        print(
            family_df[
                [
                    "feature",
                    "models_count",
                    "mean_coefficient",
                    "abs_mean_coefficient",
                    "signal_stability",
                    "model_scope",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )

    print(f"\nSaved advanced feature signals to: {output_path}")
    return analysis


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze advanced Football-Data feature signals.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the advanced feature signal analysis from the command line."""
    args = parse_args()
    analyze_advanced_feature_signals(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
