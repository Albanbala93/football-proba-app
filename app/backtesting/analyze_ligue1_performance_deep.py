"""Analyze current probability engine performance deeply for Ligue 1 only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_deep_performance_summary.csv"
DEFAULT_PROFILE_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_performance_by_profile.csv"
DEFAULT_CONFIDENCE_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_performance_by_confidence.csv"
DEFAULT_DRAW_BUCKET_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_performance_by_draw_bucket.csv"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]
PROBABILITY_COLUMNS = ["away_win_probability", "draw_probability", "home_win_probability"]
REQUIRED_COLUMNS = [
    "league",
    "actual_result",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "predicted_class_argmax",
    "predicted_class_adjusted",
    "recommended_prediction_class",
    "match_profile",
    "confidence_score",
]
CONFIDENCE_BUCKETS = [0, 20, 40, 60, 80, 100.000001]
CONFIDENCE_LABELS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
DRAW_BUCKETS = [0.00, 0.20, 0.25, 0.30, 0.35, float("inf")]
DRAW_BUCKET_LABELS = ["0.00-0.20", "0.20-0.25", "0.25-0.30", "0.30-0.35", "0.35+"]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in backtest file: {', '.join(missing)}")


def class_distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A count columns for a class series."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def probabilities_for_log_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Return normalized A/D/H probabilities for sklearn log_loss."""
    probabilities = df[PROBABILITY_COLUMNS].astype(float).copy()
    return probabilities.div(probabilities.sum(axis=1), axis=0)


def multiclass_brier_score(df: pd.DataFrame) -> float:
    """Compute the mean multiclass Brier score for H/D/A probabilities."""
    total = 0.0
    for label, probability_column in [
        ("H", "home_win_probability"),
        ("D", "draw_probability"),
        ("A", "away_win_probability"),
    ]:
        actual = (df["actual_result"] == label).astype(float)
        total += ((df[probability_column].astype(float) - actual) ** 2).mean()
    return float(total / 3)


def accuracy(df: pd.DataFrame, prediction_column: str) -> float:
    """Compute accuracy for one prediction column."""
    return float(accuracy_score(df["actual_result"], df[prediction_column]))


def group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """Build the requested metrics for one subgroup."""
    if group.empty:
        return {
            "matches": 0,
            "accuracy_recommended": pd.NA,
            "actual_home_win_rate": pd.NA,
            "actual_draw_rate": pd.NA,
            "actual_away_win_rate": pd.NA,
            "avg_home_win_probability": pd.NA,
            "avg_draw_probability": pd.NA,
            "avg_away_win_probability": pd.NA,
            "calibration_gap_draw": pd.NA,
        }

    actual_draw_rate = float((group["actual_result"] == "D").mean())
    avg_draw_probability = float(group["draw_probability"].mean())
    return {
        "matches": int(len(group)),
        "accuracy_recommended": accuracy(group, "recommended_prediction_class"),
        "actual_home_win_rate": float((group["actual_result"] == "H").mean()),
        "actual_draw_rate": actual_draw_rate,
        "actual_away_win_rate": float((group["actual_result"] == "A").mean()),
        "avg_home_win_probability": float(group["home_win_probability"].mean()),
        "avg_draw_probability": avg_draw_probability,
        "avg_away_win_probability": float(group["away_win_probability"].mean()),
        "calibration_gap_draw": avg_draw_probability - actual_draw_rate,
    }


def grouped_performance(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Aggregate requested performance metrics by one column."""
    rows = []
    for group_value, group in df.groupby(group_column, sort=True, observed=False):
        rows.append({group_column: str(group_value), **group_metrics(group)})
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row Ligue 1 summary with metrics and distributions."""
    probabilities = probabilities_for_log_loss(df)
    row: dict[str, Any] = {
        "league": "Ligue 1",
        "matches": int(len(df)),
        "accuracy_argmax": accuracy(df, "predicted_class_argmax"),
        "accuracy_adjusted": accuracy(df, "predicted_class_adjusted"),
        "accuracy_recommended": accuracy(df, "recommended_prediction_class"),
        "log_loss": float(log_loss(df["actual_result"], probabilities, labels=LOG_LOSS_LABELS)),
        "brier_score_multiclass": multiclass_brier_score(df),
    }
    row.update(class_distribution("actual_result", df["actual_result"]))
    row.update(class_distribution("predicted_argmax", df["predicted_class_argmax"]))
    row.update(class_distribution("predicted_adjusted", df["predicted_class_adjusted"]))
    row.update(class_distribution("predicted_recommended", df["recommended_prediction_class"]))
    return pd.DataFrame([row])


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add confidence and draw-probability buckets."""
    bucketed = df.copy()
    bucketed["confidence_score"] = pd.to_numeric(bucketed["confidence_score"], errors="coerce")
    bucketed["draw_probability"] = pd.to_numeric(bucketed["draw_probability"], errors="coerce")
    bucketed["confidence_score_bucket"] = pd.cut(
        bucketed["confidence_score"],
        bins=CONFIDENCE_BUCKETS,
        labels=CONFIDENCE_LABELS,
        include_lowest=True,
        right=False,
    )
    bucketed["draw_probability_bucket"] = pd.cut(
        bucketed["draw_probability"],
        bins=DRAW_BUCKETS,
        labels=DRAW_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )
    return bucketed


def analyze_ligue1_performance_deep(
    input_path: Path = DEFAULT_INPUT_PATH,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    profile_output_path: Path = DEFAULT_PROFILE_OUTPUT_PATH,
    confidence_output_path: Path = DEFAULT_CONFIDENCE_OUTPUT_PATH,
    draw_bucket_output_path: Path = DEFAULT_DRAW_BUCKET_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the Ligue 1 deep performance analysis and export CSV outputs."""
    if not input_path.exists():
        raise FileNotFoundError(f"Backtest report not found: {input_path}")

    df = pd.read_csv(input_path)
    require_columns(df, REQUIRED_COLUMNS)

    ligue1 = df[df["league"].astype(str) == "Ligue 1"].copy()
    if ligue1.empty:
        raise ValueError("No Ligue 1 rows found in probability engine backtest.")

    for column in ["home_win_probability", "draw_probability", "away_win_probability"]:
        ligue1[column] = pd.to_numeric(ligue1[column], errors="coerce")

    if ligue1[["home_win_probability", "draw_probability", "away_win_probability"]].isna().any().any():
        raise ValueError("Ligue 1 rows contain missing or invalid probability values.")
    ligue1 = add_buckets(ligue1)

    summary = build_summary(ligue1)
    by_profile = grouped_performance(ligue1, "match_profile").sort_values(
        ["matches", "match_profile"],
        ascending=[False, True],
    )
    by_confidence = grouped_performance(ligue1, "confidence_score_bucket")
    by_draw_bucket = grouped_performance(ligue1, "draw_probability_bucket")

    for output_path, output_df in [
        (summary_output_path, summary),
        (profile_output_path, by_profile),
        (confidence_output_path, by_confidence),
        (draw_bucket_output_path, by_draw_bucket),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)

    print_summary(
        summary=summary,
        by_profile=by_profile,
        by_confidence=by_confidence,
        by_draw_bucket=by_draw_bucket,
        input_path=input_path,
        output_paths=[summary_output_path, profile_output_path, confidence_output_path, draw_bucket_output_path],
    )
    return summary, by_profile, by_confidence, by_draw_bucket


def print_summary(
    summary: pd.DataFrame,
    by_profile: pd.DataFrame,
    by_confidence: pd.DataFrame,
    by_draw_bucket: pd.DataFrame,
    input_path: Path,
    output_paths: list[Path],
) -> None:
    """Print a clear console summary."""
    row = summary.iloc[0]
    print("Ligue 1 deep performance analysis")
    print(f"Input: {input_path}")
    print(f"Matches: {int(row['matches'])}")
    print(f"Accuracy argmax: {row['accuracy_argmax']:.4f}")
    print(f"Accuracy adjusted: {row['accuracy_adjusted']:.4f}")
    print(f"Accuracy recommended: {row['accuracy_recommended']:.4f}")
    print(f"Log loss: {row['log_loss']:.4f}")
    print(f"Brier score multiclass: {row['brier_score_multiclass']:.4f}")

    print("\nDistributions H/D/A:")
    distribution_columns = [
        "actual_result_H",
        "actual_result_D",
        "actual_result_A",
        "predicted_argmax_H",
        "predicted_argmax_D",
        "predicted_argmax_A",
        "predicted_adjusted_H",
        "predicted_adjusted_D",
        "predicted_adjusted_A",
        "predicted_recommended_H",
        "predicted_recommended_D",
        "predicted_recommended_A",
    ]
    print(summary[distribution_columns].to_string(index=False))

    display_columns = ["matches", "accuracy_recommended", "actual_draw_rate", "avg_draw_probability", "calibration_gap_draw"]
    print("\nBy match_profile:")
    print(by_profile[["match_profile", *display_columns]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBy confidence_score bucket:")
    print(
        by_confidence[["confidence_score_bucket", *display_columns]].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\nBy draw_probability bucket:")
    print(
        by_draw_bucket[["draw_probability_bucket", *display_columns]].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    for output_path in output_paths:
        print(f"- {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze current probability engine performance for Ligue 1.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--profile-output", type=Path, default=DEFAULT_PROFILE_OUTPUT_PATH)
    parser.add_argument("--confidence-output", type=Path, default=DEFAULT_CONFIDENCE_OUTPUT_PATH)
    parser.add_argument("--draw-bucket-output", type=Path, default=DEFAULT_DRAW_BUCKET_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the Ligue 1 deep analysis from the command line."""
    args = parse_args()
    analyze_ligue1_performance_deep(
        input_path=args.input,
        summary_output_path=args.summary_output,
        profile_output_path=args.profile_output,
        confidence_output_path=args.confidence_output,
        draw_bucket_output_path=args.draw_bucket_output,
    )


if __name__ == "__main__":
    main()
