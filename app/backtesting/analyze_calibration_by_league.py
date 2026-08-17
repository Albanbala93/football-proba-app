"""Analyze probability calibration by league."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "calibration_by_league.csv"

BUCKET_LABELS = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]
PROBABILITY_ANALYSES = {
    "H": ("home_win_probability", "H"),
    "D": ("draw_probability", "D"),
    "A": ("away_win_probability", "A"),
}


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in probability engine backtest: {', '.join(missing)}")


def _bucket_label(probability: float) -> str:
    """Return the fixed 0.10-wide calibration bucket for a probability."""
    clipped_probability = min(max(float(probability), 0.0), 1.0)
    bucket_index = min(int(clipped_probability * 10), len(BUCKET_LABELS) - 1)
    return BUCKET_LABELS[bucket_index]


def analyze_calibration_by_league(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build a bucketed calibration table for H/D/A probabilities by league."""
    if not input_path.exists():
        raise FileNotFoundError(f"Backtest report not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = ["league", "actual_result"] + [
        probability_column for probability_column, _ in PROBABILITY_ANALYSES.values()
    ]
    _require_columns(df, required_columns)

    df = df.dropna(subset=required_columns).copy()
    df["actual_result"] = df["actual_result"].astype(str).str.upper().str.strip()
    df = df[df["actual_result"].isin(["H", "D", "A"])].copy()

    rows = []
    for league, league_df in df.groupby("league", sort=True):
        for probability_type, (probability_column, actual_label) in PROBABILITY_ANALYSES.items():
            analysis_df = league_df[[probability_column, "actual_result"]].copy()
            analysis_df[probability_column] = pd.to_numeric(analysis_df[probability_column], errors="coerce")
            analysis_df = analysis_df.dropna(subset=[probability_column])
            analysis_df["bucket"] = analysis_df[probability_column].map(_bucket_label)
            analysis_df["is_actual"] = analysis_df["actual_result"].eq(actual_label)

            for bucket in BUCKET_LABELS:
                bucket_data = analysis_df[analysis_df["bucket"] == bucket]
                matches = int(len(bucket_data))
                avg_predicted_probability = float(bucket_data[probability_column].mean()) if matches else 0.0
                actual_frequency = float(bucket_data["is_actual"].mean()) if matches else 0.0
                rows.append(
                    {
                        "league": league,
                        "probability_type": probability_type,
                        "bucket": bucket,
                        "matches": matches,
                        "avg_predicted_probability": avg_predicted_probability,
                        "actual_frequency": actual_frequency,
                        "calibration_gap": avg_predicted_probability - actual_frequency,
                    }
                )

    calibration = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_path, index=False)

    print_calibration_summary(calibration, input_path, output_path, len(df))
    return calibration


def _mean_abs_gap(calibration: pd.DataFrame, league: str, probability_type: str) -> float:
    """Return the average absolute calibration gap on non-empty buckets."""
    selected = calibration[
        (calibration["league"] == league)
        & (calibration["probability_type"] == probability_type)
        & (calibration["matches"] > 0)
    ]
    if selected.empty:
        return 0.0
    return float(selected["calibration_gap"].abs().mean())


def _worst_bucket(calibration: pd.DataFrame, league: str) -> str:
    """Return the most miscalibrated non-empty bucket for a league."""
    selected = calibration[(calibration["league"] == league) & (calibration["matches"] > 0)].copy()
    if selected.empty:
        return "n/a"

    selected["abs_gap"] = selected["calibration_gap"].abs()
    worst = selected.sort_values(["abs_gap", "matches"], ascending=[False, False]).iloc[0]
    return (
        f"{worst['probability_type']} {worst['bucket']} "
        f"(gap={worst['calibration_gap']:.4f}, matches={int(worst['matches'])})"
    )


def print_calibration_summary(
    calibration: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    total_matches: int,
) -> None:
    """Print a compact per-league calibration summary."""
    print("Calibration by league")
    print(f"Input: {input_path}")
    print(f"Total valid matches: {total_matches}")

    summary_rows = []
    for league in sorted(calibration["league"].unique()):
        summary_rows.append(
            {
                "league": league,
                "mean_abs_gap_H": _mean_abs_gap(calibration, league, "H"),
                "mean_abs_gap_D": _mean_abs_gap(calibration, league, "D"),
                "mean_abs_gap_A": _mean_abs_gap(calibration, league, "A"),
                "worst_bucket": _worst_bucket(calibration, league),
            }
        )

    print("\nSummary by league:")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved calibration by league to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze probability calibration by league.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run calibration analysis from the command line."""
    args = parse_args()
    analyze_calibration_by_league(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
