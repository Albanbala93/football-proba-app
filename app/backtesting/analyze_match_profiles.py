"""Analyze probability engine performance by match profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "match_profiles_analysis.csv"
RESULT_LABELS = ["H", "D", "A"]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in probability engine backtest: {', '.join(missing)}")


def _distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A distribution columns for a series."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def _draw_signal_summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate optional draw signal metrics when columns are available."""
    summary: dict[str, float | int | None] = {}

    for column in ["is_draw_plausible", "is_strong_draw_signal", "is_very_strong_draw_signal"]:
        if column in df.columns:
            selected = df[df[column].astype(bool)]
            summary[f"{column}_matches"] = int(len(selected))
            summary[f"{column}_actual_draw_rate"] = float((selected["actual_result"] == "D").mean()) if len(selected) else 0.0
        else:
            summary[f"{column}_matches"] = None
            summary[f"{column}_actual_draw_rate"] = None

    if "calibrated_draw_signal" in df.columns:
        summary["calibrated_draw_signal_mean"] = float(df["calibrated_draw_signal"].mean())
    else:
        summary["calibrated_draw_signal_mean"] = None

    return summary


def analyze_match_profiles(input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Aggregate performance metrics by probability engine match_profile."""
    if not input_path.exists():
        raise FileNotFoundError(f"Backtest report not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(
        df,
        [
            "match_profile",
            "actual_result",
            "predicted_class",
            "is_correct",
            "confidence_score",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "home_no_loss_probability",
            "away_no_loss_probability",
        ],
    )

    rows = []
    total_matches = len(df)
    for profile, profile_df in df.groupby("match_profile", sort=True):
        row = {
            "match_profile": profile,
            "matches": int(len(profile_df)),
            "test_share_pct": len(profile_df) / total_matches * 100 if total_matches else 0.0,
            "accuracy_1N2": float(profile_df["is_correct"].mean()),
            "avg_confidence_score": float(profile_df["confidence_score"].mean()),
            "actual_home_win_rate": float((profile_df["actual_result"] == "H").mean()),
            "actual_draw_rate": float((profile_df["actual_result"] == "D").mean()),
            "actual_away_win_rate": float((profile_df["actual_result"] == "A").mean()),
            "avg_home_win_probability": float(profile_df["home_win_probability"].mean()),
            "avg_draw_probability": float(profile_df["draw_probability"].mean()),
            "avg_away_win_probability": float(profile_df["away_win_probability"].mean()),
            "avg_home_no_loss_probability": float(profile_df["home_no_loss_probability"].mean()),
            "avg_away_no_loss_probability": float(profile_df["away_no_loss_probability"].mean()),
        }
        row.update(_distribution("predicted_class", profile_df["predicted_class"]))
        row.update(_distribution("actual_result", profile_df["actual_result"]))
        rows.append(row)

    analysis = pd.DataFrame(rows).sort_values(["matches", "match_profile"], ascending=[False, True]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)

    print("Match profile analysis")
    print(f"Input: {input_path}")
    print(f"Total matches: {total_matches}")
    print("\nBy match_profile:")
    print(analysis.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nDraw signal analysis:")
    draw_summary = _draw_signal_summary(df)
    for key, value in draw_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved match profile analysis to: {output_path}")
    return analysis


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze probability engine backtest by match_profile.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run match profile analysis from the command line."""
    args = parse_args()
    analyze_match_profiles(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
