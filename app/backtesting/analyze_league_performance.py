"""Analyze probability engine performance by league."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "league_performance_analysis.csv"
RESULT_LABELS = ["H", "D", "A"]
DRAW_SIGNAL_COLUMNS = [
    "is_draw_plausible",
    "is_strong_draw_signal",
    "is_very_strong_draw_signal",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in probability engine backtest: {', '.join(missing)}")


def _distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A distribution columns for a series."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to bool."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _actual_draw_rate_for_signal(league_df: pd.DataFrame, column: str) -> float | None:
    """Compute actual draw rate where a draw signal column is true."""
    if column not in league_df.columns:
        return None

    selected = league_df[_as_bool(league_df[column])]
    if selected.empty:
        return 0.0
    return float((selected["actual_result"] == "D").mean())


def _most_frequent_profile(league_df: pd.DataFrame) -> str | None:
    """Return the most common match_profile for a league."""
    if "match_profile" not in league_df.columns or league_df["match_profile"].dropna().empty:
        return None
    return str(league_df["match_profile"].value_counts().idxmax())


def analyze_league_performance(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Aggregate probability engine performance metrics by league."""
    if not input_path.exists():
        raise FileNotFoundError(f"Backtest report not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(
        df,
        [
            "league",
            "actual_result",
            "predicted_class_argmax",
            "predicted_class_adjusted",
            "recommended_prediction_class",
            "is_correct_argmax",
            "is_correct_adjusted",
            "is_correct_recommended",
            "draw_warning",
            "confidence_score",
            "match_profile",
        ],
    )

    rows = []
    for league, league_df in df.groupby("league", sort=True):
        adjusted_draw_matches = league_df[league_df["predicted_class_adjusted"] == "D"]
        draw_warning_matches = league_df[_as_bool(league_df["draw_warning"])]
        accuracy_argmax = float(league_df["is_correct_argmax"].mean())
        accuracy_adjusted = float(league_df["is_correct_adjusted"].mean())
        accuracy_recommended = float(league_df["is_correct_recommended"].mean())
        row = {
            "league": league,
            "matches": int(len(league_df)),
            "accuracy_argmax": accuracy_argmax,
            "accuracy_adjusted": accuracy_adjusted,
            "accuracy_recommended": accuracy_recommended,
            "recommended_minus_argmax": accuracy_recommended - accuracy_argmax,
            "recommended_minus_adjusted": accuracy_recommended - accuracy_adjusted,
            "actual_home_win_rate": float((league_df["actual_result"] == "H").mean()),
            "actual_draw_rate": float((league_df["actual_result"] == "D").mean()),
            "actual_away_win_rate": float((league_df["actual_result"] == "A").mean()),
            "adjusted_draw_predictions": int(len(adjusted_draw_matches)),
            "actual_draw_rate_on_adjusted_draw_predictions": (
                float((adjusted_draw_matches["actual_result"] == "D").mean()) if len(adjusted_draw_matches) else 0.0
            ),
            "draw_warning_count": int(len(draw_warning_matches)),
            "actual_draw_rate_on_draw_warning": (
                float((draw_warning_matches["actual_result"] == "D").mean()) if len(draw_warning_matches) else 0.0
            ),
            "avg_confidence_score": float(league_df["confidence_score"].mean()),
            "most_frequent_match_profile": _most_frequent_profile(league_df),
            "actual_draw_rate_when_is_draw_plausible": _actual_draw_rate_for_signal(league_df, "is_draw_plausible"),
            "actual_draw_rate_when_is_strong_draw_signal": _actual_draw_rate_for_signal(
                league_df,
                "is_strong_draw_signal",
            ),
            "actual_draw_rate_when_is_very_strong_draw_signal": _actual_draw_rate_for_signal(
                league_df,
                "is_very_strong_draw_signal",
            ),
        }
        row.update(_distribution("predicted_class_argmax", league_df["predicted_class_argmax"]))
        row.update(_distribution("predicted_class_adjusted", league_df["predicted_class_adjusted"]))
        row.update(_distribution("recommended_prediction_class", league_df["recommended_prediction_class"]))
        rows.append(row)

    analysis = pd.DataFrame(rows).sort_values("league").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)

    print("League performance analysis")
    print(f"Input: {input_path}")
    print(f"Total matches: {len(df)}")
    print("\nBy league:")
    summary_columns = [
        "league",
        "matches",
        "accuracy_argmax",
        "accuracy_adjusted",
        "accuracy_recommended",
        "draw_warning_count",
        "actual_draw_rate_on_draw_warning",
        "recommended_minus_argmax",
        "recommended_minus_adjusted",
    ]
    print(analysis[summary_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved league performance analysis to: {output_path}")
    return analysis


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze probability engine backtest by league.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run league performance analysis from the command line."""
    args = parse_args()
    analyze_league_performance(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
