"""Optimize recommended-class thresholds for the Ligue 1 specialist engine."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_ligue1_backtest.csv"
BASELINE_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_specialist_threshold_optimization.csv"

RESULT_LABELS = ["H", "D", "A"]
REQUIRED_COLUMNS = [
    "actual_result",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "predicted_class_argmax",
    "favorite_probability",
    "top_two_margin",
]
DRAW_PROBABILITY_MINIMUMS = [round(value / 100, 2) for value in range(26, 37)]
TOP_TWO_MARGIN_MAXIMUMS = [round(value / 100, 2) for value in range(3, 13)]
FAVORITE_PROBABILITY_MAXIMUMS = [round(value / 100, 2) for value in range(38, 51)]
MIN_D_RECOMMENDED_FOR_QUALITY = 10
MIN_D_DRAW_RATE_FOR_BALANCED = 0.35


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in specialist backtest: {', '.join(missing)}")


def class_distribution(predictions: pd.Series) -> dict[str, int]:
    """Return H/D/A prediction counts."""
    counts = predictions.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"recommended_{label}": int(counts[label]) for label in RESULT_LABELS}


def baseline_accuracy() -> float | None:
    """Return current Big 5 engine Ligue 1 recommended accuracy when available."""
    if not BASELINE_INPUT_PATH.exists():
        return None

    baseline = pd.read_csv(BASELINE_INPUT_PATH)
    required = {"league", "is_correct_recommended"}
    if not required.issubset(baseline.columns):
        return None

    ligue1 = baseline[baseline["league"].astype(str) == "Ligue 1"].copy()
    if ligue1.empty:
        return None
    return float(ligue1["is_correct_recommended"].mean())


def apply_rule(
    df: pd.DataFrame,
    draw_probability_min: float,
    top_two_margin_max: float,
    favorite_probability_max: float,
) -> pd.Series:
    """Apply the tested recommendation rule."""
    recommend_draw = (
        (df["draw_probability"] >= draw_probability_min)
        & (df["top_two_margin"] <= top_two_margin_max)
        & (df["favorite_probability"] <= favorite_probability_max)
    )
    predictions = df["predicted_class_argmax"].copy()
    predictions.loc[recommend_draw] = "D"
    return predictions


def optimization_row(
    df: pd.DataFrame,
    draw_probability_min: float,
    top_two_margin_max: float,
    favorite_probability_max: float,
    baseline_recommended_accuracy: float | None,
) -> dict[str, Any]:
    """Build one optimization row."""
    predictions = apply_rule(df, draw_probability_min, top_two_margin_max, favorite_probability_max)
    d_mask = predictions == "D"
    d_recommended = int(d_mask.sum())
    actual_draw_rate_among_d = float((df.loc[d_mask, "actual_result"] == "D").mean()) if d_recommended else 0.0
    accuracy_recommended = float((predictions == df["actual_result"]).mean())

    row: dict[str, Any] = {
        "draw_probability_min": draw_probability_min,
        "top_two_margin_max": top_two_margin_max,
        "favorite_probability_max": favorite_probability_max,
        "matches": int(len(df)),
        "accuracy_recommended": accuracy_recommended,
        "d_recommended": d_recommended,
        "d_recommended_rate": d_recommended / len(df),
        "actual_draw_rate_among_d_recommended": actual_draw_rate_among_d,
        "baseline_recommended_accuracy": baseline_recommended_accuracy,
        "accuracy_delta_vs_baseline": (
            accuracy_recommended - baseline_recommended_accuracy
            if baseline_recommended_accuracy is not None
            else pd.NA
        ),
    }
    row.update(class_distribution(predictions))
    return row


def balanced_score(row: pd.Series) -> float:
    """Rank rules by accuracy, draw quality, and enough but not excessive D volume."""
    d_count = int(row["d_recommended"])
    if d_count < MIN_D_RECOMMENDED_FOR_QUALITY:
        return -1.0
    if float(row["actual_draw_rate_among_d_recommended"]) < MIN_D_DRAW_RATE_FOR_BALANCED:
        return -1.0

    d_rate = float(row["d_recommended_rate"])
    return (
        float(row["accuracy_recommended"])
        + 0.25 * float(row["actual_draw_rate_among_d_recommended"])
        + 0.05 * min(d_rate, 0.12) / 0.12
        - 0.05 * max(0.0, d_rate - 0.20)
    )


def optimize_ligue1_specialist_thresholds(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Optimize specialist Ligue 1 recommendation thresholds and export all rows."""
    if not input_path.exists():
        raise FileNotFoundError(f"Ligue 1 specialist backtest not found: {input_path}")

    df = pd.read_csv(input_path)
    require_columns(df, REQUIRED_COLUMNS)
    for column in ["draw_probability", "top_two_margin", "favorite_probability"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["draw_probability", "top_two_margin", "favorite_probability"]].isna().any().any():
        raise ValueError("Specialist backtest contains invalid threshold input values.")

    baseline_recommended_accuracy = baseline_accuracy()
    rows = [
        optimization_row(
            df=df,
            draw_probability_min=draw_probability_min,
            top_two_margin_max=top_two_margin_max,
            favorite_probability_max=favorite_probability_max,
            baseline_recommended_accuracy=baseline_recommended_accuracy,
        )
        for draw_probability_min, top_two_margin_max, favorite_probability_max in product(
            DRAW_PROBABILITY_MINIMUMS,
            TOP_TWO_MARGIN_MAXIMUMS,
            FAVORITE_PROBABILITY_MAXIMUMS,
        )
    ]

    optimization = pd.DataFrame(rows)
    optimization["balanced_score"] = optimization.apply(balanced_score, axis=1)
    optimization = optimization.sort_values(
        ["accuracy_recommended", "actual_draw_rate_among_d_recommended", "d_recommended"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimization.to_csv(output_path, index=False)
    print_summary(optimization, input_path, output_path, baseline_recommended_accuracy)
    return optimization


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a compact optimization table."""
    columns = [
        "draw_probability_min",
        "top_two_margin_max",
        "favorite_probability_max",
        "accuracy_recommended",
        "d_recommended",
        "actual_draw_rate_among_d_recommended",
        "recommended_H",
        "recommended_D",
        "recommended_A",
        "accuracy_delta_vs_baseline",
        "balanced_score",
    ]
    print(f"\n{title}")
    print(df[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_summary(
    optimization: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    baseline_recommended_accuracy: float | None,
) -> None:
    """Print requested optimization summaries."""
    print("Ligue 1 specialist threshold optimization")
    print(f"Input: {input_path}")
    print(f"Rules tested: {len(optimization)}")
    if baseline_recommended_accuracy is None:
        print("Baseline recommended accuracy: not_available")
    else:
        print(f"Baseline recommended accuracy: {baseline_recommended_accuracy:.4f}")
    print(f"Saved optimization table to: {output_path}")

    print_table("Top 10 par accuracy_recommended", optimization.head(10))

    quality = optimization[optimization["d_recommended"] >= MIN_D_RECOMMENDED_FOR_QUALITY].sort_values(
        ["actual_draw_rate_among_d_recommended", "accuracy_recommended", "d_recommended"],
        ascending=[False, False, False],
    )
    print_table("Top 10 par taux reel de nul parmi D recommandes (min 10 D)", quality.head(10))

    balanced = optimization[optimization["balanced_score"] >= 0].sort_values(
        ["balanced_score", "accuracy_recommended", "actual_draw_rate_among_d_recommended"],
        ascending=[False, False, False],
    )
    if balanced.empty:
        print(
            "\nMeilleure regle equilibree: aucune regle avec au moins "
            f"{MIN_D_RECOMMENDED_FOR_QUALITY} D recommandes et taux reel de nul >= "
            f"{MIN_D_DRAW_RATE_FOR_BALANCED:.2f}."
        )
    else:
        print_table("Meilleure regle equilibree", balanced.head(1))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimize Ligue 1 specialist recommendation thresholds.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the threshold optimization from the command line."""
    args = parse_args()
    optimize_ligue1_specialist_thresholds(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
