"""Optimize recommended prediction thresholds for Ligue 1 only."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_threshold_optimization.csv"
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
    "favorite_probability",
    "top_two_margin",
]
DRAW_PROBABILITY_MINIMUMS = [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32]
TOP_TWO_MARGIN_MAXIMUMS = [0.04, 0.06, 0.08, 0.10, 0.12]
WEAK_FAVORITE_THRESHOLD = 0.42
MIN_D_PREDICTIONS_FOR_QUALITY = 10


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in backtest file: {', '.join(missing)}")


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


def class_distribution(predictions: pd.Series) -> dict[str, int]:
    """Return H/D/A prediction counts."""
    counts = predictions.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"predicted_{label}": int(counts[label]) for label in RESULT_LABELS}


def apply_threshold_rule(
    df: pd.DataFrame,
    draw_probability_minimum: float,
    top_two_margin_maximum: float,
) -> pd.Series:
    """Apply the Ligue 1 threshold rule to produce recommended predictions."""
    draw_rule = (df["draw_probability"] >= draw_probability_minimum) & (
        df["top_two_margin"] <= top_two_margin_maximum
    )
    weak_favorite_draw_rule = (df["favorite_probability"] < WEAK_FAVORITE_THRESHOLD) & (
        df["draw_probability"] >= draw_probability_minimum
    )
    recommend_draw = draw_rule | weak_favorite_draw_rule

    predictions = df["predicted_class_argmax"].copy()
    predictions.loc[recommend_draw] = "D"
    return predictions


def optimization_row(
    df: pd.DataFrame,
    draw_probability_minimum: float,
    top_two_margin_maximum: float,
    log_loss_value: float,
    brier_score_value: float,
) -> dict[str, Any]:
    """Build one optimization result row."""
    predictions = apply_threshold_rule(df, draw_probability_minimum, top_two_margin_maximum)
    d_mask = predictions == "D"
    d_predictions = int(d_mask.sum())
    actual_draw_rate_among_d = float((df.loc[d_mask, "actual_result"] == "D").mean()) if d_predictions else 0.0
    actual_draw_rate = float((df["actual_result"] == "D").mean())
    d_prediction_rate = d_predictions / len(df)

    row: dict[str, Any] = {
        "league": "Ligue 1",
        "rule": "draw_threshold_or_weak_favorite_draw",
        "draw_probability_minimum": draw_probability_minimum,
        "top_two_margin_maximum": top_two_margin_maximum,
        "weak_favorite_threshold": WEAK_FAVORITE_THRESHOLD,
        "matches": int(len(df)),
        "accuracy": float(accuracy_score(df["actual_result"], predictions)),
        "log_loss": log_loss_value,
        "brier_score_multiclass": brier_score_value,
        "d_predictions": d_predictions,
        "d_prediction_rate": d_prediction_rate,
        "actual_draw_rate": actual_draw_rate,
        "actual_draw_rate_among_d_predictions": actual_draw_rate_among_d,
        "d_precision_lift_vs_base_draw_rate": actual_draw_rate_among_d - actual_draw_rate,
    }
    row.update(class_distribution(predictions))
    return row


def balanced_score(row: pd.Series) -> float:
    """Score rules by accuracy, useful D signal, and conservative draw volume."""
    d_rate = float(row["d_prediction_rate"])
    draw_quality = float(row["actual_draw_rate_among_d_predictions"])
    accuracy = float(row["accuracy"])

    # Reward accuracy first, then D quality, while penalizing very high D volume.
    return accuracy + (0.25 * draw_quality) + (0.10 * min(d_rate, 0.15) / 0.15) - (0.10 * max(0.0, d_rate - 0.20))


def optimize_ligue1_thresholds(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run threshold optimization for Ligue 1 and export all tested rules."""
    if not input_path.exists():
        raise FileNotFoundError(f"Backtest report not found: {input_path}")

    df = pd.read_csv(input_path)
    require_columns(df, REQUIRED_COLUMNS)
    ligue1 = df[df["league"].astype(str) == "Ligue 1"].copy()
    if ligue1.empty:
        raise ValueError("No Ligue 1 rows found in probability engine backtest.")

    numeric_columns = [
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "favorite_probability",
        "top_two_margin",
    ]
    for column in numeric_columns:
        ligue1[column] = pd.to_numeric(ligue1[column], errors="coerce")
    if ligue1[numeric_columns].isna().any().any():
        raise ValueError("Ligue 1 rows contain missing or invalid numeric values needed for optimization.")

    log_loss_value = float(log_loss(ligue1["actual_result"], probabilities_for_log_loss(ligue1), labels=LOG_LOSS_LABELS))
    brier_score_value = multiclass_brier_score(ligue1)

    rows = [
        optimization_row(
            df=ligue1,
            draw_probability_minimum=draw_probability_minimum,
            top_two_margin_maximum=top_two_margin_maximum,
            log_loss_value=log_loss_value,
            brier_score_value=brier_score_value,
        )
        for draw_probability_minimum, top_two_margin_maximum in product(
            DRAW_PROBABILITY_MINIMUMS,
            TOP_TWO_MARGIN_MAXIMUMS,
        )
    ]

    optimization = pd.DataFrame(rows)
    optimization["balanced_score"] = optimization.apply(balanced_score, axis=1)
    optimization = optimization.sort_values(
        ["accuracy", "actual_draw_rate_among_d_predictions", "d_predictions"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimization.to_csv(output_path, index=False)
    print_summary(optimization, input_path, output_path)
    return optimization


def best_row_by_draw_quality(optimization: pd.DataFrame) -> pd.Series:
    """Return the best rule by D prediction quality, avoiding tiny samples when possible."""
    eligible = optimization[optimization["d_predictions"] >= MIN_D_PREDICTIONS_FOR_QUALITY].copy()
    if eligible.empty:
        eligible = optimization.copy()
    return eligible.sort_values(
        ["actual_draw_rate_among_d_predictions", "accuracy", "d_predictions"],
        ascending=[False, False, False],
    ).iloc[0]


def print_rule(label: str, row: pd.Series) -> None:
    """Print one selected threshold rule."""
    print(f"\n{label}")
    print(f"- draw_probability_minimum: {row['draw_probability_minimum']:.2f}")
    print(f"- top_two_margin_maximum: {row['top_two_margin_maximum']:.2f}")
    print(f"- weak_favorite_threshold: {row['weak_favorite_threshold']:.2f}")
    print(f"- accuracy: {row['accuracy']:.4f}")
    print(f"- D predits: {int(row['d_predictions'])} ({row['d_prediction_rate']:.2%})")
    print(f"- taux reel de nul parmi D predits: {row['actual_draw_rate_among_d_predictions']:.4f}")
    print(f"- distribution H/D/A: {int(row['predicted_H'])}/{int(row['predicted_D'])}/{int(row['predicted_A'])}")


def print_summary(optimization: pd.DataFrame, input_path: Path, output_path: Path) -> None:
    """Print a clear console summary."""
    best_accuracy = optimization.sort_values(
        ["accuracy", "actual_draw_rate_among_d_predictions", "d_predictions"],
        ascending=[False, False, False],
    ).iloc[0]
    best_draw_quality = best_row_by_draw_quality(optimization)
    best_balanced = optimization.sort_values(
        ["balanced_score", "accuracy", "actual_draw_rate_among_d_predictions"],
        ascending=[False, False, False],
    ).iloc[0]

    print("Ligue 1 threshold optimization")
    print(f"Input: {input_path}")
    print(f"Rules tested: {len(optimization)}")
    print(f"Saved optimization table to: {output_path}")

    print_rule("Meilleure regle par accuracy", best_accuracy)
    print_rule("Meilleure regle par qualite des D predits", best_draw_quality)
    print_rule("Regle la plus equilibree accuracy / nuls / prudence", best_balanced)

    print("\nTop 10 par accuracy:")
    columns = [
        "draw_probability_minimum",
        "top_two_margin_maximum",
        "accuracy",
        "d_predictions",
        "actual_draw_rate_among_d_predictions",
        "predicted_H",
        "predicted_D",
        "predicted_A",
        "balanced_score",
    ]
    print(optimization[columns].head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimize Ligue 1 recommended prediction thresholds.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run Ligue 1 threshold optimization from the command line."""
    args = parse_args()
    optimize_ligue1_thresholds(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
