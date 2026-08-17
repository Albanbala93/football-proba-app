"""Compare the available probability engines on their backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIC_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_NO_ODDS_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_no_odds_backtest.csv"
DEFAULT_GB_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_gb_backtest.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "engine_comparison_summary.csv"
DEFAULT_BY_LEAGUE_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "engine_comparison_by_league.csv"

ENGINE_FILES = {
    "classic": DEFAULT_CLASSIC_PATH,
    "no_odds": DEFAULT_NO_ODDS_PATH,
    "gradient_boosting": DEFAULT_GB_PATH,
}

RESULT_LABELS = ["A", "D", "H"]
PREDICTION_LABELS = ["H", "D", "A"]


def safe_brier_score(report: pd.DataFrame) -> float | None:
    """Compute a multiclass Brier score when the required probability columns exist."""
    required_columns = ["home_win_probability", "draw_probability", "away_win_probability", "actual_result"]
    if not all(column in report.columns for column in required_columns):
        return None

    total = 0.0
    for label, probability_column in [
        ("H", "home_win_probability"),
        ("D", "draw_probability"),
        ("A", "away_win_probability"),
    ]:
        actual = (report["actual_result"] == label).astype(float)
        total += ((report[probability_column] - actual) ** 2).mean()
    return float(total / 3)


def safe_log_loss_1n2(report: pd.DataFrame) -> float | None:
    """Compute 1N2 log loss using the normalized match probabilities."""
    required_columns = ["home_win_probability", "draw_probability", "away_win_probability", "actual_result"]
    if not all(column in report.columns for column in required_columns):
        return None

    probabilities = report[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)
    return float(log_loss(report["actual_result"], probabilities, labels=RESULT_LABELS))


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available column from a candidate list."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def resolve_prediction_column(report: pd.DataFrame) -> str:
    """Return the best available recommended prediction column."""
    return first_existing_column(
        report,
        [
            "recommended_prediction_class",
            "predicted_class_adjusted",
            "predicted_class_argmax",
            "predicted_class",
        ],
    ) or "predicted_class"


def resolve_argmax_column(report: pd.DataFrame) -> str:
    """Return the best available argmax prediction column."""
    return first_existing_column(report, ["predicted_class_argmax", "predicted_class"]) or "predicted_class"


def resolve_adjusted_column(report: pd.DataFrame) -> str | None:
    """Return the adjusted prediction column when available."""
    return first_existing_column(report, ["predicted_class_adjusted"])


def resolved_metric_value(report: pd.DataFrame, column: str | None) -> float | None:
    """Return a mean metric value when the column exists."""
    if column is None or column not in report.columns:
        return None
    series = report[column]
    if series.empty:
        return None
    if series.dtype == bool:
        return float(series.mean())
    try:
        return float(series.mean())
    except Exception:
        return None


def classification_accuracy(report: pd.DataFrame, prediction_column: str) -> float | None:
    """Compute accuracy for a prediction column when present."""
    if prediction_column not in report.columns or "actual_result" not in report.columns:
        return None
    return float(accuracy_score(report["actual_result"], report[prediction_column]))


def recommendation_distribution(report: pd.DataFrame, recommendation_column: str) -> dict[str, int]:
    """Return H/D/A distribution for a recommendation column."""
    if recommendation_column not in report.columns:
        return {label: 0 for label in PREDICTION_LABELS}
    counts = report[recommendation_column].value_counts()
    return {label: int(counts.get(label, 0)) for label in PREDICTION_LABELS}


def draw_rate_for_subset(report: pd.DataFrame, mask: pd.Series) -> float | None:
    """Return the actual draw rate for a row subset."""
    subset = report.loc[mask]
    if subset.empty:
        return None
    return float((subset["actual_result"] == "D").mean())


def engine_summary_row(engine: str, report: pd.DataFrame) -> dict[str, Any]:
    """Build one comparison row per engine."""
    argmax_column = resolve_argmax_column(report)
    adjusted_column = resolve_adjusted_column(report)
    recommendation_column = resolve_prediction_column(report)
    recommendation_dist = recommendation_distribution(report, recommendation_column)
    d_recommended_mask = report[recommendation_column] == "D" if recommendation_column in report.columns else pd.Series(False, index=report.index)
    draw_warning_mask = report["draw_warning"].astype(bool) if "draw_warning" in report.columns else pd.Series(False, index=report.index)

    row = {
        "engine": engine,
        "matches": int(len(report)),
        "accuracy_argmax": classification_accuracy(report, argmax_column),
        "accuracy_adjusted": classification_accuracy(report, adjusted_column) if adjusted_column else None,
        "accuracy_recommended": classification_accuracy(report, recommendation_column),
        "log_loss_1N2": safe_log_loss_1n2(report),
        "brier_score_multiclass": safe_brier_score(report),
        "recommended_prediction_class_H": recommendation_dist["H"],
        "recommended_prediction_class_D": recommendation_dist["D"],
        "recommended_prediction_class_A": recommendation_dist["A"],
        "d_recommended": recommendation_dist["D"],
        "actual_draw_rate_among_d_recommended": draw_rate_for_subset(report, d_recommended_mask),
        "draw_warning_count": int(draw_warning_mask.sum()) if "draw_warning" in report.columns else None,
        "actual_draw_rate_on_draw_warning": draw_rate_for_subset(report, draw_warning_mask),
        "confidence_score_mean": resolved_metric_value(report, "confidence_score"),
    }
    return row


def engine_by_league_rows(engine: str, report: pd.DataFrame) -> list[dict[str, Any]]:
    """Build league-level comparison rows for one engine when league is available."""
    if "league" not in report.columns:
        return []

    rows: list[dict[str, Any]] = []
    recommendation_column = resolve_prediction_column(report)
    for league, league_report in report.dropna(subset=["league"]).groupby("league", dropna=False):
        recommendation_dist = recommendation_distribution(league_report, recommendation_column)
        d_recommended_mask = league_report[recommendation_column] == "D" if recommendation_column in league_report.columns else pd.Series(False, index=league_report.index)
        rows.append(
            {
                "engine": engine,
                "league": league,
                "matches": int(len(league_report)),
                "accuracy_recommended": classification_accuracy(league_report, recommendation_column),
                "log_loss_1N2": safe_log_loss_1n2(league_report),
                "brier_score_multiclass": safe_brier_score(league_report),
                "d_recommended": recommendation_dist["D"],
                "actual_draw_rate_among_d_recommended": draw_rate_for_subset(league_report, d_recommended_mask),
            }
        )
    return rows


def load_report(path: Path, engine_name: str) -> pd.DataFrame:
    """Load one engine backtest report with a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest file for {engine_name}: {path}")
    report = pd.read_csv(path)
    if "actual_result" not in report.columns:
        raise ValueError(f"Backtest file for {engine_name} has no actual_result column: {path}")
    return report


def compare_engines(
    classic_path: Path = DEFAULT_CLASSIC_PATH,
    no_odds_path: Path = DEFAULT_NO_ODDS_PATH,
    gb_path: Path = DEFAULT_GB_PATH,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    by_league_output: Path = DEFAULT_BY_LEAGUE_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the available engines and export summary CSVs."""
    path_map = {
        "classic": classic_path,
        "no_odds": no_odds_path,
        "gradient_boosting": gb_path,
    }

    summary_rows: list[dict[str, Any]] = []
    league_rows: list[dict[str, Any]] = []

    for engine, path in path_map.items():
        report = load_report(path, engine)
        summary_rows.append(engine_summary_row(engine, report))
        league_rows.extend(engine_by_league_rows(engine, report))

    summary = pd.DataFrame(summary_rows)
    by_league = pd.DataFrame(league_rows)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    by_league.to_csv(by_league_output, index=False)

    print("Engine comparison summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}" if pd.notna(value) else ""))

    if not by_league.empty:
        print("\nEngine comparison by league")
        print(by_league.to_string(index=False, float_format=lambda value: f"{value:.4f}" if pd.notna(value) else ""))
    else:
        print("\nNo league-specific comparison available.")

    print(f"\nSaved summary to: {summary_output}")
    print(f"Saved league analysis to: {by_league_output}")
    return summary, by_league


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare the available probability engines.")
    parser.add_argument("--classic", type=Path, default=DEFAULT_CLASSIC_PATH)
    parser.add_argument("--no-odds", dest="no_odds", type=Path, default=DEFAULT_NO_ODDS_PATH)
    parser.add_argument("--gb", type=Path, default=DEFAULT_GB_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--by-league-output", type=Path, default=DEFAULT_BY_LEAGUE_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the engine comparison from the command line."""
    args = parse_args()
    compare_engines(
        classic_path=args.classic,
        no_odds_path=args.no_odds,
        gb_path=args.gb,
        summary_output=args.summary_output,
        by_league_output=args.by_league_output,
    )


if __name__ == "__main__":
    main()
