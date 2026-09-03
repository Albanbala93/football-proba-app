"""Compare current Ligue 1 engine backtest with the experimental API xG engine."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREMATCH_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_prematch_features.csv"
DEFAULT_API_XG_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_model_predictions.csv"
DEFAULT_CURRENT_BACKTEST_PATH = PROJECT_ROOT / "data" / "predictions" / "probability_engine_backtest.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_vs_current_comparison.csv"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]
CALIBRATION_BUCKETS = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]


def normalize_team_name(value: Any) -> str:
    """Normalize team names for cross-source matching."""
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "", text)
    aliases = {
        "stadebrestois29": "brest",
        "parissaintgermain": "psg",
        "parissg": "psg",
        "stetienne": "saintetienne",
        "parisfc": "parisfc",
        "lehavre": "lehavre",
    }
    return aliases.get(text, text)


def add_match_key(df: pd.DataFrame, date_column: str, home_column: str, away_column: str) -> pd.DataFrame:
    """Add normalized match key columns."""
    keyed = df.copy()
    keyed["match_date_key"] = pd.to_datetime(keyed[date_column], errors="coerce").dt.date.astype(str)
    keyed["home_team_key"] = keyed[home_column].map(normalize_team_name)
    keyed["away_team_key"] = keyed[away_column].map(normalize_team_name)
    return keyed


def multiclass_brier_score(df: pd.DataFrame) -> float:
    """Compute mean multiclass Brier score for H/D/A probabilities."""
    total = 0.0
    for label, probability_column in [
        ("H", "home_win_probability"),
        ("D", "draw_probability"),
        ("A", "away_win_probability"),
    ]:
        actual = (df["actual_result"] == label).astype(float)
        total += ((df[probability_column] - actual) ** 2).mean()
    return float(total / len(RESULT_LABELS))


def probabilities_for_log_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Return A/D/H probabilities for sklearn log_loss."""
    probabilities = df[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    return probabilities.div(probabilities.sum(axis=1), axis=0)


def distribution(prefix: str, values: pd.Series) -> dict[str, int]:
    """Return H/D/A distribution columns."""
    counts = values.value_counts().reindex(RESULT_LABELS, fill_value=0)
    return {f"{prefix}_{label}": int(counts[label]) for label in RESULT_LABELS}


def prediction_column(df: pd.DataFrame) -> str:
    """Return the best available prediction class column."""
    for column in ["recommended_prediction_class", "predicted_class_argmax", "predicted_class"]:
        if column in df.columns:
            return column
    raise ValueError("No prediction class column found.")


def metric_row(model_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Build one summary metric row."""
    pred_column = prediction_column(df)
    row: dict[str, Any] = {
        "row_type": "summary",
        "model": model_name,
        "matches": int(len(df)),
        "accuracy": float(accuracy_score(df["actual_result"], df[pred_column])) if len(df) else pd.NA,
        "log_loss": (
            float(log_loss(df["actual_result"], probabilities_for_log_loss(df), labels=LOG_LOSS_LABELS))
            if len(df)
            else pd.NA
        ),
        "brier_score_multiclass": multiclass_brier_score(df) if len(df) else pd.NA,
        "mapped_matches": int(len(df)),
    }
    row.update(distribution("actual_result", df["actual_result"]) if len(df) else {})
    row.update(distribution("predicted_result", df[pred_column]) if len(df) else {})
    return row


def calibration_rows(model_name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build simple confidence calibration rows."""
    if df.empty:
        return []
    pred_column = prediction_column(df)
    probability_by_prediction = {
        "H": df["home_win_probability"],
        "D": df["draw_probability"],
        "A": df["away_win_probability"],
    }
    confidence = pd.Series(index=df.index, dtype=float)
    for label, probability in probability_by_prediction.items():
        confidence.loc[df[pred_column] == label] = probability.loc[df[pred_column] == label]

    rows = []
    buckets = pd.cut(confidence, bins=CALIBRATION_BUCKETS, include_lowest=True, right=False)
    for bucket, bucket_df in df.groupby(buckets, observed=False):
        if bucket_df.empty:
            continue
        bucket_confidence = confidence.loc[bucket_df.index]
        rows.append(
            {
                "row_type": "calibration",
                "model": model_name,
                "bucket": str(bucket),
                "matches": int(len(bucket_df)),
                "avg_confidence": float(bucket_confidence.mean()),
                "observed_accuracy": float((bucket_df[pred_column] == bucket_df["actual_result"]).mean()),
                "calibration_gap": float(bucket_confidence.mean() - (bucket_df[pred_column] == bucket_df["actual_result"]).mean()),
            }
        )
    return rows


def confident_error_rows(model_name: str, df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    """Return most confident incorrect predictions."""
    if df.empty:
        return []
    pred_column = prediction_column(df)
    confidence = df[["home_win_probability", "draw_probability", "away_win_probability"]].max(axis=1)
    errors = df[df[pred_column] != df["actual_result"]].copy()
    errors["confidence"] = confidence.loc[errors.index]
    errors = errors.sort_values("confidence", ascending=False).head(limit)
    rows = []
    for _, row in errors.iterrows():
        rows.append(
            {
                "row_type": "confident_error",
                "model": model_name,
                "date": row.get("date"),
                "home_team": row.get("home_team_name", row.get("home_team")),
                "away_team": row.get("away_team_name", row.get("away_team")),
                "actual_result": row["actual_result"],
                "predicted_result": row[pred_column],
                "confidence": row["confidence"],
                "home_win_probability": row["home_win_probability"],
                "draw_probability": row["draw_probability"],
                "away_win_probability": row["away_win_probability"],
            }
        )
    return rows


def load_api_xg_predictions(path: Path) -> pd.DataFrame:
    """Load API xG predictions."""
    if not path.exists():
        raise FileNotFoundError(f"API xG predictions not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"predicted_class_argmax": "recommended_prediction_class"})
    return add_match_key(df, "date", "home_team_name", "away_team_name")


def load_current_predictions(path: Path) -> pd.DataFrame:
    """Load current engine Ligue 1 predictions when available."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "league" in df.columns:
        df = df[df["league"].astype(str) == "Ligue 1"].copy()
    if df.empty:
        return df
    return add_match_key(df, "date", "home_team", "away_team")


def compare_ligue1_current_vs_api_xg(
    prematch_path: Path = DEFAULT_PREMATCH_PATH,
    api_xg_path: Path = DEFAULT_API_XG_PATH,
    current_backtest_path: Path = DEFAULT_CURRENT_BACKTEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Compare API xG and current engine predictions on mapped matches."""
    if not prematch_path.exists():
        raise FileNotFoundError(f"Prematch feature file not found: {prematch_path}")

    api_xg = load_api_xg_predictions(api_xg_path)
    current = load_current_predictions(current_backtest_path)

    rows: list[dict[str, Any]] = []
    rows.append(metric_row("baseline_api_xg", api_xg))
    rows.extend(calibration_rows("baseline_api_xg", api_xg))
    rows.extend(confident_error_rows("baseline_api_xg", api_xg))

    mapped_current = pd.DataFrame()
    if not current.empty:
        keys = ["match_date_key", "home_team_key", "away_team_key"]
        mapped_keys = api_xg[keys].drop_duplicates()
        mapped_current = current.merge(mapped_keys, on=keys, how="inner")
        if not mapped_current.empty:
            mapped_api_xg = api_xg.merge(mapped_current[keys].drop_duplicates(), on=keys, how="inner")
            rows.append(metric_row("baseline_api_xg_mapped", mapped_api_xg))
            rows.extend(calibration_rows("baseline_api_xg_mapped", mapped_api_xg))
            rows.extend(confident_error_rows("baseline_api_xg_mapped", mapped_api_xg))
            rows.append(metric_row("current_engine_mapped", mapped_current))
            rows.extend(calibration_rows("current_engine_mapped", mapped_current))
            rows.extend(confident_error_rows("current_engine_mapped", mapped_current))
        else:
            rows.append(
                {
                    "row_type": "summary",
                    "model": "current_engine_mapped",
                    "matches": 0,
                    "mapping_status": "no common matches by date/home/away",
                }
            )
    else:
        rows.append(
            {
                "row_type": "summary",
                "model": "current_engine_mapped",
                "matches": 0,
                "mapping_status": "current backtest unavailable or empty",
            }
        )

    comparison = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    print_summary(api_xg, current, mapped_current, comparison, output_path)
    return comparison


def print_summary(
    api_xg: pd.DataFrame,
    current: pd.DataFrame,
    mapped_current: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Print a readable comparison summary."""
    print("Ligue 1 current vs API xG comparison")
    print(f"API xG matches: {len(api_xg)}")
    print(f"Current Ligue 1 backtest matches: {len(current)}")
    print(f"Mapped current matches: {len(mapped_current)}")
    print("\nSummary rows:")
    summary = comparison[comparison["row_type"] == "summary"].copy()
    columns = [column for column in ["model", "matches", "accuracy", "log_loss", "brier_score_multiclass", "mapping_status"] if column in summary.columns]
    print(summary[columns].to_string(index=False))
    print(f"\nSaved comparison to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare current Ligue 1 engine and API xG experiment.")
    parser.add_argument("--prematch", type=Path, default=DEFAULT_PREMATCH_PATH)
    parser.add_argument("--api-xg", type=Path, default=DEFAULT_API_XG_PATH)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT_BACKTEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the comparison."""
    args = parse_args()
    compare_ligue1_current_vs_api_xg(
        prematch_path=args.prematch,
        api_xg_path=args.api_xg,
        current_backtest_path=args.current,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
