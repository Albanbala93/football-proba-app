"""Analyze tactical features against match outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tactical_features.csv"
DEFAULT_HOME_FORMATION_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "tactical_outcomes_by_home_formation.csv"
DEFAULT_AWAY_FORMATION_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "tactical_outcomes_by_away_formation.csv"
DEFAULT_MATCHUP_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "tactical_outcomes_by_matchup.csv"
DEFAULT_STABILITY_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "tactical_outcomes_by_stability.csv"
DEFAULT_STRUCTURE_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "tactical_outcomes_by_structure.csv"

STABILITY_BUCKETS = [
    ("low_stability", 0.00, 0.33),
    ("medium_stability", 0.34, 0.66),
    ("high_stability", 0.67, 1.00),
]
STRUCTURE_COLUMNS = [
    "home_back_three",
    "home_back_four",
    "home_back_five",
    "home_two_strikers",
    "away_back_three",
    "away_back_four",
    "away_back_five",
    "away_two_strikers",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in tactical features: {', '.join(missing)}")


def _outcome_rates(df: pd.DataFrame) -> dict[str, float]:
    """Return H/D/A outcome rates for a grouped dataframe."""
    return {
        "home_win_rate": float((df["result"] == "H").mean()),
        "draw_rate": float((df["result"] == "D").mean()),
        "away_win_rate": float((df["result"] == "A").mean()),
    }


def _formation_outcomes(df: pd.DataFrame, formation_column: str) -> pd.DataFrame:
    """Analyze outcomes grouped by a home or away formation column."""
    rows = []
    for formation, group in df.dropna(subset=[formation_column]).groupby(formation_column, sort=True):
        row = {
            formation_column: formation,
            "matches": int(len(group)),
            **_outcome_rates(group),
            "avg_home_goals": float(group["home_score"].mean()),
            "avg_away_goals": float(group["away_score"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["matches", formation_column], ascending=[False, True])


def _matchup_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze outcomes grouped by formation matchup."""
    rows = []
    for matchup, group in df.dropna(subset=["formation_matchup"]).groupby("formation_matchup", sort=True):
        rows.append(
            {
                "formation_matchup": matchup,
                "matches": int(len(group)),
                **_outcome_rates(group),
                "avg_total_goals": float((group["home_score"] + group["away_score"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["matches", "formation_matchup"], ascending=[False, True])


def _stability_bucket(value: float | int | None) -> str | None:
    """Return low/medium/high stability bucket for a stability value."""
    if pd.isna(value):
        return None
    value = float(value)
    for label, lower, upper in STABILITY_BUCKETS:
        if lower <= value <= upper:
            return label
    return None


def _stability_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze outcomes by home and away stability buckets."""
    working = df.copy()
    working["home_stability_bucket"] = working["home_formation_stability_5"].map(_stability_bucket)
    working["away_stability_bucket"] = working["away_formation_stability_5"].map(_stability_bucket)

    rows = []
    grouped = working.dropna(subset=["home_stability_bucket", "away_stability_bucket"]).groupby(
        ["home_stability_bucket", "away_stability_bucket"],
        sort=True,
    )
    for (home_bucket, away_bucket), group in grouped:
        rows.append(
            {
                "home_stability_bucket": home_bucket,
                "away_stability_bucket": away_bucket,
                "matches": int(len(group)),
                **_outcome_rates(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["home_stability_bucket", "away_stability_bucket"])


def _structure_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze outcomes for each structural boolean feature."""
    rows = []
    for column in STRUCTURE_COLUMNS:
        for value, group in df.groupby(column, dropna=False, sort=True):
            rows.append(
                {
                    "structure_feature": column,
                    "feature_value": bool(value) if pd.notna(value) else None,
                    "matches": int(len(group)),
                    **_outcome_rates(group),
                    "avg_home_goals": float(group["home_score"].mean()),
                    "avg_away_goals": float(group["away_score"].mean()),
                }
            )
    return pd.DataFrame(rows)


def analyze_tactical_outcomes(
    input_path: Path = DEFAULT_INPUT_PATH,
    home_formation_output: Path = DEFAULT_HOME_FORMATION_OUTPUT,
    away_formation_output: Path = DEFAULT_AWAY_FORMATION_OUTPUT,
    matchup_output: Path = DEFAULT_MATCHUP_OUTPUT,
    stability_output: Path = DEFAULT_STABILITY_OUTPUT,
    structure_output: Path = DEFAULT_STRUCTURE_OUTPUT,
) -> dict[str, pd.DataFrame]:
    """Analyze tactical features against outcomes and export reports."""
    if not input_path.exists():
        raise FileNotFoundError(f"Tactical features file not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(
        df,
        [
            "home_formation",
            "away_formation",
            "formation_matchup",
            "home_formation_stability_5",
            "away_formation_stability_5",
            "home_score",
            "away_score",
            "result",
            *STRUCTURE_COLUMNS,
        ],
    )
    df = df.dropna(subset=["home_score", "away_score", "result"]).copy()
    df["result"] = df["result"].astype(str).str.upper().str.strip()
    df = df[df["result"].isin(["H", "D", "A"])].copy()

    reports = {
        "home_formation": _formation_outcomes(df, "home_formation"),
        "away_formation": _formation_outcomes(df, "away_formation"),
        "matchup": _matchup_outcomes(df),
        "stability": _stability_outcomes(df),
        "structure": _structure_outcomes(df),
    }

    for output_path, report in [
        (home_formation_output, reports["home_formation"]),
        (away_formation_output, reports["away_formation"]),
        (matchup_output, reports["matchup"]),
        (stability_output, reports["stability"]),
        (structure_output, reports["structure"]),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    print_summary(reports, input_path, len(df))
    print("\nExported files:")
    print(f"- {home_formation_output}")
    print(f"- {away_formation_output}")
    print(f"- {matchup_output}")
    print(f"- {stability_output}")
    print(f"- {structure_output}")
    return reports


def print_summary(reports: dict[str, pd.DataFrame], input_path: Path, matches: int) -> None:
    """Print a readable tactical outcomes summary."""
    print("Tactical outcomes analysis")
    print(f"Input: {input_path}")
    print(f"Matches with results: {matches}")

    home_candidates = reports["home_formation"][reports["home_formation"]["matches"] >= 5]
    print("\nHome formations with best home_win_rate (min 5 matches):")
    print(
        home_candidates.sort_values(["home_win_rate", "matches"], ascending=[False, False])
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    away_candidates = reports["away_formation"][reports["away_formation"]["matches"] >= 5]
    print("\nAway formations with best away_win_rate (min 5 matches):")
    print(
        away_candidates.sort_values(["away_win_rate", "matches"], ascending=[False, False])
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    matchup_candidates = reports["matchup"][reports["matchup"]["matches"] >= 3]
    print("\nMatchups with highest draw_rate (min 3 matches):")
    print(
        matchup_candidates.sort_values(["draw_rate", "matches"], ascending=[False, False])
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print("\nTactical stability comparison:")
    print(reports["stability"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze tactical outcomes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run tactical outcome analysis from the command line."""
    args = parse_args()
    analyze_tactical_outcomes(input_path=args.input)


if __name__ == "__main__":
    main()
