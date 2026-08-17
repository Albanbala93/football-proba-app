"""Analyze renamed local Understat exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "understat" / "Understat"
DEFAULT_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "understat_exports_summary.csv"
DEFAULT_SITUATIONAL_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "understat_situational_stats_summary.csv"

XG_COLUMNS = {"xG", "xGA", "xPTS", "xA", "xG90", "xA90", "xG_diff", "xGperSh", "xGAperSh"}
SITUATIONAL_COLUMNS = ["shots", "goals", "shots_against", "goals_against", "xG", "xGA", "xG_diff"]
LEAGUE_TABLE_COLUMNS = ["team", "xG", "xGA", "xPTS"]


def _detect_file_type(file_path: Path) -> str:
    """Infer file type from the normalized Understat export filename."""
    name = file_path.name.lower()
    if "league_table" in name:
        return "league_table"
    if "player_stats" in name:
        return "player_stats"
    if "team_situational_stats" in name:
        return "team_situational_stats"
    return "unknown"


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns present in the DataFrame, preserving requested order."""
    return [column for column in columns if column in df.columns]


def _xg_columns(df: pd.DataFrame) -> list[str]:
    """Return xG-related columns present in the DataFrame."""
    return [column for column in df.columns if str(column) in XG_COLUMNS]


def _statistic_preview(df: pd.DataFrame) -> str:
    """Return up to five first values from statistic column."""
    if "statistic" not in df.columns:
        return ""
    values = df["statistic"].dropna().astype(str).str.strip()
    values = values[values != ""].head(5).tolist()
    return " | ".join(values)


def _league_table_console(file_path: Path, df: pd.DataFrame) -> None:
    """Print useful league table columns."""
    columns = _existing_columns(df, LEAGUE_TABLE_COLUMNS)
    if not columns:
        return
    print("\nLeague table preview:")
    print(df[columns].to_string(index=False))


def _situational_summary_rows(file_path: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build top situational statistic rows for one file."""
    if "statistic" not in df.columns:
        return []

    columns = ["statistic", *_existing_columns(df, SITUATIONAL_COLUMNS)]
    summary = df[columns].head(20).copy()
    rows = []
    for rank, (_, row) in enumerate(summary.iterrows(), start=1):
        output_row: dict[str, Any] = {
            "file_path": str(file_path.relative_to(PROJECT_ROOT)),
            "file_name": file_path.name,
            "rank": rank,
            "statistic": row.get("statistic"),
        }
        for column in SITUATIONAL_COLUMNS:
            output_row[column] = row.get(column) if column in summary.columns else pd.NA
        rows.append(output_row)
    return rows


def _situational_console(file_path: Path, df: pd.DataFrame) -> None:
    """Print top situational statistics."""
    columns = ["statistic", *_existing_columns(df, SITUATIONAL_COLUMNS)]
    if "statistic" not in columns:
        return
    print("\nTop situational statistics:")
    print(df[columns].head(20).to_string(index=False))


def analyze_understat_exports(
    input_dir: Path = DEFAULT_INPUT_DIR,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    situational_output_path: Path = DEFAULT_SITUATIONAL_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze local renamed Understat exports and write summary files."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.csv"))
    summary_rows: list[dict[str, Any]] = []
    situational_rows: list[dict[str, Any]] = []

    print("Understat exports analysis")
    print(f"Input directory: {input_dir.relative_to(PROJECT_ROOT)}")
    print(f"CSV files found: {len(files)}")

    for file_path in files:
        df = pd.read_csv(file_path, sep=";")
        file_type = _detect_file_type(file_path)
        xg_cols = _xg_columns(df)
        statistic_preview = _statistic_preview(df)

        summary_rows.append(
            {
                "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                "file_name": file_path.name,
                "file_type": file_type,
                "rows": len(df),
                "columns_count": len(df.columns),
                "columns": "; ".join(map(str, df.columns)),
                "xg_columns": "; ".join(xg_cols),
                "statistic_preview": statistic_preview,
            }
        )

        print("\n" + "=" * 80)
        print(f"File: {file_path.name}")
        print(f"Type: {file_type}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {', '.join(map(str, df.columns))}")
        print(f"xG columns: {', '.join(xg_cols) if xg_cols else 'none'}")
        if statistic_preview:
            print(f"Statistic preview: {statistic_preview}")

        if file_type == "league_table":
            _league_table_console(file_path, df)
        elif file_type == "team_situational_stats":
            _situational_console(file_path, df)
            situational_rows.extend(_situational_summary_rows(file_path, df))

    summary = pd.DataFrame(summary_rows)
    situational_summary = pd.DataFrame(
        situational_rows,
        columns=[
            "file_path",
            "file_name",
            "rank",
            "statistic",
            "shots",
            "goals",
            "shots_against",
            "goals_against",
            "xG",
            "xGA",
            "xG_diff",
        ],
    )

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output_path, index=False)
    situational_summary.to_csv(situational_output_path, index=False)

    print("\n" + "=" * 80)
    print("Files by type:")
    if summary.empty:
        print("No CSV files found.")
    else:
        print(summary["file_type"].value_counts().to_string())
    print(f"\nSaved summary to: {summary_output_path}")
    print(f"Saved situational stats summary to: {situational_output_path}")

    return summary, situational_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze renamed local Understat exports.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """Run Understat export analysis."""
    args = parse_args()
    try:
        analyze_understat_exports(input_dir=args.input_dir)
    except Exception as exc:
        print("Understat exports analysis failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
