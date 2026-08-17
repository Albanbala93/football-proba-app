"""Audit local Understat data availability for xG enrichment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "understat"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "understat_data_audit.json"
DEFAULT_CANDIDATES_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "understat_file_candidates.csv"
SUPPORTED_SUFFIXES = {".csv", ".json"}
CSV_SEPARATOR_CANDIDATES = [",", ";", "\t"]
XG_COLUMN_HINTS = ["xg", "xga", "npxg", "xpts", "deep", "scored", "missed"]


def _xg_related_columns(columns: list[str]) -> list[str]:
    """Return columns that look related to xG or Understat useful signals."""
    related = []
    for column in columns:
        normalized = column.lower()
        if any(hint in normalized for hint in XG_COLUMN_HINTS):
            related.append(column)
    return related


def _first_matching_column(columns: list[str], candidates: set[str]) -> str | None:
    """Return first column whose normalized name matches a candidate."""
    for column in columns:
        normalized = column.lower().strip().replace(" ", "_")
        if normalized in candidates:
            return column
    return None


def _read_csv_with_detected_separator(file_path: Path) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Read a CSV using the separator that yields the most columns."""
    best_df = None
    best_separator = None
    best_error = None

    for separator in CSV_SEPARATOR_CANDIDATES:
        try:
            df = pd.read_csv(file_path, sep=separator)
        except Exception as exc:
            best_error = str(exc)
            continue

        if best_df is None or len(df.columns) > len(best_df.columns):
            best_df = df
            best_separator = separator

    if best_df is None:
        return None, None, best_error or "could not read CSV with supported separators"
    return best_df, best_separator, None


def _read_preview(file_path: Path) -> tuple[int | None, list[str], str | None, str | None]:
    """Return row count, columns, detected separator, and optional error."""
    try:
        if file_path.suffix.lower() == ".csv":
            df, detected_separator, error = _read_csv_with_detected_separator(file_path)
            if error:
                return None, [], detected_separator, error
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
            detected_separator = None
        else:
            return None, [], None, "unsupported file type"
    except Exception as exc:
        return None, [], None, str(exc)

    return len(df), [str(column) for column in df.columns], detected_separator, None


def _candidate_row(file_report: dict[str, Any]) -> dict[str, Any]:
    """Build one candidate CSV row with detected metadata fields."""
    columns = file_report["columns"]
    xg_columns = file_report["xg_columns_detected"]
    detected_date = _first_matching_column(columns, {"date", "match_date", "datetime"})
    detected_league = _first_matching_column(columns, {"league", "competition", "competition_name"})
    detected_season = _first_matching_column(columns, {"season", "year"})
    detected_team = _first_matching_column(columns, {"team", "team_name", "squad"})

    normalized_xg = {column.lower().strip() for column in xg_columns}
    relevance_score = 0
    relevance_score += int(detected_date is not None)
    relevance_score += int(detected_league is not None)
    relevance_score += int(detected_season is not None)
    relevance_score += int(detected_team is not None)
    relevance_score += int(any(column == "xg" for column in normalized_xg))
    relevance_score += int(any(column == "xga" for column in normalized_xg))
    relevance_score += int(any(column == "xpts" for column in normalized_xg))

    return {
        "file_path": file_report["path"],
        "detected_separator": file_report["detected_separator"],
        "rows": file_report["rows"],
        "columns_count": file_report["columns_count"],
        "relevance_score": relevance_score,
        "detected_date_column": detected_date,
        "detected_league_column": detected_league,
        "detected_season_column": detected_season,
        "detected_team_column": detected_team,
        "detected_xg_columns": "; ".join(xg_columns),
        "columns": "; ".join(columns),
    }


def audit_understat_data(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    candidates_output_path: Path = DEFAULT_CANDIDATES_OUTPUT_PATH,
) -> dict[str, Any]:
    """Audit local Understat files and export a JSON report."""
    files = []
    if input_dir.exists():
        files = sorted(
            file_path
            for file_path in input_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    file_reports = []
    for file_path in files:
        row_count, columns, detected_separator, error = _read_preview(file_path)
        xg_columns = _xg_related_columns(columns)
        file_reports.append(
            {
                "path": str(file_path.relative_to(PROJECT_ROOT)),
                "detected_separator": detected_separator,
                "rows": row_count,
                "columns_count": len(columns),
                "columns": columns,
                "xg_columns_detected": xg_columns,
                "error": error,
            }
        )

    recommendation = None
    if not files:
        recommendation = (
            "No local Understat CSV/JSON files found. Use a local Understat/Kaggle export, "
            "or create a dedicated scraping/API script later if authorized."
        )

    report = {
        "input_dir": str(input_dir.relative_to(PROJECT_ROOT)),
        "input_dir_exists": input_dir.exists(),
        "supported_file_count": len(files),
        "files": file_reports,
        "recommendation": recommendation,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    candidate_rows = [_candidate_row(file_report) for file_report in file_reports if not file_report["error"]]
    candidates = pd.DataFrame(candidate_rows)
    candidates_output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_output_path, index=False)

    print_summary(report, output_path, candidates_output_path)
    return report


def print_summary(report: dict[str, Any], output_path: Path, candidates_output_path: Path) -> None:
    """Print a readable audit summary."""
    print("Understat data audit")
    print(f"Input directory: {report['input_dir']}")
    print(f"Directory exists: {report['input_dir_exists']}")
    print(f"Supported CSV/JSON files: {report['supported_file_count']}")

    if report["recommendation"]:
        print("\nRecommendation:")
        print(report["recommendation"])
    else:
        print("\nFiles:")
        for file_report in report["files"]:
            print(f"\n{file_report['path']}")
            if file_report["error"]:
                print(f"Error: {file_report['error']}")
                continue
            print(f"Detected separator: {repr(file_report['detected_separator'])}")
            print(f"Rows: {file_report['rows']}")
            print(f"Columns count: {file_report['columns_count']}")
            related = file_report["xg_columns_detected"]
            print(f"xG columns detected: {', '.join(related) if related else 'none'}")
            print(f"Columns: {', '.join(file_report['columns'])}")

    print(f"\nSaved Understat audit to: {output_path}")
    print(f"Saved Understat file candidates to: {candidates_output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit local Understat CSV/JSON data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the Understat data audit."""
    args = parse_args()
    try:
        audit_understat_data(
            input_dir=args.input_dir,
            output_path=args.output,
            candidates_output_path=args.candidates_output,
        )
    except Exception as exc:
        print("Understat data audit failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
