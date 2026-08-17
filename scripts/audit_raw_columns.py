"""Audit available columns in raw Football-Data CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "raw_columns_audit.csv"
CANDIDATE_COLUMNS = {
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365H",
    "B365D",
    "B365A",
}


def is_candidate_column(column: str) -> bool:
    """Return whether a raw column is a candidate for model enrichment."""
    normalized = column.strip()
    return normalized in CANDIDATE_COLUMNS or "xg" in normalized.lower()


def audit_raw_columns(raw_dir: Path = DEFAULT_RAW_DIR, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Inspect raw CSV files and export a column presence audit."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    file_columns: dict[str, list[str]] = {}
    file_row_counts: dict[str, int] = {}

    print("Raw columns audit")
    print(f"Raw directory: {raw_dir}")
    print(f"CSV files: {len(csv_files)}")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        columns = list(df.columns)
        file_columns[csv_file.name] = columns
        file_row_counts[csv_file.name] = len(df)

        print(f"\n{csv_file.name}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(columns)}")
        print(", ".join(columns))

    rows = []
    total_files = len(csv_files)
    all_columns = sorted({column for columns in file_columns.values() for column in columns})
    for column in all_columns:
        present_files = [file_name for file_name, columns in file_columns.items() if column in columns]
        rows.append(
            {
                "column": column,
                "files_present": len(present_files),
                "total_files": total_files,
                "presence_pct": len(present_files) / total_files * 100,
                "is_candidate_useful": is_candidate_column(column),
                "files": "; ".join(present_files),
            }
        )

    audit = pd.DataFrame(rows).sort_values(["is_candidate_useful", "files_present", "column"], ascending=[False, False, True])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)

    print("\nGlobal summary")
    print(f"Unique columns: {len(audit)}")
    print("\nColumn presence:")
    print(
        audit[["column", "files_present", "total_files", "presence_pct", "is_candidate_useful"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.1f}",
        )
    )

    candidate_audit = audit[audit["is_candidate_useful"]].copy()
    print("\nCandidate useful columns:")
    if candidate_audit.empty:
        print("None found")
    else:
        print(
            candidate_audit[["column", "files_present", "presence_pct"]].to_string(
                index=False,
                float_format=lambda value: f"{value:.1f}",
            )
        )

    print(f"\nSaved raw columns audit to: {output_path}")
    return audit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit available columns in data/raw CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the raw column audit from the command line."""
    args = parse_args()
    audit_raw_columns(raw_dir=args.raw_dir, output_path=args.output)


if __name__ == "__main__":
    main()
