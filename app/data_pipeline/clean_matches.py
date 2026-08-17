"""Clean and concatenate raw historical football match CSV files.

Expected core raw columns:
Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR.

Optional betting odds columns:
B365H, B365D, B365A.

Optional match statistics columns:
HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR.

The default output is ``data/processed/matches_clean.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]
OPTIONAL_COLUMNS = [
    "B365H",
    "B365D",
    "B365A",
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
]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ["source_file", "league", "season"]
LEAGUE_NAME_BY_FILE_PREFIX = {
    "premier_league": "Premier League",
    "ligue1": "Ligue 1",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
}


def find_raw_csv_files(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    """Return all CSV files found in the raw data directory."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return csv_files


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when required match columns are missing."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {joined}")


def infer_season_from_filename(input_path: Path) -> str | None:
    """Infer a season label from a filename such as premier_league_2019_2020.csv."""
    match = re.search(r"(20\d{2})[_-](20\d{2})", input_path.stem)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def infer_league_from_filename(input_path: Path) -> str | None:
    """Infer league name from raw CSV filename prefix."""
    stem = input_path.stem.lower()
    for prefix, league in LEAGUE_NAME_BY_FILE_PREFIX.items():
        if stem.startswith(f"{prefix}_"):
            return league
    return None


def load_raw_matches(input_path: Path) -> pd.DataFrame:
    """Load and normalize one raw CSV file without dropping match rows yet."""
    df = pd.read_csv(input_path)
    validate_columns(df)

    # Keep a stable output schema while allowing odds columns to be absent.
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    loaded = df.loc[:, REQUIRED_COLUMNS + OPTIONAL_COLUMNS].copy()
    loaded["source_file"] = input_path.name
    loaded["league"] = infer_league_from_filename(input_path)
    loaded["season"] = infer_season_from_filename(input_path)
    return loaded


def clean_matches(input_paths: list[Path], output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load, concatenate, clean, and save historical match data.

    The cleaning step parses dates, normalizes team/result text, coerces goals
    and odds to numeric values, removes invalid rows, and sorts matches by date.
    """
    if not input_paths:
        raise FileNotFoundError("No input CSV files were provided.")

    loaded_frames = [load_raw_matches(input_path) for input_path in input_paths]
    cleaned = pd.concat(loaded_frames, ignore_index=True)
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce", dayfirst=True)

    text_columns = ["HomeTeam", "AwayTeam", "FTR"]
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    numeric_columns = ["FTHG", "FTAG"] + OPTIONAL_COLUMNS
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["FTR"] = cleaned["FTR"].str.upper()
    cleaned = cleaned[cleaned["FTR"].isin(["H", "D", "A"])]
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.loc[:, OUTPUT_COLUMNS]
    cleaned = cleaned.sort_values(["Date", "HomeTeam", "AwayTeam", "source_file"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def print_summary(input_paths: list[Path], cleaned: pd.DataFrame) -> None:
    """Print a concise terminal summary of the cleaning run."""
    print(f"CSV files read: {len(input_paths)}")
    print("Files:")
    for input_path in input_paths:
        print(f"- {input_path.name}")

    print(f"Total cleaned matches: {len(cleaned)}")
    if cleaned.empty:
        print("Covered period: no valid matches")
    else:
        print(f"Covered period: {cleaned['Date'].min().date()} to {cleaned['Date'].max().date()}")

    if "season" in cleaned.columns and cleaned["season"].notna().any():
        print("Matches per season:")
        season_counts = cleaned["season"].value_counts(dropna=False).sort_index()
        for season, count in season_counts.items():
            label = season if pd.notna(season) else "unknown"
            print(f"- {label}: {count}")

    if "league" in cleaned.columns and cleaned["league"].notna().any():
        print("Matches per league:")
        league_counts = cleaned["league"].value_counts(dropna=False).sort_index()
        for league, count in league_counts.items():
            label = league if pd.notna(league) else "unknown"
            print(f"- {label}: {count}")

    if {"league", "season"}.issubset(cleaned.columns):
        print("Matches per league and season:")
        grouped_counts = cleaned.groupby(["league", "season"], dropna=False).size().sort_index()
        for (league, season), count in grouped_counts.items():
            league_label = league if pd.notna(league) else "unknown"
            season_label = season if pd.notna(season) else "unknown"
            print(f"- {league_label} {season_label}: {count}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cleaning script."""
    parser = argparse.ArgumentParser(description="Clean historical football match CSV data.")
    parser.add_argument(
        "input_csvs",
        nargs="?",
        type=Path,
        help="Optional path to one raw CSV file. Defaults to all CSV files in data/raw/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path. Defaults to data/processed/matches_clean.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the match cleaning pipeline from the command line."""
    args = parse_args()
    input_paths = [args.input_csvs] if args.input_csvs else find_raw_csv_files()
    cleaned = clean_matches(input_paths=input_paths, output_path=args.output)
    print_summary(input_paths, cleaned)
    print(f"Saved cleaned matches to: {args.output}")


if __name__ == "__main__":
    main()
