"""Generate upcoming fixture references from the latest raw Football-Data CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "upcoming_fixtures.csv"

LEAGUE_NAMES = {
    "premier_league": "Premier League",
    "ligue1": "Ligue 1",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
}
RAW_FILE_PATTERN = re.compile(
    r"^(premier_league|ligue1|laliga|seriea|bundesliga)_(\d{4})_(\d{4})\.csv$"
)
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]
RESULT_COLUMNS = ["FTR", "FTHG", "FTAG"]
ODDS_COLUMNS = {
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}
OUTPUT_COLUMNS = [
    "league",
    "season",
    "date",
    "home_team",
    "away_team",
    "odds_home",
    "odds_draw",
    "odds_away",
    "source_file",
]


def _require_columns(df: pd.DataFrame, columns: list[str], source_file: Path) -> None:
    """Raise a clear error if a raw file is missing required columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_file.name}: {', '.join(missing)}")


def _iter_latest(latest_files: dict[str, tuple[int, int, str, Path]]) -> list[tuple[str, int, int, str, Path]]:
    """Return latest file metadata in configured league order."""
    return [
        (league_key, *latest_files[league_key])
        for league_key in LEAGUE_NAMES
        if league_key in latest_files
    ]


def _find_latest_files(raw_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return the latest available season file for each configured league."""
    latest_files: dict[str, tuple[int, int, str, Path]] = {}

    for csv_file in raw_dir.glob("*.csv"):
        match = RAW_FILE_PATTERN.match(csv_file.name)
        if match is None:
            continue

        league_key, start_year_raw, end_year_raw = match.groups()
        start_year = int(start_year_raw)
        end_year = int(end_year_raw)
        season = f"{start_year_raw}-{end_year_raw}"
        current = latest_files.get(league_key)
        if current is None or (start_year, end_year) > (current[0], current[1]):
            latest_files[league_key] = (start_year, end_year, season, csv_file)

    missing_leagues = [league_key for league_key in LEAGUE_NAMES if league_key not in latest_files]
    if missing_leagues:
        raise FileNotFoundError(f"No raw season file found for: {', '.join(missing_leagues)}")

    return {league_key: (season, csv_file) for league_key, _, _, season, csv_file in _iter_latest(latest_files)}


def _is_missing_result_value(series: pd.Series) -> pd.Series:
    """Return a mask for missing or empty result values."""
    return series.isna() | series.astype(str).str.strip().eq("")


def _parse_football_data_dates(date_series: pd.Series) -> pd.Series:
    """Parse Football-Data date values, which are generally day-first."""
    parsed_dates = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    unresolved = parsed_dates.isna() & date_series.notna() & date_series.astype(str).str.strip().ne("")
    if unresolved.any():
        parsed_dates.loc[unresolved] = pd.to_datetime(date_series.loc[unresolved], errors="coerce")
    return parsed_dates


def _upcoming_fixtures_for_file(league_key: str, season: str, raw_file: Path) -> pd.DataFrame:
    """Extract upcoming fixtures from one latest-season raw file."""
    df = pd.read_csv(raw_file)
    _require_columns(df, REQUIRED_COLUMNS, raw_file)

    filtered = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    filtered["HomeTeam"] = filtered["HomeTeam"].astype(str).str.strip()
    filtered["AwayTeam"] = filtered["AwayTeam"].astype(str).str.strip()
    filtered = filtered[(filtered["HomeTeam"] != "") & (filtered["AwayTeam"] != "")]

    missing_result = pd.Series(False, index=filtered.index)
    for column in RESULT_COLUMNS:
        if column not in filtered.columns:
            missing_result = pd.Series(True, index=filtered.index)
            break
        missing_result = missing_result | _is_missing_result_value(filtered[column])

    fixtures = filtered[missing_result].copy()
    if fixtures.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    fixtures["_parsed_date"] = _parse_football_data_dates(fixtures["Date"])

    rows = pd.DataFrame(
        {
            "league": LEAGUE_NAMES[league_key],
            "season": season,
            "date": fixtures["_parsed_date"].dt.strftime("%Y-%m-%d"),
            "home_team": fixtures["HomeTeam"],
            "away_team": fixtures["AwayTeam"],
            "source_file": raw_file.name,
        }
    )
    for raw_column, output_column in ODDS_COLUMNS.items():
        if raw_column in fixtures.columns:
            rows[output_column] = pd.to_numeric(fixtures[raw_column], errors="coerce")
        else:
            rows[output_column] = pd.NA

    return rows[OUTPUT_COLUMNS]


def generate_upcoming_fixtures(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate and export upcoming fixtures from latest raw season files."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    latest_files = _find_latest_files(raw_dir)
    fixture_tables = []
    for league_key in LEAGUE_NAMES:
        season, raw_file = latest_files[league_key]
        fixture_tables.append(_upcoming_fixtures_for_file(league_key, season, raw_file))

    if fixture_tables:
        fixtures = pd.concat(fixture_tables, ignore_index=True)
    else:
        fixtures = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not fixtures.empty:
        fixtures["_sort_date"] = pd.to_datetime(fixtures["date"], errors="coerce")
        fixtures = fixtures.sort_values(["_sort_date", "league", "home_team"]).drop(columns="_sort_date")
        fixtures = fixtures.reset_index(drop=True)

    fixtures = fixtures.reindex(columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)

    print_upcoming_fixtures_summary(fixtures, raw_dir, output_path)
    return fixtures


def print_upcoming_fixtures_summary(fixtures: pd.DataFrame, raw_dir: Path, output_path: Path) -> None:
    """Print a compact upcoming fixture summary."""
    print("Upcoming fixtures reference")
    print(f"Raw directory: {raw_dir}")
    print(f"Total upcoming fixtures: {len(fixtures)}")

    if fixtures.empty:
        print("Aucune fixture future trouvée dans les CSV actuels.")
        print(f"\nSaved upcoming fixtures reference to: {output_path}")
        return

    print("\nFixtures by league:")
    print(fixtures["league"].value_counts().sort_index().to_string())
    print(f"\nDate min: {fixtures['date'].min()}")
    print(f"Date max: {fixtures['date'].max()}")
    print("\nNext 10 fixtures:")
    print(fixtures.head(10).to_string(index=False))
    print(f"\nSaved upcoming fixtures reference to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate upcoming fixtures from latest raw season files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run upcoming fixture reference generation from the command line."""
    args = parse_args()
    generate_upcoming_fixtures(raw_dir=args.raw_dir, output_path=args.output)


if __name__ == "__main__":
    main()
