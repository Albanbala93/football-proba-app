"""Generate the current team reference table from the latest raw season files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "current_teams.csv"

LEAGUE_NAMES = {
    "premier_league": "Premier League",
    "ligue1": "Ligue 1",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
}
EXPECTED_TEAM_COUNTS = {
    "Premier League": 20,
    "Ligue 1": 18,
    "La Liga": 20,
    "Serie A": 20,
    "Bundesliga": 18,
}
RAW_FILE_PATTERN = re.compile(
    r"^(premier_league|ligue1|laliga|seriea|bundesliga)_(\d{4})_(\d{4})\.csv$"
)


def _require_columns(df: pd.DataFrame, columns: list[str], source_file: Path) -> None:
    """Raise a clear error if a raw file is missing required columns."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_file.name}: {', '.join(missing)}")


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


def _iter_latest(latest_files: dict[str, tuple[int, int, str, Path]]) -> list[tuple[str, int, int, str, Path]]:
    """Return latest file metadata in configured league order."""
    return [
        (league_key, *latest_files[league_key])
        for league_key in LEAGUE_NAMES
        if league_key in latest_files
    ]


def _team_list(raw_file: Path) -> list[str]:
    """Read unique non-empty teams from HomeTeam and AwayTeam columns."""
    df = pd.read_csv(raw_file)
    _require_columns(df, ["HomeTeam", "AwayTeam"], raw_file)

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True)
    teams = teams.dropna().astype(str).str.strip()
    teams = teams[teams != ""]
    return sorted(teams.unique())


def generate_current_teams(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate and export the current team reference table."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    latest_files = _find_latest_files(raw_dir)
    rows = []
    summaries = []

    for league_key, league_name in LEAGUE_NAMES.items():
        season, source_file = latest_files[league_key]
        teams = _team_list(source_file)

        for team in teams:
            rows.append(
                {
                    "league": league_name,
                    "season": season,
                    "team": team,
                    "source_file": source_file.name,
                }
            )

        summaries.append(
            {
                "league": league_name,
                "season": season,
                "team_count": len(teams),
                "teams": teams,
            }
        )

    current_teams = pd.DataFrame(rows, columns=["league", "season", "team", "source_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_teams.to_csv(output_path, index=False)

    print_current_teams_summary(summaries, raw_dir, output_path)
    return current_teams


def print_current_teams_summary(summaries: list[dict[str, object]], raw_dir: Path, output_path: Path) -> None:
    """Print a readable summary and team-count warnings."""
    print("Current teams reference")
    print(f"Raw directory: {raw_dir}")

    warnings = []
    for summary in summaries:
        league = str(summary["league"])
        season = str(summary["season"])
        teams = list(summary["teams"])
        team_count = int(summary["team_count"])
        expected_count = EXPECTED_TEAM_COUNTS[league]

        print(f"\n{league} ({season})")
        print(f"Teams: {team_count}")
        print(", ".join(teams))

        if team_count != expected_count:
            warnings.append(f"{league} expected {expected_count} teams, found {team_count}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"WARNING: {warning}")

    print(f"\nSaved current teams reference to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate current teams from latest raw season files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run current team reference generation from the command line."""
    args = parse_args()
    generate_current_teams(raw_dir=args.raw_dir, output_path=args.output)


if __name__ == "__main__":
    main()
