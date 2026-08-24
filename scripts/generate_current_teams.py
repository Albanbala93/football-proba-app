"""Generate the current team reference table from the latest raw season files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "current_teams.csv"
MANUAL_OVERRIDES_PATH = PROJECT_ROOT / "data" / "reference" / "manual_current_teams.csv"

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


def _find_season_files(raw_dir: Path) -> dict[str, list[tuple[int, int, str, Path]]]:
    """Return every available season file per league, most recent first."""
    files_by_league: dict[str, list[tuple[int, int, str, Path]]] = {}

    for csv_file in raw_dir.glob("*.csv"):
        match = RAW_FILE_PATTERN.match(csv_file.name)
        if match is None:
            continue

        league_key, start_year_raw, end_year_raw = match.groups()
        season = f"{start_year_raw}-{end_year_raw}"
        entry = (int(start_year_raw), int(end_year_raw), season, csv_file)
        files_by_league.setdefault(league_key, []).append(entry)

    for entries in files_by_league.values():
        entries.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)

    missing_leagues = [league_key for league_key in LEAGUE_NAMES if league_key not in files_by_league]
    if missing_leagues:
        raise FileNotFoundError(f"No raw season file found for: {', '.join(missing_leagues)}")

    return files_by_league


def _find_latest_files(raw_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return the latest available season file for each configured league."""
    files_by_league = _find_season_files(raw_dir)
    latest_files = {league_key: entries[0] for league_key, entries in files_by_league.items()}
    return {league_key: (season, csv_file) for league_key, (_, _, season, csv_file) in latest_files.items()}


def _team_list(raw_file: Path) -> list[str]:
    """Read unique non-empty teams from HomeTeam and AwayTeam columns."""
    df = pd.read_csv(raw_file)
    _require_columns(df, ["HomeTeam", "AwayTeam"], raw_file)

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True)
    teams = teams.dropna().astype(str).str.strip()
    teams = teams[teams != ""]
    return sorted(teams.unique())


def _load_manual_overrides(path: Path = MANUAL_OVERRIDES_PATH) -> dict[str, tuple[str, list[str]]]:
    """Load hand-maintained league -> (season, teams) overrides, when present.

    Lets a real current-season roster (confirmed by hand, e.g. from the
    official league website) take priority over both the automated raw-file
    detection and the previous-season carry-over -- useful early in a season
    when neither Football-Data.co.uk nor an API-Football plan has the data
    yet. Keyed by league only (not by the raw files' detected season): if no
    raw file exists yet for the new season at all, the "latest" raw season
    would still read as the prior one, so a season match would never fire.
    When several seasons exist for one league in the override file, the most
    recent one wins. Remove or update entries here once real season data
    catches up.
    """
    if not path.exists():
        return {}
    overrides_df = pd.read_csv(path)
    _require_columns(overrides_df, ["league", "season", "team"], path)
    overrides: dict[str, tuple[str, list[str]]] = {}
    for league, group in overrides_df.groupby("league"):
        latest_season = sorted(group["season"].dropna().astype(str).unique())[-1]
        latest_group = group[group["season"].astype(str) == latest_season]
        teams = sorted(latest_group["team"].dropna().astype(str).str.strip().unique())
        overrides[str(league)] = (latest_season, teams)
    return overrides


def generate_current_teams(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate and export the current team reference table."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files_by_league = _find_season_files(raw_dir)
    manual_overrides = _load_manual_overrides()
    rows = []
    summaries = []

    for league_key, league_name in LEAGUE_NAMES.items():
        entries = files_by_league[league_key]
        _, _, season, source_file = entries[0]
        expected_count = EXPECTED_TEAM_COUNTS[league_name]
        manual_override = manual_overrides.get(league_name)

        if manual_override:
            season, teams = manual_override
            for team in teams:
                rows.append(
                    {
                        "league": league_name,
                        "season": season,
                        "team": team,
                        "source_file": f"{MANUAL_OVERRIDES_PATH.name} (renseigne a la main)",
                    }
                )
            summaries.append(
                {
                    "league": league_name,
                    "season": season,
                    "team_count": len(teams),
                    "teams": teams,
                    "carried_over_teams": [],
                    "carried_over_season": None,
                    "manual_override": True,
                }
            )
            continue

        teams = _team_list(source_file)
        carried_over_teams: list[str] = []
        carried_over_season: str | None = None

        if len(teams) < expected_count and len(entries) > 1:
            _, _, previous_season, previous_file = entries[1]
            previous_teams = _team_list(previous_file)
            carried_over_teams = sorted(set(previous_teams) - set(teams))
            carried_over_season = previous_season
            teams = sorted(set(teams) | set(carried_over_teams))

        for team in teams:
            rows.append(
                {
                    "league": league_name,
                    "season": season,
                    "team": team,
                    "source_file": (
                        f"{previous_file.name} (report suivant, pas encore joue en {season})"
                        if team in carried_over_teams
                        else source_file.name
                    ),
                }
            )

        summaries.append(
            {
                "league": league_name,
                "season": season,
                "team_count": len(teams),
                "teams": teams,
                "carried_over_teams": carried_over_teams,
                "carried_over_season": carried_over_season,
                "manual_override": False,
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

        if summary.get("manual_override"):
            print(f"Source : liste renseignee a la main dans {MANUAL_OVERRIDES_PATH.name}")

        carried_over_teams = list(summary["carried_over_teams"])
        carried_over_season = summary["carried_over_season"]
        if carried_over_teams:
            print(
                f"Completees avec {carried_over_season} (pas encore jouees en {season}): "
                + ", ".join(carried_over_teams)
            )

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
