from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"

LEAGUES = [
    ("Premier League", "E0", "premier_league"),
    ("Ligue 1", "F1", "ligue1"),
    ("Liga", "SP1", "laliga"),
    ("Serie A", "I1", "seriea"),
    ("Bundesliga", "D1", "bundesliga"),
]

SEASONS = [
    ("2019-2020", "2019_2020", "1920"),
    ("2020-2021", "2020_2021", "2021"),
    ("2021-2022", "2021_2022", "2122"),
    ("2022-2023", "2022_2023", "2223"),
    ("2023-2024", "2023_2024", "2324"),
    ("2024-2025", "2024_2025", "2425"),
    ("2025-2026", "2025_2026", "2526"),
    ("2026-2027", "2026_2027", "2627"),
]


def download_file(url: str, output_path: Path, force: bool) -> str:
    """Download one CSV unless it already exists and force is disabled."""
    if output_path.exists() and not force:
        return "SKIP"

    try:
        with urlopen(url, timeout=30) as response:
            content = response.read()
    except HTTPError as exc:
        return f"ERROR HTTP {exc.code}"
    except URLError as exc:
        return f"ERROR URL {exc.reason}"
    except TimeoutError:
        return "ERROR timeout"

    output_path.write_bytes(content)
    return "OK"


def run_downloads(force: bool = False, current_season_only: bool = False) -> int:
    """Download Big 5 CSV files and return a process exit code."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    seasons = [SEASONS[-1]] if current_season_only else SEASONS
    force = force or current_season_only

    errors = 0
    ok_count = 0
    skip_count = 0

    for league_label, league_code, league_name in LEAGUES:
        for season_label, season_file_label, season_code in seasons:
            url = URL_TEMPLATE.format(season_code=season_code, league_code=league_code)
            output_path = RAW_DATA_DIR / f"{league_name}_{season_file_label}.csv"
            status = download_file(url, output_path, force)

            if status == "OK":
                ok_count += 1
            elif status == "SKIP":
                skip_count += 1
            else:
                errors += 1

            print(f"{status} - {league_label} {season_label} -> {output_path.relative_to(PROJECT_ROOT)}")

    print()
    print(f"OK: {ok_count}")
    print(f"SKIP: {skip_count}")
    print(f"ERROR: {errors}")
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download Football-Data CSV files for Europe's Big 5 leagues.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when they already exist.")
    parser.add_argument(
        "--current-season-only",
        action="store_true",
        help="Only (re)download the latest configured season, forcing overwrite. Used for scheduled refreshes.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Big 5 data downloader."""
    args = parse_args()
    return run_downloads(force=args.force, current_season_only=args.current_season_only)


if __name__ == "__main__":
    raise SystemExit(main())
