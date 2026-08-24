"""Fetch current Big 5 team rosters from API-Football and map them onto
Football-Data.co.uk team names.

Why this exists: Football-Data.co.uk sometimes lags weeks behind the real
season start, so `generate_current_teams.py` can miss teams that have not
played yet (see scripts/generate_current_teams.py for the fallback that
handles that). API-Football's /teams endpoint returns the exact current
roster immediately, but uses different team names (e.g. "Paris Saint
Germain" instead of Football-Data.co.uk's "Paris SG"). Every downstream
feature (Elo, rolling form, historical lookups) is keyed on the
Football-Data.co.uk spelling, so an unmapped or mismapped name would make a
team silently impossible to predict (or worse, wrongly matched to another
team). This script maps aggressively but SKIPS anything it cannot confirm.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://v3.football.api-sports.io"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "current_teams.csv"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_SECONDS = 0.3

RAW_FILE_PATTERN = re.compile(r"^(premier_league|ligue1|laliga|seriea|bundesliga)_(\d{4})_(\d{4})\.csv$")

# API-Football v3 standard league IDs for Europe's Big 5.
LEAGUES = [
    ("premier_league", "Premier League", 39),
    ("ligue1", "Ligue 1", 61),
    ("laliga", "La Liga", 140),
    ("seriea", "Serie A", 135),
    ("bundesliga", "Bundesliga", 78),
]

EXPECTED_TEAM_COUNTS = {
    "Premier League": 20,
    "Ligue 1": 18,
    "La Liga": 20,
    "Serie A": 20,
    "Bundesliga": 18,
}

class RateLimitError(RuntimeError):
    """Raised when API-Football returns a rate limit response."""


# Generic club-type tokens that appear on either side of a club's real name
# (e.g. "FC Barcelona" / "Barcelona FC" / "SSC Napoli"). Stripped as whole
# words -- before founding-year digits and punctuation are removed -- so
# both orderings and both sources normalize to the same bare name.
CLUB_TYPE_TOKENS = {
    "fc",
    "cf",
    "ac",
    "as",
    "afc",
    "cfc",
    "ud",
    "sd",
    "cd",
    "rcd",
    "ca",
    "sc",
    "ss",
    "ssc",
    "us",
    "uc",
    "rc",
    "sv",
    "fsv",
    "ssd",
    "asd",
    "vfb",
    "vfl",
    "tsg",
    "calcio",
    "club",
    "sco",
    "hsc",
    "losc",
    "ogc",
    "bayer",
    "borussia",
    "olympique",
    "de",
}


def _normalize(name: str) -> str:
    """Fold a team name to a bare lowercase alphanumeric string for matching.

    Strips accents, generic club-type tokens (FC, SSC, VfL, ...), and
    founding-year digits so differently formatted names for the same club
    converge on the same string (see CLUB_TYPE_TOKENS).
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in CLUB_TYPE_TOKENS and not token.isdigit()]
    return "".join(tokens)


# Raw API-Football team name -> canonical Football-Data.co.uk name, for clubs
# whose naming differs beyond what generic normalization (accents, club-type
# tokens, founding-year digits) can bridge: nicknames, city-only forms,
# abbreviations. Keys are matched after normalization, so exact capitalization
# or punctuation in this dict does not matter. Add an entry here whenever the
# script reports an UNRESOLVED team that is clearly a known club.
NAME_OVERRIDES: dict[str, str] = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "West Bromwich Albion": "West Brom",
    "Tottenham Hotspur": "Tottenham",
    "AFC Bournemouth": "Bournemouth",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Luton Town": "Luton",
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alaves": "Alaves",
    "Real Valladolid": "Valladolid",
    "UD Almeria": "Almeria",
    "SD Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "UD Las Palmas": "Las Palmas",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Real Oviedo": "Oviedo",
    "RCD Espanyol": "Espanol",
    "Espanyol": "Espanol",
    "FC Barcelona": "Barcelona",
    "Deportivo La Coruna": "Dep. A Coruna",
    "Deportivo de La Coruna": "Dep. A Coruna",
    "Racing Santander": "Santander",
    "Real Racing Club": "Santander",
    "Malaga": "Malaga",
    "Malaga CF": "Malaga",
    # Serie A
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "FC Internazionale Milano": "Inter",
    "Hellas Verona": "Verona",
    "AC Pisa 1909": "Pisa",
    "Pisa SC": "Pisa",
    "SPAL": "Spal",
    "US SPAL 2013": "Spal",
    # Ligue 1
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "AS Saint-Etienne": "St Etienne",
    "Saint-Etienne": "St Etienne",
    "Le Havre": "Le Havre",
    "Girondins de Bordeaux": "Bordeaux",
    "Estac Troyes": "Troyes",
    "ES Troyes AC": "Troyes",
    "Olympique Lyonnais": "Lyon",
    "Stade Rennais FC": "Rennes",
    "Stade Rennais": "Rennes",
    "Stade Brestois 29": "Brest",
    "Stade Brestois": "Brest",
    "RC Strasbourg Alsace": "Strasbourg",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "FC Bayern Munchen": "Bayern Munich",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia M'gladbach": "M'gladbach",
    "1899 Hoffenheim": "Hoffenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "1 FC Union Berlin": "Union Berlin",
    "1 FC Koln": "FC Koln",
    "1. FC Koln": "FC Koln",
    "FC St Pauli": "St Pauli",
    "Hamburger SV": "Hamburg",
    "Hertha Berlin": "Hertha",
    "Hertha BSC": "Hertha",
    "Schalke 04": "Schalke 04",
    "FC Schalke 04": "Schalke 04",
    "1 FC Heidenheim": "Heidenheim",
    "1 FC Heidenheim 1846": "Heidenheim",
    "SV Darmstadt 98": "Darmstadt",
    "SC Paderborn 07": "Paderborn",
    "Fortuna Dusseldorf": "Fortuna Dusseldorf",
    "SpVgg Greuther Furth": "Greuther Furth",
    "Arminia Bielefeld": "Bielefeld",
}
NORMALIZED_NAME_OVERRIDES: dict[str, str] = {_normalize(raw_name): canonical for raw_name, canonical in NAME_OVERRIDES.items()}


def _canonical_names_by_league() -> dict[str, set[str]]:
    """Return every team name ever seen in each league's raw historical CSVs."""
    by_league: dict[str, set[str]] = {}
    for csv_path in glob.glob(str(RAW_DATA_DIR / "*.csv")):
        match = RAW_FILE_PATTERN.match(Path(csv_path).name)
        if match is None:
            continue
        league_key = match.group(1)
        df = pd.read_csv(csv_path)
        teams = set(df.get("HomeTeam", pd.Series(dtype=str)).dropna().astype(str))
        teams |= set(df.get("AwayTeam", pd.Series(dtype=str)).dropna().astype(str))
        by_league.setdefault(league_key, set()).update(teams)
    return by_league


def _request_teams(api_key: str, league_id: int, season: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Call the /teams endpoint and return raw team payloads and any errors."""
    try:
        response = requests.get(
            f"{BASE_URL}/teams",
            params={"league": league_id, "season": season},
            headers={"x-apisports-key": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return [], [f"network request failed: {exc}"]

    try:
        payload = response.json()
    except ValueError:
        return [], [f"response is not valid JSON (HTTP {response.status_code})"]

    if response.status_code == 429:
        raise RateLimitError(f"rateLimit on /teams (league={league_id}, season={season})")

    errors = payload.get("errors")
    error_messages: list[str] = []
    if isinstance(errors, dict):
        error_messages = [f"{key}: {value}" for key, value in errors.items() if value]
    elif isinstance(errors, list) and errors:
        error_messages = [str(error) for error in errors]

    if not response.ok and not error_messages:
        error_messages = [f"HTTP {response.status_code}"]

    response_items = payload.get("response")
    return (response_items if isinstance(response_items, list) else []), error_messages


def resolve_team_name(api_team_name: str, canonical_names: set[str]) -> str | None:
    """Map one API-Football team name to its Football-Data.co.uk spelling.

    Returns None when the name cannot be confidently resolved so the caller
    can skip it rather than risk mismapping a team's historical stats.
    """
    normalized_api_name = _normalize(api_team_name)
    canonical_by_normalized = {_normalize(name): name for name in canonical_names}

    if normalized_api_name in canonical_by_normalized:
        return canonical_by_normalized[normalized_api_name]
    if normalized_api_name in NORMALIZED_NAME_OVERRIDES:
        return NORMALIZED_NAME_OVERRIDES[normalized_api_name]  # may target a newly promoted team not yet in canonical_names
    return None


def fetch_current_teams(
    season: int,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Fetch and map current Big 5 rosters. Returns (rows_df, unresolved_by_league)."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        raise ValueError("API_FOOTBALL_KEY is missing from .env")

    canonical_names_by_league = _canonical_names_by_league()
    rows: list[dict[str, Any]] = []
    unresolved_by_league: dict[str, list[str]] = {}
    season_label = f"{season}-{season + 1}"

    for league_key, league_name, league_id in LEAGUES:
        canonical_names = canonical_names_by_league.get(league_key, set())
        team_payloads, errors = _request_teams(api_key, league_id, season)
        if errors:
            print(f"{league_name}: API error(s): {'; '.join(errors)}")
            continue
        if not team_payloads:
            print(f"{league_name}: empty response from API-Football (league={league_id}, season={season})")
            continue

        resolved_count = 0
        for item in team_payloads:
            team_info = item.get("team", {}) if isinstance(item, dict) else {}
            api_name = team_info.get("name")
            if not api_name:
                continue
            resolved_name = resolve_team_name(api_name, canonical_names)
            if resolved_name is None:
                unresolved_by_league.setdefault(league_name, []).append(api_name)
                continue
            resolved_count += 1
            rows.append(
                {
                    "league": league_name,
                    "season": season_label,
                    "team": resolved_name,
                    "source_file": f"api_football:teams?league={league_id}&season={season}",
                }
            )

        expected = EXPECTED_TEAM_COUNTS[league_name]
        print(f"{league_name}: {resolved_count}/{len(team_payloads)} teams resolved (expected {expected}).")
        time.sleep(sleep_seconds)

    return pd.DataFrame(rows, columns=["league", "season", "team", "source_file"]), unresolved_by_league


def print_unresolved(unresolved_by_league: dict[str, list[str]]) -> None:
    """Print a clear, actionable summary of names that could not be mapped."""
    if not unresolved_by_league:
        return
    print("\nUNRESOLVED team names (excluded from current_teams.csv to avoid mismapping):")
    for league_name, names in unresolved_by_league.items():
        for name in names:
            print(f"  - [{league_name}] '{name}' -- add a NAME_OVERRIDES entry in this script if this is a real club.")


def _merge_with_existing(fresh_teams: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Replace only the leagues present in fresh_teams; keep other leagues as-is.

    Guards against a partial API failure (e.g. quota hit after 1 of 5
    leagues, or a league not included in the current plan) silently wiping
    out leagues that simply were not re-fetched this run.
    """
    if not output_path.exists():
        return fresh_teams

    existing_teams = pd.read_csv(output_path)
    fetched_leagues = set(fresh_teams["league"].unique())
    kept_teams = existing_teams[~existing_teams["league"].isin(fetched_leagues)]
    return pd.concat([kept_teams, fresh_teams], ignore_index=True)


def run(season: int, output_path: Path, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> int:
    """Fetch, map, and save current teams. Returns a process exit code."""
    current_teams, unresolved_by_league = fetch_current_teams(season, sleep_seconds=sleep_seconds)

    if current_teams.empty:
        print("\nNo teams resolved from API-Football; keeping the existing current_teams.csv untouched.")
        print("Run scripts/generate_current_teams.py instead, or check your API-Football plan/season.")
        return 1

    fetched_leagues = set(current_teams["league"].unique())
    skipped_leagues = [league_name for _, league_name, _ in LEAGUES if league_name not in fetched_leagues]
    if skipped_leagues:
        print(
            f"\nNote: {', '.join(skipped_leagues)} returned no usable data this run -- "
            "keeping their existing entries in current_teams.csv untouched."
        )

    merged_teams = _merge_with_existing(current_teams, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_teams.to_csv(output_path, index=False)
    print(f"\nSaved current teams reference to: {output_path}")
    print_unresolved(unresolved_by_league)

    incomplete_leagues = [
        league_name
        for league_name, expected in EXPECTED_TEAM_COUNTS.items()
        if len(current_teams[current_teams["league"] == league_name]) < expected
        and league_name in fetched_leagues
    ]
    if incomplete_leagues:
        print(
            "\nNote: some freshly fetched leagues are below their expected team count "
            f"({', '.join(incomplete_leagues)}) -- not every team has played yet this season. "
            "Re-run scripts/generate_current_teams.py afterwards; its previous-season fallback will fill the gap."
        )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch current Big 5 rosters from API-Football.")
    parser.add_argument("--season", type=int, default=2026, help="API-Football season start year, e.g. 2026 for 2026-2027.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    return parser.parse_args()


def main() -> int:
    """Run the current-teams fetch from the command line."""
    args = parse_args()
    try:
        return run(season=args.season, output_path=args.output, sleep_seconds=args.sleep_seconds)
    except (ValueError, RateLimitError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
