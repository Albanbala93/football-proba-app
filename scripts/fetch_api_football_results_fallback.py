"""Fill in missing current-season results from API-Football when
football-data.co.uk is unreachable (see download_big5_data.py).

Football-data.co.uk has been failing consistently from GitHub Actions
(persistent 503 "temporarily unavailable" on the current-season file across
all 5 leagues, while a normal browser request to the exact same URL
succeeds -- see the investigation in this project's history). This script
is a fallback, not a replacement: it only adds match results that are
missing from the existing raw CSV, using the same team-name resolution
already validated for scripts/fetch_api_football_current_teams.py. Odds and
detailed match stats are not available from this path and are left blank --
football-data.co.uk remains the source of truth for those whenever it is
reachable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_api_football_current_teams import (  # noqa: E402
    LEAGUES,
    RateLimitError,
    _canonical_names_by_league,
    resolve_team_name,
)

BASE_URL = "https://v3.football.api-sports.io"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_SECONDS = 0.3
RAW_CSV_COLUMNS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
DIV_CODE_BY_LEAGUE_KEY = {
    "premier_league": "E0",
    "ligue1": "F1",
    "laliga": "SP1",
    "seriea": "I1",
    "bundesliga": "D1",
}


def _request_fixtures(api_key: str, league_id: int, season: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Call the /fixtures endpoint and return raw fixture payloads and any errors."""
    try:
        response = requests.get(
            f"{BASE_URL}/fixtures",
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
        raise RateLimitError(f"rateLimit on /fixtures (league={league_id}, season={season})")

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


def _result_from_goals(home_goals: Any, away_goals: Any) -> str | None:
    """Return H/D/A from goal counts, or None if either is missing."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _existing_raw_path(league_key: str, season: int) -> Path:
    """Return the raw CSV path for one league/season, matching download_big5_data.py's naming."""
    return RAW_DATA_DIR / f"{league_key}_{season}_{season + 1}.csv"


def _load_existing(path: Path) -> tuple[pd.DataFrame, set[tuple[str, str, str]]]:
    """Load an existing raw CSV (if any) and its set of known (date, home, away) keys.

    Membership is checked per match rather than via a "latest date seen" cutoff:
    a match can be skipped on one run (e.g. an unresolved team name) while later
    matches on the same or a later date succeed, so a date cutoff alone would
    permanently hide that match once the cutoff moves past it.
    """
    if not path.exists():
        return pd.DataFrame(columns=RAW_CSV_COLUMNS), set()
    existing = pd.read_csv(path)
    if not {"Date", "HomeTeam", "AwayTeam"}.issubset(existing.columns):
        return existing, set()
    keys = set(
        zip(
            existing["Date"].astype(str),
            existing["HomeTeam"].astype(str),
            existing["AwayTeam"].astype(str),
        )
    )
    return existing, keys


def fetch_missing_results(season: int, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> dict[str, dict[str, Any]]:
    """Fetch and merge missing current-season results for all Big 5 leagues.

    Returns a per-league summary dict for reporting.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = (os.getenv("API_FOOTBALL_KEY") or "").strip()
    if not api_key:
        raise ValueError("API_FOOTBALL_KEY is missing from .env")

    canonical_names_by_league = _canonical_names_by_league()
    summary: dict[str, dict[str, Any]] = {}

    for league_key, league_name, league_id in LEAGUES:
        raw_path = _existing_raw_path(league_key, season)
        existing, existing_keys = _load_existing(raw_path)
        canonical_names = canonical_names_by_league.get(league_key, set())

        try:
            fixtures, errors = _request_fixtures(api_key, league_id, season)
        except RateLimitError as exc:
            summary[league_name] = {"error": str(exc), "added": 0}
            continue

        if errors:
            summary[league_name] = {"error": "; ".join(errors), "added": 0}
            time.sleep(sleep_seconds)
            continue

        new_rows: list[dict[str, Any]] = []
        unresolved: set[str] = set()
        for item in fixtures:
            fixture = item.get("fixture", {}) if isinstance(item, dict) else {}
            teams = item.get("teams", {}) if isinstance(item, dict) else {}
            goals = item.get("goals", {}) if isinstance(item, dict) else {}
            status = fixture.get("status", {}) if isinstance(fixture, dict) else {}

            if status.get("short") != "FT":
                continue

            match_datetime = pd.to_datetime(fixture.get("date"), errors="coerce", utc=True)
            if pd.isna(match_datetime):
                continue
            match_date = match_datetime.tz_localize(None)

            home_api_name = (teams.get("home") or {}).get("name")
            away_api_name = (teams.get("away") or {}).get("name")
            home_team = resolve_team_name(home_api_name, canonical_names) if home_api_name else None
            away_team = resolve_team_name(away_api_name, canonical_names) if away_api_name else None
            if not home_team or not away_team:
                if home_api_name and not home_team:
                    unresolved.add(home_api_name)
                if away_api_name and not away_team:
                    unresolved.add(away_api_name)
                continue

            date_label = match_date.strftime("%d/%m/%Y")
            if (date_label, home_team, away_team) in existing_keys:
                continue

            result = _result_from_goals(goals.get("home"), goals.get("away"))
            if result is None:
                continue

            new_rows.append(
                {
                    "Div": DIV_CODE_BY_LEAGUE_KEY[league_key],
                    "Date": date_label,
                    "Time": match_date.strftime("%H:%M"),
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "FTHG": goals.get("home"),
                    "FTAG": goals.get("away"),
                    "FTR": result,
                }
            )

        if new_rows:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined["_sort_date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
            combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
            combined = combined.sort_values("_sort_date").drop(columns="_sort_date").reset_index(drop=True)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(raw_path, index=False)

        summary[league_name] = {
            "added": len(new_rows),
            "unresolved_teams": sorted(unresolved),
            "output_path": str(raw_path.relative_to(PROJECT_ROOT)),
        }
        time.sleep(sleep_seconds)

    return summary


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    """Print a compact per-league summary."""
    print("API-Football results fallback")
    for league_name, info in summary.items():
        if "error" in info:
            print(f"{league_name}: ERROR - {info['error']}")
            continue
        print(f"{league_name}: +{info['added']} match(es) -> {info['output_path']}")
        if info.get("unresolved_teams"):
            print(f"  Unresolved team names (skipped): {', '.join(info['unresolved_teams'])}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch missing current-season results from API-Football.")
    parser.add_argument("--season", type=int, required=True, help="API-Football season year, e.g. 2026.")
    return parser.parse_args()


def main() -> int:
    """Run the fallback results fetch from the command line."""
    args = parse_args()
    summary = fetch_missing_results(season=args.season)
    print_summary(summary)
    return 1 if any("error" in info for info in summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
