"""List Sportmonks leagues accessible with the current account token."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.sportmonks.com/v3/football"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "sportmonks_accessible_leagues.csv"
REQUEST_TIMEOUT_SECONDS = 20
LEAGUE_INCLUDE_CANDIDATES = [
    "country;seasons;currentSeason;latestSeason",
    "country;seasons;currentSeason",
    "country;seasons",
    "country",
    "",
]
SEARCH_TERMS = ["Premier", "Ligue", "Liga", "Serie", "Bundesliga", "France", "England", "Spain", "Italy", "Germany"]


def _load_api_token() -> str:
    """Load Sportmonks token from .env without printing it."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    if not api_token:
        raise RuntimeError("SPORTMONKS_API_TOKEN is missing from .env")
    return api_token


def _extract_error_message(payload: Any) -> str | None:
    """Extract a readable error message from a Sportmonks response."""
    if not isinstance(payload, dict):
        return None

    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message

    errors = payload.get("errors")
    if isinstance(errors, dict):
        messages = []
        for value in errors.values():
            if isinstance(value, list):
                messages.extend(str(item) for item in value if item)
            elif value:
                messages.append(str(value))
        if messages:
            return "; ".join(messages)

    if isinstance(errors, list) and errors:
        return "; ".join(str(error) for error in errors)

    return None


def _request_json(
    session: requests.Session,
    api_token: str,
    page: int,
    include: str,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """Fetch one paginated leagues page."""
    params = {
        "api_token": api_token,
        "per_page": 50,
        "page": page,
    }
    if include:
        params["include"] = include
    try:
        response = session.get(f"{BASE_URL}/leagues", params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return None, f"network request failed: {exc}", None

    try:
        payload = response.json()
    except ValueError:
        return None, "response is not valid JSON", response.status_code

    if not response.ok:
        return payload, _extract_error_message(payload) or "request failed", response.status_code

    api_error = _extract_error_message(payload)
    if api_error:
        return payload, api_error, response.status_code

    return payload, None, response.status_code


def _data_list(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return payload data as a list of dictionaries."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _pagination_next_page(payload: dict[str, Any] | None) -> int | None:
    """Return next page number from Sportmonks pagination if available."""
    if not isinstance(payload, dict):
        return None
    pagination = payload.get("pagination")
    if not isinstance(pagination, dict):
        return None
    if not pagination.get("has_more"):
        return None
    try:
        return int(pagination.get("current_page", 0)) + 1
    except (TypeError, ValueError):
        return None


def _included(value: Any) -> Any:
    """Unwrap Sportmonks include objects when they use a data key."""
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return value


def _first_id(value: Any) -> Any:
    """Return id from an included object or first included list item."""
    value = _included(value)
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        return value.get("id")
    return None


def _latest_season_id_from_seasons(league: dict[str, Any]) -> Any:
    """Return the most recent season id from included seasons when possible."""
    seasons = _included(league.get("seasons"))
    if isinstance(seasons, dict):
        return seasons.get("id")
    if not isinstance(seasons, list) or not seasons:
        return None

    sorted_seasons = sorted(
        [season for season in seasons if isinstance(season, dict)],
        key=lambda season: str(season.get("starting_at") or season.get("name") or ""),
    )
    if not sorted_seasons:
        return None
    return sorted_seasons[-1].get("id")


def _country_fields(league: dict[str, Any]) -> tuple[Any, Any]:
    """Return country id and name when available."""
    country = _included(league.get("country"))
    if isinstance(country, dict):
        return country.get("id"), country.get("name")
    return league.get("country_id"), None


def _season_count(league: dict[str, Any]) -> tuple[bool, int | None]:
    """Return whether seasons include is present and how many seasons it contains."""
    if "seasons" not in league:
        return False, None
    seasons = _included(league.get("seasons"))
    if isinstance(seasons, list):
        return True, len(seasons)
    if isinstance(seasons, dict):
        return True, 1
    if seasons is None:
        return True, 0
    return True, None


def _league_row(league: dict[str, Any]) -> dict[str, Any]:
    """Build one exported league row."""
    country_id, country_name = _country_fields(league)
    has_seasons_data, seasons_count = _season_count(league)
    return {
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "country_id": country_id,
        "country_name": country_name,
        "sport_id": league.get("sport_id"),
        "active": league.get("active"),
        "type": league.get("type"),
        "category": league.get("category"),
        "current_season_id": _first_id(league.get("currentSeason")) or league.get("current_season_id"),
        "latest_season_id": (
            _first_id(league.get("latestSeason"))
            or league.get("latest_season_id")
            or _latest_season_id_from_seasons(league)
        ),
        "has_seasons_data": has_seasons_data,
        "seasons_count": seasons_count,
    }


def _select_working_include(session: requests.Session, api_token: str) -> str:
    """Return the richest leagues include supported by the account/API."""
    last_error = None
    for include in LEAGUE_INCLUDE_CANDIDATES:
        _, error_message, _ = _request_json(session, api_token, page=1, include=include)
        if error_message is None:
            return include
        last_error = error_message
    raise RuntimeError(f"No usable leagues include found. Last error: {last_error}")


def list_accessible_leagues(output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """List accessible Sportmonks leagues and export them."""
    api_token = _load_api_token()
    rows = []

    with requests.Session() as session:
        include = _select_working_include(session, api_token)
        print(f"Using leagues include: {include or 'none'}")
        page = 1
        while page is not None:
            payload, error_message, http_status = _request_json(session, api_token, page, include)
            if error_message:
                raise RuntimeError(f"Could not fetch leagues page {page}: HTTP {http_status}, {error_message}")

            rows.extend(_league_row(league) for league in _data_list(payload))
            page = _pagination_next_page(payload)

    leagues = pd.DataFrame(
        rows,
        columns=[
            "league_id",
            "league_name",
            "country_id",
            "country_name",
            "sport_id",
            "active",
            "type",
            "category",
            "current_season_id",
            "latest_season_id",
            "has_seasons_data",
            "seasons_count",
        ],
    )
    if not leagues.empty:
        leagues = leagues.sort_values(["country_name", "league_name"], na_position="last").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    leagues.to_csv(output_path, index=False)
    print_summary(leagues, output_path)
    return leagues


def print_summary(leagues: pd.DataFrame, output_path: Path) -> None:
    """Print a readable console summary."""
    print("Sportmonks accessible leagues")
    print(f"Total accessible leagues: {len(leagues)}")

    display_columns = [
        "league_id",
        "league_name",
        "country_name",
        "current_season_id",
        "latest_season_id",
        "seasons_count",
    ]
    if not leagues.empty:
        print("\nTop 50 leagues:")
        print(leagues[display_columns].head(50).to_string(index=False))

        search_pattern = "|".join(SEARCH_TERMS)
        searchable = (
            leagues["league_name"].fillna("").astype(str)
            + " "
            + leagues["country_name"].fillna("").astype(str)
        )
        matches = leagues[searchable.str.contains(search_pattern, case=False, regex=True, na=False)].copy()
        print("\nMatching leagues:")
        if matches.empty:
            print("None")
        else:
            print(matches[display_columns].to_string(index=False))

    print(f"\nSaved Sportmonks accessible leagues to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="List Sportmonks leagues accessible with the current account.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Run accessible league listing."""
    args = parse_args()
    try:
        list_accessible_leagues(output_path=args.output)
    except Exception as exc:
        print("Sportmonks accessible leagues")
        print("ERROR - league listing failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
