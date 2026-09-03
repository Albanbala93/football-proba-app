"""Fetch API-Football Ligue 1 data for local enrichment experiments."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_LEAGUE_ID = 61
DEFAULT_SEASON = 2025
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_SECONDS = 0.3


class RateLimitError(RuntimeError):
    """Raised when API-Football returns a rate limit response."""


class APIResponseError(RuntimeError):
    """Raised when an API response is invalid for the current request."""


def _relative_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _extract_errors(payload: Any) -> list[str]:
    """Extract API-Football error messages from common response shapes."""
    if not isinstance(payload, dict):
        return []
    errors = payload.get("errors")
    if not errors:
        return []
    if isinstance(errors, dict):
        messages = []
        for key, value in errors.items():
            if isinstance(value, list):
                messages.extend(f"{key}: {item}" for item in value)
            elif value:
                messages.append(f"{key}: {value}")
        return messages
    if isinstance(errors, list):
        return [str(error) for error in errors]
    return [str(errors)]


def _response_items(payload: Any) -> list[Any]:
    """Return the response list from an API-Football payload."""
    if not isinstance(payload, dict):
        return []
    response = payload.get("response")
    return response if isinstance(response, list) else []


def _quota_from_payload(payload: Any) -> dict[str, Any] | None:
    """Return quota fields when present."""
    if not isinstance(payload, dict):
        return None
    response = payload.get("response")
    if isinstance(response, dict) and isinstance(response.get("requests"), dict):
        return response["requests"]
    if isinstance(payload.get("requests"), dict):
        return payload["requests"]
    return None


def _write_json(path: Path, payload: Any, force: bool) -> bool:
    """Write JSON unless it exists and force is false. Return whether a write happened."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    return True


def _read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _request(api_key: str, endpoint: str, params: dict[str, Any]) -> tuple[Any, list[str]]:
    """Call one API-Football endpoint."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.get(
            url,
            params=params,
            headers={"x-apisports-key": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return {"endpoint": endpoint, "params": params, "response": []}, [f"network request failed: {exc}"]

    try:
        payload = response.json()
    except ValueError:
        return {
            "endpoint": endpoint,
            "params": params,
            "http_status": response.status_code,
            "raw_text": response.text[:500],
            "response": [],
        }, ["response is not valid JSON"]

    payload["_http_status"] = response.status_code
    errors = _extract_errors(payload)
    if not response.ok and not errors:
        errors = [f"HTTP {response.status_code}"]
    return payload, errors


def safe_api_get(api_key: str, endpoint: str, params: dict[str, Any], expect_response_items: bool = True) -> Any:
    """Fetch one API-Football payload and raise on rate limit or invalid responses."""
    payload, errors = _request(api_key, endpoint, params)
    http_status = payload.get("_http_status") if isinstance(payload, dict) else None
    error_text = " ".join(errors).lower()
    if http_status == 429 or "ratelimit" in error_text or "too many requests" in error_text:
        raise RateLimitError(f"rateLimit on {endpoint}: {errors[0] if errors else 'Too many requests'}")
    if http_status is not None and http_status >= 400 and errors:
        raise APIResponseError(f"HTTP error on {endpoint}: {errors[0]}")
    if expect_response_items and not _response_items(payload):
        raise APIResponseError(f"empty response on {endpoint}")
    return payload


def _load_or_fetch(
    api_key: str,
    endpoint: str,
    params: dict[str, Any],
    path: Path,
    force: bool,
) -> tuple[Any, list[str], bool]:
    """Load an existing JSON file or fetch and save it."""
    if path.exists() and not force:
        return _read_json(path), [], False
    payload, errors = _request(api_key, endpoint, params)
    return payload, errors, True


def _load_cached_or_fetch(
    api_key: str,
    endpoint: str,
    params: dict[str, Any],
    path: Path,
    force: bool,
    expect_response_items: bool = True,
) -> tuple[Any, bool]:
    """Return a cached payload if available, otherwise fetch a fresh one."""
    if path.exists() and not force:
        return _read_json(path), False

    payload = safe_api_get(api_key, endpoint, params, expect_response_items=expect_response_items)
    _persist_if_valid(path, payload, force=force, expect_response_items=expect_response_items)
    return payload, True


def _persist_if_valid(path: Path, payload: Any, force: bool, expect_response_items: bool = True) -> bool:
    """Persist a payload only when it is valid and non-empty."""
    if expect_response_items and not _response_items(payload):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("_http_status") == 429:
        return False
    error_text = " ".join(_extract_errors(payload)).lower()
    if "ratelimit" in error_text or "too many requests" in error_text:
        return False
    return _write_json(path, payload, force=force)


def _fixture_row(
    fixture_payload: dict[str, Any],
    league_id: int,
    season: int,
    statistics_path: Path,
    lineups_path: Path,
) -> dict[str, Any]:
    """Build one fixtures_index row."""
    fixture = fixture_payload.get("fixture", {}) if isinstance(fixture_payload, dict) else {}
    teams = fixture_payload.get("teams", {}) if isinstance(fixture_payload, dict) else {}
    goals = fixture_payload.get("goals", {}) if isinstance(fixture_payload, dict) else {}
    home = teams.get("home", {}) if isinstance(teams, dict) else {}
    away = teams.get("away", {}) if isinstance(teams, dict) else {}
    status = fixture.get("status", {}) if isinstance(fixture, dict) else {}

    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "league_id": league_id,
        "season": season,
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name"),
        "status_short": status.get("short"),
        "goals_home": goals.get("home") if isinstance(goals, dict) else None,
        "goals_away": goals.get("away") if isinstance(goals, dict) else None,
        "has_statistics": statistics_path.exists(),
        "has_lineups": lineups_path.exists(),
        "statistics_path": _relative_path(statistics_path) if statistics_path.exists() else "",
        "lineups_path": _relative_path(lineups_path) if lineups_path.exists() else "",
    }


def fetch_api_football_ligue1_data(
    league_id: int = DEFAULT_LEAGUE_ID,
    season: int = DEFAULT_SEASON,
    limit_fixtures: int | None = None,
    force: bool = False,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    resume: bool = False,
) -> pd.DataFrame:
    """Fetch fixtures, per-fixture statistics/lineups, injuries, and an index CSV."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = (os.getenv("API_FOOTBALL_KEY") or "").strip()
    if not api_key:
        raise ValueError("API_FOOTBALL_KEY is missing from .env")

    output_dir = PROJECT_ROOT / "data" / "external" / "api_football" / f"ligue1_{season}"
    statistics_dir = output_dir / "statistics"
    lineups_dir = output_dir / "lineups"
    fixtures_path = output_dir / "fixtures.json"
    injuries_path = output_dir / "injuries.json"
    index_path = output_dir / "fixtures_index.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    lineups_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    latest_quota: dict[str, Any] | None = None

    try:
        fixtures_payload, fixtures_fetched = _load_cached_or_fetch(
            api_key,
            "/fixtures",
            {"league": league_id, "season": season},
            fixtures_path,
            force=force,
            expect_response_items=True,
        )
    except RateLimitError:
        raise
    except APIResponseError as exc:
        errors.append(f"fixtures: {exc}")
        if fixtures_path.exists():
            fixtures_payload = _read_json(fixtures_path)
            fixtures_fetched = False
        else:
            fixtures_payload = {"response": []}
            fixtures_fetched = True
    latest_quota = _quota_from_payload(fixtures_payload) or latest_quota

    fixtures = _response_items(fixtures_payload)
    selected_fixtures = fixtures[:limit_fixtures] if limit_fixtures is not None else fixtures

    statistics_ok = 0
    lineups_ok = 0
    rows = []

    for fixture_item in selected_fixtures:
        fixture = fixture_item.get("fixture", {}) if isinstance(fixture_item, dict) else {}
        fixture_id = fixture.get("id") if isinstance(fixture, dict) else None
        if fixture_id is None:
            errors.append("fixture without fixture.id skipped")
            continue

        statistics_path = statistics_dir / f"fixture_{fixture_id}.json"
        lineups_path = lineups_dir / f"fixture_{fixture_id}.json"

        if (resume or not force) and statistics_path.exists():
            statistics_payload = _read_json(statistics_path)
        else:
            try:
                time.sleep(sleep_seconds)
                statistics_payload = safe_api_get(
                    api_key,
                    "/fixtures/statistics",
                    {"fixture": fixture_id},
                    expect_response_items=True,
                )
                _persist_if_valid(statistics_path, statistics_payload, force=force, expect_response_items=True)
            except RateLimitError:
                errors.append(f"statistics fixture {fixture_id}: rate limit reached")
                raise
            except APIResponseError as exc:
                errors.append(f"statistics fixture {fixture_id}: {exc}")
                statistics_payload = {"response": []}
        latest_quota = _quota_from_payload(statistics_payload) or latest_quota
        if _response_items(statistics_payload):
            statistics_ok += 1

        if (resume or not force) and lineups_path.exists():
            lineups_payload = _read_json(lineups_path)
        else:
            try:
                time.sleep(sleep_seconds)
                lineups_payload = safe_api_get(
                    api_key,
                    "/fixtures/lineups",
                    {"fixture": fixture_id},
                    expect_response_items=True,
                )
                _persist_if_valid(lineups_path, lineups_payload, force=force, expect_response_items=True)
            except RateLimitError:
                errors.append(f"lineups fixture {fixture_id}: rate limit reached")
                raise
            except APIResponseError as exc:
                errors.append(f"lineups fixture {fixture_id}: {exc}")
                lineups_payload = {"response": []}
        latest_quota = _quota_from_payload(lineups_payload) or latest_quota
        if _response_items(lineups_payload):
            lineups_ok += 1

        rows.append(
            _fixture_row(
                fixture_payload=fixture_item,
                league_id=league_id,
                season=season,
                statistics_path=statistics_path,
                lineups_path=lineups_path,
            )
        )

    try:
        injuries_payload, injuries_fetched = _load_cached_or_fetch(
            api_key,
            "/injuries",
            {"league": league_id, "season": season},
            injuries_path,
            force=force,
            expect_response_items=True,
        )
    except RateLimitError:
        raise
    except APIResponseError as exc:
        errors.append(f"injuries: {exc}")
        injuries_fetched = True
        injuries_payload = {"response": []}
    latest_quota = _quota_from_payload(injuries_payload) or latest_quota
    injuries_ok = len(_response_items(injuries_payload)) > 0

    index = pd.DataFrame(rows)
    index.to_csv(index_path, index=False)

    print("API-Football Ligue 1 fetch")
    print(f"League id: {league_id}")
    print(f"Season: {season}")
    print(f"Output directory: {output_dir}")
    print(f"Fixtures available: {len(fixtures)}")
    print(f"Fixtures fetched from API: {'yes' if fixtures_fetched else 'no (cached)'}")
    print(f"Fixtures processed: {len(selected_fixtures)}")
    print(f"Statistics OK: {statistics_ok}")
    print(f"Lineups OK: {lineups_ok}")
    print(f"Injuries OK: {'yes' if injuries_ok else 'no'}")
    print(f"Injuries fetched from API: {'yes' if injuries_fetched else 'no (cached)'}")
    print(f"Fixtures index: {index_path}")
    if latest_quota:
        print(f"Quota remaining/current: {latest_quota}")
    else:
        print("Quota remaining/current: not_available")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nErrors: none")

    return index


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch API-Football Ligue 1 enrichment data.")
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--limit-fixtures", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the API-Football Ligue 1 fetcher."""
    args = parse_args()
    try:
        fetch_api_football_ligue1_data(
            league_id=args.league_id,
            season=args.season,
            limit_fixtures=args.limit_fixtures,
            force=args.force,
            sleep_seconds=args.sleep,
            resume=args.resume,
        )
    except RateLimitError as exc:
        print(f"\nRate limit reached: {exc}")
        raise SystemExit(1) from exc
    except APIResponseError as exc:
        print(f"\nAPI-Football fetch failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
