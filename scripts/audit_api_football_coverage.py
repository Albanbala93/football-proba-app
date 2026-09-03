"""Audit API-Football coverage available with the current API key."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://v3.football.api-sports.io"
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_LEAGUE_ID = 61
DEFAULT_SEASON = 2024
SENSITIVE_KEYS = {
    "account",
    "email",
    "firstname",
    "key",
    "lastname",
    "password",
    "secret",
    "token",
    "x-apisports-key",
}
STATISTICS_NAMES = {
    "Shots on Goal",
    "Total Shots",
    "Ball Possession",
    "Corner Kicks",
    "Fouls",
    "Yellow Cards",
    "Red Cards",
}
EXPECTED_GOALS_NAMES = {"Expected Goals", "expected_goals", "xG", "xg"}


def _sanitize_payload(value: Any) -> Any:
    """Recursively remove sensitive-looking values from a JSON payload."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_lower in SENSITIVE_KEYS or any(secret in key_lower for secret in SENSITIVE_KEYS):
                sanitized[key_text] = "[redacted]"
            else:
                sanitized[key_text] = _sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    return value


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


def _sample_keys(value: Any) -> list[str]:
    """Return top-level sample keys from the first useful object."""
    if isinstance(value, dict):
        response = value.get("response")
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict):
                return sorted(str(key) for key in first.keys())
        if isinstance(response, dict):
            return sorted(str(key) for key in response.keys())
        return sorted(str(key) for key in value.keys())
    return []


def _results_count(payload: Any) -> int:
    """Return API-Football results count."""
    if not isinstance(payload, dict):
        return 0
    results = payload.get("results")
    if isinstance(results, int):
        return results
    response = payload.get("response")
    if isinstance(response, list):
        return len(response)
    if isinstance(response, dict):
        return 1
    return 0


def _audit_row(
    test_name: str,
    endpoint: str,
    params: dict[str, Any] | None,
    http_status: int | None,
    payload: Any,
    request_error: str | None = None,
) -> dict[str, Any]:
    """Build a normalized audit row."""
    errors = [request_error] if request_error else _extract_errors(payload)
    success = bool(http_status and 200 <= http_status < 300 and not errors)
    return {
        "test_name": test_name,
        "endpoint": endpoint,
        "params": params or {},
        "http_status": http_status,
        "success": success,
        "errors": errors,
        "results_count": _results_count(payload),
        "has_statistics": False,
        "has_lineups": False,
        "has_injuries": False,
        "has_odds": False,
        "has_expected_goals": False,
        "sample_keys": _sample_keys(payload),
        "response_sample": _sanitize_payload(payload),
    }


def _request(api_key: str, endpoint: str, params: dict[str, Any] | None = None) -> tuple[int | None, Any, str | None]:
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
        return None, None, f"network request failed: {exc}"

    try:
        payload = response.json()
    except ValueError:
        return response.status_code, {"raw_text": response.text[:500]}, "response is not valid JSON"
    return response.status_code, payload, None


def _league_matches(payload: Any) -> list[dict[str, Any]]:
    """Return Ligue 1 / Ligue 2 league rows from /leagues response."""
    if not isinstance(payload, dict):
        return []
    response = payload.get("response")
    if not isinstance(response, list):
        return []

    matches = []
    for item in response:
        if not isinstance(item, dict):
            continue
        league = item.get("league", {})
        if not isinstance(league, dict):
            continue
        name = str(league.get("name", ""))
        if "Ligue 1" in name or "Ligue 2" in name:
            matches.append(item)
    return matches


def _first_league_id(matches: list[dict[str, Any]], league_name: str) -> int | None:
    """Return the first matching league id."""
    for item in matches:
        league = item.get("league", {})
        if isinstance(league, dict) and league_name in str(league.get("name", "")):
            league_id = league.get("id")
            return int(league_id) if league_id is not None else None
    return None


def _first_fixture_id(payload: Any) -> int | None:
    """Return the first fixture id from a fixtures response."""
    if not isinstance(payload, dict):
        return None
    response = payload.get("response")
    if not isinstance(response, list):
        return None
    for item in response:
        if not isinstance(item, dict):
            continue
        fixture = item.get("fixture", {})
        if isinstance(fixture, dict) and fixture.get("id") is not None:
            return int(fixture["id"])
    return None


def _statistics_types(payload: Any) -> set[str]:
    """Return statistic type names found in /fixtures/statistics."""
    found = set()
    if not isinstance(payload, dict):
        return found
    response = payload.get("response")
    if not isinstance(response, list):
        return found
    for team_stats in response:
        if not isinstance(team_stats, dict):
            continue
        statistics = team_stats.get("statistics")
        if not isinstance(statistics, list):
            continue
        for stat in statistics:
            if isinstance(stat, dict) and stat.get("type") is not None:
                found.add(str(stat["type"]))
    return found


def _lineup_flags(payload: Any) -> dict[str, bool]:
    """Return whether lineups expose formation, startXI, and substitutes."""
    flags = {"formation": False, "startXI": False, "substitutes": False}
    if not isinstance(payload, dict):
        return flags
    response = payload.get("response")
    if not isinstance(response, list):
        return flags
    for lineup in response:
        if not isinstance(lineup, dict):
            continue
        flags["formation"] = flags["formation"] or bool(lineup.get("formation"))
        flags["startXI"] = flags["startXI"] or isinstance(lineup.get("startXI"), list)
        flags["substitutes"] = flags["substitutes"] or isinstance(lineup.get("substitutes"), list)
    return flags


def _status_label(row: dict[str, Any]) -> str:
    """Return OK / LIMITED / ERROR for console display."""
    if not row["success"]:
        return "ERROR"
    if row["results_count"] == 0 and row["test_name"] != "status":
        return "LIMITED"
    if row["test_name"] == "statistics" and not row["has_statistics"]:
        return "LIMITED"
    if row["test_name"] == "lineups" and not row["has_lineups"]:
        return "LIMITED"
    if row["test_name"] == "injuries" and not row["has_injuries"]:
        return "LIMITED"
    if row["test_name"] == "odds" and not row["has_odds"]:
        return "LIMITED"
    return "OK"


def default_output_path(league_id: int, season: int) -> Path:
    """Return the default audit output path for one league and season."""
    return PROJECT_ROOT / "data" / "reference" / f"api_football_coverage_audit_{league_id}_{season}.json"


def audit_api_football_coverage(
    league_id: int = DEFAULT_LEAGUE_ID,
    season: int = DEFAULT_SEASON,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the API-Football coverage audit and export a JSON report."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = (os.getenv("API_FOOTBALL_KEY") or "").strip()
    if not api_key:
        raise ValueError("API_FOOTBALL_KEY is missing from .env")

    if output_path is None:
        output_path = default_output_path(league_id, season)

    tests = []

    status_code, payload, error = _request(api_key, "/status")
    tests.append(_audit_row("status", "/status", {}, status_code, payload, error))

    league_params = {"country": "France"}
    status_code, leagues_payload, error = _request(api_key, "/leagues", league_params)
    league_row = _audit_row("search_france_leagues", "/leagues", league_params, status_code, leagues_payload, error)
    league_matches = _league_matches(leagues_payload)
    league_row["matched_leagues"] = _sanitize_payload(league_matches)
    tests.append(league_row)

    fixture_id = None

    fixture_params = {"league": league_id, "season": season}
    status_code, fixtures_payload, error = _request(api_key, "/fixtures", fixture_params)
    fixture_row = _audit_row("ligue1_fixtures", "/fixtures", fixture_params, status_code, fixtures_payload, error)
    fixture_id = _first_fixture_id(fixtures_payload)
    fixture_row["fixture_id_sample"] = fixture_id
    tests.append(fixture_row)

    if fixture_id is not None:
        statistics_params = {"fixture": fixture_id}
        status_code, statistics_payload, error = _request(api_key, "/fixtures/statistics", statistics_params)
        statistics_row = _audit_row(
            "statistics",
            "/fixtures/statistics",
            statistics_params,
            status_code,
            statistics_payload,
            error,
        )
        statistics_types = _statistics_types(statistics_payload)
        statistics_row["statistics_types_found"] = sorted(statistics_types)
        statistics_row["has_statistics"] = STATISTICS_NAMES.issubset(statistics_types)
        statistics_row["has_expected_goals"] = bool(statistics_types.intersection(EXPECTED_GOALS_NAMES))
        tests.append(statistics_row)

        lineups_params = {"fixture": fixture_id}
        status_code, lineups_payload, error = _request(api_key, "/fixtures/lineups", lineups_params)
        lineups_row = _audit_row("lineups", "/fixtures/lineups", lineups_params, status_code, lineups_payload, error)
        lineups_flags = _lineup_flags(lineups_payload)
        lineups_row.update(lineups_flags)
        lineups_row["has_lineups"] = all(lineups_flags.values())
        tests.append(lineups_row)
    else:
        tests.append(
            _audit_row("statistics", "/fixtures/statistics", {"fixture": None}, None, None, "fixture_id not found")
        )
        tests.append(_audit_row("lineups", "/fixtures/lineups", {"fixture": None}, None, None, "fixture_id not found"))

    injuries_params = {"league": league_id, "season": season}
    status_code, injuries_payload, error = _request(api_key, "/injuries", injuries_params)
    injuries_row = _audit_row("injuries", "/injuries", injuries_params, status_code, injuries_payload, error)
    injuries_row["has_injuries"] = injuries_row["success"] and injuries_row["results_count"] > 0
    tests.append(injuries_row)

    odds_params = {"league": league_id, "season": season}
    status_code, odds_payload, error = _request(api_key, "/odds", odds_params)
    odds_row = _audit_row("odds", "/odds", odds_params, status_code, odds_payload, error)
    odds_row["has_odds"] = odds_row["success"] and odds_row["results_count"] > 0
    tests.append(odds_row)

    report = {
        "base_url": BASE_URL,
        "season": season,
        "league_id": league_id,
        "detected_ligue1_id": _first_league_id(league_matches, "Ligue 1"),
        "fixture_id_sample": fixture_id,
        "tests": tests,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(report, output_path)
    return report


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    """Print a readable OK / LIMITED / ERROR summary."""
    print("API-Football coverage audit")
    print(f"Base URL: {report['base_url']}")
    print(f"League id tested: {report['league_id']}")
    print(f"Season tested: {report['season']}")
    print(f"Detected Ligue 1 id from /leagues: {report['detected_ligue1_id']}")
    print(f"Fixture sample id: {report['fixture_id_sample']}")
    print("\nTests:")
    for row in report["tests"]:
        status = _status_label(row)
        errors = "; ".join(row["errors"]) if row["errors"] else ""
        print(
            f"- {status} | {row['test_name']} | HTTP {row['http_status']} | "
            f"results={row['results_count']}"
            + (f" | errors={errors}" if errors else "")
        )
    print(f"\nSaved API-Football coverage audit to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit API-Football coverage for Ligue 1 enrichment.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the coverage audit."""
    args = parse_args()
    audit_api_football_coverage(league_id=args.league_id, season=args.season, output_path=args.output)


if __name__ == "__main__":
    main()
