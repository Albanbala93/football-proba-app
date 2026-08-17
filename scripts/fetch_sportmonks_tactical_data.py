"""Fetch Sportmonks tactical fixture data for any accessible league and season."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.sportmonks.com/v3/football"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "external" / "sportmonks"
DEFAULT_LIMIT = 20
REQUEST_TIMEOUT_SECONDS = 30
FIXTURE_INCLUDES = ["participants", "formations", "lineups", "sidelined.sideline", "statistics", "scores"]


def league_slug(league_name: str) -> str:
    """Convert a league name into a stable filesystem slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", league_name.lower())
    return slug.strip("_") or "unknown_league"


def _load_api_token() -> str:
    """Load the Sportmonks API token from .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    if not api_token:
        raise RuntimeError("SPORTMONKS_API_TOKEN is missing from .env")
    return api_token


def _extract_error_message(payload: Any) -> str | None:
    """Extract a readable API error message."""
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
    path: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """Call Sportmonks and return JSON payload, error message, and HTTP status."""
    request_params = dict(params or {})
    request_params["api_token"] = api_token

    try:
        response = session.get(
            f"{BASE_URL}{path}",
            params=request_params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
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
    """Return payload data as a list of objects."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _contains_key(value: Any, target_keys: set[str]) -> bool:
    """Return whether a nested object contains one of the target keys."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if str(key).lower() in target_keys:
                return True
            if _contains_key(nested_value, target_keys):
                return True
    elif isinstance(value, list):
        return any(_contains_key(item, target_keys) for item in value)
    return False


def _coverage_flags(fixture_payload: dict[str, Any]) -> dict[str, bool]:
    """Return tactical coverage flags from a fixture payload."""
    return {
        "has_formations": _contains_key(fixture_payload, {"formations", "formation"}),
        "has_lineups": _contains_key(fixture_payload, {"lineups", "lineup"}),
        "has_sidelined": _contains_key(fixture_payload, {"sidelined", "sideline"}),
        "has_statistics": _contains_key(fixture_payload, {"statistics", "statistic"}),
    }


def _participant_name(participant: dict[str, Any]) -> str | None:
    """Return the best available participant/team name."""
    for key in ["name", "short_code", "common_name"]:
        value = participant.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _home_away_from_participants(fixture: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract home and away team names from Sportmonks participants when available."""
    participants = fixture.get("participants")
    if isinstance(participants, dict):
        participants = participants.get("data")

    home_team = None
    away_team = None
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            meta = participant.get("meta") if isinstance(participant.get("meta"), dict) else {}
            location = str(meta.get("location", "")).lower()
            name = _participant_name(participant)
            if location == "home":
                home_team = name
            elif location == "away":
                away_team = name

    if home_team and away_team:
        return home_team, away_team

    fixture_name = fixture.get("name")
    if isinstance(fixture_name, str) and " vs " in fixture_name:
        home, away = fixture_name.split(" vs ", 1)
        return home.strip() or home_team, away.strip() or away_team

    return home_team, away_team


def _fixture_date(fixture: dict[str, Any]) -> str | None:
    """Return the fixture date as YYYY-MM-DD when available."""
    starting_at = fixture.get("starting_at")
    if isinstance(starting_at, str) and starting_at:
        return starting_at[:10]
    return None


def _fixture_id(fixture: dict[str, Any]) -> int | None:
    """Return a fixture id as int when possible."""
    try:
        return int(fixture.get("id"))
    except (TypeError, ValueError):
        return None


def _fixture_data_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return one fixture data object from a fixture response payload."""
    fixtures = _data_list(payload)
    return fixtures[0] if fixtures else None


def _fixtures_from_schedule(payload: dict[str, Any] | None, league_id: int) -> list[dict[str, Any]]:
    """Extract fixtures from a Sportmonks season schedule response."""
    fixtures_by_id: dict[int, dict[str, Any]] = {}
    for stage in _data_list(payload):
        rounds = stage.get("rounds")
        if isinstance(rounds, dict):
            rounds = rounds.get("data")
        if not isinstance(rounds, list):
            continue

        for round_data in rounds:
            if not isinstance(round_data, dict):
                continue
            round_fixtures = round_data.get("fixtures")
            if isinstance(round_fixtures, dict):
                round_fixtures = round_fixtures.get("data")
            if not isinstance(round_fixtures, list):
                continue

            for fixture in round_fixtures:
                if not isinstance(fixture, dict):
                    continue
                if int(fixture.get("league_id") or 0) != league_id:
                    continue
                fixture_id = _fixture_id(fixture)
                if fixture_id is not None:
                    fixtures_by_id[fixture_id] = fixture

    return sorted(fixtures_by_id.values(), key=lambda fixture: (str(fixture.get("starting_at") or ""), fixture.get("id") or 0))


def fetch_tactical_data(
    league_id: int,
    season_id: int,
    league_name: str,
    limit: int = DEFAULT_LIMIT,
    force: bool = False,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> pd.DataFrame:
    """Fetch tactical fixture data and build an index CSV."""
    api_token = _load_api_token()
    output_dir = output_root / league_slug(league_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    rows = []
    downloaded_count = 0

    with requests.Session() as session:
        schedule_path = f"/schedules/seasons/{season_id}"
        schedule_payload, schedule_error, schedule_status = _request_json(session, api_token, schedule_path)
        if schedule_error:
            raise RuntimeError(
                f"Could not fetch schedule for season_id={season_id}: HTTP {schedule_status}, {schedule_error}"
            )

        fixtures = _fixtures_from_schedule(schedule_payload, league_id=league_id)
        selected_fixtures = fixtures[: max(limit, 0)]
        print(f"Fixtures found: {len(fixtures)}")
        print(f"Fixture limit: {len(selected_fixtures)}")

        for fixture in selected_fixtures:
            fixture_id = _fixture_id(fixture)
            if fixture_id is None:
                errors.append("Fixture without usable id skipped.")
                continue

            json_path = output_dir / f"fixture_{fixture_id}.json"
            if json_path.exists() and not force:
                try:
                    detail_payload = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    errors.append(f"Existing JSON is invalid and was skipped: {json_path}")
                    continue
            else:
                detail_payload, detail_error, detail_status = _request_json(
                    session,
                    api_token,
                    f"/fixtures/{fixture_id}",
                    params={"include": ";".join(FIXTURE_INCLUDES)},
                )
                if detail_error:
                    errors.append(f"fixture {fixture_id}: HTTP {detail_status}, {detail_error}")
                    continue
                json_path.write_text(json.dumps(detail_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                downloaded_count += 1

            fixture_detail = _fixture_data_from_payload(detail_payload) or fixture
            home_team, away_team = _home_away_from_participants(fixture_detail)
            flags = _coverage_flags(fixture_detail)
            rows.append(
                {
                    "sportmonks_fixture_id": fixture_id,
                    "date": _fixture_date(fixture_detail),
                    "league_id": league_id,
                    "league_name": league_name,
                    "season_id": season_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "has_formations": flags["has_formations"],
                    "has_lineups": flags["has_lineups"],
                    "has_sidelined": flags["has_sidelined"],
                    "has_statistics": flags["has_statistics"],
                    "json_path": str(json_path.relative_to(PROJECT_ROOT)),
                }
            )

    index = pd.DataFrame(
        rows,
        columns=[
            "sportmonks_fixture_id",
            "date",
            "league_id",
            "league_name",
            "season_id",
            "home_team",
            "away_team",
            "has_formations",
            "has_lineups",
            "has_sidelined",
            "has_statistics",
            "json_path",
        ],
    )
    index_path = output_dir / "fixtures_index.csv"
    index.to_csv(index_path, index=False)

    print("\nSportmonks tactical fetch")
    print(f"League: {league_name} ({league_id})")
    print(f"Season ID: {season_id}")
    print(f"Output directory: {output_dir}")
    print(f"Downloaded fixtures: {downloaded_count}")
    print(f"Fixtures with formations: {int(index['has_formations'].sum()) if not index.empty else 0}")
    print(f"Fixtures with lineups: {int(index['has_lineups'].sum()) if not index.empty else 0}")
    print(f"Fixtures with sidelined: {int(index['has_sidelined'].sum()) if not index.empty else 0}")
    print(f"Fixtures with statistics: {int(index['has_statistics'].sum()) if not index.empty else 0}")
    print(f"Index saved to: {index_path}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Errors: none")

    return index


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch Sportmonks tactical data for one league and season.")
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--season-id", type=int, required=True)
    parser.add_argument("--league-name", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the tactical fetch."""
    args = parse_args()
    league_name = args.league_name or f"league_{args.league_id}"

    try:
        fetch_tactical_data(
            league_id=args.league_id,
            season_id=args.season_id,
            league_name=league_name,
            limit=args.limit,
            force=args.force,
        )
    except Exception as exc:
        print("Sportmonks tactical fetch failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
