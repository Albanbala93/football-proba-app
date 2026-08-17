"""Test the Sportmonks Football API connection without persisting project data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "https://api.sportmonks.com/v3/football/leagues"
REQUEST_TIMEOUT_SECONDS = 20


def _response_error_message(payload: dict[str, Any]) -> str | None:
    """Extract a readable error message from common Sportmonks response shapes."""
    if isinstance(payload.get("message"), str):
        return payload["message"]

    errors = payload.get("errors")
    if isinstance(errors, dict):
        messages = []
        for value in errors.values():
            if isinstance(value, list):
                messages.extend(str(item) for item in value)
            elif value:
                messages.append(str(value))
        if messages:
            return "; ".join(messages)

    if isinstance(errors, list) and errors:
        return "; ".join(str(error) for error in errors)

    return None


def _count_data_items(payload: dict[str, Any]) -> int:
    """Return the number of items in the response data field."""
    data = payload.get("data")
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return 1
    return 0


def test_sportmonks_connection(endpoint: str = DEFAULT_ENDPOINT) -> int:
    """Call a read-only Sportmonks endpoint and print a compact diagnostic."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    if not api_token:
        print("Success: no")
        print("Error: SPORTMONKS_API_TOKEN is missing from .env")
        return 1

    print("Sportmonks connection test")
    print(f"Endpoint: {endpoint}")

    try:
        response = requests.get(
            endpoint,
            params={"api_token": api_token},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        print("HTTP status: not_available")
        print("Success: no")
        print(f"Error: network request failed: {exc}")
        return 1

    print(f"HTTP status: {response.status_code}")

    try:
        payload = response.json()
    except ValueError:
        print("Success: no")
        print("Error: response is not valid JSON")
        return 1

    top_level_keys = sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else []
    print(f"JSON keys: {', '.join(top_level_keys) if top_level_keys else 'none'}")

    if not response.ok:
        print("Success: no")
        error_message = _response_error_message(payload) if isinstance(payload, dict) else None
        if error_message:
            print(f"Error: {error_message}")
        else:
            print("Error: request failed without a structured error message")
        return 1

    if not isinstance(payload, dict):
        print("Success: no")
        print("Error: JSON response is not an object")
        return 1

    error_message = _response_error_message(payload)
    if error_message:
        print("Success: no")
        print(f"Error: {error_message}")
        return 1

    print("Success: yes")
    print(f"Items returned: {_count_data_items(payload)}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test Sportmonks Football API connectivity.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Read-only Sportmonks endpoint to test.")
    return parser.parse_args()


def main() -> int:
    """Run the Sportmonks connection test."""
    args = parse_args()
    return test_sportmonks_connection(endpoint=args.endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
