"""Test API-Football / API-Sports connectivity without persisting project data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_ENDPOINT = "/status"
REQUEST_TIMEOUT_SECONDS = 20
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


def _endpoint_url(endpoint: str) -> str:
    """Build an API-Football URL from a relative or absolute endpoint."""
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"{BASE_URL}/{endpoint.lstrip('/')}"


def _sanitize_payload(value: Any) -> Any:
    """Recursively remove sensitive-looking values from a JSON payload."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_KEYS or any(secret in key_text.lower() for secret in SENSITIVE_KEYS):
                sanitized[key_text] = "[redacted]"
            else:
                sanitized[key_text] = _sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    return value


def _extract_errors(payload: dict[str, Any]) -> list[str]:
    """Extract API-Football error messages from common response shapes."""
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


def _extract_quota(payload: dict[str, Any]) -> dict[str, Any]:
    """Return quota-related response fields when API-Football provides them."""
    response = payload.get("response")
    if isinstance(response, dict):
        quota = {
            key: response.get(key)
            for key in ["requests", "limit_day", "current", "remaining"]
            if key in response
        }
        if quota:
            return quota

    return {
        key: payload.get(key)
        for key in ["requests", "limit_day", "current", "remaining"]
        if key in payload
    }


def test_api_football_connection(endpoint: str = DEFAULT_ENDPOINT) -> int:
    """Call a read-only API-Football endpoint and print a compact diagnostic."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = (os.getenv("API_FOOTBALL_KEY") or "").strip()
    if not api_key:
        print("API-Football connection test")
        print("HTTP status: not_available")
        print("Success: no")
        print("Error: API_FOOTBALL_KEY is missing from .env")
        return 1

    url = _endpoint_url(endpoint)
    print("API-Football connection test")
    print(f"Endpoint: {url}")

    try:
        response = requests.get(
            url,
            headers={"x-apisports-key": api_key},
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

    sanitized_payload = _sanitize_payload(payload)
    print("Response JSON cleaned:")
    print(sanitized_payload)

    if not isinstance(payload, dict):
        print("Success: no")
        print("Error: JSON response is not an object")
        return 1

    quota = _extract_quota(payload)
    if quota:
        print(f"Quota: {_sanitize_payload(quota)}")
    else:
        print("Quota: not_available")

    errors = _extract_errors(payload)
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")

    success = response.ok and not errors
    print(f"Success: {'yes' if success else 'no'}")
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test API-Football / API-Sports connectivity.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Endpoint to test. Defaults to /status.")
    return parser.parse_args()


def main() -> int:
    """Run the API-Football connection test."""
    args = parse_args()
    return test_api_football_connection(endpoint=args.endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
