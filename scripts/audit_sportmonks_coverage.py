"""Audit Sportmonks Football API coverage for tactical data."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "sportmonks_coverage_audit.json"
BASE_URL = "https://api.sportmonks.com/v3/football"
REQUEST_TIMEOUT_SECONDS = 20
TACTICAL_INCLUDES = ["formations", "lineups", "sidelined.sideline", "statistics"]
EXPECTED_LINEUPS_INCLUDES = ["expectedLineups", "expected_lineups", "lineups"]


def _redacted_endpoint(path: str, params: dict[str, Any] | None = None) -> str:
    """Build a readable endpoint string without secrets."""
    if not params:
        return f"{BASE_URL}{path}"

    visible_params = {key: value for key, value in params.items() if key != "api_token"}
    if not visible_params:
        return f"{BASE_URL}{path}"
    query = "&".join(f"{key}={value}" for key, value in visible_params.items())
    return f"{BASE_URL}{path}?{query}"


def _extract_error_message(payload: Any) -> str | None:
    """Extract a useful API error message from common response shapes."""
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


def _data_info(payload: Any) -> tuple[str, int | None, Any]:
    """Return data type, count, and sample data for a Sportmonks payload."""
    if not isinstance(payload, dict) or "data" not in payload:
        return "missing", None, None

    data = payload["data"]
    if isinstance(data, list):
        sample = data[0] if data else None
        return "list", len(data), sample
    if isinstance(data, dict):
        return "object", 1, data
    if data is None:
        return "null", 0, None
    return type(data).__name__, None, data


def _sample_keys(sample: Any) -> list[str]:
    """Return sorted keys from a sample object."""
    if isinstance(sample, dict):
        return sorted(str(key) for key in sample.keys())
    return []


def _contains_key(value: Any, target_keys: set[str]) -> bool:
    """Return whether any nested dict/list contains a target key."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if str(key).lower() in target_keys:
                return True
            if _contains_key(nested_value, target_keys):
                return True
    elif isinstance(value, list):
        return any(_contains_key(item, target_keys) for item in value)
    return False


def _coverage_flags(payload: Any) -> dict[str, bool]:
    """Detect tactical data families in a JSON payload."""
    return {
        "has_formations": _contains_key(payload, {"formations", "formation"}),
        "has_lineups": _contains_key(payload, {"lineups", "lineup"}),
        "has_sidelined": _contains_key(payload, {"sidelined", "sideline"}),
        "has_statistics": _contains_key(payload, {"statistics", "statistic"}),
        "has_expected_lineups": _contains_key(payload, {"expectedlineups", "expected_lineups", "expectedlineup"}),
    }


def _request_json(
    session: requests.Session,
    api_token: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> tuple[int | None, Any, str | None]:
    """Call Sportmonks and return status, JSON payload, and network/parse errors."""
    request_params = dict(params or {})
    request_params["api_token"] = api_token

    try:
        response = session.get(
            f"{BASE_URL}{path}",
            params=request_params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return None, None, f"network request failed: {exc}"

    try:
        payload = response.json()
    except ValueError:
        return response.status_code, None, "response is not valid JSON"

    return response.status_code, payload, None


def _build_result(
    test_name: str,
    endpoint: str,
    http_status: int | None,
    payload: Any,
    request_error: str | None,
) -> dict[str, Any]:
    """Convert one API response into the audit result schema."""
    data_type, data_count, sample = _data_info(payload)
    response_ok = http_status is not None and 200 <= http_status < 300
    api_error = _extract_error_message(payload)
    error_message = request_error or api_error
    flags = _coverage_flags(payload)

    return {
        "test_name": test_name,
        "endpoint": endpoint,
        "http_status": http_status,
        "success": response_ok and error_message is None,
        "error_message": error_message,
        "top_level_json_keys": sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else [],
        "data_type": data_type,
        "data_count": data_count,
        "has_formations": flags["has_formations"],
        "has_lineups": flags["has_lineups"],
        "has_sidelined": flags["has_sidelined"],
        "has_statistics": flags["has_statistics"],
        "has_expected_lineups": flags["has_expected_lineups"],
        "sample_keys": _sample_keys(sample),
    }


def _status_label(result: dict[str, Any]) -> str:
    """Return OK, LIMITED, or ERROR for console reporting."""
    if not result["success"]:
        return "ERROR"
    has_any_coverage = any(
        result[key]
        for key in [
            "has_formations",
            "has_lineups",
            "has_sidelined",
            "has_statistics",
            "has_expected_lineups",
        ]
    )
    return "OK" if has_any_coverage else "LIMITED"


def _latest_fixture_id(payload: Any) -> int | None:
    """Extract one fixture id from a fixtures response."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    fixture = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
    if not isinstance(fixture, dict):
        return None

    fixture_id = fixture.get("id")
    try:
        return int(fixture_id)
    except (TypeError, ValueError):
        return None


def audit_sportmonks_coverage(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Run Sportmonks coverage tests and persist a JSON report."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_token = os.getenv("SPORTMONKS_API_TOKEN")
    if not api_token:
        raise RuntimeError("SPORTMONKS_API_TOKEN is missing from .env")

    results = []
    with requests.Session() as session:
        latest_path = "/fixtures/latest"
        latest_status, latest_payload, latest_error = _request_json(session, api_token, latest_path)
        latest_result = _build_result(
            test_name="latest_fixtures",
            endpoint=_redacted_endpoint(latest_path),
            http_status=latest_status,
            payload=latest_payload,
            request_error=latest_error,
        )
        results.append(latest_result)

        fixture_id = _latest_fixture_id(latest_payload)
        if fixture_id is not None:
            tactical_path = f"/fixtures/{fixture_id}"
            tactical_params = {"include": ";".join(TACTICAL_INCLUDES)}
            tactical_status, tactical_payload, tactical_error = _request_json(
                session,
                api_token,
                tactical_path,
                tactical_params,
            )
            results.append(
                _build_result(
                    test_name="fixture_tactical_includes",
                    endpoint=_redacted_endpoint(tactical_path, tactical_params),
                    http_status=tactical_status,
                    payload=tactical_payload,
                    request_error=tactical_error,
                )
            )

            expected_params = {"include": ";".join(EXPECTED_LINEUPS_INCLUDES)}
            expected_status, expected_payload, expected_error = _request_json(
                session,
                api_token,
                tactical_path,
                expected_params,
            )
            results.append(
                _build_result(
                    test_name="fixture_expected_lineups_probe",
                    endpoint=_redacted_endpoint(tactical_path, expected_params),
                    http_status=expected_status,
                    payload=expected_payload,
                    request_error=expected_error,
                )
            )
        else:
            results.append(
                {
                    "test_name": "fixture_tactical_includes",
                    "endpoint": None,
                    "http_status": None,
                    "success": False,
                    "error_message": "No fixture id available from latest_fixtures test.",
                    "top_level_json_keys": [],
                    "data_type": "missing",
                    "data_count": None,
                    "has_formations": False,
                    "has_lineups": False,
                    "has_sidelined": False,
                    "has_statistics": False,
                    "has_expected_lineups": False,
                    "sample_keys": [],
                }
            )
            results.append(
                {
                    "test_name": "fixture_expected_lineups_probe",
                    "endpoint": None,
                    "http_status": None,
                    "success": False,
                    "error_message": "No fixture id available from latest_fixtures test.",
                    "top_level_json_keys": [],
                    "data_type": "missing",
                    "data_count": None,
                    "has_formations": False,
                    "has_lineups": False,
                    "has_sidelined": False,
                    "has_statistics": False,
                    "has_expected_lineups": False,
                    "sample_keys": [],
                }
            )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": BASE_URL,
        "tests": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(results, output_path)
    return report


def print_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    """Print a readable audit summary."""
    print("Sportmonks coverage audit")
    for result in results:
        status = _status_label(result)
        print(f"\n{status} - {result['test_name']}")
        print(f"Endpoint: {result['endpoint']}")
        print(f"HTTP status: {result['http_status']}")
        print(f"Data: {result['data_type']} ({result['data_count']})")
        if result.get("error_message"):
            print(f"Error: {result['error_message']}")
        print(
            "Coverage: "
            f"formations={result['has_formations']}, "
            f"lineups={result['has_lineups']}, "
            f"sidelined={result['has_sidelined']}, "
            f"statistics={result['has_statistics']}, "
            f"expected_lineups={result['has_expected_lineups']}"
        )
        if result["sample_keys"]:
            print(f"Sample keys: {', '.join(result['sample_keys'])}")

    print(f"\nSaved Sportmonks coverage audit to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit Sportmonks tactical data coverage.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the Sportmonks coverage audit."""
    args = parse_args()
    try:
        audit_sportmonks_coverage(output_path=args.output)
    except Exception as exc:
        print("Sportmonks coverage audit")
        print("ERROR - audit failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
