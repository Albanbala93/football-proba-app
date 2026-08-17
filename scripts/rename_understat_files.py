"""Rename local Understat files based on detected content."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "understat" / "Understat"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "reference" / "understat_rename_report.csv"
CSV_SEPARATOR_CANDIDATES = [",", ";", "\t"]
XG_COLUMNS = {"xg", "xga", "xpts", "xa", "xg90", "xa90", "xg_diff", "xgpersh", "xgapersh"}


def _slug(value: Any) -> str:
    """Return a lowercase filesystem-safe slug."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _read_csv_with_detected_separator(file_path: Path) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Read a CSV using the separator that yields the most columns."""
    best_df = None
    best_separator = None
    best_error = None
    for separator in CSV_SEPARATOR_CANDIDATES:
        try:
            df = pd.read_csv(file_path, sep=separator)
        except Exception as exc:
            best_error = str(exc)
            continue
        if best_df is None or len(df.columns) > len(best_df.columns):
            best_df = df
            best_separator = separator
    if best_df is None:
        return None, None, best_error or "could not read CSV with supported separators"
    return best_df, best_separator, None


def _normalized_columns(columns: list[str]) -> dict[str, str]:
    """Map normalized column names to original column names."""
    return {str(column).strip().lower().replace(" ", "_"): str(column) for column in columns}


def _detected_xg_columns(columns: list[str]) -> list[str]:
    """Return detected xG-related columns."""
    detected = []
    for column in columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in XG_COLUMNS:
            detected.append(str(column))
    return detected


def _has_any(columns: dict[str, str], names: set[str]) -> bool:
    """Return whether any normalized column is present."""
    return bool(set(columns).intersection(names))


def _detect_file_type(columns: list[str]) -> str:
    """Infer Understat file type from columns."""
    normalized = _normalized_columns(columns)

    has_date = _has_any(normalized, {"date", "datetime", "match_date"})
    has_home_away = _has_any(normalized, {"home_team", "h_team", "home"}) and _has_any(
        normalized,
        {"away_team", "a_team", "away"},
    )
    has_home_away_xg = _has_any(normalized, {"home_xg", "hxg", "xg_home"}) and _has_any(
        normalized,
        {"away_xg", "axg", "xg_away"},
    )
    if has_date and has_home_away and has_home_away_xg:
        return "match_stats"

    if _has_any(normalized, {"minute", "min"}) and _has_any(normalized, {"player", "player_name"}) and _has_any(
        normalized,
        {"result", "shot_result"},
    ) and _has_any(normalized, {"xg"}):
        return "shots"

    if _has_any(normalized, {"number", "position", "rank"}) and _has_any(
        normalized,
        {"team", "team_name"},
    ) and _has_any(normalized, {"wins"}) and _has_any(normalized, {"draws"}) and _has_any(
        normalized,
        {"loses", "losses"},
    ) and _has_any(normalized, {"points"}):
        return "league_table"

    if _has_any(normalized, {"player", "player_name"}) and _has_any(normalized, {"xg", "xa"}):
        return "player_stats"

    if _has_any(normalized, {"team", "team_name"}) and _has_any(normalized, {"xg"}) and _has_any(
        normalized,
        {"xga", "xpts"},
    ):
        return "team_season_stats"

    if (
        _has_any(normalized, {"number"})
        and _has_any(normalized, {"statistic"})
        and _has_any(normalized, {"shots"})
        and _has_any(normalized, {"goals"})
        and _has_any(normalized, {"shots_against"})
        and _has_any(normalized, {"goals_against"})
        and _has_any(normalized, {"xg"})
        and _has_any(normalized, {"xga"})
        and _has_any(normalized, {"xg_diff"})
    ):
        return "team_situational_stats"

    return "unknown"


def _classification_reason(file_type: str) -> str:
    """Return a human-readable reason for the inferred file type."""
    if file_type == "team_situational_stats":
        return "aggregated situational team statistics by formation/game_state/minute/situation/shot_type"
    return ""


def _first_value_for_column(df: pd.DataFrame, candidates: set[str]) -> str | None:
    """Return first non-empty value from the first matching column."""
    normalized = _normalized_columns([str(column) for column in df.columns])
    for candidate in candidates:
        column = normalized.get(candidate)
        if column is None:
            continue
        values = df[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            return str(values.iloc[0])
    return None


def _target_name(file_type: str, league: str | None, season: str | None, index: int, suffix: str) -> str:
    """Build target filename."""
    league_slug = _slug(league)
    season_slug = _slug(season)
    if league_slug and season_slug:
        return f"understat_{file_type}_{league_slug}_{season_slug}_{index:03d}{suffix}"
    return f"understat_{file_type}_{index:03d}{suffix}"


def _unique_target_path(input_dir: Path, filename: str, original_path: Path, planned_targets: set[Path]) -> Path:
    """Return a target path that does not overwrite existing or planned files."""
    target = input_dir / filename
    if target == original_path:
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 2
    while (target.exists() and target != original_path) or target in planned_targets:
        target = input_dir / f"{stem}_{counter:03d}{suffix}"
        counter += 1
    return target


def _inspect_file(file_path: Path, index: int) -> dict[str, Any]:
    """Inspect one Understat file and return rename metadata."""
    if file_path.suffix.lower() == ".csv":
        df, separator, error = _read_csv_with_detected_separator(file_path)
        if error or df is None:
            return {
                "rows": None,
                "columns": [],
                "detected_separator": separator,
                "file_type": "unknown",
                "detected_xg_columns": [],
                "league": None,
                "season": None,
                "reason": error,
            }
        columns = [str(column) for column in df.columns]
        file_type = _detect_file_type(columns)
        return {
            "rows": len(df),
            "columns": columns,
            "detected_separator": separator,
            "file_type": file_type,
            "detected_xg_columns": _detected_xg_columns(columns),
            "league": _first_value_for_column(df, {"league", "competition", "tournament"}),
            "season": _first_value_for_column(df, {"season", "year"}),
            "reason": _classification_reason(file_type),
        }

    return {
        "rows": None,
        "columns": [],
        "detected_separator": None,
        "file_type": "unknown",
        "detected_xg_columns": [],
        "league": None,
        "season": None,
        "reason": "JSON rename classification is not implemented yet",
    }


def rename_understat_files(
    input_dir: Path = DEFAULT_INPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    apply: bool = False,
) -> pd.DataFrame:
    """Build a rename plan and optionally apply it."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in {".csv", ".json"}
    )
    planned_targets: set[Path] = set()
    rows = []

    for index, file_path in enumerate(files, start=1):
        info = _inspect_file(file_path, index)
        filename = _target_name(info["file_type"], info["league"], info["season"], index, file_path.suffix.lower())
        target_path = _unique_target_path(input_dir, filename, file_path, planned_targets)
        planned_targets.add(target_path)

        if target_path == file_path:
            action = "skipped"
            reason = "already named as target"
        elif apply:
            file_path.rename(target_path)
            action = "renamed"
            reason = info["reason"]
        else:
            action = "dry_run"
            reason = info["reason"]

        rows.append(
            {
                "original_path": str(file_path.relative_to(PROJECT_ROOT)),
                "new_path": str(target_path.relative_to(PROJECT_ROOT)),
                "detected_separator": info["detected_separator"],
                "rows": info["rows"],
                "columns_count": len(info["columns"]),
                "file_type": info["file_type"],
                "detected_xg_columns": "; ".join(info["detected_xg_columns"]),
                "action": action,
                "reason": reason,
            }
        )
        print(f"{file_path.name} -> {target_path.name} [{action}]")

    report = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(f"\nSaved Understat rename report to: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Rename local Understat files based on detected content.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the Understat rename planner/applicator."""
    args = parse_args()
    try:
        rename_understat_files(input_dir=args.input_dir, apply=args.apply)
    except Exception as exc:
        print("Understat rename failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
