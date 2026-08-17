"""Aggregate Understat player match tables into team and match xG tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "external" / "understat" / "player_match_stats"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "external" / "understat" / "match_player_tables_index.csv"
DEFAULT_TEAM_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "understat_team_match_aggregates.csv"
DEFAULT_MATCH_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "understat_matches_normalized.csv"
CSV_SEPARATOR_CANDIDATES = [",", ";", "\t", "|"]
INDEX_REQUIRED_COLUMNS = {
    "file_path",
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "team",
    "is_home",
}
FINAL_MATCH_COLUMNS = [
    "date",
    "league",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_xa",
    "away_xa",
    "home_players_count",
    "away_players_count",
    "source_files",
]


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


def _read_table(file_path: Path) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Read a supported player table and return data, detected separator, and error."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_with_detected_separator(file_path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(file_path), None, None
        except Exception as exc:
            return None, None, str(exc)
    return None, None, f"unsupported file type: {suffix}"


def _normalized_columns(columns: list[Any]) -> dict[str, str]:
    """Map normalized names to original column names."""
    return {str(column).strip().lower().replace(" ", "_"): str(column) for column in columns}


def _numeric_sum(df: pd.DataFrame, normalized: dict[str, str], column_name: str) -> float:
    """Return a numeric sum for a normalized column, defaulting to zero if absent."""
    column = normalized.get(column_name)
    if column is None:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _players_count(df: pd.DataFrame, normalized: dict[str, str]) -> int:
    """Count players when a player column exists, otherwise count table rows."""
    player_column = normalized.get("player")
    if player_column is None:
        return int(len(df))
    players = df[player_column].dropna().astype(str).str.strip()
    return int((players != "").sum())


def _parse_is_home(value: Any, team: Any, home_team: Any, away_team: Any) -> bool | None:
    """Parse home/away flags, falling back to the team name when possible."""
    if pd.isna(value):
        text = ""
    else:
        text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "home", "h", "domicile"}:
        return True
    if text in {"false", "0", "no", "n", "away", "a", "exterieur"}:
        return False

    team_text = str(team or "").strip().lower()
    home_text = str(home_team or "").strip().lower()
    away_text = str(away_team or "").strip().lower()
    if team_text and team_text == home_text:
        return True
    if team_text and team_text == away_text:
        return False
    return None


def _resolve_file_path(raw_path: Any, input_dir: Path) -> Path:
    """Resolve index file paths from absolute paths, project-relative paths, or input-dir paths."""
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    input_relative = input_dir / path
    if input_relative.exists():
        return input_relative

    return project_relative


def _display_path(path: Path) -> str:
    """Return a readable path, relative to the project when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def aggregate_team_tables(index_path: Path, input_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Aggregate one player-level file per team/match into team-level rows."""
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = pd.read_csv(index_path)
    missing_index_columns = sorted(INDEX_REQUIRED_COLUMNS.difference(index.columns))
    if missing_index_columns:
        raise ValueError(f"Index is missing required columns: {', '.join(missing_index_columns)}")

    rows = []
    missing_files = []
    read_errors = []

    for index_row in index.to_dict("records"):
        file_path = _resolve_file_path(index_row["file_path"], input_dir)
        display_path = _display_path(file_path)

        if not file_path.exists():
            missing_files.append(display_path)
            continue

        df, detected_separator, error = _read_table(file_path)
        if error or df is None:
            read_errors.append(f"{display_path}: {error}")
            continue

        normalized = _normalized_columns(list(df.columns))
        is_home = _parse_is_home(
            index_row["is_home"],
            index_row["team"],
            index_row["home_team"],
            index_row["away_team"],
        )

        rows.append(
            {
                "date": index_row["date"],
                "league": index_row["league"],
                "season": index_row["season"],
                "home_team": index_row["home_team"],
                "away_team": index_row["away_team"],
                "team": index_row["team"],
                "is_home": is_home,
                "goals": _numeric_sum(df, normalized, "goals"),
                "xg": _numeric_sum(df, normalized, "xg"),
                "shots": _numeric_sum(df, normalized, "shots"),
                "xa": _numeric_sum(df, normalized, "xa"),
                "players_count": _players_count(df, normalized),
                "source_file": display_path,
                "detected_separator": detected_separator,
            }
        )

    return pd.DataFrame(rows), missing_files, read_errors


def _side_aggregate(side_rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    """Aggregate all rows for one match side."""
    if side_rows.empty:
        return {
            f"{prefix}_goals": pd.NA,
            f"{prefix}_xg": pd.NA,
            f"{prefix}_shots": pd.NA,
            f"{prefix}_xa": pd.NA,
            f"{prefix}_players_count": pd.NA,
        }

    return {
        f"{prefix}_goals": side_rows["goals"].sum(),
        f"{prefix}_xg": side_rows["xg"].sum(),
        f"{prefix}_shots": side_rows["shots"].sum(),
        f"{prefix}_xa": side_rows["xa"].sum(),
        f"{prefix}_players_count": side_rows["players_count"].sum(),
    }


def normalize_matches(team_aggregates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot team-level aggregates into one row per match."""
    if team_aggregates.empty:
        return pd.DataFrame(columns=FINAL_MATCH_COLUMNS), pd.DataFrame()

    match_keys = ["date", "league", "season", "home_team", "away_team"]
    rows = []
    incomplete_rows = []

    for key_values, match_rows in team_aggregates.groupby(match_keys, dropna=False, sort=False):
        match = dict(zip(match_keys, key_values))
        home_rows = match_rows[match_rows["is_home"] == True]  # noqa: E712
        away_rows = match_rows[match_rows["is_home"] == False]  # noqa: E712
        missing_sides = []
        if home_rows.empty:
            missing_sides.append("home")
        if away_rows.empty:
            missing_sides.append("away")

        source_files = sorted(str(path) for path in match_rows["source_file"].dropna().unique())
        row = {
            **match,
            **_side_aggregate(home_rows, "home"),
            **_side_aggregate(away_rows, "away"),
            "source_files": " | ".join(source_files),
        }
        rows.append(row)

        if missing_sides:
            incomplete_rows.append({**match, "missing_sides": ", ".join(missing_sides)})

    matches = pd.DataFrame(rows)
    if not matches.empty:
        matches = matches.reindex(columns=FINAL_MATCH_COLUMNS)
    return matches, pd.DataFrame(incomplete_rows)


def print_summary(
    team_aggregates: pd.DataFrame,
    matches: pd.DataFrame,
    missing_files: list[str],
    read_errors: list[str],
    incomplete_matches: pd.DataFrame,
    team_output_path: Path,
    match_output_path: Path,
) -> None:
    """Print a concise run summary and required diagnostics."""
    print("Understat player match aggregation")
    print(f"Team-level rows: {len(team_aggregates)}")
    print(f"Match-level rows: {len(matches)}")

    print(f"\nMissing files: {len(missing_files)}")
    for file_path in missing_files:
        print(f"- {file_path}")

    print(f"\nUnreadable files: {len(read_errors)}")
    for error in read_errors:
        print(f"- {error}")

    print(f"\nIncomplete matches: {len(incomplete_matches)}")
    for row in incomplete_matches.to_dict("records"):
        print(
            "- "
            f"{row['date']} | {row['league']} | {row['season']} | "
            f"{row['home_team']} vs {row['away_team']} | missing: {row['missing_sides']}"
        )

    print(f"\nSaved team aggregates to: {team_output_path}")
    print(f"Saved normalized matches to: {match_output_path}")


def run(
    index_path: Path = DEFAULT_INDEX_PATH,
    input_dir: Path = DEFAULT_INPUT_DIR,
    team_output_path: Path = DEFAULT_TEAM_OUTPUT_PATH,
    match_output_path: Path = DEFAULT_MATCH_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build team-level and match-level Understat aggregate CSV files."""
    team_aggregates, missing_files, read_errors = aggregate_team_tables(index_path=index_path, input_dir=input_dir)
    matches, incomplete_matches = normalize_matches(team_aggregates)

    team_output_path.parent.mkdir(parents=True, exist_ok=True)
    match_output_path.parent.mkdir(parents=True, exist_ok=True)
    team_aggregates.to_csv(team_output_path, index=False)
    matches.to_csv(match_output_path, index=False)

    print_summary(
        team_aggregates=team_aggregates,
        matches=matches,
        missing_files=missing_files,
        read_errors=read_errors,
        incomplete_matches=incomplete_matches,
        team_output_path=team_output_path,
        match_output_path=match_output_path,
    )
    return team_aggregates, matches


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate Understat player-level match tables into team and match xG CSV files.",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--team-output", type=Path, default=DEFAULT_TEAM_OUTPUT_PATH)
    parser.add_argument("--match-output", type=Path, default=DEFAULT_MATCH_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the aggregation CLI."""
    args = parse_args()
    try:
        run(
            index_path=args.index,
            input_dir=args.input_dir,
            team_output_path=args.team_output,
            match_output_path=args.match_output,
        )
    except Exception as exc:
        print("Understat player match aggregation failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
