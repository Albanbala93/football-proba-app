"""Build Ligue 1 lineup strength / rotation features from API-Football data."""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEASON = 2025
DEFAULT_LEAGUE_ID = 61

LINEUPS_INPUT_TEMPLATE = "api_football_ligue1_{season}_lineups.csv"
INJURY_IMPACT_INPUT_TEMPLATE = "api_football_ligue1_{season}_injury_impact_features.csv"
OUTPUT_TEMPLATE = "api_football_ligue1_{season}_lineup_strength_features.csv"

LINEUP_OUTPUT_COLUMNS = [
    "lineup_available",
    "home_expected_starters_count",
    "away_expected_starters_count",
    "home_actual_regular_starters_in_xi",
    "away_actual_regular_starters_in_xi",
    "home_rotation_score",
    "away_rotation_score",
    "rotation_diff",
    "home_lineup_strength_score",
    "away_lineup_strength_score",
    "lineup_strength_diff",
]


def input_path(season: int, template: str) -> Path:
    """Return a processed input path."""
    return PROJECT_ROOT / "data" / "processed" / template.format(season=season)


def output_path(season: int) -> Path:
    """Return the lineup strength output path."""
    return PROJECT_ROOT / "data" / "processed" / OUTPUT_TEMPLATE.format(season=season)


def parse_datetime(value: Any) -> pd.Timestamp:
    """Parse a datetime value with UTC awareness."""
    return pd.to_datetime(value, errors="coerce", utc=True)


def strip_accents(value: Any) -> str:
    """Remove accents from a string-like value."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_text(value: Any) -> str:
    """Normalize text for loose matching."""
    text = strip_accents(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"['â€™`-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(st|st\.)\b", "saint", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_players(value: Any) -> list[dict[str, Any]]:
    """Parse a serialized lineup payload into a list of dictionaries."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def extract_player_fields(item: Any) -> dict[str, Any]:
    """Extract a player payload from either a nested or flat lineup item."""
    if not isinstance(item, dict):
        return {}
    player = item.get("player")
    if isinstance(player, dict):
        return player
    return item


def player_key(player_id: Any = None, player_name: Any = None) -> str | None:
    """Build a stable player key from id or name."""
    if player_id is not None and not pd.isna(player_id):
        text = str(player_id).strip()
        if text:
            try:
                numeric = float(text)
                if numeric.is_integer():
                    return str(int(numeric))
            except ValueError:
                pass
            return text
    name = normalize_text(player_name)
    return name if name else None


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV when it exists, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_inputs(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load injury-impact and lineups tables."""
    injury_path = input_path(season, INJURY_IMPACT_INPUT_TEMPLATE)
    lineups_path = input_path(season, LINEUPS_INPUT_TEMPLATE)
    if not injury_path.exists():
        raise FileNotFoundError(f"Injury impact file not found: {injury_path}")
    if not lineups_path.exists():
        raise FileNotFoundError(f"Lineups file not found: {lineups_path}")
    return load_csv(injury_path), load_csv(lineups_path)


def normalize_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert identifier columns to numeric values when possible."""
    if df.empty:
        return df
    data = df.copy()
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def prepare_lineups(lineups: pd.DataFrame) -> pd.DataFrame:
    """Normalize lineup rows and attach parsed dates."""
    if lineups.empty:
        return lineups

    data = normalize_ids(lineups.copy(), ["fixture_id", "team_id", "startXI_count", "substitutes_count"])
    if "date" in data.columns:
        data["date"] = parse_datetime(data["date"])
    if "is_home" in data.columns:
        data["is_home"] = data["is_home"].astype(str).str.lower().isin(["1", "true", "yes", "home"])
    else:
        data["is_home"] = False
    if "startXI_players" not in data.columns:
        data["startXI_players"] = "[]"
    if "substitutes_players" not in data.columns:
        data["substitutes_players"] = "[]"
    return data


def update_team_history(team_history: dict[Any, dict[str, dict[str, int]]], team_id: Any, lineup_row: pd.Series) -> None:
    """Update one team's player history using a lineup row."""
    squad_players = parse_players(lineup_row.get("startXI_players")) + parse_players(lineup_row.get("substitutes_players"))
    for item in squad_players:
        player = extract_player_fields(item)
        key = player_key(player.get("id"), player.get("name"))
        if not key:
            continue
        stats = team_history.setdefault(team_id, {}).setdefault(key, {"startxi": 0, "squad": 0})
        stats["squad"] += 1
    for item in parse_players(lineup_row.get("startXI_players")):
        player = extract_player_fields(item)
        key = player_key(player.get("id"), player.get("name"))
        if not key:
            continue
        stats = team_history.setdefault(team_id, {}).setdefault(key, {"startxi": 0, "squad": 0})
        stats["startxi"] += 1


def player_history_stats(team_history: dict[Any, dict[str, dict[str, int]]], team_id: Any, player_id: Any, player_name: Any) -> tuple[int, int, float]:
    """Return prior startXI, squad counts and starter rate for a player."""
    team_stats = team_history.get(team_id, {})
    keys: list[str] = []
    key_id = player_key(player_id, None)
    key_name = normalize_text(player_name)
    if key_id:
        keys.append(key_id)
    if key_name and key_name not in keys:
        keys.append(key_name)

    for key in keys:
        if key in team_stats:
            stats = team_stats[key]
            startxi = int(stats.get("startxi", 0))
            squad = int(stats.get("squad", 0))
            starter_rate = startxi / squad if squad > 0 else 0.0
            return startxi, squad, starter_rate

    return 0, 0, 0.0


def expected_starter_flag(starter_rate_before: float) -> bool:
    """Return whether a player is a habitual starter before the match."""
    return starter_rate_before >= 0.60


def build_side_features(
    team_history: dict[Any, dict[str, dict[str, int]]],
    team_id: Any,
    lineup_row: pd.Series,
) -> dict[str, Any]:
    """Build lineup strength features for one team and one fixture."""
    starters = parse_players(lineup_row.get("startXI_players"))
    squad = starters + parse_players(lineup_row.get("substitutes_players"))

    expected_starters_count = 0
    actual_regular_starters_in_xi = 0

    seen_squad_keys: set[str] = set()
    for item in squad:
        player = extract_player_fields(item)
        key = player_key(player.get("id"), player.get("name"))
        if not key or key in seen_squad_keys:
            continue
        seen_squad_keys.add(key)
        _, _, starter_rate_before = player_history_stats(team_history, team_id, player.get("id"), player.get("name"))
        if expected_starter_flag(starter_rate_before):
            expected_starters_count += 1

    seen_xi_keys: set[str] = set()
    for item in starters:
        player = extract_player_fields(item)
        key = player_key(player.get("id"), player.get("name"))
        if not key or key in seen_xi_keys:
            continue
        seen_xi_keys.add(key)
        _, _, starter_rate_before = player_history_stats(team_history, team_id, player.get("id"), player.get("name"))
        if expected_starter_flag(starter_rate_before):
            actual_regular_starters_in_xi += 1

    rotation_score = float(expected_starters_count - actual_regular_starters_in_xi)
    lineup_strength_score = float(actual_regular_starters_in_xi / expected_starters_count) if expected_starters_count > 0 else 0.0
    return {
        "expected_starters_count": int(expected_starters_count),
        "actual_regular_starters_in_xi": int(actual_regular_starters_in_xi),
        "rotation_score": rotation_score,
        "lineup_strength_score": lineup_strength_score,
    }


def build_api_football_ligue1_lineup_strength_features(season: int = DEFAULT_SEASON, league_id: int = DEFAULT_LEAGUE_ID) -> pd.DataFrame:
    """Build and save lineup strength features."""
    injury_impact, lineups = load_inputs(season)

    required = {"fixture_id", "date", "home_team_id", "away_team_id", "home_team_name", "away_team_name"}
    missing = sorted(required - set(injury_impact.columns))
    if missing:
        raise ValueError(f"Missing required columns in injury impact features: {', '.join(missing)}")

    if lineups.empty:
        raise ValueError("Lineups file is empty.")

    injury_data = injury_impact.copy()
    injury_data["match_datetime"] = parse_datetime(injury_data["date"])
    injury_data = injury_data.dropna(subset=["match_datetime"]).sort_values(["match_datetime", "fixture_id"]).reset_index(drop=True)

    lineups_data = prepare_lineups(lineups)
    lineups_data = lineups_data.dropna(subset=["fixture_id", "team_id"])

    team_history: dict[Any, dict[str, dict[str, int]]] = defaultdict(dict)
    lineup_rows: list[dict[str, Any]] = []
    available_fixture_count = 0

    for _, match_row in injury_data.iterrows():
        fixture_id = match_row["fixture_id"]
        home_team_id = match_row["home_team_id"]
        away_team_id = match_row["away_team_id"]

        home_rows = lineups_data[(lineups_data["fixture_id"] == fixture_id) & (lineups_data["is_home"] == True)].copy()  # noqa: E712
        away_rows = lineups_data[(lineups_data["fixture_id"] == fixture_id) & (lineups_data["is_home"] == False)].copy()  # noqa: E712

        if not home_rows.empty and not away_rows.empty:
            home_row = home_rows.iloc[0]
            away_row = away_rows.iloc[0]
            home_features = build_side_features(team_history, home_team_id, home_row)
            away_features = build_side_features(team_history, away_team_id, away_row)
            lineup_rows.append(
                {
                    "fixture_id": fixture_id,
                    "lineup_available": 1,
                    "home_expected_starters_count": home_features["expected_starters_count"],
                    "away_expected_starters_count": away_features["expected_starters_count"],
                    "home_actual_regular_starters_in_xi": home_features["actual_regular_starters_in_xi"],
                    "away_actual_regular_starters_in_xi": away_features["actual_regular_starters_in_xi"],
                    "home_rotation_score": home_features["rotation_score"],
                    "away_rotation_score": away_features["rotation_score"],
                    "rotation_diff": float(home_features["rotation_score"] - away_features["rotation_score"]),
                    "home_lineup_strength_score": home_features["lineup_strength_score"],
                    "away_lineup_strength_score": away_features["lineup_strength_score"],
                    "lineup_strength_diff": float(home_features["lineup_strength_score"] - away_features["lineup_strength_score"]),
                }
            )
            available_fixture_count += 1
        else:
            lineup_rows.append({"fixture_id": fixture_id, "lineup_available": 0})

        # Update histories after features are computed, even if only one side is available.
        if not home_rows.empty:
            update_team_history(team_history, home_team_id, home_rows.iloc[0])
        if not away_rows.empty:
            update_team_history(team_history, away_team_id, away_rows.iloc[0])

    lineup_features = pd.DataFrame(lineup_rows)
    merged = injury_data.merge(lineup_features, on="fixture_id", how="left")
    merged["lineup_available"] = merged["lineup_available"].fillna(0).astype(int)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce", utc=True)
        merged = merged.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    return merged


def _nan_rate(df: pd.DataFrame, columns: list[str]) -> float:
    """Return the NaN rate for a group of columns."""
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return 0.0
    return float(df[existing].isna().mean().mean())


def print_summary(features: pd.DataFrame, source_rows: int, available_fixture_count: int) -> None:
    """Print a readable build summary."""
    print("API-Football Ligue 1 lineup strength feature build")
    print(f"Matches read: {source_rows}")
    print(f"Matches with lineups available: {available_fixture_count}")
    print(f"Rows exported: {len(features)}")
    print("\nAverage rotation score:")
    print(f"- home: {float(pd.to_numeric(features.get('home_rotation_score'), errors='coerce').mean()):.4f}")
    print(f"- away: {float(pd.to_numeric(features.get('away_rotation_score'), errors='coerce').mean()):.4f}")
    print(f"- lineup strength diff NaN rate: {_nan_rate(features, ['lineup_strength_diff']):.2%}")

    print("\nTop 10 matches by absolute rotation_diff:")
    preview_columns = [
        column
        for column in [
            "fixture_id",
            "date",
            "home_team_name",
            "away_team_name",
            "home_rotation_score",
            "away_rotation_score",
            "rotation_diff",
            "lineup_strength_diff",
        ]
        if column in features.columns
    ]
    if preview_columns:
        top_rotation = features.reindex(columns=preview_columns).copy()
        if "rotation_diff" in top_rotation.columns:
            top_rotation["_abs"] = top_rotation["rotation_diff"].abs()
            print(top_rotation.sort_values("_abs", ascending=False).head(10).drop(columns=["_abs"]).to_string(index=False))
        else:
            print("none")
    else:
        print("none")

    print("\nNaN rates:")
    print(f"- lineup_available: {_nan_rate(features, ['lineup_available']):.2%}")
    print(f"- home_expected_starters_count: {_nan_rate(features, ['home_expected_starters_count']):.2%}")
    print(f"- away_expected_starters_count: {_nan_rate(features, ['away_expected_starters_count']):.2%}")
    print(f"- home_rotation_score: {_nan_rate(features, ['home_rotation_score']):.2%}")
    print(f"- away_rotation_score: {_nan_rate(features, ['away_rotation_score']):.2%}")
    print(f"- lineup_strength_diff: {_nan_rate(features, ['lineup_strength_diff']):.2%}")

    print("\nPreview:")
    preview_columns = [
        column
        for column in ["fixture_id", "date", "home_team_name", "away_team_name"] + LINEUP_OUTPUT_COLUMNS
        if column in features.columns
    ]
    print(features[preview_columns].head().to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build API-Football Ligue 1 lineup strength features.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    return parser.parse_args()


def main() -> None:
    """Run the lineup strength builder."""
    args = parse_args()
    features, source_path = load_inputs(args.season)
    available_fixture_count = 0
    # Rebuild to get the merged output and count from the full routine.
    output = build_api_football_ligue1_lineup_strength_features(season=args.season, league_id=args.league_id)
    destination = output_path(args.season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)

    # Count fixtures with lineups available from the output.
    if "lineup_available" in output.columns:
        available_fixture_count = int(output["lineup_available"].fillna(0).astype(int).sum())
    print_summary(output, source_rows=len(features), available_fixture_count=available_fixture_count)
    print(f"\nSaved lineup strength features to: {destination}")


if __name__ == "__main__":
    main()
