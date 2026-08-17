"""Build Ligue 1 injury impact features from API-Football data."""

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

PREMATCH_INPUT_TEMPLATE = "api_football_ligue1_{season}_prematch_features.csv"
LINEUPS_INPUT_TEMPLATE = "api_football_ligue1_{season}_lineups.csv"
INJURIES_INPUT_TEMPLATE = "api_football_ligue1_{season}_injuries.csv"
OUTPUT_TEMPLATE = "api_football_ligue1_{season}_injury_impact_features.csv"

INJURY_OUTPUT_COLUMNS = [
    "home_injury_impact_score",
    "away_injury_impact_score",
    "injury_impact_diff",
    "home_likely_starter_injuries_count",
    "away_likely_starter_injuries_count",
    "likely_starter_injuries_diff",
]


def input_path(season: int, template: str) -> Path:
    """Return a processed API-Football input path."""
    return PROJECT_ROOT / "data" / "processed" / template.format(season=season)


def output_path(season: int) -> Path:
    """Return the injury impact output path."""
    return PROJECT_ROOT / "data" / "processed" / OUTPUT_TEMPLATE.format(season=season)


def strip_accents(value: Any) -> str:
    """Remove accents from a string-like value."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_text(value: Any) -> str:
    """Normalize text for loose matching."""
    text = strip_accents(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"['’`-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(st|st\.)\b", "saint", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_datetime(value: Any) -> pd.Timestamp:
    """Parse a datetime value with UTC awareness."""
    return pd.to_datetime(value, errors="coerce", utc=True)


def parse_players(value: Any) -> list[dict[str, Any]]:
    """Parse a serialized players payload into a list of dictionaries."""
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


def parse_lineup_player_list(value: Any) -> list[dict[str, Any]]:
    """Parse lineup player payloads into dicts."""
    return parse_players(value)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV when it exists, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_and_prepare_inputs(season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prematch, lineups and injuries data."""
    prematch_path = input_path(season, PREMATCH_INPUT_TEMPLATE)
    lineups_path = input_path(season, LINEUPS_INPUT_TEMPLATE)
    injuries_path = input_path(season, INJURIES_INPUT_TEMPLATE)

    if not prematch_path.exists():
        raise FileNotFoundError(f"Prematch features file not found: {prematch_path}")

    prematch = load_csv(prematch_path)
    lineups = load_csv(lineups_path)
    injuries = load_csv(injuries_path)
    return prematch, lineups, injuries


def normalize_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert a set of identifier columns to numeric values when possible."""
    if df.empty:
        return df
    data = df.copy()
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def normalize_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    """Normalize injury rows and deduplicate by fixture/team/player."""
    if injuries.empty:
        return injuries

    data = injuries.copy()
    data = normalize_ids(data, ["fixture_id", "league_id", "season", "team_id", "player_id"])
    if "date" in data.columns:
        data["date"] = parse_datetime(data["date"])
    if "player_name" in data.columns:
        data["player_name_key"] = data["player_name"].apply(normalize_text)
    else:
        data["player_name_key"] = ""
    data["player_key"] = data.apply(lambda row: player_key(row.get("player_id"), row.get("player_name")), axis=1)
    data = data.dropna(subset=["fixture_id", "team_id"])
    data = data.drop_duplicates(subset=["fixture_id", "team_id", "player_key"], keep="first")
    return data


def prepare_lineups(lineups: pd.DataFrame, prematch: pd.DataFrame) -> pd.DataFrame:
    """Attach fixture dates to lineups and normalize identifiers."""
    if lineups.empty:
        return lineups

    data = normalize_ids(lineups.copy(), ["fixture_id", "team_id", "startXI_count", "substitutes_count"])
    fixture_dates = prematch[["fixture_id", "date"]].copy() if "fixture_id" in prematch.columns and "date" in prematch.columns else pd.DataFrame()
    if not fixture_dates.empty:
        fixture_dates = normalize_ids(fixture_dates, ["fixture_id"])
        fixture_dates["date"] = parse_datetime(fixture_dates["date"])
        data = data.merge(fixture_dates, on="fixture_id", how="left", suffixes=("", "_fixture"))
    if "date" in data.columns:
        data["date"] = parse_datetime(data["date"])
    if "home_team_name" not in data.columns and "team_name" in data.columns:
        data["home_team_name"] = data["team_name"]
    return data


def update_team_history(team_history: dict[str, dict[str, int]], lineup_row: dict[str, Any]) -> None:
    """Update one team's player history using a lineup row."""
    starters = parse_lineup_player_list(lineup_row.get("startXI_players"))
    substitutes = parse_lineup_player_list(lineup_row.get("substitutes_players"))

    for item in starters:
        player = item.get("player", {}) if isinstance(item, dict) else {}
        if not isinstance(player, dict):
            player = {}
        for key in filter(None, [player_key(player.get("id"), None), normalize_text(player.get("name"))]):
            stats = team_history.setdefault(key, {"startxi": 0, "group": 0})
            stats["startxi"] += 1
            stats["group"] += 1

    for item in substitutes:
        player = item.get("player", {}) if isinstance(item, dict) else {}
        if not isinstance(player, dict):
            player = {}
        for key in filter(None, [player_key(player.get("id"), None), normalize_text(player.get("name"))]):
            stats = team_history.setdefault(key, {"startxi": 0, "group": 0})
            stats["group"] += 1


def player_history_stats(team_history: dict[str, dict[str, int]], player_id: Any, player_name: Any) -> tuple[int, int, float]:
    """Return prior startXI, group counts and starter rate for a player."""
    keys: list[str] = []
    key_id = player_key(player_id, None)
    key_name = normalize_text(player_name)
    if key_id:
        keys.append(key_id)
    if key_name and key_name not in keys:
        keys.append(key_name)

    for key in keys:
        if key in team_history:
            stats = team_history[key]
            startxi = int(stats.get("startxi", 0))
            group = int(stats.get("group", 0))
            starter_rate = startxi / group if group > 0 else 0.0
            return startxi, group, starter_rate

    return 0, 0, 0.0


def classify_absence_weight(starter_rate_before: float, known_player: bool) -> float:
    """Assign a simple injury weight from prior starter rate."""
    if not known_player:
        return 0.5
    if starter_rate_before >= 0.70:
        return 1.5
    if starter_rate_before >= 0.30:
        return 1.0
    return 0.5


def build_side_features(
    team_history: dict[str, dict[str, int]],
    injuries_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build side-level injury features for one team and one fixture."""
    unique_injuries: dict[str, dict[str, Any]] = {}
    for idx, injury in enumerate(injuries_rows):
        key = injury.get("player_key")
        if not key:
            key = player_key(injury.get("player_id"), injury.get("player_name"))
        if key:
            unique_injuries[str(key)] = injury
        else:
            unique_injuries[f"unknown_{idx}"] = injury

    injuries_count = len(unique_injuries)
    likely_starter_injuries_count = 0
    injury_impact_score = 0.0

    for injury in unique_injuries.values():
        startxi_count, group_count, starter_rate_before = player_history_stats(
            team_history,
            injury.get("player_id"),
            injury.get("player_name"),
        )
        known_player = (startxi_count > 0) or (group_count > 0)
        weight = classify_absence_weight(starter_rate_before, known_player)
        if starter_rate_before >= 0.30:
            likely_starter_injuries_count += 1
        injury_impact_score += weight

    return {
        "injuries_count": injuries_count,
        "likely_starter_injuries_count": likely_starter_injuries_count,
        "injury_impact_score": round(float(injury_impact_score), 3),
    }


def build_injury_impact_features(season: int = DEFAULT_SEASON, league_id: int = DEFAULT_LEAGUE_ID) -> pd.DataFrame:
    """Build match-level injury impact features and merge them with prematch features."""
    prematch, lineups, injuries = load_and_prepare_inputs(season)
    if prematch.empty:
        raise ValueError("Prematch features are empty.")
    if "fixture_id" not in prematch.columns:
        raise ValueError("Prematch features must contain fixture_id.")

    prematch_data = normalize_ids(prematch.copy(), ["fixture_id", "league_id", "season", "home_team_id", "away_team_id"])
    if "date" in prematch_data.columns:
        prematch_data["date"] = parse_datetime(prematch_data["date"])
    if "league_id" in prematch_data.columns:
        prematch_data = prematch_data[prematch_data["league_id"].fillna(DEFAULT_LEAGUE_ID).astype(int) == league_id].copy()
    prematch_data = prematch_data.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    lineups = prepare_lineups(lineups, prematch_data)
    injuries = normalize_injuries(injuries)
    if "league_id" in injuries.columns:
        injuries = injuries[injuries["league_id"].fillna(DEFAULT_LEAGUE_ID).astype(int) == league_id].copy()

    lineups_lookup: dict[tuple[Any, Any], dict[str, Any]] = {}
    if not lineups.empty:
        for row in lineups.to_dict("records"):
            fixture_id = row.get("fixture_id")
            team_id = row.get("team_id")
            if pd.isna(fixture_id) or pd.isna(team_id):
                continue
            lineups_lookup[(fixture_id, team_id)] = row

    injuries_lookup: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    if not injuries.empty:
        for row in injuries.to_dict("records"):
            fixture_id = row.get("fixture_id")
            team_id = row.get("team_id")
            if pd.isna(fixture_id) or pd.isna(team_id):
                continue
            injuries_lookup[(fixture_id, team_id)].append(row)

    team_histories: dict[Any, dict[str, dict[str, int]]] = defaultdict(dict)
    output_rows: list[dict[str, Any]] = []

    for row in prematch_data.to_dict("records"):
        fixture_id = row.get("fixture_id")
        home_team_id = row.get("home_team_id")
        away_team_id = row.get("away_team_id")

        home_injuries = injuries_lookup.get((fixture_id, home_team_id), [])
        away_injuries = injuries_lookup.get((fixture_id, away_team_id), [])

        home_history = team_histories.get(home_team_id, {})
        away_history = team_histories.get(away_team_id, {})
        home_features = build_side_features(home_history, home_injuries)
        away_features = build_side_features(away_history, away_injuries)

        output_rows.append(
            {
                "fixture_id": fixture_id,
                "home_injury_impact_score": home_features["injury_impact_score"],
                "away_injury_impact_score": away_features["injury_impact_score"],
                "injury_impact_diff": (
                    home_features["injury_impact_score"] - away_features["injury_impact_score"]
                    if pd.notna(home_features["injury_impact_score"]) and pd.notna(away_features["injury_impact_score"])
                    else 0.0
                ),
                "home_likely_starter_injuries_count": home_features["likely_starter_injuries_count"],
                "away_likely_starter_injuries_count": away_features["likely_starter_injuries_count"],
                "likely_starter_injuries_diff": (
                    home_features["likely_starter_injuries_count"] - away_features["likely_starter_injuries_count"]
                ),
            }
        )

        home_lineup = lineups_lookup.get((fixture_id, home_team_id))
        away_lineup = lineups_lookup.get((fixture_id, away_team_id))
        if home_lineup is not None:
            team_histories[home_team_id] = team_histories.get(home_team_id, {})
            update_team_history(team_histories[home_team_id], home_lineup)
        if away_lineup is not None:
            team_histories[away_team_id] = team_histories.get(away_team_id, {})
            update_team_history(team_histories[away_team_id], away_lineup)

    injury_features = pd.DataFrame(output_rows)
    merged = prematch.merge(injury_features, on="fixture_id", how="left")
    for column in INJURY_OUTPUT_COLUMNS:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    if "date" in merged.columns:
        merged = merged.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    return merged


def nan_rate(df: pd.DataFrame, columns: list[str]) -> float:
    """Return the mean NaN rate across a set of columns."""
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return 0.0
    return float(df[existing].isna().mean().mean())


def print_summary(features: pd.DataFrame, season: int, prematch_rows: int, lineups_rows: int, injuries_rows: int) -> None:
    """Print a readable build summary."""
    print("API-Football Ligue 1 injury impact feature build")
    print(f"Season: {season}")
    print(f"League ID: {DEFAULT_LEAGUE_ID}")
    print(f"Prematch rows read: {prematch_rows}")
    print(f"Lineups rows read: {lineups_rows}")
    print(f"Injuries rows read: {injuries_rows}")
    print(f"Rows exported: {len(features)}")
    if "date" in features.columns:
        print(f"Date min: {features['date'].min()}")
        print(f"Date max: {features['date'].max()}")

    print("\nAverage weighted injuries:")
    if "home_injury_impact_score" in features.columns:
        print(f"- home: {float(features['home_injury_impact_score'].mean()):.3f}")
    if "away_injury_impact_score" in features.columns:
        print(f"- away: {float(features['away_injury_impact_score'].mean()):.3f}")

    impact_column = None
    if {"home_injury_impact_score", "away_injury_impact_score"}.issubset(features.columns):
        features = features.copy()
        features["_combined_injury_impact_score"] = (
            pd.to_numeric(features["home_injury_impact_score"], errors="coerce").fillna(0.0)
            + pd.to_numeric(features["away_injury_impact_score"], errors="coerce").fillna(0.0)
        )
        impact_column = "_combined_injury_impact_score"

    if impact_column is not None:
        print("\nTop 10 matches by combined injury impact:")
        preview_columns = [
            column
            for column in ["fixture_id", "date", "home_team_name", "away_team_name", impact_column]
            if column in features.columns
        ]
        top10 = features.sort_values(impact_column, ascending=False).head(10)[preview_columns]
        print(top10.to_string(index=False))

    print("\nNaN rates:")
    for column in INJURY_OUTPUT_COLUMNS:
        if column in features.columns:
            print(f"- {column}: {nan_rate(features, [column]):.2%}")

    print("\nPreview:")
    preview_columns = [
        column
        for column in ["fixture_id", "date", "home_team_name", "away_team_name"] + INJURY_OUTPUT_COLUMNS
        if column in features.columns
    ]
    print(features[preview_columns].head().to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build API-Football Ligue 1 injury impact features.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    return parser.parse_args()


def main() -> None:
    """Run the injury impact feature builder."""
    args = parse_args()
    features = build_injury_impact_features(season=args.season, league_id=args.league_id)
    destination = output_path(args.season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)

    prematch_path = input_path(args.season, PREMATCH_INPUT_TEMPLATE)
    lineups_path = input_path(args.season, LINEUPS_INPUT_TEMPLATE)
    injuries_path = input_path(args.season, INJURIES_INPUT_TEMPLATE)
    print_summary(
        features,
        season=args.season,
        prematch_rows=len(pd.read_csv(prematch_path)) if prematch_path.exists() else 0,
        lineups_rows=len(pd.read_csv(lineups_path)) if lineups_path.exists() else 0,
        injuries_rows=len(pd.read_csv(injuries_path)) if injuries_path.exists() else 0,
    )
    print(f"\nSaved injury impact features to: {destination}")


if __name__ == "__main__":
    main()
