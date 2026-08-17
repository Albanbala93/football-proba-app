"""Build leakage-safe Ligue 1 prematch features from API-Football context data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEASON = 2025
DEFAULT_WINDOW = 5

IDENTIFIER_COLUMNS = [
    "fixture_id",
    "date",
    "league_id",
    "season",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "goals_home",
    "goals_away",
    "status_short",
    "home_formation",
    "away_formation",
    "formation_matchup",
    "home_injuries_count",
    "away_injuries_count",
    "injuries_diff",
]
TEAM_STAT_SPECS = {
    "shots_on_goal": ("shots_on_goal_for", "shots_on_goal_against"),
    "total_shots": ("total_shots_for", "total_shots_against"),
    "expected_goals": ("expected_goals_for", "expected_goals_against"),
    "corners": ("corners_for", "corners_against"),
}


def input_path(season: int) -> Path:
    """Return context feature input path."""
    return PROJECT_ROOT / "data" / "processed" / f"api_football_ligue1_{season}_context_features.csv"


def output_path(season: int) -> Path:
    """Return prematch feature output path."""
    return PROJECT_ROOT / "data" / "processed" / f"api_football_ligue1_{season}_prematch_features.csv"


def _mean_recent(matches: list[dict[str, Any]], key: str, window: int) -> float | None:
    """Mean of a key over the last window previous matches."""
    values = [match.get(key) for match in matches[-window:]]
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else None


def _formation_stability(matches: list[dict[str, Any]], window: int) -> float | None:
    """Share of the most common formation in recent previous matches."""
    formations = [str(match.get("formation")) for match in matches[-window:] if pd.notna(match.get("formation"))]
    if not formations:
        return None
    counts = pd.Series(formations).value_counts()
    return float(counts.iloc[0] / len(formations))


def _recent_formations_count(matches: list[dict[str, Any]], window: int) -> int:
    """Count distinct formations in recent previous matches."""
    formations = {str(match.get("formation")) for match in matches[-window:] if pd.notna(match.get("formation"))}
    return len(formations)


def _team_history_row(row: pd.Series, side: str) -> dict[str, Any]:
    """Convert one match row into one team-history entry after features are computed."""
    opponent = "away" if side == "home" else "home"
    return {
        "fixture_id": row["fixture_id"],
        "date": row["date"],
        "team_id": row[f"{side}_team_id"],
        "opponent_team_id": row[f"{opponent}_team_id"],
        "formation": row.get(f"{side}_formation"),
        "shots_on_goal_for": row.get(f"{side}_shots_on_goal"),
        "shots_on_goal_against": row.get(f"{opponent}_shots_on_goal"),
        "total_shots_for": row.get(f"{side}_total_shots"),
        "total_shots_against": row.get(f"{opponent}_total_shots"),
        "expected_goals_for": row.get(f"{side}_expected_goals"),
        "expected_goals_against": row.get(f"{opponent}_expected_goals"),
            "corners_for": row.get(f"{side}_corner_kicks"),
            "corners_against": row.get(f"{opponent}_corner_kicks"),
        "possession": row.get(f"{side}_ball_possession"),
        "yellow_cards": row.get(f"{side}_yellow_cards"),
        "red_cards": row.get(f"{side}_red_cards"),
    }


def _side_features(prefix: str, history: list[dict[str, Any]], window: int) -> dict[str, Any]:
    """Build rolling features for one side from previous team history."""
    features: dict[str, Any] = {
        f"{prefix}_formation_stability_{window}": _formation_stability(history, window),
        f"{prefix}_recent_formations_count": _recent_formations_count(history, window),
        f"{prefix}_api_possession_{window}": _mean_recent(history, "possession", window),
        f"{prefix}_api_yellow_cards_{window}": _mean_recent(history, "yellow_cards", window),
        f"{prefix}_api_red_cards_{window}": _mean_recent(history, "red_cards", window),
    }

    for output_base, (for_key, against_key) in TEAM_STAT_SPECS.items():
        features[f"{prefix}_api_{output_base}_for_{window}"] = _mean_recent(history, for_key, window)
        features[f"{prefix}_api_{output_base}_against_{window}"] = _mean_recent(history, against_key, window)

    return features


def build_prematch_features(context: pd.DataFrame, window: int) -> pd.DataFrame:
    """Build leakage-safe prematch rows from match context data."""
    required_columns = set(IDENTIFIER_COLUMNS).union(
        {
            "home_shots_on_goal",
            "away_shots_on_goal",
            "home_total_shots",
            "away_total_shots",
            "home_expected_goals",
            "away_expected_goals",
            "home_corner_kicks",
            "away_corner_kicks",
            "home_ball_possession",
            "away_ball_possession",
            "home_yellow_cards",
            "away_yellow_cards",
            "home_red_cards",
            "away_red_cards",
        }
    )
    missing = sorted(column for column in required_columns if column not in context.columns)
    if missing:
        raise ValueError(f"Missing required columns in context features: {', '.join(missing)}")

    data = context.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values(["date", "fixture_id"]).reset_index(drop=True)

    histories: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    rows = []

    for _, row in data.iterrows():
        home_history = histories[row["home_team_id"]]
        away_history = histories[row["away_team_id"]]
        output = {column: row.get(column) for column in IDENTIFIER_COLUMNS}
        output.update(_side_features("home", home_history, window))
        output.update(_side_features("away", away_history, window))

        output[f"api_shots_on_goal_diff_{window}"] = (
            output[f"home_api_shots_on_goal_for_{window}"] - output[f"away_api_shots_on_goal_for_{window}"]
            if pd.notna(output[f"home_api_shots_on_goal_for_{window}"])
            and pd.notna(output[f"away_api_shots_on_goal_for_{window}"])
            else None
        )
        output[f"api_total_shots_diff_{window}"] = (
            output[f"home_api_total_shots_for_{window}"] - output[f"away_api_total_shots_for_{window}"]
            if pd.notna(output[f"home_api_total_shots_for_{window}"])
            and pd.notna(output[f"away_api_total_shots_for_{window}"])
            else None
        )
        output[f"api_expected_goals_diff_{window}"] = (
            output[f"home_api_expected_goals_for_{window}"] - output[f"away_api_expected_goals_for_{window}"]
            if pd.notna(output[f"home_api_expected_goals_for_{window}"])
            and pd.notna(output[f"away_api_expected_goals_for_{window}"])
            else None
        )
        output[f"api_corners_diff_{window}"] = (
            output[f"home_api_corners_for_{window}"] - output[f"away_api_corners_for_{window}"]
            if pd.notna(output[f"home_api_corners_for_{window}"])
            and pd.notna(output[f"away_api_corners_for_{window}"])
            else None
        )
        output[f"api_possession_diff_{window}"] = (
            output[f"home_api_possession_{window}"] - output[f"away_api_possession_{window}"]
            if pd.notna(output[f"home_api_possession_{window}"]) and pd.notna(output[f"away_api_possession_{window}"])
            else None
        )
        output[f"formation_stability_diff_{window}"] = (
            output[f"home_formation_stability_{window}"] - output[f"away_formation_stability_{window}"]
            if pd.notna(output[f"home_formation_stability_{window}"])
            and pd.notna(output[f"away_formation_stability_{window}"])
            else None
        )
        rows.append(output)

        histories[row["home_team_id"]].append(_team_history_row(row, "home"))
        histories[row["away_team_id"]].append(_team_history_row(row, "away"))

    return pd.DataFrame(rows)


def _nan_rate(df: pd.DataFrame, columns: list[str]) -> float:
    """Return aggregate NaN rate for a feature family."""
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return 0.0
    return float(df[existing].isna().mean().mean())


def print_summary(features: pd.DataFrame, source_rows: int, window: int) -> None:
    """Print a readable build summary."""
    rolling_columns = [column for column in features.columns if f"_{window}" in column and column.startswith(("home_api", "away_api"))]
    formation_columns = [
        f"home_formation_stability_{window}",
        f"away_formation_stability_{window}",
        "home_recent_formations_count",
        "away_recent_formations_count",
        f"formation_stability_diff_{window}",
    ]
    injury_columns = ["home_injuries_count", "away_injuries_count", "injuries_diff"]
    diff_columns = [column for column in features.columns if column.endswith(f"_diff_{window}")]

    print("API-Football Ligue 1 prematch feature build")
    print(f"Matches read: {source_rows}")
    print(f"Rows exported: {len(features)}")
    print(f"Date min: {features['date'].min()}")
    print(f"Date max: {features['date'].max()}")
    print("\nNaN rates:")
    print(f"- rolling_stats: {_nan_rate(features, rolling_columns):.2%}")
    print(f"- formations: {_nan_rate(features, formation_columns):.2%}")
    print(f"- injuries: {_nan_rate(features, injury_columns):.2%}")
    print(f"- differentials: {_nan_rate(features, diff_columns):.2%}")
    print("\nPreview:")
    print(features.head().to_string(index=False))


def build_api_football_ligue1_prematch_features(season: int = DEFAULT_SEASON, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Build and save prematch feature CSV."""
    source_path = input_path(season)
    if not source_path.exists():
        raise FileNotFoundError(f"Context features file not found: {source_path}")

    context = pd.read_csv(source_path)
    features = build_prematch_features(context, window=window)
    destination = output_path(season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)

    print_summary(features, source_rows=len(context), window=window)
    print(f"\nSaved prematch features to: {destination}")
    return features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build API-Football Ligue 1 prematch features.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    return parser.parse_args()


def main() -> None:
    """Run the prematch feature builder."""
    args = parse_args()
    build_api_football_ligue1_prematch_features(season=args.season, window=args.window)


if __name__ == "__main__":
    main()
