"""Build Ligue 1 match stakes features from API-Football context data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEASON = 2025
DEFAULT_LEAGUE_ID = 61
MATCHS_PER_MATCHDAY = 9
TOTAL_MATCHDAYS = 34

CONTEXT_INPUT_TEMPLATE = "api_football_ligue1_{season}_context_features.csv"
INJURY_IMPACT_INPUT_TEMPLATE = "api_football_ligue1_{season}_injury_impact_features.csv"
OUTPUT_TEMPLATE = "api_football_ligue1_{season}_match_stakes_features.csv"

STAKE_COLUMNS = [
    "home_points_before",
    "away_points_before",
    "home_rank_before",
    "away_rank_before",
    "home_goal_diff_before",
    "away_goal_diff_before",
    "home_matches_played_before",
    "away_matches_played_before",
    "matchday_estimate",
    "remaining_matches_estimate",
    "home_title_pressure",
    "away_title_pressure",
    "home_europe_pressure",
    "away_europe_pressure",
    "home_relegation_pressure",
    "away_relegation_pressure",
    "home_midtable_flag",
    "away_midtable_flag",
    "home_total_motivation_score",
    "away_total_motivation_score",
    "motivation_diff",
    "match_context_label",
]


def input_path(season: int, template: str) -> Path:
    """Return a processed input path."""
    return PROJECT_ROOT / "data" / "processed" / template.format(season=season)


def output_path(season: int) -> Path:
    """Return the stakes feature output path."""
    return PROJECT_ROOT / "data" / "processed" / OUTPUT_TEMPLATE.format(season=season)


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse timestamps with UTC awareness."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_input_table(season: int) -> tuple[pd.DataFrame, Path]:
    """Load the preferred injury-impact table with context fallback."""
    injury_path = input_path(season, INJURY_IMPACT_INPUT_TEMPLATE)
    context_path = input_path(season, CONTEXT_INPUT_TEMPLATE)
    if injury_path.exists():
        return pd.read_csv(injury_path), injury_path
    if context_path.exists():
        return pd.read_csv(context_path), context_path
    raise FileNotFoundError(
        f"Input table not found: neither {injury_path} nor {context_path} exists"
    )


def _initial_standings(team_ids: list[Any]) -> dict[Any, dict[str, int]]:
    """Create an empty standings table for all teams."""
    standings: dict[Any, dict[str, int]] = {}
    for team_id in team_ids:
        standings[team_id] = {"points": 0, "gf": 0, "ga": 0, "gd": 0, "matches": 0}
    return standings


def _sorted_ranks(standings: dict[Any, dict[str, int]], team_names: dict[Any, str]) -> dict[Any, int]:
    """Compute current ranks from standings."""
    ordered = sorted(
        standings.items(),
        key=lambda item: (
            -item[1]["points"],
            -item[1]["gd"],
            -item[1]["gf"],
            str(team_names.get(item[0], item[0])),
        ),
    )
    return {team_id: rank + 1 for rank, (team_id, _) in enumerate(ordered)}


def _title_pressure(rank_before: int, remaining_matches_estimate: int) -> float:
    """Return title pressure according to the requested V1 rules."""
    if 1 <= rank_before <= 3 and remaining_matches_estimate <= 10:
        return 1.0
    if 1 <= rank_before <= 5 and remaining_matches_estimate <= 15:
        return 0.5
    return 0.0


def _europe_pressure(rank_before: int, remaining_matches_estimate: int) -> float:
    """Return Europe pressure according to the requested V1 rules."""
    if 3 <= rank_before <= 7 and remaining_matches_estimate <= 10:
        return 1.0
    if 3 <= rank_before <= 9 and remaining_matches_estimate <= 15:
        return 0.5
    return 0.0


def _relegation_pressure(rank_before: int, remaining_matches_estimate: int) -> float:
    """Return relegation pressure according to the requested V1 rules."""
    if 15 <= rank_before <= 18 and remaining_matches_estimate <= 10:
        return 1.0
    if 13 <= rank_before <= 18 and remaining_matches_estimate <= 15:
        return 0.5
    return 0.0


def _midtable_flag(rank_before: int, title_pressure: float, europe_pressure: float, relegation_pressure: float) -> int:
    """Return whether a team is in midtable territory."""
    return int(8 <= rank_before <= 12 and title_pressure == 0 and europe_pressure == 0 and relegation_pressure == 0)


def _update_standings(standings: dict[Any, dict[str, int]], home_team_id: Any, away_team_id: Any, goals_home: Any, goals_away: Any) -> None:
    """Update standings after one completed match."""
    home_goals = pd.to_numeric(pd.Series([goals_home]), errors="coerce").iloc[0]
    away_goals = pd.to_numeric(pd.Series([goals_away]), errors="coerce").iloc[0]
    if pd.isna(home_goals) or pd.isna(away_goals):
        return

    home_stats = standings.setdefault(home_team_id, {"points": 0, "gf": 0, "ga": 0, "gd": 0, "matches": 0})
    away_stats = standings.setdefault(away_team_id, {"points": 0, "gf": 0, "ga": 0, "gd": 0, "matches": 0})

    home_stats["matches"] += 1
    away_stats["matches"] += 1
    home_stats["gf"] += int(home_goals)
    home_stats["ga"] += int(away_goals)
    away_stats["gf"] += int(away_goals)
    away_stats["ga"] += int(home_goals)
    home_stats["gd"] = home_stats["gf"] - home_stats["ga"]
    away_stats["gd"] = away_stats["gf"] - away_stats["ga"]

    if home_goals > away_goals:
        home_stats["points"] += 3
    elif home_goals < away_goals:
        away_stats["points"] += 3
    else:
        home_stats["points"] += 1
        away_stats["points"] += 1


def _match_result(goals_home: Any, goals_away: Any) -> str | None:
    """Return H, D or A from goals."""
    home = pd.to_numeric(pd.Series([goals_home]), errors="coerce").iloc[0]
    away = pd.to_numeric(pd.Series([goals_away]), errors="coerce").iloc[0]
    if pd.isna(home) or pd.isna(away):
        return None
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _team_stakes_label(title_pressure: float, europe_pressure: float, relegation_pressure: float, midtable_flag: int) -> str:
    """Return a compact stakes label for one team."""
    if title_pressure >= 1:
        return "title_race"
    if europe_pressure >= 1:
        return "europe_race"
    if relegation_pressure >= 1:
        return "relegation_battle"
    if midtable_flag == 1:
        return "low_stakes"
    return "standard"


def build_api_football_ligue1_match_stakes_features(season: int = DEFAULT_SEASON, league_id: int = DEFAULT_LEAGUE_ID) -> pd.DataFrame:
    """Build and save match stakes features."""
    source, source_path = load_input_table(season)

    required = {"fixture_id", "date", "home_team_id", "away_team_id", "home_team_name", "away_team_name", "goals_home", "goals_away"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns in source features ({source_path}): {', '.join(missing)}")

    source_data = source.copy()
    source_data["match_datetime"] = parse_datetime(source_data["date"])
    source_data = source_data.dropna(subset=["match_datetime"]).sort_values(["match_datetime", "fixture_id"]).reset_index(drop=True)

    team_ids = sorted(
        pd.concat([source_data["home_team_id"], source_data["away_team_id"]], ignore_index=True)
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    team_names: dict[Any, str] = {}
    for _, row in source_data.iterrows():
        if pd.notna(row["home_team_id"]):
            team_names[int(row["home_team_id"])] = str(row["home_team_name"])
        if pd.notna(row["away_team_id"]):
            team_names[int(row["away_team_id"])] = str(row["away_team_name"])

    standings = _initial_standings(team_ids)
    stakes_rows: list[dict[str, Any]] = []
    completed_matches_before = 0

    for _, row in source_data.iterrows():
        fixture_id = row["fixture_id"]
        home_team_id = int(row["home_team_id"])
        away_team_id = int(row["away_team_id"])

        ranks = _sorted_ranks(standings, team_names)
        home_stats = standings.get(home_team_id, {"points": 0, "gf": 0, "ga": 0, "gd": 0, "matches": 0})
        away_stats = standings.get(away_team_id, {"points": 0, "gf": 0, "ga": 0, "gd": 0, "matches": 0})

        matchday_estimate = min(TOTAL_MATCHDAYS, (completed_matches_before // MATCHS_PER_MATCHDAY) + 1)
        remaining_matches_estimate = max(0, TOTAL_MATCHDAYS - matchday_estimate)
        home_rank = int(ranks.get(home_team_id, len(standings)))
        away_rank = int(ranks.get(away_team_id, len(standings)))
        home_title_pressure = _title_pressure(home_rank, remaining_matches_estimate)
        away_title_pressure = _title_pressure(away_rank, remaining_matches_estimate)
        home_europe_pressure = _europe_pressure(home_rank, remaining_matches_estimate)
        away_europe_pressure = _europe_pressure(away_rank, remaining_matches_estimate)
        home_relegation_pressure = _relegation_pressure(home_rank, remaining_matches_estimate)
        away_relegation_pressure = _relegation_pressure(away_rank, remaining_matches_estimate)
        home_total_motivation_score = round(home_title_pressure + home_europe_pressure + home_relegation_pressure, 4)
        away_total_motivation_score = round(away_title_pressure + away_europe_pressure + away_relegation_pressure, 4)
        home_midtable_flag = _midtable_flag(home_rank, home_title_pressure, home_europe_pressure, home_relegation_pressure)
        away_midtable_flag = _midtable_flag(away_rank, away_title_pressure, away_europe_pressure, away_relegation_pressure)
        home_label = _team_stakes_label(home_title_pressure, home_europe_pressure, home_relegation_pressure, home_midtable_flag)
        away_label = _team_stakes_label(away_title_pressure, away_europe_pressure, away_relegation_pressure, away_midtable_flag)
        if home_title_pressure >= 1 or away_title_pressure >= 1:
            match_context_label = "title_race"
        elif home_europe_pressure >= 1 or away_europe_pressure >= 1:
            match_context_label = "europe_race"
        elif home_relegation_pressure >= 1 or away_relegation_pressure >= 1:
            match_context_label = "relegation_battle"
        elif home_midtable_flag == 1 and away_midtable_flag == 1:
            match_context_label = "low_stakes"
        elif home_label != away_label:
            match_context_label = "mixed_stakes"
        else:
            match_context_label = "standard"

        stakes_rows.append(
            {
                "fixture_id": fixture_id,
                "home_points_before": int(home_stats.get("points", 0)),
                "away_points_before": int(away_stats.get("points", 0)),
                "home_rank_before": home_rank,
                "away_rank_before": away_rank,
                "home_goal_diff_before": int(home_stats.get("gd", 0)),
                "away_goal_diff_before": int(away_stats.get("gd", 0)),
                "home_matches_played_before": int(home_stats.get("matches", 0)),
                "away_matches_played_before": int(away_stats.get("matches", 0)),
                "matchday_estimate": matchday_estimate,
                "remaining_matches_estimate": remaining_matches_estimate,
                "home_title_pressure": home_title_pressure,
                "away_title_pressure": away_title_pressure,
                "home_europe_pressure": home_europe_pressure,
                "away_europe_pressure": away_europe_pressure,
                "home_relegation_pressure": home_relegation_pressure,
                "away_relegation_pressure": away_relegation_pressure,
                "home_midtable_flag": home_midtable_flag,
                "away_midtable_flag": away_midtable_flag,
                "home_total_motivation_score": home_total_motivation_score,
                "away_total_motivation_score": away_total_motivation_score,
                "motivation_diff": round(home_total_motivation_score - away_total_motivation_score, 4),
                "match_context_label": match_context_label,
            }
        )

        _update_standings(standings, home_team_id, away_team_id, row["goals_home"], row["goals_away"])
        completed_matches_before += 1

    stakes = pd.DataFrame(stakes_rows)
    merged = source.merge(stakes, on="fixture_id", how="left")
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


def print_summary(features: pd.DataFrame, source_rows: int, source_path: Path) -> None:
    """Print a readable build summary."""
    print("API-Football Ligue 1 match stakes feature build")
    print(f"Input source: {source_path}")
    print(f"Matches read: {source_rows}")
    print(f"Rows exported: {len(features)}")
    if "date" in features.columns:
        print(f"Date min: {features['date'].min()}")
        print(f"Date max: {features['date'].max()}")
    print("\nNaN rates:")
    print(f"- standings: {_nan_rate(features, ['home_points_before', 'away_points_before', 'home_rank_before', 'away_rank_before']):.2%}")
    print(f"- pressures: {_nan_rate(features, ['home_title_pressure', 'away_title_pressure', 'home_europe_pressure', 'away_europe_pressure', 'home_relegation_pressure', 'away_relegation_pressure']):.2%}")
    print(f"- motivation: {_nan_rate(features, ['home_total_motivation_score', 'away_total_motivation_score', 'motivation_diff']):.2%}")
    print(f"- labels: {_nan_rate(features, ['match_context_label']):.2%}")
    if "match_context_label" in features.columns:
        print("\nMatch context label distribution:")
        print(features["match_context_label"].fillna("unknown").value_counts(dropna=False).to_string())
        print("\nHigh stakes examples:")
        high_stakes = features[features["match_context_label"].isin(["title_race", "europe_race", "relegation_battle"])]
        if not high_stakes.empty:
            preview_columns = [column for column in ["fixture_id", "date", "home_team_name", "away_team_name", "match_context_label"] if column in high_stakes.columns]
            print(high_stakes[preview_columns].head(10).to_string(index=False))
        else:
            print("none")
    print("\nPreview:")
    preview_columns = [
        column
        for column in ["fixture_id", "date", "home_team_name", "away_team_name"] + STAKE_COLUMNS
        if column in features.columns
    ]
    print(features[preview_columns].head().to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build API-Football Ligue 1 match stakes features.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    return parser.parse_args()


def main() -> None:
    """Run the match stakes builder."""
    args = parse_args()
    features = build_api_football_ligue1_match_stakes_features(season=args.season, league_id=args.league_id)
    destination = output_path(args.season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)
    source_df, source_path = load_input_table(args.season)
    print_summary(features, source_rows=len(source_df), source_path=source_path)
    print(f"\nSaved match stakes features to: {destination}")


if __name__ == "__main__":
    main()
