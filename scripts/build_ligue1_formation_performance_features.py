"""Build leakage-safe Ligue 1 formation performance prematch features."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEASON = 2025


def input_path(season: int) -> Path:
    """Return API-Football prematch feature input path."""
    return PROJECT_ROOT / "data" / "processed" / f"api_football_ligue1_{season}_prematch_features.csv"


def context_path(season: int) -> Path:
    """Return API-Football context feature path with match-level stats."""
    return PROJECT_ROOT / "data" / "processed" / f"api_football_ligue1_{season}_context_features.csv"


def output_path(season: int) -> Path:
    """Return formation performance feature output path."""
    return PROJECT_ROOT / "data" / "processed" / f"api_football_ligue1_{season}_formation_performance_features.csv"


def signal_strength(sample_size: int) -> str:
    """Return signal strength from historical sample size."""
    if sample_size >= 10:
        return "strong"
    if sample_size >= 5:
        return "medium"
    return "weak"


def result_for_side(goals_for: Any, goals_against: Any) -> str | None:
    """Return W/D/L for one team side."""
    if pd.isna(goals_for) or pd.isna(goals_against):
        return None
    if float(goals_for) > float(goals_against):
        return "W"
    if float(goals_for) < float(goals_against):
        return "L"
    return "D"


def result_hda(goals_home: Any, goals_away: Any) -> str | None:
    """Return H/D/A result for a fixture."""
    if pd.isna(goals_home) or pd.isna(goals_away):
        return None
    if float(goals_home) > float(goals_away):
        return "H"
    if float(goals_home) < float(goals_away):
        return "A"
    return "D"


def mean_or_na(values: list[Any]) -> float | None:
    """Return numeric mean or None."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else None


def team_formation_metrics(history: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Build historical performance metrics for one team+formation."""
    matches = len(history)
    if matches == 0:
        return {
            f"{prefix}_matches_before": 0,
            f"{prefix}_win_rate_before": None,
            f"{prefix}_draw_rate_before": None,
            f"{prefix}_loss_rate_before": None,
            f"{prefix}_xg_for_avg_before": None,
            f"{prefix}_xg_against_avg_before": None,
            f"{prefix}_goal_diff_avg_before": None,
        }

    results = [item["result"] for item in history if item.get("result") in {"W", "D", "L"}]
    result_count = len(results)
    return {
        f"{prefix}_matches_before": matches,
        f"{prefix}_win_rate_before": results.count("W") / result_count if result_count else None,
        f"{prefix}_draw_rate_before": results.count("D") / result_count if result_count else None,
        f"{prefix}_loss_rate_before": results.count("L") / result_count if result_count else None,
        f"{prefix}_xg_for_avg_before": mean_or_na([item.get("xg_for") for item in history]),
        f"{prefix}_xg_against_avg_before": mean_or_na([item.get("xg_against") for item in history]),
        f"{prefix}_goal_diff_avg_before": mean_or_na([item.get("goal_diff") for item in history]),
    }


def matchup_metrics(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build historical performance metrics for one home-formation vs away-formation matchup."""
    matches = len(history)
    if matches == 0:
        return {
            "formation_matchup_matches_before": 0,
            "formation_matchup_home_win_rate_before": None,
            "formation_matchup_draw_rate_before": None,
            "formation_matchup_away_win_rate_before": None,
            "formation_matchup_avg_total_goals_before": None,
            "formation_matchup_avg_xg_total_before": None,
        }

    results = [item["result"] for item in history if item.get("result") in {"H", "D", "A"}]
    result_count = len(results)
    return {
        "formation_matchup_matches_before": matches,
        "formation_matchup_home_win_rate_before": results.count("H") / result_count if result_count else None,
        "formation_matchup_draw_rate_before": results.count("D") / result_count if result_count else None,
        "formation_matchup_away_win_rate_before": results.count("A") / result_count if result_count else None,
        "formation_matchup_avg_total_goals_before": mean_or_na([item.get("total_goals") for item in history]),
        "formation_matchup_avg_xg_total_before": mean_or_na([item.get("xg_total") for item in history]),
    }


def build_formation_performance_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build formation performance features with strict previous-match history."""
    required_columns = [
        "date",
        "fixture_id",
        "home_team_id",
        "away_team_id",
        "home_formation",
        "away_formation",
        "goals_home",
        "goals_away",
        "home_expected_goals",
        "away_expected_goals",
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    rows = []
    sorted_data = data.copy()
    sorted_data["date"] = pd.to_datetime(sorted_data["date"], errors="coerce")
    sorted_data = sorted_data.dropna(subset=["date"]).sort_values(["date", "fixture_id"]).reset_index(drop=True)

    team_formation_history: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    formation_matchup_history: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)

    for _, match in sorted_data.iterrows():
        home_key = (match["home_team_id"], match["home_formation"])
        away_key = (match["away_team_id"], match["away_formation"])
        matchup_key = (match["home_formation"], match["away_formation"])

        output = match.to_dict()
        home_metrics = team_formation_metrics(team_formation_history[home_key], "home_team_formation")
        away_metrics = team_formation_metrics(team_formation_history[away_key], "away_team_formation")
        matchup = matchup_metrics(formation_matchup_history[matchup_key])
        output.update(home_metrics)
        output.update(away_metrics)
        output.update(matchup)
        output["home_formation_signal_strength"] = signal_strength(
            int(home_metrics["home_team_formation_matches_before"])
        )
        output["away_formation_signal_strength"] = signal_strength(
            int(away_metrics["away_team_formation_matches_before"])
        )
        output["matchup_signal_strength"] = signal_strength(int(matchup["formation_matchup_matches_before"]))
        rows.append(output)

        home_result = result_for_side(match["goals_home"], match["goals_away"])
        away_result = result_for_side(match["goals_away"], match["goals_home"])
        fixture_result = result_hda(match["goals_home"], match["goals_away"])
        if home_result is not None and pd.notna(match["home_formation"]):
            team_formation_history[home_key].append(
                {
                    "result": home_result,
                    "xg_for": match.get("home_expected_goals"),
                    "xg_against": match.get("away_expected_goals"),
                    "goal_diff": float(match["goals_home"]) - float(match["goals_away"]),
                }
            )
        if away_result is not None and pd.notna(match["away_formation"]):
            team_formation_history[away_key].append(
                {
                    "result": away_result,
                    "xg_for": match.get("away_expected_goals"),
                    "xg_against": match.get("home_expected_goals"),
                    "goal_diff": float(match["goals_away"]) - float(match["goals_home"]),
                }
            )
        if fixture_result is not None and pd.notna(match["home_formation"]) and pd.notna(match["away_formation"]):
            xg_total = None
            if pd.notna(match.get("home_expected_goals")) and pd.notna(match.get("away_expected_goals")):
                xg_total = float(match["home_expected_goals"]) + float(match["away_expected_goals"])
            formation_matchup_history[matchup_key].append(
                {
                    "result": fixture_result,
                    "total_goals": float(match["goals_home"]) + float(match["goals_away"]),
                    "xg_total": xg_total,
                }
            )

    return pd.DataFrame(rows)


def build_ligue1_formation_performance_features(season: int = DEFAULT_SEASON) -> pd.DataFrame:
    """Build and save the formation performance feature table."""
    source_path = input_path(season)
    if not source_path.exists():
        raise FileNotFoundError(f"Prematch feature file not found: {source_path}")

    data = pd.read_csv(source_path)
    xg_columns = ["home_expected_goals", "away_expected_goals"]
    context_feature_path = context_path(season)
    missing_xg_columns = [column for column in xg_columns if column not in data.columns]
    if missing_xg_columns and context_feature_path.exists():
        context = pd.read_csv(context_feature_path)
        context_columns = ["fixture_id"] + [column for column in xg_columns if column in context.columns]
        if len(context_columns) > 1:
            data = data.merge(context[context_columns], on="fixture_id", how="left")
    for column in xg_columns:
        if column not in data.columns:
            data[column] = pd.NA

    output = build_formation_performance_features(data)
    destination = output_path(season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)

    print("Ligue 1 formation performance feature build")
    print(f"Input: {source_path}")
    print(f"Rows read: {len(data)}")
    print(f"Rows exported: {len(output)}")
    print(f"Date min: {output['date'].min()}")
    print(f"Date max: {output['date'].max()}")
    print("\nSignal strength counts:")
    for column in ["home_formation_signal_strength", "away_formation_signal_strength", "matchup_signal_strength"]:
        print(f"{column}:")
        print(output[column].value_counts().to_string())
    print(f"\nSaved formation performance features to: {destination}")
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build Ligue 1 formation performance features.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    return parser.parse_args()


def main() -> None:
    """Run the feature builder."""
    args = parse_args()
    build_ligue1_formation_performance_features(season=args.season)


if __name__ == "__main__":
    main()
