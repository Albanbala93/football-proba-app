"""Build Ligue 1 fatigue / calendar features from API-Football data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEASON = 2025
DEFAULT_LEAGUE_ID = 61

INJURY_IMPACT_INPUT_TEMPLATE = "api_football_ligue1_{season}_injury_impact_features.csv"
PREMATCH_INPUT_TEMPLATE = "api_football_ligue1_{season}_prematch_features.csv"
OUTPUT_TEMPLATE = "api_football_ligue1_{season}_fatigue_features.csv"

FATIGUE_COLUMNS = [
    "home_days_since_last_match",
    "away_days_since_last_match",
    "rest_days_diff",
    "home_matches_last_7_days",
    "away_matches_last_7_days",
    "matches_last_7_days_diff",
    "home_matches_last_14_days",
    "away_matches_last_14_days",
    "matches_last_14_days_diff",
    "home_matches_last_21_days",
    "away_matches_last_21_days",
    "matches_last_21_days_diff",
    "home_short_rest_flag",
    "away_short_rest_flag",
    "short_rest_diff",
    "home_rest_advantage_flag",
    "away_rest_advantage_flag",
    "home_fatigue_score",
    "away_fatigue_score",
    "fatigue_score_diff",
]


def input_path(season: int, template: str) -> Path:
    """Return a processed input path."""
    return PROJECT_ROOT / "data" / "processed" / template.format(season=season)


def output_path(season: int) -> Path:
    """Return the fatigue feature output path."""
    return PROJECT_ROOT / "data" / "processed" / OUTPUT_TEMPLATE.format(season=season)


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse timestamps with UTC awareness."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_input_table(season: int) -> tuple[pd.DataFrame, Path]:
    """Load the preferred injury-impact table with prematch fallback."""
    injury_path = input_path(season, INJURY_IMPACT_INPUT_TEMPLATE)
    prematch_path = input_path(season, PREMATCH_INPUT_TEMPLATE)
    if injury_path.exists():
        return pd.read_csv(injury_path), injury_path
    if prematch_path.exists():
        return pd.read_csv(prematch_path), prematch_path
    raise FileNotFoundError(
        f"Input table not found: neither {injury_path} nor {prematch_path} exists"
    )


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _matches_before(history: list[pd.Timestamp], current_date: pd.Timestamp, days: int) -> int:
    """Count prior matches in the given lookback window."""
    lower_bound = current_date - pd.Timedelta(days=days)
    return int(sum(1 for match_date in history if lower_bound < match_date < current_date))


def _days_since_last_match(history: list[pd.Timestamp], current_date: pd.Timestamp) -> float:
    """Return days since the last prior match, or NaN if unavailable."""
    prior_dates = [match_date for match_date in history if match_date < current_date]
    if not prior_dates:
        return float("nan")
    last_match = max(prior_dates)
    return float((current_date - last_match) / pd.Timedelta(days=1))


def build_api_football_ligue1_fatigue_features(
    season: int = DEFAULT_SEASON,
    league_id: int = DEFAULT_LEAGUE_ID,
) -> pd.DataFrame:
    """Build and save fatigue features."""
    source, source_path = load_input_table(season)

    required = {"fixture_id", "date", "home_team_id", "away_team_id", "home_team_name", "away_team_name"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns in source features ({source_path}): {', '.join(missing)}")

    source_data = source.copy()
    source_data["match_datetime"] = parse_datetime(source_data["date"])
    source_data = source_data.dropna(subset=["match_datetime"]).sort_values(["match_datetime", "fixture_id"]).reset_index(drop=True)

    team_history: dict[Any, list[pd.Timestamp]] = defaultdict(list)
    fatigue_rows: list[dict[str, Any]] = []

    for _, row in source_data.iterrows():
        fixture_id = row["fixture_id"]
        match_datetime = row["match_datetime"]
        home_team_id = row["home_team_id"]
        away_team_id = row["away_team_id"]

        home_history = team_history[home_team_id]
        away_history = team_history[away_team_id]

        home_days_since_last_match = _days_since_last_match(home_history, match_datetime)
        away_days_since_last_match = _days_since_last_match(away_history, match_datetime)

        home_matches_last_7_days = _matches_before(home_history, match_datetime, 7)
        away_matches_last_7_days = _matches_before(away_history, match_datetime, 7)
        home_matches_last_14_days = _matches_before(home_history, match_datetime, 14)
        away_matches_last_14_days = _matches_before(away_history, match_datetime, 14)
        home_matches_last_21_days = _matches_before(home_history, match_datetime, 21)
        away_matches_last_21_days = _matches_before(away_history, match_datetime, 21)

        home_short_rest_flag = int(pd.notna(home_days_since_last_match) and home_days_since_last_match <= 3)
        away_short_rest_flag = int(pd.notna(away_days_since_last_match) and away_days_since_last_match <= 3)
        short_rest_diff = home_short_rest_flag - away_short_rest_flag

        rest_days_diff = float("nan")
        if pd.notna(home_days_since_last_match) and pd.notna(away_days_since_last_match):
            rest_days_diff = float(home_days_since_last_match - away_days_since_last_match)

        home_rest_advantage_flag = int(
            pd.notna(home_days_since_last_match)
            and pd.notna(away_days_since_last_match)
            and (home_days_since_last_match - away_days_since_last_match) >= 2
        )
        away_rest_advantage_flag = int(
            pd.notna(home_days_since_last_match)
            and pd.notna(away_days_since_last_match)
            and (away_days_since_last_match - home_days_since_last_match) >= 2
        )

        home_fatigue_score = float(home_matches_last_7_days * 1.5 + home_matches_last_14_days * 1.0 + home_short_rest_flag * 1.5)
        away_fatigue_score = float(away_matches_last_7_days * 1.5 + away_matches_last_14_days * 1.0 + away_short_rest_flag * 1.5)
        fatigue_score_diff = float(home_fatigue_score - away_fatigue_score)

        fatigue_rows.append(
            {
                "fixture_id": fixture_id,
                "home_days_since_last_match": home_days_since_last_match,
                "away_days_since_last_match": away_days_since_last_match,
                "rest_days_diff": rest_days_diff,
                "home_matches_last_7_days": home_matches_last_7_days,
                "away_matches_last_7_days": away_matches_last_7_days,
                "matches_last_7_days_diff": home_matches_last_7_days - away_matches_last_7_days,
                "home_matches_last_14_days": home_matches_last_14_days,
                "away_matches_last_14_days": away_matches_last_14_days,
                "matches_last_14_days_diff": home_matches_last_14_days - away_matches_last_14_days,
                "home_matches_last_21_days": home_matches_last_21_days,
                "away_matches_last_21_days": away_matches_last_21_days,
                "matches_last_21_days_diff": home_matches_last_21_days - away_matches_last_21_days,
                "home_short_rest_flag": home_short_rest_flag,
                "away_short_rest_flag": away_short_rest_flag,
                "short_rest_diff": short_rest_diff,
                "home_rest_advantage_flag": home_rest_advantage_flag,
                "away_rest_advantage_flag": away_rest_advantage_flag,
                "home_fatigue_score": home_fatigue_score,
                "away_fatigue_score": away_fatigue_score,
                "fatigue_score_diff": fatigue_score_diff,
            }
        )

        team_history[home_team_id].append(match_datetime)
        team_history[away_team_id].append(match_datetime)

    fatigue = pd.DataFrame(fatigue_rows)
    merged = source.merge(fatigue, on="fixture_id", how="left")
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
    print("API-Football Ligue 1 fatigue feature build")
    print(f"Input source: {source_path}")
    print(f"Matches read: {source_rows}")
    print(f"Rows exported: {len(features)}")
    print(f"NaN rate days_since_last_match: {_nan_rate(features, ['home_days_since_last_match', 'away_days_since_last_match']):.2%}")

    print("\nTop 10 matches by absolute fatigue_score_diff:")
    preview_columns = [
        column
        for column in [
            "fixture_id",
            "date",
            "home_team_name",
            "away_team_name",
            "home_fatigue_score",
            "away_fatigue_score",
            "fatigue_score_diff",
        ]
        if column in features.columns
    ]
    if preview_columns:
        top_fatigue = features.reindex(columns=preview_columns).copy()
        if "fatigue_score_diff" in top_fatigue.columns:
            top_fatigue["_abs"] = top_fatigue["fatigue_score_diff"].abs()
            print(top_fatigue.sort_values("_abs", ascending=False).head(10).drop(columns=["_abs"]).to_string(index=False))
        else:
            print("none")
    else:
        print("none")

    print("\nExamples with short rest:")
    short_rest = features[
        (features.get("home_short_rest_flag", 0) == 1) | (features.get("away_short_rest_flag", 0) == 1)
    ]
    if not short_rest.empty:
        short_preview = [
            column
            for column in [
                "fixture_id",
                "date",
                "home_team_name",
                "away_team_name",
                "home_days_since_last_match",
                "away_days_since_last_match",
                "home_short_rest_flag",
                "away_short_rest_flag",
            ]
            if column in short_rest.columns
        ]
        print(short_rest[short_preview].head(10).to_string(index=False))
    else:
        print("none")

    print("\nNaN rates:")
    print(f"- days_since_last_match: {_nan_rate(features, ['home_days_since_last_match', 'away_days_since_last_match']):.2%}")
    print(f"- rest days diff: {_nan_rate(features, ['rest_days_diff']):.2%}")
    print(f"- matches_last_7_days: {_nan_rate(features, ['home_matches_last_7_days', 'away_matches_last_7_days']):.2%}")
    print(f"- matches_last_14_days: {_nan_rate(features, ['home_matches_last_14_days', 'away_matches_last_14_days']):.2%}")
    print(f"- matches_last_21_days: {_nan_rate(features, ['home_matches_last_21_days', 'away_matches_last_21_days']):.2%}")
    print(f"- fatigue scores: {_nan_rate(features, ['home_fatigue_score', 'away_fatigue_score', 'fatigue_score_diff']):.2%}")

    print("\nPreview:")
    preview_columns = [
        column
        for column in ["fixture_id", "date", "home_team_name", "away_team_name"] + FATIGUE_COLUMNS
        if column in features.columns
    ]
    print(features[preview_columns].head().to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build API-Football Ligue 1 fatigue features.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    return parser.parse_args()


def main() -> None:
    """Run the fatigue builder."""
    args = parse_args()
    features = build_api_football_ligue1_fatigue_features(season=args.season, league_id=args.league_id)
    destination = output_path(args.season)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)
    source_df, source_path = load_input_table(args.season)
    print_summary(features, source_rows=len(source_df), source_path=source_path)
    print(f"\nSaved fatigue features to: {destination}")


if __name__ == "__main__":
    main()
