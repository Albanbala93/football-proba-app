"""Elo rating utilities for football match feature engineering."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 20.0
DEFAULT_HOME_ADVANTAGE = 50.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return the expected score for rating_a against rating_b."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating: float, expected: float, actual: float, k: float = DEFAULT_K_FACTOR) -> float:
    """Update one Elo rating from expected and actual result values."""
    return rating + k * (actual - expected)


def _actual_home_score(result: str) -> float:
    """Convert full-time result H/D/A into the home team's Elo score."""
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    if result == "A":
        return 0.0
    raise ValueError(f"Unsupported result for Elo calculation: {result}")


def compute_elo_features(
    df: pd.DataFrame,
    initial_rating: float = INITIAL_RATING,
    k: float = DEFAULT_K_FACTOR,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> pd.DataFrame:
    """Compute pre-match Elo features for a chronological match dataframe.

    Ratings are recorded before each match result is applied. For matches on
    the same date, all pre-match Elo values are read first, then ratings are
    updated after the whole date group has been processed.
    """
    required_columns = ["Date", "HomeTeam", "AwayTeam", "FTR"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required Elo columns: {', '.join(missing_columns)}")

    matches = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).copy()
    ratings: defaultdict[str, float] = defaultdict(lambda: float(initial_rating))
    elo_rows: list[dict[str, float]] = []

    for _, date_matches in matches.groupby("Date", sort=True):
        pending_updates: list[tuple[str, float, str, float]] = []

        for index, row in date_matches.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            home_elo = ratings[home_team]
            away_elo = ratings[away_team]
            home_elo_with_advantage = home_elo + home_advantage

            elo_rows.append(
                {
                    "index": index,
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_diff": home_elo - away_elo,
                    "home_elo_with_advantage": home_elo_with_advantage,
                    "elo_diff_with_home_advantage": home_elo_with_advantage - away_elo,
                }
            )

            expected_home = expected_score(home_elo_with_advantage, away_elo)
            actual_home = _actual_home_score(str(row["FTR"]).upper())
            expected_away = 1 - expected_home
            actual_away = 1 - actual_home

            pending_updates.append((home_team, update_elo(home_elo, expected_home, actual_home, k), away_team, update_elo(away_elo, expected_away, actual_away, k)))

        # Apply updates only after recording all pre-match ratings for the date.
        for home_team, new_home_elo, away_team, new_away_elo in pending_updates:
            ratings[home_team] = new_home_elo
            ratings[away_team] = new_away_elo

    elo_features = pd.DataFrame(elo_rows).set_index("index").sort_index()
    return elo_features
