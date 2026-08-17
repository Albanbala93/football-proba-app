"""Predict probabilities for an upcoming match from historical data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.elo_model import DEFAULT_HOME_ADVANTAGE  # noqa: E402
from app.core.probability_engine import predict_match_probabilities  # noqa: E402
from app.core.probability_engine_no_odds import predict_match_probabilities_no_odds  # noqa: E402


MATCHES_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
MATCHES_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
ALLOWED_LEAGUES = ["Premier League", "Ligue 1", "La Liga", "Serie A", "Bundesliga"]

BASE_REQUIRED_FEATURES = [
    "home_form_5",
    "away_form_5",
    "home_goals_for_5",
    "away_goals_for_5",
    "home_goals_against_5",
    "away_goals_against_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_goal_diff_5",
    "away_goal_diff_5",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_elo_with_advantage",
    "elo_diff_with_home_advantage",
]

ADVANCED_REQUIRED_FEATURES = [
    "home_shots_for_5",
    "home_shots_against_5",
    "home_shots_on_target_for_5",
    "home_shots_on_target_against_5",
    "home_corners_for_5",
    "home_corners_against_5",
    "home_fouls_for_5",
    "home_fouls_against_5",
    "home_yellow_cards_5",
    "home_red_cards_5",
    "away_shots_for_5",
    "away_shots_against_5",
    "away_shots_on_target_for_5",
    "away_shots_on_target_against_5",
    "away_corners_for_5",
    "away_corners_against_5",
    "away_fouls_for_5",
    "away_fouls_against_5",
    "away_yellow_cards_5",
    "away_red_cards_5",
]

DERIVED_REQUIRED_FEATURES = [
    "home_shots_diff_5",
    "away_shots_diff_5",
    "shots_diff_between_teams",
    "home_shots_on_target_diff_5",
    "away_shots_on_target_diff_5",
    "shots_on_target_diff_between_teams",
    "home_attack_pressure_5",
    "away_attack_pressure_5",
    "home_defensive_pressure_5",
    "away_defensive_pressure_5",
    "attack_pressure_diff",
    "defensive_pressure_diff",
    "home_attack_vs_away_defense_5",
    "away_attack_vs_home_defense_5",
    "attack_matchup_diff_5",
    "home_corner_diff_5",
    "away_corner_diff_5",
    "corner_diff_between_teams",
    "home_discipline_risk_5",
    "away_discipline_risk_5",
    "discipline_risk_diff",
]

ODDS_FEATURES = [
    "B365H",
    "B365D",
    "B365A",
]

REQUIRED_FEATURES = BASE_REQUIRED_FEATURES + ADVANCED_REQUIRED_FEATURES + DERIVED_REQUIRED_FEATURES + ODDS_FEATURES

CLASSIC_MODEL_PATHS = {
    "home_win": PROJECT_ROOT / "models" / "outcome_models" / "home_win_model_classic.pkl",
    "away_win": PROJECT_ROOT / "models" / "outcome_models" / "away_win_model_classic.pkl",
    "draw": PROJECT_ROOT / "models" / "outcome_models" / "draw_model_classic.pkl",
}

NO_ODDS_MODEL_PATHS = {
    "home_win": PROJECT_ROOT / "models" / "outcome_models_no_odds" / "home_win_model_no_odds.pkl",
    "away_win": PROJECT_ROOT / "models" / "outcome_models_no_odds" / "away_win_model_no_odds.pkl",
    "draw": PROJECT_ROOT / "models" / "outcome_models_no_odds" / "draw_model_no_odds.pkl",
}

HISTORICAL_ADVANCED_SOURCE_COLUMNS = [
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
]


def _load_matches() -> pd.DataFrame:
    """Load cleaned historical matches."""
    if not MATCHES_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Cleaned matches file not found: {MATCHES_CLEAN_PATH}")
    matches = pd.read_csv(MATCHES_CLEAN_PATH, parse_dates=["Date"])
    return matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def _load_features() -> pd.DataFrame:
    """Load historical feature rows."""
    if not MATCHES_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {MATCHES_FEATURES_PATH}")
    features = pd.read_csv(MATCHES_FEATURES_PATH, parse_dates=["Date"])
    return features.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def _team_exists(team: str, matches: pd.DataFrame) -> bool:
    """Return whether a team appears in historical matches."""
    return bool(((matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)).any())


def _filter_by_league(data: pd.DataFrame, league: str | None) -> pd.DataFrame:
    """Filter data by league when a league column is available."""
    if league is None or "league" not in data.columns:
        return data
    filtered = data[data["league"].astype(str) == league].copy()
    if filtered.empty:
        raise ValueError(f"No historical data found for league '{league}'.")
    return filtered


def _league_values(data: pd.DataFrame) -> set[str]:
    """Return available league values when the column exists."""
    if "league" not in data.columns:
        return set()
    return set(data["league"].dropna().astype(str).unique())


def _team_leagues(team: str, matches: pd.DataFrame) -> set[str]:
    """Return leagues where a team appears."""
    if "league" not in matches.columns:
        return set()
    team_matches = matches[(matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)]
    return set(team_matches["league"].dropna().astype(str).unique())


def resolve_league(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    home_team: str,
    away_team: str,
    league: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the league to use without mixing competitions."""
    if league is not None:
        if league not in ALLOWED_LEAGUES:
            raise ValueError(f"Unsupported league '{league}'. Expected one of: {', '.join(ALLOWED_LEAGUES)}")
        return league, None

    available_leagues = _league_values(matches) | _league_values(features)
    if len(available_leagues) <= 1:
        return None, None

    home_leagues = _team_leagues(home_team, matches)
    away_leagues = _team_leagues(away_team, matches)
    shared_leagues = sorted(home_leagues & away_leagues)
    if len(shared_leagues) == 1:
        inferred_league = shared_leagues[0]
        return (
            inferred_league,
            "WARNING: plusieurs ligues sont presentes; league inferee "
            f"'{inferred_league}'. Utilisez --league pour lever toute ambiguite.",
        )

    raise ValueError(
        "Multiple leagues are present and the league cannot be inferred safely. "
        "Use --league with one of: " + ", ".join(ALLOWED_LEAGUES)
    )


def _team_match_record(row: pd.Series, team: str) -> dict[str, float]:
    """Build team-centric stats from one historical match row."""
    if row["HomeTeam"] == team:
        goals_for = float(row["FTHG"])
        goals_against = float(row["FTAG"])
        result = row["FTR"]
        points = 3.0 if result == "H" else 1.0 if result == "D" else 0.0
    elif row["AwayTeam"] == team:
        goals_for = float(row["FTAG"])
        goals_against = float(row["FTHG"])
        result = row["FTR"]
        points = 3.0 if result == "A" else 1.0 if result == "D" else 0.0
    else:
        raise ValueError(f"Team {team} is not present in the supplied match row.")

    return {
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "win": 1.0 if points == 3 else 0.0,
        "goal_diff": goals_for - goals_against,
    }


def get_team_recent_stats(team: str, historical_matches: pd.DataFrame, before_date: str | pd.Timestamp, n: int = 5) -> dict[str, float]:
    """Return rolling stats for a team using only matches before before_date."""
    cutoff = pd.to_datetime(before_date)
    team_matches = historical_matches[
        (historical_matches["Date"] < cutoff)
        & ((historical_matches["HomeTeam"] == team) | (historical_matches["AwayTeam"] == team))
    ].sort_values("Date")

    if team_matches.empty:
        raise ValueError(f"No historical matches found for team '{team}' before {cutoff.date()}.")

    recent_matches = team_matches.tail(n)
    records = [_team_match_record(row, team) for _, row in recent_matches.iterrows()]
    match_count = len(records)
    wins = sum(record["win"] for record in records)

    return {
        "form_5": float(sum(record["points"] for record in records)),
        "goals_for_5": float(sum(record["goals_for"] for record in records)),
        "goals_against_5": float(sum(record["goals_against"] for record in records)),
        "win_rate_5": float(wins / match_count) if match_count else 0.0,
        "goal_diff_5": float(sum(record["goal_diff"] for record in records)),
    }


def get_latest_team_elo(team: str, matches_features: pd.DataFrame, before_date: str | pd.Timestamp) -> float:
    """Return the latest available pre-match Elo for a team before before_date."""
    cutoff = pd.to_datetime(before_date)
    required_columns = ["Date", "HomeTeam", "AwayTeam", "home_elo", "away_elo"]
    missing_columns = [column for column in required_columns if column not in matches_features.columns]
    if missing_columns:
        raise ValueError(f"Missing Elo feature columns: {', '.join(missing_columns)}")

    historical_features = matches_features[matches_features["Date"] < cutoff].sort_values("Date")
    team_rows = historical_features[
        (historical_features["HomeTeam"] == team) | (historical_features["AwayTeam"] == team)
    ]
    if team_rows.empty:
        raise ValueError(f"No Elo history found for team '{team}' before {cutoff.date()}.")

    latest = team_rows.iloc[-1]
    if latest["HomeTeam"] == team:
        return float(latest["home_elo"])
    return float(latest["away_elo"])


def _numeric_or_zero(row: pd.Series, column: str) -> float:
    """Return a numeric value for a match stat, defaulting to zero when missing."""
    value = pd.to_numeric(row.get(column, 0.0), errors="coerce")
    if pd.isna(value):
        return 0.0
    return float(value)


def _team_match_advanced_record(row: pd.Series, team: str) -> dict[str, float]:
    """Build advanced team-centric stats from one historical match row."""
    if row["HomeTeam"] == team:
        return {
            "shots_for": _numeric_or_zero(row, "HS"),
            "shots_against": _numeric_or_zero(row, "AS"),
            "shots_on_target_for": _numeric_or_zero(row, "HST"),
            "shots_on_target_against": _numeric_or_zero(row, "AST"),
            "corners_for": _numeric_or_zero(row, "HC"),
            "corners_against": _numeric_or_zero(row, "AC"),
            "fouls_for": _numeric_or_zero(row, "HF"),
            "fouls_against": _numeric_or_zero(row, "AF"),
            "yellow_cards": _numeric_or_zero(row, "HY"),
            "red_cards": _numeric_or_zero(row, "HR"),
        }
    if row["AwayTeam"] == team:
        return {
            "shots_for": _numeric_or_zero(row, "AS"),
            "shots_against": _numeric_or_zero(row, "HS"),
            "shots_on_target_for": _numeric_or_zero(row, "AST"),
            "shots_on_target_against": _numeric_or_zero(row, "HST"),
            "corners_for": _numeric_or_zero(row, "AC"),
            "corners_against": _numeric_or_zero(row, "HC"),
            "fouls_for": _numeric_or_zero(row, "AF"),
            "fouls_against": _numeric_or_zero(row, "HF"),
            "yellow_cards": _numeric_or_zero(row, "AY"),
            "red_cards": _numeric_or_zero(row, "AR"),
        }
    raise ValueError(f"Team {team} is not present in the supplied match row.")


def get_team_recent_advanced_stats(
    team: str,
    historical_matches: pd.DataFrame,
    before_date: str | pd.Timestamp,
    n: int = 5,
) -> dict[str, float]:
    """Return rolling advanced stats for a team using only matches before before_date."""
    cutoff = pd.to_datetime(before_date)
    required_columns = [
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
        "HY",
        "AY",
        "HR",
        "AR",
    ]
    missing_columns = [column for column in required_columns if column not in historical_matches.columns]
    if missing_columns:
        raise ValueError(f"Missing advanced historical columns: {', '.join(missing_columns)}")

    team_matches = historical_matches[
        (historical_matches["Date"] < cutoff)
        & ((historical_matches["HomeTeam"] == team) | (historical_matches["AwayTeam"] == team))
    ].sort_values("Date")

    if team_matches.empty:
        raise ValueError(f"No historical advanced stats found for team '{team}' before {cutoff.date()}.")

    recent_matches = team_matches.tail(n)
    records = [_team_match_advanced_record(row, team) for _, row in recent_matches.iterrows()]
    match_count = len(records)

    def avg(field: str) -> float:
        return float(sum(record[field] for record in records) / match_count) if match_count else 0.0

    shots_for = avg("shots_for")
    shots_against = avg("shots_against")
    shots_on_target_for = avg("shots_on_target_for")
    shots_on_target_against = avg("shots_on_target_against")
    corners_for = avg("corners_for")
    corners_against = avg("corners_against")
    fouls_for = avg("fouls_for")
    fouls_against = avg("fouls_against")
    yellow_cards = avg("yellow_cards")
    red_cards = avg("red_cards")

    shots_diff = shots_for - shots_against
    shots_on_target_diff = shots_on_target_for - shots_on_target_against
    attack_pressure = shots_for + shots_on_target_for
    defensive_pressure = shots_against + shots_on_target_against
    attack_vs_defense = shots_for + shots_on_target_for + shots_against + shots_on_target_against
    corner_diff = corners_for - corners_against
    discipline_risk = yellow_cards + 3 * red_cards

    return {
        "shots_for_5": shots_for,
        "shots_against_5": shots_against,
        "shots_on_target_for_5": shots_on_target_for,
        "shots_on_target_against_5": shots_on_target_against,
        "corners_for_5": corners_for,
        "corners_against_5": corners_against,
        "fouls_for_5": fouls_for,
        "fouls_against_5": fouls_against,
        "yellow_cards_5": yellow_cards,
        "red_cards_5": red_cards,
        "shots_diff_5": shots_diff,
        "shots_on_target_diff_5": shots_on_target_diff,
        "attack_pressure_5": attack_pressure,
        "defensive_pressure_5": defensive_pressure,
        "attack_vs_defense_5": attack_vs_defense,
        "corner_diff_5": corner_diff,
        "discipline_risk_5": discipline_risk,
    }


def _derived_advanced_features(feature_row: dict[str, float]) -> dict[str, float]:
    """Build higher-level advanced features from already computed pre-match stats."""
    required_groups = [
        ["home_shots_for_5", "home_shots_against_5", "away_shots_for_5", "away_shots_against_5"],
        [
            "home_shots_on_target_for_5",
            "home_shots_on_target_against_5",
            "away_shots_on_target_for_5",
            "away_shots_on_target_against_5",
        ],
        ["home_corners_for_5", "home_corners_against_5", "away_corners_for_5", "away_corners_against_5"],
        ["home_yellow_cards_5", "away_yellow_cards_5", "home_red_cards_5", "away_red_cards_5"],
    ]
    if not all(all(column in feature_row for column in group) for group in required_groups):
        return {}

    home_shots_for = float(feature_row["home_shots_for_5"])
    home_shots_against = float(feature_row["home_shots_against_5"])
    away_shots_for = float(feature_row["away_shots_for_5"])
    away_shots_against = float(feature_row["away_shots_against_5"])
    home_sot_for = float(feature_row["home_shots_on_target_for_5"])
    home_sot_against = float(feature_row["home_shots_on_target_against_5"])
    away_sot_for = float(feature_row["away_shots_on_target_for_5"])
    away_sot_against = float(feature_row["away_shots_on_target_against_5"])
    home_corners_for = float(feature_row["home_corners_for_5"])
    home_corners_against = float(feature_row["home_corners_against_5"])
    away_corners_for = float(feature_row["away_corners_for_5"])
    away_corners_against = float(feature_row["away_corners_against_5"])
    home_yellow_cards = float(feature_row["home_yellow_cards_5"])
    away_yellow_cards = float(feature_row["away_yellow_cards_5"])
    home_red_cards = float(feature_row["home_red_cards_5"])
    away_red_cards = float(feature_row["away_red_cards_5"])

    home_shots_diff = home_shots_for - home_shots_against
    away_shots_diff = away_shots_for - away_shots_against
    home_sot_diff = home_sot_for - home_sot_against
    away_sot_diff = away_sot_for - away_sot_against
    home_attack_pressure = home_shots_for + home_sot_for
    away_attack_pressure = away_shots_for + away_sot_for
    home_defensive_pressure = home_shots_against + home_sot_against
    away_defensive_pressure = away_shots_against + away_sot_against
    home_attack_vs_away_defense = home_shots_for + home_sot_for + away_shots_against + away_sot_against
    away_attack_vs_home_defense = away_shots_for + away_sot_for + home_shots_against + home_sot_against
    home_corner_diff = home_corners_for - home_corners_against
    away_corner_diff = away_corners_for - away_corners_against
    home_discipline_risk = home_yellow_cards + 3 * home_red_cards
    away_discipline_risk = away_yellow_cards + 3 * away_red_cards

    return {
        "home_shots_diff_5": home_shots_diff,
        "away_shots_diff_5": away_shots_diff,
        "shots_diff_between_teams": home_shots_diff - away_shots_diff,
        "home_shots_on_target_diff_5": home_sot_diff,
        "away_shots_on_target_diff_5": away_sot_diff,
        "shots_on_target_diff_between_teams": home_sot_diff - away_sot_diff,
        "home_attack_pressure_5": home_attack_pressure,
        "away_attack_pressure_5": away_attack_pressure,
        "home_defensive_pressure_5": home_defensive_pressure,
        "away_defensive_pressure_5": away_defensive_pressure,
        "attack_pressure_diff": home_attack_pressure - away_attack_pressure,
        "defensive_pressure_diff": away_defensive_pressure - home_defensive_pressure,
        "home_attack_vs_away_defense_5": home_attack_vs_away_defense,
        "away_attack_vs_home_defense_5": away_attack_vs_home_defense,
        "attack_matchup_diff_5": home_attack_vs_away_defense - away_attack_vs_home_defense,
        "home_corner_diff_5": home_corner_diff,
        "away_corner_diff_5": away_corner_diff,
        "corner_diff_between_teams": home_corner_diff - away_corner_diff,
        "home_discipline_risk_5": home_discipline_risk,
        "away_discipline_risk_5": away_discipline_risk,
        "discipline_risk_diff": home_discipline_risk - away_discipline_risk,
    }


@lru_cache(maxsize=2)
def _expected_model_features(mode: str) -> list[str]:
    """Load the feature names expected by the trained models."""
    if mode == "no-odds":
        model_paths = NO_ODDS_MODEL_PATHS
    else:
        model_paths = CLASSIC_MODEL_PATHS

    feature_names: set[str] = set()
    for path in model_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            continue
        feature_names.update(list(model_features))
    if not feature_names:
        raise ValueError(f"Unable to determine expected features for mode '{mode}'.")
    return sorted(feature_names)


def _validate_expected_features(match_features: dict[str, float], mode: str) -> None:
    """Ensure the upcoming match payload contains all model input columns."""
    expected_features = _expected_model_features(mode)
    missing_features = [feature for feature in expected_features if feature not in match_features]
    if missing_features:
        raise ValueError(
            "Unable to build required upcoming match features. Missing columns: "
            + ", ".join(missing_features)
            + ". Relancez python scripts/rebuild_all.py si le CSV de features n'est pas a jour."
        )


def _validate_historical_advanced_columns(matches: pd.DataFrame) -> None:
    """Ensure the raw historical CSV contains the Football-Data advanced columns."""
    missing_columns = [column for column in HISTORICAL_ADVANCED_SOURCE_COLUMNS if column not in matches.columns]
    if missing_columns:
        raise ValueError(
            "Les features avancees sont absentes du CSV charge. "
            "Relancez python scripts/rebuild_all.py. Colonnes manquantes: "
            + ", ".join(missing_columns)
        )


def _estimated_market_odds(home_elo_with_advantage: float, away_elo: float) -> dict[str, float]:
    """Estimate bookmaker-like odds when real upcoming odds are unavailable.

    The trained models currently expect B365H/B365D/B365A. For upcoming matches
    without market data, this creates conservative pseudo-odds from Elo strength
    while reserving a baseline draw probability.
    """
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo_with_advantage) / 400))
    draw_probability = 0.24
    home_probability = expected_home * (1 - draw_probability)
    away_probability = (1 - expected_home) * (1 - draw_probability)
    return {
        "B365H": 1 / max(home_probability, 0.01),
        "B365D": 1 / draw_probability,
        "B365A": 1 / max(away_probability, 0.01),
    }


WITH_ODDS_REQUIRED_MESSAGE = "Le mode with-odds nécessite --odds-home, --odds-draw et --odds-away."
PSEUDO_ODDS_WARNING = "Attention : les pseudo-cotes sont dérivées de l'Elo et ne remplacent pas des cotes réelles."


def _validate_manual_odds(odds: dict[str, float | None] | None) -> dict[str, float]:
    """Validate required manual 1N2 odds supplied for an upcoming match."""
    if odds is None or all(value is None for value in odds.values()):
        raise ValueError(WITH_ODDS_REQUIRED_MESSAGE)

    missing = [label for label, value in odds.items() if value is None]
    if missing:
        raise ValueError(WITH_ODDS_REQUIRED_MESSAGE)

    invalid = [label for label, value in odds.items() if value is not None and value <= 1.0]
    if invalid:
        raise ValueError(f"Odds must be greater than 1.0. Invalid: {', '.join(invalid)}")

    return {
        "B365H": float(odds["home"]),
        "B365D": float(odds["draw"]),
        "B365A": float(odds["away"]),
    }


def build_upcoming_match_features(
    home_team: str,
    away_team: str,
    match_date: str,
    league: str | None = None,
    odds_mode: str = "none",
    manual_odds: dict[str, float | None] | None = None,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Build probability engine features for an upcoming fixture."""
    if odds_mode not in {"none", "manual", "elo_pseudo"}:
        raise ValueError("odds_mode must be 'none', 'manual' or 'elo_pseudo'.")

    cutoff = pd.to_datetime(match_date)
    all_matches = _load_matches()
    all_features = _load_features()
    resolved_league, league_warning = resolve_league(all_matches, all_features, home_team, away_team, league)
    matches = _filter_by_league(all_matches, resolved_league)
    features = _filter_by_league(all_features, resolved_league)
    _validate_historical_advanced_columns(matches)

    if not _team_exists(home_team, matches):
        raise ValueError(f"Home team '{home_team}' was not found in historical CSV files.")
    if not _team_exists(away_team, matches):
        raise ValueError(f"Away team '{away_team}' was not found in historical CSV files.")

    home_stats = get_team_recent_stats(home_team, matches, cutoff)
    away_stats = get_team_recent_stats(away_team, matches, cutoff)
    home_advanced_stats = get_team_recent_advanced_stats(home_team, matches, cutoff)
    away_advanced_stats = get_team_recent_advanced_stats(away_team, matches, cutoff)
    home_elo = get_latest_team_elo(home_team, features, cutoff)
    away_elo = get_latest_team_elo(away_team, features, cutoff)
    home_elo_with_advantage = home_elo + DEFAULT_HOME_ADVANTAGE

    match_features = {
        "league": resolved_league,
        "home_form_5": home_stats["form_5"],
        "away_form_5": away_stats["form_5"],
        "home_goals_for_5": home_stats["goals_for_5"],
        "away_goals_for_5": away_stats["goals_for_5"],
        "home_goals_against_5": home_stats["goals_against_5"],
        "away_goals_against_5": away_stats["goals_against_5"],
        "home_win_rate_5": home_stats["win_rate_5"],
        "away_win_rate_5": away_stats["win_rate_5"],
        "home_goal_diff_5": home_stats["goal_diff_5"],
        "away_goal_diff_5": away_stats["goal_diff_5"],
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_elo_with_advantage": home_elo_with_advantage,
        "elo_diff_with_home_advantage": home_elo_with_advantage - away_elo,
    }

    match_features.update(
        {
            "home_shots_for_5": home_advanced_stats["shots_for_5"],
            "home_shots_against_5": home_advanced_stats["shots_against_5"],
            "home_shots_on_target_for_5": home_advanced_stats["shots_on_target_for_5"],
            "home_shots_on_target_against_5": home_advanced_stats["shots_on_target_against_5"],
            "home_corners_for_5": home_advanced_stats["corners_for_5"],
            "home_corners_against_5": home_advanced_stats["corners_against_5"],
            "home_fouls_for_5": home_advanced_stats["fouls_for_5"],
            "home_fouls_against_5": home_advanced_stats["fouls_against_5"],
            "home_yellow_cards_5": home_advanced_stats["yellow_cards_5"],
            "home_red_cards_5": home_advanced_stats["red_cards_5"],
            "away_shots_for_5": away_advanced_stats["shots_for_5"],
            "away_shots_against_5": away_advanced_stats["shots_against_5"],
            "away_shots_on_target_for_5": away_advanced_stats["shots_on_target_for_5"],
            "away_shots_on_target_against_5": away_advanced_stats["shots_on_target_against_5"],
            "away_corners_for_5": away_advanced_stats["corners_for_5"],
            "away_corners_against_5": away_advanced_stats["corners_against_5"],
            "away_fouls_for_5": away_advanced_stats["fouls_for_5"],
            "away_fouls_against_5": away_advanced_stats["fouls_against_5"],
            "away_yellow_cards_5": away_advanced_stats["yellow_cards_5"],
            "away_red_cards_5": away_advanced_stats["red_cards_5"],
        }
    )

    match_features.update(_derived_advanced_features(match_features))

    if odds_mode == "manual":
        match_features.update(_validate_manual_odds(manual_odds))
    elif odds_mode == "elo_pseudo":
        match_features.update(_estimated_market_odds(home_elo_with_advantage, away_elo))

    validation_mode = "no-odds" if odds_mode == "none" else "classic"
    _validate_expected_features(match_features, validation_mode)

    return match_features, resolved_league, league_warning


def predict_upcoming_match(
    home_team: str,
    away_team: str,
    match_date: str,
    mode: str = "no-odds",
    league: str | None = None,
    odds_home: float | None = None,
    odds_draw: float | None = None,
    odds_away: float | None = None,
) -> dict[str, Any]:
    """Predict probabilities for an upcoming match."""
    if mode not in {"no-odds", "with-odds", "pseudo-odds"}:
        raise ValueError("mode must be 'no-odds', 'with-odds' or 'pseudo-odds'.")

    manual_odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}
    if mode == "no-odds" and any(value is not None for value in manual_odds.values()):
        raise ValueError("Les cotes ne sont pas acceptées en mode no-odds.")
    if mode == "pseudo-odds" and any(value is not None for value in manual_odds.values()):
        raise ValueError("Les cotes manuelles ne sont pas acceptées en mode pseudo-odds.")

    odds_mode = "manual" if mode == "with-odds" else "elo_pseudo" if mode == "pseudo-odds" else "none"

    match_features, resolved_league, league_warning = build_upcoming_match_features(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        league=league,
        odds_mode=odds_mode,
        manual_odds=manual_odds,
    )
    if mode == "no-odds":
        prediction = predict_match_probabilities_no_odds(match_features)
    else:
        prediction = predict_match_probabilities(match_features)
    odds_source = "manual" if mode == "with-odds" else "elo_pseudo" if mode == "pseudo-odds" else "none"
    warnings = [warning for warning in [league_warning, PSEUDO_ODDS_WARNING if mode == "pseudo-odds" else None] if warning]
    return {
        "match": {
            "home_team": home_team,
            "away_team": away_team,
            "league": resolved_league,
            "date": match_date,
            "mode": mode,
            "odds_source": odds_source,
            "warning": "\n".join(warnings) if warnings else None,
        },
        "features": match_features,
        "prediction": prediction,
    }


def _print_prediction(result: dict[str, Any]) -> None:
    """Print a readable console summary."""
    match = result["match"]
    prediction = result["prediction"]
    probabilities = prediction["probabilities"]
    analysis = prediction["analysis"]

    print(f"Match: {match['home_team']} vs {match['away_team']}")
    if match.get("league"):
        print(f"League: {match['league']}")
    print(f"Date: {match['date']}")
    print(f"Mode: {match.get('mode', 'no-odds')}")
    print(f"Odds source: {match.get('odds_source', 'none')}")
    if match.get("warning"):
        print(f"\nWARNING: {match['warning']}")
    print("\nProbabilites 1N2")
    print(f"- Victoire domicile: {probabilities['match']['home_win']:.2%}")
    print(f"- Nul: {probabilities['match']['draw']:.2%}")
    print(f"- Victoire exterieur: {probabilities['match']['away_win']:.2%}")
    print("\nEquipe domicile")
    print(f"- Victoire: {probabilities['home']['win']:.2%}")
    print(f"- Nul: {probabilities['home']['draw']:.2%}")
    print(f"- Defaite: {probabilities['home']['loss']:.2%}")
    print(f"- Non-defaite: {probabilities['home']['no_loss']:.2%}")
    print("\nEquipe exterieure")
    print(f"- Victoire: {probabilities['away']['win']:.2%}")
    print(f"- Nul: {probabilities['away']['draw']:.2%}")
    print(f"- Defaite: {probabilities['away']['loss']:.2%}")
    print(f"- Non-defaite: {probabilities['away']['no_loss']:.2%}")
    print("\nAnalyse")
    print(f"- Match profile: {analysis.get('match_profile', 'not_available')}")
    print(f"- Confidence score: {analysis.get('confidence_score', 'not_available')}/100")
    print(f"- Favorite team: {analysis['favorite_team']}")
    print(f"- Model version: {analysis.get('model_version')}")
    print(f"- Calibrated draw signal: {analysis.get('calibrated_draw_signal')}")
    print(f"- Draw plausible: {analysis.get('is_draw_plausible')}")
    print(f"- Strong draw signal: {analysis.get('is_strong_draw_signal')}")
    print(f"- Very strong draw signal: {analysis.get('is_very_strong_draw_signal')}")
    print("\nJSON")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict an upcoming football match.")
    parser.add_argument("--home", required=True, help="Home team name exactly as it appears in the CSV files.")
    parser.add_argument("--away", required=True, help="Away team name exactly as it appears in the CSV files.")
    parser.add_argument("--date", required=True, help="Match date, for example 2026-05-20.")
    parser.add_argument("--league", choices=ALLOWED_LEAGUES, default=None, help="Optional league name.")
    parser.add_argument(
        "--mode",
        choices=["no-odds", "with-odds", "pseudo-odds"],
        default="no-odds",
        help="Prediction mode. no-odds is the default; with-odds requires real odds; pseudo-odds derives odds from Elo.",
    )
    parser.add_argument("--odds-home", type=float, default=None, help="Manual home win odds for --mode with-odds.")
    parser.add_argument("--odds-draw", type=float, default=None, help="Manual draw odds for --mode with-odds.")
    parser.add_argument("--odds-away", type=float, default=None, help="Manual away win odds for --mode with-odds.")
    return parser.parse_args()


def main() -> None:
    """Run upcoming match prediction from the command line."""
    args = parse_args()
    try:
        result = predict_upcoming_match(
            home_team=args.home,
            away_team=args.away,
            match_date=args.date,
            mode=args.mode,
            league=args.league,
            odds_home=args.odds_home,
            odds_draw=args.odds_draw,
            odds_away=args.odds_away,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
    _print_prediction(result)


if __name__ == "__main__":
    main()
