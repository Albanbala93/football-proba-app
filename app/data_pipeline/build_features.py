"""Build pre-match form features from cleaned football match data.

The input file is expected at ``data/processed/matches_clean.csv`` and the
default output is ``data/processed/matches_features.csv``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.elo_model import compute_elo_features  # noqa: E402

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
FORM_WINDOW = 5
UNKNOWN_LEAGUE = "unknown"
ADVANCED_FEATURE_COLUMNS = [
    "home_shots_for_5",
    "home_shots_against_5",
    "away_shots_for_5",
    "away_shots_against_5",
    "home_shots_on_target_for_5",
    "home_shots_on_target_against_5",
    "away_shots_on_target_for_5",
    "away_shots_on_target_against_5",
    "home_corners_for_5",
    "home_corners_against_5",
    "away_corners_for_5",
    "away_corners_against_5",
    "home_fouls_for_5",
    "home_fouls_against_5",
    "away_fouls_for_5",
    "away_fouls_against_5",
    "home_yellow_cards_5",
    "away_yellow_cards_5",
    "home_red_cards_5",
    "away_red_cards_5",
]
DERIVED_ADVANCED_FEATURE_COLUMNS = [
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
DRAW_SIGNAL_FEATURE_COLUMNS = [
    "home_draw_rate_5",
    "away_draw_rate_5",
    "h2h_draw_rate",
    "h2h_matches_count",
    "abs_elo_diff",
    "abs_form_diff_5",
]


def _numeric_or_zero(row: pd.Series, column: str) -> float:
    """Return a numeric match stat value, using zero when unavailable."""
    value = pd.to_numeric(row.get(column, 0.0), errors="coerce")
    if pd.isna(value):
        return 0.0
    return float(value)


def _has_columns(row: dict[str, float] | pd.Series, columns: list[str]) -> bool:
    """Return True when all source columns are available in the row."""
    return all(column in row for column in columns)


def _safe_feature_value(row: dict[str, float] | pd.Series, column: str) -> float:
    """Read a pre-match feature value, defaulting to zero when missing."""
    value = pd.to_numeric(row.get(column, 0.0), errors="coerce")
    if pd.isna(value):
        return 0.0
    return float(value)


def _derived_advanced_features(feature_row: dict[str, float], row: pd.Series) -> dict[str, float]:
    """Build higher-level advanced features from already computed pre-match stats."""
    required_groups = [
        ["home_shots_for_5", "home_shots_against_5", "away_shots_for_5", "away_shots_against_5"],
        [
            "home_shots_on_target_for_5",
            "home_shots_on_target_against_5",
            "away_shots_on_target_for_5",
            "away_shots_on_target_against_5",
        ],
        [
            "home_corners_for_5",
            "home_corners_against_5",
            "away_corners_for_5",
            "away_corners_against_5",
        ],
        ["home_yellow_cards_5", "away_yellow_cards_5", "home_red_cards_5", "away_red_cards_5"],
    ]
    if not all(_has_columns(feature_row, group) for group in required_groups):
        return {}

    home_shots_for = _safe_feature_value(feature_row, "home_shots_for_5")
    home_shots_against = _safe_feature_value(feature_row, "home_shots_against_5")
    away_shots_for = _safe_feature_value(feature_row, "away_shots_for_5")
    away_shots_against = _safe_feature_value(feature_row, "away_shots_against_5")
    home_sot_for = _safe_feature_value(feature_row, "home_shots_on_target_for_5")
    home_sot_against = _safe_feature_value(feature_row, "home_shots_on_target_against_5")
    away_sot_for = _safe_feature_value(feature_row, "away_shots_on_target_for_5")
    away_sot_against = _safe_feature_value(feature_row, "away_shots_on_target_against_5")
    home_corners_for = _safe_feature_value(feature_row, "home_corners_for_5")
    home_corners_against = _safe_feature_value(feature_row, "home_corners_against_5")
    away_corners_for = _safe_feature_value(feature_row, "away_corners_for_5")
    away_corners_against = _safe_feature_value(feature_row, "away_corners_against_5")
    home_yellow_cards = _safe_feature_value(feature_row, "home_yellow_cards_5")
    away_yellow_cards = _safe_feature_value(feature_row, "away_yellow_cards_5")
    home_red_cards = _safe_feature_value(feature_row, "home_red_cards_5")
    away_red_cards = _safe_feature_value(feature_row, "away_red_cards_5")

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


def _result_points(row: pd.Series, team_side: str) -> int:
    """Return match points for home or away team from the full-time result."""
    if row["FTR"] == "D":
        return 1
    if team_side == "home" and row["FTR"] == "H":
        return 3
    if team_side == "away" and row["FTR"] == "A":
        return 3
    return 0


def _match_record(row: pd.Series, team_side: str) -> dict[str, float]:
    """Build a team-centric record for one completed match."""
    if team_side == "home":
        goals_for = float(row["FTHG"])
        goals_against = float(row["FTAG"])
        shots_for = _numeric_or_zero(row, "HS")
        shots_against = _numeric_or_zero(row, "AS")
        shots_on_target_for = _numeric_or_zero(row, "HST")
        shots_on_target_against = _numeric_or_zero(row, "AST")
        corners_for = _numeric_or_zero(row, "HC")
        corners_against = _numeric_or_zero(row, "AC")
        fouls_for = _numeric_or_zero(row, "HF")
        fouls_against = _numeric_or_zero(row, "AF")
        yellow_cards = _numeric_or_zero(row, "HY")
        red_cards = _numeric_or_zero(row, "HR")
    else:
        goals_for = float(row["FTAG"])
        goals_against = float(row["FTHG"])
        shots_for = _numeric_or_zero(row, "AS")
        shots_against = _numeric_or_zero(row, "HS")
        shots_on_target_for = _numeric_or_zero(row, "AST")
        shots_on_target_against = _numeric_or_zero(row, "HST")
        corners_for = _numeric_or_zero(row, "AC")
        corners_against = _numeric_or_zero(row, "HC")
        fouls_for = _numeric_or_zero(row, "AF")
        fouls_against = _numeric_or_zero(row, "HF")
        yellow_cards = _numeric_or_zero(row, "AY")
        red_cards = _numeric_or_zero(row, "AR")

    points = float(_result_points(row, team_side))
    return {
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "win": 1.0 if points == 3 else 0.0,
        "draw": 1.0 if row["FTR"] == "D" else 0.0,
        "goal_diff": goals_for - goals_against,
        "shots_for": shots_for,
        "shots_against": shots_against,
        "shots_on_target_for": shots_on_target_for,
        "shots_on_target_against": shots_on_target_against,
        "corners_for": corners_for,
        "corners_against": corners_against,
        "fouls_for": fouls_for,
        "fouls_against": fouls_against,
        "yellow_cards": yellow_cards,
        "red_cards": red_cards,
    }


def _sum_recent(records: Iterable[dict[str, float]], field: str) -> float:
    """Sum a field over a team's available historical records."""
    return float(sum(record[field] for record in records))


def _team_form_features(
    team_history: dict[tuple[str, str], deque[dict[str, float]]],
    league: str,
    team: str,
    prefix: str,
) -> dict[str, float]:
    """Return rolling form features using only matches already in history.

    When fewer than five prior matches exist, the function uses the available
    prior matches. New teams receive zero-valued form features.
    """
    recent_matches = list(team_history[(league, team)])
    matches_count = len(recent_matches)
    wins = _sum_recent(recent_matches, "win")
    draws = _sum_recent(recent_matches, "draw")

    return {
        f"{prefix}_form_5": _sum_recent(recent_matches, "points"),
        f"{prefix}_goals_for_5": _sum_recent(recent_matches, "goals_for"),
        f"{prefix}_goals_against_5": _sum_recent(recent_matches, "goals_against"),
        f"{prefix}_win_rate_5": wins / matches_count if matches_count else 0.0,
        f"{prefix}_draw_rate_5": draws / matches_count if matches_count else 0.0,
        f"{prefix}_goal_diff_5": _sum_recent(recent_matches, "goal_diff"),
        f"{prefix}_shots_for_5": _sum_recent(recent_matches, "shots_for"),
        f"{prefix}_shots_against_5": _sum_recent(recent_matches, "shots_against"),
        f"{prefix}_shots_on_target_for_5": _sum_recent(recent_matches, "shots_on_target_for"),
        f"{prefix}_shots_on_target_against_5": _sum_recent(recent_matches, "shots_on_target_against"),
        f"{prefix}_corners_for_5": _sum_recent(recent_matches, "corners_for"),
        f"{prefix}_corners_against_5": _sum_recent(recent_matches, "corners_against"),
        f"{prefix}_fouls_for_5": _sum_recent(recent_matches, "fouls_for"),
        f"{prefix}_fouls_against_5": _sum_recent(recent_matches, "fouls_against"),
        f"{prefix}_yellow_cards_5": _sum_recent(recent_matches, "yellow_cards"),
        f"{prefix}_red_cards_5": _sum_recent(recent_matches, "red_cards"),
    }


def compute_league_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo features independently for each league."""
    if "league" not in matches.columns:
        return compute_elo_features(matches)

    elo_frames = []
    for _, league_matches in matches.groupby("league", sort=True, dropna=False):
        elo_frames.append(compute_elo_features(league_matches))
    return pd.concat(elo_frames).sort_index()


def _h2h_key(home_team: str, away_team: str) -> frozenset[str]:
    """Return an order-independent key identifying a pair of teams."""
    return frozenset((home_team, away_team))


def _h2h_features(h2h_history: dict[frozenset[str], list[bool]], home_team: str, away_team: str) -> dict[str, float]:
    """Return the pre-match head-to-head draw rate between two teams.

    Uses full history (not just the last five matches) since two teams
    typically only meet once or twice per season.
    """
    past_results = h2h_history[_h2h_key(home_team, away_team)]
    matches_count = len(past_results)
    draw_rate = sum(past_results) / matches_count if matches_count else 0.0
    return {
        "h2h_draw_rate": draw_rate,
        "h2h_matches_count": float(matches_count),
    }


def _draw_signal_features(feature_row: dict[str, float]) -> dict[str, float]:
    """Return magnitude-based features useful to spot evenly matched teams.

    A signed difference (elo_diff, form diff) tells a linear model which side
    is favored, but not how close the match is. Evenly matched teams (small
    absolute gaps) draw more often, so the absolute gap is its own signal.
    """
    elo_diff = _safe_feature_value(feature_row, "elo_diff")
    form_diff = _safe_feature_value(feature_row, "home_form_5") - _safe_feature_value(feature_row, "away_form_5")
    return {
        "abs_elo_diff": abs(elo_diff),
        "abs_form_diff_5": abs(form_diff),
    }


def _implied_probabilities(row: pd.Series) -> dict[str, float]:
    """Convert bookmaker odds into normalized implied probabilities."""
    implied_home = 1 / row["B365H"]
    implied_draw = 1 / row["B365D"]
    implied_away = 1 / row["B365A"]
    implied_total = implied_home + implied_draw + implied_away

    return {
        "implied_home_prob": implied_home / implied_total,
        "implied_draw_prob": implied_draw / implied_total,
        "implied_away_prob": implied_away / implied_total,
    }


def build_features(input_path: Path = DEFAULT_INPUT_PATH, output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Create rolling five-match form features for each fixture.

    Features are calculated before updating team histories with the current
    match result, so a match never uses its own result to build its row.
    Matches with the same date are also isolated from one another: the whole
    date group is featurized first, then the group results are added to history.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned matches file not found: {input_path}")

    matches = pd.read_csv(input_path, parse_dates=["Date"])
    if "league" not in matches.columns:
        matches["league"] = UNKNOWN_LEAGUE
    if "season" not in matches.columns:
        matches["season"] = pd.NA

    matches["league"] = matches["league"].fillna(UNKNOWN_LEAGUE).astype(str)
    matches = matches.sort_values(["Date", "league", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    elo_features = compute_league_elo_features(matches)

    team_history: dict[tuple[str, str], deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h_history: dict[frozenset[str], list[bool]] = defaultdict(list)
    feature_rows = []

    for _, date_matches in matches.groupby("Date", sort=True):
        for row_index, row in date_matches.iterrows():
            league = row["league"]
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]

            feature_row = {
                "Date": row["Date"],
                "league": league,
                "season": row["season"],
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "FTHG": row["FTHG"],
                "FTAG": row["FTAG"],
                "FTR": row["FTR"],
                "B365H": row["B365H"],
                "B365D": row["B365D"],
                "B365A": row["B365A"],
                "target": row["FTR"],
            }
            feature_row.update(_team_form_features(team_history, league, home_team, "home"))
            feature_row.update(_team_form_features(team_history, league, away_team, "away"))
            feature_row.update(elo_features.loc[row_index].to_dict())
            feature_row.update(_implied_probabilities(row))
            feature_row.update(_derived_advanced_features(feature_row, row))
            feature_row.update(_h2h_features(h2h_history, home_team, away_team))
            feature_row.update(_draw_signal_features(feature_row))
            feature_rows.append(feature_row)

        for _, row in date_matches.iterrows():
            league = row["league"]
            team_history[(league, row["HomeTeam"])].append(_match_record(row, "home"))
            team_history[(league, row["AwayTeam"])].append(_match_record(row, "away"))
            h2h_history[_h2h_key(row["HomeTeam"], row["AwayTeam"])].append(row["FTR"] == "D")

    features = pd.DataFrame(feature_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature generation."""
    parser = argparse.ArgumentParser(description="Build rolling form features from cleaned matches.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input cleaned CSV path. Defaults to data/processed/matches_clean.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output feature CSV path. Defaults to data/processed/matches_features.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Run feature generation from the command line."""
    args = parse_args()
    print(f"Reading cleaned matches from: {args.input}")
    features = build_features(input_path=args.input, output_path=args.output)
    print(f"Created {len(features)} feature rows.")
    print("Added Elo features: home_elo, away_elo, elo_diff, home_elo_with_advantage, elo_diff_with_home_advantage.")
    added_advanced_features = [column for column in ADVANCED_FEATURE_COLUMNS if column in features.columns]
    print(f"Added advanced rolling features: {len(added_advanced_features)}")
    if added_advanced_features:
        print(", ".join(added_advanced_features))
    added_derived_features = [column for column in DERIVED_ADVANCED_FEATURE_COLUMNS if column in features.columns]
    print(f"Added derived advanced features: {len(added_derived_features)}")
    if added_derived_features:
        print(", ".join(added_derived_features))
    added_draw_signal_features = [column for column in DRAW_SIGNAL_FEATURE_COLUMNS if column in features.columns]
    print(f"Added draw signal features: {len(added_draw_signal_features)}")
    if added_draw_signal_features:
        print(", ".join(added_draw_signal_features))
    if not features.empty:
        print(f"Date range: {features['Date'].min().date()} to {features['Date'].max().date()}")
    print(f"Saved features to: {args.output}")


if __name__ == "__main__":
    main()
