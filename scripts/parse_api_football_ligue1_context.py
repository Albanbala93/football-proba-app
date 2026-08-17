"""Parse raw API-Football Ligue 1 JSON into normalized context tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEAGUE_ID = 61
DEFAULT_SEASON = 2025

STATISTIC_TYPE_TO_COLUMN = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "total_shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_insidebox",
    "Shots outsidebox": "shots_outsidebox",
    "Fouls": "fouls",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "ball_possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes",
    "Passes accurate": "passes_accurate",
    "Passes %": "passes_percentage",
    "expected_goals": "expected_goals",
    "Expected Goals": "expected_goals",
    "xG": "expected_goals",
}
STAT_COLUMNS = [
    "shots_on_goal",
    "shots_off_goal",
    "total_shots",
    "blocked_shots",
    "shots_insidebox",
    "shots_outsidebox",
    "fouls",
    "corner_kicks",
    "offsides",
    "ball_possession",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
    "total_passes",
    "passes_accurate",
    "passes_percentage",
    "expected_goals",
]


def _read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _response_items(path: Path) -> list[Any]:
    """Return the response list from an API-Football JSON file."""
    if not path.exists():
        return []
    payload = _read_json(path)
    response = payload.get("response") if isinstance(payload, dict) else None
    return response if isinstance(response, list) else []


def _to_number(value: Any) -> float | int | None:
    """Convert API numeric strings and percentages to numbers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _compact_players(players: list[Any]) -> str:
    """Serialize player list into compact JSON with stable useful fields."""
    compact = []
    for item in players:
        if not isinstance(item, dict):
            continue
        player = item.get("player", {})
        if not isinstance(player, dict):
            continue
        compact.append(
            {
                "id": player.get("id"),
                "name": player.get("name"),
                "number": player.get("number"),
                "pos": player.get("pos"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def parse_fixtures(input_dir: Path, league_id: int, season: int) -> pd.DataFrame:
    """Parse fixtures.json into one row per match."""
    fixtures_path = input_dir / "fixtures.json"
    rows = []
    response_items = _response_items(fixtures_path)
    if response_items:
        source_items = response_items
    else:
        index_path = input_dir / "fixtures_index.csv"
        if not index_path.exists():
            return pd.DataFrame()
        index_df = pd.read_csv(index_path)
        if index_df.empty:
            return pd.DataFrame()
        source_items = index_df.to_dict("records")

    for item in source_items:
        if not isinstance(item, dict):
            continue
        if "fixture_id" in item and "home_team_name" in item and "away_team_name" in item:
            rows.append(
                {
                    "fixture_id": item.get("fixture_id"),
                    "date": item.get("date"),
                    "league_id": league_id,
                    "season": season,
                    "home_team_id": item.get("home_team_id"),
                    "home_team_name": item.get("home_team_name"),
                    "away_team_id": item.get("away_team_id"),
                    "away_team_name": item.get("away_team_name"),
                    "goals_home": item.get("goals_home"),
                    "goals_away": item.get("goals_away"),
                    "status_short": item.get("status_short"),
                }
            )
            continue

        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home = teams.get("home", {}) if isinstance(teams, dict) else {}
        away = teams.get("away", {}) if isinstance(teams, dict) else {}
        status = fixture.get("status", {}) if isinstance(fixture, dict) else {}
        rows.append(
            {
                "fixture_id": fixture.get("id") if isinstance(fixture, dict) else None,
                "date": fixture.get("date") if isinstance(fixture, dict) else None,
                "league_id": league_id,
                "season": season,
                "home_team_id": home.get("id"),
                "home_team_name": home.get("name"),
                "away_team_id": away.get("id"),
                "away_team_name": away.get("name"),
                "goals_home": goals.get("home") if isinstance(goals, dict) else None,
                "goals_away": goals.get("away") if isinstance(goals, dict) else None,
                "status_short": status.get("short") if isinstance(status, dict) else None,
            }
        )
    return pd.DataFrame(rows)


def parse_statistics(input_dir: Path, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Parse per-fixture statistics into one row per team per fixture."""
    rows = []
    statistics_dir = input_dir / "statistics"
    for fixture in fixtures.to_dict("records"):
        fixture_id = fixture["fixture_id"]
        path = statistics_dir / f"fixture_{fixture_id}.json"
        for team_stats in _response_items(path):
            if not isinstance(team_stats, dict):
                continue
            team = team_stats.get("team", {})
            team_id = team.get("id") if isinstance(team, dict) else None
            row = {
                "fixture_id": fixture_id,
                "team_id": team_id,
                "team_name": team.get("name") if isinstance(team, dict) else None,
                "is_home": team_id == fixture["home_team_id"],
            }
            row.update({column: None for column in STAT_COLUMNS})
            statistics = team_stats.get("statistics", [])
            if isinstance(statistics, list):
                for stat in statistics:
                    if not isinstance(stat, dict):
                        continue
                    column = STATISTIC_TYPE_TO_COLUMN.get(str(stat.get("type")))
                    if column:
                        row[column] = _to_number(stat.get("value"))
            rows.append(row)
    return pd.DataFrame(rows)


def parse_lineups(input_dir: Path, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Parse per-fixture lineups into one row per team per fixture."""
    rows = []
    lineups_dir = input_dir / "lineups"
    for fixture in fixtures.to_dict("records"):
        fixture_id = fixture["fixture_id"]
        path = lineups_dir / f"fixture_{fixture_id}.json"
        for lineup in _response_items(path):
            if not isinstance(lineup, dict):
                continue
            team = lineup.get("team", {})
            coach = lineup.get("coach", {})
            team_id = team.get("id") if isinstance(team, dict) else None
            start_xi = lineup.get("startXI", [])
            substitutes = lineup.get("substitutes", [])
            start_xi = start_xi if isinstance(start_xi, list) else []
            substitutes = substitutes if isinstance(substitutes, list) else []
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "team_name": team.get("name") if isinstance(team, dict) else None,
                    "is_home": team_id == fixture["home_team_id"],
                    "formation": lineup.get("formation"),
                    "coach_name": coach.get("name") if isinstance(coach, dict) else None,
                    "startXI_count": len(start_xi),
                    "substitutes_count": len(substitutes),
                    "startXI_players": _compact_players(start_xi),
                    "substitutes_players": _compact_players(substitutes),
                }
            )
    return pd.DataFrame(rows)


def parse_injuries(input_dir: Path) -> pd.DataFrame:
    """Parse injuries.json into normalized injury rows."""
    rows = []
    injuries_path = input_dir / "injuries.json"
    for item in _response_items(injuries_path):
        if not isinstance(item, dict):
            continue
        player = item.get("player", {})
        team = item.get("team", {})
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        rows.append(
            {
                "fixture_id": fixture.get("id") if isinstance(fixture, dict) else None,
                "league_id": league.get("id") if isinstance(league, dict) else None,
                "season": league.get("season") if isinstance(league, dict) else None,
                "team_id": team.get("id") if isinstance(team, dict) else None,
                "team_name": team.get("name") if isinstance(team, dict) else None,
                "player_id": player.get("id") if isinstance(player, dict) else None,
                "player_name": player.get("name") if isinstance(player, dict) else None,
                "type": player.get("type") if isinstance(player, dict) else None,
                "reason": player.get("reason") if isinstance(player, dict) else None,
                "date": fixture.get("date") if isinstance(fixture, dict) else None,
            }
        )
    return pd.DataFrame(rows)


def _side_value(df: pd.DataFrame, fixture_id: Any, is_home: bool, column: str) -> Any:
    """Return one side-specific value from a team-level table."""
    if df.empty or column not in df.columns:
        return None
    rows = df[(df["fixture_id"] == fixture_id) & (df["is_home"] == is_home)]
    if rows.empty:
        return None
    return rows.iloc[0][column]


def build_context_features(
    fixtures: pd.DataFrame,
    statistics: pd.DataFrame,
    lineups: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Build one match-level context row per fixture."""
    rows = []
    injury_counts = pd.DataFrame()
    if not injuries.empty and {"fixture_id", "team_id"}.issubset(injuries.columns):
        injury_counts = injuries.groupby(["fixture_id", "team_id"], dropna=False).size().reset_index(name="injuries")

    for fixture in fixtures.to_dict("records"):
        fixture_id = fixture["fixture_id"]
        home_team_id = fixture["home_team_id"]
        away_team_id = fixture["away_team_id"]
        home_formation = _side_value(lineups, fixture_id, True, "formation")
        away_formation = _side_value(lineups, fixture_id, False, "formation")

        if injury_counts.empty:
            home_injuries = 0
            away_injuries = 0
        else:
            home_rows = injury_counts[
                (injury_counts["fixture_id"] == fixture_id) & (injury_counts["team_id"] == home_team_id)
            ]
            away_rows = injury_counts[
                (injury_counts["fixture_id"] == fixture_id) & (injury_counts["team_id"] == away_team_id)
            ]
            home_injuries = int(home_rows["injuries"].iloc[0]) if not home_rows.empty else 0
            away_injuries = int(away_rows["injuries"].iloc[0]) if not away_rows.empty else 0

        row = {
            **fixture,
            "home_formation": home_formation,
            "away_formation": away_formation,
            "formation_matchup": (
                f"{home_formation} vs {away_formation}" if pd.notna(home_formation) and pd.notna(away_formation) else None
            ),
            "home_injuries_count": home_injuries,
            "away_injuries_count": away_injuries,
            "injuries_diff": home_injuries - away_injuries,
        }
        for column in [
            "shots_on_goal",
            "total_shots",
            "ball_possession",
            "corner_kicks",
            "fouls",
            "yellow_cards",
            "red_cards",
            "expected_goals",
        ]:
            row[f"home_{column}"] = _side_value(statistics, fixture_id, True, column)
            row[f"away_{column}"] = _side_value(statistics, fixture_id, False, column)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_api_football_ligue1_context(league_id: int = DEFAULT_LEAGUE_ID, season: int = DEFAULT_SEASON) -> None:
    """Parse raw API-Football files into normalized processed CSVs."""
    input_dir = PROJECT_ROOT / "data" / "external" / "api_football" / f"ligue1_{season}"
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = parse_fixtures(input_dir, league_id=league_id, season=season)
    statistics = parse_statistics(input_dir, fixtures)
    lineups = parse_lineups(input_dir, fixtures)
    injuries = parse_injuries(input_dir)
    context = build_context_features(fixtures, statistics, lineups, injuries)

    match_stats_path = output_dir / f"api_football_ligue1_{season}_match_stats.csv"
    lineups_path = output_dir / f"api_football_ligue1_{season}_lineups.csv"
    injuries_path = output_dir / f"api_football_ligue1_{season}_injuries.csv"
    context_path = output_dir / f"api_football_ligue1_{season}_context_features.csv"

    statistics.to_csv(match_stats_path, index=False)
    lineups.to_csv(lineups_path, index=False)
    injuries.to_csv(injuries_path, index=False)
    context.to_csv(context_path, index=False)

    print("API-Football Ligue 1 context parsing")
    print(f"Input directory: {input_dir}")
    print(f"Fixtures: {len(fixtures)}")
    print(f"Team match stats rows: {len(statistics)}")
    print(f"Lineup rows: {len(lineups)}")
    print(f"Injury rows: {len(injuries)}")
    print(f"Context feature rows: {len(context)}")
    print(f"Expected goals present: {statistics['expected_goals'].notna().any() if not statistics.empty else False}")
    print("\nSaved outputs:")
    print(f"- {match_stats_path}")
    print(f"- {lineups_path}")
    print(f"- {injuries_path}")
    print(f"- {context_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse API-Football Ligue 1 context data.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--league-id", type=int, default=DEFAULT_LEAGUE_ID)
    return parser.parse_args()


def main() -> None:
    """Run the parser."""
    args = parse_args()
    parse_api_football_ligue1_context(league_id=args.league_id, season=args.season)


if __name__ == "__main__":
    main()
