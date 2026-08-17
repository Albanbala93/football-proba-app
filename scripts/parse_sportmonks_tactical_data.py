"""Parse raw Sportmonks tactical JSON files into model-ready CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LINEUPS_PATH = PROJECT_ROOT / "data" / "external" / "lineups" / "lineups.csv"
DEFAULT_FORMATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "match_formations.csv"
DEFAULT_TACTICAL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "tactical_features.csv"

STARTER_TYPE_ID = 11
BENCH_TYPE_ID = 12
POSITION_NAMES = {
    24: "Goalkeeper",
    25: "Defender",
    26: "Midfielder",
    27: "Forward",
    148: "Attacker",
    149: "Defender",
    150: "Midfielder",
    151: "Goalkeeper",
}


def _as_list(value: Any) -> list[Any]:
    """Return a Sportmonks value as a list, unwrapping data if needed."""
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _fixture_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the fixture object from a Sportmonks JSON payload."""
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def _participant_name(participant: dict[str, Any]) -> str | None:
    """Return the best available team name."""
    for key in ["name", "short_code", "common_name"]:
        value = participant.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _participants_by_team(fixture: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return participants indexed by team id."""
    participants = {}
    for participant in _as_list(fixture.get("participants")):
        if not isinstance(participant, dict):
            continue
        team_id = participant.get("id")
        if team_id is None:
            continue
        participants[int(team_id)] = participant
    return participants


def _home_away_teams(participants: dict[int, dict[str, Any]], fixture_name: str | None) -> tuple[int | None, int | None]:
    """Return home and away team ids from participant metadata."""
    home_team_id = None
    away_team_id = None
    for team_id, participant in participants.items():
        meta = participant.get("meta") if isinstance(participant.get("meta"), dict) else {}
        location = str(meta.get("location", "")).lower()
        if location == "home":
            home_team_id = team_id
        elif location == "away":
            away_team_id = team_id

    if (home_team_id is None or away_team_id is None) and fixture_name and " vs " in fixture_name:
        home_name, away_name = [part.strip() for part in fixture_name.split(" vs ", 1)]
        for team_id, participant in participants.items():
            name = _participant_name(participant)
            if name == home_name:
                home_team_id = team_id
            elif name == away_name:
                away_team_id = team_id

    return home_team_id, away_team_id


def _formations_by_team(fixture: dict[str, Any]) -> dict[int, str]:
    """Return formation strings indexed by team id."""
    formations = {}
    for formation in _as_list(fixture.get("formations")):
        if not isinstance(formation, dict):
            continue
        team_id = formation.get("participant_id") or formation.get("team_id")
        formation_value = formation.get("formation")
        if team_id is not None and isinstance(formation_value, str) and formation_value.strip():
            formations[int(team_id)] = formation_value.strip()
    return formations


def _lineup_type(type_id: Any) -> str | None:
    """Return a readable lineup type from Sportmonks type_id."""
    if type_id == STARTER_TYPE_ID:
        return "starter"
    if type_id == BENCH_TYPE_ID:
        return "bench"
    return str(type_id) if type_id is not None else None


def _is_starter(lineup: dict[str, Any]) -> bool:
    """Return whether a lineup row is a starter."""
    if lineup.get("type_id") == STARTER_TYPE_ID:
        return True
    formation_field = lineup.get("formation_field")
    return isinstance(formation_field, str) and bool(formation_field.strip())


def _position_name(lineup: dict[str, Any]) -> str | None:
    """Return position name from nested position data or known ids."""
    position = lineup.get("position")
    if isinstance(position, dict):
        name = position.get("name")
        if isinstance(name, str):
            return name
    position_id = lineup.get("position_id")
    try:
        return POSITION_NAMES.get(int(position_id))
    except (TypeError, ValueError):
        return None


def _fixture_date(fixture: dict[str, Any]) -> str | None:
    """Return fixture date as YYYY-MM-DD."""
    starting_at = fixture.get("starting_at")
    if isinstance(starting_at, str) and starting_at:
        return starting_at[:10]
    return None


def _fixture_scores(fixture: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    """Extract final/current home and away score plus H/D/A result when available."""
    score_rows = [score for score in _as_list(fixture.get("scores")) if isinstance(score, dict)]
    if not score_rows:
        return None, None, None

    selected_rows = [
        score
        for score in score_rows
        if str(score.get("description", "")).upper() in {"CURRENT", "FT", "FULL_TIME"}
    ]
    if not selected_rows:
        selected_rows = score_rows

    home_score = None
    away_score = None
    for score_row in selected_rows:
        score = score_row.get("score") if isinstance(score_row.get("score"), dict) else {}
        participant = str(score.get("participant", "")).lower()
        goals = score.get("goals")
        try:
            goals = int(goals)
        except (TypeError, ValueError):
            continue
        if participant == "home":
            home_score = goals
        elif participant == "away":
            away_score = goals

    if home_score is None or away_score is None:
        return home_score, away_score, None
    if home_score > away_score:
        return home_score, away_score, "H"
    if home_score < away_score:
        return home_score, away_score, "A"
    return home_score, away_score, "D"


def _parse_formation(formation: str | None) -> dict[str, Any]:
    """Convert a formation like 4-2-3-1 into tactical counts."""
    if not formation:
        return {
            "defenders_count": None,
            "midfielders_count": None,
            "forwards_count": None,
            "back_three": False,
            "back_four": False,
            "back_five": False,
            "two_strikers": False,
        }

    parts = []
    for value in str(formation).replace(" ", "").split("-"):
        try:
            parts.append(int(value))
        except ValueError:
            return {
                "defenders_count": None,
                "midfielders_count": None,
                "forwards_count": None,
                "back_three": False,
                "back_four": False,
                "back_five": False,
                "two_strikers": False,
            }

    defenders = parts[0] if parts else None
    forwards = parts[-1] if parts else None
    midfielders = sum(parts[1:-1]) if len(parts) > 2 else 0
    return {
        "defenders_count": defenders,
        "midfielders_count": midfielders,
        "forwards_count": forwards,
        "back_three": defenders == 3,
        "back_four": defenders == 4,
        "back_five": defenders == 5,
        "two_strikers": forwards == 2,
    }


def _diff(left: Any, right: Any) -> Any:
    """Return left minus right when both values are present."""
    if left is None or right is None:
        return None
    return left - right


def _stability_from_history(current_formation: str | None, history: list[str], lookback: int = 5) -> tuple[float | None, int]:
    """Return formation stability against the previous available formations."""
    recent_formations = [formation for formation in history[-lookback:] if formation]
    if not recent_formations or not current_formation:
        return None, len(recent_formations)
    same_formation_count = sum(formation == current_formation for formation in recent_formations)
    return same_formation_count / len(recent_formations), len(recent_formations)


def _add_formation_stability(tactical_features: pd.DataFrame) -> pd.DataFrame:
    """Add home/away formation stability using strictly previous matches by team."""
    if tactical_features.empty:
        tactical_features["home_formation_stability_5"] = pd.Series(dtype=float)
        tactical_features["away_formation_stability_5"] = pd.Series(dtype=float)
        tactical_features["home_recent_formations_count"] = pd.Series(dtype=int)
        tactical_features["away_recent_formations_count"] = pd.Series(dtype=int)
        return tactical_features

    enriched = tactical_features.copy()
    enriched["_parsed_date"] = pd.to_datetime(enriched["date"], errors="coerce")
    enriched["_original_order"] = range(len(enriched))
    enriched = enriched.sort_values(["_parsed_date", "sportmonks_fixture_id", "_original_order"], na_position="last")

    history_by_team: dict[str, list[str]] = {}
    stability_values: dict[int, dict[str, Any]] = {}

    for row_index, row in enriched.iterrows():
        home_team = row.get("home_team_name")
        away_team = row.get("away_team_name")
        home_formation = row.get("home_formation")
        away_formation = row.get("away_formation")

        home_history = history_by_team.get(str(home_team), []) if not _is_missing(home_team) else []
        away_history = history_by_team.get(str(away_team), []) if not _is_missing(away_team) else []
        home_stability, home_count = _stability_from_history(home_formation, home_history)
        away_stability, away_count = _stability_from_history(away_formation, away_history)
        stability_values[row_index] = {
            "home_formation_stability_5": home_stability,
            "away_formation_stability_5": away_stability,
            "home_recent_formations_count": home_count,
            "away_recent_formations_count": away_count,
        }

        if not _is_missing(home_team) and not _is_missing(home_formation):
            history_by_team.setdefault(str(home_team), []).append(str(home_formation))
        if not _is_missing(away_team) and not _is_missing(away_formation):
            history_by_team.setdefault(str(away_team), []).append(str(away_formation))

    for column in [
        "home_formation_stability_5",
        "away_formation_stability_5",
        "home_recent_formations_count",
        "away_recent_formations_count",
    ]:
        enriched[column] = pd.Series(
            {row_index: values[column] for row_index, values in stability_values.items()}
        )

    return (
        enriched.sort_values("_original_order")
        .drop(columns=["_parsed_date", "_original_order"])
        .reset_index(drop=True)
    )


def _is_missing(value: Any) -> bool:
    """Return whether a metadata value is missing or empty."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _load_fixture_metadata(input_dir: Path) -> dict[int, dict[str, Any]]:
    """Load fixture metadata from fixtures_index.csv when available."""
    index_path = input_dir / "fixtures_index.csv"
    if not index_path.exists():
        print("No fixtures_index.csv found, using JSON metadata only")
        return {}

    print("Using fixtures_index.csv for metadata enrichment")
    index = pd.read_csv(index_path)
    if "sportmonks_fixture_id" not in index.columns:
        return {}

    metadata = {}
    for _, row in index.iterrows():
        try:
            fixture_id = int(row["sportmonks_fixture_id"])
        except (TypeError, ValueError):
            continue
        metadata[fixture_id] = row.to_dict()
    return metadata


def parse_tactical_data(
    input_dir: Path,
    lineups_path: Path = DEFAULT_LINEUPS_PATH,
    formations_path: Path = DEFAULT_FORMATIONS_PATH,
    tactical_features_path: Path = DEFAULT_TACTICAL_FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse Sportmonks fixture JSON files and export tactical CSVs."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("fixture_*.json"))
    fixture_metadata = _load_fixture_metadata(input_dir)
    warnings = []
    lineup_rows = []
    formation_rows = []
    feature_rows = []
    parsed_matches = 0

    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"{json_file.name}: invalid JSON ({exc})")
            continue

        fixture = _fixture_data(payload)
        if fixture is None:
            warnings.append(f"{json_file.name}: missing fixture data")
            continue

        fixture_id = fixture.get("id")
        try:
            fixture_id_int = int(fixture_id)
        except (TypeError, ValueError):
            fixture_id_int = None
        metadata = fixture_metadata.get(fixture_id_int or -1, {})
        participants = _participants_by_team(fixture)
        if not participants:
            warnings.append(f"{json_file.name}: missing teams/participants")

        home_team_id, away_team_id = _home_away_teams(participants, fixture.get("name"))
        if home_team_id is None or away_team_id is None:
            warnings.append(f"{json_file.name}: could not identify home/away teams")

        formations_by_team = _formations_by_team(fixture)
        if not formations_by_team:
            warnings.append(f"{json_file.name}: missing formations")

        lineups = [lineup for lineup in _as_list(fixture.get("lineups")) if isinstance(lineup, dict)]
        if not lineups:
            warnings.append(f"{json_file.name}: missing lineups")

        date_value = _fixture_date(fixture)
        league_id = fixture.get("league_id")
        season_id = fixture.get("season_id")
        home_team = participants.get(home_team_id or -1, {})
        away_team = participants.get(away_team_id or -1, {})
        home_team_name = _participant_name(home_team)
        away_team_name = _participant_name(away_team)
        home_formation = formations_by_team.get(home_team_id or -1)
        away_formation = formations_by_team.get(away_team_id or -1)
        league_name = None
        home_score, away_score, result = _fixture_scores(fixture)

        if _is_missing(league_id):
            league_id = metadata.get("league_id")
        if _is_missing(league_name):
            league_name = metadata.get("league_name")
        if _is_missing(season_id):
            season_id = metadata.get("season_id")
        if _is_missing(date_value):
            date_value = metadata.get("date")
        if _is_missing(home_team_name):
            home_team_name = metadata.get("home_team")
        if _is_missing(away_team_name):
            away_team_name = metadata.get("away_team")

        for lineup in lineups:
            team_id = lineup.get("team_id")
            try:
                team_id_int = int(team_id)
            except (TypeError, ValueError):
                team_id_int = None

            team = participants.get(team_id_int or -1, {})
            opponent_id = away_team_id if team_id_int == home_team_id else home_team_id if team_id_int == away_team_id else None
            opponent = participants.get(opponent_id or -1, {})
            lineup_rows.append(
                {
                    "sportmonks_fixture_id": fixture_id,
                    "date": date_value,
                    "league_id": league_id,
                    "league_name": league_name,
                    "season_id": season_id,
                    "team_id": team_id_int,
                    "team_name": _participant_name(team),
                    "opponent_team_id": opponent_id,
                    "opponent_team_name": _participant_name(opponent),
                    "is_home": team_id_int == home_team_id if team_id_int is not None else None,
                    "formation": formations_by_team.get(team_id_int or -1),
                    "player_id": lineup.get("player_id"),
                    "player_name": lineup.get("player_name"),
                    "position_id": lineup.get("position_id"),
                    "position_name": _position_name(lineup),
                    "formation_position": lineup.get("formation_position"),
                    "is_starter": _is_starter(lineup),
                    "lineup_type": _lineup_type(lineup.get("type_id")),
                    "jersey_number": lineup.get("jersey_number"),
                }
            )

        formation_matchup = (
            f"{home_formation}_vs_{away_formation}" if home_formation and away_formation else None
        )
        formation_rows.append(
            {
                "sportmonks_fixture_id": fixture_id,
                "date": date_value,
                "league_id": league_id,
                "league_name": league_name,
                "season_id": season_id,
                "home_team_id": home_team_id,
                "home_team_name": home_team_name,
                "away_team_id": away_team_id,
                "away_team_name": away_team_name,
                "home_formation": home_formation,
                "away_formation": away_formation,
                "formation_matchup": formation_matchup,
            }
        )

        home_counts = _parse_formation(home_formation)
        away_counts = _parse_formation(away_formation)
        feature_rows.append(
            {
                "sportmonks_fixture_id": fixture_id,
                "date": date_value,
                "league_id": league_id,
                "league_name": league_name,
                "season_id": season_id,
                "home_team_name": home_team_name,
                "away_team_name": away_team_name,
                "home_formation": home_formation,
                "away_formation": away_formation,
                "home_score": home_score,
                "away_score": away_score,
                "result": result,
                "formation_matchup": formation_matchup,
                "home_defenders_count": home_counts["defenders_count"],
                "home_midfielders_count": home_counts["midfielders_count"],
                "home_forwards_count": home_counts["forwards_count"],
                "away_defenders_count": away_counts["defenders_count"],
                "away_midfielders_count": away_counts["midfielders_count"],
                "away_forwards_count": away_counts["forwards_count"],
                "midfield_density_diff": _diff(home_counts["midfielders_count"], away_counts["midfielders_count"]),
                "attacking_line_diff": _diff(home_counts["forwards_count"], away_counts["forwards_count"]),
                "defensive_line_diff": _diff(home_counts["defenders_count"], away_counts["defenders_count"]),
                "home_back_three": home_counts["back_three"],
                "home_back_four": home_counts["back_four"],
                "home_back_five": home_counts["back_five"],
                "away_back_three": away_counts["back_three"],
                "away_back_four": away_counts["back_four"],
                "away_back_five": away_counts["back_five"],
                "home_two_strikers": home_counts["two_strikers"],
                "away_two_strikers": away_counts["two_strikers"],
            }
        )
        parsed_matches += 1

    lineups = pd.DataFrame(lineup_rows)
    formations = pd.DataFrame(formation_rows)
    tactical_features = _add_formation_stability(pd.DataFrame(feature_rows))

    for output_path, df in [
        (lineups_path, lineups),
        (formations_path, formations),
        (tactical_features_path, tactical_features),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    print("Sportmonks tactical parser")
    print(f"JSON files read: {len(json_files)}")
    print(f"Matches parsed: {parsed_matches}")
    print(f"Lineup player rows: {len(lineups)}")
    print(
        "Matches with home formation: "
        f"{int(formations['home_formation'].notna().sum()) if 'home_formation' in formations else 0}"
    )
    print(
        "Matches with away formation: "
        f"{int(formations['away_formation'].notna().sum()) if 'away_formation' in formations else 0}"
    )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    print("\nExported files:")
    print(f"- {lineups_path}")
    print(f"- {formations_path}")
    print(f"- {tactical_features_path}")

    return lineups, formations, tactical_features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse Sportmonks tactical JSON files into CSVs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--lineups-output", type=Path, default=DEFAULT_LINEUPS_PATH)
    parser.add_argument("--formations-output", type=Path, default=DEFAULT_FORMATIONS_PATH)
    parser.add_argument("--features-output", type=Path, default=DEFAULT_TACTICAL_FEATURES_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the tactical parser."""
    args = parse_args()
    try:
        parse_tactical_data(
            input_dir=args.input_dir,
            lineups_path=args.lineups_output,
            formations_path=args.formations_output,
            tactical_features_path=args.features_output,
        )
    except Exception as exc:
        print("Sportmonks tactical parser failed")
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
