"""Analyze generated tactical feature data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tactical_features.csv"
DEFAULT_FORMATIONS_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "tactical_formations_frequency.csv"
DEFAULT_MATCHUPS_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "tactical_matchups_frequency.csv"
DEFAULT_TEAM_STABILITY_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "tactical_team_stability.csv"
DEFAULT_STRUCTURE_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "tactical_structure_summary.csv"


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in tactical features: {', '.join(missing)}")


def _frequency_table(series: pd.Series, value_column: str) -> pd.DataFrame:
    """Return matches and share percentage for one categorical series."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    total = len(values)
    frequency = values.value_counts().rename_axis(value_column).reset_index(name="matches")
    frequency["share_pct"] = frequency["matches"] / total * 100 if total else 0.0
    return frequency


def formation_frequency(tactical_features: pd.DataFrame) -> pd.DataFrame:
    """Build home, away, and global formation frequency rows."""
    home = _frequency_table(tactical_features["home_formation"], "formation")
    home.insert(0, "scope", "home")

    away = _frequency_table(tactical_features["away_formation"], "formation")
    away.insert(0, "scope", "away")

    global_formations = _frequency_table(
        pd.concat([tactical_features["home_formation"], tactical_features["away_formation"]], ignore_index=True),
        "formation",
    )
    global_formations.insert(0, "scope", "global")

    return pd.concat([home, away, global_formations], ignore_index=True)


def matchup_frequency(tactical_features: pd.DataFrame) -> pd.DataFrame:
    """Build formation matchup frequency rows."""
    return _frequency_table(tactical_features["formation_matchup"], "formation_matchup")


def _team_rows(
    tactical_features: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """Return team-level rows for one home/away side."""
    return pd.DataFrame(
        {
            "team_name": tactical_features[f"{side}_team_name"],
            "formation": tactical_features[f"{side}_formation"],
            "formation_stability_5": tactical_features[f"{side}_formation_stability_5"],
            "recent_formations_count": tactical_features[f"{side}_recent_formations_count"],
        }
    )


def team_stability(tactical_features: pd.DataFrame) -> pd.DataFrame:
    """Build average formation stability by team."""
    team_rows = pd.concat(
        [_team_rows(tactical_features, "home"), _team_rows(tactical_features, "away")],
        ignore_index=True,
    )
    team_rows = team_rows.dropna(subset=["team_name"]).copy()

    rows = []
    for team_name, team_df in team_rows.groupby("team_name", sort=True):
        formations = team_df["formation"].dropna().astype(str).str.strip()
        rows.append(
            {
                "team_name": team_name,
                "matches": int(len(team_df)),
                "avg_formation_stability_5": float(team_df["formation_stability_5"].mean()),
                "avg_recent_formations_count": float(team_df["recent_formations_count"].mean()),
                "most_common_formation": formations.value_counts().idxmax() if not formations.empty else None,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["avg_formation_stability_5", "matches", "team_name"],
        ascending=[False, False, True],
    )


def structure_summary(tactical_features: pd.DataFrame) -> pd.DataFrame:
    """Build home/away structural feature rates."""
    rows = []
    for side in ["home", "away"]:
        rows.append(
            {
                "side": side,
                "matches": int(len(tactical_features)),
                "back_three_rate": float(tactical_features[f"{side}_back_three"].mean()),
                "back_four_rate": float(tactical_features[f"{side}_back_four"].mean()),
                "back_five_rate": float(tactical_features[f"{side}_back_five"].mean()),
                "two_strikers_rate": float(tactical_features[f"{side}_two_strikers"].mean()),
            }
        )
    return pd.DataFrame(rows)


def analyze_tactical_features(
    input_path: Path = DEFAULT_INPUT_PATH,
    formations_output_path: Path = DEFAULT_FORMATIONS_OUTPUT_PATH,
    matchups_output_path: Path = DEFAULT_MATCHUPS_OUTPUT_PATH,
    team_stability_output_path: Path = DEFAULT_TEAM_STABILITY_OUTPUT_PATH,
    structure_output_path: Path = DEFAULT_STRUCTURE_OUTPUT_PATH,
) -> dict[str, pd.DataFrame]:
    """Analyze tactical features and export summary reports."""
    if not input_path.exists():
        raise FileNotFoundError(f"Tactical features file not found: {input_path}")

    tactical_features = pd.read_csv(input_path)
    _require_columns(
        tactical_features,
        [
            "home_team_name",
            "away_team_name",
            "home_formation",
            "away_formation",
            "formation_matchup",
            "home_formation_stability_5",
            "away_formation_stability_5",
            "home_recent_formations_count",
            "away_recent_formations_count",
            "home_back_three",
            "home_back_four",
            "home_back_five",
            "away_back_three",
            "away_back_four",
            "away_back_five",
            "home_two_strikers",
            "away_two_strikers",
        ],
    )

    reports = {
        "formations": formation_frequency(tactical_features),
        "matchups": matchup_frequency(tactical_features),
        "team_stability": team_stability(tactical_features),
        "structure": structure_summary(tactical_features),
    }

    for output_path, report in [
        (formations_output_path, reports["formations"]),
        (matchups_output_path, reports["matchups"]),
        (team_stability_output_path, reports["team_stability"]),
        (structure_output_path, reports["structure"]),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    print_summary(reports, input_path, len(tactical_features))
    print("\nExported files:")
    print(f"- {formations_output_path}")
    print(f"- {matchups_output_path}")
    print(f"- {team_stability_output_path}")
    print(f"- {structure_output_path}")
    return reports


def print_summary(reports: dict[str, pd.DataFrame], input_path: Path, total_matches: int) -> None:
    """Print a clear console summary."""
    print("Tactical features analysis")
    print(f"Input: {input_path}")
    print(f"Matches: {total_matches}")

    formations = reports["formations"]
    global_formations = formations[formations["scope"] == "global"].copy()
    print("\nTop 10 formations:")
    print(global_formations.head(10).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    print("\nTop 10 matchups:")
    print(reports["matchups"].head(10).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    team_stability_report = reports["team_stability"].copy()
    print("\nTop 10 most stable teams:")
    print(team_stability_report.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nTop 10 least stable teams:")
    print(
        team_stability_report.sort_values(
            ["avg_formation_stability_5", "matches", "team_name"],
            ascending=[True, False, True],
        )
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze generated tactical features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--formations-output", type=Path, default=DEFAULT_FORMATIONS_OUTPUT_PATH)
    parser.add_argument("--matchups-output", type=Path, default=DEFAULT_MATCHUPS_OUTPUT_PATH)
    parser.add_argument("--team-stability-output", type=Path, default=DEFAULT_TEAM_STABILITY_OUTPUT_PATH)
    parser.add_argument("--structure-output", type=Path, default=DEFAULT_STRUCTURE_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run tactical feature analysis from the command line."""
    args = parse_args()
    analyze_tactical_features(
        input_path=args.input,
        formations_output_path=args.formations_output,
        matchups_output_path=args.matchups_output,
        team_stability_output_path=args.team_stability_output,
        structure_output_path=args.structure_output,
    )


if __name__ == "__main__":
    main()
