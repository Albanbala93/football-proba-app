"""Analyze the last Ligue 1 scored matches to understand recent prediction failures."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.probability_engine_ligue1_api_xg_v2 import FEATURE_COLUMNS as LIGUE1_API_XG_V2_FEATURE_COLUMNS  # noqa: E402
from app.core.probability_engine_ligue1_api_xg_v2 import predict_ligue1_api_xg_v2  # noqa: E402


DEFAULT_PREMATCH_PATH = PROJECT_ROOT / "data" / "processed" / "api_football_ligue1_2025_prematch_features.csv"
DEFAULT_V2_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_api_xg_v2_model_predictions.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "ligue1_last_round_postmortem.csv"
DEFAULT_STREAMLIT_LOG_PATH = PROJECT_ROOT / "data" / "predictions" / "upcoming_predictions_log.csv"
RESULT_LABELS = ["H", "D", "A"]
LOG_LOSS_LABELS = ["A", "D", "H"]
LEAGUE_NAME = "Ligue 1"


def normalize_team_name(value: Any) -> str:
    """Normalize a team name for cross-file matching."""
    text = str(value or "").lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "", text)
    aliases = {
        "stadebrestois29": "brest",
        "parissaintgermain": "psg",
        "parissg": "psg",
        "psg": "psg",
        "lehavre": "lehavre",
        "parisfc": "parisfc",
        "redstarfc93": "redstar",
        "sainte": "saintetienne",
        "stetienne": "saintetienne",
    }
    return aliases.get(normalized, normalized)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def derive_result(row: pd.Series) -> str | None:
    """Derive H/D/A result from final goals."""
    if pd.isna(row["goals_home"]) or pd.isna(row["goals_away"]):
        return None
    home_goals = float(row["goals_home"])
    away_goals = float(row["goals_away"])
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def load_prematch_data(input_path: Path = DEFAULT_PREMATCH_PATH) -> pd.DataFrame:
    """Load the Ligue 1 prematch feature table."""
    if not input_path.exists():
        raise FileNotFoundError(f"Prematch features file not found: {input_path}")

    data = pd.read_csv(input_path)
    required = [
        "date",
        "home_team_name",
        "away_team_name",
        "goals_home",
        "goals_away",
        "home_api_expected_goals_for_5",
        "away_api_expected_goals_for_5",
        "api_expected_goals_diff_5",
        "home_formation_stability_5",
        "away_formation_stability_5",
        *LIGUE1_API_XG_V2_FEATURE_COLUMNS,
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in prematch features: {', '.join(missing)}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if "league" in data.columns:
        data = data[data["league"].astype(str) == LEAGUE_NAME].copy()
    data["actual_result"] = data.apply(derive_result, axis=1)
    data = data.dropna(subset=["date", "actual_result"]).copy()
    data = data.sort_values(["date", "fixture_id" if "fixture_id" in data.columns else "date"]).reset_index(drop=True)
    return data


def select_last_matches(data: pd.DataFrame, count: int = 9) -> pd.DataFrame:
    """Return the last scored matches in chronological order."""
    if len(data) < count:
        raise ValueError(f"Not enough scored Ligue 1 matches to select the last {count} rows.")
    return data.tail(count).copy().reset_index(drop=True)


def load_v2_predictions(path: Path = DEFAULT_V2_PREDICTIONS_PATH) -> pd.DataFrame:
    """Load the saved v2 prediction log when available."""
    if not path.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(path)
    if predictions.empty:
        return predictions

    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions = predictions.dropna(subset=["date"]).copy()
    predictions["match_date_key"] = predictions["date"].dt.date.astype(str)
    predictions["home_team_key"] = predictions["home_team_name"].map(normalize_team_name)
    predictions["away_team_key"] = predictions["away_team_name"].map(normalize_team_name)
    return predictions


def load_streamlit_log(path: Path = DEFAULT_STREAMLIT_LOG_PATH) -> pd.DataFrame:
    """Load the Streamlit prediction log when available."""
    if not path.exists():
        return pd.DataFrame()
    try:
        log = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "match_date" in log.columns:
        log["match_date"] = pd.to_datetime(log["match_date"], errors="coerce")
    return log


def build_match_key(df: pd.DataFrame, date_column: str, home_column: str, away_column: str) -> pd.DataFrame:
    """Add normalized match keys for merging."""
    keyed = df.copy()
    keyed["match_date_key"] = pd.to_datetime(keyed[date_column], errors="coerce").dt.date.astype(str)
    keyed["home_team_key"] = keyed[home_column].map(normalize_team_name)
    keyed["away_team_key"] = keyed[away_column].map(normalize_team_name)
    return keyed


def _confidence_score(probabilities: dict[str, float]) -> float:
    """Compute the same confidence score shape used by the v2 engine."""
    sorted_probabilities = sorted(probabilities.values(), reverse=True)
    favorite_probability = sorted_probabilities[0]
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - favorite_probability
    score = (favorite_probability * 70) + (top_two_margin * 80) - (uncertainty_score * 35)
    return round(max(0.0, min(100.0, score)), 2)


def v2_prediction_for_row(row: pd.Series, prediction_lookup: pd.DataFrame) -> dict[str, Any]:
    """Return v2 prediction values from the saved prediction file or by recomputing them."""
    if not prediction_lookup.empty:
        match = prediction_lookup[
            (prediction_lookup["match_date_key"] == row["match_date_key"])
            & (prediction_lookup["home_team_key"] == row["home_team_key"])
            & (prediction_lookup["away_team_key"] == row["away_team_key"])
        ]
        if not match.empty:
            selected = match.iloc[0]
            return {
                "prediction_source": "saved_prediction_log",
                "home_win_probability": float(selected["home_win_probability"]),
                "draw_probability": float(selected["draw_probability"]),
                "away_win_probability": float(selected["away_win_probability"]),
                "predicted_class_argmax": str(selected["predicted_class_argmax"]),
                "confidence_score": float(selected.get("confidence_score", _confidence_score({
                    "H": float(selected["home_win_probability"]),
                    "D": float(selected["draw_probability"]),
                    "A": float(selected["away_win_probability"]),
                }))),
                "model_version": str(selected.get("model_version", "ligue1_api_xg_v2")),
            }

    match_features = row[LIGUE1_API_XG_V2_FEATURE_COLUMNS].to_dict()
    prediction = predict_ligue1_api_xg_v2(match_features)
    return {
        "prediction_source": "recomputed_v2",
        "home_win_probability": float(prediction["home_win_probability"]),
        "draw_probability": float(prediction["draw_probability"]),
        "away_win_probability": float(prediction["away_win_probability"]),
        "predicted_class_argmax": str(prediction["predicted_class_argmax"]),
        "confidence_score": float(prediction["confidence_score"]),
        "model_version": str(prediction["model_version"]),
    }


def multiclass_brier_score(df: pd.DataFrame) -> float:
    """Compute the mean multiclass Brier score for H/D/A probabilities."""
    total = 0.0
    for label, probability_column in [
        ("H", "home_win_probability"),
        ("D", "draw_probability"),
        ("A", "away_win_probability"),
    ]:
        actual = (df["actual_result"] == label).astype(float)
        total += ((df[probability_column] - actual) ** 2).mean()
    return float(total / 3)


def safe_log_loss(df: pd.DataFrame) -> float | None:
    """Compute log loss if the probability columns are usable."""
    if df.empty:
        return None
    probabilities = df[["away_win_probability", "draw_probability", "home_win_probability"]].copy()
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)
    return float(log_loss(df["actual_result"], probabilities, labels=LOG_LOSS_LABELS))


def summarize_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize visible failure patterns."""
    correct = df[df["is_correct"]]
    errors = df[~df["is_correct"]]
    open_matches = df[df["top_probability"] < 0.45]
    strong_errors = df[(~df["is_correct"]) & (df["top_probability"] > 0.50)]

    def _mean(column: str, frame: pd.DataFrame) -> float | None:
        if frame.empty:
            return None
        return float(pd.to_numeric(frame[column], errors="coerce").mean())

    return {
        "correct_matches": int(len(correct)),
        "error_matches": int(len(errors)),
        "strong_errors": int(len(strong_errors)),
        "open_matches": int(len(open_matches)),
        "correct_avg_confidence": _mean("confidence_score", correct),
        "error_avg_confidence": _mean("confidence_score", errors),
        "correct_avg_api_expected_goals_diff_5": _mean("api_expected_goals_diff_5", correct),
        "error_avg_api_expected_goals_diff_5": _mean("api_expected_goals_diff_5", errors),
        "correct_avg_formation_stability": _mean("home_formation_stability_5", correct),
        "error_avg_formation_stability": _mean("home_formation_stability_5", errors),
    }


def build_postmortem(
    prematch_path: Path = DEFAULT_PREMATCH_PATH,
    v2_predictions_path: Path = DEFAULT_V2_PREDICTIONS_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    streamlit_log_path: Path = DEFAULT_STREAMLIT_LOG_PATH,
) -> pd.DataFrame:
    """Build and export the last-round postmortem."""
    prematch = load_prematch_data(prematch_path)
    last_matches = select_last_matches(prematch, 9)
    last_matches = build_match_key(last_matches, "date", "home_team_name", "away_team_name")

    saved_predictions = load_v2_predictions(v2_predictions_path)
    streamlit_log = load_streamlit_log(streamlit_log_path)

    rows: list[dict[str, Any]] = []
    for _, row in last_matches.iterrows():
        prediction = v2_prediction_for_row(row, saved_predictions)
        actual_result = row["actual_result"]
        predicted_class = prediction["predicted_class_argmax"]
        top_probability = max(
            prediction["home_win_probability"],
            prediction["draw_probability"],
            prediction["away_win_probability"],
        )

        streamlit_match = pd.DataFrame()
        if not streamlit_log.empty and {"match_date", "home_team", "away_team"}.issubset(streamlit_log.columns):
            streamlit_match = streamlit_log[
                (pd.to_datetime(streamlit_log["match_date"], errors="coerce").dt.date.astype(str) == row["match_date_key"])
                & (streamlit_log["home_team"].map(normalize_team_name) == row["home_team_key"])
                & (streamlit_log["away_team"].map(normalize_team_name) == row["away_team_key"])
            ]

        rows.append(
            {
                "date": row["date"],
                "league": row.get("league", LEAGUE_NAME),
                "home_team_name": row["home_team_name"],
                "away_team_name": row["away_team_name"],
                "score_reel": f"{int(row['goals_home'])}-{int(row['goals_away'])}",
                "actual_result": actual_result,
                "prediction_source": prediction["prediction_source"] if streamlit_match.empty else "streamlit_log_or_v2",
                "home_win_probability": prediction["home_win_probability"],
                "draw_probability": prediction["draw_probability"],
                "away_win_probability": prediction["away_win_probability"],
                "predicted_class_argmax": predicted_class,
                "recommended_prediction_class": predicted_class,
                "confidence_score": prediction["confidence_score"],
                "home_api_expected_goals_for_5": row.get("home_api_expected_goals_for_5"),
                "away_api_expected_goals_for_5": row.get("away_api_expected_goals_for_5"),
                "api_expected_goals_diff_5": row.get("api_expected_goals_diff_5"),
                "home_formation_stability_5": row.get("home_formation_stability_5"),
                "away_formation_stability_5": row.get("away_formation_stability_5"),
                "is_correct": predicted_class == actual_result,
                "is_correct_recommended": predicted_class == actual_result,
                "top_probability": top_probability,
                "strong_error": (predicted_class != actual_result) and (top_probability > 0.50),
                "very_open_match": top_probability < 0.45,
                "model_version": prediction["model_version"],
            }
        )

    report = pd.DataFrame(rows)
    report["home_win_probability"] = pd.to_numeric(report["home_win_probability"], errors="coerce")
    report["draw_probability"] = pd.to_numeric(report["draw_probability"], errors="coerce")
    report["away_win_probability"] = pd.to_numeric(report["away_win_probability"], errors="coerce")
    report["confidence_score"] = pd.to_numeric(report["confidence_score"], errors="coerce")
    report["top_probability"] = pd.to_numeric(report["top_probability"], errors="coerce")
    report["is_correct"] = report["is_correct"].astype(bool)
    report["is_correct_recommended"] = report["is_correct_recommended"].astype(bool)
    report["strong_error"] = report["strong_error"].astype(bool)
    report["very_open_match"] = report["very_open_match"].astype(bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print_summary(report, output_path)
    return report


def print_summary(report: pd.DataFrame, output_path: Path) -> None:
    """Print a concise console postmortem."""
    accuracy = accuracy_score(report["actual_result"], report["predicted_class_argmax"])
    logloss = safe_log_loss(report)
    brier = multiclass_brier_score(report)
    strong_errors = report[report["strong_error"]]
    open_matches = report[report["very_open_match"]]

    print("Ligue 1 last-round postmortem")
    print(f"Matches analyzed: {len(report)}")
    print(f"Accuracy: {accuracy:.4f}")
    if logloss is not None:
        print(f"Log loss: {logloss:.4f}")
    else:
        print("Log loss: not_available")
    print(f"Brier score multiclass: {brier:.4f}")
    print(f"Correctly predicted matches: {int(report['is_correct'].sum())}")
    print(f"Strong confidence errors: {len(strong_errors)}")
    print(f"Very open matches: {len(open_matches)}")

    print("\nCorrect matches:")
    correct_cols = ["date", "home_team_name", "away_team_name", "score_reel", "actual_result", "predicted_class_argmax", "confidence_score"]
    print(report[report["is_correct"]][correct_cols].to_string(index=False))

    if not strong_errors.empty:
        print("\nStrong confidence errors:")
        error_cols = [
            "date",
            "home_team_name",
            "away_team_name",
            "score_reel",
            "actual_result",
            "predicted_class_argmax",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "confidence_score",
        ]
        print(strong_errors[error_cols].to_string(index=False))
    else:
        print("\nStrong confidence errors: none")

    if not open_matches.empty:
        print("\nOpen matches:")
        open_cols = [
            "date",
            "home_team_name",
            "away_team_name",
            "score_reel",
            "actual_result",
            "predicted_class_argmax",
            "top_probability",
            "confidence_score",
        ]
        print(open_matches[open_cols].to_string(index=False))
    else:
        print("\nOpen matches: none")

    patterns = summarize_patterns(report)
    print("\nVisible patterns:")
    print(f"- correct_avg_confidence: {patterns['correct_avg_confidence']}")
    print(f"- error_avg_confidence: {patterns['error_avg_confidence']}")
    print(f"- correct_avg_api_expected_goals_diff_5: {patterns['correct_avg_api_expected_goals_diff_5']}")
    print(f"- error_avg_api_expected_goals_diff_5: {patterns['error_avg_api_expected_goals_diff_5']}")
    print(f"- correct_avg_formation_stability: {patterns['correct_avg_formation_stability']}")
    print(f"- error_avg_formation_stability: {patterns['error_avg_formation_stability']}")
    print(f"\nSaved postmortem to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze the last Ligue 1 round prediction failures.")
    parser.add_argument("--prematch", type=Path, default=DEFAULT_PREMATCH_PATH)
    parser.add_argument("--v2-predictions", type=Path, default=DEFAULT_V2_PREDICTIONS_PATH)
    parser.add_argument("--streamlit-log", type=Path, default=DEFAULT_STREAMLIT_LOG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the postmortem analysis."""
    args = parse_args()
    build_postmortem(
        prematch_path=args.prematch,
        v2_predictions_path=args.v2_predictions,
        output_path=args.output,
        streamlit_log_path=args.streamlit_log,
    )


if __name__ == "__main__":
    main()
