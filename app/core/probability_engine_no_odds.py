"""No-odds probability engine using form and Elo features only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "outcome_models_no_odds"

MODEL_PATHS = {
    "home_win": MODELS_DIR / "home_win_model_no_odds.pkl",
    "away_win": MODELS_DIR / "away_win_model_no_odds.pkl",
    "draw": MODELS_DIR / "draw_model_no_odds.pkl",
}


def _load_model(path: Path, label: str) -> Any:
    """Load a no-odds model or raise a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing no-odds {label} model: {path}. Run `python app/core/outcome_models_no_odds.py` first.")
    return joblib.load(path)


def load_no_odds_models() -> dict[str, Any]:
    """Load all no-odds models."""
    return {name: _load_model(path, name) for name, path in MODEL_PATHS.items()}


def _feature_columns_for_model(model: Any) -> list[str] | None:
    """Return model feature names if available."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)
    return None


def _positive_probability(model: Any, features: pd.DataFrame, model_name: str) -> float:
    """Predict probability of positive class 1."""
    feature_columns = _feature_columns_for_model(model)
    if feature_columns:
        missing = [column for column in feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing features for {model_name}: {', '.join(missing)}")
        features = features.loc[:, feature_columns]
    probabilities = model.predict_proba(features)
    classes = list(model.named_steps["classifier"].classes_)
    return float(probabilities[0, classes.index(1)])


def _normalize(home_win: float, draw: float, away_win: float) -> dict[str, float]:
    """Normalize 1N2 probabilities to sum to 1."""
    total = home_win + draw + away_win
    if total <= 0:
        raise ValueError("Raw no-odds probabilities sum to zero or less.")
    return {"home_win": home_win / total, "draw": draw / total, "away_win": away_win / total}


def _round(value: float) -> float:
    """Round for JSON-compatible output."""
    return round(float(value), 6)


def _match_profile(
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
    top_two_margin: float,
    uncertainty_score: float,
) -> str:
    """Classify the broad shape of the match probability distribution.

    Kept identical to probability_engine.py so both engines describe a match
    profile the same way regardless of which one produced the probabilities.
    """
    if home_win_probability >= 0.55:
        return "clear_home_advantage"
    if away_win_probability >= 0.55:
        return "clear_away_advantage"
    if draw_probability >= 0.32:
        return "very_strong_draw_signal"
    if draw_probability >= 0.30:
        return "strong_draw_signal"
    if top_two_margin <= 0.07:
        return "balanced_match"
    if draw_probability >= 0.27 and top_two_margin <= 0.10:
        return "draw_plausible"
    if uncertainty_score >= 0.62:
        return "high_uncertainty"
    return "moderate_advantage"


def _confidence_score(favorite_probability: float, top_two_margin: float, uncertainty_score: float) -> float:
    """Compute a bounded 0-100 confidence score from probability shape."""
    score = (favorite_probability * 70) + (top_two_margin * 80) - (uncertainty_score * 35)
    return round(max(0.0, min(100.0, score)), 2)


def predict_match_probabilities_no_odds(match_features: dict[str, Any]) -> dict[str, Any]:
    """Predict no-odds match probabilities."""
    if not match_features:
        raise ValueError("match_features cannot be empty.")
    features = pd.DataFrame([match_features])
    models = load_no_odds_models()
    raw_home_win = _positive_probability(models["home_win"], features, "home_win_no_odds")
    raw_away_win = _positive_probability(models["away_win"], features, "away_win_no_odds")
    raw_draw = _positive_probability(models["draw"], features, "draw_no_odds")
    normalized = _normalize(raw_home_win, raw_draw, raw_away_win)

    home_win = normalized["home_win"]
    draw = normalized["draw"]
    away_win = normalized["away_win"]
    sorted_probabilities = sorted([home_win, draw, away_win], reverse=True)
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - sorted_probabilities[0]
    favorite_team = "home" if home_win > away_win else "away" if away_win > home_win else "none"
    favorite_probability = max(home_win, away_win)
    match_profile = _match_profile(
        home_win_probability=home_win,
        draw_probability=draw,
        away_win_probability=away_win,
        top_two_margin=top_two_margin,
        uncertainty_score=uncertainty_score,
    )
    confidence_score = _confidence_score(
        favorite_probability=favorite_probability,
        top_two_margin=top_two_margin,
        uncertainty_score=uncertainty_score,
    )

    return {
        "probabilities": {
            "home": {"win": _round(home_win), "draw": _round(draw), "loss": _round(away_win), "no_loss": _round(home_win + draw)},
            "away": {"win": _round(away_win), "draw": _round(draw), "loss": _round(home_win), "no_loss": _round(away_win + draw)},
            "match": {"home_win": _round(home_win), "draw": _round(draw), "away_win": _round(away_win)},
        },
        "raw_probabilities": {"home_win": _round(raw_home_win), "draw": _round(raw_draw), "away_win": _round(raw_away_win)},
        "analysis": {
            "model_version": "no_odds_elo_v1",
            "favorite_team": favorite_team,
            "favorite_probability": _round(favorite_probability),
            "uncertainty_score": _round(uncertainty_score),
            "top_two_margin": _round(top_two_margin),
            "match_profile": match_profile,
            "confidence_score": confidence_score,
            "is_draw_plausible": bool(draw >= 0.27),
            "is_strong_draw_signal": bool(draw >= 0.30),
            "is_very_strong_draw_signal": bool(draw >= 0.32),
        },
    }


def predict_from_latest_features_csv(row_index: int = -1) -> dict[str, Any]:
    """Predict from a row in matches_features.csv."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
    features = pd.read_csv(FEATURES_PATH)
    row = features.iloc[row_index]
    ignored = {"Date", "date", "match_date", "HomeTeam", "home_team", "AwayTeam", "away_team", "FTR", "result", "target", "source_file", "season", "B365H", "B365D", "B365A", "home_odds", "draw_odds", "away_odds"}
    result = predict_match_probabilities_no_odds({key: value for key, value in row.to_dict().items() if key not in ignored})
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Predict with no-odds probability engine.")
    parser.add_argument("--row-index", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    """Run a smoke prediction."""
    args = parse_args()
    predict_from_latest_features_csv(args.row_index)


if __name__ == "__main__":
    main()
