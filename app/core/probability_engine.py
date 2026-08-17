"""Central probability engine for football match analysis.

V3 uses classic Elo-aware binary outcome models for the main normalized 1N2
probabilities, then uses the calibrated sigmoid draw model as a secondary draw
signal when it is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"
OUTCOME_MODELS_DIR = PROJECT_ROOT / "models" / "outcome_models"
CALIBRATED_MODELS_DIR = PROJECT_ROOT / "models" / "calibrated_outcome_models"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_logistic_model.pkl"

CLASSIC_MODEL_PATHS = {
    "home_win": OUTCOME_MODELS_DIR / "home_win_model_classic.pkl",
    "away_win": OUTCOME_MODELS_DIR / "away_win_model_classic.pkl",
    "draw": OUTCOME_MODELS_DIR / "draw_model_classic.pkl",
}

CALIBRATED_DRAW_SIGNAL_MODEL_PATH = CALIBRATED_MODELS_DIR / "draw_model_sigmoid.pkl"


def _load_model(model_path: Path, label: str) -> Any:
    """Load a joblib model or raise a clear error."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {label} model: {model_path}. "
            "Run `python app/core/outcome_models.py` before using the probability engine."
        )
    return joblib.load(model_path)


def load_outcome_models() -> dict[str, Any]:
    """Load classic binary outcome models used for V3 main probabilities."""
    return {name: _load_model(path, f"classic {name}") for name, path in CLASSIC_MODEL_PATHS.items()}


def load_calibrated_draw_signal_model_if_available() -> Any | None:
    """Load the optional calibrated sigmoid draw model."""
    if CALIBRATED_DRAW_SIGNAL_MODEL_PATH.exists():
        return joblib.load(CALIBRATED_DRAW_SIGNAL_MODEL_PATH)
    return None


def load_baseline_model_if_available() -> Any | None:
    """Load the optional 1N2 baseline model when present."""
    if BASELINE_MODEL_PATH.exists():
        return joblib.load(BASELINE_MODEL_PATH)
    return None


def _feature_columns_for_model(model: Any) -> list[str] | None:
    """Return model feature names when scikit-learn exposes them."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)
    return None


def _predict_positive_probability(model: Any, features: pd.DataFrame, model_name: str) -> float:
    """Predict the probability of class 1 for a binary model."""
    feature_columns = _feature_columns_for_model(model)
    if feature_columns:
        missing_columns = [column for column in feature_columns if column not in features.columns]
        if missing_columns:
            raise ValueError(
                f"Missing features for {model_name}: {', '.join(missing_columns)}. "
                "Provide the same feature columns used during training."
            )
        features = features.loc[:, feature_columns]

    probabilities = model.predict_proba(features)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError(f"Model {model_name} cannot predict positive class 1.")
    positive_index = classes.index(1)
    return float(probabilities[0, positive_index])


def _normalize_probabilities(home_win: float, draw: float, away_win: float) -> dict[str, float]:
    """Normalize raw H/D/A probabilities so they sum to 1."""
    total = home_win + draw + away_win
    if total <= 0:
        raise ValueError("Raw probabilities sum to zero or less; cannot normalize.")
    return {
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
    }


def _favorite(probabilities: dict[str, float]) -> tuple[str, float]:
    """Return favorite side and probability from normalized match probabilities."""
    home_win = probabilities["home_win"]
    away_win = probabilities["away_win"]

    if home_win == away_win:
        return "none", home_win
    if home_win > away_win:
        return "home", home_win
    return "away", away_win


def _match_profile(
    home_win_probability: float,
    draw_probability: float,
    away_win_probability: float,
    top_two_margin: float,
    uncertainty_score: float,
) -> str:
    """Classify the broad shape of the match probability distribution."""
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


def _round_probability(value: float) -> float:
    """Round probabilities for JSON-compatible output."""
    return round(float(value), 6)


def _draw_signal_reaches(value: float | None, threshold: float) -> bool:
    """Return whether an optional draw signal reaches a threshold."""
    return value is not None and value >= threshold


def predict_match_probabilities(match_features: dict[str, Any]) -> dict[str, Any]:
    """Predict normalized football match probabilities from one feature dictionary."""
    if not match_features:
        raise ValueError("match_features cannot be empty.")

    features = pd.DataFrame([match_features])
    outcome_models = load_outcome_models()
    calibrated_draw_signal_model = load_calibrated_draw_signal_model_if_available()
    _ = load_baseline_model_if_available()

    raw_home_win = _predict_positive_probability(outcome_models["home_win"], features, "home_win")
    raw_away_win = _predict_positive_probability(outcome_models["away_win"], features, "away_win")
    raw_draw = _predict_positive_probability(outcome_models["draw"], features, "draw")
    calibrated_draw_signal = (
        _predict_positive_probability(calibrated_draw_signal_model, features, "calibrated_draw_signal")
        if calibrated_draw_signal_model is not None
        else None
    )

    normalized = _normalize_probabilities(raw_home_win, raw_draw, raw_away_win)
    home_win_probability = normalized["home_win"]
    draw_probability = normalized["draw"]
    away_win_probability = normalized["away_win"]

    sorted_probabilities = sorted([home_win_probability, draw_probability, away_win_probability], reverse=True)
    top_two_margin = sorted_probabilities[0] - sorted_probabilities[1]
    uncertainty_score = 1 - sorted_probabilities[0]
    favorite_team, favorite_probability = _favorite(normalized)
    is_draw_plausible = draw_probability >= 0.27 or _draw_signal_reaches(calibrated_draw_signal, 0.27)
    is_strong_draw_signal = draw_probability >= 0.30 or _draw_signal_reaches(calibrated_draw_signal, 0.30)
    is_very_strong_draw_signal = draw_probability >= 0.32 or _draw_signal_reaches(calibrated_draw_signal, 0.32)
    match_profile = _match_profile(
        home_win_probability=home_win_probability,
        draw_probability=draw_probability,
        away_win_probability=away_win_probability,
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
            "home": {
                "win": _round_probability(home_win_probability),
                "draw": _round_probability(draw_probability),
                "loss": _round_probability(away_win_probability),
                "no_loss": _round_probability(home_win_probability + draw_probability),
            },
            "away": {
                "win": _round_probability(away_win_probability),
                "draw": _round_probability(draw_probability),
                "loss": _round_probability(home_win_probability),
                "no_loss": _round_probability(away_win_probability + draw_probability),
            },
            "match": {
                "home_win": _round_probability(home_win_probability),
                "draw": _round_probability(draw_probability),
                "away_win": _round_probability(away_win_probability),
            },
        },
        "raw_probabilities": {
            "home_win": _round_probability(raw_home_win),
            "draw": _round_probability(raw_draw),
            "away_win": _round_probability(raw_away_win),
        },
        "analysis": {
            "model_version": "classic_elo_v3_with_calibrated_draw_signal",
            "main_probability_source": "classic_models",
            "draw_signal_source": "draw_model_sigmoid" if calibrated_draw_signal is not None else "none",
            "calibrated_draw_signal": _round_probability(calibrated_draw_signal) if calibrated_draw_signal is not None else None,
            "favorite_team": favorite_team,
            "favorite_probability": _round_probability(favorite_probability),
            "uncertainty_score": _round_probability(uncertainty_score),
            "top_two_margin": _round_probability(top_two_margin),
            "match_profile": match_profile,
            "confidence_score": confidence_score,
            "is_draw_plausible": bool(is_draw_plausible),
            "is_strong_draw_signal": bool(is_strong_draw_signal),
            "is_very_strong_draw_signal": bool(is_very_strong_draw_signal),
        },
    }


def predict_from_latest_features_csv(row_index: int = -1) -> dict[str, Any]:
    """Predict probabilities from one row of data/processed/matches_features.csv."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}. "
            "Run `python app/data_pipeline/build_features.py` first."
        )

    features = pd.read_csv(FEATURES_PATH)
    if features.empty:
        raise ValueError(f"Feature file is empty: {FEATURES_PATH}")

    try:
        row = features.iloc[row_index]
    except IndexError as exc:
        raise IndexError(f"row_index {row_index} is out of range for {len(features)} rows.") from exc

    ignored_columns = {"Date", "date", "match_date", "HomeTeam", "home_team", "AwayTeam", "away_team", "FTR", "result", "target"}
    match_features = {key: value for key, value in row.to_dict().items() if key not in ignored_columns}
    result = predict_match_probabilities(match_features)

    print(f"Using row_index={row_index} from: {FEATURES_PATH}")
    if "HomeTeam" in row and "AwayTeam" in row:
        print(f"Match: {row['HomeTeam']} vs {row['AwayTeam']}")
    if "Date" in row:
        print(f"Date: {row['Date']}")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for probability engine smoke testing."""
    parser = argparse.ArgumentParser(description="Predict match probabilities from the features CSV.")
    parser.add_argument("--row-index", type=int, default=-1, help="Row index from matches_features.csv. Defaults to -1.")
    return parser.parse_args()


def main() -> None:
    """Run a prediction from the latest features CSV row."""
    args = parse_args()
    predict_from_latest_features_csv(row_index=args.row_index)


if __name__ == "__main__":
    main()
